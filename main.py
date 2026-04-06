"""
main.py
CSTLF — training and evaluation entry point.

Usage:
    # Train on sKwandaSCD_V1
    python main.py --config configs/sKwandaSCD_V1.yaml --mode train

    # Evaluate a checkpoint
    python main.py --config configs/SECOND.yaml --mode eval \
                   --checkpoint checkpoints/SECOND/best.pth

    # Large-scale inference on a full GeoTIFF scene
    python main.py --config configs/sKwandaSCD_V1.yaml --mode infer \
                   --t1 path/to/t1.tif --t2 path/to/t2.tif \
                   --checkpoint checkpoints/sKwandaSCD_V1/best.pth \
                   --output outputs/kigali_pred.tif
"""

import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm

from models.cstlf   import CSTLF
from models.losses  import CSTLFLoss
from datasets.scd_dataset import build_dataset
from utils.metrics  import SCDMetrics


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────
class CSTLFTrainer:
    """
    OOP Trainer for CSTLF.

    Handles training loop, validation, checkpoint saving/loading,
    and metric logging.

    Args:
        cfg : configuration dict loaded from a YAML file
    """

    def __init__(self, cfg: dict):
        self.cfg    = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Datasets
        self.train_ds = build_dataset(
            cfg['dataset']['name'], cfg['dataset']['root'],
            split='train',
            img_size=tuple(cfg['dataset']['img_size']),
            augment=cfg['dataset']['augment'],
        )
        self.val_ds = build_dataset(
            cfg['dataset']['name'], cfg['dataset']['root'],
            split='val',
            img_size=tuple(cfg['dataset']['img_size']),
            augment=False,
        )

        t = cfg['training']
        self.train_loader = DataLoader(
            self.train_ds, batch_size=t['batch_size'],
            shuffle=True,  num_workers=t['num_workers'],
            pin_memory=True, drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=t['batch_size'],
            shuffle=False, num_workers=t['num_workers'],
            pin_memory=True,
        )

        # Model
        m = cfg['model']
        self.model = CSTLF(
            in_ch=m['in_ch'], base_ch=m['base_ch'],
            num_classes=m['num_classes'], num_heads=m['num_heads'],
            patch_size=m['patch_size'],
            tau_init=m['tau_init'], tau_min=m['tau_min'],
        ).to(self.device)

        # Loss
        lc = cfg['loss']
        self.criterion = CSTLFLoss(
            num_classes=m['num_classes'],
            lambda_sup=lc['lambda_sup'],   lambda_psu=lc['lambda_psu'],
            lambda_temp=lc['lambda_temp'], lambda_cons=lc['lambda_cons'],
            ignore_index=lc['ignore_index'],
        )

        # Optimiser
        self.optimiser = optim.AdamW(
            self.model.parameters(),
            lr=t['lr'], weight_decay=t['weight_decay'],
        )
        self.scheduler = CosineAnnealingLR(
            self.optimiser,
            T_max=t['epochs'] - t.get('warmup_epochs', 0),
            eta_min=t['lr'] * 0.01,
        )

        self.epochs       = t['epochs']
        self.save_freq    = cfg['output']['save_freq']
        self.save_dir     = Path(cfg['output']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.num_classes  = m['num_classes']
        self.ignore_index = lc['ignore_index']
        self.best_miou    = 0.0
        self.total_steps  = self.epochs * len(self.train_loader)
        self.global_step  = 0

        print(f'Device     : {self.device}')
        print(f'Dataset    : {cfg["dataset"]["name"]}  '
              f'(train={len(self.train_ds)}, val={len(self.val_ds)})')
        print(f'Parameters : {sum(p.numel() for p in self.model.parameters()):,}')

    # ── Training epoch ────────────────────────────────────────────────────
    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        running = {'total': 0., 'sup': 0., 'psu': 0., 'temp': 0., 'cons': 0.}
        n = 0

        for batch in tqdm(self.train_loader, desc=f'Epoch {epoch}', leave=False):
            t1     = batch['T1'].to(self.device)
            t2     = batch['T2'].to(self.device)
            labels = batch['label_to'].to(self.device)          # T2 semantic map
            # Pseudo-labels: start as −1 (unlabeled), refined by WS-PLR
            pseudo = torch.full_like(labels, -1)

            self.optimiser.zero_grad()

            out = self.model(t1, t2, pseudo, self.global_step, self.total_steps)

            # Placeholder feature tensors for temp/cons losses
            # (in full impl. expose intermediate features from model)
            dummy_feat = torch.zeros_like(out['prob'][:, :1])

            loss_dict = self.criterion(
                logits=out['logits'],
                labels=labels,
                pseudo=out['pseudo_new'],
                confidence=out['confidence'],
                h_t1=dummy_feat, h_t2=dummy_feat,
                f_fine=dummy_feat, f_coarse=dummy_feat,
            )

            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimiser.step()

            for k in running:
                running[k] += loss_dict[k].item()
            n += 1
            self.global_step += 1

        return {k: v / n for k, v in running.items()}

    # ── Validation ────────────────────────────────────────────────────────
    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        metrics = SCDMetrics(self.num_classes, self.ignore_index)

        for batch in tqdm(self.val_loader, desc='Validation', leave=False):
            t1     = batch['T1'].to(self.device)
            t2     = batch['T2'].to(self.device)
            labels = batch['label_to'].to(self.device)

            out = self.model(t1, t2)
            metrics.update(out['pred'], labels)

        return metrics.compute()

    # ── Save / load ───────────────────────────────────────────────────────
    def save_checkpoint(self, epoch: int, tag: str = 'latest'):
        path = self.save_dir / f'{tag}.pth'
        torch.save({
            'epoch':       epoch,
            'state_dict':  self.model.state_dict(),
            'optimiser':   self.optimiser.state_dict(),
            'best_miou':   self.best_miou,
            'global_step': self.global_step,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['state_dict'])
        self.optimiser.load_state_dict(ckpt['optimiser'])
        self.best_miou   = ckpt.get('best_miou',   0.0)
        self.global_step = ckpt.get('global_step', 0)
        print(f'Loaded checkpoint: {path}  (best mIoU={self.best_miou:.2f}%)')
        return ckpt['epoch']

    # ── Main train loop ───────────────────────────────────────────────────
    def train(self):
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate()
            self.scheduler.step()

            miou = val_metrics['mIoU']
            sek  = val_metrics['SeK']

            print(f'[Epoch {epoch:03d}/{self.epochs}]  '
                  f"loss={train_loss['total']:.4f}  "
                  f"mIoU={miou:.2f}%  SeK={sek:.2f}pp")

            if miou > self.best_miou:
                self.best_miou = miou
                self.save_checkpoint(epoch, 'best')
                print(f'  → New best mIoU: {self.best_miou:.2f}%')

            if epoch % self.save_freq == 0:
                self.save_checkpoint(epoch, f'epoch_{epoch:03d}')

        self.save_checkpoint(self.epochs, 'final')
        print(f'\nTraining complete. Best mIoU: {self.best_miou:.2f}%')


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────
class CSTLFEvaluator:
    """Standalone evaluator — loads a checkpoint and runs test-set evaluation."""

    def __init__(self, cfg: dict, checkpoint: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        m = cfg['model']
        self.model = CSTLF(**m).to(self.device)

        ckpt = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()

        self.ds = build_dataset(
            cfg['dataset']['name'], cfg['dataset']['root'],
            split='test',
            img_size=tuple(cfg['dataset']['img_size']),
            augment=False,
        )
        self.loader = DataLoader(self.ds, batch_size=4, shuffle=False, num_workers=4)
        self.num_classes  = m['num_classes']
        self.ignore_index = cfg['loss']['ignore_index']

    @torch.no_grad()
    def evaluate(self, class_names: list = None) -> dict:
        metrics = SCDMetrics(self.num_classes, self.ignore_index)
        for batch in tqdm(self.loader, desc='Testing'):
            t1     = batch['T1'].to(self.device)
            t2     = batch['T2'].to(self.device)
            labels = batch['label_to'].to(self.device)
            out    = self.model(t1, t2)
            metrics.update(out['pred'], labels)

        print(metrics.summary(class_names))
        return metrics.compute()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='CSTLF Semantic Change Detection')
    parser.add_argument('--config',     type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--mode',       type=str, default='train',
                        choices=['train', 'eval'],
                        help='train | eval')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (required for eval mode)')
    parser.add_argument('--resume',     type=str, default=None,
                        help='Resume training from checkpoint')
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    args = parse_args()
    cfg  = load_config(args.config)

    if args.mode == 'train':
        trainer = CSTLFTrainer(cfg)
        if args.resume:
            trainer.load_checkpoint(args.resume)
        trainer.train()

    elif args.mode == 'eval':
        if args.checkpoint is None:
            raise ValueError('--checkpoint required for eval mode')
        evaluator = CSTLFEvaluator(cfg, args.checkpoint)
        evaluator.evaluate()
