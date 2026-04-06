<div align="center">

<h1>CSTLF: From Asymmetric Bi-Temporal Encoding to Semantic Transition Mapping<br>in High-Resolution Remote Sensing under Weak Supervision</h1>

<p>
  <a href="https://github.com/BoazGithub/CSTLF_V1">
    <img src="https://img.shields.io/badge/Code-GitHub-blue?logo=github" alt="GitHub"/>
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Status-Under%20Review-orange" alt="Status"/>
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/License-Non--Commercial%20Research-lightgrey" alt="License"/>
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Python-3.8%2B-green?logo=python" alt="Python"/>
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/PyTorch-1.12%2B-red?logo=pytorch" alt="PyTorch"/>
  </a>
</p>

<p>
  <b>Boaz Mwubahimana<sup>1</sup></b>,
  Dingruibo Miao<sup>1*</sup>,
  Yan Jianguo<sup>1*</sup>,
  Le Ma<sup>1</sup>,
  Clarisse Kagoyire<sup>2</sup>,
  Ange Felix Nsanziyera<sup>3</sup>,
  Swalpa Kumar Roy<sup>4</sup>,
  Yumin Tan<sup>5</sup>,
  Ruisheng Wang<sup>6</sup>,
  Xiao Huang<sup>7</sup>
</p>

<p>
  <sup>1</sup>LIESMARS, Wuhan University &nbsp;|&nbsp;
  <sup>2</sup>CGIS, University of Rwanda &nbsp;|&nbsp;
  <sup>3</sup>INES Ruhengeri &nbsp;|&nbsp;
  <sup>4</sup>Tezpur University &nbsp;|&nbsp;
  <sup>5</sup>Beihang University &nbsp;|&nbsp;
  <sup>6</sup>Shenzhen University &nbsp;|&nbsp;
  <sup>7</sup>Emory University
</p>

<p><sup>*</sup>Corresponding authors:
  <a href="mailto:miaodrb@whu.edu.cn">miaodrb@whu.edu.cn</a> &nbsp;|&nbsp;
  <a href="mailto:jgyan@whu.edu.cn">jgyan@whu.edu.cn</a>
</p>

</div>

---

## Abstract

Semantic change detection (SCD) from high-resolution multi-temporal imagery requires simultaneous localization and semantic labeling of land-cover transitions — a fundamentally harder problem than binary change detection (BCD) that remains unsolved under realistic operational constraints. Three structural challenges persist: (*i*) symmetric bi-temporal encoders that cannot account for the heterogeneous land-cover distributions arising between acquisition epochs; (*ii*) fixed-scale spatial representations that are inadequate under heterogeneous imaging conditions; and (*iii*) dense pixel-level annotation requirements that restrict scalability in data-scarce regions.

To address these limitations, we propose the **Cross-Spatio-Temporal Learning Framework (CSTLF)**, a unified end-to-end architecture integrating four tightly coupled components: a **Multi-Scale Feature Fusion Network (MSFFN)** for adaptive hierarchical spatial encoding; a **Dual-branch Temporal–Spatial Attention Mechanism (DbTSAM)** for joint intra- and inter-temporal dependency modeling; a **Cross-Temporal Fusion Network (CTFN)** coupling LSTM-based sequential memory with Transformer-based global reasoning; and a **Weakly Supervised Pseudo-Label Refinement (WS-PLR)** strategy that progressively suppresses label noise from spatially coarse supervision.

Experiments on **sKwandaSCD\_V1**, **SECOND**, and **LsSCD-Ex** demonstrate mIoU improvements of up to **8.3%** and Separated Kappa (SeK) gains of up to **5.1 pp** while reducing computational complexity by **23%** relative to eight state-of-the-art comparison methods.

---

## Updates

| Date | Status |
|------|--------|
| January 2026 | Manuscript submitted for peer review |
| January 2026 | sKwandaSCD\_V1 dataset and pre-trained weights released |
| December 2025 | CSTLF codebase finalized and tested on three benchmarks |

---

## Key Contributions

- **Asymmetric temporal encoding** via DbTSAM, explicitly modeling heterogeneous land-cover distributions between acquisition epochs — the fundamental problem that symmetric Siamese encoders cannot address.
- **Adaptive multi-scale fusion** via MSFFN, jointly weighting fine, medium, and coarse spatial representations based on local context across heterogeneous landscapes.
- **Hybrid sequential–global temporal reasoning** via CTFN, combining LSTM gated memory and Transformer self-attention through adaptive gating to capture both gradual and abrupt semantic transitions.
- **Confidence-guided weakly supervised refinement** via WS-PLR, progressively suppressing non-change label dominance and pseudo-label noise from low-resolution GLC supervision without requiring synthetic data generation.
- **Separated Kappa (SeK) evaluation** decoupling non-change pixels from semantic change scoring, providing a bias-corrected metric under severe label imbalance.

---

## Architecture Overview

```
HR Multi-Temporal Input (T1, T2)
         │
         ▼
  ┌─────────────┐
  │    MSFFN    │  Fine / Medium / Coarse branches → Adaptive fusion
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │   DbTSAM   │  Spatial branch (intra-temporal) ⊗ Temporal branch (inter-temporal)
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │    CTFN     │  LSTM (sequential memory) + Transformer (global) → Adaptive gate
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │   WS-PLR   │  Confidence filtering → Adaptive threshold τ → Pseudo-label refinement
  └──────┬──────┘
         │
         ▼
  Semantic Change Map  +  Uncertainty / Confidence Maps
```

The full architecture diagram is provided in `figures/CSTLF_framework_design4.pdf`.

---

## Datasets

CSTLF is evaluated on three large-scale SCD benchmarks.

### sKwandaSCD\_V1 *(introduced in this work)*

| Property | Value |
|----------|-------|
| Coverage | Kigali, Nyagatare, Bugesera — Rwanda |
| Area | ~75,000 km² |
| Resolution | 1.07 m (Google Earth, quarterly 2020–2024) |
| Semantic classes | 8 (built-up, road, bare land, low vegetation, water, farmland, forest, wetlands) |
| Annotation | Crowd-sourced + expert validation |
| Supervision | Weakly supervised (GLC products: ESA, ESRI, NLA) |
| Download | [[Request access]](https://github.com/BoazGithub/CSTLF_V1) |

### SECOND *(Yang et al., IEEE TGRS 2023)*

| Property | Value |
|----------|-------|
| Coverage | Multiple Chinese cities |
| Image pairs | 2,968 pairs of 512×512 patches |
| Resolution | 0.3–0.5 m |
| Semantic classes | 6 (non-vegetated ground, tree, low vegetation, water, buildings, playground) |
| Download | [[Official release]](https://captain-whu.github.io/SCD/) |

### LsSCD-Ex *(Tang et al., JAG 2026)*

| Property | Value |
|----------|-------|
| Coverage | Nanjing, China (~471 km²) |
| Image pairs | 100 pairs of 2048×2048 pixels |
| Temporal span | September 2013 – August 2015 |
| Resolution | 0.6 m |
| Semantic classes | 8 (OpenEarthMap-aligned) |
| Download | [[DreamCD repository]](https://github.com/tangkai-RS/DreamCD) |

### Dataset Structure

```
data/
├── sKwandaSCD_V1/
│   ├── train/
│   │   ├── T1/          # Pre-change imagery (512×512 patches)
│   │   ├── T2/          # Post-change imagery (512×512 patches)
│   │   ├── label_from/  # Semantic map at T1
│   │   └── label_to/    # Semantic map at T2
│   ├── val/
│   └── test/
├── SECOND/
│   └── [official structure]
└── LsSCD_Ex/
    └── [official structure]
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/BoazGithub/CSTLF_V1.git
cd CSTLF_V1

# Create and activate environment
conda create -n cstlf python=3.8 -y
conda activate cstlf

# Install dependencies
pip install torch==1.12.0 torchvision==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

### Requirements

```
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
opencv-python>=4.5.0
timm>=0.6.0
einops>=0.4.0
tqdm>=4.62.0
tensorboard>=2.8.0
```

---

## Training

```bash
# Train on sKwandaSCD_V1
python train.py \
    --dataset sKwandaSCD_V1 \
    --data_root ./data/sKwandaSCD_V1 \
    --model CSTLF \
    --epochs 200 \
    --batch_size 8 \
    --lr 1e-4 \
    --weak_supervision \
    --save_dir ./checkpoints/sKwandaSCD_V1

# Train on SECOND
python train.py \
    --dataset SECOND \
    --data_root ./data/SECOND \
    --model CSTLF \
    --epochs 200 \
    --batch_size 8 \
    --lr 1e-4 \
    --save_dir ./checkpoints/SECOND

# Train on LsSCD-Ex
python train.py \
    --dataset LsSCD_Ex \
    --data_root ./data/LsSCD_Ex \
    --model CSTLF \
    --epochs 200 \
    --batch_size 4 \
    --lr 5e-5 \
    --save_dir ./checkpoints/LsSCD_Ex
```

---

## Evaluation

```bash
# Evaluate on sKwandaSCD_V1 (reports mIoU, OA, F1, SeK)
python evaluate.py \
    --dataset sKwandaSCD_V1 \
    --data_root ./data/sKwandaSCD_V1 \
    --checkpoint ./checkpoints/sKwandaSCD_V1/best_model.pth \
    --metrics mIoU OA F1 SeK

# Large-scale inference (full scene, sliding window)
python inference_large_scale.py \
    --image_t1 ./images/kigali_2020.tif \
    --image_t2 ./images/kigali_2024.tif \
    --checkpoint ./checkpoints/sKwandaSCD_V1/best_model.pth \
    --output_dir ./outputs/kigali_large_scale \
    --tile_size 512 \
    --overlap 64
```

---

## Quantitative Results

### Main comparison (mIoU / OA / SeK)

| Method | sKwandaSCD\_V1 | | | SECOND | | | LsSCD-Ex | | |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | mIoU↑ | OA↑ | SeK↑ | mIoU↑ | OA↑ | SeK↑ | mIoU↑ | OA↑ | SeK↑ |
| ClearSCD | 67.3 | 84.2 | 14.1 | 63.8 | 82.1 | 11.8 | 69.1 | 84.7 | 15.2 |
| HRSCD-S1 | 69.7 | 85.8 | 16.3 | 66.2 | 83.7 | 13.7 | 71.2 | 86.3 | 17.4 |
| HRSCD-S2 | 68.9 | 85.1 | 15.8 | 65.4 | 83.2 | 13.2 | 70.8 | 85.9 | 16.9 |
| ChangeMamba | 72.4 | 87.3 | 19.6 | 69.5 | 85.4 | 17.1 | 74.8 | 88.7 | 21.3 |
| ChangeMask | 74.1 | 88.2 | 21.4 | 71.3 | 86.8 | 18.9 | 76.3 | 89.4 | 23.1 |
| SSCD-1 | 75.8 | 89.1 | 23.2 | 73.0 | 87.5 | 20.5 | 78.1 | 90.2 | 25.0 |
| BiSRNet | 76.5 | 89.7 | 24.1 | 74.2 | 88.1 | 21.3 | 79.0 | 90.8 | 25.8 |
| ScanNet | <u>77.2</u> | <u>90.1</u> | <u>25.0</u> | <u>75.1</u> | <u>88.6</u> | <u>22.1</u> | <u>79.7</u> | <u>91.1</u> | <u>26.6</u> |
| **CSTLF (Ours)** | **84.8** | **93.4** | **30.1** | **81.9** | **92.3** | **27.2** | **84.9** | **93.6** | **31.7** |

*Bold: best. Underline: strongest baseline. SeK: Separated Kappa coefficient (Yang et al., 2023) — decouples non-change pixels from semantic change scoring.*

### Computational efficiency

| Method | Params (M) | VRAM (GB) | Latency (ms) | mIoU (avg) |
|--------|:---:|:---:|:---:|:---:|
| ChangeMamba | 124.7 | 12.4 | 145 | 72.0 |
| BiSRNet | 118.9 | 10.8 | 132 | 75.9 |
| ScanNet | 89.3 | 8.6 | 98 | 76.7 |
| SSCD-1 | 76.4 | 7.8 | 89 | 75.6 |
| **CSTLF (Ours)** | **109.4** | **7.2** | **87** | **83.9** |

CSTLF achieves the highest mIoU while consuming the lowest peak VRAM (7.2 GB) and lowest inference latency (87 ms) among all methods with comparable or higher accuracy.

---

## Ablation Study

| Configuration | F1 (%) | Params (M) | VRAM (GB) | Latency (ms) |
|---|:---:|:---:|:---:|:---:|
| Base CNN only | 75.4 | 24.7 | 4.2 | 45 |
| + Attention (DbTSAM) | 77.9 | 28.4 | 5.8 | 54 |
| + LSTM (CTFN) | 80.4 | 35.2 | 6.9 | 73 |
| + Transformer (CTFN) | 82.4 | 35.8 | 7.1 | 62 |
| **Full CSTLF** | **87.2** | **109.4** | **7.2** | **87** |

WS-PLR improves pseudo-label accuracy from 67.3% to 84.7%, filters 31.2% of low-confidence samples, and reduces annotation requirements by 40.3%.

---

## Evaluation Metric: Separated Kappa (SeK)

Standard metrics (OA, κ) are unreliable for SCD due to the dominance of non-change pixels. CSTLF adopts the **SeK coefficient** (Yang et al., IEEE TGRS 2023), which decouples non-change class scoring from semantic change evaluation:

$$\text{SeK} = \frac{\text{IoU}_2(1 - \text{IoU}_1)}{1 - \text{IoU}_1 \cdot \text{IoU}_2}$$

where IoU₁ measures non-change class accuracy and IoU₂ measures semantic change class accuracy collectively. SeK penalizes models that inflate performance by predicting mostly non-change pixels.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{mwubahimana2026cstlf,
  author    = {Mwubahimana, Boaz and Miao, Dingruibo and Jianguo, Yan
               and Ma, Le and Kagoyire, Clarisse and Nsanziyera, Ange Felix
               and Roy, Swalpa Kumar and Tan, Yumin
               and Wang, Ruisheng and Huang, Xiao},
  title     = {CSTLF: From Asymmetric Bi-Temporal Encoding to Semantic
               Transition Mapping in High-Resolution Remote Sensing
               under Weak Supervision},
  journal   = {Under Review},
  year      = {2026}
}
```

*This entry will be updated with full journal details upon acceptance.*

---

## Related Work

| Paper | Venue | Relevance |
|-------|-------|-----------|
| [ASN + SECOND dataset](https://ieeexplore.ieee.org/document/10015667) | IEEE TGRS 2023 | SECOND dataset; SeK metric; asymmetric SCD |
| [DreamCD + LsSCD-Ex](https://doi.org/10.1016/j.jag.2026.105125) | JAG 2026 | LsSCD-Ex dataset; weakly supervised SCD |
| [C2FNet](https://doi.org/10.1109/TGRS.2025.3598681) | IEEE TGRS 2025 | sKwandaSCD\_V1; weak supervision |
| [ChangeMamba](https://doi.org/10.1109/TGRS.2024.3417253) | IEEE TGRS 2024 | State-space SCD baseline |
| [GSTM-SCD](https://doi.org/10.1016/j.isprsjprs.2025.09.003) | ISPRS P&RS 2025 | Graph-enhanced spatio-temporal SCD |

---

## License

This repository is released for **non-commercial and research purposes only**. For commercial applications, please contact the corresponding authors.

---

## Acknowledgments

This work was supported by the State Key Laboratory of Information Engineering in Surveying, Mapping and Remote Sensing (LIESMARS), Wuhan University. The authors thank the National Land Authority of Rwanda (NLA), ESA, and USGS for providing reference datasets, and the providers of the SECOND and LsSCD-Ex benchmarks for enabling comparative evaluation.

---

## Contact

For questions, dataset access requests, or collaboration enquiries, please contact **Boaz Mwubahimana** at [aiboaz1896@gmail.com](mailto:aiboaz1896@gmail.com) or open an issue in this repository.
