<div align="center">

<h1><b>CSTLF: From Asymmetric Bi-Temporal Encoding to Semantic Transition Mapping<br>in High-Resolution Remote Sensing under Weak Supervision</b></h1>

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
</p>

</div>

---

<h2 align="left">Authors</h2>

**Boaz Mwubahimana**<sup>1</sup>,
Dingruibo Miao<sup>1*</sup>,
Yan Jianguo<sup>1*</sup>,
Le Ma<sup>1</sup>,
Clarisse Kagoyire<sup>2</sup>,
Ange Felix Nsanziyera<sup>3</sup>,
Swalpa Kumar Roy<sup>4</sup>,
Yumin Tan<sup>5</sup>,
Ruisheng Wang<sup>6</sup>,
Xiao Huang<sup>7</sup>

<sup>1</sup> LIESMARS, Wuhan University &nbsp;|&nbsp;
<sup>2</sup> CGIS, University of Rwanda &nbsp;|&nbsp;
<sup>3</sup> INES Ruhengeri &nbsp;|&nbsp;
<sup>4</sup> Tezpur University &nbsp;|&nbsp;
<sup>5</sup> Beihang University &nbsp;|&nbsp;
<sup>6</sup> Shenzhen University &nbsp;|&nbsp;
<sup>7</sup> Emory University

<sup>*</sup> Corresponding authors:
[miaodrb@whu.edu.cn](mailto:miaodrb@whu.edu.cn) &nbsp;|&nbsp;
[jgyan@whu.edu.cn](mailto:jgyan@whu.edu.cn)

---

## Updates

| | |
|---|---|
| :zap: | April 2026: CSTLF manuscript submitted for peer review at ISPRS Journal of Photogrammetry and Remote Sensing. |
| :zap: | April 2026: sKwandaSCD\_V1 dataset and codebased is available and pretrained weights will be available upon review finals. |

---

## Overview

This study introduces the **Cross-Spatio-Temporal Learning Framework (CSTLF)**, a unified architecture for semantic land-cover change detection. To address the limitations of symmetric bi-temporal encoders and dense annotation bottlenecks, CSTLF harmonizes **CNN-based spatial feature extraction**, **LSTM-based sequential state propagation**, and **Transformer-driven global attention** within a single end-to-end architecture. By leveraging weakly supervised pseudo-labels derived from low-resolution global land-cover products (GLC), the model achieves high-fidelity semantic transition mapping without requiring dense pixel-level annotations.

CSTLF and the **sKwandaSCD\_V1** dataset (Nyagatare, Kigali) can be accessed [[here](https://github.com/BoazGithub/CSTLF_V1)].

---

## Graphical Abstract

> Graphical abstract illustrating the synergistic multi-scale and multi-temporal
> feature extraction within the CSTLF architecture. It highlights the integration
> of CNN, LSTM, and Transformer branches to resolve spatial heterogeneity and
> temporal dependencies in high-resolution remote sensing imagery.
<img width="863" height="259" alt="image" src="https://github.com/user-attachments/assets/c0f0019b-2f4d-422a-8482-7a1f96d09415" />


## Requirements

> Full environment details and pre-trained weights will be released upon
> acceptance of the associated manuscript.

---

## Description

### Study Area

The Cross-Spatio-Temporal Learning Framework (CSTLF) was trained and validated
over the **Nyagatare agricultural district** and the **Kigali urban corridor**
in Rwanda. Cross-dataset generalization was evaluated on the **SECOND** and
**LsSCD-Ex** benchmarks to assess robustness under diverse geographic and
imaging conditions.

### Study Area — Data Descriptions

| Image Ref. | Site | Acquisition Period | GT Date |
|:---:|:---|:---:|:---:|
| Img (1) | Kigali City | 2020 – 2024 | 2024-05-15 |
| Img (2) | Nyagatare | 2021 – 2023 | 2023-11-20 |
| Img (3) | SECOND (Chinese cities) | 2020 – 2022 | — |
| Img (4) | LsSCD-Ex (Nanjing) | 2013 – 2015 | — |

---

## CSTLF Architecture

The CSTLF architecture integrates four core components:

- **Multi-Scale Feature Fusion Network (MSFFN)** — hierarchical spatial
  representation at fine, medium, and coarse resolutions via adaptive
  attention weighting.
- **Dual-branch Temporal–Spatial Attention Mechanism (DbTSAM)** — joint
  intra-temporal spatial attention and inter-temporal dependency modeling,
  designed to handle asymmetric land-cover distributions across acquisition
  epochs.
- **Cross-Temporal Fusion Network (CTFN)** — parallel LSTM sequential memory
  and Transformer global reasoning, fused via an adaptive gating mechanism.
- **Weakly Supervised Pseudo-Label Refinement (WS-PLR)** — confidence-guided
  iterative pseudo-label correction that suppresses non-change label dominance
  and noise from coarse GLC supervision.

These components utilize cross-attention and residual learning to suppress
stochastic noise while preserving sharp semantic transition boundaries in
heterogeneous landscapes.

### Algorithm Flow

```
HR Multi-Temporal Input (T1, T2)
         │
         ▼
   ┌───────────┐
   │   MSFFN   │  Fine / Medium / Coarse branches → Adaptive fusion
   └─────┬─────┘
         │
         ▼
   ┌───────────┐
   │  DbTSAM   │  Spatial branch ⊗ Temporal branch → Joint attention
   └─────┬─────┘
         │
         ▼
   ┌───────────┐
   │   CTFN    │  LSTM + Transformer → Adaptive gating → Fused features
   └─────┬─────┘
         │
         ▼
   ┌───────────┐
   │  WS-PLR   │  Confidence filtering → Pseudo-label refinement
   └─────┬─────┘
         │
         ▼
   Semantic Change Map + Uncertainty Maps
```

### CSTLF Network Workflows

Full network diagrams are provided in `figures/`:

- `CSTLF_framework_design4.pdf` — overall tri-branch architecture
- `MSFFN.pdf` — multi-scale feature fusion design
- `CNN_LSTM_V2.pdf` — DbTSAM sequential modeling
- `cross_temporal_CNN_Transformer_network_V6.pdf` — CTFN design

---

## Results

### Tracking Training Progress

Convergence is monitored across **mIoU**, **OA**, **SeK**, and **F1-score**
metrics. The tri-branch fusion ensures stable loss decline while maximizing
semantic change detection performance across both urban and agricultural classes.

### Feature Pattern Recognition

Visualization of the feature flow demonstrating how MSFFN and DbTSAM improve
pattern recognition for multi-scale global and local dependencies across
heterogeneous land-cover distributions.

---

## Quantitative Results

### Main Comparison (mIoU / OA / SeK)

| Method | sKwandaSCD\_V1 mIoU↑ | OA↑ | SeK↑ | SECOND mIoU↑ | OA↑ | SeK↑ | LsSCD-Ex mIoU↑ | OA↑ | SeK↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ClearSCD | 67.3 | 84.2 | 14.1 | 63.8 | 82.1 | 11.8 | 69.1 | 84.7 | 15.2 |
| HRSCD-S1 | 69.7 | 85.8 | 16.3 | 66.2 | 83.7 | 13.7 | 71.2 | 86.3 | 17.4 |
| HRSCD-S2 | 68.9 | 85.1 | 15.8 | 65.4 | 83.2 | 13.2 | 70.8 | 85.9 | 16.9 |
| ChangeMamba | 72.4 | 87.3 | 19.6 | 69.5 | 85.4 | 17.1 | 74.8 | 88.7 | 21.3 |
| ChangeMask | 74.1 | 88.2 | 21.4 | 71.3 | 86.8 | 18.9 | 76.3 | 89.4 | 23.1 |
| SSCD-1 | 75.8 | 89.1 | 23.2 | 73.0 | 87.5 | 20.5 | 78.1 | 90.2 | 25.0 |
| BiSRNet | 76.5 | 89.7 | 24.1 | 74.2 | 88.1 | 21.3 | 79.0 | 90.8 | 25.8 |
| ScanNet | <u>77.2</u> | <u>90.1</u> | <u>25.0</u> | <u>75.1</u> | <u>88.6</u> | <u>22.1</u> | <u>79.7</u> | <u>91.1</u> | <u>26.6</u> |
| **CSTLF (Ours)** | **84.8** | **93.4** | **30.1** | **81.9** | **92.3** | **27.2** | **84.9** | **93.6** | **31.7** |

*Bold: best. Underline: strongest baseline.*
*SeK: Separated Kappa (Yang et al., IEEE TGRS 2023) — decouples non-change dominance from semantic change scoring.*

---

## Qualitative Results

### State-of-the-Art Comparison

Qualitative comparisons of CSTLF against all eight comparison methods are
provided across all three benchmarks, including:

- Semantic change maps and changed-pixel binary masks
- Probability threshold and uncertainty maps from WS-PLR inference
- Large-scale predictions over Kigali (Kimihurura and Kanombe districts, 0.5 m)
- Few-shot predictions under noisy pseudo-labels from GLC products (ESA, ESRI, NLA)

### Comparison Methods (Paper Baselines)

📖 📖 📖

| Method | Reference |
|:---|:---|
| ClearSCD | [Béchaz et al., ISPRS P&RS 2026](https://doi.org/10.1016/j.isprsjprs.2025.11.024) |
| HRSCD-S1 / HRSCD-S2 | [Daudt et al., ICIP 2018](https://doi.org/10.1109/ICIP.2018.8451652) |
| ChangeMamba | [Chen et al., IEEE TGRS 2024](https://doi.org/10.1109/TGRS.2024.3417253) |
| ChangeMask | [Zhu & Wu, IJCSIT 2025](https://doi.org/10.62051/ijcsit.v5n2.08) |
| SSCD-1 | [Zheng et al., IGARSS 2021](https://doi.org/10.1109/IGARSS47720.2021.9553768) |
| BiSRNet | [Chen et al., IEEE TGRS 2021](https://doi.org/10.1109/TGRS.2021.3095166) |
| ScanNet | [Robinson et al., CVPR 2019](https://doi.org/10.1109/CVPR.2019.01301) |

📖 📖 📖

---

## Dataset Preparation

### sKwandaSCD\_V1 Dataset Overview

The **sKwandaSCD\_V1** dataset consists of **512 × 512** pixel bi-temporal
patches at **1.07 m** ground sampling distance, acquired from Google Earth
quarterly between 2020 and 2024 over three major metropolitan areas in Rwanda
(Kigali, Nyagatare, Bugesera). It provides **8 semantic change categories**:
built-up, road/impervious, bare land, low vegetation, water, farmland, forest,
and wetlands. Weak supervision is derived from aggregated GLC products (ESA,
ESRI, NLA).

### Dataset Structure

```
data/
├── sKwandaSCD_V1/
│   ├── train/
│   │   ├── T1/      # Pre-change imagery (512×512 patches)
│   │   ├── T2/      # Post-change imagery (512×512 patches)
│   │   └── label/   # Weakly supervised pseudo-labels
│   ├── val/
│   └── test/
├── SECOND/
│   └── [official structure — see Yang et al. 2023]
└── LsSCD_Ex/
    └── [official structure — see Tang et al. 2026]
```

### Benchmark Datasets

| Dataset | Source | Download |
|:---|:---|:---|
| sKwandaSCD\_V1 | This work | [[Request access]](https://github.com/BoazGithub/CSTLF_V1) |
| SECOND | Yang et al., IEEE TGRS 2023 | [[Official]](https://captain-whu.github.io/SCD/) |
| LsSCD-Ex | Tang et al., JAG 2026 | [[DreamCD repo]](https://github.com/tangkai-RS/DreamCD) |

---

## Citation

If you use CSTLF or the sKwandaSCD\_V1 dataset in your research, please cite:

```bibtex
@article{mwubahimana2026cstlf,
  author  = {Mwubahimana, Boaz and Miao, Dingruibo and Jianguo, Yan
             and Ma, Le and Kagoyire, Clarisse and Nsanziyera, {Ange Felix}
             and Roy, {Swalpa Kumar} and Tan, Yumin
             and Wang, Ruisheng and Huang, Xiao},
  title   = {CSTLF: From Asymmetric Bi-Temporal Encoding to Semantic
             Transition Mapping in High-Resolution Remote Sensing
             under Weak Supervision},
  journal = {Under Review},
  year    = {2026}
}
```

*This entry will be updated with full journal and DOI details upon acceptance.*

---

## Contact

For questions, dataset access requests, or collaboration enquiries, please
reach out to **Boaz Mwubahimana** at
[aiboaz1896@gmail.com](mailto:aiboaz1896@gmail.com)
or open an issue in this repository.

---

## License

The code and datasets are released for **non-commercial and research purposes
only**. For commercial applications, please contact the corresponding authors.

---

## Acknowledgement

This work was supported by the **Planetary Science Research Group** at the
State Key Laboratory of Information Engineering in Surveying, Mapping and
Remote Sensing (LIESMARS), **Wuhan University**. The authors thank the
National Land Authority of Rwanda (NLA), ESA, and USGS for providing
satellite imagery and reference datasets, and the providers of the SECOND
and LsSCD-Ex benchmarks for enabling rigorous comparative evaluation.

