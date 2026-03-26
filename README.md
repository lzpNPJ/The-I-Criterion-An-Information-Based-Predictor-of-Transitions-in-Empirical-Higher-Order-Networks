# The-I-Criterion-An-Information-Based-Predictor-of-Transitions-in-Empirical-Higher-Order-Networks
This repository contains code for validating the I-criterion as a predictor of explosive (first-order) phase transitions in complex networks with higher-order interactions. The I-criterion quantifies the sharpness of transitions and effectively distinguishes between continuous and explosive dynamics.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# I-Criterion Validation on Empirical Hypergraphs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Validation of I-criterion for detecting explosive transitions in higher-order networks.

## Overview

This repository contains code for validating the I-criterion as a predictor of explosive (first-order) phase transitions in complex networks with higher-order interactions. The I-criterion quantifies the sharpness of transitions and effectively distinguishes between continuous and explosive dynamics.

### Key Features
- **7 real-world hypergraph networks** (C. elegans, mouse, rhesus monkey, etc.)
- **3 dynamical processes**: SIS epidemic, Kuramoto synchronization, Evolutionary game theory
- **630 data samples** (7 networks × 3 dynamics × 30 parameter points)
- **Leave-one-network-out cross-validation**
- **Publication-ready figures** (PDF/PNG) for journal submission

## Files

### `NJP_hypersis_7_3.py`
**Hysteresis data generator** - Generates raw hysteresis data for all 7 networks across 3 dynamical processes.

- Loads GraphML network files and extracts maximal cliques (size ≥ 3) as hyperedges
- Builds 3-body tensors (W₃) and projected adjacency matrices
- Simulates forward/backward dynamics with 30 parameter points per dynamics
- Uses parallel processing (6 workers) for efficient computation
- Outputs: CSV files for each network/dynamics combination

**Output format:**
hysteresis_sis.csv # λ₁ parameter, low_init, high_init
hysteresis_kuramoto.csv # σ₁ parameter, low_init, high_init
hysteresis_game.csv # r parameter, low_init, high_init

text

### `NJP.py`
**Analysis and visualization** - Computes I-criterion and generates all publication figures.

- Computes I-value for each sample (I = log(v_info / H))
- Implements leave-one-network-out cross-validation
- Finds optimal I_c thresholds and evaluates classification accuracy
- Generates 3 main figures + 1 table:

**Figure 1**: I-value distribution (boxplot), ROC curve, confusion matrix
**Figure 2**: Network-level accuracy bar chart + per-dynamics heatmap
**Figure 3**: I-value vs control parameter scatter plots (all 7 networks)
**Table 1**: Network statistics (N, ⟨k⟩, T, λ_max) and test accuracy

## Installation

```bash
git clone https://github.com/yourusername/I-criterion-validation.git
cd I-criterion-validation
pip install numpy scipy matplotlib pandas networkx scikit-learn
Usage
1. Generate hysteresis data
bash
python NJP_hypersis_7_3.py
Important: Update network file paths in the script:

python
networks = {
    'C.elegans_pharynx': r'your_path/c.elegans.herm_pharynx_1.graphml',
    # ... update other paths
}
2. Analyze results and generate figures
bash
python NJP.py
Important: Update data paths in the script:

python
DATA_PATH = r'path/to/your/hysteresis_data'
JSON_PATH = r'path/to/your/structural_params.json'
Data Format
Input
GraphML files: 7 empirical hypergraph networks (not included)

JSON file: Structural parameters (N, ⟨k⟩, T, λ_max) for each network

Output
CSV files: Hysteresis data (30 points per dynamics)

Figures: Figure1.pdf/png, Figure2.pdf/png, Figure3.pdf/png

Table: Table1.tex/csv

Results Summary
The I-criterion achieves:

Mean test accuracy: ~85% (leave-one-network-out)

AUC: ~0.92

Consistent performance across all 3 dynamical models

Citation
If you use this code in your research, please cite:

bibtex
@article{your_paper_2024,
  title={I-Criterion Validation on Empirical Hypergraphs},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
License
MIT License - see LICENSE for details.

Contact
For questions or issues, please open an issue on GitHub.

text

## `requirements.txt`

```txt
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.4.0
pandas>=1.3.0
networkx>=2.6.0
scikit-learn>=0.24.0
.gitignore
gitignore
# Python
__pycache__/
*.py[cod]
*.so
.Python
env/
venv/
.venv/

# Data files (not to be committed)
*.graphml
*.csv
*.json

# Results
*.pdf
*.png
*.tex

# IDE
.vscode/
.idea/
.DS_Store
仓库结构
text
I-criterion-validation/
├── README.md                      # 项目说明
├── requirements.txt               # 依赖包
├── .gitignore                     # Git忽略规则
├── LICENSE                        # MIT许可证
├── NJP_hypersis_7_3.py           # 数据生成脚本
└── NJP.py                         # 分析绘图脚本


