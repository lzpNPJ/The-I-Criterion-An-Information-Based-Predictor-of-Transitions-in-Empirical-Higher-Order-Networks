# I-Criterion for Explosive Transitions in Higher-Order Networks

## Overview

This repository contains the code and data for the paper:

**"The I-Criterion: An Information-Based Predictor of Explosive Transitions in Empirical Higher-Order Networks"**

The I-criterion is a novel information-based predictor that captures the competition between information propagation and localization. It requires only steady-state hysteresis data and achieves **94.0% average test accuracy** across seven empirical neural networks and three dynamical models.

## Repository Structure
I-criterion-validation/
├── README.md # This file
├── requirements.txt # Python dependencies
├── generate_hysteresis.py # Generate hysteresis data (30 points per dynamics)
├── analyze_I_criterion.py # Main analysis code (I-value calculation, cross-validation, figures)
├── data/
│ └── all_results_paper.json # Network structural parameters (N, ⟨k⟩, T, λ_max) - used only for Table 1
├── results/
│ ├── Figure1.pdf/png # I-value distribution + ROC + confusion matrix
│ ├── Figure2.pdf/png # Network accuracy + per-dynamics heatmap
│ ├── Figure3.pdf/png # I-value vs parameter for all 7 networks
│ └── Table1.tex/csv # Network statistics table
└── paper/
└── main.tex # LaTeX source for the paper


## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt

numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
pandas>=1.3.0
networkx>=2.6.0
scikit-learn>=1.0.0

Usage
Step 1: Generate Hysteresis Data
bash
python generate_hysteresis.py
This script:

Loads the 7 empirical networks from GraphML files

Simulates SIS, Kuramoto, and evolutionary game dynamics

Scans 30 parameter points per dynamics

Saves 21 CSV files (7 networks × 3 dynamics) with columns: [parameter, low_init, high_init]

Output: all_networks_hysteresis_30points_YYYYMMDD_HHMMSS/ directory containing 21 CSV files.

Step 2: Analyze I-Criterion
bash
python analyze_I_criterion.py
This script:

Reads all 21 CSV files

Computes I-values using $\mathcal{I} = \log(v/H)$

Performs leave-one-network-out cross-validation

Generates all figures and tables

Output:

Figure1.pdf/png - I-value distribution, ROC curve, confusion matrix

Figure2.pdf/png - Network accuracy bar chart, per-dynamics heatmap

Figure3.pdf/png - I-value vs parameter for all 7 networks

Table1.tex/csv - Network statistics (N, ⟨k⟩, T, λ_max, Accuracy)

Step 3: Reproduce Paper Results
The analysis results are saved in the same directory as the CSV files. Key results:

Metric	Value
Mean test accuracy	94.0% ± 6.2%
Total samples	630 (7 × 3 × 30)
Optimal threshold I_c	≈ 1.25 (consistent across networks)
Data Sources
Empirical Networks
Network	Nodes	Source
C. elegans pharynx	279	NeuroData
Mixed species brain	65	NeuroData
Mouse visual cortex 1	29	NeuroData
Mouse visual cortex 2	195	NeuroData
P. pacificus synaptic	54	NeuroData
Rhesus brain 2	91	NeuroData
Rhesus cerebral cortex 1	91	NeuroData
Structural Parameters
The file data/all_results_paper.json contains network structural parameters (N, ⟨k⟩, T, λ_max) extracted from previous simulations. These are used only for Table 1 and are independent of the I-criterion calculations.

Citation
If you use this code in your research, please cite:

bibtex
@article{li2026Icrit,
  title={The I-Criterion: An Information-Based Predictor of Explosive Transitions in Empirical Higher-Order Networks},
  author={Li, Zhenpeng and Yan, Zhihua and Tang, Xijin},
  journal={New Journal of Physics},
  year={2026},
  note={Submitted}
}
License
This code is released under the MIT License.

Contact
For questions or issues, please contact: lizhenpeng@amss.ac.cn

