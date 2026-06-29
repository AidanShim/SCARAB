# SCARAB-
Convolutional autoencoder framework for identifying rare Standard Model 
processes in LArTPC neutrino detector simulation data. Developed as part 
of the Worster Summer Research Fellowship and senior thesis project in 
the Caratelli group at UC Santa Barbara.

## Project Overview
SCARAB applies unsupervised anomaly detection to SBND Monte Carlo data,
targeting eta meson production, lambda baryon production, and di-muon 
events. The framework builds on the FIREFLY methodology 
(Chung et al., arXiv:2509.21817).

## Repository Structure
- `notebooks/` — Tutorial notebooks and analysis
- `scarab/` — Core Python package (models, scoring, utilities)
- `data/` — Local data storage (not tracked by git)
- `results/` — Output plots and metrics
- `docs/` — Documentation and references

## Setup
conda create -n scarab python=3.11
conda activate scarab
pip install -r requirements.txt

## Tutorials
1. `01_mnist_anomaly_detection.ipynb` — Proof of concept on MNIST
2. `02_microboone_open_data.ipynb` — Prototype on real LArTPC data
3. `03_sbnd_analysis.ipynb` — Full SBND analysis (in progress)
