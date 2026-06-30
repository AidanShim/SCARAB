# SCARAB- Standard-Candle Autoencoder for Rare Anomaly Benchmarking

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
- `notebooks/` — Analysis notebooks
- `tutorial/` - Tutorial notebooms (MNIST, MicroBoone open dataset)
- `scarab/` — Core Python package (models, scoring, utilities)
- `data/` — Local data storage (not tracked by git)
- `results/` — Output plots and metrics
- `docs/` — Documentation and references

## Setup

conda create -n scarab python=3.11

conda activate scarab

### For CPU-only (most users):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

### For GPU (if you have CUDA):
pip install -r requirements.txt

## Tutorial folders
1. `MNIST` — Proof of concept on MNIST data
2. `MBOONE` — Prototype on real LArTPC data

## Anomaly Detection
1. `SBND_anomaly_detection.ipynb`- WIP anomaly detection
