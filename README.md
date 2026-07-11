# SCARAB- Standard-Candle Autoencoder for Rare Anomaly Benchmarking
<img width="640" height="400" alt="Untitled design (1)" src="https://github.com/user-attachments/assets/021fd1ca-4432-4381-8346-81e43df2e323" />



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
- `tutorial/` - Tutorial notebooks (MNIST, MicroBoone open dataset)
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
2. `MBOONE` — Prototype on simulated LArTPC data sourced by the microBoone open samples: https://github.com/uboone/OpenSamples/tree/main 

## Anomaly Detection
1. `SBND_anomaly_detection.ipynb`- WIP anomaly detection

## Funding
This work was graciously funded graciously by the Worster family.
