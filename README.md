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
2. `MBOONE` — Prototype on simulated LArTPC data sourced by the microBoone open samples: https://github.com/uboone/OpenSamples/tree/main. Following the procedures developed by https://github.com/NevisRAD/RAD4LArTPC/tree/main and https://arxiv.org/pdf/2509.21817 

### tutorial/MNIST 
The MNIST tutorial notebooks will study the reconstruction losses of digital MNIST data and a practice anomaly detection notebook for a set of testing data with anomalous features (i.e. noise) added to it. The goal of this notebook is to provide a solid foundation for key fundamentals of anomaly detection and autoencoders. This is skippable for anyone focussing only on the applications for neutrino detection.


### tutorial/MBOONE
The MBOONE tutorial notebooks and scripts will study the procedure of anomaly detection for using wire data from the Open Samples repository from Fermilab. The core idea is that we will build an unsupervised autoencoder following the RAD4HEP group at Columbia: https://github.com/NevisRAD/RAD4LArTPC/tree/main, and build a solely "teacher" autoencoder to read wire pixel data from the microbooNe open samples. The goal is to prototype with accessible monte-carlo data and build an unsupervised model which should follow their "teacher" autoencoder very closely. Any relevant tests and studies will be found in this folder. 

Any more advanced studies that examine or isolate neutrino events and omit pure space, search for specific topological events, will be found in the main project folder under the main_project/MBOONE folder, where the more original studies will go.

However, before toying around with this autoencoder, please practice with the OpenSamples repository and the notebooks provided. Through anaconda also ensure you have the `ubopendata` environment prepared. You may need additional libraries and downloads prior to using every notebook provided here beyond the ones downloaded through the `ubopendata` envioronment.

## Main project (main_project folder)
1. `MBOONE` - Additional studies on microBoone data from simulated LArTPC data sourced by the microBoone open samples: https://github.com/uboone/OpenSamples/tree/main. Focussing on specfic physical searches such as neutrino vertices, di-muon, eta meson, and lambda baryon production in microBooNe.
2. `SBND` - Anomaly detection to identify di-muon, eta meson, and lambda baryon production in SBND.

## Funding
This work was funded graciously by the Worster family.
