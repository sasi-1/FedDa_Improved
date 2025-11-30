# FedDA - Independent Reimplementation with some improvements
This repository contains an Implementation of federated training ideas inspired by the FedDA paper.

## What changed in this reimplementation
- Implemented the core federated training flow with a new, modular trainer and aggregation hooks.
- Replaced the simple predictor with a more expressive model architecture (`models/model_def.py`) that combines convolutional feature extraction, an LSTM encoder, and attention pooling to improve temporal modelling capacity.
- Provided preprocessing helpers, dataset instructions
  <img width="2012" height="1764" alt="system_diagram" src="https://github.com/user-attachments/assets/eee1ce32-c6e6-4d40-b3e2-1b2d79666f50" />
 

## Dataset
 We have used milano dataset


## Inspired from
C. Zhang, S. Dang, B. Shihada and M. -S. Alouini, "Dual Attention-Based Federated Learning for Wireless Traffic Prediction," IEEE INFOCOM 2021 - IEEE Conference on Computer Communications, 2021, pp. 1-10.

