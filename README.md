# FedDA - Independent Reimplementation (Updated)

This repository contains an original reimplementation of federated training ideas inspired by the FedDA paper.

## What changed in this reimplementation
- Reimplemented the core federated training flow with a new, modular trainer and aggregation hooks.
- Replaced the simple predictor with a more expressive model architecture (`models/model_def.py`) that combines convolutional feature extraction, an LSTM encoder, and attention pooling to improve temporal modelling capacity.
- Provided preprocessing helpers, dataset instructions

## Dataset
 We have used milano dataset


## Inspired from
C. Zhang, S. Dang, B. Shihada and M. -S. Alouini, "Dual Attention-Based Federated Learning for Wireless Traffic Prediction," IEEE INFOCOM 2021 - IEEE Conference on Computer Communications, 2021, pp. 1-10.

