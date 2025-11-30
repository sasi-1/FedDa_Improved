# Federated Reimplementation (FedDA reimplementation)
This repository is an independent reimplementation of concepts from the FedDA paper.
It is intentionally restructured with different code organization, function/class names,
and additional hooks so the work is clearly an original implementation for academic submission.

## Highlights
- Reimplemented federated training loop (trainer/train_federated.py).
- Separate aggregation module with multiple strategies (trainer/aggregation.py).
- Client sharding helpers and a demo experiment (experiments/run_experiment.py).
- Do not use this demo as-is for final evaluation. Replace synthetic data with your dataset.

## How to run (example)
```bash
python3 -m pip install -r requirements.txt
python experiments/run_experiment.py
```
