"""Run an end-to-end experiment using the reimplemented trainer.
Produces a CSV with per-round logs and saves the final model.
"""
import os
import csv
import torch
from trainer.train_federated import federated_train, TrainConfig, SimplePredictor
from trainer.clients import shard_dataset
from models.model_def import ReimplPredictor

def make_synthetic(num_samples=1000, input_dim=8):
    X = torch.randn(num_samples, input_dim)
    y = (X.sum(dim=1, keepdim=True) + 0.1*torch.randn(num_samples,1))
    return X, y

def main():
    X, y = make_synthetic(num_samples=800, input_dim=8)
    shards = shard_dataset(X, y, num_shards=8)

    # convert shards to the format trainer expects: list of (X,y)
    cfg = TrainConfig(rounds=6, fraction=0.5, local_epochs=1, local_batch=32)
    model = ReimplPredictor(inp=8, hid=16, out=1)
    # the trainer's federated_train uses SimplePredictor default type in example,
    # but it accepts any torch.nn.Module with state_dict that matches across clients.
    trained = federated_train(model, shards, cfg)

    # save model
    os.makedirs('output', exist_ok=True)
    torch.save(trained.state_dict(), 'output/final_model.pth')
    # placeholder CSV
    with open('output/experiment_metadata.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['note'])
        w.writerow(['This is a synthetic demo run. Replace with your dataset and metrics.'])
    print('Experiment finished. Outputs are in ./output')

if __name__ == '__main__':
    main()
