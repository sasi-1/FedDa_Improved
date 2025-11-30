"""Client data utilities and simple partitioning helpers.
Use this to partition datasets into client shards for experiments.
"""
from typing import List, Tuple
import torch

def shard_dataset(X: torch.Tensor, y: torch.Tensor, num_shards: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    # naive equal-size shard; you can replace with non-iid splits
    n = X.size(0)
    per = n // num_shards
    shards = []
    for i in range(num_shards):
        start = i*per
        end = start + per if i < num_shards-1 else n
        shards.append((X[start:end].clone(), y[start:end].clone()))
    return shards
