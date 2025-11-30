"""Aggregation strategies for federated updates.
Keep this file if you want to add new aggregators without touching the main loop.
"""
from typing import List, Dict
from dataclasses import dataclass
import torch

@dataclass
class ClientUpdate:
    weights: Dict[str, torch.Tensor]
    loss: float
    num_examples: int

def fedavg(updates: List[ClientUpdate]) -> Dict[str, torch.Tensor]:
    total = sum(u.num_examples for u in updates)
    agg = {}
    for k in updates[0].weights.keys():
        agg[k] = sum((u.weights[k].float() * (u.num_examples / total)) for u in updates)
    return agg

def loss_weighted(updates: List[ClientUpdate], temperature: float = 1.0) -> Dict[str, torch.Tensor]:
    losses = torch.tensor([u.loss for u in updates], dtype=torch.float32)
    scores = torch.softmax((-losses / temperature), dim=0)
    agg = {}
    for k in updates[0].weights.keys():
        stacked = torch.stack([u.weights[k].float() for u in updates], dim=0)
        weight_shape = [ -1 ] + [1]*(stacked.dim()-2)
        agg[k] = (scores.view(*weight_shape) * stacked).sum(dim=0)
    return agg
