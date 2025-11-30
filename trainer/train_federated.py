"""Federated training loop 
Keep this file as the main training loop module.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import copy
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

@dataclass
class TrainConfig:
    rounds: int = 20
    fraction: float = 0.3
    local_epochs: int = 2
    local_batch: int = 32
    lr: float = 1e-3
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42

@dataclass
class ClientUpdate:
    weights: Dict[str, torch.Tensor]
    loss: float
    num_examples: int

class SimplePredictor(nn.Module):
    def __init__(self, input_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)

def run_local_training(seed: int, model: nn.Module,
                       data: torch.Tensor, targets: torch.Tensor,
                       cfg: TrainConfig) -> ClientUpdate:
    torch.manual_seed(seed)
    local_model = copy.deepcopy(model).to(cfg.device)
    local_model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(local_model.parameters(), lr=cfg.lr)

    dataset = TensorDataset(data, targets)
    loader = DataLoader(dataset, batch_size=cfg.local_batch, shuffle=True)

    epoch_losses = []
    for _ in range(cfg.local_epochs):
        batch_losses = []
        for xb, yb in loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            optimizer.zero_grad()
            out = local_model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        epoch_losses.append(sum(batch_losses) / max(1, len(batch_losses)))

    avg_loss = sum(epoch_losses) / max(1, len(epoch_losses))
    return ClientUpdate(weights=local_model.state_dict(),
                        loss=float(avg_loss),
                        num_examples=len(dataset))

def fedavg_aggregate(updates: List[ClientUpdate]) -> Dict[str, torch.Tensor]:
    total_examples = sum(u.num_examples for u in updates)
    if total_examples == 0:
        raise ValueError("No examples present in updates")
    agg = {}
    for k in updates[0].weights.keys():
        agg[k] = sum((u.weights[k].float() * (u.num_examples / total_examples)) for u in updates)
    return agg

def simple_attention_aggregate(updates: List[ClientUpdate], temp: float = 1.0) -> Dict[str, torch.Tensor]:
    import torch as _torch
    losses = _torch.tensor([u.loss for u in updates], dtype=_torch.float32)
    scores = _torch.softmax((-losses / temp), dim=0)
    agg = {}
    for k in updates[0].weights.keys():
        stacked = _torch.stack([u.weights[k].float() for u in updates], dim=0)
        # ensure broadcasting works for parameters of different dims
        weight_shape = [ -1 ] + [1]*(stacked.dim()-2)
        agg[k] = (scores.view(*weight_shape) * stacked).sum(dim=0)
    return agg

def federated_train(global_model: nn.Module,
                    all_client_data: List[Tuple[torch.Tensor, torch.Tensor]],
                    cfg: TrainConfig,
                    aggregator: Callable[[List[ClientUpdate]], Dict[str, torch.Tensor]] = fedavg_aggregate,
                    parallel_clients: int = 4) -> nn.Module:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    num_clients = len(all_client_data)
    if num_clients == 0:
        raise ValueError("Need at least one client")

    for rnd in range(1, cfg.rounds + 1):
        m = max(1, int(cfg.fraction * num_clients))
        selected = random.sample(range(num_clients), k=m)

        client_updates: List[ClientUpdate] = []
        with ThreadPoolExecutor(max_workers=parallel_clients) as pool:
            futures = []
            for cid in selected:
                client_x, client_y = all_client_data[cid]
                fut = pool.submit(run_local_training, cfg.seed + rnd + cid,
                                  copy.deepcopy(global_model), client_x, client_y, cfg)
                futures.append(fut)
            for fut in as_completed(futures):
                client_updates.append(fut.result())

        aggregated_state = aggregator(client_updates)
        global_model.load_state_dict(aggregated_state)

        round_loss = sum(u.loss * u.num_examples for u in client_updates) / sum(u.num_examples for u in client_updates)
        print(f"[Round {rnd}/{cfg.rounds}] selected={m} clients, weighted_loss={round_loss:.4f}")

    return global_model

if __name__ == "__main__":
    # quick smoke run with synthetic data
    n_clients = 8
    all_data = []
    for i in range(n_clients):
        X = torch.randn(80, 8)
        y = (X.sum(dim=1, keepdim=True) + 0.1*torch.randn(80,1))
        all_data.append((X, y))

    model = SimplePredictor(input_dim=8, hidden=16, out_dim=1)
    cfg = TrainConfig(rounds=5, fraction=0.5, local_epochs=1)
    _ = federated_train(model, all_data, cfg, aggregator=fedavg_aggregate)
