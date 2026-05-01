from copy import deepcopy
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: str,
        learning_rate: float,
        weight_decay: float,
        early_stopping_patience: int,
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.early_stopping_patience = early_stopping_patience

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ) -> List[Dict[str, float]]:
        best_state = None
        best_val_loss = float("inf")
        patience_count = 0
        history: list[dict[str, float]] = []

        for epoch in range(1, epochs + 1):
            train_loss = self._train_one_epoch(train_loader)
            val_loss = self.evaluate_loss(val_loader)

            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = deepcopy(self.model.state_dict())
                patience_count = 0
            else:
                patience_count += 1

            if patience_count >= self.early_stopping_patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    def _train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        total_count = 0

        for X, y in loader:
            X = X.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            pred = self.model(X)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()

            batch_size = X.shape[0]
            total_loss += loss.item() * batch_size
            total_count += batch_size

        return total_loss / max(1, total_count)

    @torch.no_grad()
    def evaluate_loss(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        total_count = 0

        for X, y in loader:
            X = X.to(self.device)
            y = y.to(self.device)
            pred = self.model(X)
            loss = self.criterion(pred, y)

            batch_size = X.shape[0]
            total_loss += loss.item() * batch_size
            total_count += batch_size

        return total_loss / max(1, total_count)

    @torch.no_grad()
    def predict_loader(self, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        preds = []
        targets = []
        for X, y in loader:
            X = X.to(self.device)
            pred = self.model(X).cpu()
            preds.append(pred)
            targets.append(y.cpu())
        return torch.cat(targets, dim=0), torch.cat(preds, dim=0)
