from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass(slots=True)
class TrainConfig:
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 0.001
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4


@dataclass(slots=True)
class EpochStats:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float | None
    val_accuracy: float | None


@dataclass(slots=True)
class TrainResult:
    best_epoch: int
    best_val_loss: float | None
    epochs_ran: int
    stopped_early: bool
    history: list[EpochStats]

    def to_dict(self) -> dict:
        out = asdict(self)
        out["history"] = [asdict(h) for h in self.history]
        return out


def _run_eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x_batch, y_batch in tqdm(
            loader,
            desc="eval batches",
            unit="batch",
            leave=False,
        ):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            preds = torch.argmax(logits, dim=1)
            batch_size = int(y_batch.shape[0])
            total_samples += batch_size
            total_correct += int((preds == y_batch).sum().item())
            total_loss += float(loss.item()) * batch_size

    if total_samples == 0:
        return 0.0, 0.0

    return total_loss / total_samples, total_correct / total_samples


def _run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x_batch, y_batch in tqdm(
        loader,
        desc="train batches",
        unit="batch",
        leave=False,
    ):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        batch_size = int(y_batch.shape[0])
        total_samples += batch_size
        total_correct += int((preds == y_batch).sum().item())
        total_loss += float(loss.item()) * batch_size

    if total_samples == 0:
        return 0.0, 0.0

    return total_loss / total_samples, total_correct / total_samples


def _is_improved(
    best_score: float | None,
    current_score: float,
    min_delta: float,
) -> bool:
    if best_score is None:
        return True
    return current_score < (best_score - min_delta)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    config: TrainConfig,
) -> TrainResult:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: list[EpochStats] = []
    best_epoch = 1
    best_val_loss: float | None = None
    best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
    no_improve = 0

    epochs = max(1, int(config.epochs))

    for epoch in tqdm(
        range(1, epochs + 1),
        desc="train epochs",
        unit="epoch",
    ):
        train_loss, train_acc = _run_train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss: float | None = None
        val_acc: float | None = None
        score_for_early_stop = train_loss

        if val_loader is not None:
            val_loss, val_acc = _run_eval_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )
            score_for_early_stop = val_loss

        history.append(
            EpochStats(
                epoch=epoch,
                train_loss=float(train_loss),
                train_accuracy=float(train_acc),
                val_loss=float(val_loss) if val_loss is not None else None,
                val_accuracy=float(val_acc) if val_acc is not None else None,
            )
        )

        improved = _is_improved(
            best_score=best_val_loss,
            current_score=float(score_for_early_stop),
            min_delta=float(config.early_stopping_min_delta),
        )

        if improved:
            best_val_loss = float(score_for_early_stop)
            best_epoch = epoch
            best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(config.early_stopping_patience):
                break

    model.load_state_dict(best_state_dict)
    model.eval()

    epochs_ran = len(history)
    stopped_early = epochs_ran < epochs

    return TrainResult(
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        epochs_ran=epochs_ran,
        stopped_early=stopped_early,
        history=history,
    )
