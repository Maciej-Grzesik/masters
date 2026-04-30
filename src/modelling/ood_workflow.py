from __future__ import annotations

import argparse
import json
from itertools import combinations
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from src.dataset import COCODroneBirdCrops
from src.modelling.backbone_utils import (
    choose_best_block_from_csv,
    choose_best_model_from_csv,
    load_backbone_and_target_block,
)
from src.modelling.gmm import GMMOOD
from src.modelling.mahobian import MahalanobisOOD
from src.modelling.ood_plots import (
    plot_fold_metrics_summary,
    plot_ood_curves,
    plot_score_distribution,
)
from src.modelling.statistical_protocol import run_paired_tests
from src.utils.train import TrainConfig, train
from src.utils.get_device import get_device
from src.utils.set_seed import set_seed
from src.utils.vision import imagenet_eval_transform_224
import src.utils.const as CONST


@dataclass(slots=True)
class ExperimentConfig:
    dataset_root: Path
    id_labels: tuple[str, ...]
    ood_label: str
    k_folds: int
    seed: int
    batch_size: int
    output_dir: Path
    methods: tuple[str, ...]
    model_selection_csv: Path
    layer_selection_csv: Path | None
    train_epochs: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int
    early_stopping_min_delta: float
    train_val_ratio: float
    plots_dir: Path


DEFAULT_METHODS = (
    "mahalanobis",
    "gmm",
)


OOD_CANDIDATES = {"bird"}
ID_CANDIDATES = {
    "helicopter",
    "drone",
    "airplane",
}


class RemappedSubset(Dataset):
    def __init__(
        self,
        base_dataset: COCODroneBirdCrops,
        indices: np.ndarray,
        original_label_names: np.ndarray,
        label_to_id: dict[str, int],
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.original_label_names = original_label_names
        self.label_to_id = label_to_id

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int):
        ds_idx = int(self.indices[idx])
        x, _ = self.base_dataset[ds_idx]
        name = str(self.original_label_names[ds_idx])
        y = int(self.label_to_id[name])
        return x, y


def _norm(name: str) -> str:
    return str(name).strip().lower()


def _split_train_val_indices(
    train_indices: np.ndarray,
    label_names: np.ndarray,
    label_to_id: dict[str, int],
    seed: int,
    fold_id: int,
    val_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    if train_indices.shape[0] < 4:
        return train_indices, np.array([], dtype=np.int64)

    mapped = np.array(
        [label_to_id[str(label_names[idx])] for idx in train_indices],
        dtype=np.int64,
    )

    split_ratio = float(np.clip(val_ratio, 0.05, 0.4))
    n_classes = len(np.unique(mapped))
    min_required = max(2 * n_classes, 4)
    if train_indices.shape[0] < min_required:
        return train_indices, np.array([], dtype=np.int64)

    try:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=split_ratio,
            random_state=seed + fold_id,
        )
        train_rel, val_rel = next(splitter.split(train_indices, mapped))
        return train_indices[train_rel], train_indices[val_rel]
    except ValueError:
        return train_indices, np.array([], dtype=np.int64)


def load_features_and_labels(
    dataset_root: Path,
    model_name: str,
    weight_name: str,
    is_random: bool,
    block_name: str,
    include_labels: tuple[str, ...] | None = None,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = get_device()

    dataset = COCODroneBirdCrops(
        dataset_root=dataset_root,
        include_labels=include_labels,
        transform=imagenet_eval_transform_224(),
    )

    model, block = load_backbone_and_target_block(
        model_name=model_name,
        weight_name=weight_name,
        is_random=is_random,
        block_name=block_name,
        device=device,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    def _hook(_module, _input, output):
        pooled = F.adaptive_avg_pool2d(output, output_size=(1, 1)).flatten(1)
        x_list.append(pooled.detach().cpu().numpy())

    handle = block.register_forward_hook(_hook)
    with torch.no_grad():
        for x_batch, y_batch in tqdm(loader, desc="Feature extraction", unit="batch"):
            _ = model(x_batch.to(device))
            y_list.append(y_batch.cpu().numpy())
    handle.remove()

    x = np.concatenate(x_list, axis=0).astype(np.float64)
    y = np.concatenate(y_list, axis=0).astype(np.int64)

    label_names = np.array(
        [dataset.index_to_label[int(idx)] for idx in y],
        dtype=object,
    )
    return x, y, label_names


def train_model_for_fold(
    model: torch.nn.Module,
    dataset: COCODroneBirdCrops,
    train_indices: np.ndarray,
    label_names: np.ndarray,
    id_class_names: tuple[str, ...],
    batch_size: int,
    device: torch.device,
    seed: int,
    fold_id: int,
    train_config: TrainConfig,
    train_val_ratio: float = 0.1,
) -> dict:
    set_seed(seed + fold_id)

    label_to_id = {name: idx for idx, name in enumerate(id_class_names)}
    train_main_indices, val_indices = _split_train_val_indices(
        train_indices=train_indices,
        label_names=label_names,
        label_to_id=label_to_id,
        seed=seed,
        fold_id=fold_id,
        val_ratio=train_val_ratio,
    )

    train_subset = RemappedSubset(
        base_dataset=dataset,
        indices=train_main_indices,
        original_label_names=label_names,
        label_to_id=label_to_id,
    )
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = None
    if val_indices.shape[0] > 0:
        val_subset = RemappedSubset(
            base_dataset=dataset,
            indices=val_indices,
            original_label_names=label_names,
            label_to_id=label_to_id,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    in_features = int(model.fc.in_features)
    model.fc = nn.Linear(in_features, len(id_class_names)).to(device)
    train_result = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=TrainConfig(
            epochs=train_config.epochs,
            learning_rate=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
            early_stopping_patience=train_config.early_stopping_patience,
            early_stopping_min_delta=train_config.early_stopping_min_delta,
        ),
    )
    return {
        **train_result.to_dict(),
        "fold": int(fold_id),
        "train_size": int(train_main_indices.shape[0]),
        "val_size": int(val_indices.shape[0]),
        "epochs": int(train_config.epochs),
        "learning_rate": float(train_config.learning_rate),
        "weight_decay": float(train_config.weight_decay),
        "early_stopping_patience": int(train_config.early_stopping_patience),
        "early_stopping_min_delta": float(train_config.early_stopping_min_delta),
    }


def extract_block_features_for_indices(
    model: torch.nn.Module,
    block: torch.nn.Module,
    dataset: COCODroneBirdCrops,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    subset = Subset(dataset, indices.tolist())
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    z_list: list[np.ndarray] = []

    def _hook(_module, _input, output):
        pooled = F.adaptive_avg_pool2d(output, output_size=(1, 1)).flatten(1)
        z_list.append(pooled.detach().cpu().numpy())

    handle = block.register_forward_hook(_hook)
    with torch.no_grad():
        for x_batch, _ in tqdm(
            loader,
            desc="Block feature extraction",
            unit="batch",
            leave=False,
        ):
            _ = model(x_batch.to(device))
    handle.remove()

    return np.concatenate(z_list, axis=0).astype(np.float64)


def infer_labels(dataset_root: Path) -> tuple[tuple[str, ...], str]:
    dataset = COCODroneBirdCrops(
        dataset_root=dataset_root,
        transform=imagenet_eval_transform_224(),
    )
    available = tuple(dataset.label_to_index.keys())

    ood_match = [lbl for lbl in available if _norm(lbl) in OOD_CANDIDATES]
    if not ood_match:
        raise ValueError(
            f"Nie znaleziono etykiety OOD (bird/ptak) w dataset. Dostępne: {available}"
        )

    ood_label = ood_match[0]
    id_labels = tuple(lbl for lbl in available if lbl != ood_label and _norm(lbl) in ID_CANDIDATES)

    if len(id_labels) < 3:
        id_labels = tuple(lbl for lbl in available if lbl != ood_label)

    return id_labels, ood_label


def build_methods(method_names: tuple[str, ...], seed: int, fold_id: int) -> dict[str, object]:
    supported = {
        "mahalanobis": lambda: MahalanobisOOD(regularization=1e-5),
        "gmm": lambda: GMMOOD(n_components=3, covariance_type="full", random_state=seed + fold_id),
    }

    unknown = [m for m in method_names if m not in supported]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Supported: {tuple(supported.keys())}")

    return {name: supported[name]() for name in method_names}


def evaluate_fold(
    train_x: np.ndarray,
    eval_x: np.ndarray,
    eval_y: np.ndarray,
    fold_id: int,
    seed: int,
    method_names: tuple[str, ...],
    plots_dir: Path,
) -> list[dict]:
    methods = build_methods(method_names=method_names, seed=seed, fold_id=fold_id)

    rows: list[dict] = []
    for method_name, model in tqdm(
        methods.items(),
        desc=f"Fold {fold_id}: methods",
        unit="method",
        leave=False,
    ):
        model.fit(train_x)
        scores = model.score_samples(eval_x)
        preds = model.predict(eval_x)

        auroc = roc_auc_score(eval_y, scores)
        loss = 1.0 - float(auroc)

        method_plot_dir = plots_dir / method_name
        plot_ood_curves(
            y_true=eval_y.astype(np.int64),
            ood_scores=scores.astype(np.float64),
            method_name=method_name,
            fold_id=fold_id,
            out_dir=method_plot_dir,
        )
        plot_score_distribution(
            y_true=eval_y.astype(np.int64),
            ood_scores=scores.astype(np.float64),
            method_name=method_name,
            fold_id=fold_id,
            out_dir=method_plot_dir,
        )

        rows.append(
            {
                "fold": fold_id,
                "method": method_name,
                "auroc": float(auroc),
                "loss": float(loss),
                "accuracy": float(accuracy_score(eval_y, preds)),
                "f1": float(f1_score(eval_y, preds, zero_division=0)),
                "n_eval": int(eval_x.shape[0]),
                "n_ood": int(np.sum(eval_y == 1)),
                "n_id": int(np.sum(eval_y == 0)),
            }
        )

    return rows


def build_evaluation_set(
    id_x_test: np.ndarray,
    ood_x: np.ndarray,
    fold_id: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 1000 + fold_id)
    n_id = id_x_test.shape[0]

    if ood_x.shape[0] <= n_id:
        chosen_ood = ood_x
    else:
        idx = rng.choice(ood_x.shape[0], size=n_id, replace=False)
        chosen_ood = ood_x[idx]

    eval_x = np.concatenate([id_x_test, chosen_ood], axis=0)
    eval_y = np.concatenate(
        [np.zeros(id_x_test.shape[0], dtype=np.int64), np.ones(chosen_ood.shape[0], dtype=np.int64)],
        axis=0,
    )
    return eval_x, eval_y


def run_experiment(config: ExperimentConfig) -> dict:
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    model_name, weight_name, is_random = choose_best_model_from_csv(
        config.model_selection_csv
    )

    layer_csv = config.layer_selection_csv
    if layer_csv is None:
        layer_csv = (
            CONST.PROJECT_ROOT
            / "outputs"
            / "transrate"
            / f"layer_selection_{model_name}__{weight_name}.csv"
        )
    if not layer_csv.exists():
        raise FileNotFoundError(
            f"Layer-selection CSV not found: {layer_csv}. "
            "Run layer selection first or pass --layer-selection-csv."
        )
    block_name = choose_best_block_from_csv(layer_csv)

    selected_labels = tuple(dict.fromkeys([*config.id_labels, config.ood_label]))
    dataset = COCODroneBirdCrops(
        dataset_root=config.dataset_root,
        include_labels=selected_labels,
        transform=imagenet_eval_transform_224(),
    )

    label_names = np.array([s.label_name for s in dataset.samples], dtype=object)
    mask_ood = np.array([_norm(lbl) == _norm(config.ood_label) for lbl in label_names])
    mask_id = ~mask_ood

    all_indices = np.arange(len(dataset), dtype=np.int64)
    id_indices = all_indices[mask_id]
    ood_indices = all_indices[mask_ood]

    y_id_names = label_names[id_indices]
    unique_id, y_id = np.unique(y_id_names, return_inverse=True)

    device = get_device()

    skf = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=config.seed)

    all_rows: list[dict] = []
    training_runs: list[dict] = []
    split_iter = list(skf.split(id_indices, y_id))
    for fold_idx, (train_rel, test_rel) in enumerate(
        tqdm(split_iter, desc="OOD folds", unit="fold"),
        start=1,
    ):
        id_train_indices = id_indices[train_rel]
        id_test_indices = id_indices[test_rel]

        model, block = load_backbone_and_target_block(
            model_name=model_name,
            weight_name=weight_name,
            is_random=is_random,
            block_name=block_name,
            device=device,
        )

        train_info = train_model_for_fold(
            model=model,
            dataset=dataset,
            train_indices=id_train_indices,
            label_names=label_names,
            id_class_names=tuple(str(v) for v in unique_id.tolist()),
            batch_size=config.batch_size,
            device=device,
            seed=config.seed,
            fold_id=fold_idx,
            train_config=TrainConfig(
                epochs=config.train_epochs,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_min_delta=config.early_stopping_min_delta,
            ),
            train_val_ratio=config.train_val_ratio,
        )
        training_runs.append(train_info)

        id_train_x = extract_block_features_for_indices(
            model=model,
            block=block,
            dataset=dataset,
            indices=id_train_indices,
            batch_size=config.batch_size,
            device=device,
        )
        id_test_x = extract_block_features_for_indices(
            model=model,
            block=block,
            dataset=dataset,
            indices=id_test_indices,
            batch_size=config.batch_size,
            device=device,
        )
        ood_x = extract_block_features_for_indices(
            model=model,
            block=block,
            dataset=dataset,
            indices=ood_indices,
            batch_size=config.batch_size,
            device=device,
        )

        eval_x, eval_y = build_evaluation_set(id_test_x, ood_x, fold_idx, config.seed)
        all_rows.extend(
            evaluate_fold(
                id_train_x,
                eval_x,
                eval_y,
                fold_idx,
                config.seed,
                method_names=config.methods,
                plots_dir=config.plots_dir / "per_method",
            )
        )

    df = pd.DataFrame(all_rows).sort_values(["method", "fold"]).reset_index(drop=True)
    df.to_csv(config.output_dir / "fold_metrics.csv", index=False)

    plot_fold_metrics_summary(
        results_df=df,
        out_dir=config.plots_dir / "summary",
        metrics=("auroc", "f1", "accuracy", "loss"),
    )

    losses = {
        method: df[df["method"] == method].sort_values("fold")["loss"].to_numpy(dtype=np.float64)
        for method in df["method"].unique()
    }

    statistical_tests: list[dict] = []
    pairs = list(combinations(df["method"].unique().tolist(), 2))
    for method_a, method_b in tqdm(
        pairs,
        desc="Statistical comparisons",
        unit="pair",
        leave=False,
    ):
        stat = run_paired_tests(
            method_a=method_a,
            method_b=method_b,
            metric_name="loss=1-auroc",
            losses_a=losses[method_a],
            losses_b=losses[method_b],
        )
        statistical_tests.append(stat.to_dict())

    summary = {
        "config": {
            **asdict(config),
            "dataset_root": str(config.dataset_root),
            "output_dir": str(config.output_dir),
            "plots_dir": str(config.plots_dir),
            "model_selection_csv": str(config.model_selection_csv),
            "layer_selection_csv": (
                str(config.layer_selection_csv) if config.layer_selection_csv is not None else None
            ),
        },
        "selected_backbone": {
            "model": model_name,
            "weight": weight_name,
            "is_random": bool(is_random),
            "block_name": block_name,
        },
        "id_classes": [str(v) for v in unique_id.tolist()],
        "ood_class": config.ood_label,
        "n_samples": {
            "id": int(id_indices.shape[0]),
            "ood": int(ood_indices.shape[0]),
            "total": int(len(dataset)),
        },
        "means": (
            df.groupby("method")[["auroc", "loss", "accuracy", "f1"]]
            .mean()
            .round(6)
            .to_dict(orient="index")
        ),
        "training_runs": training_runs,
        "statistical_tests": statistical_tests,
    }

    with (config.output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OOD workflow: best model/layer from TransRate + k-fold statistical testing."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=CONST.DEFAULT_DATASET_ROOT,
        help="Path to AOD_4 dataset root.",
    )
    parser.add_argument(
        "--id-labels",
        type=str,
        nargs="*",
        default=None,
        help="ID labels, e.g. helicopter drone airplane. If omitted, inferred automatically.",
    )
    parser.add_argument(
        "--ood-label",
        type=str,
        default=None,
        help="OOD label, e.g. bird. If omitted, inferred automatically.",
    )
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--train-val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CONST.PROJECT_ROOT / "outputs" / "ood_training",
    )
    parser.add_argument(
        "--model-selection-csv",
        type=Path,
        default=CONST.PROJECT_ROOT / "outputs" / "transrate" / "resnet_transrate_results.csv",
        help="CSV with model/weight TransRate ranking.",
    )
    parser.add_argument(
        "--layer-selection-csv",
        type=Path,
        default=None,
        help="Optional layer-selection CSV. If missing, inferred from selected model/weight.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Directory for OOD evaluation plots. Defaults to <output-dir>/plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.id_labels is None or args.ood_label is None:
        inferred_id, inferred_ood = infer_labels(args.dataset_root)
    else:
        inferred_id, inferred_ood = tuple(args.id_labels), args.ood_label

    config = ExperimentConfig(
        dataset_root=args.dataset_root,
        id_labels=inferred_id,
        ood_label=inferred_ood,
        k_folds=args.k_folds,
        seed=CONST.SEED,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        methods=tuple(DEFAULT_METHODS),
        model_selection_csv=args.model_selection_csv,
        layer_selection_csv=args.layer_selection_csv,
        train_epochs=args.train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        train_val_ratio=args.train_val_ratio,
        plots_dir=args.plots_dir or (args.output_dir / "plots"),
    )

    summary = run_experiment(config)

    print("=== OOD experiment finished ===")
    print(f"Backbone: {summary['selected_backbone']}")
    print(f"ID classes: {summary['id_classes']}")
    print(f"OOD class: {summary['ood_class']}")
    print(f"Saved summary: {config.output_dir / 'run_summary.json'}")
    print(f"Saved metrics: {config.output_dir / 'fold_metrics.csv'}")


if __name__ == "__main__":
    main()
