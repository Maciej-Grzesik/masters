from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.dataset import COCODroneBirdCrops
from src.modelling.backbone_utils import (
    choose_best_block_from_csv,
    choose_best_model_from_csv,
    load_backbone_and_target_block,
)
from src.modelling.ood_workflow import (
    extract_block_features_for_indices,
    infer_labels,
    train_model_for_fold,
)
from src.utils.train import TrainConfig
import src.utils.const as CONST
from src.utils.get_device import get_device
from src.utils.vision import imagenet_eval_transform_224


def _sample_indices(y: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if y.shape[0] <= max_samples:
        return np.arange(y.shape[0])

    rng = np.random.default_rng(seed)
    idx = rng.choice(y.shape[0], size=max_samples, replace=False)
    idx.sort()
    return idx


def _norm(name: str) -> str:
    return str(name).strip().lower()


def _resolve_backbone_choice(args: argparse.Namespace) -> tuple[str, str, bool, str]:
    if args.model is None or args.weight is None:
        model_name, weight_name, is_random = choose_best_model_from_csv(
            args.model_selection_csv
        )
    else:
        model_name = str(args.model)
        weight_name = str(args.weight)
        is_random = bool(args.is_random)

    block_name = args.block_name
    if block_name is None:
        layer_csv = args.layer_selection_csv
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
                "Run layer selection first or pass --layer-selection-csv/--block-name."
            )
        block_name = choose_best_block_from_csv(layer_csv)

    return model_name, weight_name, is_random, str(block_name)


def _plot_embedding(df: pd.DataFrame, x_col: str, y_col: str, title: str, path: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue="label",
        style="label",
        alpha=0.75,
        s=45,
        ax=ax,
    )
    ax.set_title(title)
    ax.legend(title="Class", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize dataset embeddings with UMAP and t-SNE."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=CONST.DEFAULT_DATASET_ROOT,
        help="Path to AOD_4 dataset root.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=CONST.SEED)
    parser.add_argument("--train-epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--train-val-ratio", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=10_000)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument(
        "--model-selection-csv",
        type=Path,
        default=CONST.PROJECT_ROOT / "outputs" / "transrate" / "resnet_transrate_results.csv",
    )
    parser.add_argument("--layer-selection-csv", type=Path, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--is-random", action="store_true")
    parser.add_argument("--block-name", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CONST.PROJECT_ROOT / "outputs" / "embeddings",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional path for saving trained backbone checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    UMAP = importlib.import_module("umap").UMAP

    model_name, weight_name, is_random, block_name = _resolve_backbone_choice(args)

    id_labels, ood_label = infer_labels(args.dataset_root)
    labels = (*id_labels, ood_label)

    dataset = COCODroneBirdCrops(
        dataset_root=args.dataset_root,
        include_labels=labels,
        transform=imagenet_eval_transform_224(),
    )

    label_names = np.array([sample.label_name for sample in dataset.samples], dtype=object)
    all_indices = np.arange(len(dataset), dtype=np.int64)
    id_indices = all_indices[np.array([_norm(lbl) != _norm(ood_label) for lbl in label_names])]

    device = get_device()
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
        train_indices=id_indices,
        label_names=label_names,
        id_class_names=id_labels,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        fold_id=0,
        train_config=TrainConfig(
            epochs=args.train_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
        ),
        train_val_ratio=args.train_val_ratio,
    )

    checkpoint_path = args.checkpoint_path or (
        args.output_dir
        / f"trained_backbone_{model_name}__{weight_name}__{block_name.replace('.', '_')}.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "weight_name": weight_name,
            "is_random": bool(is_random),
            "block_name": block_name,
            "id_labels": list(id_labels),
            "ood_label": ood_label,
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    x = extract_block_features_for_indices(
        model=model,
        block=block,
        dataset=dataset,
        indices=all_indices,
        batch_size=args.batch_size,
        device=device,
    )

    idx = _sample_indices(label_names, max_samples=args.max_samples, seed=args.seed)
    x = x[idx]
    label_names = label_names[idx]
    ood_flags = np.array([_norm(lbl) == _norm(ood_label) for lbl in label_names], dtype=np.int64)

    x_std = StandardScaler().fit_transform(x)

    umap_2d = UMAP(n_components=2, random_state=args.seed)
    umap_emb = umap_2d.fit_transform(x_std)

    tsne_2d = TSNE(
        n_components=2,
        random_state=args.seed,
        perplexity=args.tsne_perplexity,
        init="pca",
        learning_rate="auto",
    )
    tsne_emb = tsne_2d.fit_transform(x_std)

    out_df = pd.DataFrame(
        {
            "label": label_names,
            "is_ood": ood_flags,
            "umap_1": umap_emb[:, 0],
            "umap_2": umap_emb[:, 1],
            "tsne_1": tsne_emb[:, 0],
            "tsne_2": tsne_emb[:, 1],
        }
    )

    csv_path = args.output_dir / "dataset_umap_tsne_embeddings.csv"
    out_df.to_csv(csv_path, index=False)

    metadata = {
        "model_name": model_name,
        "weight_name": weight_name,
        "is_random": bool(is_random),
        "block_name": block_name,
        "id_labels": list(id_labels),
        "ood_label": ood_label,
        "checkpoint_path": str(checkpoint_path),
        "n_samples_total": int(len(dataset)),
        "n_samples_used_for_plot": int(len(out_df)),
        "training": train_info,
    }
    with (args.output_dir / "embedding_run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    _plot_embedding(
        out_df,
        x_col="umap_1",
        y_col="umap_2",
        title="UMAP projection of AOD_4 dataset",
        path=args.output_dir / "umap_projection.png",
    )
    _plot_embedding(
        out_df,
        x_col="tsne_1",
        y_col="tsne_2",
        title="t-SNE projection of AOD_4 dataset",
        path=args.output_dir / "tsne_projection.png",
    )

    print(f"Saved embeddings CSV: {csv_path}")
    print(f"Saved trained model: {checkpoint_path}")
    print(f"Saved metadata: {args.output_dir / 'embedding_run_metadata.json'}")
    print(f"Saved plots in: {args.output_dir}")


if __name__ == "__main__":
    main()
