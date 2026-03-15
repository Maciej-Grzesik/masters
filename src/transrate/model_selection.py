from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

from src.dataset import COCODroneBirdCrops
from src.transrate.transrate import transrate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


SEED = 1410
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset" / "AOD_4"
OUTPUT_DIR = PROJECT_ROOT

RESNET_MODEL_NAMES = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_transform() -> Compose:
    return Compose(
        [
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def extract_features_and_labels(
    feature_extractor: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    progress_desc: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    feature_extractor.eval()
    z_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    with torch.no_grad():
        for x_batch, y_batch in tqdm(
            data_loader,
            desc=progress_desc or "batches",
            unit="batch",
            leave=False,
        ):
            x_batch = x_batch.to(device)
            features = feature_extractor(x_batch)
            z_list.append(features.flatten(1).cpu().numpy())
            y_list.append(y_batch.cpu().numpy())

    z = np.concatenate(z_list, axis=0).astype(np.float64)
    y = np.concatenate(y_list, axis=0).astype(np.int64)
    return z, y


def get_weight_entries(model_name: str):
    weight_enum = models.get_model_weights(model_name)
    weight_entries = [(w.name, w) for w in weight_enum]
    weight_entries.append((f"random_seed_{SEED}", None))
    return weight_entries


def evaluate_single_setup(
    model_name: str,
    weight_name: str,
    weight_obj,
    data_loader: DataLoader,
    device: torch.device,
) -> dict:
    if weight_obj is None:
        set_seed(SEED)

    model = models.get_model(model_name, weights=weight_obj)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(device)

    z, y = extract_features_and_labels(
        feature_extractor,
        data_loader,
        device,
        progress_desc=f"{model_name}:{weight_name}",
    )

    _, y_contiguous = np.unique(y, return_inverse=True)
    score = float(transrate(z, y_contiguous))

    return {
        "model": model_name,
        "weight": weight_name,
        "is_random": weight_obj is None,
        "transrate": score,
        "n_samples": int(z.shape[0]),
        "feature_dim": int(z.shape[1]),
    }


def make_scatter_plot(results_df: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    df = results_df.copy()

    model_order = RESNET_MODEL_NAMES.copy()
    weight_order = sorted(df["weight"].astype(str).unique().tolist())

    markers_cycle = ["o", "s", "^", "D", "P", "X", "*", "v", "<", ">", "h", "8"]
    marker_map = {
        weight: markers_cycle[i % len(markers_cycle)]
        for i, weight in enumerate(weight_order)
    }

    palette = sns.color_palette("tab20", n_colors=max(3, len(weight_order)))
    color_map = {weight: palette[i % len(palette)] for i, weight in enumerate(weight_order)}

    offsets = np.linspace(-0.3, 0.3, num=max(1, len(weight_order)))
    offset_map = {weight: offsets[i] for i, weight in enumerate(weight_order)}
    x_map = {name: idx for idx, name in enumerate(model_order)}

    rng = np.random.default_rng(SEED)
    df["x"] = df["model"].map(x_map).astype(float)
    df["x"] = df["x"] + df["weight"].map(offset_map).astype(float)
    df["x"] = df["x"] + rng.normal(0.0, 0.015, size=len(df))

    fig, ax = plt.subplots(figsize=(18, 8))
    for weight_name in weight_order:
        part = df[df["weight"] == weight_name]
        ax.scatter(
            part["x"],
            part["transrate"],
            s=85,
            marker=marker_map[weight_name],
            color=color_map[weight_name],
            edgecolors="black",
            linewidths=0.5,
            alpha=0.9,
            label=weight_name,
        )

    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order, rotation=30)
    ax.set_title("TransRate")
    ax.set_xlabel("Model")
    ax.set_ylabel("TransRate")
    ax.legend(title="Weight", bbox_to_anchor=(1.02, 1.0), loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close()


def run_model_selection(results_csv_path: Path, figure_path: Path) -> None:
    set_seed(SEED)
    results_csv_path.parent.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    dataset = COCODroneBirdCrops(
        dataset_root=DEFAULT_DATASET_ROOT,
        transform=default_transform(),
    )

    loader = DataLoader(dataset=dataset, batch_size=64, num_workers=0, shuffle=False)

    results: list[dict] = []
    total_runs = sum(len(get_weight_entries(name)) for name in RESNET_MODEL_NAMES)
    progress = tqdm(total=total_runs, desc="Model selection", unit="setup")

    for model_name in RESNET_MODEL_NAMES:
        for weight_name, weight_obj in tqdm(
            get_weight_entries(model_name),
            desc=f"{model_name} weights",
            unit="weight",
            leave=False,
        ):
            record = evaluate_single_setup(
                model_name=model_name,
                weight_name=weight_name,
                weight_obj=weight_obj,
                data_loader=loader,
                device=device,
            )
            results.append(record)
            progress.set_postfix(model=model_name, weight=weight_name)
            progress.update(1)

    progress.close()

    results_df = pd.DataFrame(results).sort_values(["model", "weight"])
    results_df.to_csv(results_csv_path, index=False)
    make_scatter_plot(results_df, figure_path)

    print(f"Saved results: {results_csv_path}")
    print(f"Saved plot: {figure_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ResNet TransRate model selection and/or build a scatter plot from CSV."
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip model evaluation and build the plot from an existing CSV file.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=OUTPUT_DIR / "resnet_transrate_results.csv",
        help="Path to model selection CSV (input for --plot-only, output otherwise).",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=OUTPUT_DIR / "resnet_transrate_scatter.png",
        help="Output path for the scatter plot.",
    )
    args = parser.parse_args()

    if args.plot_only:
        df = pd.read_csv(args.results_csv)
        make_scatter_plot(df, args.plot_path)
        print(f"Loaded CSV: {args.results_csv}")
        print(f"Saved plot: {args.plot_path}")
        return

    run_model_selection(results_csv_path=args.results_csv, figure_path=args.plot_path)


if __name__ == "__main__":
    main()
