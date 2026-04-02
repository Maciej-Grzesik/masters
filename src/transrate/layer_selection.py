from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import COCODroneBirdCrops
from src.modelling.backbone_utils import (
    choose_best_model_from_csv,
    get_resnet_blocks as backbone_get_resnet_blocks,
    load_backbone_and_target_block,
)
from src.transrate.transrate import transrate

from src.utils import get_device, set_seed
import src.utils.const as CONST
from src.utils.vision import imagenet_eval_transform_224


def choose_best_from_csv(csv_path: Path) -> tuple[str, str, bool]:
    return choose_best_model_from_csv(csv_path)


def select_from_end(
    blocks: list[tuple[str, torch.nn.Module]], count: int
) -> list[tuple[str, str, torch.nn.Module]]:
    selected = list(reversed(blocks))[:count]
    out: list[tuple[str, str, torch.nn.Module]] = []
    for idx, (name, module) in enumerate(selected):
        tag = "L" if idx == 0 else f"L-{idx}"
        out.append((tag, name, module))
    return out


def extract_block_features_and_labels(
    model: torch.nn.Module,
    block: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    progress_desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    z_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    def _hook(_module, _input, output):
        pooled = F.adaptive_avg_pool2d(output, output_size=(1, 1)).flatten(1)
        z_list.append(pooled.detach().cpu().numpy())

    handle = block.register_forward_hook(_hook)

    with torch.no_grad():
        for x_batch, y_batch in tqdm(
            data_loader, desc=progress_desc, unit="batch", leave=False
        ):
            _ = model(x_batch.to(device))
            y_list.append(y_batch.cpu().numpy())

    handle.remove()

    z = np.concatenate(z_list, axis=0).astype(np.float64)
    y = np.concatenate(y_list, axis=0).astype(np.int64)
    return z, y


def evaluate_layers(
    model_name: str,
    weight_name: str,
    is_random: bool,
    dataset_root: Path,
    max_depth: int,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    set_seed(CONST.SEED)
    if is_random:
        set_seed(CONST.SEED)

    device = get_device()
    print(f"Device: {device}")

    model, _ = load_backbone_and_target_block(
        model_name=model_name,
        weight_name=weight_name,
        is_random=is_random,
        block_name="layer4.0",
        device=device,
    )

    dataset = COCODroneBirdCrops(
        dataset_root=dataset_root, transform=imagenet_eval_transform_224()
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    blocks = list(backbone_get_resnet_blocks(model).items())
    selected = select_from_end(blocks, count=max_depth + 1)

    records: list[dict] = []
    for tag, block_name, block_module in tqdm(
        selected, desc="Layer selection", unit="layer"
    ):
        z, y = extract_block_features_and_labels(
            model=model,
            block=block_module,
            data_loader=loader,
            device=device,
            progress_desc=f"{model_name} {tag} ({block_name})",
        )
        _, y_contiguous = np.unique(y, return_inverse=True)
        score = float(transrate(z, y_contiguous))
        records.append(
            {
                "layer_tag": tag,
                "block_name": block_name,
                "transrate": score,
                "feature_dim": int(z.shape[1]),
                "n_samples": int(z.shape[0]),
                "model": model_name,
                "weight": weight_name,
                "is_random": is_random,
            }
        )

    df = pd.DataFrame(records)

    def _order_key(tag: str) -> int:
        return 0 if tag == "L" else int(tag.split("-")[1])

    df["order"] = df["layer_tag"].map(_order_key)
    df = df.sort_values("order").reset_index(drop=True)
    return df


def make_plot(df: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    plot_df = df.copy()

    if "order" not in plot_df.columns:

        def _order_key(tag: str) -> int:
            return 0 if tag == "L" else int(str(tag).split("-")[1])

        plot_df["order"] = plot_df["layer_tag"].map(_order_key)

    plot_df = plot_df.sort_values("order").reset_index(drop=True)
    order = plot_df["layer_tag"].tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=plot_df,
        x="layer_tag",
        y="transrate",
        hue="block_name",
        style="block_name",
        s=120,
        legend="brief",
        ax=ax,
    )
    ax.set_title("Layer Selection (TransRate)")
    ax.set_xlabel("Layer from the end")
    ax.set_ylabel("TransRate")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.legend(title="Block", bbox_to_anchor=(1.02, 1.0), loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer selection for ResNet: evaluate L, L-1, ..., L-6 using TransRate."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=CONST.DEFAULT_DATASET_ROOT,
        help="Path to the dataset root (AOD_4).",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=CONST.DEFAULT_RESULTS_CSV,
        help="Model selection CSV used to auto-pick the best model if --model/--weight are not provided.",
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Model name (optional)."
    )
    parser.add_argument(
        "--weight", type=str, default=None, help="Weight enum name (optional)."
    )
    parser.add_argument(
        "--is-random",
        action="store_true",
        help="Use random initialization instead of pretrained weights.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum depth from the end. 6 means evaluate L..L-6.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CONST.DEFAULT_OUTPUT_DIR,
        help="Output directory for CSV and plot.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip feature extraction/evaluation and only build plot from an existing layer-selection CSV.",
    )
    parser.add_argument(
        "--layer-csv",
        type=Path,
        default=None,
        help="Path to an existing layer-selection CSV (required for --plot-only).",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Optional output path for plot image. If omitted, it is derived from output-dir/model/weight.",
    )
    args = parser.parse_args()

    if args.plot_only:
        plot_df = pd.read_csv(args.layer_csv)
        if args.plot_path is None:
            fig_path = args.layer_csv.with_suffix(".png")
        else:
            fig_path = args.plot_path
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        make_plot(plot_df, fig_path)
        print(f"Loaded layer CSV: {args.layer_csv}")
        print(f"Saved plot: {fig_path}")
        return

    if args.model is None or args.weight is None:
        model_name, weight_name, is_random = choose_best_from_csv(args.results_csv)
        print(
            f"Best from CSV -> model={model_name}, weight={weight_name}, is_random={is_random}"
        )
    else:
        model_name, weight_name, is_random = args.model, args.weight, args.is_random

    df = evaluate_layers(
        model_name=model_name,
        weight_name=weight_name,
        is_random=is_random,
        dataset_root=args.dataset_root,
        max_depth=args.max_depth,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name}__{weight_name}"
    csv_path = args.output_dir / f"layer_selection_{stem}.csv"
    fig_path = args.plot_path or (args.output_dir / f"layer_selection_{stem}.png")

    df.to_csv(csv_path, index=False)
    make_plot(df, fig_path)

    print(f"Saved results: {csv_path}")
    print(f"Saved plot: {fig_path}")


if __name__ == "__main__":
    main()
