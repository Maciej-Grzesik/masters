from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.dataset import COCODroneBirdCrops, CropSample

SEED = 1410


class FeatureExtractionDataset(Dataset):
    def __init__(self, base_ds: COCODroneBirdCrops, transform) -> None:
        self.base_ds = base_ds
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_ds.samples)

    def __getitem__(self, idx: int):
        sample: CropSample = self.base_ds.samples[idx]
        with Image.open(sample.image_path) as img:
            rgb = img.convert("RGB")
            crop = rgb.crop(sample.bbox)
        x = self.transform(crop)
        return x, sample.label, idx


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


def fallback_transform() -> Compose:
    return Compose(
        [
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def parse_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def choose_best_from_csv(csv_path: Path) -> tuple[str, str, bool]:
    df = pd.read_csv(csv_path)

    best = df.sort_values("transrate", ascending=False).iloc[0]
    return str(best["model"]), str(best["weight"]), parse_bool(best["is_random"])


def choose_best_model_from_layer_csv(layer_csv_path: Path) -> tuple[str, str, bool]:
    df = pd.read_csv(layer_csv_path)

    best = df.sort_values("transrate", ascending=False).iloc[0]
    return str(best["model"]), str(best["weight"]), parse_bool(best["is_random"])


def resolve_model_choice(
    model: str | None,
    weight: str | None,
    is_random: bool,
    results_csv: Path,
    layer_results_csv: Path,
) -> tuple[str, str, bool]:
    if model is not None and weight is not None:
        return model, weight, is_random

    try:
        model_name, weight_name, random_flag = choose_best_from_csv(results_csv)
        print(
            f"Best from model-selection CSV -> model={model_name}, weight={weight_name}, is_random={random_flag}"
        )
        return model_name, weight_name, random_flag
    except (FileNotFoundError, IsADirectoryError, ValueError) as model_csv_error:
        try:
            model_name, weight_name, random_flag = choose_best_model_from_layer_csv(
                layer_results_csv
            )
            print(
                f"Best from layer-selection CSV -> model={model_name}, weight={weight_name}, is_random={random_flag}"
            )
            return model_name, weight_name, random_flag
        except (FileNotFoundError, IsADirectoryError, ValueError) as layer_csv_error:
            raise ValueError(
                "Could not auto-select model/weights. Provide --model and --weight, "
                "or pass valid --results-csv / --layer-results-csv files. "
                f"model_csv_error={model_csv_error}; layer_csv_error={layer_csv_error}"
            )


def get_resnet_blocks(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    blocks: list[tuple[str, torch.nn.Module]] = []
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, layer_name, None)
        if layer is None:
            continue
        for i, block in enumerate(layer):
            blocks.append((f"{layer_name}.{i}", block))

    return blocks


def build_layer_tag_to_block_name(model: torch.nn.Module) -> dict[str, str]:
    blocks = get_resnet_blocks(model)
    selected = list(reversed(blocks))
    out: dict[str, str] = {}
    for idx, (block_name, _block_module) in enumerate(selected):
        tag = "L" if idx == 0 else f"L-{idx}"
        out[tag] = block_name
    return out


def choose_best_layer_from_csv(
    layer_csv_path: Path,
    model_name: str,
    weight_name: str,
    is_random: bool,
) -> tuple[str, str]:
    df = pd.read_csv(layer_csv_path)

    if {"model", "weight", "is_random"}.issubset(df.columns):
        df = df[
            (df["model"].astype(str) == model_name)
            & (df["weight"].astype(str) == weight_name)
            & (df["is_random"].map(parse_bool) == bool(is_random))
        ]

    best = df.sort_values("transrate", ascending=False).iloc[0]
    return str(best["layer_tag"]), str(best["block_name"])


def resolve_weight(model_name: str, weight_name: str, is_random: bool):
    if is_random:
        return None

    enum_cls = models.get_model_weights(model_name)

    return getattr(enum_cls, weight_name)


def model_slug(model_name: str, weight_name: str, is_random: bool) -> str:
    if is_random:
        return f"{model_name}__random_seed_{SEED}"
    return f"{model_name}__{weight_name}"


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def extract_features_with_hook(
    model: torch.nn.Module,
    block_module: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    z_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    idx_list: list[np.ndarray] = []

    def _hook(_module, _inp, out):
        pooled = F.adaptive_avg_pool2d(out, output_size=(1, 1)).flatten(1)
        z_list.append(pooled.detach().cpu().numpy().astype(np.float32))

    handle = block_module.register_forward_hook(_hook)

    with torch.no_grad():
        for x_batch, y_batch, idx_batch in tqdm(loader, desc="Extracting", unit="batch"):
            _ = model(x_batch.to(device, non_blocking=True))
            y_list.append(y_batch.cpu().numpy())
            idx_list.append(idx_batch.cpu().numpy())

    handle.remove()

    z = np.concatenate(z_list, axis=0)
    y = np.concatenate(y_list, axis=0).astype(np.int64)
    idx = np.concatenate(idx_list, axis=0).astype(np.int64)
    return z, y, idx


def extract_and_save(
    dataset_root: Path,
    model_name: str,
    weight_name: str,
    is_random: bool,
    layer_tag: str,
    block_name: str,
    batch_size: int,
    num_workers: int,
) -> Path:
    device = get_device()
    print(f"Device: {device}")

    if is_random:
        set_seed(SEED)

    weight_obj = resolve_weight(model_name, weight_name, is_random)
    model = models.get_model(model_name, weights=weight_obj).to(device)

    block_lookup = dict(get_resnet_blocks(model))

    target_block = block_lookup[block_name]

    transform = (
        weight_obj.transforms() if weight_obj is not None else fallback_transform()
    )

    base_ds = COCODroneBirdCrops(dataset_root=dataset_root, transform=None)
    ex_ds = FeatureExtractionDataset(base_ds, transform)
    loader = DataLoader(
        ex_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    target_root = (
        dataset_root
        / "DeepFeatures"
        / model_slug(model_name, weight_name, is_random)
        / f"layer_{sanitize_name(layer_tag)}__{sanitize_name(block_name)}"
    )
    target_root.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    feat, labels, indices = extract_features_with_hook(
        model=model,
        block_module=target_block,
        loader=loader,
        device=device,
    )

    for i, sample_idx in enumerate(indices.tolist()):
        sample = base_ds.samples[sample_idx]
        split_dir = target_root / sample.source_split / sample.label_name
        split_dir.mkdir(parents=True, exist_ok=True)

        file_name = f"{sample.image_path.stem}__ann_{sample.annotation_id}.npy"
        out_path = split_dir / file_name
        np.save(out_path, feat[i])

        records.append(
            {
                "split": sample.source_split,
                "label": int(labels[i]),
                "label_name": sample.label_name,
                "annotation_id": int(sample.annotation_id),
                "image_path": str(sample.image_path),
                "feature_path": str(out_path),
                "feature_dim": int(feat.shape[1]),
                "model": model_name,
                "weight": weight_name,
                "is_random": is_random,
                "layer_tag": layer_tag,
                "block_name": block_name,
            }
        )

    index_path = target_root / "features_index.csv"
    pd.DataFrame(records).to_csv(index_path, index=False)
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract deep features using the best ResNet model and save per split/class."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "AOD_4",
        help="Path to the AOD_4 dataset root",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=PROJECT_ROOT / "resnet_transrate_results.csv",
        help="CSV produced by the model selection step",
    )
    parser.add_argument(
        "--layer-results-csv",
        type=Path,
        default=PROJECT_ROOT / "layer_selection_results.csv",
        help="Layer selection CSV used to auto-pick the best layer",
    )
    parser.add_argument("--model", type=str, default=None, help="Model name (optional)")
    parser.add_argument(
        "--weight", type=str, default=None, help="Weight enum name (optional)"
    )
    parser.add_argument(
        "--is-random",
        action="store_true",
        help="Use random initialization instead of pretrained weights",
    )
    parser.add_argument(
        "--layer-tag",
        type=str,
        default=None,
        help="Layer tag to extract (e.g. L, L-1). If omitted, use best layer from --layer-results-csv.",
    )
    parser.add_argument(
        "--block-name",
        type=str,
        default=None,
        help="Explicit block name override (e.g. layer3.1). If omitted, resolved from layer tag.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    model_name, weight_name, is_random = resolve_model_choice(
        model=args.model,
        weight=args.weight,
        is_random=args.is_random,
        results_csv=args.results_csv,
        layer_results_csv=args.layer_results_csv,
    )

    layer_tag = args.layer_tag
    block_name = args.block_name

    if layer_tag is None and block_name is None:
        try:
            layer_tag, block_name = choose_best_layer_from_csv(
                layer_csv_path=args.layer_results_csv,
                model_name=model_name,
                weight_name=weight_name,
                is_random=is_random,
            )
            print(
                f"Best layer from CSV -> layer_tag={layer_tag}, block_name={block_name}"
            )
        except FileNotFoundError:
            layer_tag = "L"

    weight_obj = resolve_weight(model_name, weight_name, is_random)
    layer_model = models.get_model(model_name, weights=weight_obj)
    layer_map = build_layer_tag_to_block_name(layer_model)

    if block_name is None:
        if layer_tag is None:
            layer_tag = "L"
        block_name = layer_map[layer_tag]

    if layer_tag is None:
        reverse_map = {v: k for k, v in layer_map.items()}
        layer_tag = reverse_map.get(block_name, "custom")

    print(
        f"Feature source -> model={model_name}, weight={weight_name}, layer_tag={layer_tag}, block_name={block_name}"
    )

    index_path = extract_and_save(
        dataset_root=args.dataset_root,
        model_name=model_name,
        weight_name=weight_name,
        is_random=is_random,
        layer_tag=layer_tag,
        block_name=block_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Saved feature index: {index_path}")


if __name__ == "__main__":
    main()
