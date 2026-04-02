from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torchvision import models

import src.utils.const as CONST
from src.utils.set_seed import set_seed


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def choose_best_model_from_csv(csv_path: Path) -> tuple[str, str, bool]:
    df = pd.read_csv(csv_path)
    best = df.sort_values("transrate", ascending=False).iloc[0]
    return str(best["model"]), str(best["weight"]), parse_bool(best["is_random"])


def choose_best_block_from_csv(csv_path: Path) -> str:
    df = pd.read_csv(csv_path)
    best = df.sort_values("transrate", ascending=False).iloc[0]
    return str(best["block_name"])


def resolve_weight(model_name: str, weight_name: str, is_random: bool):
    if is_random:
        return None
    enum_cls = models.get_model_weights(model_name)
    return getattr(enum_cls, weight_name)


def get_resnet_blocks(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    out: dict[str, torch.nn.Module] = {}
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, layer_name, None)
        if layer is None:
            continue
        for i, block in enumerate(layer):
            out[f"{layer_name}.{i}"] = block
    return out


def load_backbone_and_target_block(
    model_name: str,
    weight_name: str,
    is_random: bool,
    block_name: str,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    if is_random:
        set_seed(CONST.SEED)

    weight_obj = resolve_weight(model_name, weight_name, is_random)
    model = models.get_model(model_name, weights=weight_obj).to(device)
    model.eval()

    blocks = get_resnet_blocks(model)
    if block_name not in blocks:
        available = ", ".join(sorted(blocks.keys()))
        raise ValueError(
            f"Block '{block_name}' not found in {model_name}. Available: {available}"
        )

    return model, blocks[block_name]
