from __future__ import annotations

import argparse
import copy
import importlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

from src.dataset import COCODroneBirdCrops, CropSample

SEED = 1410


@dataclass(slots=True)
class WorkflowConfig:
    dataset_root: Path
    transrate_csv: Path
    checkpoint_dir: Path | None
    checkpoint_index_csv: Path | None
    output_dir: Path
    epochs: int
    batch_size: int
    num_workers: int
    learning_rate: float
    weight_decay: float


class OODSplitDataset(Dataset):
    def __init__(
        self,
        base_ds: COCODroneBirdCrops,
        indices: list[int],
        transform,
        id_label_map: dict[str, int],
        ood_label_name: str,
    ) -> None:
        self.base_ds = base_ds
        self.indices = indices
        self.transform = transform
        self.id_label_map = id_label_map
        self.ood_label_name = ood_label_name

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        sample_idx = self.indices[idx]
        sample: CropSample = self.base_ds.samples[sample_idx]
        with sample.image_path.open("rb") as f:
            from PIL import Image

            img = Image.open(f).convert("RGB")
            crop = img.crop(sample.bbox)

        x = self.transform(crop)
        is_ood = int(sample.label_name == self.ood_label_name)
        y = -1 if is_ood else self.id_label_map[sample.label_name]

        return {
            "image": x,
            "target": y,
            "is_ood": is_ood,
            "label_name": sample.label_name,
            "source_split": sample.source_split,
            "image_path": str(sample.image_path),
            "annotation_id": int(sample.annotation_id),
        }


def set_seed(seed: int = SEED) -> None:
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


def _parse_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _resolve_weight(model_name: str, weight_name: str, is_random: bool):
    if is_random:
        return None

    enum_cls = models.get_model_weights(model_name)
    return getattr(enum_cls, weight_name)


def choose_best_model_from_transrate_csv(transrate_csv: Path) -> tuple[str, str, bool, float]:
    df = pd.read_csv(transrate_csv)
    best = df.sort_values("transrate", ascending=False).iloc[0]
    return (
        str(best["model"]),
        str(best["weight"]),
        _parse_bool(best.get("is_random", False)),
        float(best["transrate"]),
    )


def _extract_transrate_from_object(obj) -> float | None:
    if not isinstance(obj, dict):
        return None

    for key in ["transrate", "best_transrate"]:
        val = obj.get(key)
        if isinstance(val, (int, float)):
            return float(val)

    metadata = obj.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("transrate"), (int, float)):
        return float(metadata["transrate"])

    return None


def _extract_transrate_from_name(ckpt_path: Path) -> float | None:
    for part in ckpt_path.stem.split("__"):
        if not part.startswith("transrate-"):
            continue
        try:
            return float(part.replace("transrate-", ""))
        except ValueError:
            return None
    return None


def _checkpoint_transrate_value(ckpt_path: Path) -> float | None:
    try:
        obj = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        return None

    value = _extract_transrate_from_object(obj)
    if value is not None:
        return value

    return _extract_transrate_from_name(ckpt_path)


def choose_best_checkpoint(
    checkpoint_dir: Path | None,
    checkpoint_index_csv: Path | None,
) -> Path | None:
    if checkpoint_index_csv is not None and checkpoint_index_csv.exists():
        idx_df = pd.read_csv(checkpoint_index_csv)
        if {"checkpoint_path", "transrate"}.issubset(idx_df.columns):
            best = idx_df.sort_values("transrate", ascending=False).iloc[0]
            return Path(best["checkpoint_path"]).expanduser().resolve()

    if checkpoint_dir is None or not checkpoint_dir.exists():
        return None

    candidates = sorted(
        list(checkpoint_dir.glob("*.pt"))
        + list(checkpoint_dir.glob("*.pth"))
        + list(checkpoint_dir.glob("*.ckpt"))
    )
    if not candidates:
        return None

    scored = []
    for path in candidates:
        score = _checkpoint_transrate_value(path)
        if score is not None:
            scored.append((score, path))

    if scored:
        return sorted(scored, key=lambda x: x[0], reverse=True)[0][1]
    return None


def build_splits(
    dataset_root: Path,
    transform,
    ood_label_name: str = "bird",
) -> tuple[OODSplitDataset, OODSplitDataset, OODSplitDataset, dict[str, int]]:
    base_ds = COCODroneBirdCrops(dataset_root=dataset_root, transform=None)

    present_labels = sorted({s.label_name for s in base_ds.samples})
    id_labels = [lbl for lbl in present_labels if lbl != ood_label_name]
    id_label_map = {lbl: i for i, lbl in enumerate(id_labels)}

    train_indices = [
        i
        for i, s in enumerate(base_ds.samples)
        if s.source_split == "train" and s.label_name in id_label_map
    ]
    valid_indices = [
        i
        for i, s in enumerate(base_ds.samples)
        if s.source_split == "valid" and s.label_name in id_label_map
    ]
    test_indices = [
        i
        for i, s in enumerate(base_ds.samples)
        if s.source_split == "test" and (s.label_name in id_label_map or s.label_name == ood_label_name)
    ]

    train_ds = OODSplitDataset(base_ds, train_indices, transform, id_label_map, ood_label_name)
    valid_ds = OODSplitDataset(base_ds, valid_indices, transform, id_label_map, ood_label_name)
    test_ds = OODSplitDataset(base_ds, test_indices, transform, id_label_map, ood_label_name)
    return train_ds, valid_ds, test_ds, id_label_map


def _extract_state_dict(obj: dict) -> dict:
    if "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    return obj


def train_id_only(
    model: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[dict, float]:
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state = None
    best_valid_acc = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False):
            x = batch["image"].to(device)
            y = batch["target"].to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        valid_preds = []
        valid_targets = []
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Epoch {epoch}/{epochs} [valid]", leave=False):
                x = batch["image"].to(device)
                y = batch["target"].to(device)
                logits = model(x)
                pred = torch.argmax(logits, dim=1)
                valid_preds.append(pred.cpu().numpy())
                valid_targets.append(y.cpu().numpy())

        y_pred = np.concatenate(valid_preds, axis=0)
        y_true = np.concatenate(valid_targets, axis=0)
        valid_acc = float(accuracy_score(y_true, y_pred))
        avg_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f}, valid_acc={valid_acc:.4f}")

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return best_state, best_valid_acc


def extract_test_embeddings(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict] = []
    feat_store: list[np.ndarray] = []

    def _hook(_module, _inp, out):
        pooled = out.flatten(1)
        feat_store.append(pooled.detach().cpu().numpy())

    hook_handle = model.avgpool.register_forward_hook(_hook)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing + embedding extraction"):
            x = batch["image"].to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            max_prob, pred_idx = torch.max(probs, dim=1)

            feats = feat_store.pop(0)
            for i in range(x.size(0)):
                label_name = batch["label_name"][i]
                is_ood = int(batch["is_ood"][i])
                true_id_target = int(batch["target"][i])
                pred = int(pred_idx[i].item())

                rows.append(
                    {
                        "is_ood": is_ood,
                        "label_name": label_name,
                        "true_target": true_id_target,
                        "pred_target": pred,
                        "max_softmax": float(max_prob[i].item()),
                        "image_path": batch["image_path"][i],
                        "annotation_id": int(batch["annotation_id"][i]),
                        "source_split": batch["source_split"][i],
                        **{f"f_{j}": float(feats[i, j]) for j in range(feats.shape[1])},
                    }
                )

    hook_handle.remove()
    return pd.DataFrame(rows)


def plot_manifold(
    emb: np.ndarray,
    labels: np.ndarray,
    is_ood: np.ndarray,
    method_name: str,
    output_path: Path,
) -> None:
    plot_df = pd.DataFrame(
        {
            "x": emb[:, 0],
            "y": emb[:, 1],
            "label_name": labels,
            "domain": np.where(is_ood == 1, "OOD (bird)", "ID"),
        }
    )

    unique_labels = sorted(plot_df["label_name"].unique().tolist())
    palette = sns.color_palette("tab10", n_colors=max(3, len(unique_labels)))
    color_map = {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}
    if "bird" in color_map:
        color_map["bird"] = (0.85, 0.2, 0.2)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        data=plot_df,
        x="x",
        y="y",
        hue="label_name",
        style="domain",
        palette=color_map,
        markers={"ID": "o", "OOD (bird)": "X"},
        s=60,
        edgecolor="black",
        linewidth=0.25,
        alpha=0.85,
        ax=ax,
    )
    ax.set_title(f"{method_name}: ID vs OOD Feature Separation")
    ax.set_xlabel(f"{method_name}-1")
    ax.set_ylabel(f"{method_name}-2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_workflow(config: WorkflowConfig) -> None:
    set_seed(SEED)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = choose_best_checkpoint(config.checkpoint_dir, config.checkpoint_index_csv)

    model_name, weight_name, is_random, transrate_score = choose_best_model_from_transrate_csv(config.transrate_csv)
    print(
        f"Best model by TransRate: model={model_name}, weight={weight_name}, is_random={is_random}, transrate={transrate_score:.6f}"
    )

    weight_obj = _resolve_weight(model_name, weight_name, is_random)
    transform = weight_obj.transforms() if weight_obj is not None else default_transform()

    train_ds, valid_ds, test_ds, id_label_map = build_splits(
        dataset_root=config.dataset_root,
        transform=transform,
        ood_label_name="bird",
    )
    print(
        f"Split sizes -> train(ID): {len(train_ds)}, valid(ID): {len(valid_ds)}, test(ID+OOD): {len(test_ds)}"
    )
    print(f"ID classes: {list(id_label_map.keys())}")

    device = get_device()
    model = models.get_model(model_name, weights=weight_obj)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, len(id_label_map))

    if best_ckpt is not None and best_ckpt.exists():
        print(f"Loading selected checkpoint: {best_ckpt}")
        obj = torch.load(best_ckpt, map_location="cpu")
        state_dict = _extract_state_dict(obj)
        model.load_state_dict(state_dict, strict=False)

    initial_state = copy.deepcopy(model.state_dict())

    current_batch_size = config.batch_size
    while True:
        train_loader = DataLoader(
            train_ds,
            batch_size=current_batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=current_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        model.load_state_dict(initial_state)
        model = model.to(device)

        try:
            best_state, best_valid_acc = train_id_only(
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                device=device,
                epochs=config.epochs,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            break
        except torch.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                if current_batch_size > 8:
                    current_batch_size = max(8, current_batch_size // 2)
                    print(f"CUDA OOM: retrying with batch_size={current_batch_size}")
                    continue

                print("CUDA OOM persists. Falling back to CPU.")
                device = torch.device("cpu")
                continue
            raise

    model.load_state_dict(best_state)

    test_loader = DataLoader(
        test_ds,
        batch_size=current_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    weights_path = config.output_dir / "best_id_model_state_dict.pt"
    torch.save(
        {
            "model_name": model_name,
            "weight_name": weight_name,
            "is_random": bool(is_random),
            "transrate": float(transrate_score),
            "id_classes": list(id_label_map.keys()),
            "state_dict": best_state,
            "valid_accuracy": float(best_valid_acc),
        },
        weights_path,
    )
    print(f"Saved decoupled .pt weights: {weights_path}")

    while True:
        try:
            eval_df = extract_test_embeddings(
                model=model,
                test_loader=test_loader,
                device=device,
            )
            break
        except torch.OutOfMemoryError:
            if device.type == "cuda" and current_batch_size > 8:
                torch.cuda.empty_cache()
                current_batch_size = max(8, current_batch_size // 2)
                print(f"CUDA OOM during evaluation: retrying with batch_size={current_batch_size}")
                test_loader = DataLoader(
                    test_ds,
                    batch_size=current_batch_size,
                    shuffle=False,
                    num_workers=config.num_workers,
                    pin_memory=torch.cuda.is_available(),
                )
                continue
            raise

    feat_cols = [c for c in eval_df.columns if c.startswith("f_")]
    X = eval_df[feat_cols].to_numpy(dtype=np.float64)
    X = StandardScaler().fit_transform(X)

    tsne = TSNE(n_components=2, random_state=SEED, learning_rate="auto", init="pca")
    x_tsne = tsne.fit_transform(X)
    plot_manifold(
        emb=x_tsne,
        labels=eval_df["label_name"].to_numpy(),
        is_ood=eval_df["is_ood"].to_numpy(),
        method_name="t-SNE",
        output_path=config.output_dir / "manifold_tsne.png",
    )

    umap_module = importlib.import_module("umap") if importlib.util.find_spec("umap") else None
    if umap_module is not None:
        reducer = umap_module.UMAP(n_components=2, random_state=SEED, n_neighbors=20, min_dist=0.1)
        x_umap = reducer.fit_transform(X)
        plot_manifold(
            emb=x_umap,
            labels=eval_df["label_name"].to_numpy(),
            is_ood=eval_df["is_ood"].to_numpy(),
            method_name="UMAP",
            output_path=config.output_dir / "manifold_umap.png",
        )
    else:
        print("UMAP is unavailable. Install `umap-learn` to generate UMAP plots.")

    eval_csv_path = config.output_dir / "test_embeddings_predictions.csv"
    eval_df.to_csv(eval_csv_path, index=False)

    summary = {
        "model_name": model_name,
        "weight_name": weight_name,
        "is_random": bool(is_random),
        "best_transrate": float(transrate_score),
        "id_classes": list(id_label_map.keys()),
        "valid_accuracy": float(best_valid_acc),
        "weights_path": str(weights_path),
        "embedding_csv": str(eval_csv_path),
        "tsne_plot": str(config.output_dir / "manifold_tsne.png"),
        "umap_plot": str(config.output_dir / "manifold_umap.png"),
    }
    with (config.output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Workflow complete.")


def parse_args() -> WorkflowConfig:
    parser = argparse.ArgumentParser(
        description="OOD-aware workflow: best model selection by TransRate, ID-only training, and manifold visualization.",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/AOD_4"))
    parser.add_argument("--transrate-csv", type=Path, default=Path("resnet_transrate_results.csv"))
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-index-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ood_training"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    return WorkflowConfig(
        dataset_root=args.dataset_root,
        transrate_csv=args.transrate_csv,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_index_csv=args.checkpoint_index_csv,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )


def main() -> None:
    cfg = parse_args()
    run_workflow(cfg)


if __name__ == "__main__":
    main()
