from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
from torch import Generator
from torch.utils.data import Dataset, Subset, random_split


@dataclass(slots=True)
class CropSample:
    image_path: Path
    bbox: tuple[float, float, float, float]
    label: int
    label_name: str
    source_split: str
    annotation_id: int


class COCODroneBirdCrops(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        include_labels: Iterable[str] | None = None,
        transform=None,
        min_bbox_size: float = 4.0,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.transform = transform
        self.min_bbox_size = min_bbox_size

        if include_labels is None:
            self.selected_labels: tuple[str, ...] | None = None
            self.label_to_index: dict[str, int] = {}
        else:
            selected = tuple(dict.fromkeys(lbl.lower() for lbl in include_labels))
            self.selected_labels = selected
            self.label_to_index = {name: idx for idx, name in enumerate(selected)}

        self.index_to_label = {idx: name for name, idx in self.label_to_index.items()}
        self.samples: list[CropSample] = []

        self._build_index()

    def _build_index(self) -> None:
        annotation_root = self.dataset_root / "Annotations" / "COCO Annotation format"
        image_root = self.dataset_root / "Images"
        ann_paths = sorted(annotation_root.glob("*/_annotations.coco.json"))

        if self.selected_labels is None:
            all_labels: set[str] = set()
            for ann_path in ann_paths:
                with ann_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                all_labels.update(
                    str(c["name"]).lower() for c in data.get("categories", [])
                )

            selected = tuple(sorted(all_labels))
            self.label_to_index = {name: idx for idx, name in enumerate(selected)}
            self.index_to_label = {
                idx: name for name, idx in self.label_to_index.items()
            }

        for ann_path in ann_paths:
            split = ann_path.parent.name

            with ann_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            category_map = {
                int(c["id"]): str(c["name"]).lower() for c in data.get("categories", [])
            }
            images_by_id = {int(img["id"]): img for img in data.get("images", [])}

            for ann in data.get("annotations", []):
                sample = self._build_sample(
                    split, ann, category_map, images_by_id, image_root
                )
                if sample is not None:
                    self.samples.append(sample)

    def _build_sample(
        self,
        split: str,
        ann: dict,
        category_map: dict[int, str],
        images_by_id: dict[int, dict],
        image_root: Path,
    ) -> CropSample | None:
        category_name = category_map.get(int(ann["category_id"]), "")
        if category_name not in self.label_to_index:
            return None

        img_info = images_by_id.get(int(ann["image_id"]))
        if img_info is None:
            return None

        bbox = self._clip_bbox(ann.get("bbox", [0.0, 0.0, 0.0, 0.0]), img_info)
        if bbox is None:
            return None

        img_path = image_root / split / str(img_info["file_name"])
        if not img_path.exists():
            return None

        return CropSample(
            image_path=img_path,
            bbox=bbox,
            label=self.label_to_index[category_name],
            label_name=category_name,
            source_split=split,
            annotation_id=int(ann.get("id", -1)),
        )

    def _clip_bbox(
        self, bbox: list[float], img_info: dict
    ) -> tuple[float, float, float, float] | None:
        x, y, w, h = map(float, bbox)
        if w < self.min_bbox_size or h < self.min_bbox_size:
            return None

        img_w = float(img_info.get("width", 0))
        img_h = float(img_info.get("height", 0))
        x0 = max(0.0, min(x, img_w - 1.0))
        y0 = max(0.0, min(y, img_h - 1.0))
        x1 = max(x0 + 1.0, min(x + w, img_w))
        y1 = max(y0 + 1.0, min(y + h, img_h))
        return x0, y0, x1, y1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.image_path) as img:
            rgb = img.convert("RGB")
            crop = rgb.crop(sample.bbox)

        if self.transform is not None:
            crop = self.transform(crop)

        return crop, sample.label


def train_valid_test_split(
    dataset: Dataset,
    train_ratio,
    test_ratio,
    seed: int,
) -> tuple[Subset, Subset, Subset]:
    n = len(dataset)
    train_len = int(n * train_ratio)
    test_len = int(n * test_ratio)
    valid_len = n - train_len - test_len

    generator = Generator().manual_seed(seed)
    train_set, valid_set, test_set = random_split(
        dataset,
        lengths=[train_len, valid_len, test_len],
        generator=generator,
    )
    return train_set, valid_set, test_set
