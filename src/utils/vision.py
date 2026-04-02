from __future__ import annotations

from torchvision.transforms import Compose, Normalize, Resize, ToTensor


def imagenet_eval_transform_224() -> Compose:
    return Compose(
        [
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
