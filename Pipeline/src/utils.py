from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass
class SaveConfig:
    output_dir: str = "outputs"
    save_images: bool = True
    save_video: bool = True


@dataclass
class PipelineConfig:
    model_path: str = "models/best.pt"
    device: str = ""
    imgsz: int = 1280
    conf: float = 0.25
    iou: float = 0.5
    show: bool = False
    stream_source: str = "0"
    classes: list[str] = field(default_factory=lambda: ["person", "car"])
    count_classes: list[str] = field(default_factory=lambda: ["person"])
    tracker: str = "bytetrack.yaml"
    save: SaveConfig = field(default_factory=SaveConfig)


def pipeline_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> PipelineConfig:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    defaults = PipelineConfig()
    save_data = data.get("save", {})
    save_cfg = SaveConfig(
        output_dir=save_data.get("output_dir", defaults.save.output_dir),
        save_images=save_data.get("save_images", defaults.save.save_images),
        save_video=save_data.get("save_video", defaults.save.save_video),
    )
    return PipelineConfig(
        model_path=data.get("model_path", defaults.model_path),
        device=data.get("device", defaults.device),
        imgsz=int(data.get("imgsz", defaults.imgsz)),
        conf=float(data.get("conf", defaults.conf)),
        iou=float(data.get("iou", defaults.iou)),
        show=bool(data.get("show", defaults.show)),
        stream_source=str(data.get("stream_source", defaults.stream_source)),
        classes=list(data.get("classes", defaults.classes)),
        count_classes=list(data.get("count_classes", defaults.count_classes)),
        tracker=str(data.get("tracker", defaults.tracker)),
        save=save_cfg,
    )


def resolve_path(value: str, base: Optional[Path] = None) -> Path:
    base = base or pipeline_root()
    path = Path(value)
    return path if path.is_absolute() else (base / path)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_video_path(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def is_stream_source(value: str) -> bool:
    lowered = value.lower()
    return lowered.startswith(("rtsp://", "rtmp://", "http://", "https://"))


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def collect_image_paths(path: Path) -> list[Path]:
    if path.is_file() and is_image_path(path):
        return [path]
    if path.is_dir():
        paths = [p for p in path.iterdir() if p.is_file() and is_image_path(p)]
        return sorted(paths)
    return []


def resolve_class_ids(model_names: dict[int, str], targets: Iterable[object]) -> list[int]:
    name_to_id = {v: k for k, v in model_names.items()}
    resolved: list[int] = []
    for item in targets:
        if isinstance(item, int) and item in model_names:
            resolved.append(item)
            continue
        if isinstance(item, str) and item in name_to_id:
            resolved.append(name_to_id[item])
    return sorted(set(resolved))


def to_numpy(value: object) -> np.ndarray:
    try:
        return value.cpu().numpy()
    except AttributeError:
        return np.asarray(value)


def color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = [
        (255, 99, 71),
        (65, 105, 225),
        (46, 139, 87),
        (218, 165, 32),
        (199, 21, 133),
        (0, 139, 139),
    ]
    return palette[class_id % len(palette)]


def draw_boxes(
    image: np.ndarray,
    boxes: object,
    names: dict[int, str],
    track_ids: Optional[np.ndarray] = None,
) -> None:
    if boxes is None or len(boxes) == 0:
        return
    xyxy = to_numpy(boxes.xyxy)
    cls_ids = to_numpy(boxes.cls).astype(int)
    confs = to_numpy(boxes.conf)
    track_ids = track_ids if track_ids is None else track_ids.astype(int)

    for i, coords in enumerate(xyxy):
        x1, y1, x2, y2 = [int(v) for v in coords]
        class_id = int(cls_ids[i])
        conf = float(confs[i])
        label = f"{names.get(class_id, str(class_id))} {conf:.2f}"
        if track_ids is not None:
            label = f"{label} id={int(track_ids[i])}"
        color = color_for_class(class_id)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            label,
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )


def count_by_class(boxes: object, target_ids: list[int]) -> dict[int, int]:
    counts = {class_id: 0 for class_id in target_ids}
    if boxes is None or len(boxes) == 0:
        return counts
    cls_ids = to_numpy(boxes.cls).astype(int)
    for class_id in cls_ids:
        if class_id in counts:
            counts[class_id] += 1
    return counts


def overlay_counts(image: np.ndarray, counts: dict[int, int], names: dict[int, str]) -> None:
    y = 25
    for class_id, count in counts.items():
        label = f"{names.get(class_id, str(class_id))}: {count}"
        cv2.putText(
            image,
            label,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24
