from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

from utils import (
    collect_image_paths,
    draw_boxes,
    ensure_dir,
    is_video_path,
    is_stream_source,
    load_config,
    pipeline_root,
    resolve_class_ids,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track targets with YOLO trackers.")
    parser.add_argument("--source", required=True, help="Image, folder, video, or webcam index")
    parser.add_argument("--config", default=str(pipeline_root() / "config.yaml"))
    parser.add_argument("--output", default=None, help="Override output directory")
    parser.add_argument("--show", action="store_true", help="Show annotated frames")
    return parser.parse_args()


def run_images(model: YOLO, image_paths: list[Path], class_ids: list[int], cfg) -> None:
    output_root = resolve_path(cfg.save.output_dir)
    output_dir = output_root / "images"
    if cfg.save.save_images:
        ensure_dir(output_dir)

    tracker = cfg.tracker
    tracker_path = resolve_path(cfg.tracker)
    if tracker_path.exists():
        tracker = str(tracker_path)

    for image_path in image_paths:
        results = model.track(
            source=str(image_path),
            imgsz=cfg.imgsz,
            conf=cfg.conf,
            iou=cfg.iou,
            classes=class_ids or None,
            device=cfg.device or None,
            tracker=tracker,
            persist=True,
            verbose=False,
        )
        result = results[0]
        frame = result.orig_img.copy()
        track_ids = result.boxes.id if result.boxes is not None else None
        draw_boxes(frame, result.boxes, model.names, track_ids=track_ids)

        if cfg.save.save_images:
            out_path = output_dir / f"{image_path.stem}_track{image_path.suffix}"
            cv2.imwrite(str(out_path), frame)

        if cfg.show:
            cv2.imshow("tracking", frame)
            cv2.waitKey(0)

    if cfg.show:
        cv2.destroyAllWindows()


def run_video(model: YOLO, source: object, class_ids: list[int], cfg) -> None:
    output_root = resolve_path(cfg.save.output_dir)
    output_dir = output_root / "videos"
    if cfg.save.save_video:
        ensure_dir(output_dir)

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if cfg.save.save_video:
        name = (
            "webcam"
            if isinstance(source, int)
            else ("stream" if isinstance(source, str) and "://" in source else Path(str(source)).stem)
        )
        output_path = output_dir / f"{name}_tracking.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    tracker = cfg.tracker
    tracker_path = resolve_path(cfg.tracker)
    if tracker_path.exists():
        tracker = str(tracker_path)

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        results = model.track(
            source=frame,
            imgsz=cfg.imgsz,
            conf=cfg.conf,
            iou=cfg.iou,
            classes=class_ids or None,
            device=cfg.device or None,
            tracker=tracker,
            persist=True,
            verbose=False,
        )
        result = results[0]
        track_ids = result.boxes.id if result.boxes is not None else None
        draw_boxes(frame, result.boxes, model.names, track_ids=track_ids)

        if writer is not None:
            writer.write(frame)

        if cfg.show:
            cv2.imshow("tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    capture.release()
    if writer is not None:
        writer.release()
    if cfg.show:
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    cfg.show = args.show
    if args.output:
        cfg.save.output_dir = args.output

    model_path = resolve_path(cfg.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. Update config.yaml or place weights there."
        )

    model = YOLO(str(model_path))
    class_ids = resolve_class_ids(model.names, cfg.classes)

    source = args.source
    if source.isdigit():
        run_video(model, int(source), class_ids, cfg)
        return

    if is_stream_source(source):
        run_video(model, source, class_ids, cfg)
        return

    source_path = Path(source)
    image_paths = collect_image_paths(source_path)
    if image_paths:
        run_images(model, image_paths, class_ids, cfg)
        return

    if source_path.is_file() and is_video_path(source_path):
        run_video(model, str(source_path), class_ids, cfg)
        return

    raise FileNotFoundError(f"Unsupported source: {source}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI entrypoint
        print(str(exc), file=sys.stderr)
        sys.exit(1)
