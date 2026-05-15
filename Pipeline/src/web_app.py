from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
from flask import Flask, Response, flash, redirect, render_template, request, send_from_directory, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename

from utils import (
    IMAGE_EXTS,
    VIDEO_EXTS,
    count_by_class,
    draw_boxes,
    ensure_dir,
    load_config,
    overlay_counts,
    pipeline_root,
    resolve_class_ids,
    resolve_path,
)

CFG_PATH = pipeline_root() / "config.yaml"
CFG = load_config(CFG_PATH)
OUTPUT_ROOT = resolve_path(CFG.save.output_dir)
UPLOAD_DIR = OUTPUT_ROOT / "web_uploads"
WEB_OUTPUT_DIR = OUTPUT_ROOT / "web_outputs"
WEB_IMAGES_DIR = WEB_OUTPUT_DIR / "images"
WEB_VIDEOS_DIR = WEB_OUTPUT_DIR / "videos"

APP = Flask(
    __name__,
    template_folder=str(pipeline_root() / "web" / "templates"),
    static_folder=str(pipeline_root() / "web" / "static"),
)
APP.config["SECRET_KEY"] = "local-dev"

_MODEL: Optional[YOLO] = None
_CLASS_IDS: list[int] = []
_COUNT_IDS: list[int] = []


def get_model() -> YOLO:
    global _MODEL, _CLASS_IDS, _COUNT_IDS
    if _MODEL is None:
        model_path = resolve_path(CFG.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {model_path}. Update config.yaml or place weights there."
            )
        _MODEL = YOLO(str(model_path))
        _CLASS_IDS = resolve_class_ids(_MODEL.names, CFG.classes)
        _COUNT_IDS = resolve_class_ids(_MODEL.names, CFG.count_classes)
    return _MODEL


def ensure_web_dirs() -> None:
    ensure_dir(UPLOAD_DIR)
    ensure_dir(WEB_IMAGES_DIR)
    ensure_dir(WEB_VIDEOS_DIR)


def relative_to_output(path: Path) -> str:
    return path.relative_to(OUTPUT_ROOT).as_posix()


def run_image(image_path: Path) -> tuple[Path, dict[str, int]]:
    model = get_model()
    results = model.predict(
        source=str(image_path),
        imgsz=CFG.imgsz,
        conf=CFG.conf,
        iou=CFG.iou,
        classes=_CLASS_IDS or None,
        device=CFG.device or None,
        verbose=False,
    )
    result = results[0]
    frame = result.orig_img.copy()
    counts = count_by_class(result.boxes, _COUNT_IDS)
    draw_boxes(frame, result.boxes, model.names)
    overlay_counts(frame, counts, model.names)

    out_path = WEB_IMAGES_DIR / f"{image_path.stem}_web{image_path.suffix}"
    cv2.imwrite(str(out_path), frame)

    labeled_counts = {model.names.get(k, str(k)): v for k, v in counts.items()}
    return out_path, labeled_counts


def run_video(video_path: Path) -> Path:
    model = get_model()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = WEB_VIDEOS_DIR / f"{video_path.stem}_web.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        results = model.predict(
            source=frame,
            imgsz=CFG.imgsz,
            conf=CFG.conf,
            iou=CFG.iou,
            classes=_CLASS_IDS or None,
            device=CFG.device or None,
            verbose=False,
        )
        result = results[0]
        counts = count_by_class(result.boxes, _COUNT_IDS)
        draw_boxes(frame, result.boxes, model.names)
        overlay_counts(frame, counts, model.names)
        writer.write(frame)

    capture.release()
    writer.release()
    return output_path


def resolve_stream_source(override: Optional[str] = None) -> object:
    source = override or str(getattr(CFG, "stream_source", "0"))
    source = str(source).strip()
    return int(source) if source.isdigit() else source


def generate_stream(override: Optional[str] = None):
    model = get_model()
    source = resolve_stream_source(override)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open stream source: {source}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            results = model.predict(
                source=frame,
                imgsz=CFG.imgsz,
                conf=CFG.conf,
                iou=CFG.iou,
                classes=_CLASS_IDS or None,
                device=CFG.device or None,
                verbose=False,
            )
            result = results[0]
            counts = count_by_class(result.boxes, _COUNT_IDS)
            draw_boxes(frame, result.boxes, model.names)
            overlay_counts(frame, counts, model.names)

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
    finally:
        capture.release()


def normalize_filename(filename: str, mime_type: str) -> str:
    safe_name = secure_filename(filename)
    if Path(safe_name).suffix:
        return safe_name
    if mime_type.startswith("image/"):
        return f"{safe_name}.jpg"
    if mime_type.startswith("video/"):
        return f"{safe_name}.mp4"
    return safe_name


def allowed_file(path: Path, mime_type: str) -> bool:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTS or suffix in VIDEO_EXTS:
        return True
    return mime_type.startswith("image/") or mime_type.startswith("video/")


@APP.route("/", methods=["GET", "POST"])
def index():
    ensure_web_dirs()
    result_url = None
    result_type = None
    counts = None

    if request.method == "POST":
        file = request.files.get("file")
        if file is None or file.filename == "":
            flash("Select an image or video to upload.")
            return redirect(url_for("index"))

        mime_type = file.mimetype or ""
        filename = normalize_filename(file.filename, mime_type)
        upload_path = UPLOAD_DIR / filename
        if not allowed_file(upload_path, mime_type):
            flash("Unsupported file type. Use common image or video formats.")
            return redirect(url_for("index"))

        file.save(str(upload_path))

        try:
            if upload_path.suffix.lower() in IMAGE_EXTS:
                output_path, counts = run_image(upload_path)
                result_type = "image"
            else:
                output_path = run_video(upload_path)
                result_type = "video"
        except Exception as exc:  # pragma: no cover - UI error handling
            flash(str(exc))
            return redirect(url_for("index"))

        result_url = url_for("outputs", filename=relative_to_output(output_path))

    return render_template(
        "index.html",
        result_url=result_url,
        result_type=result_type,
        counts=counts,
        image_exts=sorted(IMAGE_EXTS),
        video_exts=sorted(VIDEO_EXTS),
    )


@APP.route("/outputs/<path:filename>")
def outputs(filename: str):
    return send_from_directory(OUTPUT_ROOT, filename)


@APP.route("/stream")
def stream():
    try:
        source = request.args.get("source")
        return Response(
            generate_stream(source), mimetype="multipart/x-mixed-replace; boundary=frame"
        )
    except Exception as exc:  # pragma: no cover - UI error handling
        return str(exc), 400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flask web interface for drone detection.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=5000, type=int)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    APP.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
