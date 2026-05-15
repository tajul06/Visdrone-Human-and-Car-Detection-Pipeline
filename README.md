# Drone Detection Pipeline

This folder contains a lightweight inference pipeline for person and car detection, counting, and tracking on drone footage using Ultralytics YOLO models. The scripts run on images, folders, videos, or a webcam stream.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Place your trained weights at `pipeline/models/best.pt` or update `model_path` in `config.yaml`.

## Configuration

- `config.yaml` controls model path, input size, confidence thresholds, and class names.
- The default target classes are `person` and `car`.

## Usage

Detection and counting on a single image:

```bash
python pipeline/src/detect_count.py --source path/to/image.jpg
```

Detection and counting on a video:

```bash
python pipeline/src/detect_count.py --source path/to/video.mp4 --show
```

Tracking on a video (ByteTrack by default):

```bash
python pipeline/src/track.py --source path/to/video.mp4
```

Tracking uses BoT-SORT by default. Switch to another tracker by editing
`tracker` in `config.yaml` (e.g., `trackers/botsort.yaml` or `trackers/bytetrack.yaml`).

Available tracker configs under `pipeline/trackers/`:
- `botsort.yaml`
- `bytetrack.yaml`

Webcam stream:

```bash
python pipeline/src/detect_count.py --source 0
```

Web interface (Flask):

```bash
python pipeline/src/web_app.py --host 127.0.0.1 --port 5000
```

Realtime CLI examples:

```bash
python pipeline/src/detect_count.py --source 0
python pipeline/src/detect_count.py --source "rtsp://user:pass@host:554/stream"
```

Realtime web stream:

- Set `stream_source` in `config.yaml` to a webcam index (e.g. `0`) or an RTSP URL.
- Open the web UI and use the Live Stream section.

## Outputs

- Annotated images are saved under `pipeline/outputs/images`.
- Annotated videos are saved under `pipeline/outputs/videos`.
- Web uploads are saved under `pipeline/outputs/web_uploads`.
- Web outputs are saved under `pipeline/outputs/web_outputs`.

## Web UI Preview In README

Use Markdown image syntax to show screenshots (PNG or JPG):

```md
![Web UI Preview](Pipeline/outputs/web_outputs/images/0000000_02021_d_0000005_web.jpg)
```

If you want to control image width in GitHub README, use HTML:

```html
<img src="Pipeline/outputs/web_outputs/images/0000000_02021_d_0000005_web.jpg" alt="Web UI Preview" width="900" />
```

Example render:

![Web UI Preview](Pipeline/outputs/web_outputs/images/0000000_02021_d_0000005_web.jpg)

If you keep screenshots in the `Web UI/` folder at the repository root, reference them directly (note spaces are URL-encoded for reliable rendering):

```md
![Web UI Screenshot](Web%20UI/Screenshot%202026-05-15%20181911.png)
```

Or with size control using HTML:

```html
<img src="Web%20UI/Screenshot%202026-05-15%20184459.png" alt="Web UI Screenshot" width="900" />
```

## Notes

- The pipeline checks for missing paths and empty detections.
- Update `count_classes` in `config.yaml` if you want to count different classes.
