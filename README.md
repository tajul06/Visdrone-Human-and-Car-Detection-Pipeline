# Drone Detection Pipeline

This Project contains a lightweight inference pipeline for person and car detection, counting, and tracking on drone footage using Ultralytics YOLO models. The scripts run on images, folders, videos, or a webcam stream.



## Project Structure

```
Visdrone-Human-and-Car-Detection-Pipeline/
│
├── Dataset Exploration/
│   ├── exploration_result.ipynb     # Phase 1: full EDA and dataset analysis notebook
│   └── results/                     # Saved charts (class distribution, bbox sizes, samples)
│
├── Pipeline/                        # Phase 5: inference pipeline
│   ├── config.yaml                  # Runtime config — model path, imgsz, conf, tracker
│   ├── requirements.txt
│   ├── models/
│   │   └── yolo26s_visdrone.pt      # Fine-tuned checkpoint (place weights here)
│   ├── outputs/
│   │   ├── images/                  # Annotated image outputs
│   │   └── web_uploads/             # Uploads and results from Flask UI
│   ├── src/
│   │   ├── detect_count.py          # Detection + counting entry point
│   │   ├── track.py                 # Tracking entry point (BoT-SORT / ByteTrack)
│   │   ├── web_app.py               # Flask web interface
│   │   ├── utils.py                 # Shared helpers
│   │   └── __init__.py
│   ├── trackers/
│   │   ├── botsort.yaml             # BoT-SORT config (primary — has GMC)
│   │   └── bytetrack.yaml           # ByteTrack config (fast fallback)
│   └── web/
│       ├── static/
│       │   └── styles.css
│       └── templates/
│           └── index.html
│
├── Training & Fine Tuning/          # Phase 3: training notebooks
│   ├── visdrone-rt-detr-training.ipynb
│   ├── yolo11s_visdrone.ipynb
│   ├── yolo26m_visdrone.ipynb
│   ├── yolo26s-hbb-visdrone.ipynb
│   ├── yolo26s-obb-visdrone.ipynb
│   └── Training Results/            # Per-model saved weights and inference results
│       ├── RETDER-l/
│       │   ├── results.csv
│       │   └── Inference/
│       ├── Yolo11s/
│       │   └── Inference/
│       │       └── results.csv
│       ├── Yolo26m/
│       │   └── Inference/
│       ├── yolo26s/
│       │   └── Inference/
│       └── Yolo26s-OBB/
│           └── Inference/
│
├── Web UI/                          # Flask web interface screenshots
└── README.md
```

---

## Tech Stack

| Tool | Role |
|------|------|
| **Python 3.10+** | Core language |
| **Ultralytics (YOLO)** | Model training, inference, tracking |
| **PyTorch** | Deep learning backend |
| **OpenCV** | Frame reading, annotation, video I/O |
| **NumPy** | Bounding box math, label manipulation |
| **Flask** | Web inference interface |
| **PyYAML** | Config management |
| **Matplotlib / Seaborn** | Plots and dataset visualizations |
| **Kaggle / T4 GPU** | Training environment (2× Tesla T4, 16 GB VRAM each) |

---

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

## Web UI

Run the Flask web interface to view live stream and upload results:

```bash
python pipeline/src/web_app.py --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000 in a browser. The web UI shows a Live Stream viewer and a Web Uploads gallery.

Screenshots (kept in the repository `Web UI/` folder):

![Web UI 1](Web%20UI/Screenshot%202026-05-15%20181911.png)

![Web UI 2](Web%20UI/Screenshot%202026-05-15%20184459.png)

## Notes

- The pipeline checks for missing paths and empty detections.
- Update `count_classes` in `config.yaml` if you want to count different classes.
