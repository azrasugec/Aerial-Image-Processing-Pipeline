# Aerial Image Processing Pipeline

A Python pipeline for processing aerial remote sensing images through noise injection, median filtering, and object detection.

---

## Pipeline Overview

| Phase | Description |
|-------|-------------|
| A | RGB channel splitting — extracts R, G, B as separate grayscale PNGs |
| B | Red channel histogram generation |
| C | Salt-and-pepper noise injection (manual implementation, no noise library) |
| D | Manual pixel audit — 5×5 window extraction, DN value sorting, median calculation |
| E | Manual 3×3 median filter (no cv2 / scipy / skimage used) |
| F | Texture sensitivity analysis — compares edge strength and contrast across scene types |
| G | Object detection — manual Otsu thresholding, morphological opening, BFS connected components, bounding boxes |

---

## Dataset

Images are from the [AID (Aerial Image Dataset)](https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets) — a large-scale aerial scene classification dataset collected from Google Earth.

Tested on:
- `Bridge_62.jpg`
- `Center_62.jpg`
- `Church_62.jpg`

---

## Requirements

```
numpy
Pillow
matplotlib
tkinter  # built-in with Python
```

```bash
pip install numpy Pillow matplotlib
```

---

## Usage

```bash
python main.py
```

On startup:
1. A dialog asks how many images to process
2. File pickers open for each image (`.jpg`, `.jpeg`, `.png`)
3. Enter a noise ratio (e.g. `0.05` for 5%)
4. Enter row/col coordinates for the Phase D pixel audit per image
5. All outputs are saved automatically

---

## Output Structure

```
outputs/
├── phaseA/    # channel PNGs
├── phaseB/    # histograms
├── phaseC/    # noisy images
├── phaseD/    # audit reports (.txt) + charts
├── phaseE/    # filtered images + difference maps
├── phaseF/    # texture analysis report + bar charts
└── phaseG/    # detection results + full pipeline comparisons
```

---

## Sample Results

### Phase A — RGB Channel Split

| Bridge | Center | Church |
|--------|--------|--------|
| ![](bridge_62_channel_split.png) | ![](center_62_channel_split.png) | ![](church_62_channel_split.png) |

---

### Phase C — Salt & Pepper Noise

| Bridge | Center | Church |
|--------|--------|--------|
| ![](bridge_62_noise_comparison.png) | ![](center_62_noise_comparison.png) | ![](church_62_noise_comparison.png) |

---

### Phase E — Median Filter

| Bridge | Center | Church |
|--------|--------|--------|
| ![](bridge_62_filter_comparison.png) | ![](center_62_filter_comparison.png) | ![](church_62_filter_comparison.png) |

---

### Phase F — Texture Analysis

![](texture_metrics_chart.png)

---

### Phase G — Object Detection

**Full comparison (filtered input + detections):**

| Bridge | Center | Church |
|--------|--------|--------|
| ![](bridge_62_detection_comparison.png) | ![](center_62_detection_comparison.png) | ![](church_62_detection_comparison.png) |

**Detection close-ups:**

| Bridge | Center | Church |
|--------|--------|--------|
| ![](detected_bridge_62_closeup.png) | ![](detected_center_62_closeup.png) | ![](detected_church_62_closeup.png) |

---

## Key Implementation Notes

- **Noise injection** uses a random float map with a fixed seed (42) — no external noise library
- **Median filter** uses only Python's built-in `list.sort()` on 3×3 neighbourhoods
- **Object detection** runs manual Otsu thresholding and a pure-Python BFS labeller — no OpenCV
- **MSE and PSNR** are computed from scratch to measure filter quality
