# Crowd Rush Prediction & Early Warning System

An end-to-end **Computer Vision and Machine Learning** decision-support system for estimating crowd count, spatial density, pedestrian movement dynamics, overcrowding risk (`LOW`, `MODERATE`, `HIGH`, `CRITICAL`), `+5/+10/+15` minute risk trends, and explainable safety recommendations.

> **Scientific & Ethical Notice:** This system is an **early-warning decision-support tool**, not an automated stampede guarantee. Public Computer Vision datasets provide head points and tracking boxes rather than real-world disaster labels; accordingly, real CV benchmark observations are strictly separated from documented synthetic safety scenario modeling.

---

## Project Architecture & Folder Structure

```text
crowd-rush-prediction/
├── dashboard/
│   └── app.py                          # 7-page Streamlit Dashboard (Dataset Explorer, no manual upload)
├── datasets/
│   ├── nwpu_crowd/                     # 5 schema samples + real_val_subset/ (10 official NWPU val photos)
│   ├── ucf_qnrf/                       # 3 UCF-QNRF schema samples + splits
│   ├── ucf_crowd/                      # 2 optical flow (.npy) & segmentation (_seg.png) samples
│   ├── motsynth/                       # MOTSynth_001_rush (30 frames + MOT gt.txt)
│   └── ucf_tracking/                   # ucf_dense_crowd_corridor.mp4 (125 frames) + tracks
├── docs/
│   ├── FINAL_PROJECT_REPORT.md         # Full academic report with real observed metrics
│   ├── SYSTEM_ARCHITECTURE.md          # Mermaid architecture, DFD Level 0/1, flowcharts, model diagrams
│   ├── PROJECT_EXPLANATION_AND_LEARNING_GUIDE.md  # Beginner-friendly guide to every model & file
│   ├── PRESENTATION_SLIDES.md          # 15-slide defense presentation deck
│   └── VIVA_QUESTIONS_AND_ANSWERS.md   # 45 project-specific viva Q&As
├── models/
│   ├── detection/yolov8n.pt            # Ultralytics YOLOv8n pretrained weights
│   ├── crowd_counting/csrnet_weights.pth # 8-epoch CPU-calibrated CSRNet weights
│   └── risk_prediction/                # rf_risk_model.joblib, xgb_risk_model.joblib, scaler.joblib
├── results/
│   ├── metrics/                        # detection, counting, tracking, ucf_crowd_flow, risk JSONs
│   ├── graphs/                         # confusion_matrices, feature_importance, roc_curves, counting_mae_comparison, risk_timeline
│   ├── heatmaps/                       # Saved YOLOv8 overlays, CSRNet density heatmaps, flow & tracking frames
│   └── predictions/                    # extracted_crowd_features.csv, simulated_risk_scenarios.csv, experimental_eval_summary.csv
├── src/
│   ├── data_preprocessing/             # Dataset inspector, parsers, Gaussian density maps, NWPU subset downloader
│   ├── detection/                      # Module A: YOLOv8n person detector & Point-in-Box / IoU evaluator
│   ├── counting/                       # Module B: CSRNet VGG-16 + dilated CNN & counting evaluator
│   ├── tracking/                       # Module C: ByteTrack two-stage tracker & motion entropy analyzer
│   ├── feature_extraction/             # Module D: Spatio-temporal crowd feature extractor
│   ├── risk_prediction/                # Synthetic scenarios, RF vs XGBoost, +5/10/15m forecaster, explainer
│   ├── early_warning/                  # Rule-based 4-tier warning & crowd safety directive engine
│   └── evaluation/                     # Benchmark runner & consolidated CSV metrics logger
├── main.py                             # Unified CLI orchestrator
├── requirements.txt
└── README.md
```

---

## Setup & Installation (Windows PowerShell)

```powershell
# 1. Create and activate Python virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install verified dependencies
pip install -r requirements.txt
```

---

## Running the Project (`main.py` & Streamlit)

```powershell
# Inspect all datasets, modalities, and annotations
.\venv\Scripts\python.exe main.py --inspect

# Download the 10-image Official NWPU-Crowd Validation Subset (via HTTP Range, ~30 MB)
.\venv\Scripts\python.exe main.py --download-nwpu

# View official access guide for all 5 full external datasets
.\venv\Scripts\python.exe main.py --dataset-guide

# Run lightweight evaluation (UCF Flow, UCF Video Telemetry, Risk Timeline, Summary CSV)
.\venv\Scripts\python.exe main.py --evaluate

# Recompute all heavy CV benchmarks (YOLOv8 detection, CSRNet counting, ByteTrack, Features)
.\venv\Scripts\python.exe main.py --evaluate --full

# Launch the interactive Streamlit Dashboard
.\venv\Scripts\python.exe main.py --dashboard
```

---

## Verified Experimental Highlights

| Module | Model | Dataset / Evaluation Basis | Measured Metrics |
|---|---|---|---|
| **Module A: Detection** | `YOLOv8n` | **NWPU-Crowd Official Real Val (10 photos)** | Precision `0.914`, Low Recall `0.889`, Extreme Recall `0.0024` (`312.1 ms`, `3.2 FPS`) |
| **Module B: Counting** | `CSRNet` (8-ep CPU) | **NWPU-Crowd Official Real Val (10 photos)** | Overall MAE `1039.31`; **Extreme MAE `685.85`** vs. YOLOv8 **`2281.50`** (`3.33x` better) |
| **Module C: Tracking** | `ByteTrack` | `MOTSynth_001_rush` (GT-noisy box fallback) | MOTA `0.9187` (`91.9%`), IDF1 `0.9681` (`96.8%`), `14` ID switches, `14.7 FPS` |
| **Module D: Risk** | `Random Forest` | Synthetic Fruin/Still Scenarios (`400` test) | Accuracy `1.0000`, Macro F1 `1.0000`, `CRITICAL` Recall `1.0000`, `0.082 ms` |
| **Module D: Risk** | `XGBoost` | Synthetic Fruin/Still Scenarios (`400` test) | Accuracy `0.9950`, Macro F1 `0.9950`, `CRITICAL` Recall `1.0000`, `0.005 ms` |

---

## Academic Documentation Suite (`docs/`)

- [FINAL_PROJECT_REPORT.md](docs/FINAL_PROJECT_REPORT.md) — Complete academic project report, methodology, real metrics, limitations.
- [SYSTEM_ARCHITECTURE.md](docs/SYSTEM_ARCHITECTURE.md) — Mermaid system architecture, DFD Level 0 & 1, flowchart, and neural network diagrams.
- [PROJECT_EXPLANATION_AND_LEARNING_GUIDE.md](docs/PROJECT_EXPLANATION_AND_LEARNING_GUIDE.md) — Beginner-friendly explanation of all 11 core models/technologies and every Python file.
- [PRESENTATION_SLIDES.md](docs/PRESENTATION_SLIDES.md) — 15-slide defense presentation deck.
- [VIVA_QUESTIONS_AND_ANSWERS.md](docs/VIVA_QUESTIONS_AND_ANSWERS.md) — 45 project-specific viva questions with clear answers.
