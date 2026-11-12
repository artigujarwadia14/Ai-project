"""
End-to-End Evaluation & Benchmark Runner
Coordinates evaluations across all 5 dataset directories, generates optical flow
and video tracking evaluations for UCF Crowd & UCF Tracking, scores extracted
crowd features with trained RF/XGBoost models to produce risk_timeline.png, and
consolidates experimental_eval_summary.csv.
"""

import json
import sys
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from typing import Dict, Any, List
import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_preprocessing.dataset_inspector import DatasetInspector
from src.data_preprocessing.ucf_crowd_parser import UCFCrowdDatasetParser
from src.data_preprocessing.ucf_tracking_parser import UCFTrackingDatasetParser
from src.tracking.byte_tracker import ByteTracker
from src.tracking.motion_analyzer import CrowdMotionAnalyzer
from src.risk_prediction.risk_classifier import FEATURE_COLUMNS, CLASS_NAMES
from src.evaluation.metrics_logger import build_experimental_summary_csv


def evaluate_ucf_crowd_flow(
    ucf_crowd_dir: str = "datasets/ucf_crowd",
    output_json: str = "results/metrics/ucf_crowd_flow_metrics.json",
    save_vis_dir: str = "results/heatmaps/ucf_crowd_flow"
) -> Dict[str, Any]:
    """
    Evaluates optical flow fields (.npy) and spatial segmentation masks (_seg.png)
    in the UCF Crowd Segmentation dataset.
    """
    parser = UCFCrowdDatasetParser(ucf_crowd_dir)
    samples = parser.list_samples()
    vis_dir = Path(save_vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    per_sample = []
    latencies = []

    for s_name in samples:
        t0 = time.perf_counter()
        info = parser.parse_sample(s_name)
        img_bgr = cv2.imread(info["image_path"])
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        mean_mag, max_mag, dominant_deg, crowd_cov_pct = 0.0, 0.0, 0.0, 0.0
        vis_img = img_rgb.copy()

        if info["seg_path"] and Path(info["seg_path"]).exists():
            seg = cv2.imread(info["seg_path"], cv2.IMREAD_GRAYSCALE)
            if seg is not None:
                crowd_cov_pct = float(round((np.sum(seg > 127) / float(seg.size)) * 100.0, 2))
                mask_color = np.zeros_like(vis_img)
                mask_color[:, :, 0] = np.where(seg > 127, 255, 0)  # highlight segmented crowd in red tint
                vis_img = cv2.addWeighted(vis_img, 0.75, mask_color, 0.25, 0)

        if info["flow_path"] and Path(info["flow_path"]).exists():
            flow = np.load(info["flow_path"])
            vx = flow[:, :, 0]
            vy = flow[:, :, 1]
            mag = np.sqrt(vx ** 2 + vy ** 2)
            active = mag > 0.05
            if np.any(active):
                mean_mag = float(round(np.mean(mag[active]), 3))
                max_mag = float(round(np.max(mag[active]), 3))
                dominant_deg = float(round(np.degrees(np.arctan2(np.mean(vy[active]), np.mean(vx[active]))) % 360, 1))

                # Draw grid flow arrows on visualization
                step = 32
                for y in range(step // 2, h, step):
                    for x in range(step // 2, w, step):
                        fx, fy = float(vx[y, x]), float(vy[y, x])
                        if np.hypot(fx, fy) > 0.2:
                            cv2.arrowedLine(
                                vis_img,
                                (x, y),
                                (int(x + fx * 6), int(y + fy * 6)),
                                (0, 255, 255),
                                2,
                                tipLength=0.35
                            )

        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)
        cv2.imwrite(str(vis_dir / f"flow_{Path(s_name).stem}.jpg"), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

        per_sample.append({
            "sample_name": s_name,
            "resolution": f"{w}x{h}",
            "mean_flow_magnitude_px": mean_mag,
            "max_flow_magnitude_px": max_mag,
            "dominant_flow_angle_deg": dominant_deg,
            "crowd_segmentation_coverage_pct": crowd_cov_pct,
            "latency_ms": float(round(dt_ms, 2)),
        })

    summary = {
        "dataset": "UCF Crowd Segmentation (Schema Samples)",
        "sample_count": len(per_sample),
        "mean_flow_magnitude_px": float(round(np.mean([x["mean_flow_magnitude_px"] for x in per_sample]), 3)) if per_sample else 0.0,
        "mean_crowd_area_pct": float(round(np.mean([x["crowd_segmentation_coverage_pct"] for x in per_sample]), 2)) if per_sample else 0.0,
        "mean_latency_ms": float(round(np.mean(latencies), 2)) if latencies else 0.0,
        "per_sample": per_sample,
    }

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def evaluate_ucf_tracking_video(
    ucf_tracking_dir: str = "datasets/ucf_tracking",
    tracking_json: str = "results/metrics/tracking_metrics.json"
) -> Dict[str, Any]:
    """
    Runs ByteTrack motion & trajectory telemetry on UCF Tracking dense video sequences
    and appends ucf_tracking_results into tracking_metrics.json.
    """
    parser = UCFTrackingDatasetParser(ucf_tracking_dir)
    videos = parser.list_videos()
    results = {}

    for vid in videos:
        meta = parser.get_video_metadata(vid)
        fps = float(meta.get("fps", 25.0))
        tracks_file = meta.get("tracks_file")
        if not tracks_file or not Path(tracks_file).exists():
            continue

        tracks_df = pd.read_csv(tracks_file, comment="#", names=["frame", "track_id", "x", "y", "vx", "vy"])
        tracker = ByteTracker(fps=fps)
        analyzer = CrowdMotionAnalyzer()

        cap = cv2.VideoCapture(meta["video_path"])
        frame_i = 0
        speeds, entropies, counts = [], [], []
        sample_vis_saved = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            f_sub = tracks_df[tracks_df["frame"] == frame_i]
            boxes = np.array(
                [[max(0, r["x"] - 6), max(0, r["y"] - 6), min(w, r["x"] + 6), min(h, r["y"] + 12)] for _, r in f_sub.iterrows()],
                dtype=np.float32
            ) if len(f_sub) > 0 else np.zeros((0, 4), dtype=np.float32)
            confs = np.full((len(boxes),), 0.80, dtype=np.float32)

            active_tracks = tracker.update(boxes, confs)
            motion = analyzer.analyze_frame_motion(active_tracks, (h, w))
            counts.append(motion["active_tracks_count"])
            if frame_i > 0:
                speeds.append(motion["mean_speed_px_sec"])
                entropies.append(motion["flow_entropy"])

            if frame_i == 30 and not sample_vis_saved:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                vis = analyzer.render_tracking_frame(rgb, active_tracks, motion)
                out_dir = Path("results/heatmaps/tracking")
                out_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_dir / f"track_{Path(vid).stem}_f30.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                sample_vis_saved = True

            frame_i += 1
        cap.release()

        results[vid] = {
            "frames_processed": frame_i,
            "fps": fps,
            "mean_tracked_density_agents": float(round(np.mean(counts), 1)) if counts else 0.0,
            "mean_speed_px_sec": float(round(np.mean(speeds), 1)) if speeds else 0.0,
            "mean_flow_entropy": float(round(np.mean(entropies), 3)) if entropies else 0.0,
        }

    t_path = Path(tracking_json)
    if t_path.exists():
        with open(t_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = {}
    existing["ucf_tracking_video_telemetry"] = results
    with open(t_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    return results


def generate_risk_timeline_graph(
    features_csv: str = "results/predictions/extracted_crowd_features.csv",
    models_dir: str = "models/risk_prediction",
    output_png: str = "results/graphs/risk_timeline.png"
) -> str:
    """
    Scores extracted CV temporal features using trained Random Forest and XGBoost models
    and plots crowd count, movement speed, directional entropy, and risk score timeline.
    """
    df = pd.read_csv(features_csv)
    if df.empty:
        return ""

    p_models = Path(models_dir)
    scaler = joblib.load(p_models / "scaler.joblib")
    rf = joblib.load(p_models / "rf_risk_model.joblib")
    xgb = joblib.load(p_models / "xgb_risk_model.joblib")

    X_scaled = scaler.transform(df[FEATURE_COLUMNS].values)
    rf_preds = rf.predict(X_scaled)
    xgb_preds = xgb.predict(X_scaled)
    df["rf_risk_code"] = rf_preds
    df["rf_risk_label"] = [CLASS_NAMES[int(c)] for c in rf_preds]
    df["xgb_risk_code"] = xgb_preds
    df["xgb_risk_label"] = [CLASS_NAMES[int(c)] for c in xgb_preds]
    df.to_csv(features_csv, index=False)

    # Plot timelines for each sequence present
    sequences = df["sequence"].unique().tolist()
    fig, axes = plt.subplots(len(sequences), 1, figsize=(11, 4.2 * len(sequences)), squeeze=False)

    for idx, seq in enumerate(sequences):
        sub = df[df["sequence"] == seq].sort_values("timestamp_sec")
        ax1 = axes[idx, 0]
        ax1.plot(sub["timestamp_sec"], sub["crowd_count"], color="#2563EB", lw=2, label="Observed Count")
        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Crowd Count (persons)", color="#2563EB")
        ax1.tick_params(axis="y", labelcolor="#2563EB")
        ax1.grid(alpha=0.25)

        ax2 = ax1.twinx()
        ax2.plot(sub["timestamp_sec"], sub["flow_entropy"], color="#D97706", ls="--", lw=1.8, label="Directional Entropy")
        ax2.step(sub["timestamp_sec"], sub["rf_risk_code"], where="post", color="#DC2626", lw=2.0, label="RF Risk Tier (0=LOW..3=CRIT)")
        ax2.set_ylabel("Entropy / Risk Level", color="#DC2626")
        ax2.set_ylim(-0.2, 3.6)

        ax1.set_title(f"Temporal Crowd Telemetry & Risk Prediction Timeline — {seq}")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    out_p = Path(output_png)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_p, dpi=250)
    plt.close(fig)
    return str(out_p)


def run_all_benchmarks(recompute_heavy: bool = False):
    """Runs complete evaluation pipeline across all modules and updates summaries."""
    print("=== 1. Dataset Audit ===")
    inspector = DatasetInspector()
    print(inspector.generate_markdown_report())

    if recompute_heavy:
        from src.detection.yolo_detector import YOLOPersonDetector
        from src.detection.detection_evaluator import run_nwpu_detection_benchmark
        from src.counting.csrnet_model import CSRNetDensityEstimator
        from src.counting.counting_evaluator import run_counting_benchmark
        from src.tracking.tracking_evaluator import run_tracking_benchmark
        from src.feature_extraction.crowd_features import extract_features_from_datasets

        det = YOLOPersonDetector()
        run_nwpu_detection_benchmark(det)
        csr = CSRNetDensityEstimator()
        run_counting_benchmark(csr, det)
        run_tracking_benchmark()
        extract_features_from_datasets()

    print("=== 2. UCF Crowd Optical Flow & Segmentation Evaluation ===")
    flow_res = evaluate_ucf_crowd_flow()
    print("UCF Crowd Flow:", flow_res["sample_count"], "samples, mean flow:", flow_res["mean_flow_magnitude_px"], "px")

    print("=== 3. UCF Tracking Dense Video Telemetry ===")
    trk_vid_res = evaluate_ucf_tracking_video()
    print("UCF Tracking Video Telemetry:", trk_vid_res)

    print("=== 4. Risk Timeline Generation ===")
    tl_path = generate_risk_timeline_graph()
    print("Saved risk timeline graph to:", tl_path)

    print("=== 5. Experimental Evaluation Summary CSV ===")
    summary_df = build_experimental_summary_csv()
    print(summary_df[["Module", "Model", "Primary_Metric_1", "Primary_Metric_2", "Data_Type"]].to_string(index=False))


if __name__ == "__main__":
    run_all_benchmarks(recompute_heavy=False)
