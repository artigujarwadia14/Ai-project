"""
Streamlit Web Dashboard for Crowd Rush Prediction & Early Warning System
A multi-page Computer Vision & Machine Learning decision-support application.

Pages:
  1. Home / Project Overview
  2. Dataset Explorer (Direct dataset browsing without file upload buttons)
  3. Person Detection (YOLOv8n)
  4. Crowd Density Analysis (CSRNet & Gaussian Heatmaps)
  5. Tracking and Movement Analysis (ByteTrack & Trajectories)
  6. Risk Prediction & Temporal Lookahead (RF vs XGBoost, 5/10/15 min forecasting)
  7. Reports & Results (Dynamic metrics from JSON/CSV)
"""

import json
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.data_preprocessing.dataset_inspector import DatasetInspector
from src.data_preprocessing.nwpu_parser import NWPUDatasetParser
from src.data_preprocessing.ucf_qnrf_parser import UCFQNRFDatasetParser
from src.data_preprocessing.ucf_crowd_parser import UCFCrowdDatasetParser
from src.data_preprocessing.motsynth_parser import MOTSynthDatasetParser
from src.data_preprocessing.ucf_tracking_parser import UCFTrackingDatasetParser
from src.data_preprocessing.density_map_generator import overlay_density_on_image
from src.detection.yolo_detector import YOLOPersonDetector
from src.counting.csrnet_model import CSRNetDensityEstimator
from src.tracking.byte_tracker import ByteTracker
from src.tracking.motion_analyzer import CrowdMotionAnalyzer
from src.risk_prediction.temporal_forecaster import TemporalRiskForecaster, CLASS_NAMES
from src.risk_prediction.explainability import RiskExplainer
from src.early_warning.warning_engine import EarlyWarningEngine

st.set_page_config(
    page_title="Crowd Rush Prediction & Early Warning System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title { font-size: 2.1rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.4rem; }
</style>
""", unsafe_allow_html=True)


def load_json_if_exists(path_str: str) -> dict:
    p = Path(path_str)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_resource
def get_yolo_detector(conf: float = 0.25, iou: float = 0.45):
    return YOLOPersonDetector(conf_threshold=conf, iou_threshold=iou)


@st.cache_resource
def get_csrnet_estimator():
    return CSRNetDensityEstimator()


@st.cache_resource
def get_forecaster():
    return TemporalRiskForecaster()


@st.cache_resource
def get_explainer():
    return RiskExplainer()


@st.cache_resource
def get_warning_engine():
    return EarlyWarningEngine()


def render_image_fit(img_obj, caption: str = ""):
    st.image(img_obj, caption=caption, width="stretch")


# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.title("Crowd Risk System")
st.sidebar.caption("Computer Vision & ML Early Warning")

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "Home / Project Overview",
        "Dataset Explorer",
        "Person Detection",
        "Crowd Density Analysis",
        "Tracking & Movement Analysis",
        "Risk Prediction",
        "Reports and Results"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Loaded Models:**
- Ultralytics YOLOv8n (`models/detection/yolov8n.pt`)
- CSRNet Dilated CNN (`models/crowd_counting/csrnet_weights.pth`)
- ByteTrack Multi-Object Tracker
- Random Forest & XGBoost Risk Classifiers
""")


# =========================================================================
# PAGE A: HOME / PROJECT OVERVIEW
# =========================================================================
if page == "Home / Project Overview":
    st.markdown('<div class="main-title">Crowd Rush Prediction & Early Warning System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">A Computer Vision and Machine Learning Decision-Support System</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("### Project Objective")
        st.write("""
        Overcrowding in high-density public spaces (pilgrimages, stadiums, transport corridors, festivals)
        can rapidly escalate into hazardous congestion and crush risks. This system analyzes crowd images
        and video sequences to:
        - **Detect & Localize People:** YOLOv8n bounding-box person detection in sparse and moderate scenes.
        - **Estimate Crowd Size & Density Maps:** CSRNet dilated CNN density regression where severe occlusion breaks bounding-box detection.
        - **Track & Analyze Movement:** ByteTrack two-stage association across video frames to compute trajectories, velocity, and directional turbulence entropy.
        - **Extract Spatio-Temporal Features:** Count, spatial density, growth rates, flow variation, capacity utilization, and congestion index.
        - **Estimate Risk & Future Trends:** Random Forest vs. XGBoost risk classification (`LOW`, `MODERATE`, `HIGH`, `CRITICAL`) and `+5/+10/+15` minute forecasting via Greenshields speed-decay physics.
        - **Generate Explainable Warnings:** Rule-based operational guidance highlighting exact driving indicators.
        """)

        st.markdown("### System Workflow Pipeline")
        st.code("""
Public Benchmarks (NWPU Real Subset + Schema Samples)
        │
        ▼
Dataset Inspection & Preprocessing (Images vs. Video Modalities)
        │
  ┌─────┴──────────────────────┬───────────────────────────┐
  ▼                            ▼                           ▼
YOLOv8n Detection       CSRNet Density Map        ByteTrack + Motion
(Boxes & Confidence)    (Continuous Heatmap)      (Trajectories & Entropy)
  └────────────────────────────┬───────────────────────────┘
                               ▼
              Crowd Feature Extraction (CSV Telemetry)
                               │
                               ▼
         Random Forest vs. XGBoost Risk Prediction
        (+5 / +10 / +15 Min Lookahead Forecasting)
                               │
                               ▼
         Explainable Early Warning & Recommendations
        """, language="text")

    with col2:
        st.markdown("### Dataset Coverage & Provenance")
        inspector = DatasetInspector()
        audit = inspector.inspect_all()
        rows = []
        for k, v in audit.items():
            cnt_txt = f"{v['image_count']} imgs" if v["type"] == "Image" else (
                f"{v['video_count']} vid" if v["video_count"] > 0 else f"{v['image_count']} frames"
            )
            rows.append({
                "Dataset": v["name"],
                "Modality": v["type"],
                "Samples": cnt_txt,
                "Source": v["source_type"]
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        st.markdown("### Scientific Scope & Disclaimers")
        st.warning("""
        - **Early-Warning Tool:** Estimates congestion risk indicators; does not guarantee stampede prediction.
        - **Real vs. Synthetic:** Real CV metrics are measured on the **10-image Official NWPU-Crowd Validation Subset**. Tabular risk labels come from a **documented synthetic safety scenario generator** (`simulated_risk_scenarios.csv`) because public crowd datasets do not carry stampede labels.
        """)


# =========================================================================
# PAGE B: DATASET EXPLORER (NO MANUAL UPLOAD)
# =========================================================================
elif page == "Dataset Explorer":
    st.markdown('<div class="main-title">Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Select a dataset to browse stored samples, ground-truth annotations, and live model outputs.</div>', unsafe_allow_html=True)

    inspector = DatasetInspector()
    audit_data = inspector.inspect_all()

    ds_key = st.selectbox(
        "Select Dataset:",
        options=list(audit_data.keys()),
        format_func=lambda k: f"{audit_data[k]['name']} [{audit_data[k]['type']}]"
    )
    info = audit_data[ds_key]
    st.markdown(
        f"**Source:** `{info['source_type']}` | **Modality:** `{info['type']}` | "
        f"**Task:** `{info['primary_task']}` | [Dataset Homepage]({info['homepage']})"
    )

    if ds_key in ["nwpu_crowd_real_val", "nwpu_crowd"]:
        folder = "datasets/nwpu_crowd/real_val_subset" if ds_key == "nwpu_crowd_real_val" else "datasets/nwpu_crowd"
        parser = NWPUDatasetParser(folder)
        samples = parser.list_samples()
        if samples:
            sel_sample = st.selectbox("Select Image Sample:", samples)
            ann = parser.parse_annotation(sel_sample)

            img_bgr = cv2.imread(ann["image_path"])
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            detector = get_yolo_detector()
            csrnet = get_csrnet_estimator()

            det_res = detector.detect(img_bgr)
            det_vis = detector.draw_detections(img_rgb, det_res)
            csr_res = csrnet.predict(img_rgb)
            csr_overlay = overlay_density_on_image(img_rgb, csr_res["density_map"])

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ground Truth Count", ann["ground_truth_count"])
            c2.metric("Density Category", ann["density_category"])
            c3.metric("YOLOv8 Detected", det_res["count"])
            c4.metric("CSRNet Count", f"{csr_res['estimated_count']:.1f}")

            t1, t2, t3 = st.tabs(["Original Image", "YOLOv8 Detections", "CSRNet Density Map"])
            with t1:
                render_image_fit(img_rgb, f"Original: {sel_sample} ({ann['image_size'][0]}x{ann['image_size'][1]})")
            with t2:
                render_image_fit(det_vis, f"YOLOv8n Detections: {det_res['count']} persons")
            with t3:
                render_image_fit(csr_overlay, f"CSRNet Density Heatmap: {csr_res['estimated_count']:.1f} estimated")

    elif ds_key == "ucf_qnrf":
        parser = UCFQNRFDatasetParser("datasets/ucf_qnrf")
        samples = parser.list_samples()
        if samples:
            sel_sample = st.selectbox("Select UCF-QNRF Image:", samples)
            ann = parser.parse_annotation(sel_sample)
            img_bgr = cv2.imread(ann["image_path"])
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            csrnet = get_csrnet_estimator()
            csr_res = csrnet.predict(img_rgb)
            csr_overlay = overlay_density_on_image(img_rgb, csr_res["density_map"])

            c1, c2 = st.columns(2)
            c1.metric("Ground Truth annPoints", ann["ground_truth_count"])
            c2.metric("CSRNet Predicted Count", f"{csr_res['estimated_count']:.1f}")
            colA, colB = st.columns(2)
            with colA:
                render_image_fit(img_rgb, f"Original Sample: {sel_sample}")
            with colB:
                render_image_fit(csr_overlay, "CSRNet Predicted Density Map")

    elif ds_key == "ucf_crowd":
        parser = UCFCrowdDatasetParser("datasets/ucf_crowd")
        samples = parser.list_samples()
        if samples:
            sel_sample = st.selectbox("Select UCF Crowd Flow Sample:", samples)
            ann = parser.parse_sample(sel_sample)
            img_rgb = np.array(Image.open(ann["image_path"]).convert("RGB"))
            vis_flow_path = Path("results/heatmaps/ucf_crowd_flow") / f"flow_{Path(sel_sample).stem}.jpg"

            colA, colB = st.columns(2)
            with colA:
                render_image_fit(img_rgb, f"Original Image: {sel_sample}")
            with colB:
                if vis_flow_path.exists():
                    render_image_fit(str(vis_flow_path), "Optical Flow Field & Spatial Crowd Mask")
                else:
                    render_image_fit(img_rgb, "Flow visualization available after running evaluation")

    elif ds_key == "motsynth":
        parser = MOTSynthDatasetParser("datasets/motsynth")
        seqs = parser.list_sequences()
        if seqs:
            sel_seq = st.selectbox("Select MOTSynth Sequence:", seqs)
            frames = parser.list_sequence_frames(sel_seq)
            frame_idx = st.slider("Select Sequence Frame:", 1, len(frames), 1)
            frame_path = Path("datasets/motsynth/frames") / sel_seq / frames[frame_idx - 1]
            img_rgb = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)

            gt_df = parser.parse_ground_truth(sel_seq)
            frame_gt = gt_df[gt_df["frame"] == frame_idx]
            gt_vis = img_rgb.copy()
            for _, row in frame_gt.iterrows():
                x1, y1 = int(row["bb_left"]), int(row["bb_top"])
                x2, y2 = int(x1 + row["bb_width"]), int(y1 + row["bb_height"])
                cv2.rectangle(gt_vis, (x1, y1), (x2, y2), (50, 220, 120), 2)
                cv2.putText(gt_vis, f"GT:{int(row['track_id'])}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 220, 120), 1)

            st.write(f"Sequence `{sel_seq}` | Frame `{frames[frame_idx - 1]}` | Annotated Pedestrians: `{len(frame_gt)}`")
            render_image_fit(gt_vis, f"MOTSynth Frame {frame_idx} with Ground Truth Boxes")

    elif ds_key == "ucf_tracking":
        parser = UCFTrackingDatasetParser("datasets/ucf_tracking")
        vids = parser.list_videos()
        if vids:
            sel_vid = st.selectbox("Select Video:", vids)
            meta = parser.get_video_metadata(sel_vid)
            st.write(
                f"Resolution: `{meta['resolution'][0]}x{meta['resolution'][1]}` | "
                f"FPS: `{meta['fps']}` | Frames: `{meta['frame_count']}` | "
                f"Duration: `{meta['duration_seconds']:.1f}s`"
            )
            f_idx = st.slider("Select Video Frame:", 0, max(0, meta["frame_count"] - 1), 25)
            frame = parser.extract_frame(sel_vid, frame_idx=f_idx)
            if frame is not None:
                render_image_fit(frame, f"{sel_vid} — Frame {f_idx}")


# =========================================================================
# PAGE C: PERSON DETECTION
# =========================================================================
elif page == "Person Detection":
    st.markdown('<div class="main-title">Module A: Person Detection (YOLOv8n)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Person localization across Low, Medium, High, and Extreme crowd densities.</div>', unsafe_allow_html=True)

    conf_slider = st.sidebar.slider("Confidence Threshold:", 0.10, 0.90, 0.25, 0.05)
    iou_slider = st.sidebar.slider("NMS IoU Threshold:", 0.20, 0.80, 0.45, 0.05)

    subset_choice = st.radio(
        "Dataset Source:",
        ["NWPU-Crowd Official Real Validation Subset (10 images)", "NWPU-Crowd Synthetic Schema Samples"],
        horizontal=True
    )
    folder = "datasets/nwpu_crowd/real_val_subset" if "Official" in subset_choice else "datasets/nwpu_crowd"
    parser = NWPUDatasetParser(folder)
    samples = parser.list_samples()
    sel_sample = st.selectbox("Select Sample Image:", samples)
    ann = parser.parse_annotation(sel_sample)

    img_bgr = cv2.imread(ann["image_path"])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    detector = get_yolo_detector(conf=conf_slider, iou=iou_slider)
    det_res = detector.detect(img_bgr)
    det_vis = detector.draw_detections(img_rgb, det_res)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Density Category", ann["density_category"])
    c2.metric("Ground Truth Count", ann["ground_truth_count"])
    c3.metric("YOLOv8 Detected", det_res["count"])
    c4.metric("Latency / Speed", f"{det_res['inference_time_ms']:.1f} ms", f"{det_res['fps']:.1f} FPS")

    render_image_fit(det_vis, f"YOLOv8n Detections (Conf >= {conf_slider:.2f}, Mean Conf: {det_res['mean_confidence']:.2f})")

    d_data = load_json_if_exists("results/metrics/detection_metrics.json")
    if d_data:
        st.markdown("### Evaluated YOLOv8n Metrics (Real NWPU-Crowd Validation Subset)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{d_data.get('overall_precision', 0):.3f}")
        m2.metric("Recall", f"{d_data.get('overall_recall', 0):.3f}")
        m3.metric("Localization AP", f"{d_data.get('localization_ap', 0):.3f}")
        m4.metric("Mean Latency", f"{d_data.get('mean_inference_time_ms', 0):.1f} ms")

        strat = d_data.get("density_stratified", {})
        if strat:
            rows = []
            for cat in ["Negative", "Low", "Medium", "High", "Extreme"]:
                if cat in strat:
                    s = strat[cat]
                    rows.append({
                        "Density Tier": cat,
                        "Images": s["samples"],
                        "GT Persons": s["gt_total"],
                        "YOLOv8 Detected": s["detected_total"],
                        "Precision": s["precision"],
                        "Recall": s["recall"],
                        "F1-Score": s["f1"]
                    })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        st.info(d_data.get("density_observation", ""))


# =========================================================================
# PAGE D: CROWD DENSITY ANALYSIS
# =========================================================================
elif page == "Crowd Density Analysis":
    st.markdown('<div class="main-title">Module B: Crowd Counting & Density Estimation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">CSRNet Dilated CNN continuous density estimation vs. YOLOv8 bounding-box counting.</div>', unsafe_allow_html=True)

    ds_choice = st.radio(
        "Dataset Source:",
        ["NWPU-Crowd Official Real Val (10 images)", "NWPU-Crowd Schema Samples", "UCF-QNRF Schema Samples"],
        horizontal=True
    )
    if "Real Val" in ds_choice:
        parser = NWPUDatasetParser("datasets/nwpu_crowd/real_val_subset")
        samples = parser.list_samples()
        sel_sample = st.selectbox("Select Image:", samples)
        ann = parser.parse_annotation(sel_sample)
        cat_label = ann["density_category"]
    elif "NWPU" in ds_choice:
        parser = NWPUDatasetParser("datasets/nwpu_crowd")
        samples = parser.list_samples()
        sel_sample = st.selectbox("Select Image:", samples)
        ann = parser.parse_annotation(sel_sample)
        cat_label = ann["density_category"]
    else:
        parser = UCFQNRFDatasetParser("datasets/ucf_qnrf")
        samples = parser.list_samples()
        sel_sample = st.selectbox("Select Image:", samples)
        ann = parser.parse_annotation(sel_sample)
        cat_label = "High" if ann["ground_truth_count"] <= 1000 else "Extreme"

    img_bgr = cv2.imread(ann["image_path"])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    csrnet = get_csrnet_estimator()
    csr_res = csrnet.predict(img_rgb)
    blended = overlay_density_on_image(img_rgb, csr_res["density_map"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Density Category", cat_label)
    c2.metric("Ground Truth Count", ann["ground_truth_count"])
    c3.metric("CSRNet Predicted Count", f"{csr_res['estimated_count']:.1f}")
    c4.metric("Absolute Error", f"{abs(csr_res['estimated_count'] - ann['ground_truth_count']):.1f}")

    colA, colB = st.columns(2)
    with colA:
        render_image_fit(img_rgb, f"Original Scene: {sel_sample}")
    with colB:
        render_image_fit(blended, f"CSRNet Density Heatmap Overlay (Peak: {csr_res['max_density']:.4f})")

    c_data = load_json_if_exists("results/metrics/counting_metrics.json")
    if c_data:
        st.markdown("### Benchmark Comparison: CSRNet vs. YOLOv8n (NWPU-Crowd Official Real Val)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CSRNet Overall MAE", f"{c_data['csrnet_metrics']['mae']:.2f}")
        m2.metric("CSRNet Overall RMSE", f"{c_data['csrnet_metrics']['rmse']:.2f}")
        m3.metric("YOLOv8n Overall MAE", f"{c_data['yolo_metrics']['mae']:.2f}")
        m4.metric("YOLOv8n Overall RMSE", f"{c_data['yolo_metrics']['rmse']:.2f}")

        strat = c_data.get("density_stratified_comparison", {})
        if strat:
            rows = []
            for tier in ["Negative", "Low", "Medium", "High", "Extreme"]:
                if tier in strat:
                    vals = strat[tier]
                    rows.append({
                        "Density Tier": tier,
                        "Samples": vals["sample_count"],
                        "CSRNet MAE": vals["csrnet_mae"],
                        "CSRNet RMSE": vals["csrnet_rmse"],
                        "YOLOv8 MAE": vals["yolo_mae"],
                        "YOLOv8 RMSE": vals["yolo_rmse"],
                        "Lower Error": "CSRNet (Density)" if vals["csrnet_mae"] < vals["yolo_mae"] else "YOLOv8n (Boxes)"
                    })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        mae_chart = Path("results/graphs/counting_mae_comparison.png")
        if mae_chart.exists():
            render_image_fit(str(mae_chart), "Density-Stratified MAE Crossover: YOLOv8n vs. CSRNet")

        st.info(c_data.get("academic_insight", ""))


# =========================================================================
# PAGE E: TRACKING AND MOVEMENT ANALYSIS
# =========================================================================
elif page == "Tracking & Movement Analysis":
    st.markdown('<div class="main-title">Module C: Person Tracking & Movement Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">ByteTrack two-stage association, trajectory paths, velocity vectors, and directional turbulence.</div>', unsafe_allow_html=True)

    video_source = st.radio("Video Dataset:", ["MOTSynth Frame Sequence", "UCF High-Density Tracking Video"], horizontal=True)
    detector = get_yolo_detector()
    analyzer = CrowdMotionAnalyzer()

    if "MOTSynth" in video_source:
        parser = MOTSynthDatasetParser("datasets/motsynth")
        seqs = parser.list_sequences()
        sel_seq = st.selectbox("Select MOTSynth Sequence:", seqs)
        frames = parser.list_sequence_frames(sel_seq)
        frame_idx = st.slider("Frame Timeline:", 1, len(frames), min(15, len(frames)))
        gt_df = parser.parse_ground_truth(sel_seq)

        tracker = ByteTracker(fps=30.0)
        target_img_rgb, target_tracks, target_stats = None, [], {}

        for i in range(1, frame_idx + 1):
            f_p = Path("datasets/motsynth/frames") / sel_seq / frames[i - 1]
            bgr = cv2.imread(str(f_p))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            det = detector.detect(bgr)
            bxs, cfs = det["bboxes"], det["confidences"]
            if len(bxs) == 0 and len(gt_df) > 0:
                c_gt = gt_df[gt_df["frame"] == i]
                bxs = np.array([[r["bb_left"], r["bb_top"], r["bb_left"] + r["bb_width"], r["bb_top"] + r["bb_height"]] for _, r in c_gt.iterrows()], dtype=np.float32)
                cfs = np.full((len(bxs),), 0.85, dtype=np.float32)

            active_t = tracker.update(bxs, cfs)
            m_stat = analyzer.analyze_frame_motion(active_t, rgb.shape[:2])
            if i == frame_idx:
                target_img_rgb, target_tracks, target_stats = rgb, active_t, m_stat

        rendered = analyzer.render_tracking_frame(target_img_rgb, target_tracks, target_stats)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Active Tracked People", target_stats.get("active_tracks_count", 0))
        c2.metric("Mean Speed", f"{target_stats.get('mean_speed_px_sec', 0.0):.1f} px/s")
        c3.metric("Dominant Direction", f"{target_stats.get('dominant_direction_cardinal', 'None')} ({target_stats.get('dominant_direction_deg', 0):.0f}°)")
        c4.metric("Directional Entropy", f"{target_stats.get('flow_entropy', 0.0):.2f}")
        render_image_fit(rendered, f"ByteTrack Output — {sel_seq} Frame {frame_idx}")

    else:
        parser = UCFTrackingDatasetParser("datasets/ucf_tracking")
        vids = parser.list_videos()
        sel_vid = st.selectbox("Select UCF Tracking Video:", vids)
        meta = parser.get_video_metadata(sel_vid)
        fps = float(meta.get("fps", 25.0))
        max_f = max(1, int(meta.get("frame_count", 125)) - 1)
        target_f = st.slider("Select Frame Index:", 1, max_f, min(30, max_f))

        tracks_file = meta.get("tracks_file")
        tracks_df = pd.read_csv(tracks_file, comment="#", names=["frame", "track_id", "x", "y", "vx", "vy"]) if tracks_file and Path(tracks_file).exists() else None
        tracker = ByteTracker(fps=fps)
        cap = cv2.VideoCapture(meta["video_path"])
        target_rgb, target_tracks, target_stats = None, [], {}
        for i in range(target_f + 1):
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            det = detector.detect(frame)
            bxs, cfs = det["bboxes"], det["confidences"]
            if len(bxs) == 0 and tracks_df is not None:
                f_sub = tracks_df[tracks_df["frame"] == i]
                bxs = np.array([[max(0, r["x"] - 6), max(0, r["y"] - 6), min(w, r["x"] + 6), min(h, r["y"] + 12)] for _, r in f_sub.iterrows()], dtype=np.float32)
                cfs = np.full((len(bxs),), 0.80, dtype=np.float32)
            active_t = tracker.update(bxs, cfs)
            m_stat = analyzer.analyze_frame_motion(active_t, (h, w))
            if i == target_f:
                target_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                target_tracks, target_stats = active_t, m_stat
        cap.release()

        if target_rgb is not None:
            rendered = analyzer.render_tracking_frame(target_rgb, target_tracks, target_stats)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Active Tracked Agents", target_stats.get("active_tracks_count", 0))
            c2.metric("Mean Speed", f"{target_stats.get('mean_speed_px_sec', 0.0):.1f} px/s")
            c3.metric("Dominant Flow", target_stats.get("dominant_direction_cardinal", "None"))
            c4.metric("Directional Entropy", f"{target_stats.get('flow_entropy', 0.0):.2f}")
            render_image_fit(rendered, f"UCF High-Density Tracking — {sel_vid} Frame {target_f}")

    t_data = load_json_if_exists("results/metrics/tracking_metrics.json")
    if t_data:
        st.markdown("### Evaluated Tracking Metrics (MOTSynth Benchmark)")
        ca, cb, cc = st.columns(3)
        ca.metric("MOTA", f"{t_data.get('aggregate_mota', 0)*100:.2f}%")
        cb.metric("IDF1", f"{t_data.get('aggregate_idf1', 0)*100:.2f}%")
        cc.metric("ID Switches", t_data.get("total_id_switches", 0))
        st.caption(t_data.get("tracking_observation", ""))


# =========================================================================
# PAGE F: RISK PREDICTION
# =========================================================================
elif page == "Risk Prediction":
    st.markdown('<div class="main-title">Crowd-Risk Prediction & Early Warning</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Random Forest vs. XGBoost classification, 5/10/15-min lookahead forecasting, and explainable recommendations.</div>', unsafe_allow_html=True)

    mode = st.radio(
        "Feature Input Mode:",
        ["Interactive Scenario Simulator", "Replay Extracted Video Features (CSV)"],
        horizontal=True
    )

    if mode == "Replay Extracted Video Features (CSV)":
        feat_csv = Path("results/predictions/extracted_crowd_features.csv")
        if feat_csv.exists():
            df_feat = pd.read_csv(feat_csv)
            sel_seq = st.selectbox("Select Extracted Sequence:", df_feat["sequence"].unique().tolist())
            seq_df = df_feat[df_feat["sequence"] == sel_seq].reset_index(drop=True)
            row_idx = st.slider("Select Sequence Time Step:", 0, len(seq_df) - 1, 0)
            row = seq_df.iloc[row_idx]
            sim_density = float(row["crowd_density_p_m2"])
            sim_speed = float(row["avg_movement_speed_ms"])
            sim_capacity = float(row["capacity_utilization"])
            sim_entropy = float(row["flow_entropy"])
            sim_inflow = float(row["crowd_growth_rate_per_min"])
            sim_features = {
                "crowd_count": float(row["crowd_count"]),
                "crowd_density_p_m2": sim_density,
                "avg_movement_speed_ms": sim_speed,
                "capacity_utilization": sim_capacity,
                "congestion_index": float(row["congestion_index"]),
                "flow_entropy": sim_entropy,
                "flow_variation": float(row["flow_variation"]),
                "crowd_growth_rate_per_min": sim_inflow,
                "speed_variance": float(row["speed_variance"]),
            }
            st.caption(f"Loaded real extracted CV telemetry: `timestamp={row['timestamp_sec']}s`, `count={row['crowd_count']}`, `speed={sim_speed} m/s`, `entropy={sim_entropy}`")
        else:
            st.error("Extracted feature CSV missing. Run `python main.py --evaluate --full`.")
            st.stop()
    else:
        st.sidebar.markdown("### Scenario Parameters")
        sim_density = st.sidebar.slider("Crowd Density (persons/m²):", 0.1, 7.0, 2.5, 0.1)
        sim_speed = st.sidebar.slider("Walking Speed (m/s):", 0.05, 1.8, 0.65, 0.05)
        sim_capacity = st.sidebar.slider("Capacity Utilization:", 0.1, 1.5, 0.85, 0.05)
        sim_entropy = st.sidebar.slider("Directional Entropy:", 0.2, 3.2, 1.6, 0.1)
        sim_inflow = st.sidebar.slider("Net Inflow Rate (persons/min):", -20.0, 60.0, 15.0, 5.0)
        sim_features = {
            "crowd_count": float(round(sim_density * 375.0, 1)),
            "crowd_density_p_m2": sim_density,
            "avg_movement_speed_ms": sim_speed,
            "capacity_utilization": sim_capacity,
            "congestion_index": float(round(sim_density / max(0.1, sim_speed), 2)),
            "flow_entropy": sim_entropy,
            "flow_variation": float(round((sim_entropy / 3.0) * 0.7, 3)),
            "crowd_growth_rate_per_min": sim_inflow,
            "speed_variance": 0.15
        }

    forecaster = get_forecaster()
    explainer = get_explainer()
    warning_engine = get_warning_engine()

    model_choice = st.radio("Active Classifier:", ["Random Forest", "XGBoost"], horizontal=True)
    m_code = "rf" if model_choice == "Random Forest" else "xgb"

    forecast_res = forecaster.forecast_future_risk(
        current_features=sim_features,
        inflow_rate_per_min=sim_inflow,
        model_name=m_code
    )

    scaled_curr = forecaster.scaler.transform([[
        sim_features["crowd_density_p_m2"],
        sim_features["avg_movement_speed_ms"],
        sim_features["capacity_utilization"],
        sim_features["congestion_index"],
        sim_features["flow_entropy"],
        sim_features["flow_variation"],
        sim_features["crowd_growth_rate_per_min"],
        sim_features["speed_variance"]
    ]])
    clf = forecaster.rf_model if m_code == "rf" else forecaster.xgb_model
    curr_risk_code = int(clf.predict(scaled_curr)[0])
    curr_risk_label = CLASS_NAMES[curr_risk_code]
    curr_probs = clf.predict_proba(scaled_curr)[0]

    rf_label = CLASS_NAMES[int(forecaster.rf_model.predict(scaled_curr)[0])]
    xgb_label = CLASS_NAMES[int(forecaster.xgb_model.predict(scaled_curr)[0])]
    st.write(f"**Model Agreement:** Random Forest -> `{rf_label}` | XGBoost -> `{xgb_label}`")

    warning_info = warning_engine.evaluate_warnings(curr_risk_label, sim_features)

    st.markdown(f"""
    <div style="background-color: {warning_info['color_hex']}22; border: 2px solid {warning_info['color_hex']}; padding: 16px; border-radius: 10px; margin-bottom: 18px;">
        <h3 style="color: {warning_info['color_hex']}; margin: 0 0 6px 0;">{warning_info['badge']} — {warning_info['status_title']}</h3>
        <p style="margin: 0;">{warning_info['description']}</p>
        <p style="margin: 6px 0 0 0; font-weight: 600;">Operational Posture: {warning_info['operational_posture']}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Class Probability Distribution")
        prob_df = pd.DataFrame({
            "Risk Category": CLASS_NAMES,
            "Probability (%)": [p * 100.0 for p in curr_probs]
        })
        st.bar_chart(prob_df.set_index("Risk Category"), color="#2563EB")

    with col2:
        st.markdown("### 5-, 10-, and 15-Minute Lookahead Forecast")
        f_data = forecast_res["forecasts"]
        f_rows = []
        for dt_key in ["5_min", "10_min", "15_min"]:
            f_item = f_data[dt_key]
            f_rows.append({
                "Horizon": f"+{f_item['horizon_minutes']} Mins",
                "Projected Count": f_item["projected_crowd_count"],
                "Projected Density": f"{f_item['projected_density_p_m2']:.2f} p/m²",
                "Projected Speed": f"{f_item['projected_speed_ms']:.2f} m/s",
                "Forecasted Risk": f_item["predicted_risk_level"]
            })
        st.dataframe(pd.DataFrame(f_rows), width="stretch", hide_index=True)

    st.markdown("### Feature Explainability & Top Driving Factors")
    feat_names = ["crowd_density_p_m2", "avg_movement_speed_ms", "capacity_utilization", "congestion_index", "flow_entropy", "flow_variation", "crowd_growth_rate_per_min", "speed_variance"]
    feat_imps = {f: float(imp) for f, imp in zip(feat_names, clf.feature_importances_)}
    explanation = explainer.explain_instance(sim_features, curr_risk_label, feat_imps)
    st.info(f"**Explanation:** {explanation['explanation_summary']}")
    for tf in explanation["top_driving_factors"]:
        st.write(f"- **{tf['name']}:** {tf['detail']} (Weighted Impact: `{tf['impact_score']:.1f}`)")

    st.markdown("### Actionable Preventive Recommendations")
    for rec in warning_info["actionable_recommendations"]:
        st.warning(rec)
    st.caption(warning_info["decision_support_disclaimer"])


# =========================================================================
# PAGE G: REPORTS AND RESULTS
# =========================================================================
elif page == "Reports and Results":
    st.markdown('<div class="main-title">Reports and Experimental Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Measured evaluation metrics loaded dynamically from pipeline outputs, graphs, and downloadable CSVs.</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Experimental Summary & Graphs", "Downloadable CSV Datasets"])

    with tab1:
        summary_csv = Path("results/predictions/experimental_eval_summary.csv")
        if summary_csv.exists():
            st.markdown("### Consolidated Model Evaluation Table (Actual Run Outputs)")
            st.dataframe(pd.read_csv(summary_csv), width="stretch", hide_index=True)

        risk_data = load_json_if_exists("results/metrics/risk_model_comparison.json")
        if risk_data:
            st.markdown("### Random Forest vs. XGBoost Class-Wise Metrics (Synthetic Risk Scenarios)")
            c_rf, c_xgb = st.columns(2)
            with c_rf:
                st.write(f"**Random Forest** — Accuracy: `{risk_data['random_forest']['accuracy']*100:.2f}%` | Latency: `{risk_data['random_forest']['inference_latency_ms']} ms`")
                st.dataframe(pd.DataFrame(risk_data["random_forest"]["class_wise_metrics"]).T, width="stretch")
            with c_xgb:
                st.write(f"**XGBoost** — Accuracy: `{risk_data['xgboost']['accuracy']*100:.2f}%` | Latency: `{risk_data['xgboost']['inference_latency_ms']} ms`")
                st.dataframe(pd.DataFrame(risk_data["xgboost"]["class_wise_metrics"]).T, width="stretch")

        st.markdown("### Computed Evaluation Charts")
        g1, g2 = st.columns(2)
        if Path("results/graphs/confusion_matrices.png").exists():
            with g1:
                render_image_fit("results/graphs/confusion_matrices.png", "RF vs. XGBoost Confusion Matrices")
        if Path("results/graphs/feature_importance.png").exists():
            with g2:
                render_image_fit("results/graphs/feature_importance.png", "Feature Importance Comparison")

        g3, g4 = st.columns(2)
        if Path("results/graphs/roc_curves.png").exists():
            with g3:
                render_image_fit("results/graphs/roc_curves.png", "Multi-Class ROC-AUC Curves")
        if Path("results/graphs/counting_mae_comparison.png").exists():
            with g4:
                render_image_fit("results/graphs/counting_mae_comparison.png", "CSRNet vs. YOLOv8 MAE Across Densities")

        if Path("results/graphs/risk_timeline.png").exists():
            render_image_fit("results/graphs/risk_timeline.png", "Extracted Video Telemetry & Predicted Risk Timeline")

    with tab2:
        st.markdown("### Download Generated CSV Results")
        for label, path_str in [
            ("Experimental Evaluation Summary CSV", "results/predictions/experimental_eval_summary.csv"),
            ("Extracted Crowd Features CSV", "results/predictions/extracted_crowd_features.csv"),
            ("Synthetic Risk Scenarios CSV (2,000 rows)", "results/predictions/simulated_risk_scenarios.csv"),
        ]:
            p = Path(path_str)
            if p.exists():
                df_dl = pd.read_csv(p)
                st.download_button(
                    label=f"Download {label} ({len(df_dl)} rows)",
                    data=df_dl.to_csv(index=False),
                    file_name=p.name,
                    mime="text/csv"
                )
