"""
Module A: Person Detection powered by YOLOv8
Implements person detection using Ultralytics YOLOv8 with confidence thresholding,
bounding box extraction, latency/FPS benchmarking, and visualization.

Architectural details:
- Backbone: Modified CSPDarknet53 with C2f (Cross-Stage Partial with 2 Convolutions) modules
  combining high-level semantic features and rich low-level spatial features.
- Neck: Path Aggregation Network (PANet) and Feature Pyramid Network (FPN) for multiscale feature fusion.
- Head: Decoupled, anchor-free detection head predicting bounding box offsets (DFL) and class probabilities.
- Post-processing: Non-Maximum Suppression (NMS) with IoU threshold.

Limitation in Dense Crowds:
In high/extreme crowd densities (>200 people/view), individual bounding boxes severely overlap.
NMS suppresses true overlapping persons as duplicates, and small head pixels lack distinct bodily features,
causing false negatives. Density estimation models (e.g. CSRNet) address this limitation.
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class YOLOPersonDetector:
    """YOLOv8 Person Detection and Benchmarking Engine."""

    PERSON_CLASS_ID = 0  # COCO person class ID is 0

    def __init__(
        self,
        weights_path: str = "models/detection/yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu"
    ):
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None

        self._load_model()

    def _load_model(self):
        """Loads YOLOv8 model from local weights, falling back gracefully if necessary."""
        if YOLO is None:
            raise RuntimeError("ultralytics library is not installed.")

        # If local weights exist, load directly; otherwise load default yolov8n
        if self.weights_path.exists():
            self.model = YOLO(str(self.weights_path))
        else:
            self.model = YOLO("yolov8n.pt")
            # Save a local copy
            self.weights_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.weights_path))

    def detect(
        self,
        image_input: Union[str, Path, np.ndarray],
        imgsz: int = 640
    ) -> Dict[str, Any]:
        """
        Run person detection on a single image.

        Args:
            image_input: Filepath or BGR/RGB numpy array
            imgsz: Inference image size

        Returns:
            Dict containing:
                - count: Number of detected persons
                - bboxes: (N, 4) array in [x1, y1, x2, y2] format
                - confidences: (N,) array of confidence scores
                - inference_time_ms: Wall-clock forward pass latency in ms
                - fps: Theoretical inference speed
                - original_shape: (H, W)
        """
        if isinstance(image_input, (str, Path)):
            img_bgr = cv2.imread(str(image_input))
            if img_bgr is None:
                raise ValueError(f"Could not read image from {image_input}")
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input if len(image_input.shape) == 3 else cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        else:
            raise TypeError("Unsupported image input type.")

        h, w = img_bgr.shape[:2]

        t0 = time.perf_counter()
        results = self.model.predict(
            source=img_bgr,
            classes=[self.PERSON_CLASS_ID],
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=imgsz,
            device=self.device,
            verbose=False
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

        r = results[0]
        boxes_data = r.boxes.xyxy.cpu().numpy() if len(r.boxes) > 0 else np.zeros((0, 4), dtype=np.float32)
        confs_data = r.boxes.conf.cpu().numpy() if len(r.boxes) > 0 else np.zeros((0,), dtype=np.float32)

        return {
            "count": int(len(boxes_data)),
            "bboxes": boxes_data,
            "confidences": confs_data,
            "mean_confidence": float(np.mean(confs_data)) if len(confs_data) > 0 else 0.0,
            "inference_time_ms": float(round(latency_ms, 2)),
            "fps": float(round(fps, 1)),
            "original_shape": (h, w)
        }

    def draw_detections(
        self,
        image_input: Union[str, Path, np.ndarray],
        detection_result: Dict[str, Any],
        box_color: Tuple[int, int, int] = (0, 220, 110),
        text_color: Tuple[int, int, int] = (255, 255, 255),
        line_thickness: int = 2
    ) -> np.ndarray:
        """
        Draws bounding boxes and metadata overlay on image.
        Returns RGB image suitable for dashboard and visualization.
        """
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image_input.copy()
            if len(img_rgb.shape) == 2:
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)

        bboxes = detection_result["bboxes"]
        confs = detection_result["confidences"]

        # Draw each detected person
        for i, (box, conf) in enumerate(zip(bboxes, confs)):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), box_color, line_thickness)
            
            label = f"Person {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img_rgb, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, max(th + 6, y1)), box_color, -1)
            cv2.putText(img_rgb, label, (x1 + 2, max(th + 2, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)

        # Draw summary overlay banner
        count = detection_result["count"]
        fps = detection_result["fps"]
        latency = detection_result["inference_time_ms"]
        banner_text = f"YOLOv8 Detected: {count} | Latency: {latency:.1f}ms ({fps:.1f} FPS) | Conf: >={self.conf_threshold:.2f}"

        (bw, bh), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_rgb, (10, 10), (10 + bw + 16, 10 + bh + 16), (20, 20, 25), -1)
        cv2.rectangle(img_rgb, (10, 10), (10 + bw + 16, 10 + bh + 16), (0, 180, 255), 1)
        cv2.putText(img_rgb, banner_text, (18, 10 + bh + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 240, 255), 2)

        return img_rgb
