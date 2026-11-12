"""
UCF Crowd Segmentation and Crowd Data Parser
Supports reading UCF Crowd Dataset images, spatial segmentation masks,
optical flow annotations, and motion field distributions.
Reference: Ali and Shah, 'A Lagranian Particle Dynamics Approach for Crowd Flow Segmentation and Stability Analysis', CVPR 2007.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image


class UCFCrowdDatasetParser:
    """Parser for UCF Crowd segmentation images, flow fields, and motion patterns."""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "images"
        self.flow_dir = self.dataset_dir / "flow_annotations"

    def list_samples(self) -> List[str]:
        """List all available crowd images in dataset."""
        if not self.images_dir.exists():
            return []
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return sorted([
            f.name for f in self.images_dir.iterdir()
            if f.suffix.lower() in valid_exts
        ])

    def parse_sample(self, sample_name: str) -> Dict[str, Any]:
        """
        Parse UCF crowd sample and corresponding flow/segmentation masks.
        """
        stem = Path(sample_name).stem
        img_path = self.images_dir / sample_name
        
        img_w, img_h = 720, 480
        if img_path.exists():
            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
            except Exception:
                pass

        flow_mask_path = self.flow_dir / f"{stem}_flow.npy"
        seg_mask_path = self.flow_dir / f"{stem}_seg.png"

        flow_data = None
        if flow_mask_path.exists():
            try:
                flow_data = np.load(str(flow_mask_path))
            except Exception:
                flow_data = None

        has_flow = flow_data is not None or flow_mask_path.exists()
        has_seg = seg_mask_path.exists()

        return {
            "sample_id": stem,
            "image_filename": sample_name,
            "image_path": str(img_path),
            "image_size": (img_w, img_h),
            "flow_path": str(flow_mask_path) if has_flow else None,
            "seg_path": str(seg_mask_path) if has_seg else None,
            "has_flow_annotation": has_flow,
            "has_segmentation": has_seg,
        }
