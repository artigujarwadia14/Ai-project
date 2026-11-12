"""
NWPU-Crowd Dataset Parser
Supports reading NWPU-Crowd samples, JSON/MAT/TXT annotations,
head coordinate extraction, density classification, and validation.
Reference: Wang et al., 'NWPU-Crowd: A Large-Scale Benchmark for Crowd Counting and Localization', IEEE TPAMI 2020.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image

try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None


class NWPUDatasetParser:
    """Parser and Inspector for NWPU-Crowd dataset."""
    
    DENSITY_CATEGORIES = {
        "Negative": (0, 0),
        "Low": (1, 50),
        "Medium": (51, 500),
        "High": (501, 2000),
        "Extreme": (2001, float("inf")),
    }
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "images"
        self.annotations_dir = self.dataset_dir / "annotations"
        self.splits_dir = self.dataset_dir / "splits"
        
    def get_density_category(self, count: int) -> str:
        """Classify crowd density level by head count."""
        for cat, (low, high) in self.DENSITY_CATEGORIES.items():
            if low <= count <= high:
                return cat
        return "Unknown"

    def list_samples(self) -> List[str]:
        """List all available image IDs/filenames in dataset."""
        if not self.images_dir.exists():
            return []
        valid_exts = {".jpg", ".jpeg", ".png"}
        return sorted([
            f.name for f in self.images_dir.iterdir()
            if f.suffix.lower() in valid_exts
        ])

    def parse_annotation(self, sample_name: str) -> Dict[str, Any]:
        """
        Parse annotations corresponding to an image sample.
        Supports .json, .mat, or .txt format.
        """
        stem = Path(sample_name).stem
        img_path = self.images_dir / sample_name
        
        # Determine image dimensions
        img_w, img_h = 1024, 768
        if img_path.exists():
            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
            except Exception:
                pass

        points: List[Tuple[float, float]] = []
        bboxes: List[List[float]] = []
        bboxes_xyxy: List[List[float]] = []
        meta: Dict[str, Any] = {}
        is_official_real = False

        # 1. Try JSON annotation (standard in NWPU-Crowd release)
        json_path = self.annotations_dir / f"{stem}.json"
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                meta = data.get("meta", {})
                is_official_real = ("human_num" in data) or (meta.get("box_format") == "xyxy")

                if "points" in data and data["points"]:
                    points = [tuple(p) for p in data["points"]]

                if "boxes" in data and data["boxes"]:
                    bboxes = data["boxes"]
                    for b in bboxes:
                        if is_official_real:
                            # Official NWPU format: [xmin, ymin, xmax, ymax]
                            bboxes_xyxy.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
                        else:
                            # Synthetic schema format: [x, y, w, h]
                            bboxes_xyxy.append([float(b[0]), float(b[1]), float(b[0] + b[2]), float(b[1] + b[3])])

                    if not points:
                        for b_xyxy in bboxes_xyxy:
                            cx = (b_xyxy[0] + b_xyxy[2]) / 2.0
                            cy = (b_xyxy[1] + b_xyxy[3]) / 2.0
                            points.append((cx, cy))
            except Exception as e:
                meta["parse_error"] = str(e)

        # 2. Try MAT annotation
        mat_path = self.annotations_dir / f"{stem}.mat"
        if not points and mat_path.exists() and loadmat is not None:
            try:
                mat = loadmat(str(mat_path))
                if "annPoints" in mat:
                    pts = mat["annPoints"]
                    points = [tuple(p) for p in pts]
                elif "point" in mat:
                    pts = mat["point"]
                    points = [tuple(p) for p in pts]
            except Exception as e:
                meta["mat_parse_error"] = str(e)

        # 3. Try TXT format (x y per line)
        txt_path = self.annotations_dir / f"{stem}.txt"
        if not points and txt_path.exists():
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            points.append((float(parts[0]), float(parts[1])))
            except Exception as e:
                meta["txt_parse_error"] = str(e)

        points_arr = np.array(points, dtype=np.float32) if points else np.zeros((0, 2), dtype=np.float32)
        count = len(points_arr)
        density_category = self.get_density_category(count)

        return {
            "sample_id": stem,
            "image_filename": sample_name,
            "image_path": str(img_path),
            "image_size": (img_w, img_h),
            "ground_truth_count": count,
            "points": points_arr,
            "bboxes": bboxes,
            "bboxes_xyxy": bboxes_xyxy,
            "is_official_real": is_official_real,
            "density_category": density_category,
            "meta": meta,
            "has_ground_truth": len(points_arr) > 0 or json_path.exists() or mat_path.exists() or txt_path.exists()
        }
