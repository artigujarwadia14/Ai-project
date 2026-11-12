"""
UCF-QNRF Dataset Parser
Supports reading UCF-QNRF images, MATLAB .mat annotations (annPoints),
JSON/TXT fallback annotations, and density statistics.
Reference: Idrees et al., 'Composition Loss for Counting, Density Map Estimation and Localization in Dense Crowds', ECCV 2018.
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


class UCFQNRFDatasetParser:
    """Parser and Inspector for UCF-QNRF dataset."""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "images"
        self.annotations_dir = self.dataset_dir / "annotations"
        self.splits_dir = self.dataset_dir / "splits"

    def list_samples(self) -> List[str]:
        """List all available image files in the dataset."""
        if not self.images_dir.exists():
            return []
        valid_exts = {".jpg", ".jpeg", ".png"}
        return sorted([
            f.name for f in self.images_dir.iterdir()
            if f.suffix.lower() in valid_exts
        ])

    def parse_annotation(self, sample_name: str) -> Dict[str, Any]:
        """
        Parse UCF-QNRF ground truth for a given image.
        In standard UCF-QNRF, the annotation is img_xxxx_ann.mat containing annPoints.
        Also supports img_xxxx.json and img_xxxx.txt for portability across platforms.
        """
        stem = Path(sample_name).stem
        img_path = self.images_dir / sample_name
        
        img_w, img_h = 1024, 768
        if img_path.exists():
            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
            except Exception:
                pass

        points: List[Tuple[float, float]] = []
        meta: Dict[str, Any] = {}

        # 1. Standard UCF-QNRF MAT format (img_xxxx_ann.mat or img_xxxx.mat)
        mat_candidates = [
            self.annotations_dir / f"{stem}_ann.mat",
            self.annotations_dir / f"{stem}.mat"
        ]
        for mat_path in mat_candidates:
            if mat_path.exists() and loadmat is not None:
                try:
                    mat = loadmat(str(mat_path))
                    if "annPoints" in mat:
                        pts = mat["annPoints"]
                        points = [tuple(p) for p in pts]
                        break
                    elif "point" in mat:
                        pts = mat["point"]
                        points = [tuple(p) for p in pts]
                        break
                except Exception as e:
                    meta["mat_error"] = str(e)

        # 2. JSON format fallback (img_xxxx.json or img_xxxx_ann.json)
        if not points:
            for json_path in [self.annotations_dir / f"{stem}.json", self.annotations_dir / f"{stem}_ann.json"]:
                if json_path.exists():
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if "points" in data:
                                points = [tuple(p) for p in data["points"]]
                            elif "annPoints" in data:
                                points = [tuple(p) for p in data["annPoints"]]
                            meta.update(data.get("meta", {}))
                            break
                    except Exception as e:
                        meta["json_error"] = str(e)

        # 3. Plain text format fallback (img_xxxx.txt)
        if not points:
            txt_path = self.annotations_dir / f"{stem}.txt"
            if txt_path.exists():
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                points.append((float(parts[0]), float(parts[1])))
                except Exception as e:
                    meta["txt_error"] = str(e)

        points_arr = np.array(points, dtype=np.float32) if points else np.zeros((0, 2), dtype=np.float32)
        count = len(points_arr)

        return {
            "sample_id": stem,
            "image_filename": sample_name,
            "image_path": str(img_path),
            "image_size": (img_w, img_h),
            "ground_truth_count": count,
            "points": points_arr,
            "meta": meta,
            "has_ground_truth": len(points_arr) > 0 or any(p.exists() for p in mat_candidates)
        }
