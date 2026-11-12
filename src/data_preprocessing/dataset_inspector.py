"""
Dataset Inspector Module
Verifies the directory structure, modalities, sample counts, and annotation types
for all 5 benchmark crowd datasets:
  1. NWPU-Crowd (Images + Head points/boxes)
  2. UCF-QNRF (Images + Head points)
  3. UCF Crowd Segmentation / Crowd Data (Images/Sequences + Flow/Segments)
  4. MOTSynth (Video frames + MOTChallenge tracking bounding boxes)
  5. UCF Tracking in High-Density Crowds (Videos/Sequences + Trajectories)

Adheres to dataset rules:
- Identifies whether each dataset contains images or videos.
- Never treats static image datasets as video datasets.
- Inspects actual files without fabricating access.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd


class DatasetInspector:
    """Audits and inventories local dataset files and annotation completeness."""

    def __init__(self, base_datasets_dir: str = "datasets"):
        self.base_dir = Path(base_datasets_dir)
        self.dataset_configs = {
            "nwpu_crowd_real_val": {
                "name": "NWPU-Crowd (Official Real Validation Subset)",
                "type": "Image",
                "source_type": "Official Real Subset",
                "homepage": "https://gjy3035.github.io/NWPU-Crowd-Sample-Code/",
                "primary_task": "Crowd Counting, Density Estimation & Detection Localization",
                "annotation_type": "Official Head Points (x, y) & Head Boxes [xmin, ymin, xmax, ymax]",
                "sample_image_dir": self.base_dir / "nwpu_crowd" / "real_val_subset" / "images",
                "sample_ann_dir": self.base_dir / "nwpu_crowd" / "real_val_subset" / "annotations",
                "splits_dir": self.base_dir / "nwpu_crowd" / "real_val_subset" / "splits",
            },
            "nwpu_crowd": {
                "name": "NWPU-Crowd (Schema Sample)",
                "type": "Image",
                "source_type": "Schema-Compliant Synthetic Sample",
                "homepage": "https://gjy3035.github.io/NWPU-Crowd-Sample-Code/",
                "primary_task": "Crowd Counting & Density Estimation",
                "annotation_type": "Point Head Annotations (x, y) & Bounding Boxes",
                "sample_image_dir": self.base_dir / "nwpu_crowd" / "images",
                "sample_ann_dir": self.base_dir / "nwpu_crowd" / "annotations",
                "splits_dir": self.base_dir / "nwpu_crowd" / "splits",
            },
            "ucf_qnrf": {
                "name": "UCF-QNRF (Schema Sample)",
                "type": "Image",
                "source_type": "Schema-Compliant Synthetic Sample",
                "homepage": "https://www.crcv.ucf.edu/research/data-sets/ucf-qnrf/",
                "primary_task": "Extreme Density Crowd Counting",
                "annotation_type": "Dense Point Annotations (annPoints in MAT/JSON)",
                "sample_image_dir": self.base_dir / "ucf_qnrf" / "images",
                "sample_ann_dir": self.base_dir / "ucf_qnrf" / "annotations",
                "splits_dir": self.base_dir / "ucf_qnrf" / "splits",
            },
            "ucf_crowd": {
                "name": "UCF Crowd Segmentation (Schema Sample)",
                "type": "Image / Short Flow Sequence",
                "source_type": "Schema-Compliant Synthetic Sample",
                "homepage": "https://www.crcv.ucf.edu/data/crowd.php",
                "primary_task": "Crowd Density Distribution & Spatial Flow",
                "annotation_type": "Flow Fields & Spatial Segmentation Masks",
                "sample_image_dir": self.base_dir / "ucf_crowd" / "images",
                "sample_ann_dir": self.base_dir / "ucf_crowd" / "flow_annotations",
                "splits_dir": None,
            },
            "motsynth": {
                "name": "MOTSynth (Schema Sample)",
                "type": "Video Frames",
                "source_type": "Schema-Compliant Synthetic Sample",
                "homepage": "https://motchallenge.net/data/MOTSynth/",
                "primary_task": "Multi-Object Tracking (MOT) & Trajectory Analysis",
                "annotation_type": "MOTChallenge Format (frame, id, bbox, conf)",
                "sample_image_dir": self.base_dir / "motsynth" / "frames",
                "sample_ann_dir": self.base_dir / "motsynth" / "gt",
                "splits_dir": None,
            },
            "ucf_tracking": {
                "name": "UCF Tracking in High-Density Crowds (Schema Sample)",
                "type": "Video",
                "source_type": "Schema-Compliant Synthetic Sample",
                "homepage": "https://www.crcv.ucf.edu/data/crowd.php",
                "primary_task": "High-Density Video Crowd Tracking",
                "annotation_type": "Multi-Person Dense Tracking Trajectories",
                "sample_image_dir": self.base_dir / "ucf_tracking" / "videos",
                "sample_ann_dir": self.base_dir / "ucf_tracking" / "tracks",
                "splits_dir": None,
            },
        }

    def inspect_dataset(self, key: str) -> Dict[str, Any]:
        """Inspects single dataset directory."""
        if key not in self.dataset_configs:
            raise ValueError(f"Unknown dataset key: {key}")

        cfg = self.dataset_configs[key]
        img_dir: Path = cfg["sample_image_dir"]
        ann_dir: Path = cfg["sample_ann_dir"]

        img_count = 0
        img_samples = []
        video_count = 0
        video_samples = []
        ann_count = 0
        ann_samples = []

        if img_dir.exists():
            for p in img_dir.glob("**/*"):
                if p.is_file():
                    ext = p.suffix.lower()
                    if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                        img_count += 1
                        if len(img_samples) < 10:
                            img_samples.append(str(p.relative_to(self.base_dir)))
                    elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
                        video_count += 1
                        if len(video_samples) < 10:
                            video_samples.append(str(p.relative_to(self.base_dir)))

        if ann_dir and ann_dir.exists():
            for p in ann_dir.glob("**/*"):
                if p.is_file() and p.suffix.lower() in [".json", ".mat", ".txt", ".csv", ".xml", ".npy", ".png"]:
                    ann_count += 1
                    if len(ann_samples) < 10:
                        ann_samples.append(str(p.relative_to(self.base_dir)))

        is_available = (img_count > 0 or video_count > 0)
        has_annotations = ann_count > 0

        status_msg = "Available with annotations" if (is_available and has_annotations) else (
            "Available (unannotated samples)" if is_available else "Not populated locally (requires download or setup script)"
        )

        return {
            "key": key,
            "name": cfg["name"],
            "type": cfg["type"],
            "source_type": cfg.get("source_type", "Unknown"),
            "primary_task": cfg["primary_task"],
            "annotation_type": cfg["annotation_type"],
            "homepage": cfg["homepage"],
            "is_available": is_available,
            "has_annotations": has_annotations,
            "image_count": img_count,
            "video_count": video_count,
            "annotation_count": ann_count,
            "status": status_msg,
            "sample_files": img_samples or video_samples,
            "sample_annotations": ann_samples,
        }

    def inspect_all(self) -> Dict[str, Any]:
        """Inspect all 5 datasets and return summary report."""
        summary = {}
        for key in self.dataset_configs:
            summary[key] = self.inspect_dataset(key)
        return summary

    def generate_markdown_report(self) -> str:
        """Generate a formatted markdown table summarizing dataset inspection."""
        results = self.inspect_all()
        lines = [
            "# Dataset Audit and Inspection Report",
            "",
            "| Dataset Key | Official Name | Modality | Primary Task | Sample Count | Annotations | Status |",
            "|---|---|---|---|---|---|---|",
        ]
        for key, info in results.items():
            count_str = f"{info['image_count']} imgs" if info["type"] == "Image" else (
                f"{info['video_count']} vids" if info["video_count"] > 0 else f"{info['image_count']} frames"
            )
            ann_str = f"{info['annotation_count']} files" if info["has_annotations"] else "None"
            lines.append(
                f"| `{key}` | **{info['name']}** | {info['type']} | {info['primary_task']} | {count_str} | {ann_str} | {info['status']} |"
            )
        lines.append("")
        return "\n".join(lines)


if __name__ == "__main__":
    inspector = DatasetInspector()
    print(inspector.generate_markdown_report())
