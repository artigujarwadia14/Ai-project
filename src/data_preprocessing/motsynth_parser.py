"""
MOTSynth Dataset Parser
Supports reading MOTSynth synthetic video sequences and ground truth tracking annotations.
MOT format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
Reference: Fabbri et al., 'MOTSynth: How Can Synthetic Data Help Pedestrian Detection and Tracking?', ICCV 2021.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image


class MOTSynthDatasetParser:
    """Parser for MOTSynth sequences and MOTChallenge format tracking ground truth."""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.frames_dir = self.dataset_dir / "frames"
        self.gt_dir = self.dataset_dir / "gt"

    def list_sequences(self) -> List[str]:
        """List available sequence folders."""
        if not self.frames_dir.exists():
            return []
        return sorted([d.name for d in self.frames_dir.iterdir() if d.is_dir()])

    def list_sequence_frames(self, seq_name: str) -> List[str]:
        """List frame images in a sequence in chronological order."""
        seq_path = self.frames_dir / seq_name
        if not seq_path.exists():
            return []
        valid_exts = {".jpg", ".jpeg", ".png"}
        return sorted(
            [f.name for f in seq_path.iterdir() if f.suffix.lower() in valid_exts],
            key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)
        )

    def parse_ground_truth(self, seq_name: str) -> pd.DataFrame:
        """
        Parse gt.txt for the given sequence.
        Returns DataFrame with columns:
        [frame, track_id, bb_left, bb_top, bb_width, bb_height, conf, class_id, visibility]
        """
        gt_path = self.gt_dir / seq_name / "gt.txt"
        if not gt_path.exists():
            # Check alternative location
            gt_path = self.gt_dir / f"{seq_name}_gt.txt"
            
        columns = ["frame", "track_id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "class_id", "visibility"]
        if not gt_path.exists():
            return pd.DataFrame(columns=columns)

        try:
            df = pd.read_csv(gt_path, header=None, names=columns)
            # Filter for pedestrians (class 1 in MOTChallenge or class 2)
            if "class_id" in df.columns:
                df = df[df["class_id"].isin([1, 2])]
            return df
        except Exception:
            return pd.DataFrame(columns=columns)

    def get_frame_annotations(self, seq_name: str, frame_idx: int) -> Dict[str, Any]:
        """Get bounding boxes and track IDs for a specific frame index."""
        df = self.parse_ground_truth(seq_name)
        frame_df = df[df["frame"] == frame_idx]
        
        boxes = []
        track_ids = []
        for _, row in frame_df.iterrows():
            boxes.append([row["bb_left"], row["bb_top"], row["bb_width"], row["bb_height"]])
            track_ids.append(int(row["track_id"]))
            
        return {
            "sequence": seq_name,
            "frame_idx": frame_idx,
            "count": len(boxes),
            "bboxes": np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32),
            "track_ids": track_ids
        }
