"""
UCF Tracking in High-Density Crowds Parser
Supports reading dense crowd videos, ground truth trajectory points,
frame extraction, and track statistics.
Reference: Rodriguez, Sivic, Laptev, Shah, 'Data-driven Crowd Analysis in Videos', ICCV 2011.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2


class UCFTrackingDatasetParser:
    """Parser for UCF High-Density Crowd Tracking videos and tracks."""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.videos_dir = self.dataset_dir / "videos"
        self.tracks_dir = self.dataset_dir / "tracks"

    def list_videos(self) -> List[str]:
        """List available video files in dataset."""
        if not self.videos_dir.exists():
            return []
        valid_exts = {".mp4", ".avi", ".mov", ".mkv"}
        return sorted([
            f.name for f in self.videos_dir.iterdir()
            if f.suffix.lower() in valid_exts
        ])

    def get_video_metadata(self, video_filename: str) -> Dict[str, Any]:
        """Extract duration, FPS, resolution, and frame count for a video."""
        video_path = self.videos_dir / video_filename
        if not video_path.exists():
            return {}

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"error": f"Could not open video {video_filename}"}

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = frame_count / fps if fps > 0 else 0.0
        cap.release()

        stem = Path(video_filename).stem
        track_file = self.tracks_dir / f"{stem}_tracks.txt"
        has_tracks = track_file.exists()

        return {
            "video_filename": video_filename,
            "video_path": str(video_path),
            "fps": fps,
            "frame_count": frame_count,
            "resolution": (width, height),
            "duration_seconds": duration_sec,
            "has_tracks": has_tracks,
            "tracks_file": str(track_file) if has_tracks else None,
        }

    def extract_frame(self, video_filename: str, frame_idx: int) -> Optional[np.ndarray]:
        """Extract a specific frame from video as RGB numpy array."""
        video_path = self.videos_dir / video_filename
        if not video_path.exists():
            return None

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if ret and frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
