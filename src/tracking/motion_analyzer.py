"""
Module C: Crowd Motion Analysis & Spatio-Temporal Dynamics
Extracts trajectory vectors, velocity, speed, flow coherence, and directional entropy
from ByteTrack multi-object tracking results.

Mathematical formulations:
  - Velocity: v = (dx/dt, dy/dt) in pixels/sec
  - Speed: s = sqrt(vx^2 + vy^2)
  - Dominant Angle: theta = arctan2(mean(vy), mean(vx))
  - Directional Flow Entropy: H(theta) = -sum(p_i * log2(p_i)) over 8 directional bins.
    Low entropy (< 1.5): Unidirectional laminar crowd flow.
    High entropy (> 2.4): Turbulent multi-directional movement or panic disorder.
  - Congestion Index: C = Density / max(0.5, Mean Speed) (Fluid dynamics analogy).
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2
from src.tracking.byte_tracker import Tracklet


class CrowdMotionAnalyzer:
    """Analyzes spatio-temporal trajectories and extracts kinetic crowd features."""

    def __init__(self, num_direction_bins: int = 8):
        self.num_bins = num_direction_bins
        self.bin_edges = np.linspace(0, 360, num_direction_bins + 1)

    def compute_directional_entropy(self, angles_deg: List[float]) -> float:
        """
        Calculates Shannon entropy of motion directions across angular bins.
        """
        if len(angles_deg) == 0:
            return 0.0

        hist, _ = np.histogram(angles_deg, bins=self.bin_edges)
        total = np.sum(hist)
        if total == 0:
            return 0.0

        probs = hist / float(total)
        # Avoid log(0)
        non_zero = probs[probs > 0]
        entropy = -float(np.sum(non_zero * np.log2(non_zero)))
        return float(round(entropy, 3))

    def analyze_frame_motion(
        self,
        active_tracks: List[Tracklet],
        frame_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Extracts aggregate crowd movement dynamics for the current frame.
        """
        h, w = frame_shape
        num_tracked = len(active_tracks)

        if num_tracked == 0:
            return {
                "active_tracks_count": 0,
                "mean_speed_px_sec": 0.0,
                "speed_std": 0.0,
                "dominant_direction_deg": 0.0,
                "dominant_direction_cardinal": "None",
                "flow_entropy": 0.0,
                "flow_variation": 0.0,
                "congestion_ratio": 0.0,
                "track_summaries": []
            }

        speeds = [t.speed for t in active_tracks]
        directions = [t.direction_deg for t in active_tracks]
        vxs = [t.vx for t in active_tracks]
        vys = [t.vy for t in active_tracks]

        mean_speed = float(np.mean(speeds))
        std_speed = float(np.std(speeds))

        mean_vx = float(np.mean(vxs))
        mean_vy = float(np.mean(vys))
        dominant_angle = float(np.degrees(np.arctan2(mean_vy, mean_vx)) % 360)
        cardinal_dir = self.degrees_to_cardinal(dominant_angle)

        entropy = self.compute_directional_entropy(directions)
        # Flow variation: coefficient of variation of speed + normalized entropy
        flow_variation = float(round((std_speed / (mean_speed + 1e-5)) * 0.5 + (entropy / 3.0) * 0.5, 3))

        # Congestion ratio: tracked count relative to mean velocity
        congestion_ratio = float(round(num_tracked / max(1.0, mean_speed / 5.0), 2))

        track_summaries = []
        for t in active_tracks:
            track_summaries.append({
                "track_id": t.track_id,
                "bbox": [float(x) for x in t.bbox],
                "speed": float(round(t.speed, 1)),
                "direction_deg": float(round(t.direction_deg, 1)),
                "history_len": len(t.history)
            })

        return {
            "active_tracks_count": num_tracked,
            "mean_speed_px_sec": float(round(mean_speed, 1)),
            "speed_std": float(round(std_speed, 1)),
            "dominant_direction_deg": float(round(dominant_angle, 1)),
            "dominant_direction_cardinal": cardinal_dir,
            "flow_entropy": entropy,
            "flow_variation": flow_variation,
            "congestion_ratio": congestion_ratio,
            "track_summaries": track_summaries
        }

    @staticmethod
    def degrees_to_cardinal(deg: float) -> str:
        """Converts angle in degrees to compass heading."""
        dirs = ["E", "ENE", "NE", "NNE", "N", "NNW", "NW", "WNW", "W", "WSW", "SW", "SSW", "S", "SSE", "SE", "ESE"]
        idx = int(round(deg / 22.5)) % 16
        return dirs[idx]

    def render_tracking_frame(
        self,
        frame_rgb: np.ndarray,
        active_tracks: List[Tracklet],
        motion_stats: Dict[str, Any]
    ) -> np.ndarray:
        """
        Renders bounding boxes, unique track IDs, historical trajectory trails,
        velocity vectors, and real-time kinetic telemetry HUD on video frame.
        """
        out = frame_rgb.copy()

        # Draw trajectory trails
        for t in active_tracks:
            # Color based on track ID
            np.random.seed(t.track_id * 13)
            color = (
                int(np.random.randint(60, 245)),
                int(np.random.randint(60, 245)),
                int(np.random.randint(60, 245))
            )

            # Draw history polyline
            if len(t.history) > 1:
                pts = np.array([(int(pt[0]), int(pt[1])) for pt in t.history], dtype=np.int32)
                cv2.polylines(out, [pts], isClosed=False, color=color, thickness=2)

            # Bounding box
            x1, y1, x2, y2 = map(int, t.bbox)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Draw ID label
            id_label = f"ID:{t.track_id} ({t.speed:.0f}px/s)"
            (tw, th), _ = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(out, (x1, max(0, y1 - th - 4)), (x1 + tw + 2, y1), color, -1)
            cv2.putText(out, id_label, (x1 + 1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Velocity vector arrow
            if t.speed > 5.0:
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                arrow_scale = 0.4
                end_x = int(cx + t.vx * arrow_scale)
                end_y = int(cy + t.vy * arrow_scale)
                cv2.arrowedLine(out, (cx, cy), (end_x, end_y), (0, 255, 255), 2, tipLength=0.3)

        # Draw Telemetry HUD Banner
        count = motion_stats.get("active_tracks_count", 0)
        speed = motion_stats.get("mean_speed_px_sec", 0.0)
        entropy = motion_stats.get("flow_entropy", 0.0)
        cardinal = motion_stats.get("dominant_direction_cardinal", "None")

        hud_lines = [
            f"ByteTrack Active People: {count} | Avg Speed: {speed:.1f} px/s",
            f"Dominant Flow: {cardinal} | Directional Entropy: {entropy:.2f} (Turbulence)"
        ]

        # Draw dark glass banner
        cv2.rectangle(out, (10, 10), (520, 68), (15, 18, 22), -1)
        cv2.rectangle(out, (10, 10), (520, 68), (0, 200, 255), 1)
        cv2.putText(out, hud_lines[0], (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.putText(out, hud_lines[1], (18, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 1)

        return out
