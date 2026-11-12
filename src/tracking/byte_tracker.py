"""
Module C: ByteTrack Multi-Object Tracker & Movement Analysis
Implementation of ByteTrack: Association of every detection box (high and low confidence).
Reference: Zhang et al., 'ByteTrack: Multi-Object Tracking by Associating Every Detection Box', ECCV 2022.

Key innovations:
  1. Primary matching: Matches tracklets against high-confidence detections using IoU distance.
  2. Secondary matching: Recovers occluded/blurred targets by matching remaining tracklets
     against low-confidence detections (which classical SORT/DeepSORT discard).
  3. Trajectory management: Preserves track history, computes velocity, speed, and heading angle.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


def box_iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Computes pairwise IoU matrix between N boxes_a and M boxes_b.
    Boxes are in [x1, y1, x2, y2] format.
    Returns: (N, M) matrix with IoU values.
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    # Coordinates
    x1_a, y1_a, x2_a, y2_a = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    x1_b, y1_b, x2_b, y2_b = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    # Intersection
    xx1 = np.maximum(x1_a[:, None], x1_b[None, :])
    yy1 = np.maximum(y1_a[:, None], y1_b[None, :])
    xx2 = np.minimum(x2_a[:, None], x2_b[None, :])
    yy2 = np.minimum(y2_a[:, None], y2_b[None, :])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    intersection = w * h

    # Union
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    union = area_a[:, None] + area_b[None, :] - intersection

    iou = np.zeros_like(intersection)
    valid_mask = union > 0
    iou[valid_mask] = intersection[valid_mask] / union[valid_mask]
    return iou


class Tracklet:
    """Represents a single tracked person trajectory over time."""

    def __init__(self, track_id: int, bbox: np.ndarray, conf: float, frame_idx: int):
        self.track_id = track_id
        self.bbox = bbox.astype(np.float32)  # [x1, y1, x2, y2]
        self.conf = conf
        self.start_frame = frame_idx
        self.last_frame = frame_idx
        self.lost_frames = 0
        self.is_active = True

        # Center coordinates
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.history: List[Tuple[float, float, int]] = [(cx, cy, frame_idx)]

        # Velocity vector (vx, vy in pixels/sec)
        self.vx = 0.0
        self.vy = 0.0
        self.speed = 0.0
        self.direction_deg = 0.0

    def update(self, bbox: np.ndarray, conf: float, frame_idx: int, fps: float = 25.0):
        """Updates tracklet state with a newly associated detection box."""
        self.bbox = bbox.astype(np.float32)
        self.conf = conf
        self.lost_frames = 0
        self.is_active = True

        new_cx = (bbox[0] + bbox[2]) / 2.0
        new_cy = (bbox[1] + bbox[3]) / 2.0

        if len(self.history) > 0:
            old_cx, old_cy, old_f = self.history[-1]
            df = max(1, frame_idx - old_f)
            dt = df / fps
            self.vx = float((new_cx - old_cx) / dt)
            self.vy = float((new_cy - old_cy) / dt)
            self.speed = float(np.sqrt(self.vx**2 + self.vy**2))
            self.direction_deg = float(np.degrees(np.arctan2(self.vy, self.vx)) % 360)

        self.history.append((new_cx, new_cy, frame_idx))
        if len(self.history) > 40:
            self.history.pop(0)

        self.last_frame = frame_idx

    def mark_lost(self):
        """Increments lost counter when no detection matches this frame."""
        self.lost_frames += 1
        if self.lost_frames > 2:
            self.is_active = False


class ByteTracker:
    """ByteTrack Tracker implementation with two-stage association."""

    def __init__(
        self,
        high_thresh: float = 0.50,
        low_thresh: float = 0.15,
        match_thresh: float = 0.35,
        max_lost_frames: int = 30,
        fps: float = 25.0
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_lost_frames = max_lost_frames
        self.fps = fps

        self.next_id = 1
        self.tracklets: List[Tracklet] = []
        self.frame_idx = 0

    def update(self, bboxes: np.ndarray, confidences: np.ndarray) -> List[Tracklet]:
        """
        Processes detections for the current frame and updates tracks.

        Args:
            bboxes: (N, 4) in [x1, y1, x2, y2]
            confidences: (N,) confidence scores
        """
        self.frame_idx += 1
        num_dets = len(bboxes)

        if num_dets == 0:
            for t in self.tracklets:
                t.mark_lost()
            self.tracklets = [t for t in self.tracklets if t.lost_frames <= self.max_lost_frames]
            return [t for t in self.tracklets if t.is_active]

        # 1. Split detections into high-conf and low-conf sets
        high_mask = confidences >= self.high_thresh
        low_mask = (confidences >= self.low_thresh) & (confidences < self.high_thresh)

        high_boxes = bboxes[high_mask]
        high_confs = confidences[high_mask]
        low_boxes = bboxes[low_mask]
        low_confs = confidences[low_mask]

        active_tracks = [t for t in self.tracklets if t.lost_frames <= self.max_lost_frames]

        # ---------------- Stage 1: Match Active Tracks with High-Conf Dets ----------------
        matched_tracks_1 = set()
        matched_dets_1 = set()

        if len(active_tracks) > 0 and len(high_boxes) > 0:
            track_boxes = np.array([t.bbox for t in active_tracks])
            iou_matrix = box_iou_batch(track_boxes, high_boxes)

            # Greedy bipartite matching
            while True:
                max_iou = np.max(iou_matrix) if iou_matrix.size > 0 else 0.0
                if max_iou < self.match_thresh:
                    break
                t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                active_tracks[t_idx].update(high_boxes[d_idx], high_confs[d_idx], self.frame_idx, self.fps)
                matched_tracks_1.add(t_idx)
                matched_dets_1.add(d_idx)
                # Invalidate row and column
                iou_matrix[t_idx, :] = -1.0
                iou_matrix[:, d_idx] = -1.0

        # Remaining unmatched tracks
        unmatched_track_indices = [i for i in range(len(active_tracks)) if i not in matched_tracks_1]
        unmatched_high_det_indices = [i for i in range(len(high_boxes)) if i not in matched_dets_1]

        # ---------------- Stage 2: Match Unmatched Tracks with Low-Conf Dets ----------------
        matched_tracks_2 = set()
        if len(unmatched_track_indices) > 0 and len(low_boxes) > 0:
            rem_track_boxes = np.array([active_tracks[i].bbox for i in unmatched_track_indices])
            iou_matrix_2 = box_iou_batch(rem_track_boxes, low_boxes)

            while True:
                max_iou = np.max(iou_matrix_2) if iou_matrix_2.size > 0 else 0.0
                if max_iou < self.match_thresh:
                    break
                t_idx_sub, d_idx = np.unravel_index(np.argmax(iou_matrix_2), iou_matrix_2.shape)
                real_t_idx = unmatched_track_indices[t_idx_sub]
                active_tracks[real_t_idx].update(low_boxes[d_idx], low_confs[d_idx], self.frame_idx, self.fps)
                matched_tracks_2.add(real_t_idx)
                iou_matrix_2[t_idx_sub, :] = -1.0
                iou_matrix_2[:, d_idx] = -1.0

        # ---------------- Stage 3: Initiate New Tracks from Unmatched High-Conf Dets ----------------
        for d_idx in unmatched_high_det_indices:
            new_track = Tracklet(self.next_id, high_boxes[d_idx], high_confs[d_idx], self.frame_idx)
            self.next_id += 1
            self.tracklets.append(new_track)

        # ---------------- Stage 4: Mark Remaining Tracks as Lost ----------------
        all_matched_t = matched_tracks_1.union(matched_tracks_2)
        for i, t in enumerate(active_tracks):
            if i not in all_matched_t:
                t.mark_lost()

        # Prune dead tracks
        self.tracklets = [t for t in self.tracklets if t.lost_frames <= self.max_lost_frames]

        return [t for t in self.tracklets if t.is_active]
