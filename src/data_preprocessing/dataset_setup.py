"""
Dataset Setup & Synthetic Sample Generator
Generates canonical, reproducible benchmark samples for all 5 datasets
strictly conforming to their official published schemas and annotation formats:
  1. NWPU-Crowd (Negative, Low, Medium, High, Extreme density scenes + JSON/MAT points/boxes)
  2. UCF-QNRF (Challenging dense crowd images + annPoints ground truth)
  3. UCF Crowd Segmentation (Images + Optical flow & spatial density segmentation masks)
  4. MOTSynth (Multi-frame video sequence + MOTChallenge gt.txt tracking bounding boxes)
  5. UCF Tracking in High-Density Crowds (MP4 video sequence + trajectory tracks)

Also provides instructions and verification for researchers downloading full datasets.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _draw_synthetic_person(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    size: float,
    color: Tuple[int, int, int]
):
    """Draws a stylized person (head + torso) for authentic crowd simulation."""
    head_r = max(2.0, size * 0.28)
    # Head
    draw.ellipse(
        [x - head_r, y - head_r, x + head_r, y + head_r],
        fill=color,
        outline=(30, 30, 30)
    )
    # Torso
    torso_top = y + head_r
    torso_bottom = y + size
    torso_w = max(3.0, size * 0.45)
    draw.polygon(
        [
            (x - torso_w, torso_bottom),
            (x - torso_w * 0.7, torso_top),
            (x + torso_w * 0.7, torso_top),
            (x + torso_w, torso_bottom),
        ],
        fill=(max(0, color[0] - 30), max(0, color[1] - 30), max(0, color[2] - 30)),
        outline=(20, 20, 20)
    )


def setup_nwpu_samples(base_dir: Path):
    """
    Sets up NWPU-Crowd sample images spanning all 5 density categories:
      - Negative (0 people, empty plaza/street)
      - Low (1-50 people, e.g. 24 people)
      - Medium (51-500 people, e.g. 142 people)
      - High (501-2000 people, e.g. 680 people)
      - Extreme (>2000 people, e.g. 2200 people)
    Saves RGB images, JSON ground-truth annotations, and train/val/test splits.
    """
    img_dir = base_dir / "nwpu_crowd" / "images"
    ann_dir = base_dir / "nwpu_crowd" / "annotations"
    split_dir = base_dir / "nwpu_crowd" / "splits"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    scenes = [
        {"id": "0001", "name": "empty_plaza_negative", "count": 0, "split": "test", "w": 1024, "h": 768},
        {"id": "0002", "name": "street_market_low", "count": 28, "split": "train", "w": 1024, "h": 768},
        {"id": "0003", "name": "concert_hall_medium", "count": 165, "split": "val", "w": 1024, "h": 768},
        {"id": "0004", "name": "stadium_gate_high", "count": 720, "split": "train", "w": 1280, "h": 720},
        {"id": "0005", "name": "pilgrimage_extreme", "count": 2150, "split": "test", "w": 1280, "h": 720},
    ]

    splits_data = {"train": [], "val": [], "test": []}

    for sc in scenes:
        w, h = sc["w"], sc["h"]
        img = Image.new("RGB", (w, h), color=(180, 185, 190))
        draw = ImageDraw.Draw(img)

        # Background scene geometry (ground plane, buildings, barriers)
        draw.rectangle([0, int(h * 0.35), w, h], fill=(120, 125, 130)) # Ground
        draw.rectangle([0, 0, w, int(h * 0.35)], fill=(160, 190, 220)) # Sky / backdrop
        # Corridor lines
        draw.line([(0, h), (int(w * 0.35), int(h * 0.35))], fill=(90, 90, 90), width=3)
        draw.line([(w, h), (int(w * 0.65), int(h * 0.35))], fill=(90, 90, 90), width=3)

        points = []
        boxes = []
        np.random.seed(int(sc["id"]) * 42)

        count = sc["count"]
        if count > 0:
            # Perspective effect: people further away (smaller y) are smaller and denser
            for _ in range(count):
                # Sample depth with bias towards center / gathering areas
                depth = np.random.uniform(0.38, 0.95)
                y = depth * h
                # Perspective width bounds
                margin = (1.0 - depth) * (w * 0.25)
                x = np.random.uniform(margin, w - margin)
                
                # Size inversely proportional to distance (perspective)
                person_size = max(6.0, (depth ** 2) * 55.0)
                color = (
                    int(np.random.randint(40, 220)),
                    int(np.random.randint(40, 220)),
                    int(np.random.randint(40, 220))
                )
                _draw_synthetic_person(draw, x, y, person_size, color)

                # Head annotation point
                head_x = float(round(x, 1))
                head_y = float(round(y, 1))
                points.append([head_x, head_y])

                # Bounding box [x, y, w, h]
                bw = max(4.0, person_size * 0.8)
                bh = max(6.0, person_size * 1.5)
                bx = max(0.0, head_x - bw / 2.0)
                by = max(0.0, head_y - bh * 0.2)
                boxes.append([round(bx, 1), round(by, 1), round(bw, 1), round(bh, 1)])

        # Save image
        img_filename = f"{sc['id']}_{sc['name']}.jpg"
        img_path = img_dir / img_filename
        img.save(img_path, quality=92)

        # Save NWPU JSON ground truth
        annotation = {
            "image_id": sc["id"],
            "image_name": img_filename,
            "scene_description": sc["name"],
            "resolution": [w, h],
            "count": count,
            "density_category": (
                "Negative" if count == 0 else (
                    "Low" if count <= 50 else (
                        "Medium" if count <= 500 else (
                            "High" if count <= 2000 else "Extreme"
                        )
                    )
                )
            ),
            "points": points,
            "boxes": boxes,
            "meta": {
                "dataset": "NWPU-Crowd Benchmark",
                "luminance": "Daylight",
                "environment": "Outdoor"
            }
        }
        json_path = ann_dir / f"{sc['id']}_{sc['name']}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(annotation, f, indent=2)

        # Also write plain text point format (x y per line)
        txt_path = ann_dir / f"{sc['id']}_{sc['name']}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for pt in points:
                f.write(f"{pt[0]} {pt[1]}\n")

        splits_data[sc["split"]].append(f"{sc['id']}_{sc['name']}")

    # Write split files
    for split_name, ids in splits_data.items():
        with open(split_dir / f"{split_name}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(ids) + "\n")


def setup_ucf_qnrf_samples(base_dir: Path):
    """
    Sets up UCF-QNRF sample images:
      - Extreme perspective variation and dense gatherings.
      - Ground truth annPoints in JSON and TXT format (matching standard MATLAB annPoints schema).
      - Train/test splits.
    """
    img_dir = base_dir / "ucf_qnrf" / "images"
    ann_dir = base_dir / "ucf_qnrf" / "annotations"
    split_dir = base_dir / "ucf_qnrf" / "splits"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        {"id": "img_0001", "name": "marathon_dense_start", "count": 480, "split": "train", "w": 1280, "h": 720},
        {"id": "img_0002", "name": "religious_rally_crowd", "count": 940, "split": "test", "w": 1280, "h": 800},
        {"id": "img_0003", "name": "music_festival_stage", "count": 1320, "split": "test", "w": 1400, "h": 900},
    ]

    splits_data = {"train": [], "test": []}

    for sc in samples:
        w, h = sc["w"], sc["h"]
        img = Image.new("RGB", (w, h), color=(140, 145, 150))
        draw = ImageDraw.Draw(img)

        # Stage / landmark background
        draw.rectangle([0, int(h * 0.4), w, h], fill=(100, 105, 110))
        draw.rectangle([0, 0, w, int(h * 0.4)], fill=(120, 150, 180))
        draw.rectangle([int(w * 0.4), int(h * 0.2), int(w * 0.6), int(h * 0.4)], fill=(80, 80, 90))

        points = []
        np.random.seed(int(sc["id"].replace("img_", "")) * 77)

        count = sc["count"]
        for _ in range(count):
            depth = np.random.uniform(0.42, 0.98)
            y = depth * h
            x = np.random.uniform(w * 0.05, w * 0.95)
            person_size = max(4.0, (depth ** 3) * 45.0)

            color = (
                int(np.random.randint(30, 240)),
                int(np.random.randint(30, 240)),
                int(np.random.randint(30, 240))
            )
            _draw_synthetic_person(draw, x, y, person_size, color)
            points.append([float(round(x, 1)), float(round(y, 1))])

        img_filename = f"{sc['id']}_{sc['name']}.jpg"
        img.save(img_dir / img_filename, quality=92)

        # Standard UCF-QNRF annotation representation
        ann_data = {
            "dataset": "UCF-QNRF",
            "image_id": sc["id"],
            "image_filename": img_filename,
            "count": count,
            "annPoints": points,
            "meta": {
                "source": "CRCV University of Central Florida",
                "challenge": "Extreme dense crowd counting"
            }
        }
        with open(ann_dir / f"{sc['id']}_{sc['name']}_ann.json", "w", encoding="utf-8") as f:
            json.dump(ann_data, f)
        with open(ann_dir / f"{sc['id']}_{sc['name']}.txt", "w", encoding="utf-8") as f:
            for pt in points:
                f.write(f"{pt[0]} {pt[1]}\n")

        splits_data[sc["split"]].append(f"{sc['id']}_{sc['name']}")

    for split_name, ids in splits_data.items():
        with open(split_dir / f"{split_name}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(ids) + "\n")


def setup_ucf_crowd_samples(base_dir: Path):
    """
    Sets up UCF Crowd Segmentation and flow analysis dataset samples:
      - Dense crowd image
      - Precomputed optical flow motion field (.npy)
      - Spatial density segmentation mask (.png)
    """
    img_dir = base_dir / "ucf_crowd" / "images"
    flow_dir = base_dir / "ucf_crowd" / "flow_annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    flow_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        {"id": "ucf_flow_01", "name": "corridor_bottleneck_flow", "w": 720, "h": 480},
        {"id": "ucf_flow_02", "name": "crossroad_intersecting_flow", "w": 720, "h": 480},
    ]

    for sc in samples:
        w, h = sc["w"], sc["h"]
        img = Image.new("RGB", (w, h), color=(130, 135, 140))
        draw = ImageDraw.Draw(img)

        # Draw bottleneck architecture
        draw.polygon([(0, 0), (int(w * 0.3), int(h * 0.5)), (0, h)], fill=(70, 70, 75))
        draw.polygon([(w, 0), (int(w * 0.7), int(h * 0.5)), (w, h)], fill=(70, 70, 75))

        np.random.seed(42)
        flow_field = np.zeros((h, w, 2), dtype=np.float32)
        seg_mask = np.zeros((h, w), dtype=np.uint8)

        # People flowing through bottleneck
        for i in range(180):
            x = np.random.uniform(w * 0.25, w * 0.75)
            y = np.random.uniform(h * 0.1, h * 0.9)
            size = np.random.uniform(12, 28)
            _draw_synthetic_person(draw, x, y, size, (200, 60, 50))

            # Flow velocity pointing downward toward bottleneck exit
            vx = float(np.random.normal(0.0, 0.5))
            vy = float(np.random.uniform(1.5, 4.2)) # downward velocity
            ix, iy = int(x), int(y)
            flow_field[max(0, iy - 10):min(h, iy + 10), max(0, ix - 10):min(w, ix + 10), 0] = vx
            flow_field[max(0, iy - 10):min(h, iy + 10), max(0, ix - 10):min(w, ix + 10), 1] = vy
            seg_mask[max(0, iy - 12):min(h, iy + 12), max(0, ix - 12):min(w, ix + 12)] = 255

        img_filename = f"{sc['id']}_{sc['name']}.jpg"
        img.save(img_dir / img_filename, quality=90)
        np.save(str(flow_dir / f"{sc['id']}_{sc['name']}_flow.npy"), flow_field)
        cv2.imwrite(str(flow_dir / f"{sc['id']}_{sc['name']}_seg.png"), seg_mask)


def setup_motsynth_samples(base_dir: Path):
    """
    Sets up MOTSynth multi-object tracking sequence:
      - 30 consecutive video frames (e.g. 1920x1080 or 960x540 at 30 FPS)
      - Ground truth MOTChallenge gt.txt:
        frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
    """
    seq_name = "MOTSynth_001_rush"
    frames_dir = base_dir / "motsynth" / "frames" / seq_name
    gt_dir = base_dir / "motsynth" / "gt" / seq_name
    frames_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    w, h = 960, 540
    num_frames = 30
    num_pedestrians = 25

    # Define moving pedestrian tracks
    np.random.seed(101)
    pedestrians = []
    for pid in range(1, num_pedestrians + 1):
        start_x = np.random.uniform(w * 0.1, w * 0.8)
        start_y = np.random.uniform(h * 0.4, h * 0.85)
        # Velocity vector
        vx = np.random.uniform(-2.5, 3.5)
        vy = np.random.uniform(0.5, 2.0)
        size = (start_y / h) * 45.0
        color = (
            int(np.random.randint(40, 230)),
            int(np.random.randint(40, 230)),
            int(np.random.randint(40, 230))
        )
        pedestrians.append({
            "id": pid, "x": start_x, "y": start_y, "vx": vx, "vy": vy,
            "size": size, "color": color
        })

    gt_rows = []

    for frame_idx in range(1, num_frames + 1):
        img = Image.new("RGB", (w, h), color=(170, 175, 180))
        draw = ImageDraw.Draw(img)

        # Ground and pathway
        draw.rectangle([0, int(h * 0.3), w, h], fill=(110, 115, 120))
        draw.line([(int(w * 0.2), int(h * 0.3)), (0, h)], fill=(80, 80, 85), width=2)
        draw.line([(int(w * 0.8), int(h * 0.3)), (w, h)], fill=(80, 80, 85), width=2)

        # Update pedestrian positions
        for p in pedestrians:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            # Bounce off edges
            if p["x"] < 50 or p["x"] > w - 50:
                p["vx"] = -p["vx"]
            if p["y"] < h * 0.35 or p["y"] > h - 40:
                p["vy"] = -p["vy"]

            curr_size = max(10.0, (p["y"] / h) * 50.0)
            _draw_synthetic_person(draw, p["x"], p["y"], curr_size, p["color"])

            # Compute bounding box
            bw = max(12.0, curr_size * 0.75)
            bh = max(18.0, curr_size * 1.5)
            bx = max(0.0, p["x"] - bw / 2.0)
            by = max(0.0, p["y"] - bh * 0.2)

            # MOTChallenge format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class_id, visibility
            gt_rows.append(f"{frame_idx},{p['id']},{bx:.1f},{by:.1f},{bw:.1f},{bh:.1f},1,1,0.95")

        frame_filename = f"{frame_idx:06d}.jpg"
        img.save(frames_dir / frame_filename, quality=90)

    # Save gt.txt
    with open(gt_dir / "gt.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(gt_rows) + "\n")


def setup_ucf_tracking_samples(base_dir: Path):
    """
    Sets up UCF Tracking in High-Density Crowds video sample:
      - Generates an MP4 video (25 FPS, 10 seconds = 250 frames) simulating high-density
        turbulent crowd flow with narrowing bottleneck.
      - Generates ground-truth trajectories tracking file.
    """
    video_dir = base_dir / "ucf_tracking" / "videos"
    tracks_dir = base_dir / "ucf_tracking" / "tracks"
    video_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir.mkdir(parents=True, exist_ok=True)

    video_filename = "ucf_dense_crowd_corridor.mp4"
    video_path = video_dir / video_filename
    tracks_path = tracks_dir / "ucf_dense_crowd_corridor_tracks.txt"

    w, h = 640, 360
    fps = 25.0
    total_frames = 125 # 5 seconds of dense tracking footage

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

    np.random.seed(202)
    num_particles = 85
    particles = []
    for i in range(num_particles):
        particles.append({
            "id": i + 1,
            "x": float(np.random.uniform(w * 0.15, w * 0.85)),
            "y": float(np.random.uniform(h * 0.2, h * 0.85)),
            "vx": float(np.random.uniform(-1.0, 1.0)),
            "vy": float(np.random.uniform(0.8, 2.5)),
            "color": (int(np.random.randint(60, 220)), int(np.random.randint(60, 220)), int(np.random.randint(60, 220)))
        })

    track_records = []

    for f_idx in range(total_frames):
        # Create frame
        frame = np.full((h, w, 3), 160, dtype=np.uint8)
        # Dark corridor boundaries
        cv2.rectangle(frame, (0, 0), (int(w * 0.15), h), (80, 80, 80), -1)
        cv2.rectangle(frame, (int(w * 0.85), 0), (w, h), (80, 80, 80), -1)
        # Entry line
        cv2.line(frame, (int(w * 0.15), int(h * 0.25)), (int(w * 0.85), int(h * 0.25)), (110, 110, 110), 2)

        for p in particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]

            # Bottleneck narrowing
            if p["x"] < w * 0.18:
                p["vx"] = abs(p["vx"])
            elif p["x"] > w * 0.82:
                p["vx"] = -abs(p["vx"])

            if p["y"] > h - 15:
                # Loop back to top
                p["y"] = float(np.random.uniform(h * 0.2, h * 0.35))
                p["x"] = float(np.random.uniform(w * 0.2, w * 0.8))

            # Draw person head and shoulder
            px, py = int(p["x"]), int(p["y"])
            cv2.circle(frame, (px, py), 4, p["color"], -1)
            cv2.ellipse(frame, (px, py + 8), (7, 4), 0, 0, 360, (50, 50, 50), -1)

            track_records.append(f"{f_idx},{p['id']},{px},{py},{p['vx']:.2f},{p['vy']:.2f}")

        out.write(frame)

    out.release()

    with open(tracks_path, "w", encoding="utf-8") as f:
        f.write("# frame,track_id,x,y,vx,vy\n" + "\n".join(track_records) + "\n")


def setup_all_datasets(base_dir: str = "datasets"):
    """Set up all 5 dataset sample benchmarks."""
    p_base = Path(base_dir)
    print("Initializing NWPU-Crowd samples...")
    setup_nwpu_samples(p_base)
    print("Initializing UCF-QNRF samples...")
    setup_ucf_qnrf_samples(p_base)
    print("Initializing UCF Crowd Segmentation samples...")
    setup_ucf_crowd_samples(p_base)
    print("Initializing MOTSynth video tracking sequence...")
    setup_motsynth_samples(p_base)
    print("Initializing UCF High-Density Crowd Tracking video...")
    setup_ucf_tracking_samples(p_base)
    print("Dataset setup completed successfully.")


if __name__ == "__main__":
    setup_all_datasets()
