"""
Official Dataset Subset Downloader & Access Documentation Utility
Downloads a manageable, stratified 10-image validation subset of the official
NWPU-Crowd dataset (2 images per density tier: Negative, Low, Medium, High, Extreme)
directly from HuggingFace (Junyu3/NWPU-Crowd) using HTTP Range requests, avoiding
the need to download all ~19.3 GB of image archives at once.

Also prints official access URLs, folder expectations, and academic citations
for all 5 benchmark datasets used in this project.
"""

import argparse
import io
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
import requests


NWPU_SUBSET_IDS = [
    ("3143", "Negative"),
    ("3150", "Negative"),
    ("3274", "Low"),
    ("3154", "Low"),
    ("3334", "Medium"),
    ("3200", "Medium"),
    ("3318", "High"),
    ("3468", "High"),
    ("3126", "Extreme"),
    ("3156", "Extreme"),
]

HF_BASE = "https://huggingface.co/datasets/Junyu3/NWPU-Crowd/resolve/main"


class HttpRangeFile(io.IOBase):
    """Seekable HTTP Range reader enabling selective extraction from remote ZIP archives."""

    def __init__(self, url: str):
        self.session = requests.Session()
        head = self.session.head(url, allow_redirects=True, timeout=20)
        head.raise_for_status()
        self.final_url = head.url
        self.size = int(head.headers["Content-Length"])
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n == -1:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b""
        end = min(self.size - 1, self.pos + n - 1)
        resp = self.session.get(
            self.final_url,
            headers={"Range": f"bytes={self.pos}-{end}"},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.content
        self.pos += len(data)
        return data


def download_nwpu_real_subset(target_dir: str = "datasets/nwpu_crowd/real_val_subset") -> Dict[str, object]:
    """
    Downloads 10 official NWPU-Crowd validation images and their official JSON annotations.
    """
    out_root = Path(target_dir)
    img_dir = out_root / "images"
    ann_dir = out_root / "annotations"
    splits_dir = out_root / "splits"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    needed_ids = [item[0] for item in NWPU_SUBSET_IDS]
    existing_imgs = {p.stem for p in img_dir.glob("*.jpg")}
    existing_jsons = {p.stem for p in ann_dir.glob("*.json")}
    if all(sid in existing_imgs and sid in existing_jsons for sid in needed_ids):
        print(f"Real NWPU-Crowd validation subset (10 images) already present in {out_root}")
        return {"downloaded": 0, "total_present": len(needed_ids), "target_dir": str(out_root)}

    # 1. Fetch official val.txt and readme.md
    val_resp = requests.get(f"{HF_BASE}/val.txt", timeout=20)
    val_resp.raise_for_status()
    val_lines = {line.split()[0]: line for line in val_resp.text.splitlines() if line.strip()}

    # 2. Fetch jsons.zip (~28 MB) and extract only the 10 subset JSONs
    print("Downloading NWPU-Crowd official jsons.zip (28 MB) to extract subset annotations...")
    jsons_resp = requests.get(f"{HF_BASE}/jsons.zip", timeout=120)
    jsons_resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(jsons_resp.content)) as zf_json:
        for sid, tier in NWPU_SUBSET_IDS:
            jname = f"{sid}.json"
            if jname in zf_json.namelist():
                raw = zf_json.read(jname).decode("utf-8")
                data = json.loads(raw)
                data["meta"] = {
                    "dataset": "NWPU-Crowd Official Validation Subset",
                    "source_url": "https://huggingface.co/datasets/Junyu3/NWPU-Crowd",
                    "density_tier": tier,
                    "box_format": "xyxy",
                    "val_txt_entry": val_lines.get(sid, sid),
                }
                with open(ann_dir / jname, "w", encoding="utf-8") as f:
                    json.dump(data, f)

    # 3. Extract the 10 validation images from remote images_part4.zip via HTTP Range
    print("Extracting 10 official NWPU-Crowd validation images from images_part4.zip via HTTP Range...")
    range_file = HttpRangeFile(f"{HF_BASE}/images_part4.zip")
    downloaded_count = 0
    with zipfile.ZipFile(range_file) as zf_img:
        for sid, tier in NWPU_SUBSET_IDS:
            img_name = f"{sid}.jpg"
            dest_path = img_dir / img_name
            if img_name in zf_img.namelist():
                img_bytes = zf_img.read(img_name)
                with open(dest_path, "wb") as f:
                    f.write(img_bytes)
                downloaded_count += 1
                print(f"  Saved {img_name} ({tier}, {len(img_bytes)/1024:.1f} KB)")

    with open(splits_dir / "val_subset.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(needed_ids) + "\n")

    return {"downloaded": downloaded_count, "total_present": len(needed_ids), "target_dir": str(out_root)}


def print_full_dataset_instructions():
    """Prints official dataset access instructions and license notes."""
    guide = """
================================================================================
OFFICIAL DATASET ACCESS & FULL EVALUATION INSTRUCTIONS
================================================================================
1. NWPU-Crowd (Images + Head Points & Boxes, 5,109 images, ~19.3 GB)
   - Official Site : https://gjy3035.github.io/NWPU-Crowd-Sample-Code/
   - Mirror / Data : https://huggingface.co/datasets/Junyu3/NWPU-Crowd
   - Access        : Public academic/non-commercial use. Download images_part1..5b.zip,
                     jsons.zip, mats.zip, train.txt, val.txt, test.txt into datasets/nwpu_crowd/.

2. UCF-QNRF (Extreme Dense Crowd Images + annPoints .mat, 1,535 images, ~4.3 GB)
   - Official Site : https://www.crcv.ucf.edu/research/data-sets/ucf-qnrf/
   - Access        : Academic download archive from CRCV UCF (note: UCF site occasionally
                     requires manual browser download due to institutional TLS chains).
                     Extract Train/ and Test/ into datasets/ucf_qnrf/.

3. UCF Crowd Segmentation / Crowd Data (Motion Flow & Segmentation)
   - Official Site : https://www.crcv.ucf.edu/data/crowd.php
   - Access        : Download crowd sequence archives directly from CRCV UCF web portal
                     and place images/flow files into datasets/ucf_crowd/.

4. MOTSynth (Synthetic Multi-Object Pedestrian Tracking, 764 Full-HD sequences, >200 GB)
   - Official Site : https://motchallenge.net/data/MOTSynth/ (and CVPR/ICCV MOTSynth release)
   - Access        : Requires downloading selected sequence archives and MOT gt.txt files
                     via MOTChallenge / official scripts into datasets/motsynth/.

5. UCF Tracking in High-Density Crowds (Dense Video Crowd Tracking)
   - Official Site : https://www.crcv.ucf.edu/data/crowd.php
   - Access        : Download high-density crowd videos & track ground truth from CRCV UCF
                     into datasets/ucf_tracking/.
================================================================================
"""
    print(guide)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NWPU-Crowd subset or show dataset access guide.")
    parser.add_argument("--instructions", action="store_true", help="Print full dataset access instructions.")
    parser.add_argument("--download-nwpu", action="store_true", help="Download 10-image official NWPU-Crowd val subset.")
    args = parser.parse_args()

    if args.instructions:
        print_full_dataset_instructions()
    if args.download_nwpu or (not args.instructions and not args.download_nwpu):
        res = download_nwpu_real_subset()
        print("Result:", res)
