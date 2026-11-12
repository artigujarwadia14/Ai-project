"""
CSRNet Training & Density Map Calibration Script
Trains/Fine-tunes CSRNet backend dilated layers on benchmark density maps using pixel-wise MSE Loss:
  L = (1 / 2N) * sum_i || D_pred - D_gt ||^2

This fits the dilated convolutional filters to transform VGG-16 high-level feature maps
into calibrated crowd density distributions.
"""

import sys
from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.counting.csrnet_model import CSRNet, CSRNetDensityEstimator
from src.data_preprocessing.nwpu_parser import NWPUDatasetParser
from src.data_preprocessing.ucf_qnrf_parser import UCFQNRFDatasetParser
from src.data_preprocessing.density_map_generator import generate_density_map_adaptive, generate_density_map_fixed


def train_csrnet(
    epochs: int = 15,
    lr: float = 1e-5,
    save_path: str = "models/crowd_counting/csrnet_weights.pth"
):
    print("Preparing training dataset for CSRNet calibration...")
    device = torch.device("cpu")
    model = CSRNet(load_weights=True).to(device)

    # Freeze frontend VGG backbone to prevent destroying pretrained features, train dilated backend
    for param in model.frontend.parameters():
        param.requires_grad = False
    for param in model.backend.parameters():
        param.requires_grad = True
    for param in model.output_layer.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4
    )
    criterion = nn.MSELoss(reduction='sum')

    nwpu = NWPUDatasetParser("datasets/nwpu_crowd")
    qnrf = UCFQNRFDatasetParser("datasets/ucf_qnrf")

    training_samples = []

    # Prepare NWPU
    for s_name in nwpu.list_samples():
        ann = nwpu.parse_annotation(s_name)
        img_path = ann["image_path"]
        img = cv2.imread(img_path)
        if img is None:
            continue
        # Resize to fixed manageable 512x384 for fast CPU training
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]
        img_small = cv2.resize(img_rgb, (512, 384), interpolation=cv2.INTER_AREA)

        target_h, target_w = 384 // 8, 512 // 8
        pts = ann["points"]
        if len(pts) > 0:
            scale_x = 512.0 / orig_w / 8.0
            scale_y = 384.0 / orig_h / 8.0
            scaled_pts = pts.copy()
            scaled_pts[:, 0] *= scale_x
            scaled_pts[:, 1] *= scale_y
        else:
            scaled_pts = np.zeros((0, 2), dtype=np.float32)

        gt_density = generate_density_map_fixed((target_h, target_w), scaled_pts, sigma=1.8)

        training_samples.append({
            "img_rgb": img_small,
            "gt_density": gt_density,
            "count": ann["ground_truth_count"]
        })

    # Prepare QNRF
    for s_name in qnrf.list_samples():
        ann = qnrf.parse_annotation(s_name)
        img_path = ann["image_path"]
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]
        img_small = cv2.resize(img_rgb, (512, 384), interpolation=cv2.INTER_AREA)

        target_h, target_w = 384 // 8, 512 // 8
        pts = ann["points"]
        if len(pts) > 0:
            scale_x = 512.0 / orig_w / 8.0
            scale_y = 384.0 / orig_h / 8.0
            scaled_pts = pts.copy()
            scaled_pts[:, 0] *= scale_x
            scaled_pts[:, 1] *= scale_y
        else:
            scaled_pts = np.zeros((0, 2), dtype=np.float32)

        gt_density = generate_density_map_fixed((target_h, target_w), scaled_pts, sigma=1.8)

        training_samples.append({
            "img_rgb": img_small,
            "gt_density": gt_density,
            "count": ann["ground_truth_count"]
        })

    print(f"Loaded {len(training_samples)} training pairs. Commencing {epochs} training epochs...")
    estimator = CSRNetDensityEstimator()

    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        t0 = time.time()
        for sample in training_samples:
            img_rgb = sample["img_rgb"]
            gt_density = sample["gt_density"]

            tensor, _ = estimator.preprocess(img_rgb)
            gt_tensor = torch.from_numpy(gt_density).unsqueeze(0).unsqueeze(0).float().to(device)

            optimizer.zero_grad()
            output = model(tensor)

            # Match sizes
            if output.shape != gt_tensor.shape:
                gt_tensor = nn.functional.interpolate(gt_tensor, size=output.shape[2:], mode='bilinear', align_corners=False)

            loss = criterion(output, gt_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        dt = time.time() - t0
        print(f"Epoch {ep:02d}/{epochs:02d} | Loss: {total_loss:.4f} | Time: {dt:.2f}s", flush=True)

    # Save trained weights
    save_p = Path(save_path)
    save_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(save_p))
    print(f"Calibrated CSRNet weights successfully saved to {save_path}", flush=True)


if __name__ == "__main__":
    train_csrnet(epochs=8)
