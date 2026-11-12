"""
Module B: CSRNet (Congested Scene Recognition Network)
PyTorch implementation of CSRNet for crowd counting and density map estimation.
Reference: Li, Zhang, Chen, 'CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes', CVPR 2018.

Architecture:
  - Front-end: First 10 convolutional layers of VGG-16 with 3 MaxPool layers (receptive field & semantic features).
  - Back-end: 6 dilated convolutional layers (dilation rate = 2) maintaining spatial resolution while enlarging receptive field.
  - Output layer: 1x1 Convolution yielding continuous 2D crowd density map.
  - Integration: Predicted crowd count = sum(density_map).
"""

import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import cv2
from PIL import Image


class CSRNet(nn.Module):
    """
    CSRNet Architecture for Congested Scene Crowd Counting.
    """
    def __init__(self, load_weights: bool = True):
        super(CSRNet, self).__init__()
        self.seen = 0

        # Front-end: VGG16 first 10 Conv layers (conv1_1 to conv4_3)
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]

        self.frontend = self._make_layers(self.frontend_feat, in_channels=3)
        self.backend = self._make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self._initialize_weights()

        if load_weights:
            self._load_vgg_frontend()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass generating continuous density map."""
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        # Density must be non-negative
        return torch.relu(x)

    def _initialize_weights(self):
        """Initializes backend and output layers."""
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)

    def _load_vgg_frontend(self):
        """Loads pretrained VGG16 weights for feature extraction backbone."""
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            vgg_dict = vgg.features.state_dict()
            frontend_dict = self.frontend.state_dict()

            # Transfer corresponding weights
            transferred = 0
            for (k_front, v_front), (k_vgg, v_vgg) in zip(frontend_dict.items(), vgg_dict.items()):
                if v_front.size() == v_vgg.size():
                    frontend_dict[k_front] = v_vgg
                    transferred += 1
            self.frontend.load_state_dict(frontend_dict)
        except Exception as e:
            # Fallback gracefully if offline
            pass

    def _make_layers(self, cfg, in_channels=3, batch_norm=False, dilation=False):
        layers = []
        d_rate = 2 if dilation else 1
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


class CSRNetDensityEstimator:
    """Wrapper and inference engine for CSRNet crowd density estimation."""

    def __init__(
        self,
        weights_path: Optional[str] = "models/crowd_counting/csrnet_weights.pth",
        calibration_scale: float = 0.36,
        device: str = "cpu"
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.model = CSRNet(load_weights=True).to(self.device)
        self.weights_path = Path(weights_path) if weights_path else None
        self.calibration_scale = calibration_scale

        if self.weights_path and self.weights_path.exists():
            try:
                state = torch.load(str(self.weights_path), map_location=self.device)
                self.model.load_state_dict(state)
            except Exception:
                pass
        self.model.eval()

        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def preprocess(self, img_rgb: np.ndarray, max_dim: int = 1024) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocesses RGB image: resize if too large, normalize, convert to tensor.
        """
        h, w = img_rgb.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w, new_h = int(w * scale), int(h * scale)
            # Ensure divisible by 8 for VGG pooling layers
            new_w = (new_w // 8) * 8
            new_h = (new_h // 8) * 8
            img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            new_w = (w // 8) * 8
            new_h = (h // 8) * 8
            img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA) if (w % 8 != 0 or h % 8 != 0) else img_rgb

        img_norm = (img_resized.astype(np.float32) / 255.0 - self.mean) / self.std
        tensor = torch.from_numpy(img_norm.transpose((2, 0, 1))).unsqueeze(0).float().to(self.device)
        return tensor, (h, w)

    def predict(self, image_input: Union[str, Path, np.ndarray], max_dim: int = 1024) -> Dict[str, Any]:
        """
        Predict continuous density map and estimated crowd count.
        """
        import time
        if isinstance(image_input, (str, Path)):
            img_bgr = cv2.imread(str(image_input))
            if img_bgr is None:
                raise ValueError(f"Could not load image from {image_input}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image_input

        orig_h, orig_w = img_rgb.shape[:2]
        tensor, _ = self.preprocess(img_rgb, max_dim=max_dim)

        t0 = time.perf_counter()
        with torch.no_grad():
            density_out = self.model(tensor)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        density_map_raw = density_out.squeeze().cpu().numpy() * self.calibration_scale
        # Spatial integral count
        estimated_count = float(density_map_raw.sum())

        # Resize density map back to original image size for display and alignment
        density_map_full = cv2.resize(density_map_raw, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        # Ensure non-negative
        density_map_full = np.clip(density_map_full, 0.0, None)

        # Recalibrate integral to match raw count
        full_sum = density_map_full.sum()
        if full_sum > 0:
            density_map_full = density_map_full * (estimated_count / full_sum)

        return {
            "estimated_count": float(round(estimated_count, 1)),
            "density_map": density_map_full,
            "density_map_raw": density_map_raw,
            "max_density": float(density_map_full.max()),
            "inference_time_ms": float(round(latency_ms, 2)),
            "fps": float(round(1000.0 / latency_ms, 1)) if latency_ms > 0 else 0.0,
            "image_shape": (orig_h, orig_w)
        }

    def save_weights(self, path: Optional[str] = None):
        """Saves current CSRNet model weights to disk."""
        target_path = Path(path) if path else self.weights_path
        if target_path:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), str(target_path))
