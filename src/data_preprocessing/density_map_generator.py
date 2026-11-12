"""
Crowd Density Map Generator
Implements standard Gaussian kernel convolution for head point annotations
with count preservation property: sum(D) == N.
Supports fixed Gaussian and k-NN geometry-adaptive Gaussian kernels.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
import cv2
from typing import List, Tuple, Optional


def generate_density_map_fixed(
    image_shape: Tuple[int, int],
    points: np.ndarray,
    sigma: float = 15.0
) -> np.ndarray:
    """
    Generate crowd density map using a fixed Gaussian kernel.
    
    Args:
        image_shape: (height, width) of target map
        points: Array of shape (N, 2) containing (x, y) coordinates
        sigma: Standard deviation of Gaussian kernel
        
    Returns:
        density_map: 2D float32 array where sum(density_map) == len(points)
    """
    h, w = image_shape
    density_map = np.zeros((h, w), dtype=np.float32)
    num_points = len(points)
    
    if num_points == 0:
        return density_map

    for pt in points:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < w and 0 <= y < h:
            density_map[y, x] += 1.0

    # Convolve with Gaussian filter
    # Mode='constant' with truncate=4.0 ensures Gaussian falls to near zero
    density_map = gaussian_filter(density_map, sigma=sigma, mode='constant')
    
    # Strictly preserve integral count
    current_sum = density_map.sum()
    if current_sum > 0:
        density_map = density_map * (num_points / current_sum)
        
    return density_map


def generate_density_map_adaptive(
    image_shape: Tuple[int, int],
    points: np.ndarray,
    k: int = 3,
    beta: float = 0.3,
    min_sigma: float = 4.0,
    max_sigma: float = 30.0
) -> np.ndarray:
    """
    Generate geometry-adaptive density map using k-NN distance.
    As proposed by Zhang et al. (MCNN) and used in UCF-QNRF/NWPU-Crowd:
    sigma_i = beta * mean(d_i^k), where d_i^k is distance to k nearest neighbors.
    
    Args:
        image_shape: (height, width)
        points: (N, 2) array of (x, y) coordinates
        k: Number of nearest neighbors
        beta: Scaling factor (typically 0.3)
        min_sigma: Minimum allowable sigma
        max_sigma: Maximum allowable sigma
    """
    h, w = image_shape
    density_map = np.zeros((h, w), dtype=np.float32)
    num_points = len(points)
    
    if num_points == 0:
        return density_map
    if num_points == 1:
        return generate_density_map_fixed(image_shape, points, sigma=15.0)

    # Build KDTree for k-NN queries
    tree = KDTree(points)
    # query k+1 because the point itself is returned at distance 0
    query_k = min(k + 1, num_points)
    distances, _ = tree.query(points, k=query_k)

    for i, pt in enumerate(points):
        x, y = int(round(pt[0])), int(round(pt[1]))
        if not (0 <= x < w and 0 <= y < h):
            continue
            
        # Mean distance to k nearest neighbors (excluding self at index 0)
        k_dists = distances[i, 1:]
        mean_dist = np.mean(k_dists) if len(k_dists) > 0 else 15.0
        sigma = np.clip(beta * mean_dist, min_sigma, max_sigma)
        
        # Local Gaussian patch to avoid filtering full image for every single point
        radius = int(round(3 * sigma))
        x_min, x_max = max(0, x - radius), min(w, x + radius + 1)
        y_min, y_max = max(0, y - radius), min(h, y + radius + 1)
        
        y_grid, x_grid = np.ogrid[y_min - y:y_max - y, x_min - x:x_max - x]
        patch = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        patch_sum = patch.sum()
        if patch_sum > 0:
            patch = patch / patch_sum
            density_map[y_min:y_max, x_min:x_max] += patch

    # Count preservation check
    current_sum = density_map.sum()
    if current_sum > 0:
        density_map = density_map * (num_points / current_sum)
        
    return density_map


def colorize_density_map(
    density_map: np.ndarray,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Convert a 2D float density map into a colorful RGB heatmap image.
    
    Args:
        density_map: 2D float array
        colormap: OpenCV colormap constant (default COLORMAP_JET)
        
    Returns:
        RGB heatmap image (uint8, shape HxWx3)
    """
    if density_map.max() > 0:
        norm_map = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-8)
    else:
        norm_map = density_map
    uint_map = (norm_map * 255.0).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(uint_map, colormap)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def overlay_density_on_image(
    image_rgb: np.ndarray,
    density_map: np.ndarray,
    alpha: float = 0.55
) -> np.ndarray:
    """
    Blend density heatmap over original RGB image.
    
    Args:
        image_rgb: RGB uint8 image (HxWx3)
        density_map: 2D float density map (HxW)
        alpha: Heatmap transparency factor
    """
    h, w = image_rgb.shape[:2]
    if density_map.shape != (h, w):
        density_map_resized = cv2.resize(density_map, (w, h), interpolation=cv2.INTER_CUBIC)
    else:
        density_map_resized = density_map
        
    heatmap_rgb = colorize_density_map(density_map_resized)
    blended = cv2.addWeighted(image_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)
    return blended
