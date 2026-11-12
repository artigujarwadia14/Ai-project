"""
Crowd Risk Scenario Generator (Synthetic Benchmark Dataset)
Generates realistic, physically grounded crowd risk simulation records based on established
Crowd Safety Engineering standards (John Fruin's Levels of Service A-F, Keith Still's Crowd Safety Analysis,
and Dirk Helbing's crowd turbulence equations).

CRITICAL SCIENTIFIC DISTINCTION:
  - This dataset contains synthetic simulation scenarios with explicit ground truth risk levels.
  - Public computer vision datasets (NWPU-Crowd, UCF-QNRF) contain image counts and tracking points,
    NOT real-world stampede disaster labels.
  - This simulation dataset is used strictly to train and benchmark risk classification algorithms
    without fabricating real-world disaster claims.

Risk Class Definitions:
  - LOW (Class 0): Normal free-flowing crowd. Density < 1.5 p/m^2, Speed > 1.0 m/s, Capacity < 60%.
  - MODERATE (Class 1): Increasing density, restricted walking speeds. Density 1.5 - 2.8 p/m^2, Speed 0.6 - 1.0 m/s, Capacity 60% - 85%.
  - HIGH (Class 2): Overcrowding indicators, stop-and-go waves. Density 2.8 - 4.5 p/m^2, Speed 0.25 - 0.6 m/s, Capacity 85% - 110%.
  - CRITICAL (Class 3): Severe crushing congestion, complete blockage. Density > 4.5 p/m^2, Speed < 0.25 m/s, Capacity > 110%.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


def generate_crowd_risk_scenarios(
    num_samples_per_class: int = 500,
    random_seed: int = 42,
    output_csv: str = "results/predictions/simulated_risk_scenarios.csv"
) -> pd.DataFrame:
    """
    Generates a balanced, multivariate synthetic dataset spanning LOW, MODERATE, HIGH, and CRITICAL risk levels.
    Incorporates physical correlations (e.g. as density increases, speed drops non-linearly).
    """
    np.random.seed(random_seed)
    records = []

    classes = ["LOW", "MODERATE", "HIGH", "CRITICAL"]

    for class_idx, class_name in enumerate(classes):
        for i in range(num_samples_per_class):
            if class_name == "LOW":
                # Unconstrained laminar flow
                density = float(np.random.uniform(0.1, 1.45))
                # Fruin Greenshields model: speed inversely related to density
                speed = float(np.clip(np.random.normal(1.35, 0.18) - 0.2 * density, 0.9, 1.8))
                speed_var = float(np.random.uniform(0.05, 0.22))
                flow_entropy = float(np.random.uniform(0.4, 1.4))
                flow_variation = float(np.random.uniform(0.05, 0.35))
                growth_rate = float(np.random.normal(0.0, 5.0))
                capacity_util = float(np.clip(density / 3.0, 0.05, 0.58))
                
            elif class_name == "MODERATE":
                # Increasing density, pedestrian maneuvering restricted
                density = float(np.random.uniform(1.5, 2.75))
                speed = float(np.clip(np.random.normal(0.85, 0.12) - 0.15 * (density - 1.5), 0.55, 1.05))
                speed_var = float(np.random.uniform(0.15, 0.35))
                flow_entropy = float(np.random.uniform(1.2, 1.95))
                flow_variation = float(np.random.uniform(0.30, 0.60))
                growth_rate = float(np.random.uniform(5.0, 25.0)) # rising attendance
                capacity_util = float(np.random.uniform(0.60, 0.84))

            elif class_name == "HIGH":
                # Significant overcrowding, stop-and-go flow, intermittent compression
                density = float(np.random.uniform(2.8, 4.45))
                speed = float(np.clip(np.random.normal(0.42, 0.08) - 0.12 * (density - 2.8), 0.22, 0.58))
                speed_var = float(np.random.uniform(0.25, 0.50))
                flow_entropy = float(np.random.uniform(1.85, 2.55))
                flow_variation = float(np.random.uniform(0.55, 0.85))
                growth_rate = float(np.random.uniform(15.0, 45.0))
                capacity_util = float(np.random.uniform(0.85, 1.10))

            else:  # CRITICAL
                # Physical crush zone, shockwaves, involuntary movement, near-zero forward progress
                density = float(np.random.uniform(4.5, 7.5))
                speed = float(np.clip(np.random.uniform(0.02, 0.22), 0.01, 0.25))
                speed_var = float(np.random.uniform(0.02, 0.15))
                flow_entropy = float(np.random.uniform(2.4, 3.2)) # turbulent
                flow_variation = float(np.random.uniform(0.80, 1.45))
                growth_rate = float(np.random.uniform(30.0, 75.0))
                capacity_util = float(np.random.uniform(1.12, 1.65))

            # Venue area baseline 375 m^2
            scene_area = 375.0
            crowd_count = float(round(density * scene_area, 1))
            density_growth_rate = float(round(growth_rate / scene_area, 4))
            congestion_index = float(round(density / max(0.1, speed), 2))

            records.append({
                "scenario_id": f"SCEN_{class_name[:3]}_{i+1:04d}",
                "crowd_count": crowd_count,
                "crowd_density_p_m2": float(round(density, 3)),
                "crowd_growth_rate_per_min": float(round(growth_rate, 2)),
                "density_growth_rate_per_min": density_growth_rate,
                "avg_movement_speed_ms": float(round(speed, 3)),
                "speed_variance": float(round(speed_var, 3)),
                "flow_entropy": float(round(flow_entropy, 3)),
                "flow_variation": float(round(flow_variation, 3)),
                "capacity_utilization": float(round(capacity_util, 3)),
                "congestion_index": congestion_index,
                "risk_label": class_name,
                "risk_code": class_idx
            })

    df = pd.DataFrame(records)
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    out_p = Path(output_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_p, index=False)
    print(f"Generated {len(df)} synthetic crowd risk scenarios saved to {output_csv}")
    return df


if __name__ == "__main__":
    generate_crowd_risk_scenarios()
