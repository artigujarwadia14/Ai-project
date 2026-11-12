"""
Temporal Forecaster for 5-, 10-, and 15-Minute Lookahead Crowd Risk
Implements physically grounded kinematic projection using Taylor series rate-of-change extrapolation
and Greenshields-Underwood crowd fluid dynamics.

Methodology:
  1. Historical trend extraction: Computes first-order influx rate (dN/dt) and second-order acceleration (d^2N/dt^2).
  2. Future count & density projection:
       N_hat(t + dt) = N(t) + (dN/dt) * dt + 0.5 * (d^2N/dt^2) * dt^2
       D_hat(t + dt) = N_hat(t + dt) / Area
  3. Speed decay modeling via Greenshields continuum model:
       v_hat(t + dt) = v_free * max(0.05, 1.0 - D_hat(t + dt) / D_jam)
       where v_free = 1.35 m/s, D_jam = 5.4 persons/m^2.
  4. Future classification: Projected feature vectors are passed to trained RF/XGB models
     to forecast risk probability distributions 5, 10, and 15 minutes ahead.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import joblib

FEATURE_COLUMNS = [
    "crowd_density_p_m2",
    "avg_movement_speed_ms",
    "capacity_utilization",
    "congestion_index",
    "flow_entropy",
    "flow_variation",
    "crowd_growth_rate_per_min",
    "speed_variance"
]

CLASS_NAMES = ["LOW", "MODERATE", "HIGH", "CRITICAL"]


class TemporalRiskForecaster:
    """Predicts crowd risk conditions 5, 10, and 15 minutes into the future."""

    def __init__(
        self,
        models_dir: str = "models/risk_prediction",
        scene_area_m2: float = 375.0,
        safe_capacity: int = 300,
        v_free: float = 1.35,
        d_jam: float = 5.4
    ):
        self.p_models = Path(models_dir)
        self.scene_area_m2 = scene_area_m2
        self.safe_capacity = safe_capacity
        self.v_free = v_free
        self.d_jam = d_jam

        self.scaler = joblib.load(self.p_models / "scaler.joblib")
        self.rf_model = joblib.load(self.p_models / "rf_risk_model.joblib")
        self.xgb_model = joblib.load(self.p_models / "xgb_risk_model.joblib")

    def forecast_future_risk(
        self,
        current_features: Dict[str, float],
        inflow_rate_per_min: float,
        acceleration_per_min2: float = 0.0,
        model_name: str = "rf"
    ) -> Dict[str, Any]:
        """
        Generates 5-, 10-, and 15-minute lookahead risk forecasts.

        Args:
            current_features: Present crowd state dictionary
            inflow_rate_per_min: Net crowd arrival rate per minute (inflow - outflow)
            acceleration_per_min2: Acceleration in arrival rate (e.g. concert let-out or train arrival)
            model_name: 'rf' (Random Forest) or 'xgb' (XGBoost)
        """
        clf = self.rf_model if model_name.lower() == "rf" else self.xgb_model

        curr_count = float(current_features.get("crowd_count", 150.0))
        curr_density = float(current_features.get("crowd_density_p_m2", curr_count / self.scene_area_m2))
        curr_speed = float(current_features.get("avg_movement_speed_ms", 1.0))
        curr_entropy = float(current_features.get("flow_entropy", 1.0))
        curr_flow_var = float(current_features.get("flow_variation", 0.2))

        lookahead_minutes = [5, 10, 15]
        forecasts = {}

        for dt in lookahead_minutes:
            # 1. Project future crowd count via Taylor series
            proj_count = max(0.0, curr_count + inflow_rate_per_min * dt + 0.5 * acceleration_per_min2 * (dt ** 2))
            proj_density = max(0.0, proj_count / self.scene_area_m2)

            # 2. Greenshields speed decay
            density_ratio = proj_density / self.d_jam
            proj_speed = float(np.clip(self.v_free * max(0.03, 1.0 - density_ratio), 0.03, self.v_free))
            proj_speed_var = float(np.clip(0.1 + 0.08 * proj_density, 0.05, 0.6))

            # 3. Turbulence and entropy growth
            # As density increases toward jam density, flow becomes disorganized
            entropy_multiplier = 1.0 + 0.35 * max(0.0, proj_density - 2.0)
            proj_entropy = float(np.clip(curr_entropy * entropy_multiplier, 0.5, 3.2))
            proj_flow_var = float(np.clip(curr_flow_var * entropy_multiplier, 0.1, 1.4))

            # 4. Capacity utilization and congestion index
            proj_capacity_util = float(round(proj_count / float(self.safe_capacity), 3))
            proj_congestion_index = float(round(proj_density / max(0.1, proj_speed), 2))

            # Build feature vector
            feat_vec = np.array([[
                proj_density,
                proj_speed,
                proj_capacity_util,
                proj_congestion_index,
                proj_entropy,
                proj_flow_var,
                inflow_rate_per_min,
                proj_speed_var
            ]])

            feat_scaled = self.scaler.transform(feat_vec)
            pred_code = int(clf.predict(feat_scaled)[0])
            pred_probs = clf.predict_proba(feat_scaled)[0]

            risk_label = CLASS_NAMES[pred_code]

            forecasts[f"{dt}_min"] = {
                "horizon_minutes": dt,
                "projected_crowd_count": float(round(proj_count, 1)),
                "projected_density_p_m2": float(round(proj_density, 2)),
                "projected_speed_ms": float(round(proj_speed, 2)),
                "projected_capacity_utilization": proj_capacity_util,
                "predicted_risk_level": risk_label,
                "risk_code": pred_code,
                "class_probabilities": {
                    cname: float(round(prob, 4))
                    for cname, prob in zip(CLASS_NAMES, pred_probs)
                }
            }

        return {
            "current_state": {
                "count": curr_count,
                "density": curr_density,
                "speed": curr_speed
            },
            "inflow_rate_assumed": inflow_rate_per_min,
            "forecasts": forecasts,
            "scientific_note": (
                "Forecasts extrapolate current rate of influx using Greenshields speed-density decay. "
                "Forecasted risk levels reflect anticipated overcrowding assuming no intervention occurs."
            )
        }
