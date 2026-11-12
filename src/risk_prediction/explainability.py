"""
Risk Explainability & Decision Breakdown Module
Provides global feature importance rankings and local instance-level explanations
for crowd risk predictions. Helps human crowd managers and academic reviewers
understand exactly WHY a specific risk level (LOW, MODERATE, HIGH, CRITICAL) was assigned.
"""

from typing import Dict, Any, List, Tuple
import numpy as np


class RiskExplainer:
    """Explains model risk outputs using feature deviation and decision thresholds."""

    # Normal baseline thresholds for safe crowd movement
    NORMAL_THRESHOLDS = {
        "crowd_density_p_m2": 1.5,           # Safe upper bound
        "avg_movement_speed_ms": 1.0,         # Normal pedestrian walking speed
        "capacity_utilization": 0.70,         # 70% capacity
        "congestion_index": 2.0,              # Fluid congestion factor
        "flow_entropy": 1.5,                  # Directional disorganization
        "flow_variation": 0.4,                # Flow variance
        "crowd_growth_rate_per_min": 10.0,    # Influx rate
    }

    FEATURE_DESCRIPTIONS = {
        "crowd_density_p_m2": "Crowd Density (persons/m²)",
        "avg_movement_speed_ms": "Average Movement Speed (m/s)",
        "capacity_utilization": "Venue Capacity Occupancy Ratio",
        "congestion_index": "Fluid Dynamic Congestion Ratio",
        "flow_entropy": "Directional Movement Entropy (Turbulence)",
        "flow_variation": "Flow Velocity Variation",
        "crowd_growth_rate_per_min": "Crowd Arrival Rate (persons/min)",
        "speed_variance": "Speed Dispersion Across Individuals"
    }

    def explain_instance(
        self,
        features: Dict[str, float],
        predicted_risk: str,
        feature_importances: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Generates local explanation for a single prediction.
        Identifies top contributing risk factors and feature deviations.
        """
        contributions = []

        for feat_name, normal_val in self.NORMAL_THRESHOLDS.items():
            if feat_name in features:
                curr_val = features[feat_name]
                imp = feature_importances.get(feat_name, 0.1)

                # Special case: for speed, LOWER is more dangerous
                if feat_name == "avg_movement_speed_ms":
                    if curr_val < normal_val:
                        severity = (normal_val - curr_val) / normal_val
                        impact_score = severity * imp * 100.0
                        contributions.append({
                            "feature": feat_name,
                            "name": self.FEATURE_DESCRIPTIONS[feat_name],
                            "current_value": curr_val,
                            "normal_threshold": normal_val,
                            "impact_score": float(round(impact_score, 2)),
                            "direction": "CRITICAL DROP (Speed restricted/stalled)" if curr_val < 0.3 else "RESTRICTED (Slow movement)",
                            "detail": f"Speed is {curr_val:.2f} m/s (below normal {normal_val:.2f} m/s), indicating severe physical constriction."
                        })
                else:
                    if curr_val > normal_val:
                        severity = (curr_val - normal_val) / normal_val
                        impact_score = severity * imp * 100.0
                        contributions.append({
                            "feature": feat_name,
                            "name": self.FEATURE_DESCRIPTIONS[feat_name],
                            "current_value": curr_val,
                            "normal_threshold": normal_val,
                            "impact_score": float(round(impact_score, 2)),
                            "direction": "ELEVATED ABOVE SAFE LIMIT",
                            "detail": f"{self.FEATURE_DESCRIPTIONS[feat_name]} is {curr_val:.2f} (exceeds safe baseline {normal_val:.2f})."
                        })

        # Sort by impact score descending
        contributions.sort(key=lambda x: x["impact_score"], reverse=True)
        top_factors = contributions[:3]

        if predicted_risk == "LOW":
            summary_text = "All observed parameters remain within normal safety bounds. Pedestrians maintain free walking speeds with low density."
        elif predicted_risk == "MODERATE":
            summary_text = f"Elevated crowd indicators detected, driven primarily by {top_factors[0]['name'] if top_factors else 'rising density'}. Pedestrian speed is declining."
        elif predicted_risk == "HIGH":
            primary_drivers = ", ".join([f["name"] for f in top_factors[:2]]) if top_factors else "density and bottlenecking"
            summary_text = f"Significant overcrowding detected driven by {primary_drivers}. Flow variation indicates onset of stop-and-go waves."
        else:  # CRITICAL
            primary_drivers = ", ".join([f["name"] for f in top_factors[:2]]) if top_factors else "extreme density and velocity collapse"
            summary_text = f"SEVERE CONGESTION ALERT: Multiple critical indicators breached ({primary_drivers}). Movement has virtually halted, risking crushing pressure."

        return {
            "predicted_risk": predicted_risk,
            "explanation_summary": summary_text,
            "top_driving_factors": top_factors,
            "all_deviations": contributions
        }
