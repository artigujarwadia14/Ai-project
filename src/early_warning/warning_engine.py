"""
Phase 8: Early-Warning & Decision Support Recommendation Engine
Translates Computer Vision crowd metrics and Machine Learning risk classifications
into actionable, explainable operational alerts and crowd safety recommendations.

Academic & Ethical Guardrail:
The system is explicitly designed as a human-in-the-loop decision-support tool.
It does NOT claim to guarantee automated stampede prevention, but provides early indicators
enabling proactive intervention before physical crushing thresholds occur.
"""

from typing import Dict, Any, List


class EarlyWarningEngine:
    """Generates tiered warnings and context-specific preventive recommendations."""

    ALERT_LEVELS = {
        "LOW": {
            "status": "NORMAL CONDITIONS",
            "color": "#28a745", # Green
            "badge": "🟢 LOW RISK",
            "description": "Crowd movement is free-flowing and unconstrained. Density is well below safe capacity limits.",
            "operational_posture": "Routine surveillance and passive monitoring."
        },
        "MODERATE": {
            "status": "ADVISORY - INCREASING CONGESTION",
            "color": "#ffc107", # Amber
            "badge": "🟡 MODERATE RISK",
            "description": "Pedestrian density is rising and forward walking speeds are beginning to decline. Entry rates exceed baseline outflow.",
            "operational_posture": "Heighten active monitoring, inspect bottleneck points, prepare auxiliary gates."
        },
        "HIGH": {
            "status": "WARNING - SIGNIFICANT OVERCROWDING",
            "color": "#fd7e14", # Orange
            "badge": "🟠 HIGH RISK",
            "description": "Significant overcrowding indicators detected. Influx is outpacing capacity, with stop-and-go waves forming.",
            "operational_posture": "Enact proactive entry metering, open secondary release routes, deploy ground marshals."
        },
        "CRITICAL": {
            "status": "EMERGENCY - SEVERE COMPRESSION RISK",
            "color": "#dc3545", # Red
            "badge": "🔴 CRITICAL RISK",
            "description": "Multiple severe congestion indicators breached. Density exceeds crush threshold with forward velocity near zero. Extreme danger of compressive collapse.",
            "operational_posture": "IMMEDIATE EMERGENCY ACTION: Halt all inflows, open all emergency exits, broadcast directional announcements."
        }
    }

    def evaluate_warnings(
        self,
        risk_level: str,
        features: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Synthesizes model risk level with rule-based domain logic
        to generate granular, explainable preventive recommendations.
        """
        risk_info = self.ALERT_LEVELS.get(risk_level, self.ALERT_LEVELS["LOW"])
        recommendations = []
        action_triggers = []

        density = float(features.get("crowd_density_p_m2", 0.0))
        speed = float(features.get("avg_movement_speed_ms", 1.0))
        capacity = float(features.get("capacity_utilization", 0.5))
        entropy = float(features.get("flow_entropy", 1.0))
        growth = float(features.get("crowd_growth_rate_per_min", 0.0))

        # Rule 1: Capacity Overutilization
        if capacity >= 1.10:
            action_triggers.append("Capacity exceeded 110%")
            recommendations.append("HALT INTAKE: Immediately close all primary entrance turnstiles; the area has exceeded maximum physical holding capacity.")
        elif capacity >= 0.85:
            action_triggers.append("Capacity over 85%")
            recommendations.append("THROTTLE INFLOW: Initiate staggered pulse-entry metering at upstream ticket gates to control inward rush.")

        # Rule 2: Velocity Stagnation / Bottleneck
        if speed <= 0.25 and density >= 2.5:
            action_triggers.append("Pedestrian velocity collapsed below 0.25 m/s")
            recommendations.append("CLEAR BOTTLENECK: Ground marshals must immediately inspect narrowings, turns, and exit pinch points to remove physical obstructions.")
        elif speed <= 0.55:
            action_triggers.append("Walking speed slowed below 0.55 m/s")
            recommendations.append("EXPAND DISCHARGE: Open all secondary egress lanes and wide-angle doors to restore forward momentum.")

        # Rule 3: Cross-Flow Turbulence & Entropy
        if entropy >= 2.2:
            action_triggers.append("Directional movement entropy > 2.2 (Turbulent flow)")
            recommendations.append("SEPARATE FLOWS: Install physical stanchions or rope barriers to enforce unidirectional tidal flow and eliminate dangerous counter-flow shearing.")

        # Rule 4: Rapid Influx Surge
        if growth >= 25.0:
            action_triggers.append("Surge arrival rate > 25 persons/min")
            recommendations.append("DIVERSIFY ROUTING: Broadcast PA announcements advising incoming crowds to utilize alternate entrances or peripheral holding areas.")

        # Default routine recommendations if low risk
        if not recommendations:
            recommendations.append("Maintain continuous automated visual surveillance of designated bottleneck zones.")
            recommendations.append("Ensure exit corridors and emergency passages remain unobstructed.")

        return {
            "risk_level": risk_level,
            "badge": risk_info["badge"],
            "status_title": risk_info["status"],
            "color_hex": risk_info["color"],
            "description": risk_info["description"],
            "operational_posture": risk_info["operational_posture"],
            "action_triggers": action_triggers,
            "actionable_recommendations": recommendations,
            "decision_support_disclaimer": (
                "Decision Support System Notice: These automated recommendations are decision-support guidelines "
                "intended to assist venue safety officers. The system monitors observable visual parameters and does not "
                "replace professional crowd management assessment."
            )
        }
