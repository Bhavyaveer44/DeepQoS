"""Map predicted application classes to transparent QoS targets."""

from __future__ import annotations

QOS_PROFILES = {
    "interactive": {"priority": 0, "deadline_ms": 120, "max_latency_ms": 80, "description": "latency-sensitive"},
    "streaming": {"priority": 1, "deadline_ms": 1_000, "max_latency_ms": 500, "description": "steady delivery"},
    "bulk": {"priority": 2, "deadline_ms": 10_000, "max_latency_ms": 5_000, "description": "throughput-oriented"},
}


def profile_for(label: str) -> dict:
    """Return a copy of the configured QoS profile; unknown labels use safe best-effort."""
    return dict(QOS_PROFILES.get(str(label), {"priority": 1, "deadline_ms": 1_000,
        "max_latency_ms": 1_000, "description": "best-effort fallback"}))


def apply_confidence_policy(label: str, confidence: float, threshold: float = 0.60) -> str:
    """Route uncertain predictions to best-effort (streaming) priority."""
    return str(label) if confidence >= threshold and str(label) in QOS_PROFILES else "streaming"

