"""Synthetic labeled flow generation for a deterministic, self-contained demo."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, extract_flows

CLASSES = ("interactive", "streaming", "bulk")


def make_synthetic_packets(flows_per_class: int = 120, seed: int = 7) -> pd.DataFrame:
    """Create packet metadata with class-specific timing/size patterns (not real captures)."""
    rng = np.random.default_rng(seed)
    packets = []
    patterns = {
        "interactive": (0.055, 180, 14, 0.018),
        "streaming": (0.025, 1050, 32, 0.11),
        "bulk": (0.008, 1250, 70, 0.045),
    }
    timestamp = 1_700_000_000.0
    for class_index, label in enumerate(CLASSES):
        for flow_index in range(flows_per_class):
            count = int(rng.integers(24, 55))
            mean_iat, mean_size, size_sd, burst = patterns[label]
            start = timestamp + class_index * 100_000 + flow_index * 10
            client = (f"10.{class_index}.{flow_index // 250}.{flow_index % 250 + 1}", 30_000 + flow_index)
            server = (f"192.0.2.{class_index + 1}", 443)
            t = start
            for packet_index in range(count):
                # Small burst modulation creates realistic variation without using class labels as features.
                t += max(0.0005, rng.exponential(mean_iat) * (0.65 if rng.random() < burst else 1.0))
                is_forward = packet_index == 0 or rng.random() < 0.56
                src, dst = (client, server) if is_forward else (server, client)
                size = int(np.clip(rng.normal(mean_size if is_forward else mean_size * 0.55, size_sd), 40, 1450))
                packets.append({
                    "timestamp": t, "src_ip": src[0], "dst_ip": dst[0],
                    "src_port": src[1], "dst_port": dst[1], "protocol": "TCP",
                    "length": size, "label": label,
                })
    return pd.DataFrame(packets)


def make_synthetic_flows(flows_per_class: int = 120, seed: int = 7) -> pd.DataFrame:
    packets = make_synthetic_packets(flows_per_class=flows_per_class, seed=seed)
    flows = extract_flows(packets.to_dict(orient="records"))
    # Stable, explicit columns simplify CSV exchange and model use.
    return flows[["flow_id", "label", "protocol", *FEATURE_COLUMNS]]


def make_synthetic_sequences(flows_per_class: int = 120, seed: int = 7, max_packets: int = 40) -> pd.DataFrame:
    """Create fixed-window packet sequences with schema [direction, size, delta-time]."""
    rng = np.random.default_rng(seed)
    rows = []
    patterns = {
        "interactive": (0.055, 180, 14, 0.018),
        "streaming": (0.025, 1050, 32, 0.11),
        "bulk": (0.008, 1250, 70, 0.045),
    }
    for label in CLASSES:
        mean_iat, mean_size, size_sd, burst = patterns[label]
        for flow_index in range(flows_per_class):
            length = int(rng.integers(max(8, max_packets // 2), max_packets + 1))
            sequence = []
            for packet_index in range(length):
                direction = 1.0 if packet_index == 0 or rng.random() < .56 else -1.0
                size = float(np.clip(rng.normal(mean_size if direction > 0 else mean_size * .55, size_sd), 40, 1450))
                iat = max(.0005, rng.exponential(mean_iat) * (.65 if rng.random() < burst else 1.0))
                sequence.append([direction, size, iat])
            rows.append({"flow_id": f"{label}-{flow_index:05d}", "label": label,
                         "packet_sequence": sequence})
    return pd.DataFrame(rows)

