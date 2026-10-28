"""Run the user's own flows (CSV or PCAP) through the bottleneck-link scheduler.

Unlike `demo`, every flow in the input becomes exactly one request, sized from
the flow's real byte counts when the columns exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .qos import apply_confidence_policy, profile_for
from .simulator import simulate, summarize

SIZE_PAIRS = [("forward_bytes", "backward_bytes"), ("fwd_bytes", "bwd_bytes"),
              ("src_bytes", "dst_bytes"), ("bytes_fwd", "bytes_bwd")]
SIZE_SINGLE = ["total_bytes", "flow_bytes", "bytes", "byte_count"]
START_COLS = ["start_time", "start_s", "flow_start", "first_seen", "timestamp"]
FALLBACK_SIZE = {"interactive": 6_000, "streaming": 70_000, "bulk": 240_000}


def flow_sizes(raw: pd.DataFrame, classes: pd.Series) -> tuple[np.ndarray, str]:
    """Real per-flow bytes if the CSV has them, else a per-class typical size."""
    for a, b in SIZE_PAIRS:
        if a in raw and b in raw:
            return (raw[a].fillna(0) + raw[b].fillna(0)).to_numpy(float), f"{a}+{b}"
    for col in SIZE_SINGLE:
        if col in raw:
            return raw[col].fillna(0).to_numpy(float), col
    sizes = classes.map(lambda c: FALLBACK_SIZE.get(str(c), 70_000)).to_numpy(float)
    return sizes, "per-class default (no byte columns found)"


def arrival_times(raw: pd.DataFrame, n: int, mode: str, window_s: float, seed: int) -> np.ndarray:
    if mode == "auto":
        for col in START_COLS:
            if col in raw and pd.api.types.is_numeric_dtype(raw[col]):
                t = raw[col].to_numpy(float)
                return t - np.nanmin(t)
        mode = "burst"
    rng = np.random.default_rng(seed)
    if mode == "burst":
        return np.zeros(n)
    if mode == "even":
        return np.linspace(0, window_s, n, endpoint=False)
    if mode == "random":
        return rng.uniform(0, window_s, n)
    raise ValueError("arrival must be burst, even, random, or auto")


def build_requests(predicted: pd.DataFrame, raw: pd.DataFrame, arrival: str = "burst",
                   window_s: float = 10.0, seed: int = 41) -> tuple[pd.DataFrame, str]:
    if len(predicted) != len(raw):
        raise ValueError(f"Prediction rows ({len(predicted)}) and input rows ({len(raw)}) differ; "
                         "cannot align flow sizes.")
    n = len(predicted)
    sizes, size_source = flow_sizes(raw.reset_index(drop=True), predicted["predicted_class"].reset_index(drop=True))
    flow_ids = (predicted["flow_id"].astype(str).to_numpy() if "flow_id" in predicted
                else np.arange(n).astype(str))
    frame = pd.DataFrame({
        "request_id": [f"{fid}-{i}" for i, fid in enumerate(flow_ids)],
        "flow_id": flow_ids,
        "arrival_s": arrival_times(raw.reset_index(drop=True), n, arrival, window_s, seed),
        "size_bytes": np.maximum(sizes, 1_000).astype(int),
        "traffic_class": predicted["predicted_class"].astype(str).to_numpy(),
        "confidence": predicted["confidence"].astype(float).to_numpy(),
    })
    if "label" in predicted:
        frame["true_class"] = predicted["label"].astype(str).to_numpy()
    return frame, size_source


def run_schedule(predicted: pd.DataFrame, raw: pd.DataFrame, output_dir: str | Path,
                 bandwidth_mbps: float = 50.0, arrival: str = "burst", window_s: float = 10.0,
                 confidence_threshold: float = 0.60, seed: int = 41) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    requests, size_source = build_requests(predicted, raw, arrival, window_s, seed)

    policies = ["fifo", "static_priority", "model_guided"]
    runs = {p: simulate(requests, p, bandwidth_mbps, confidence_threshold) for p in policies}
    if "true_class" in requests:  # upper bound only when ground truth exists
        oracle_in = requests.assign(traffic_class=requests["true_class"], confidence=1.0)
        runs["oracle"] = simulate(oracle_in, "oracle", bandwidth_mbps, confidence_threshold)

    for name, df in runs.items():
        df["policy"] = name
    pd.concat(runs.values(), ignore_index=True).to_csv(out / "scheduler_results.csv", index=False)
    summary = {"bandwidth_mbps": bandwidth_mbps, "arrival": arrival, "size_source": size_source,
               "flows": int(len(requests)), "policies": [summarize(df, p) for p, df in runs.items()]}
    (out / "scheduler_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Per-flow answer: what service each of YOUR flows got.
    guided, fifo = runs["model_guided"], runs["fifo"].set_index("request_id")
    def _priority(row):
        chosen = apply_confidence_policy(row["traffic_class"], float(row["confidence"]), confidence_threshold)
        return profile_for(chosen)["priority"], chosen
    pr = guided.apply(_priority, axis=1)
    per_flow = pd.DataFrame({
        "flow_id": guided["flow_id"],
        "predicted_class": guided["traffic_class"],
        "confidence": guided["confidence"],
        "service_class_applied": [c for _, c in pr],
        "priority": [p for p, _ in pr],
        "size_bytes": guided["size_bytes"],
        "arrival_s": guided["arrival_s"],
        "wait_ms": (guided["start_s"] - guided["arrival_s"]) * 1000,
        "latency_ms": guided["latency_ms"],
        "deadline_ms": guided["deadline_ms"],
        "deadline_met": guided["deadline_met"],
        "fifo_latency_ms": fifo.loc[guided["request_id"], "latency_ms"].to_numpy(),
    })
    per_flow["latency_saved_ms"] = per_flow["fifo_latency_ms"] - per_flow["latency_ms"]
    per_flow.to_csv(out / "flow_qos.csv", index=False)
    return summary