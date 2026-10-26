"""Shared bottleneck-link simulation for concurrent request scheduling."""

from __future__ import annotations

import heapq
from pathlib import Path

import numpy as np
import pandas as pd

from .qos import apply_confidence_policy, profile_for


def make_requests(flows: pd.DataFrame, seed: int = 41, requests_per_flow: int = 3) -> pd.DataFrame:
    """Turn classified flows into concurrent variable-size requests."""
    rng = np.random.default_rng(seed)
    rows = []
    # Keep the default demo below link capacity but bursty enough to create queueing.
    base = {"interactive": (0.30, 6_000, 0.05), "streaming": (0.22, 70_000, 0.18), "bulk": (0.18, 240_000, 0.12)}
    for row in flows.itertuples(index=False):
        label = getattr(row, "predicted_class", getattr(row, "label", "streaming"))
        conf = float(getattr(row, "confidence", 1.0))
        interval, size, jitter = base.get(str(label), base["streaming"])
        start = float(rng.uniform(0, 8.0))
        for request_index in range(requests_per_flow):
            rows.append({
                "request_id": f"{getattr(row, 'flow_id', len(rows))}-{request_index}",
                "flow_id": str(getattr(row, "flow_id", len(rows))),
                "arrival_s": start + request_index * interval + float(rng.uniform(0, jitter)),
                "size_bytes": max(2_000, int(rng.lognormal(np.log(size), 0.55))),
                "traffic_class": str(label), "confidence": conf,
            })
    return pd.DataFrame(rows).sort_values(["arrival_s", "request_id"]).reset_index(drop=True)


def simulate(requests: pd.DataFrame, policy: str = "fifo", bandwidth_mbps: float = 12.0,
             confidence_threshold: float = 0.60) -> pd.DataFrame:
    """Non-preemptive single-link scheduler; request transfers are serialized.

    Policies: fifo, static_priority (request's supplied class), model_guided (same
    priority but confidence falls back to best-effort), and oracle (true class).
    """
    if bandwidth_mbps <= 0:
        raise ValueError("bandwidth_mbps must be positive")
    required = {"request_id", "arrival_s", "size_bytes", "traffic_class"}
    if required - set(requests.columns):
        raise ValueError(f"Requests missing columns: {', '.join(sorted(required - set(requests.columns)))}")
    ordered = requests.sort_values(["arrival_s", "request_id"]).to_dict(orient="records")
    waiting: list[tuple] = []
    results = []
    cursor = 0
    now = 0.0
    active_classes = set(requests["traffic_class"].astype(str))
    while cursor < len(ordered) or waiting:
        if not waiting and cursor < len(ordered):
            now = max(now, float(ordered[cursor]["arrival_s"]))
        while cursor < len(ordered) and float(ordered[cursor]["arrival_s"]) <= now:
            request = ordered[cursor]
            label = str(request["traffic_class"])
            confidence = float(request.get("confidence", 1.0))
            if policy == "fifo":
                priority = 0
            elif policy == "model_guided":
                chosen = apply_confidence_policy(label, confidence, confidence_threshold)
                priority = profile_for(chosen)["priority"]
            elif policy in {"static_priority", "oracle"}:
                priority = profile_for(label)["priority"]
            else:
                raise ValueError("policy must be fifo, static_priority, model_guided, or oracle")
            tie = float(request["arrival_s"]) if policy == "fifo" else cursor
            heapq.heappush(waiting, (priority, tie, cursor, request))
            cursor += 1
        _, _, _, request = heapq.heappop(waiting)
        service_s = float(request["size_bytes"]) * 8 / (bandwidth_mbps * 1_000_000)
        start = max(now, float(request["arrival_s"]))
        finish = start + service_s
        label = str(request["traffic_class"])
        profile = profile_for(label)
        latency_ms = (finish - float(request["arrival_s"])) * 1000
        results.append({**request, "service_s": service_s, "start_s": start, "finish_s": finish,
            "latency_ms": latency_ms, "deadline_ms": profile["deadline_ms"],
            "deadline_met": latency_ms <= profile["deadline_ms"], "priority_deadline_met": latency_ms <= profile["max_latency_ms"]})
        now = finish
    return pd.DataFrame(results)


def summarize(results: pd.DataFrame, policy: str) -> dict:
    if results.empty:
        return {"policy": policy, "requests": 0}
    duration = max(float(results["finish_s"].max() - results["arrival_s"].min()), 1e-9)
    summary = {
        "policy": policy, "requests": int(len(results)),
        "mean_latency_ms": float(results["latency_ms"].mean()),
        "p95_latency_ms": float(results["latency_ms"].quantile(.95)),
        "deadline_miss_rate": float(1 - results["deadline_met"].mean()),
        "priority_sla_miss_rate": float(1 - results["priority_deadline_met"].mean()),
        "throughput_mbps": float(results["size_bytes"].sum() * 8 / duration / 1_000_000),
        "class_metrics": {},
    }
    for label, group in results.groupby("traffic_class"):
        summary["class_metrics"][str(label)] = {
            "requests": int(len(group)), "mean_latency_ms": float(group["latency_ms"].mean()),
            "p95_latency_ms": float(group["latency_ms"].quantile(.95)),
            "deadline_miss_rate": float(1 - group["deadline_met"].mean()),
        }
    return summary


def compare_schedulers(requests: pd.DataFrame, bandwidth_mbps: float = 12.0,
                       confidence_threshold: float = .60) -> tuple[pd.DataFrame, dict]:
    outcomes = []
    summaries = []
    for policy in ("fifo", "static_priority", "model_guided", "oracle"):
        result = simulate(requests, policy, bandwidth_mbps, confidence_threshold)
        result["policy"] = policy
        outcomes.append(result)
        summaries.append(summarize(result, policy))
    return pd.concat(outcomes, ignore_index=True), {"bandwidth_mbps": bandwidth_mbps, "policies": summaries}


def save_comparison(requests: pd.DataFrame, output_dir: str | Path, bandwidth_mbps: float = 12.0) -> dict:
    import json
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results, summary = compare_schedulers(requests, bandwidth_mbps)
    results.to_csv(output / "scheduler_results.csv", index=False)
    (output / "scheduler_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

