"""Flow metadata extraction from packet records or a CSV export."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PACKET_COLUMNS = {"timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "length"}
FEATURE_COLUMNS = [
    "duration", "packet_count", "mean_packet_size", "std_packet_size",
    "min_packet_size", "max_packet_size", "mean_iat", "std_iat",
    "forward_packets", "backward_packets", "forward_bytes", "backward_bytes",
    "bytes_per_second", "packets_per_second", "burstiness",
]


def _flow_key(packet: dict) -> tuple:
    left = (str(packet["src_ip"]), int(packet["src_port"]))
    right = (str(packet["dst_ip"]), int(packet["dst_port"]))
    endpoints = tuple(sorted((left, right)))
    return str(packet["protocol"]).upper(), endpoints


def extract_flows(packets: Iterable[dict], idle_timeout: float = 30.0) -> pd.DataFrame:
    """Group bidirectional packet records into flows and compute payload-blind features.

    Input packet dictionaries require timestamp, src_ip, dst_ip, src_port, dst_port,
    protocol, and length. A label field is optional and must be constant per flow.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for raw in packets:
        missing = PACKET_COLUMNS - raw.keys()
        if missing:
            raise ValueError(f"Packet missing required fields: {', '.join(sorted(missing))}")
        packet = dict(raw)
        packet["timestamp"] = float(packet["timestamp"])
        packet["length"] = max(0, int(packet["length"]))
        groups[_flow_key(packet)].append(packet)

    rows = []
    for key, flow_packets in groups.items():
        flow_packets.sort(key=lambda item: item["timestamp"])
        # Split reused 5-tuples after a long idle period.
        sessions: list[list[dict]] = [[]]
        for packet in flow_packets:
            if sessions[-1] and packet["timestamp"] - sessions[-1][-1]["timestamp"] > idle_timeout:
                sessions.append([])
            sessions[-1].append(packet)
        for session_index, session in enumerate(sessions):
            first = session[0]
            origin = (str(first["src_ip"]), int(first["src_port"]))
            directions = [(str(p["src_ip"]), int(p["src_port"])) == origin for p in session]
            forward = [p for p, is_forward in zip(session, directions) if is_forward]
            backward = [p for p, is_forward in zip(session, directions) if not is_forward]
            times = np.asarray([p["timestamp"] for p in session], dtype=float)
            sizes = np.asarray([p["length"] for p in session], dtype=float)
            iats = np.diff(times)
            duration = max(float(times[-1] - times[0]), 1e-6)
            intervals = iats[iats > 0]
            burstiness = float(np.std(intervals) / (np.mean(intervals) + 1e-9)) if len(intervals) else 0.0
            labels = {p.get("label") for p in session if p.get("label") is not None}
            if len(labels) > 1:
                raise ValueError(f"Conflicting labels within flow {key}")
            row = {
                "flow_id": f"{hash(key) & 0xffffffff:08x}-{session_index}",
                "protocol": key[0], "duration": duration, "packet_count": len(session),
                "mean_packet_size": float(np.mean(sizes)), "std_packet_size": float(np.std(sizes)),
                "min_packet_size": float(np.min(sizes)), "max_packet_size": float(np.max(sizes)),
                "mean_iat": float(np.mean(iats)) if len(iats) else 0.0,
                "std_iat": float(np.std(iats)) if len(iats) else 0.0,
                "forward_packets": len(forward), "backward_packets": len(backward),
                "forward_bytes": sum(int(p["length"]) for p in forward),
                "backward_bytes": sum(int(p["length"]) for p in backward),
                "bytes_per_second": float(np.sum(sizes) / duration),
                "packets_per_second": float(len(session) / duration), "burstiness": burstiness,
            }
            if labels:
                row["label"] = labels.pop()
            rows.append(row)
    return pd.DataFrame(rows)


def load_packet_csv(path: str | Path) -> pd.DataFrame:
    """Extract flow features from a CSV with the required packet columns."""
    return extract_flows(pd.read_csv(path).to_dict(orient="records"))


def load_packet_csv_sequences(path: str | Path, idle_timeout: float = 30.0, max_packets: int = 40) -> pd.DataFrame:
    """Load packet metadata CSV and return early packet sequences."""
    return extract_sequences(pd.read_csv(path).to_dict(orient="records"), idle_timeout, max_packets)


def load_flow_csv(path: str | Path) -> pd.DataFrame:
    """Load already aggregated flow features, validating required model columns."""
    frame = pd.read_csv(path)
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Flow CSV missing feature columns: {', '.join(sorted(missing))}")
    return frame


def load_pcap(path: str | Path, idle_timeout: float = 30.0) -> pd.DataFrame:
    """Read classic PCAP/PCAPNG via Scapy and extract IP/TCP/UDP packet metadata.

    Transport payloads are never read. Network-layer length includes transport and
    encrypted payload lengths, as is standard for size-only traffic analysis.
    """
    try:
        from scapy.all import IP, IPv6, TCP, UDP, PcapReader
    except ImportError as exc:
        raise RuntimeError("PCAP support requires Scapy. Install with: pip install -e '.[pcap]'") from exc

    packets = []
    with PcapReader(str(path)) as reader:
        for packet in reader:
            network = packet.getlayer(IP) or packet.getlayer(IPv6)
            if network is None:
                continue
            transport = packet.getlayer(TCP) or packet.getlayer(UDP)
            if transport is None:
                continue
            protocol = "TCP" if packet.haslayer(TCP) else "UDP"
            packets.append({
                "timestamp": float(packet.time), "src_ip": network.src, "dst_ip": network.dst,
                "src_port": int(transport.sport), "dst_port": int(transport.dport),
                "protocol": protocol, "length": int(len(network)),
            })
    if not packets:
        raise ValueError("No IPv4/IPv6 TCP or UDP packets found in capture")
    return extract_flows(packets, idle_timeout=idle_timeout)


def extract_sequences(packets: Iterable[dict], idle_timeout: float = 30.0, max_packets: int = 40) -> pd.DataFrame:
    """Build per-flow early packet sequences as [direction, length, inter-arrival]."""
    if max_packets < 1:
        raise ValueError("max_packets must be positive")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for raw in packets:
        missing = PACKET_COLUMNS - raw.keys()
        if missing:
            raise ValueError(f"Packet missing required fields: {', '.join(sorted(missing))}")
        packet = dict(raw)
        packet["timestamp"] = float(packet["timestamp"])
        packet["length"] = max(0, int(packet["length"]))
        groups[_flow_key(packet)].append(packet)
    rows = []
    for key, flow_packets in groups.items():
        flow_packets.sort(key=lambda item: item["timestamp"])
        sessions: list[list[dict]] = [[]]
        for packet in flow_packets:
            if sessions[-1] and packet["timestamp"] - sessions[-1][-1]["timestamp"] > idle_timeout:
                sessions.append([])
            sessions[-1].append(packet)
        for session_index, session in enumerate(sessions):
            origin = (str(session[0]["src_ip"]), int(session[0]["src_port"]))
            previous_time = session[0]["timestamp"]
            sequence = []
            for index, packet in enumerate(session[:max_packets]):
                direction = 1.0 if (str(packet["src_ip"]), int(packet["src_port"])) == origin else -1.0
                delta = 0.0 if index == 0 else max(0.0, packet["timestamp"] - previous_time)
                sequence.append([direction, float(packet["length"]), delta])
                previous_time = packet["timestamp"]
            labels = {packet.get("label") for packet in session if packet.get("label") is not None}
            if len(labels) > 1:
                raise ValueError(f"Conflicting labels within flow {key}")
            row = {"flow_id": f"{hash(key) & 0xffffffff:08x}-{session_index}",
                   "packet_sequence": sequence, "protocol": key[0]}
            if labels:
                row["label"] = next(iter(labels))
            rows.append(row)
    return pd.DataFrame(rows)


def load_pcap_sequences(path: str | Path, idle_timeout: float = 30.0, max_packets: int = 40) -> pd.DataFrame:
    """Read PCAP/PCAPNG and return unlabeled per-flow packet sequences."""
    try:
        from scapy.all import IP, IPv6, TCP, UDP, PcapReader
    except ImportError as exc:
        raise RuntimeError("PCAP support requires Scapy. Install with: pip install -e '.[pcap]'") from exc
    packets = []
    with PcapReader(str(path)) as reader:
        for packet in reader:
            network = packet.getlayer(IP) or packet.getlayer(IPv6)
            transport = packet.getlayer(TCP) or packet.getlayer(UDP)
            if network is None or transport is None:
                continue
            packets.append({"timestamp": float(packet.time), "src_ip": network.src, "dst_ip": network.dst,
                "src_port": int(transport.sport), "dst_port": int(transport.dport),
                "protocol": "TCP" if packet.haslayer(TCP) else "UDP", "length": int(len(network))})
    if not packets:
        raise ValueError("No IPv4/IPv6 TCP or UDP packets found in capture")
    return extract_sequences(packets, idle_timeout, max_packets)

