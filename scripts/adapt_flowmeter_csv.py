"""Convert a CICFlowMeter-style flow CSV (87 columns) into this project's flow contract.

Run from the repository root:

    python scripts/adapt_flowmeter_csv.py --input data/raw/iscx/raw_data.csv --output data/flows_all.csv

What it does:
  * keeps only flows whose `ProtocolName` (application) is listed in APP_TO_CLASS
    and maps them to interactive / streaming / bulk (edit APP_TO_CLASS!)
  * computes the 15 FEATURE_COLUMNS from the CICFlowMeter columns
  * removes inf/NaN rows, reverse-direction duplicates and very short flows
  * creates a `capture_id` group column so evaluate-dataset can split by group
  * caps flows per class so training stays fast

The application label and the dataset's own columns (IPs, ports, Label, ProtocolName)
are NOT written as features.
"""

from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from traffic_qos.features import FEATURE_COLUMNS

# ProtocolName (nDPI application, upper-cased) -> project class.
# This is a DRAFT. List the real names first (see the value_counts command),
# then edit. Applications not listed here are dropped on purpose.
APP_TO_CLASS = {
    # interactive: chat, calls, remote desktop
    "SKYPE": "interactive", "WHATSAPP": "interactive", "TELEGRAM": "interactive",
    "TEAMVIEWER": "interactive", "MSN": "interactive", "SLACK": "interactive",
    "TEAMSPEAK": "interactive", "VIBER": "interactive",
    # streaming: audio/video playback
    "YOUTUBE": "streaming", "NETFLIX": "streaming", "SPOTIFY": "streaming",
    "TWITCH": "streaming", "DEEZER": "streaming",
    # bulk: file sync, downloads, torrents
    "DROPBOX": "bulk", "MS_ONE_DRIVE": "bulk", "APPLE_ICLOUD": "bulk",
    "BITTORRENT": "bulk", "TORRENT": "bulk", "FTP_DATA": "bulk",
    "OWNCLOUD": "bulk", "UBUNTUONE": "bulk", "WINDOWS_UPDATE": "bulk",
}

TEXT_COLS = {"Flow.ID", "Source.IP", "Destination.IP", "Timestamp", "ProtocolName"}
USECOLS = [
    "Flow.ID", "Source.IP", "Destination.IP", "Protocol", "Timestamp", "ProtocolName",
    "Flow.Duration", "Total.Fwd.Packets", "Total.Backward.Packets",
    "Total.Length.of.Fwd.Packets", "Total.Length.of.Bwd.Packets",
    "Min.Packet.Length", "Max.Packet.Length", "Packet.Length.Mean", "Packet.Length.Std",
    "Flow.Bytes.s", "Flow.Packets.s", "Flow.IAT.Mean", "Flow.IAT.Std",
]


def read_mapped(path: Path, chunksize: int):
    """Stream the big CSV, keeping only rows whose application maps to a class."""
    kept: list[pd.DataFrame] = []
    unmapped = pd.Series(dtype="float64")
    total = 0
    for chunk in pd.read_csv(path, usecols=USECOLS, chunksize=chunksize, low_memory=False):
        total += len(chunk)
        names = chunk["ProtocolName"].astype(str).str.strip().str.upper()
        label = names.map(APP_TO_CLASS)
        unmapped = unmapped.add(names[label.isna()].value_counts(), fill_value=0)
        chunk = chunk.assign(label=label)
        kept.append(chunk[chunk["label"].notna()])
        print(f"  scanned {total:,} rows, kept {sum(len(k) for k in kept):,}", flush=True)
    return kept, unmapped, total


def detect_scale(df: pd.DataFrame) -> tuple[str, float]:
    """Find the time unit of Flow.Duration by comparing against Flow.Bytes.s."""
    sample = df[(df["Flow.Duration"] > 0) & (df["Flow.Bytes.s"] > 0)].head(200_000)
    if sample.empty:
        return "microseconds", 1e6
    total_bytes = sample["Total.Length.of.Fwd.Packets"] + sample["Total.Length.of.Bwd.Packets"]
    best = ("microseconds", 1e6)
    best_err = np.inf
    for name, scale in (("microseconds", 1e6), ("milliseconds", 1e3), ("seconds", 1.0)):
        ratio = (total_bytes / (sample["Flow.Duration"] / scale)) / sample["Flow.Bytes.s"]
        median = ratio[ratio > 0].median()
        err = abs(np.log(median)) if np.isfinite(median) and median > 0 else np.inf
        if err < best_err:
            best, best_err = (name, scale), err
    return best


def make_groups(df: pd.DataFrame, mode: str, n_groups: int) -> pd.Series:
    """Group id per flow so related flows never land on both sides of the split."""
    if mode == "hour":
        stamps = pd.to_datetime(df["Timestamp"], errors="coerce")
        return stamps.dt.strftime("%Y-%m-%d-%H")
    # 'client': the less frequent endpoint of each flow is treated as the client host.
    # Servers/proxies appear in very many flows, clients in fewer. Hosts are hashed into
    # n_groups buckets so the group list stays small and reproducible.
    src = df["Source.IP"].astype(str)
    dst = df["Destination.IP"].astype(str)
    counts = pd.concat([src, dst]).value_counts()
    src_n, dst_n = src.map(counts), dst.map(counts)
    host = src.where((src_n < dst_n) | ((src_n == dst_n) & (src < dst)), dst)
    lookup = {h: f"host-bucket-{zlib.crc32(h.encode()) % n_groups:03d}" for h in host.unique()}
    return host.map(lookup)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt a CICFlowMeter-style flow CSV")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/flows_all.csv")
    parser.add_argument("--min-packets", type=int, default=5)
    parser.add_argument("--max-per-class", type=int, default=100_000, help="0 = no cap")
    parser.add_argument("--group-mode", choices=["client", "hour"], default="client")
    parser.add_argument("--n-groups", type=int, default=200, help="Host buckets for --group-mode client")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--keep-duplicates", action="store_true", help="Keep reverse-direction duplicate flows")
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        sys.exit(f"Input not found: {path}")
    if Path(args.output).resolve() == path.resolve():
        sys.exit("--output must be a different file from --input")

    print(f"Reading {path} ...")
    kept, unmapped, total = read_mapped(path, args.chunksize)
    kept_rows = sum(len(k) for k in kept)
    print(f"\nTop applications NOT mapped to a class (edit APP_TO_CLASS to include some):")
    print(unmapped.sort_values(ascending=False).head(40).astype(int).to_string())
    if kept_rows == 0:
        sys.exit("\nNo rows matched APP_TO_CLASS. Edit it using the application names above and rerun.")

    df = pd.concat(kept, ignore_index=True)
    numeric = [c for c in USECOLS if c not in TEXT_COLS]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    n0 = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric)
    n_bad = n0 - len(df)

    n_dup = 0
    if not args.keep_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=["Flow.ID", "Timestamp"])
        n_dup = before - len(df)

    packets = df["Total.Fwd.Packets"] + df["Total.Backward.Packets"]
    df = df[(packets >= args.min_packets) & (df["Flow.Duration"] > 0)].copy()
    if df.empty:
        sys.exit("No flows left after cleaning.")

    unit, scale = detect_scale(df)
    print(f"\nDetected Flow.Duration unit: {unit}")

    groups = make_groups(df, args.group_mode, args.n_groups)
    valid = groups.notna()
    df, groups = df[valid], groups[valid]

    fwd, bwd = df["Total.Fwd.Packets"], df["Total.Backward.Packets"]
    iat_mean, iat_std = df["Flow.IAT.Mean"] / scale, df["Flow.IAT.Std"] / scale
    out = pd.DataFrame({
        "duration": df["Flow.Duration"] / scale,
        "packet_count": fwd + bwd,
        "mean_packet_size": df["Packet.Length.Mean"],
        "std_packet_size": df["Packet.Length.Std"],
        "min_packet_size": df["Min.Packet.Length"],
        "max_packet_size": df["Max.Packet.Length"],
        "mean_iat": iat_mean,
        "std_iat": iat_std,
        "forward_packets": fwd,
        "backward_packets": bwd,
        "forward_bytes": df["Total.Length.of.Fwd.Packets"],
        "backward_bytes": df["Total.Length.of.Bwd.Packets"],
        "bytes_per_second": df["Flow.Bytes.s"],
        "packets_per_second": df["Flow.Packets.s"],
        "burstiness": iat_std / (iat_mean + 1e-9),
    })[FEATURE_COLUMNS]
    out["protocol"] = df["Protocol"].map({6: "TCP", 17: "UDP"}).fillna("OTHER")
    out["label"] = df["label"]
    out["capture_id"] = groups

    if args.max_per_class > 0:
        out = out.sample(frac=1, random_state=args.seed).groupby("label").head(args.max_per_class)
    out = out.reset_index(drop=True)
    out.insert(0, "flow_id", [f"flow{i}" for i in range(len(out))])

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(destination, index=False)

    print(f"\nScanned {total:,} rows; kept {kept_rows:,} mapped; dropped {n_bad:,} inf/NaN, {n_dup:,} duplicates.")
    print(f"Wrote {len(out):,} flows to {destination}")
    summary = out.groupby("label").agg(flows=("flow_id", "count"), groups=("capture_id", "nunique"))
    print(summary.to_string())
    thin = summary[summary["groups"] < 2]
    if not thin.empty:
        print(f"\nWARNING: classes with fewer than 2 groups: {', '.join(thin.index)}")


if __name__ == "__main__":
    main()