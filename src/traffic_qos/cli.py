"""Command-line workflow for generating, training, predicting and simulating."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import make_synthetic_flows, make_synthetic_sequences
from .evaluation import evaluate_grouped, load_labeled_dataset
from .features import load_flow_csv, load_packet_csv, load_packet_csv_sequences, load_pcap, load_pcap_sequences
from .model import predict_cnn, predict_flows, train_baseline, train_cnn
from .simulator import make_requests, save_comparison


def main() -> None:
    parser = argparse.ArgumentParser(prog="traffic-qos", description="Encrypted-flow classification and QoS demo")
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate", help="Generate deterministic synthetic labeled flows")
    gen.add_argument("--output", default="data/demo_flows.csv")
    gen.add_argument("--flows-per-class", type=int, default=120)
    gen.add_argument("--seed", type=int, default=7)
    seqgen = commands.add_parser("generate-sequences", help="Generate labeled packet sequences for the CNN")
    seqgen.add_argument("--output", default="data/demo_sequences.csv")
    seqgen.add_argument("--flows-per-class", type=int, default=120)
    seqgen.add_argument("--seed", type=int, default=7)
    train = commands.add_parser("train", help="Train baseline classifiers and write evaluation metrics")
    train.add_argument("--input", default="data/demo_flows.csv")
    train.add_argument("--output-dir", default="outputs")
    train.add_argument("--cnn", action="store_true", help="Train optional TensorFlow CNN using packet_sequence column")
    train.add_argument("--model-only", action="store_true", help="Skip tabular model training and train only --cnn")
    predict = commands.add_parser("predict", help="Predict classes and confidence for flow feature CSV")
    predict.add_argument("--input", default="data/demo_flows.csv")
    predict.add_argument("--model", default="outputs/flow_classifier.joblib")
    predict.add_argument("--output", default="outputs/predictions.csv")
    predict.add_argument("--cnn", action="store_true", help="Use saved TensorFlow sequence CNN")
    demo = commands.add_parser("demo", help="Run multi-flow scheduler comparison end-to-end")
    demo.add_argument("--input", default="data/demo_flows.csv")
    demo.add_argument("--model", default="outputs/flow_classifier.joblib")
    demo.add_argument("--output-dir", default="outputs")
    demo.add_argument("--bandwidth-mbps", type=float, default=12.0)
    demo.add_argument("--max-flows", type=int, default=30, help="Maximum flows to replay (balanced across classes)")
    demo.add_argument("--oracle-labels", action="store_true", help="Use dataset labels as perfect classifier upper bound")
    packet = commands.add_parser("extract", help="Extract flow features from packet-metadata CSV")
    packet.add_argument("--input", required=True)
    packet.add_argument("--output", default="data/flows.csv")
    pcap = commands.add_parser("extract-pcap", help="Extract flow features directly from PCAP/PCAPNG")
    pcap.add_argument("--input", required=True)
    pcap.add_argument("--output", default="data/flows.csv")
    pcap.add_argument("--idle-timeout", type=float, default=30.0)
    seq_extract = commands.add_parser("extract-sequences", help="Export early packet sequences from packet CSV or PCAP")
    seq_extract.add_argument("--input", required=True)
    seq_extract.add_argument("--output", default="data/sequences.csv")
    seq_extract.add_argument("--pcap", action="store_true", help="Read PCAP/PCAPNG instead of packet metadata CSV")
    seq_extract.add_argument("--max-packets", type=int, default=40)
    seq_extract.add_argument("--idle-timeout", type=float, default=30.0)
    evaluate = commands.add_parser("evaluate-dataset", help="Evaluate labeled ISCX/QUIC-style flow CSV with capture-group holdout")
    evaluate.add_argument("--input", required=True)
    evaluate.add_argument("--dataset", required=True, choices=["iscx-vpn-nonvpn", "quic", "custom"])
    evaluate.add_argument("--label-column", default="label")
    evaluate.add_argument("--group-column", default="capture_id", help="Capture/session identifier; groups never cross split")
    evaluate.add_argument("--label-map", help="Optional JSON object mapping source labels into project classes")
    evaluate.add_argument("--test-size", type=float, default=.25)
    evaluate.add_argument("--seed", type=int, default=7)
    evaluate.add_argument("--output-dir", default="outputs")
    evaluate.add_argument("--cnn", action="store_true", help="Also evaluate 1D-CNN; input needs packet_sequence")
    args = parser.parse_args()

    try:
        _run(args)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.error(str(exc))


def _run(args: argparse.Namespace) -> None:
    if args.command == "generate":
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame = make_synthetic_flows(args.flows_per_class, args.seed)
        frame.to_csv(destination, index=False)
        print(f"Wrote {len(frame)} synthetic flows to {destination}")
    elif args.command == "extract":
        frame = load_packet_csv(args.input)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)
        print(f"Extracted {len(frame)} flows to {destination}")
    elif args.command == "extract-pcap":
        frame = load_pcap(args.input, idle_timeout=args.idle_timeout)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)
        print(f"Extracted {len(frame)} flows from PCAP to {destination}")
    elif args.command == "extract-sequences":
        loader = load_pcap_sequences if args.pcap else load_packet_csv_sequences
        frame = loader(args.input, idle_timeout=args.idle_timeout, max_packets=args.max_packets)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)
        print(f"Wrote {len(frame)} packet sequences to {destination}")
    elif args.command == "generate-sequences":
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        make_synthetic_sequences(args.flows_per_class, args.seed).to_csv(destination, index=False)
        print(f"Wrote labeled packet sequences to {destination}")
    elif args.command == "evaluate-dataset":
        mapping = json.loads(args.label_map) if args.label_map else None
        frame = load_labeled_dataset(args.input, args.group_column, args.label_column, args.dataset, mapping)
        metrics = evaluate_grouped(frame, args.output_dir, args.group_column, args.test_size,
                                   args.seed, args.dataset, args.cnn)
        print(json.dumps({"dataset": args.dataset, "split_strategy": metrics["split_strategy"],
            "train_rows": metrics["train_rows"], "test_rows": metrics["test_rows"],
            "train_groups": len(metrics["train_groups"]), "test_groups": len(metrics["test_groups"]),
            "group_overlap": metrics["group_overlap"], "models": {
                name: {"accuracy": values["accuracy"], "macro_f1": values["macro_f1"]}
                for name, values in metrics["models"].items()}}, indent=2))
    elif args.command == "train":
        frame = pd.read_csv(args.input)
        if not args.model_only:
            tabular = load_flow_csv(args.input)
            summary = train_baseline(tabular, args.output_dir)
            print(json.dumps({"selected_model": summary["selected_model"], "models": {
                name: {"accuracy": summary[name]["accuracy"], "macro_f1": summary[name]["macro_f1"]}
                for name in ("random_forest", "logistic_regression")}}, indent=2))
        if args.cnn:
            print("CNN:", json.dumps(train_cnn(frame, args.output_dir), indent=2))
        elif args.model_only:
            raise ValueError("--model-only requires --cnn")
    elif args.command == "predict":
        frame = pd.read_csv(args.input)
        result = predict_cnn(args.model, frame) if args.cnn else predict_flows(args.model, load_flow_csv(args.input))
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(destination, index=False)
        print(f"Wrote {len(result)} flow predictions to {destination}")
    elif args.command == "demo":
        frame = load_flow_csv(args.input)
        if args.oracle_labels:
            if "label" not in frame:
                raise ValueError("--oracle-labels requires a label column")
            frame["predicted_class"] = frame["label"].astype(str)
            frame["confidence"] = 1.0
        else:
            frame = predict_flows(args.model, frame)
        # A compact balanced slice gives a readable, non-saturated review demo.
        if len(frame) > args.max_flows:
            frame = frame.groupby("predicted_class", group_keys=False).head(max(1, args.max_flows // frame["predicted_class"].nunique()))
        requests = make_requests(frame)
        summary = save_comparison(requests, args.output_dir, args.bandwidth_mbps)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

