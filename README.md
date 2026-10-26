# Encrypted Flow Classification and QoS Prediction

This research prototype studies whether **early, payload-blind flow metadata** can identify broad traffic classes and help schedule many concurrent requests over a constrained shared link.

It targets a reviewable, solvable problem: classify a flow from packet timing, packet sizes, direction counts, and rates, map its predicted class to a transparent QoS profile, and measure how FIFO and priority scheduling affect request latency and deadline misses. HTTPS/VPN payloads are not decrypted or inspected.

> **Research honesty:** The bundled data generator creates synthetic traces for a reproducible software demo. Its metrics are not evidence of performance on ISCX VPN-nonVPN, QUIC, or real network traffic. Use properly labeled captures and capture/session-level splits before making dataset claims. Application labels do not intrinsically define QoS requirements; the profiles below are explicit experiment assumptions.

## Research question

Can a classifier using only the first observable flow statistics identify interactive, streaming, and bulk traffic well enough to improve deadline satisfaction when multiple requests compete for a bottleneck link? Evaluate the classifier and the scheduling outcome separately. Compare FIFO, static priority, model-guided priority, and the oracle-label upper bound.

## Architecture

```text
Packet metadata CSV / synthetic packets
              |
              v
     Bidirectional flow grouping
     + payload-blind statistics
              |
              v
   Logistic regression + Random Forest
              |
        class + confidence
              |
              v
     Explicit QoS profile mapping
              |
              v
 Concurrent request/link simulator
       |       |        |       |
      FIFO  Priority  Model   Oracle
              |
              v
  Per-class latency, throughput,
  deadline/SLA misses and JSON/CSV
```

### Flow features

The extractor groups packets bidirectionally by protocol and endpoint pair and splits reused tuples after a 30-second idle gap. It computes duration, packet count and size statistics, inter-arrival statistics, directional packet/byte counts, rates, and timing burstiness. It does not use payload bytes or ports as model features. The protocol and endpoint values are retained only for extraction/debugging.

### QoS assumptions

| Class | Scheduling priority | Deadline target | Latency target |
|---|---:|---:|---:|
| Interactive | 0 (highest) | 120 ms | 80 ms |
| Streaming | 1 | 1,000 ms | 500 ms |
| Bulk | 2 (lowest) | 10,000 ms | 5,000 ms |

These are demo parameters, not standards or claims about every application. The model-guided policy sends low-confidence predictions to the middle best-effort priority. The simulator is a non-preemptive, single-server bottleneck approximation; it is not a packet-accurate network emulator.

## Setup

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .
```

Optional packages:

```powershell
python -m pip install -e ".[cnn]"  # TensorFlow CNN extension
python -m pip install -e ".[pcap]" # Scapy for PCAP/PCAPNG input
python -m pip install -e ".[dashboard]" # Streamlit + Plotly review dashboard
```

## Demo workflow

Run from the repository root:

```powershell
traffic-qos generate
traffic-qos train
traffic-qos demo --bandwidth-mbps 12
```

If the console entry point is not on PATH, use `python -m traffic_qos.cli` after setting `PYTHONPATH=src` (PowerShell: `$env:PYTHONPATH = "src"`). For example:

```powershell
$env:PYTHONPATH = "src"
python -m traffic_qos.cli generate
python -m traffic_qos.cli train
python -m traffic_qos.cli demo --bandwidth-mbps 12
```

The demo replays a balanced sample of 30 flows by default, creates concurrent request arrivals, serializes request service over a shared link, and compares four scheduling policies. At a review, show `outputs/scheduler_summary.json` first, then `outputs/scheduler_results.csv` for individual requests and `outputs/evaluation.json` for classification metrics. Increase load with `--max-flows 60` or reduce bandwidth (for example 4 Mbps) for stress tests. Results are deterministic given the fixed seeds and defaults.

On the bundled synthetic default run, the current queue simulation typically shows deadline misses falling from about 32% with FIFO to about 12% with priority scheduling, with similar throughput. Treat this only as a pipeline demo result: the synthetic classes were deliberately generated with separable timing/size patterns, and the simulator is an abstraction. Re-run the commands to regenerate artifacts for the exact current environment and settings.

For a perfect-label upper bound on the synthetic labeled data:

```powershell
traffic-qos demo --oracle-labels --bandwidth-mbps 12
```

The normal demo uses the trained model. Oracle mode bypasses classification and should be described as an upper bound, not a deployable result.

### Optional 1D-CNN sequence workflow

The CNN is trained on the first 40 packets per flow. Each packet contributes direction (`+1/-1`), network-layer packet length, and inter-arrival time. Generate reproducible sequence data, then train and predict:

```powershell
python -m pip install -e ".[cnn]"
traffic-qos generate-sequences --output data/demo_sequences.csv
traffic-qos train --input data/demo_sequences.csv --cnn --model-only
traffic-qos predict --input data/demo_sequences.csv --model outputs/flow_cnn.keras --cnn --output outputs/cnn_predictions.csv
```

The CNN writes `outputs/flow_cnn.keras`, `outputs/cnn_preprocessing.json`, and `outputs/cnn_evaluation.json`. Its generated data is synthetic like the tabular demo; CNN metrics are not real dataset results.

## ISCX VPN-nonVPN and QUIC evaluation

The evaluation runner accepts already-extracted, labeled flow CSVs. Each row must have the statistical feature columns listed above, a label, and a capture/session grouping column. If a sequence-CNN evaluation is requested, include `packet_sequence` as JSON arrays of `[direction, packet_length, inter_arrival_seconds]`. Keep separate capture/session IDs per independent source capture; a random row split is deliberately not used.

Map source-specific column names and labels explicitly. Example after preparing an ISCX flow CSV:

```powershell
traffic-qos evaluate-dataset --input data/iscx_flows.csv --dataset iscx-vpn-nonvpn --label-column application --group-column capture_id --label-map '{"Browsing":"interactive","Streaming":"streaming","BulkTransfer":"bulk"}'
```

QUIC example:

```powershell
traffic-qos evaluate-dataset --input data/quic_flows.csv --dataset quic --label-column app_class --group-column trace_id --test-size 0.25
```

For the optional CNN, install `.[cnn]`, provide packet sequences, and add `--cnn`. Outputs include `outputs/<dataset>_evaluation.json` (metrics plus exact train/test groups and zero-overlap audit) and `outputs/<dataset>_test_predictions.csv`. The grouping column must have at least two values, and each class should occur in both sides of the split for meaningful per-class comparison. Raw dataset releases differ in schemas and labels, so adapt them into the documented flow contract and report the source label mapping. No dataset is downloaded automatically and no results are claimed until this command is run on actual captures.

## Review dashboard

Install and launch the local dashboard after generating the demo artifacts:

```powershell
python -m pip install -e ".[dashboard]"
traffic-qos generate
traffic-qos train
traffic-qos demo
streamlit run src/traffic_qos/dashboard.py
```

The dashboard shows classification scores and confusion matrices, capture split audits, scheduler latency/deadline comparisons, request-level records, and optional uploaded-flow predictions. To inspect a dataset run, choose its `<dataset>_evaluation.json` from the sidebar. Dashboard predictions use the tabular Random Forest/logistic model artifact.

## Using packet metadata from Wireshark

Export packet metadata to CSV with these columns: `timestamp,src_ip,dst_ip,src_port,dst_port,protocol,length`. Keep one packet per row; timestamps must be seconds (numeric or parseable as numeric). Then run:

```powershell
traffic-qos extract --input data/packets.csv --output data/flows.csv
```

For PCAP or PCAPNG directly (requires the optional Scapy dependency):

```powershell
python -m pip install -e ".[pcap]"
traffic-qos extract-pcap --input data/capture.pcapng --output data/flows.csv
traffic-qos extract-sequences --input data/capture.pcapng --pcap --output data/sequences.csv
```

Sequence extraction from packet-metadata CSV uses the same required columns:

```powershell
traffic-qos extract-sequences --input data/packets.csv --output data/sequences.csv
```

To train a CNN on extracted captures, add a consistent `label` column to each flow row and keep `packet_sequence` as JSON arrays of `[direction, length, inter_arrival_seconds]`, then use `train --cnn --model-only`. The PCAP reader only processes IPv4/IPv6 TCP/UDP packets and does not decrypt or inspect transport payload.

For training, add a `label` column to extracted flow rows and keep labels consistent within each flow. The included dataset contract expects the generated `flow_id` and the features listed under `traffic_qos.features.FEATURE_COLUMNS`. Already-aggregated flow CSVs can be passed directly to `train` and `demo` if they contain those feature columns. Do not train/evaluate by randomly splitting packets from the same capture across both sides; split by capture/session/host to reduce leakage.

## Commands

| Command | Action | Main outputs |
|---|---|---|
| `traffic-qos generate` | Generate synthetic labeled flow CSV | `data/demo_flows.csv` |
| `traffic-qos extract --input packets.csv` | Aggregate packet metadata into flow features | `data/flows.csv` |
| `traffic-qos extract-pcap --input capture.pcapng` | Aggregate TCP/UDP flows from PCAP/PCAPNG | `data/flows.csv` |
| `traffic-qos generate-sequences` | Generate synthetic packet-sequence dataset | `data/demo_sequences.csv` |
| `traffic-qos extract-sequences --input packets.csv` | Export early packet sequences from packet metadata | `data/sequences.csv` |
| `traffic-qos train --input sequences.csv --cnn --model-only` | Train optional TensorFlow 1D-CNN | `outputs/flow_cnn.keras`, `outputs/cnn_evaluation.json` |
| `traffic-qos evaluate-dataset --input flows.csv --dataset iscx-vpn-nonvpn` | Capture/session-separated ISCX or QUIC evaluation | `outputs/<dataset>_evaluation.json`, test predictions CSV |
| `traffic-qos train` | Compare Random Forest and logistic regression | `outputs/flow_classifier.joblib`, `outputs/evaluation.json` |
| `traffic-qos predict` | Add class/confidence predictions to flow rows | `outputs/predictions.csv` |
| `traffic-qos demo` | Run concurrent-flow scheduler comparison | `outputs/scheduler_results.csv`, `outputs/scheduler_summary.json` |

## Metrics and interpretation

Classification reports include accuracy, macro F1, per-class precision/recall/F1, and a confusion matrix. Scheduler summaries include mean/p95 request latency, deadline-miss rate, latency-target miss rate, throughput, and per-class measurements. Check both classifier quality and per-class scheduler metrics: a favorable aggregate can hide harms to one class. The oracle comparison estimates the headroom available if classification were perfect.

## Scope and research limits

The project provides offline adapters and a grouped evaluation path, but the ISCX/QUIC datasets are not distributed with this repository. Real dataset results require obtaining those datasets, preserving capture/session IDs, establishing a defensible label mapping, and running the grouped evaluation. The simulator is a research prototype and the generated data is synthetic; neither should be treated as production network control or real-world evidence.

