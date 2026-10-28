# DeepQoS: Encrypted Traffic Classification and QoS Scheduling

A research prototype that asks one question: **can packet timing, sizes and directions alone (no payload) identify what kind of traffic a flow carries, well enough to schedule many competing requests over a shared link more fairly?**

HTTPS and VPN payloads are never decrypted or read. The project classifies each flow from its early metadata, maps the predicted class to a QoS profile, and measures how different scheduling policies change latency and deadline misses.

> **Research honesty.** The built-in data generator produces *synthetic* traffic so the pipeline can be demonstrated anywhere. Its metrics say nothing about performance on ISCX VPN-nonVPN, QUIC or real networks. The QoS profiles are explicit experiment assumptions, not standards. Only capture-separated evaluation on real datasets supports dataset claims.

## How it works

```text
 Packet metadata CSV / PCAP / synthetic packets
                    |
                    v
        Bidirectional flow grouping
        + 15 payload-blind features         (features.py)
                    |
                    v
   Random forest / logistic regression      (model.py)
   or optional 1D-CNN on packet sequences
                    |
            class + confidence
                    |
                    v
      QoS profile + confidence fallback     (qos.py)
                    |
                    v
     Single-link request simulator          (simulator.py, schedule_flows.py)
        |        |         |        |
      FIFO    Static    Model-    Oracle
              priority  guided
                    |
                    v
   Latency, deadline misses, throughput     (JSON / CSV / dashboard)
```

1. **Extract.** Packets are grouped into bidirectional flows by protocol and endpoint pair. A new flow starts after a 30-second idle gap. Each flow yields duration, packet-size and inter-arrival statistics, forward/backward packet and byte counts, rates and burstiness. Ports and IPs are not model features.
2. **Classify.** A random forest and a logistic regression are trained and the one with the higher macro F1 is saved. An optional TensorFlow 1D-CNN learns from the first 40 packets of each flow as `[direction, length, inter-arrival]`.
3. **Map to QoS.** Each class has a priority, a deadline and a latency target. Predictions below the confidence threshold (default 0.60) fall back to middle-priority best-effort.
4. **Schedule.** Requests compete for one bottleneck link. Four policies are compared: FIFO, static priority, model-guided priority, and an oracle that uses true labels as an upper bound.
5. **Review.** Results are written to JSON and CSV and explored in a Streamlit dashboard.

### Traffic classes and QoS assumptions

| Class | Priority | Deadline | Latency target | Typical use |
|---|---:|---:|---:|---|
| `interactive` | 0 (highest) | 120 ms | 80 ms | browsing, chat, calls |
| `streaming` | 1 | 1,000 ms | 500 ms | video and audio |
| `bulk` | 2 (lowest) | 10,000 ms | 5,000 ms | downloads, backups |

These are tunable demo parameters in `src/traffic_qos/qos.py`. The simulator is a non-preemptive single-server queue, not a packet-accurate network emulator.

## Quick start

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dashboard]"
```

On macOS or Linux, activate with `source .venv/bin/activate`.

Run the full demo:

```powershell
traffic-qos generate
traffic-qos train
traffic-qos demo --bandwidth-mbps 12
streamlit run src/traffic_qos/dashboard.py
```

If the `traffic-qos` command is not on your PATH, run modules directly:

```powershell
$env:PYTHONPATH = "src"
python -m traffic_qos.cli generate
```

Optional extras:

| Extra | Install | Adds |
|---|---|---|
| `dashboard` | `pip install -e ".[dashboard]"` | Streamlit and Plotly |
| `cnn` | `pip install -e ".[cnn]"` | TensorFlow 1D-CNN |
| `pcap` | `pip install -e ".[pcap]"` | Scapy for PCAP/PCAPNG input |
| `dev` | `pip install -e ".[dev]"` | pytest |

The demo replays 30 flows by default (balanced across classes), turns each into three concurrent requests, and compares the four policies. Add `--max-flows 60` or lower the bandwidth (for example 4 Mbps) to stress the queue. Add `--oracle-labels` to use true labels instead of the trained model. Everything is deterministic for fixed seeds. On the bundled synthetic data, priority scheduling typically cuts deadline misses substantially compared with FIFO at similar throughput, but regenerate the artifacts yourself and treat this only as a pipeline check.

## Dashboard

```powershell
streamlit run src/traffic_qos/dashboard.py
```

The dashboard is dark-themed and has six tabs:

| Tab | What it shows |
|---|---|
| **About** | What the project does, plus a traffic explorer: pick a traffic type and see the packet sizes, directions and gaps a classifier sees. |
| **Playground** | A small interactive queue simulation. Change bandwidth, number of flows and arrival window and compare first-come-first-served with priority-by-class. |
| **Overview** | Dataset, split strategy, capture groups, train/test rows, best macro F1 and a leakage check. |
| **Classification** | Model comparison, confusion matrix (counts or row shares) and per-class precision, recall and F1. |
| **QoS scheduler** | Policy filter, p95 latency, deadline misses and per-class latency from `scheduler_results.csv`. |
| **Predict flows** | Upload a flow CSV, get a class and confidence per flow, filter results, then simulate QoS service for your own flows. |

The About and Playground tabs use generated example traffic with illustrative classes (VoIP, chat, browsing, video, file transfer). They explain the idea and do not use your results.

To get the dark theme on Streamlit's own widgets, keep `.streamlit/config.toml` in the folder you launch from. Choose an evaluation file in the sidebar to inspect a dataset run.

## Using your own data

**Wireshark CSV.** Export one packet per row with columns `timestamp,src_ip,dst_ip,src_port,dst_port,protocol,length` (timestamps in seconds), then:

```powershell
traffic-qos extract --input data/packets.csv --output data/flows.csv
```

**PCAP / PCAPNG** (needs the `pcap` extra; IPv4/IPv6 TCP and UDP only, no live capture):

```powershell
traffic-qos extract-pcap --input data/capture.pcapng --output data/flows.csv
traffic-qos extract-sequences --input data/capture.pcapng --pcap --output data/sequences.csv
```

**Schedule your flows** with a trained model. Each flow becomes one request, sized from real byte columns when present:

```powershell
traffic-qos schedule --input data/flows.csv --bandwidth-mbps 50 --arrival random --window-s 30
traffic-qos schedule --input data/capture.pcapng --pcap --arrival burst
```

Arrival patterns are `burst` (all at once, worst case), `even`, `random`, or `auto` (uses a start-time column if present). Outputs go to `outputs/schedule/` as `flow_qos.csv`, `scheduler_results.csv` and `scheduler_summary.json`.

To train on your own labeled flows, add a consistent `label` column and pass the CSV to `train`. Flow CSVs need the 15 columns in `FEATURE_COLUMNS`.

**Flow-meter exports.** Datasets such as ISCX ship as CSVs from a flow-meter tool with their own column names. `scripts/adapt_flowmeter_csv.py` converts such an export into the 15-column flow format this project expects (check the script header for its arguments). Large raw files belong in `data/raw/` and are not committed.

**Try the dashboard upload.** In the Predict flows tab, upload any CSV with the feature columns, for example a small sample cut from your flows file, and the dashboard classifies it and simulates QoS service.

### Bandwidth sweeps

Running the scheduler at several link speeds shows where each policy starts to break down. For example, in PowerShell:

```powershell
foreach ($bw in 25, 50, 100, 200) {
  traffic-qos schedule --input data/flows_all.csv --bandwidth-mbps $bw --output-dir "outputs_bw$bw"
}
```

Each run writes its own `scheduler_results.csv`, `scheduler_summary.json` and `flow_qos.csv`. In the dashboard, type a folder such as `outputs_bw50` into **Artifacts directory** to switch between runs. The Predict flows tab reads the model from the **Model path** box, which defaults to `outputs/flow_classifier.joblib`.

## Dataset evaluation (ISCX VPN-nonVPN, QUIC)

`evaluate-dataset` takes already-extracted, labeled flow CSVs and splits by **capture or session group**, so no group appears on both sides of the split. It verifies zero overlap and fails if a class is missing from either side.

```powershell
traffic-qos evaluate-dataset --input data/iscx_flows.csv --dataset iscx-vpn-nonvpn `
  --label-column application --group-column capture_id `
  --label-map '{"Browsing":"interactive","Streaming":"streaming","BulkTransfer":"bulk"}'

traffic-qos evaluate-dataset --input data/quic_flows.csv --dataset quic --label-column app_class --group-column trace_id
```

Add `--cnn` to also evaluate the 1D-CNN (the CSV then needs a `packet_sequence` column). Outputs are `outputs/<dataset>_evaluation.json` (metrics, exact train/test groups, overlap audit) and `outputs/<dataset>_test_predictions.csv`.

The datasets are large and are **not** included or downloaded. Download them from their official sources, convert them to the flow contract, and report your label mapping. `data/` and `outputs*/` are git-ignored.

## 1D-CNN workflow

```powershell
python -m pip install -e ".[cnn]"
traffic-qos generate-sequences --output data/demo_sequences.csv
traffic-qos train --input data/demo_sequences.csv --cnn --model-only
traffic-qos predict --input data/demo_sequences.csv --model outputs/flow_cnn.keras --cnn --output outputs/cnn_predictions.csv
```

This writes `flow_cnn.keras`, `cnn_preprocessing.json` and `cnn_evaluation.json`. The generated data is synthetic, so these numbers are not real dataset results.

## Commands

| Command | Action | Main outputs |
|---|---|---|
| `generate` | Synthetic labeled flow CSV | `data/demo_flows.csv` |
| `generate-sequences` | Synthetic packet-sequence CSV | `data/demo_sequences.csv` |
| `extract` | Packet CSV to flow features | `data/flows.csv` |
| `extract-pcap` | PCAP/PCAPNG to flow features | `data/flows.csv` |
| `extract-sequences` | Early packet sequences from CSV or PCAP | `data/sequences.csv` |
| `train` | Compare random forest and logistic regression (`--cnn` adds the CNN) | `flow_classifier.joblib`, `evaluation.json` |
| `predict` | Add class and confidence to flows (`--cnn` for the CNN) | `outputs/predictions.csv` |
| `demo` | Four-policy scheduler comparison on synthetic or classified flows | `scheduler_results.csv`, `scheduler_summary.json` |
| `schedule` | QoS scheduling of your own flows (CSV or PCAP) | `outputs/schedule/*` |
| `evaluate-dataset` | Capture-grouped evaluation for ISCX or QUIC | `<dataset>_evaluation.json` |

Run `traffic-qos <command> --help` for all options.

## Project structure

```text
.
├── .streamlit/config.toml     dark theme for the dashboard
├── scripts/
│   └── adapt_flowmeter_csv.py converts an exported flow-meter CSV into the flow format
├── src/traffic_qos/
│   ├── features.py            flow and sequence extraction, FEATURE_COLUMNS
│   ├── data.py                synthetic packets, flows and sequences
│   ├── model.py               training and prediction (RF, logistic, CNN)
│   ├── evaluation.py          capture-grouped evaluation
│   ├── qos.py                 class profiles and confidence fallback
│   ├── simulator.py           bottleneck-link simulator and policies
│   ├── schedule_flows.py      scheduling for user-supplied flows
│   ├── cli.py                 traffic-qos command line
│   └── dashboard.py           Streamlit dashboard
├── data/                      local only: raw/, demo_flows.csv, flows_all.csv, upload_sample.csv
├── outputs/                   local only: default models and results
├── outputs_bw25 ... bw200     local only: bandwidth-sweep runs (see below)
├── features.md                feature list and demo narration
├── pyproject.toml
└── README.md
```

`data/` and every `outputs*/` folder are git-ignored because the dataset and trained models are far larger than GitHub's 100 MB file limit. After cloning, regenerate them with the commands in this file.

## Metrics and how to read them

Classification reports give accuracy, macro F1, per-class precision, recall and F1, and a confusion matrix. Scheduler summaries give mean and p95 latency, deadline-miss rate, latency-target miss rate, throughput and per-class breakdowns. Look at both: a good aggregate can hide harm to one class. The oracle row estimates the headroom if classification were perfect.

## Limitations

- `train` uses a random stratified split on rows. Only `evaluate-dataset` separates by capture group, so use it for any claim about unseen traffic.
- Three classes only: `interactive`, `streaming`, `bulk`. Other labels fall back to best-effort.
- The simulator is a single non-preemptive link with requests generated from class-typical sizes, unless real byte columns are supplied.
- `flow_id` values come from Python's `hash`, so they can differ between runs.
- A CNN trained through `evaluate-dataset` saves `<dataset>_cnn_preprocessing.json`, but `predict --cnn` reads `cnn_preprocessing.json`. Rename the file or retrain with `train --cnn` before predicting.
- Not production network control. No live capture.