# Feature progress and explanation guide

This is the short review log. Each row says what a feature does, where it lives and how to demonstrate it. **Status describes implementation in this repository, not research validation on a real dataset.**

## Features

| ID  | Feature                            | Status                      | What it does                                                                                                                                                                                                                                                   | Where / how to run                                                                                    |
| --- | ---------------------------------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| F1  | Bidirectional flow extraction      | Implemented                 | Groups packet metadata by protocol and endpoint pair and starts a new flow after an idle timeout (default 30 s). Payloads are never inspected.                                                                                                                 | `features.py`; `traffic-qos extract`                                                                  |
| F2  | Statistical feature engineering    | Implemented                 | Computes 15 payload-blind features per flow: duration, packet count, size statistics, inter-arrival statistics, forward/backward packets and bytes, rates, burstiness.                                                                                         | `FEATURE_COLUMNS` in `features.py`                                                                    |
| F3  | Reproducible synthetic data        | Implemented                 | Generates labeled interactive, streaming and bulk flows and packet sequences so the project runs without external captures. Synthetic metrics are not real-world results.                                                                                      | `data.py`; `generate`, `generate-sequences`                                                           |
| F4  | Classical model baselines          | Implemented                 | Trains random forest and logistic regression on a stratified holdout, selects by macro F1 and saves the winner.                                                                                                                                                | `model.py`; `train`                                                                                   |
| F5  | Prediction confidence              | Implemented                 | Attaches the predicted class and its highest class probability to each flow.                                                                                                                                                                                   | `predict_flows`; `predict`                                                                            |
| F6  | QoS class mapping                  | Implemented                 | Maps each class to a priority, deadline and latency target. These are tunable experiment assumptions.                                                                                                                                                          | `qos.py`; table in `README.md`                                                                        |
| F7  | Confidence-aware fallback          | Implemented                 | Predictions below the threshold (default 0.60) or with an unknown class are served as middle-priority best-effort.                                                                                                                                             | `apply_confidence_policy`                                                                             |
| F8  | Multi-request bottleneck simulator | Implemented                 | Turns flows into concurrent requests and serves them over a shared link of configurable bandwidth, including queue wait.                                                                                                                                       | `simulator.py`; `demo --bandwidth-mbps`                                                               |
| F9  | Scheduler policy comparison        | Implemented                 | Compares FIFO, static priority, model-guided priority and the oracle (true labels, an upper bound).                                                                                                                                                            | `compare_schedulers`; `scheduler_results.csv`                                                         |
| F10 | QoS outcome metrics                | Implemented                 | Reports mean and p95 latency, deadline and latency-target misses, throughput and per-class breakdowns.                                                                                                                                                         | `scheduler_summary.json`                                                                              |
| F11 | 1D-CNN sequence classifier         | Implemented, optional       | Learns from the first 40 packets as `[direction, size, inter-arrival]`. Saves the model and its preprocessing metadata.                                                                                                                                        | `train --cnn --model-only`; `predict --cnn`; extra `.[cnn]`                                           |
| F12 | Capture-grouped dataset evaluation | Implemented, data-dependent | Splits ISCX VPN-nonVPN or QUIC flow CSVs by capture group, checks for zero overlap and missing classes, and saves metrics, confusion matrices and test predictions. Optional CNN uses the same split. No dataset result is claimed until run on real captures. | `evaluate-dataset`; `<dataset>_evaluation.json`                                                       |
| F13 | Wireshark CSV and PCAP input       | Implemented, optional       | Extracts flow features or early sequences from packet CSVs and from IPv4/IPv6 TCP/UDP PCAP/PCAPNG files. No live capture.                                                                                                                                      | `extract`, `extract-pcap`, `extract-sequences`; extra `.[pcap]`                                       |
| F14 | Scheduling for your own flows      | Implemented                 | Each flow in a CSV or PCAP becomes one request, sized from real byte columns when available (else per-class defaults). Outputs per-flow service, latency, FIFO comparison and latency saved.                                                                   | `schedule_flows.py`; `traffic-qos schedule`                                                           |
| F15 | Dark interactive dashboard         | Implemented, optional       | Six tabs: About, Playground, Overview, Classification, QoS scheduler, Predict flows. Includes a traffic explorer, a queue playground, model and policy filters, confusion matrix toggle, CSV downloads and in-app QoS simulation for uploaded flows.           | `streamlit run src/traffic_qos/dashboard.py`; extra `.[dashboard]`; theme in `.streamlit/config.toml` |
| F16 | Flow-meter CSV adapter             | Implemented                 | Converts an exported flow-meter CSV (such as an ISCX export) into the project's flow format so large real datasets can be used.                                                                                                                                | `scripts/adapt_flowmeter_csv.py`; raw files in `data/raw/`                                            |
| F17 | Bandwidth sweeps                   | Supported by the CLI        | Runs the scheduler at several link speeds (for example 25, 50, 100 and 200 Mbps) into separate output folders to compare policies as the link tightens. Each run can be opened in the dashboard.                                                               | `traffic-qos schedule --bandwidth-mbps N --output-dir outputs_bwN`                                    |

## Dashboard tour

| Tab            | Try this                                                                                                                                                           |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| About          | Pick each traffic type and compare packet size, direction and gap patterns. Press "Generate another flow" for a new sample.                                        |
| Playground     | Raise the number of flows or lower bandwidth until first-come-first-served starts missing deadlines, then see how priority-by-class changes it.                    |
| Overview       | Check that capture groups are disjoint and note the best model.                                                                                                    |
| Classification | Switch models, toggle the confusion matrix to row shares, read per-class scores.                                                                                   |
| QoS scheduler  | Filter policies and compare p95 latency and deadline misses. Change **Artifacts directory** to `outputs_bw25`, `outputs_bw50` and so on to compare bandwidth runs. |
| Predict flows  | Upload a flow CSV, filter by class and confidence, then adjust bandwidth, arrival pattern and threshold.                                                           |

The About traffic explorer and the Playground use illustrative generated traffic with example classes (VoIP, chat, browsing, video, file transfer). They are explanations, not results from the trained model.

## Repository notes

- `data/` and all `outputs*/` folders are git-ignored (the raw dataset and trained models exceed GitHub's 100 MB limit). Regenerate them with the commands in `README.md`.
- The dashboard theme needs `.streamlit/config.toml` in the folder you launch from.

## Known limitations

- `train` uses a random stratified split. Use `evaluate-dataset` for capture-separated results.
- The simulator is a single non-preemptive link, not packet-level emulation.
- Flow IDs use Python's `hash` and may differ between runs.
- CNN models from `evaluate-dataset` use a differently named preprocessing file than `predict --cnn` expects.

## Demo narration

1. Generate labeled flows and explain that inputs are packet metadata, never payload.
2. Train the baselines and show accuracy, macro F1 and the confusion matrix.
3. Run the demo at a chosen bandwidth and explain that many requests compete for one queue.
4. Compare FIFO, static, model-guided and oracle. Focus on p95 latency and per-class deadline misses, not a single average.
5. Show the CNN separately on generated sequences if time allows.
6. Close with the caveat: oracle results are an upper bound, synthetic data validates the pipeline only, and a research claim needs real captures split by capture group.
