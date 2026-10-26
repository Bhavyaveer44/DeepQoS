# Feature progress and explanation guide

This file is the short review log: each row states what the feature does, where it lives, and how to demonstrate it. Status describes implementation in this repository, not research validation on a real dataset.

| ID | Feature | Status | What it does / how to explain it | Where / evidence |
|---|---|---|---|---|
| F1 | Bidirectional flow extraction | Implemented | Groups packet metadata by protocol and endpoint pair, then starts a new flow after an idle timeout. No payload inspection. | `src/traffic_qos/features.py`; `traffic-qos extract` |
| F2 | Statistical feature engineering | Implemented | Computes packet size, direction, timing, rate, and burstiness statistics for each flow. | `FEATURE_COLUMNS` in `features.py`; `data/demo_flows.csv` |
| F3 | Reproducible synthetic demo data | Implemented | Produces labeled interactive, streaming, and bulk examples so the software can be demonstrated without external captures. Synthetic metrics must not be represented as real-world results. | `src/traffic_qos/data.py`; `traffic-qos generate` |
| F4 | Classical model baselines | Implemented | Compares Random Forest with logistic regression using stratified holdout data; selects by macro F1 and saves the model. | `src/traffic_qos/model.py`; `outputs/evaluation.json` |
| F5 | Prediction confidence | Implemented | Attaches the predicted class and maximum class probability to each flow. | `predict_flows`; `traffic-qos predict` |
| F6 | QoS class mapping | Implemented | Maps classes to documented priority, deadline, and latency targets. These are experiment assumptions that can be tuned. | `src/traffic_qos/qos.py`; table in `README.md` |
| F7 | Confidence-aware fallback | Implemented | Low-confidence or unsupported predictions use middle-priority best-effort treatment. | `apply_confidence_policy`; `model_guided` policy |
| F8 | Multi-request bottleneck simulator | Implemented | Creates several requests per flow and serializes them over a configurable shared link, including concurrent arrivals and queue wait. | `src/traffic_qos/simulator.py`; `--bandwidth-mbps` |
| F9 | Scheduler baselines | Implemented | Compares FIFO, static priority, model-guided priority, and perfect-label oracle scheduling. Oracle is an upper bound. | `compare_schedulers`; `outputs/scheduler_results.csv` |
| F10 | QoS outcome metrics | Implemented | Reports mean/p95 latency, deadline and latency-target misses, throughput, and per-class breakdowns. | `outputs/scheduler_summary.json` |
| F11 | 1D-CNN sequence classifier | Implemented / optional dependency | Trains on fixed early-packet sequences with direction, packet size, and inter-arrival time; saves the model and preprocessing metadata; predicts labels and confidence. | `generate-sequences`; `train --cnn --model-only`; `predict --cnn`; install `.[cnn]` |
| F12 | ISCX VPN-nonVPN / QUIC grouped evaluation | Implemented / data-dependent | Runs Random Forest and logistic baselines on capture/session-separated holdout groups, verifies zero group overlap, saves metrics, per-class reports, confusion matrices and test predictions. Optional CNN uses the same split. Dataset-specific mapping and actual captures are supplied by the user; no dataset result is claimed until run. | `traffic-qos evaluate-dataset`; `outputs/<dataset>_evaluation.json` |
| F13 | Wireshark CSV / PCAP integration | Implemented / optional PCAP dependency | Extracts flow statistics or early sequences from packet metadata CSV and directly from IPv4/IPv6 TCP/UDP PCAP/PCAPNG. It does not perform live capture. | `extract`, `extract-pcap`, `extract-sequences`; install `.[pcap]` for PCAP |
| F14 | Interactive research dashboard | Implemented / optional dependency | Shows experiment status, classifier metrics/confusion matrix, grouped-split audit, scheduler policy and class-level tradeoffs, request records, and uploaded-flow prediction. | `streamlit run src/traffic_qos/dashboard.py`; install `.[dashboard]` |

## Demo narration

1. Generate labeled example flows and explain that all inputs are packet metadata, not payload.
2. Train the two baselines and show accuracy plus macro F1 and the confusion matrix.
3. Run the demo at a chosen link bandwidth; explain that many requests from different flows compete for one service queue.
4. Compare FIFO, static, model-guided, and oracle metrics. Focus on p95 latency and per-class deadline misses, not just a single average.
5. Optionally show the CNN separately with generated packet sequences. Explain that oracle labels show an upper bound, synthetic data validates the pipeline only, and real dataset/capture splits are required for a research claim.

