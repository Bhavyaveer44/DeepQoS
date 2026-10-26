"""Streamlit review dashboard for classification and QoS experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    try:
        import plotly.express as px
        import streamlit as st
    except ImportError as exc:
        raise SystemExit("Install dashboard dependencies: pip install -e '.[dashboard]'") from exc

    st.set_page_config(page_title="Encrypted Traffic QoS Lab", page_icon="📶", layout="wide")
    st.title("Encrypted Traffic Classification & QoS Lab")
    st.caption("Payload-blind flow classification, capture-separated evaluation, and multi-request scheduling.")
    root = Path.cwd()
    outputs = Path(st.sidebar.text_input("Artifacts directory", "outputs"))
    evaluation_files = sorted(outputs.glob("*_evaluation.json"))
    default_eval = outputs / "evaluation.json"
    if default_eval.exists() and default_eval not in evaluation_files:
        evaluation_files.insert(0, default_eval)
    eval_options = [str(path) for path in evaluation_files]
    selected_eval = st.sidebar.selectbox("Evaluation artifact", ["(none)", *eval_options])
    scheduler_path = outputs / "scheduler_results.csv"
    predictions_upload = st.sidebar.file_uploader("Optional flow CSV for prediction", type="csv")

    tab_overview, tab_classifier, tab_scheduler, tab_predict = st.tabs(
        ["Overview", "Classification", "QoS scheduler", "Predict flows"])

    metrics = None
    if selected_eval != "(none)":
        metrics = json.loads(Path(selected_eval).read_text(encoding="utf-8"))
    scheduler = pd.read_csv(scheduler_path) if scheduler_path.exists() else pd.DataFrame()

    with tab_overview:
        st.subheader("Experiment status")
        if metrics:
            left, middle, right = st.columns(3)
            left.metric("Dataset", metrics.get("dataset", "demo"))
            middle.metric("Split", metrics.get("split_strategy", "stratified holdout"))
            right.metric("Capture groups", len(metrics.get("train_groups", [])) + len(metrics.get("test_groups", [])))
            st.write(f"Train rows: **{metrics.get('train_rows', '—')}** · Test rows: **{metrics.get('test_rows', '—')}**")
            if metrics.get("group_overlap"):
                st.error(f"Capture leakage detected: {metrics['group_overlap']}")
            elif "group_overlap" in metrics:
                st.success("Capture/session groups are disjoint across train and test.")
        else:
            st.info("No evaluation selected. Generate demo artifacts or choose a dataset evaluation from the sidebar.")
        if not scheduler.empty:
            policy_count = scheduler["policy"].nunique()
            st.metric("Requests replayed", f"{len(scheduler) // max(policy_count, 1):,}")
            st.caption("Scheduler results are a queueing simulation, not packet-level emulation.")
        st.markdown("**Research interpretation:** synthetic data verifies the workflow; only capture-separated ISCX/QUIC metrics support dataset claims.")

    with tab_classifier:
        st.subheader("Classification results")
        if not metrics or "models" not in metrics:
            st.info("Select an evaluation JSON artifact.")
        else:
            rows = [{"Model": name, "Accuracy": result.get("accuracy"), "Macro F1": result.get("macro_f1")}
                    for name, result in metrics["models"].items()]
            score_frame = pd.DataFrame(rows).sort_values("Macro F1", ascending=False)
            st.dataframe(score_frame, use_container_width=True, hide_index=True)
            model_name = st.selectbox("Confusion matrix model", list(metrics["models"]))
            result = metrics["models"][model_name]
            matrix = result.get("confusion_matrix")
            labels = metrics.get("classes", [])
            if matrix and labels:
                chart = px.imshow(matrix, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues",
                                  labels={"x": "Predicted", "y": "Actual", "color": "Flows"})
                st.plotly_chart(chart, use_container_width=True)
                report = result.get("classification_report", {})
                class_rows = [{"Class": label, **report.get(label, {})} for label in labels]
                st.dataframe(pd.DataFrame(class_rows), use_container_width=True, hide_index=True)
            with st.expander("Split audit"):
                st.json({"group_column": metrics.get("group_column"), "train_groups": metrics.get("train_groups"),
                         "test_groups": metrics.get("test_groups"), "overlap": metrics.get("group_overlap")})

    with tab_scheduler:
        st.subheader("Concurrent request scheduling")
        if scheduler.empty:
            st.info("No scheduler_results.csv found in the selected artifacts directory. Run `traffic-qos demo` first.")
        else:
            summary_rows = []
            for policy, group in scheduler.groupby("policy"):
                summary_rows.append({"Policy": policy, "Mean latency (ms)": group["latency_ms"].mean(),
                    "p95 latency (ms)": group["latency_ms"].quantile(.95),
                    "Deadline miss rate": 1 - group["deadline_met"].mean(),
                    "Requests": group["request_id"].nunique()})
            summary_frame = pd.DataFrame(summary_rows)
            a, b = st.columns(2)
            a.plotly_chart(px.bar(summary_frame, x="Policy", y="p95 latency (ms)", title="p95 latency by policy"), use_container_width=True)
            b.plotly_chart(px.bar(summary_frame, x="Policy", y="Deadline miss rate", title="Deadline misses by policy"), use_container_width=True)
            per_class = scheduler.groupby(["policy", "traffic_class"], as_index=False).agg(
                mean_latency_ms=("latency_ms", "mean"), deadline_miss_rate=("deadline_met", lambda s: 1-s.mean()))
            st.plotly_chart(px.bar(per_class, x="traffic_class", y="mean_latency_ms", color="policy", barmode="group",
                                   title="Mean latency by traffic class"), use_container_width=True)
            with st.expander("Request-level results"):
                st.dataframe(scheduler, use_container_width=True, hide_index=True)

    with tab_predict:
        st.subheader("Classify an uploaded flow CSV")
        if predictions_upload is None:
            st.info("Upload a CSV containing the aggregate flow feature columns.")
        else:
            try:
                from .features import FEATURE_COLUMNS
                from .model import predict_flows
                frame = pd.read_csv(predictions_upload)
                model_path = Path(st.text_input("Model path", str(outputs / "flow_classifier.joblib")))
                missing = set(FEATURE_COLUMNS) - set(frame.columns)
                if missing:
                    st.error(f"Missing feature columns: {', '.join(sorted(missing))}")
                elif model_path.exists():
                    predicted = predict_flows(model_path, frame)
                    st.dataframe(predicted, use_container_width=True, hide_index=True)
                    st.download_button("Download predictions", predicted.to_csv(index=False),
                                       file_name="flow_predictions.csv", mime="text/csv")
                else:
                    st.warning(f"Model not found: {model_path}. Train first with `traffic-qos train`.")
            except Exception as exc:  # User-provided model/input errors are displayed in the dashboard.
                st.error(str(exc))


if __name__ == "__main__":
    main()
