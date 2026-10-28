"""Streamlit review dashboard for classification and QoS experiment artifacts."""

from __future__ import annotations

import html
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

BG, SURFACE, EDGE, TEXT, MUTED = "#0B1020", "#121A30", "#242F4D", "#E6EAF5", "#8D9AB8"
CYAN, VIOLET, AMBER, ROSE, MINT = "#38BDF8", "#A78BFA", "#FBBF24", "#FB7185", "#34D399"
PALETTE = [CYAN, VIOLET, AMBER, ROSE, MINT, "#94A3B8"]
HEAT = [[0, SURFACE], [1, CYAN]]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500..800&family=Instrument+Sans:wght@400;500;600&display=swap');
.stApp {{ background: {BG}; color: {TEXT}; font-family: 'Instrument Sans', system-ui, sans-serif; }}
.block-container {{ padding-top: 1.5rem; padding-bottom: 4rem; max-width: 1280px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
h1, h2, h3, h4 {{ font-family: 'Bricolage Grotesque', sans-serif; letter-spacing: -.01em; color: {TEXT}; }}
[data-testid="stSidebar"] {{ background: #0E1528; border-right: 1px solid {EDGE}; }}
[data-baseweb="input"] > div, [data-baseweb="select"] > div, [data-baseweb="textarea"] {{ background: {SURFACE}; border-color: {EDGE}; }}
.stApp label, .stApp p, .stApp li, .stApp span {{ color: inherit; }}
.brand {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.2rem; }}
.sub {{ color: {MUTED}; font-size: .9rem; margin-bottom: 1rem; }}

.hero {{ border: 1px solid {EDGE}; background: {SURFACE}; border-radius: 20px; padding: 2rem 2.2rem; display: flex;
         gap: 2rem; align-items: center; justify-content: space-between; flex-wrap: wrap; margin-bottom: 1.2rem; }}
.hero h1 {{ font-size: 2.4rem; line-height: 1.1; margin: 0 0 .6rem; padding: 0; max-width: 22ch; }}
.hero p {{ color: {MUTED}; margin: 0; max-width: 58ch; line-height: 1.55; }}
.ticks {{ display: flex; gap: 5px; }}
.tick {{ display: flex; flex-direction: column; }}
.tick .u, .tick .d {{ width: 6px; height: 48px; display: flex; }}
.tick .u {{ align-items: flex-end; }}
.tick i {{ display: block; width: 6px; border-radius: 3px; }}
.tick .u i {{ background: {CYAN}; }} .tick .d i {{ background: {AMBER}; }}

.card {{ background: {SURFACE}; border: 1px solid {EDGE}; border-radius: 14px; padding: 1.1rem 1.2rem; height: 100%; }}
.card h4 {{ margin: 0 0 .4rem; font-size: 1.1rem; }}
.card p {{ color: {MUTED}; margin: 0; line-height: 1.55; font-size: .95rem; }}
.kpi-l {{ color: {MUTED}; font-size: .88rem; }}
.kpi-v {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.65rem; margin-top: .15rem; word-break: break-word; }}
.kpi-n {{ color: {MUTED}; font-size: .82rem; margin-top: .25rem; }}
.good {{ color: {MINT}; }} .warn {{ color: {AMBER}; }} .bad {{ color: {ROSE}; }} .info {{ color: {CYAN}; }}
.empty {{ border: 1.5px dashed {EDGE}; border-radius: 14px; padding: 2rem; text-align: center; background: {SURFACE}; }}
.empty b {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.15rem; }}
.empty div {{ color: {MUTED}; margin-top: .3rem; }}

.stTabs [data-baseweb="tab-list"] {{ gap: .3rem; border-bottom: 1px solid {EDGE}; }}
.stTabs [data-baseweb="tab"] {{ height: 2.8rem; padding: 0 1rem; color: {MUTED}; font-weight: 500; }}
.stTabs [aria-selected="true"] {{ color: {TEXT}; font-weight: 600; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {CYAN}; height: 3px; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background: {SURFACE}; border-color: {EDGE}; border-radius: 14px; }}
.stDownloadButton button {{ border-radius: 10px; border: 1px solid {CYAN}; color: {CYAN}; background: transparent; font-weight: 600; }}
.stDownloadButton button:hover {{ background: {CYAN}; color: {BG}; }}
@media (max-width: 700px) {{ .hero h1 {{ font-size: 1.7rem; }} .ticks {{ display: none; }} }}
</style>
"""

# Illustrative traffic archetypes: priority (1 = served first), typical flow size (MB), deadline (ms).
ARCHETYPES = {
    "VoIP call": dict(prio=1, mb=0.3, deadline=150, note="Tiny packets in both directions at a steady beat. Any delay is heard, so it goes first."),
    "Chat": dict(prio=2, mb=0.05, deadline=300, note="Sparse small messages with long silences. Light on bandwidth but people notice lag."),
    "Web browsing": dict(prio=3, mb=0.6, deadline=1000, note="A small request upstream, then a burst of large replies downstream."),
    "Video streaming": dict(prio=4, mb=8, deadline=3000, note="Mostly downstream and full-size packets in bursts. Buffers hide a little delay."),
    "File transfer": dict(prio=5, mb=20, deadline=30000, note="Steady full-size packets downstream with small acknowledgements upstream. Delay-tolerant."),
}


def packets(kind: str, seed: int, n: int = 40) -> pd.DataFrame:
    """Synthetic packet sequence for one archetype: what a payload-blind classifier sees."""
    r = np.random.default_rng(seed)
    i = np.arange(n)
    if kind == "VoIP call":
        d, s, t = np.where(i % 2 == 0, 1, -1), r.normal(172, 10, n), r.normal(20, 2, n)
    elif kind == "Chat":
        d, s, t = r.choice([1, -1], n), r.lognormal(5, .5, n), r.exponential(400, n)
    elif kind == "Web browsing":
        d = np.where(i % 7 == 0, 1, -1)
        s, t = np.where(d > 0, r.normal(480, 90, n), r.choice([1448, 1448, 900, 300], n)), r.exponential(25, n)
    elif kind == "Video streaming":
        d = np.where(r.random(n) < .08, 1, -1)
        s, t = np.where(d < 0, r.normal(1380, 60, n), 66), r.exponential(6, n)
    else:
        d = np.where(i % 4 == 3, 1, -1)
        s, t = np.where(d < 0, 1448, 66), r.exponential(1.5, n)
    s = np.clip(s, 40, 1500)
    return pd.DataFrame({"Packet": i + 1, "Size (bytes)": s * d, "Direction": np.where(d > 0, "Upstream", "Downstream"),
                         "Gap (ms)": t})


def toy_schedule(bandwidth: float, n: int, window: float, seed: int) -> pd.DataFrame:
    """Single-link queue replay: first-come-first-served vs priority by traffic class."""
    r = np.random.default_rng(seed)
    names = list(ARCHETYPES)
    cls = r.choice(len(names), n, p=[.2, .2, .3, .15, .15])
    size = np.array([ARCHETYPES[names[c]]["mb"] for c in cls]) * r.lognormal(0, .4, n)
    arrive = np.sort(r.uniform(0, window, n)) if window > 0 else np.zeros(n)
    service = size * 8 / bandwidth
    prio = np.array([ARCHETYPES[names[c]]["prio"] for c in cls])
    rows = []
    for policy in ("First come, first served", "Priority by class"):
        t, left = 0.0, list(range(n))
        while left:
            ready = [k for k in left if arrive[k] <= t] or [min(left, key=lambda k: arrive[k])]
            k = ready[0] if policy.startswith("First") else min(ready, key=lambda k: (prio[k], arrive[k]))
            t = max(t, arrive[k]) + service[k]
            left.remove(k)
            lat = (t - arrive[k]) * 1000
            rows.append({"Policy": policy, "Class": names[cls[k]], "Latency (ms)": lat,
                         "Met deadline": lat <= ARCHETYPES[names[cls[k]]]["deadline"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- UI helpers
def hero(title: str, subtitle: str) -> str:
    bars = ""
    for i, h in enumerate([18, 34, 12, 40, 26, 8, 44, 20, 30, 14, 38, 22, 10, 36, 28, 16, 42, 24, 12, 32]):
        up = i % 3 != 1
        bars += (f'<div class="tick"><div class="u">{f"<i style=height:{h}px></i>" if up else ""}</div>'
                 f'<div class="d">{"" if up else f"<i style=height:{h}px></i>"}</div></div>')
    return (f'<div class="hero"><div><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>'
            f'<div class="ticks" title="Packet direction and size over one flow">{bars}</div></div>')


def kpi(label, value, note="", tone="") -> str:
    n = f'<div class="kpi-n">{html.escape(str(note))}</div>' if note else ""
    return (f'<div class="card"><div class="kpi-l">{html.escape(str(label))}</div>'
            f'<div class="kpi-v {tone}">{html.escape(str(value))}</div>{n}</div>')


def kpi_row(st, cards) -> None:
    for col, card in zip(st.columns(len(cards)), cards):
        col.markdown(card, unsafe_allow_html=True)


def info_card(title: str, body: str) -> str:
    return f'<div class="card"><h4>{title}</h4><p>{body}</p></div>'


def empty(st, title: str, body: str) -> None:
    st.markdown(f'<div class="empty"><b>{html.escape(title)}</b><div>{body}</div></div>', unsafe_allow_html=True)


def chart(st, fig, height: int = 360) -> None:
    fig.update_layout(
        height=height, colorway=PALETTE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Instrument Sans, sans-serif", color=TEXT, size=13),
        title=dict(font=dict(family="Bricolage Grotesque, sans-serif", size=16), x=0, xanchor="left"),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title_text=""),
        hoverlabel=dict(bgcolor=BG, font_color=TEXT, bordercolor=CYAN))
    fig.update_xaxes(gridcolor="rgba(255,255,255,.07)", zerolinecolor=EDGE)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.07)", zerolinecolor=EDGE)
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def main() -> None:
    try:
        import plotly.express as px
        import streamlit as st
    except ImportError as exc:
        raise SystemExit("Install dashboard dependencies: pip install -e '.[dashboard]'") from exc

    st.set_page_config(page_title="Encrypted Traffic QoS Lab", page_icon="📶", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    num = st.column_config.NumberColumn
    policy_cfg = {"Mean latency (ms)": num(format="%.1f"), "p95 latency (ms)": num(format="%.1f"),
                  "Deadline miss rate (%)": num(format="%.1f%%"), "Throughput (Mbps)": num(format="%.2f")}
    prog = lambda: st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f")  # noqa: E731

    # ------------------------------------------------------------ sidebar
    st.sidebar.markdown('<div class="brand">Traffic QoS Lab</div><div class="sub">Load a run to review it.</div>',
                        unsafe_allow_html=True)
    outputs = Path(st.sidebar.text_input("Artifacts directory", "outputs"))
    evaluation_files = sorted(outputs.glob("*_evaluation.json"))
    default_eval = outputs / "evaluation.json"
    if default_eval.exists() and default_eval not in evaluation_files:
        evaluation_files.insert(0, default_eval)
    selected_eval = st.sidebar.selectbox("Evaluation artifact", ["(none)", *map(str, evaluation_files)])
    scheduler_path = outputs / "scheduler_results.csv"
    st.sidebar.divider()
    predictions_upload = st.sidebar.file_uploader("Flow CSV to classify", type="csv")
    model_path = Path(st.sidebar.text_input("Model path", str(outputs / "flow_classifier.joblib")))

    metrics = json.loads(Path(selected_eval).read_text(encoding="utf-8")) if selected_eval != "(none)" else None
    scheduler = pd.read_csv(scheduler_path) if scheduler_path.exists() else pd.DataFrame()

    st.markdown(hero("Telling encrypted traffic apart without opening it",
                     "Classify network flows from packet sizes, directions and timing alone, then use the "
                     "result to give each kind of traffic the service it needs."), unsafe_allow_html=True)

    t_about, t_play, t_over, t_cls, t_sched, t_pred = st.tabs(
        ["About", "Playground", "Overview", "Classification", "QoS scheduler", "Predict flows"])

    # ------------------------------------------------------------ about
    with t_about:
        st.subheader("What this project does")
        a, b, c = st.columns(3)
        a.markdown(info_card("1. Observe flows", "Each flow is reduced to aggregate features and a short packet "
                             "sequence: direction, size and gaps between packets. The payload is never read, so "
                             "this works on encrypted traffic."), unsafe_allow_html=True)
        b.markdown(info_card("2. Classify", "Random forest and logistic regression models run on flow features, and "
                             "an optional 1D-CNN runs on packet sequences. The best model by macro F1 is kept."),
                   unsafe_allow_html=True)
        c.markdown(info_card("3. Schedule", "Predicted classes drive a QoS scheduler. Latency-sensitive traffic "
                             "goes first, and flows the model is unsure about are treated as best-effort."),
                   unsafe_allow_html=True)
        st.write("")
        st.subheader("See what the classifier sees")
        st.caption("Pick a traffic type. These sequences are generated for illustration, not taken from your data.")
        kind = st.radio("Traffic type", list(ARCHETYPES), horizontal=True, label_visibility="collapsed")
        if "seed" not in st.session_state:
            st.session_state.seed = 7
        if st.button("Generate another flow"):
            st.session_state.seed += 1
        seq = packets(kind, st.session_state.seed)
        left, right = st.columns([3, 2])
        with left:
            chart(st, px.bar(seq, x="Packet", y="Size (bytes)", color="Direction",
                             color_discrete_map={"Upstream": CYAN, "Downstream": AMBER},
                             title=f"{kind}: packet size by direction"), 340)
        with right:
            info = ARCHETYPES[kind]
            st.markdown(info_card(kind, info["note"]), unsafe_allow_html=True)
            st.write("")
            kpi_row(st, [kpi("Mean packet", f"{seq['Size (bytes)'].abs().mean():.0f} B"),
                         kpi("Mean gap", f"{seq['Gap (ms)'].mean():.0f} ms"),
                         kpi("Downstream", f"{(seq['Direction'] == 'Downstream').mean():.0%}")])
            st.write("")
            st.markdown(info_card("How it is scheduled", f"Priority {info['prio']} of {len(ARCHETYPES)}. "
                                  f"Deadline about {info['deadline']:,} ms."), unsafe_allow_html=True)
        with st.expander("Why evaluate on separate captures?"):
            st.write("Packets from one capture session look alike, so a random split lets the model memorise "
                     "sessions and inflates scores. Holding out whole capture groups tests whether it works on "
                     "traffic it has never seen.")
        with st.expander("Glossary"):
            st.markdown("- **Macro F1:** F1 averaged over classes, so small classes count equally.\n"
                        "- **p95 latency:** 95% of requests finished at least this fast.\n"
                        "- **Deadline miss rate:** share of requests that finished after their deadline.\n"
                        "- **Best-effort:** served after prioritised traffic, with no guarantee.")

    # ------------------------------------------------------------ playground
    with t_play:
        st.subheader("Scheduling playground")
        st.caption("A toy queue on one shared link. Change the load and compare how the two policies treat each class.")
        with st.container(border=True):
            p1, p2, p3 = st.columns(3)
            bw = p1.slider("Link bandwidth (Mbps)", 5, 200, 80, step=5)
            nflows = p2.slider("Number of flows", 10, 150, 60, step=10)
            window = p3.slider("Arrival window (s, 0 = all at once)", 0, 60, 15)
        sim = toy_schedule(float(bw), nflows, float(window), 11)
        miss = sim.groupby("Policy")["Met deadline"].mean().rsub(1).mul(100)
        kpi_row(st, [kpi("Deadline misses, first come first served", f"{miss.iloc[0]:.0f}%", "", "bad"),
                     kpi("Deadline misses, priority by class", f"{miss.iloc[1]:.0f}%", "", "good")])
        st.write("")
        per = sim.groupby(["Policy", "Class"], as_index=False)["Latency (ms)"].mean()
        order = list(ARCHETYPES)
        chart(st, px.bar(per, x="Class", y="Latency (ms)", color="Policy", barmode="group", log_y=True,
                         category_orders={"Class": order}, title="Mean latency by class (log scale)"), 380)
        st.caption("Raise the load until the first-come policy starts missing VoIP deadlines, "
                   "then see how priority protects them.")

    # ------------------------------------------------------------ overview
    with t_over:
        st.subheader("Experiment status")
        if metrics:
            models = metrics.get("models", {})
            best = max(models, key=lambda n: models[n].get("macro_f1") or 0) if models else None
            f1 = models[best].get("macro_f1") if best else None
            kpi_row(st, [kpi("Dataset", metrics.get("dataset", "demo")),
                         kpi("Split", metrics.get("split_strategy", "stratified holdout")),
                         kpi("Capture groups", len(metrics.get("train_groups", [])) + len(metrics.get("test_groups", []))),
                         kpi("Train / test rows", f"{metrics.get('train_rows', '—')} / {metrics.get('test_rows', '—')}"),
                         kpi("Best macro F1", f"{f1:.3f}" if f1 is not None else "—", best or "", "info")])
            st.write("")
            if metrics.get("group_overlap"):
                st.error(f"Capture leakage detected: {metrics['group_overlap']}")
            elif "group_overlap" in metrics:
                st.success("Capture and session groups are disjoint across train and test.")
        else:
            empty(st, "No evaluation selected", "Choose an evaluation file in the sidebar, or run the demo to create one.")
        if not scheduler.empty:
            st.write("")
            kpi_row(st, [kpi("Requests replayed", f"{len(scheduler) // max(scheduler['policy'].nunique(), 1):,}",
                             "Queueing simulation, not packet-level emulation"),
                         kpi("Policies compared", scheduler["policy"].nunique())])
        st.caption("Synthetic data only confirms the workflow runs. Only capture-separated ISCX or QUIC metrics "
                   "support claims about a dataset.")

    # ------------------------------------------------------------ classification
    with t_cls:
        st.subheader("Classification results")
        if not metrics or "models" not in metrics:
            empty(st, "No evaluation loaded", "Pick an evaluation JSON file in the sidebar to compare models.")
        else:
            score = pd.DataFrame([{"Model": n, "Accuracy": r.get("accuracy"), "Macro F1": r.get("macro_f1")}
                                  for n, r in metrics["models"].items()]).sort_values("Macro F1", ascending=False)
            left, right = st.columns([3, 2])
            with left:
                chart(st, px.bar(score.melt(id_vars="Model", var_name="Metric", value_name="Score"), x="Model",
                                 y="Score", color="Metric", barmode="group", range_y=[0, 1],
                                 title="Model comparison"), 320)
            with right:
                st.write("")
                st.dataframe(score, use_container_width=True, hide_index=True,
                             column_config={"Accuracy": prog(), "Macro F1": prog()})
            name = st.selectbox("Model to inspect", list(metrics["models"]))
            result, labels = metrics["models"][name], metrics.get("classes", [])
            if result.get("confusion_matrix") and labels:
                share = st.toggle("Show each row as a share of its actual class")
                z = np.asarray(result["confusion_matrix"], dtype=float)
                if share:
                    z = z / np.maximum(z.sum(axis=1, keepdims=True), 1)
                left, right = st.columns([3, 2])
                with left:
                    chart(st, px.imshow(z, x=labels, y=labels, text_auto=".0%" if share else True,
                                        color_continuous_scale=HEAT, aspect="auto",
                                        labels={"x": "Predicted", "y": "Actual", "color": "Share" if share else "Flows"},
                                        title=f"Confusion matrix: {name}"), 460)
                with right:
                    report = result.get("classification_report", {})
                    st.markdown("**Per-class scores**")
                    st.dataframe(pd.DataFrame([{"Class": l, **report.get(l, {})} for l in labels]),
                                 use_container_width=True, hide_index=True, height=420,
                                 column_config={"precision": prog(), "recall": prog(), "f1-score": prog()})
            with st.expander("Split audit"):
                st.json({"group_column": metrics.get("group_column"), "train_groups": metrics.get("train_groups"),
                         "test_groups": metrics.get("test_groups"), "overlap": metrics.get("group_overlap")})

    # ------------------------------------------------------------ scheduler
    with t_sched:
        st.subheader("Concurrent request scheduling")
        if scheduler.empty:
            empty(st, "No scheduler results yet", "Run <code>traffic-qos demo</code>, then reload.")
        else:
            chosen = st.multiselect("Policies to show", sorted(scheduler["policy"].unique()),
                                    default=sorted(scheduler["policy"].unique()))
            view = scheduler[scheduler["policy"].isin(chosen)]
            if view.empty:
                empty(st, "No policies selected", "Pick at least one policy above.")
            else:
                rows = [{"Policy": p, "Mean latency (ms)": g["latency_ms"].mean(),
                         "p95 latency (ms)": g["latency_ms"].quantile(.95),
                         "Deadline miss rate (%)": (1 - g["deadline_met"].mean()) * 100,
                         "Requests": g["request_id"].nunique()} for p, g in view.groupby("policy")]
                summary = pd.DataFrame(rows)
                bl, bd = summary.loc[summary["p95 latency (ms)"].idxmin()], summary.loc[summary["Deadline miss rate (%)"].idxmin()]
                kpi_row(st, [kpi("Lowest p95 latency", bl["Policy"], f"{bl['p95 latency (ms)']:.1f} ms", "good"),
                             kpi("Fewest deadline misses", bd["Policy"], f"{bd['Deadline miss rate (%)']:.1f}% missed", "good"),
                             kpi("Policies shown", len(summary))])
                st.write("")
                a, b = st.columns(2)
                with a:
                    chart(st, px.bar(summary, x="Policy", y="p95 latency (ms)", color="Policy",
                                     title="p95 latency by policy").update_layout(showlegend=False))
                with b:
                    chart(st, px.bar(summary, x="Policy", y="Deadline miss rate (%)", color="Policy",
                                     title="Deadline misses by policy").update_layout(showlegend=False))
                per_class = view.groupby(["policy", "traffic_class"], as_index=False).agg(mean_latency_ms=("latency_ms", "mean"))
                chart(st, px.bar(per_class, x="traffic_class", y="mean_latency_ms", color="policy", barmode="group",
                                 title="Mean latency by traffic class",
                                 labels={"traffic_class": "Traffic class", "mean_latency_ms": "Mean latency (ms)"}), 380)
                st.dataframe(summary, use_container_width=True, hide_index=True, column_config=policy_cfg)
                with st.expander("Request-level results"):
                    st.dataframe(view, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------ predict
    with t_pred:
        st.subheader("Classify an uploaded flow CSV")
        if predictions_upload is None:
            empty(st, "Upload a flow CSV", "Use the sidebar uploader. The file needs the aggregate flow feature columns.")
        else:
            try:
                from traffic_qos.features import FEATURE_COLUMNS
                from traffic_qos.model import predict_flows
                frame = pd.read_csv(predictions_upload)
                missing = set(FEATURE_COLUMNS) - set(frame.columns)
                if missing:
                    st.error(f"Missing feature columns: {', '.join(sorted(missing))}")
                elif model_path.exists():
                    predicted = predict_flows(model_path, frame)
                    kpi_row(st, [kpi("Flows classified", f"{len(predicted):,}"),
                                 kpi("Classes found", predicted["predicted_class"].nunique()),
                                 kpi("Mean confidence", f"{predicted['confidence'].mean():.2f}", "", "good"),
                                 kpi("Lowest confidence", f"{predicted['confidence'].min():.2f}", "", "warn")])
                    st.write("")
                    counts = predicted["predicted_class"].value_counts().rename_axis("Class").reset_index(name="Flows")
                    l, r = st.columns(2)
                    with l:
                        chart(st, px.pie(counts, names="Class", values="Flows", hole=.55, title="Predicted class mix"), 340)
                    with r:
                        chart(st, px.histogram(predicted, x="confidence", nbins=20, range_x=[0, 1],
                                               title="Confidence distribution").update_layout(bargap=.06, yaxis_title="Flows"), 340)
                    f1c, f2c = st.columns([2, 1])
                    keep = f1c.multiselect("Show classes", sorted(predicted["predicted_class"].unique()),
                                           default=sorted(predicted["predicted_class"].unique()))
                    floor = f2c.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
                    shown = predicted[predicted["predicted_class"].isin(keep) & (predicted["confidence"] >= floor)]
                    st.caption(f"Showing {len(shown):,} of {len(predicted):,} flows.")
                    st.dataframe(shown, use_container_width=True, hide_index=True,
                                 column_config={"confidence": st.column_config.ProgressColumn(
                                     min_value=0, max_value=1, format="%.2f")})
                    st.download_button("Download predictions", predicted.to_csv(index=False),
                                       file_name="flow_predictions.csv", mime="text/csv")

                    from traffic_qos.schedule_flows import run_schedule
                    st.divider()
                    st.subheader("QoS service for these flows")
                    with st.container(border=True):
                        c1, c2, c3 = st.columns(3)
                        bandwidth = c1.slider("Link bandwidth (Mbps)", 10, 1000, 50, step=10)
                        arrival = c2.selectbox("Arrival pattern", ["random", "even", "burst", "auto"],
                                               help="burst = all flows at once (worst case); auto = use start-time column if present")
                        window_s = c3.number_input("Arrival window (s)", min_value=1.0, max_value=600.0, value=30.0)
                        threshold = st.slider("Confidence threshold (below this, a flow is treated as best-effort)",
                                              0.0, 1.0, 0.60, 0.05)
                    with tempfile.TemporaryDirectory() as tmp:
                        result = run_schedule(predicted, predicted, tmp, float(bandwidth), arrival,
                                              float(window_s), float(threshold))
                        flow_qos = pd.read_csv(Path(tmp) / "flow_qos.csv")
                        results = pd.read_csv(Path(tmp) / "scheduler_results.csv")
                    if "per-class default" in result["size_source"]:
                        st.warning("No byte-count columns found, so typical per-class sizes are used instead of real sizes.")
                    else:
                        st.caption(f"Flow sizes come from {result['size_source']} for {result['flows']} flows.")
                    table = pd.DataFrame([{"Policy": p["policy"], "Mean latency (ms)": p["mean_latency_ms"],
                                           "p95 latency (ms)": p["p95_latency_ms"],
                                           "Deadline miss rate (%)": p["deadline_miss_rate"] * 100,
                                           "Throughput (Mbps)": p["throughput_mbps"]} for p in result["policies"]])
                    st.dataframe(table, use_container_width=True, hide_index=True, column_config=policy_cfg)
                    per_class = results.groupby(["policy", "traffic_class"], as_index=False).agg(
                        mean_latency_ms=("latency_ms", "mean"))
                    l, r = st.columns(2)
                    with l:
                        chart(st, px.bar(table, x="Policy", y="p95 latency (ms)", color="Policy",
                                         title="p95 latency by policy").update_layout(showlegend=False))
                    with r:
                        chart(st, px.bar(per_class, x="traffic_class", y="mean_latency_ms", color="policy",
                                         barmode="group", title="Mean latency by traffic class"))
                    st.markdown("**Service each of your flows received (model-guided policy)**")
                    st.dataframe(flow_qos, use_container_width=True, hide_index=True)
                    st.download_button("Download per-flow QoS results", flow_qos.to_csv(index=False),
                                       file_name="flow_qos.csv", mime="text/csv")
                else:
                    st.warning(f"Model not found: {model_path}. Train first with `traffic-qos train`.")
            except Exception as exc:  # User-provided model/input errors are displayed in the dashboard.
                st.error(str(exc))


if __name__ == "__main__":
    main()