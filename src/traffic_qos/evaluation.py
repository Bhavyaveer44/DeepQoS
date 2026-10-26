"""Capture/session-grouped evaluation for labeled ISCX/QUIC flow exports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS


def evaluate_grouped(frame: pd.DataFrame, output_dir: str | Path, group_column: str = "capture_id",
                     test_size: float = .25, seed: int = 7, dataset_name: str = "custom",
                     use_cnn: bool = False) -> dict:
    """Evaluate classifiers with groups isolated between train and test.

    Input must contain label, group_column, and tabular FEATURES. CNN additionally
    requires packet_sequence arrays. Writes all metrics and split membership for audit.
    """
    required = {"label", group_column, *FEATURE_COLUMNS}
    if use_cnn:
        required.add("packet_sequence")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {', '.join(sorted(missing))}")
    data = frame.dropna(subset=["label", group_column]).copy().reset_index(drop=True)
    if data[group_column].nunique() < 2:
        raise ValueError(f"At least two distinct {group_column} groups are required for grouped evaluation")
    y = data["label"].astype(str)
    groups = data[group_column].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(data, y, groups))
    train_groups, test_groups = set(groups.iloc[train_idx]), set(groups.iloc[test_idx])
    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError("Group leakage detected between train and test partitions")
    classes_all = set(y)
    missing_train = classes_all - set(y.iloc[train_idx])
    missing_test = classes_all - set(y.iloc[test_idx])
    if missing_train or missing_test:
        raise ValueError(
            "Capture-group split does not represent every class in both partitions. "
            f"Missing from train: {sorted(missing_train)}; missing from test: {sorted(missing_test)}. "
            "Use more capture groups per class or revise test_size/grouping."
        )
    x = data[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)
    labels = sorted(y.unique())
    metrics: dict = {
        "dataset": dataset_name, "split_strategy": "GroupShuffleSplit",
        "group_column": group_column, "test_size_requested": test_size,
        "train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)),
        "train_groups": sorted(train_groups), "test_groups": sorted(test_groups),
        "group_overlap": sorted(overlap), "classes": labels, "models": {},
    }
    candidates = {
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed,
                                                  n_jobs=-1),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000,
                                                                                     class_weight="balanced")),
    }
    for name, model in candidates.items():
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        predictions = model.predict(x.iloc[test_idx])
        metrics["models"][name] = {
            "accuracy": float(accuracy_score(y.iloc[test_idx], predictions)),
            "macro_f1": float(f1_score(y.iloc[test_idx], predictions, average="macro", zero_division=0)),
            "classification_report": classification_report(y.iloc[test_idx], predictions, labels=labels,
                                                              output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y.iloc[test_idx], predictions, labels=labels).tolist(),
        }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics["selected_model"] = max(metrics["models"], key=lambda key: metrics["models"][key]["macro_f1"])
    (out / f"{dataset_name}_evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    prediction_frame = data.iloc[test_idx][["flow_id", group_column, "label"]].copy() if "flow_id" in data else data.iloc[test_idx][[group_column, "label"]].copy()
    selected = candidates[metrics["selected_model"]]
    prediction_frame["predicted_class"] = selected.predict(x.iloc[test_idx])
    if hasattr(selected, "predict_proba"):
        prediction_frame["confidence"] = selected.predict_proba(x.iloc[test_idx]).max(axis=1)
    prediction_frame.to_csv(out / f"{dataset_name}_test_predictions.csv", index=False)

    if use_cnn:
        from .model import evaluate_cnn_grouped
        metrics["models"]["cnn_1d"] = evaluate_cnn_grouped(data, train_idx, test_idx, out, dataset_name, seed)
        (out / f"{dataset_name}_evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_labeled_dataset(path: str | Path, group_column: str, label_column: str,
                         dataset_name: str, label_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Load CSV, map source labels, and attach dataset name as provenance."""
    frame = pd.read_csv(path)
    if label_column not in frame or group_column not in frame:
        raise ValueError(f"Input CSV must contain label column '{label_column}' and group column '{group_column}'")
    frame = frame.rename(columns={label_column: "label"})
    if label_map:
        normalized_map = {str(key): str(value) for key, value in label_map.items()}
        frame["label"] = frame["label"].astype(str).map(normalized_map).fillna(frame["label"].astype(str))
    frame["dataset"] = dataset_name
    return frame

