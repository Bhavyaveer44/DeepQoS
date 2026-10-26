"""Train and apply payload-blind flow classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from .features import FEATURE_COLUMNS


def train_baseline(frame: pd.DataFrame, output_dir: str | Path, seed: int = 7) -> dict:
    """Train RF and logistic baselines; save the stronger model and evaluation artifacts."""
    if "label" not in frame:
        raise ValueError("Training data needs a 'label' column")
    x = frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = frame["label"].astype(str)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=seed, stratify=y
    )
    candidates = {
        "random_forest": RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=seed),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    }
    evaluated = {}
    for name, classifier in candidates.items():
        classifier.fit(x_train, y_train)
        predicted = classifier.predict(x_test)
        evaluated[name] = {
            "model": classifier,
            "accuracy": float(accuracy_score(y_test, predicted)),
            "macro_f1": float(f1_score(y_test, predicted, average="macro")),
            "report": classification_report(y_test, predicted, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, predicted, labels=sorted(y.unique())).tolist(),
            "labels": sorted(y.unique()),
        }
    selected_name = max(evaluated, key=lambda name: evaluated[name]["macro_f1"])
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(evaluated[selected_name]["model"], destination / "flow_classifier.joblib")
    summary = {name: {key: value for key, value in result.items() if key != "model"} for name, result in evaluated.items()}
    summary["selected_model"] = selected_name
    (destination / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def predict_flows(model_path: str | Path, frame: pd.DataFrame) -> pd.DataFrame:
    model = joblib.load(model_path)
    x = frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)
    result = frame.copy()
    result["predicted_class"] = model.predict(x)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
        result["confidence"] = probabilities.max(axis=1)
    else:
        result["confidence"] = 1.0
    return result


def predict_cnn(model_path: str | Path, frame: pd.DataFrame) -> pd.DataFrame:
    """Apply a saved packet-sequence CNN and its matching preprocessing metadata."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("CNN prediction requires TensorFlow. Install with: pip install -e '.[cnn]'") from exc
    if "packet_sequence" not in frame:
        raise ValueError("CNN prediction input requires packet_sequence column")
    model_file = Path(model_path)
    metadata = json.loads((model_file.parent / "cnn_preprocessing.json").read_text(encoding="utf-8"))
    sequences = [json.loads(value) if isinstance(value, str) else value for value in frame["packet_sequence"]]
    max_len = int(metadata["max_len"])
    values = np.zeros((len(sequences), max_len, 3), dtype=np.float32)
    for index, sequence in enumerate(sequences):
        sequence = np.asarray(sequence, dtype=np.float32)[:max_len]
        values[index, :len(sequence)] = sequence
    values = (values - np.asarray(metadata["center"], dtype=np.float32)) / np.asarray(metadata["scale"], dtype=np.float32)
    model = tf.keras.models.load_model(model_file)
    probabilities = model.predict(values, verbose=0)
    labels = np.asarray(metadata["classes"])
    result = frame.copy()
    result["predicted_class"] = labels[probabilities.argmax(axis=1)]
    result["confidence"] = probabilities.max(axis=1)
    return result


def evaluate_cnn_grouped(frame: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray,
                         output_dir: str | Path, dataset_name: str, seed: int = 7) -> dict:
    """Train/evaluate 1D-CNN on a precomputed capture-group holdout split."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("CNN evaluation requires TensorFlow. Install with: pip install -e '.[cnn]'") from exc
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from sklearn.preprocessing import LabelEncoder

    tf.keras.utils.set_random_seed(seed)
    sequences = [json.loads(value) if isinstance(value, str) else value for value in frame["packet_sequence"]]
    max_len = min(40, max(map(len, sequences)))
    x = np.zeros((len(sequences), max_len, 3), dtype=np.float32)
    for index, sequence in enumerate(sequences):
        sequence = np.asarray(sequence, dtype=np.float32)[:max_len]
        x[index, :len(sequence)] = sequence
    encoder = LabelEncoder()
    y = encoder.fit_transform(frame["label"].astype(str))
    # Fail explicitly when a class appears only in one capture partition.
    if set(y[train_idx]) != set(y[test_idx]):
        raise ValueError("Grouped CNN split must contain every class in both partitions; adjust capture groups or test size")
    center = np.mean(x[train_idx].reshape(-1, 3), axis=0)
    scale = np.maximum(np.std(x[train_idx].reshape(-1, 3), axis=0), 1e-6)
    center[0], scale[0] = 0.0, 1.0
    x = (x - center) / scale
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_len, 3)),
        tf.keras.layers.Conv1D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling1D(2), tf.keras.layers.Conv1D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.GlobalAveragePooling1D(), tf.keras.layers.Dropout(.25),
        tf.keras.layers.Dense(len(encoder.classes_), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x[train_idx], y[train_idx], epochs=20, batch_size=32, validation_split=.15, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])
    probabilities = model.predict(x[test_idx], verbose=0)
    predicted = probabilities.argmax(axis=1)
    labels = list(range(len(encoder.classes_)))
    result = {
        "accuracy": float(accuracy_score(y[test_idx], predicted)),
        "macro_f1": float(f1_score(y[test_idx], predicted, average="macro", zero_division=0)),
        "classification_report": classification_report(y[test_idx], predicted, labels=labels,
                                                         target_names=encoder.classes_.tolist(), output_dict=True,
                                                         zero_division=0),
        "confusion_matrix": confusion_matrix(y[test_idx], predicted, labels=labels).tolist(),
    }
    output = Path(output_dir)
    model.save(output / f"{dataset_name}_cnn.keras")
    (output / f"{dataset_name}_cnn_preprocessing.json").write_text(json.dumps({"max_len": max_len,
        "center": center.tolist(), "scale": scale.tolist(), "classes": encoder.classes_.tolist()}, indent=2), encoding="utf-8")
    return result


def train_cnn(frame: pd.DataFrame, output_dir: str | Path, seed: int = 7) -> dict:
    """Optional TensorFlow 1D-CNN on padded packet sequences.

    Input packet sequences are JSON arrays of [direction, length, inter-arrival].
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("Install optional CNN dependencies with: pip install -e '.[cnn]'") from exc
    if "packet_sequence" not in frame or "label" not in frame:
        raise ValueError("CNN input requires packet_sequence and label columns")
    from sklearn.preprocessing import LabelEncoder
    sequences = [json.loads(value) if isinstance(value, str) else value for value in frame["packet_sequence"]]
    max_len = min(40, max(map(len, sequences)))
    values = np.zeros((len(sequences), max_len, 3), dtype=np.float32)
    for index, sequence in enumerate(sequences):
        sequence = np.asarray(sequence, dtype=np.float32)[:max_len]
        values[index, :len(sequence)] = sequence
    encoder = LabelEncoder()
    labels = encoder.fit_transform(frame["label"].astype(str))
    x_train, x_test, y_train, y_test = train_test_split(values, labels, test_size=.25, random_state=seed, stratify=labels)
    # Normalize size and timing while preserving direction sign; fit only on training data.
    scale = np.maximum(np.std(x_train.reshape(-1, 3), axis=0), 1e-6)
    center = np.mean(x_train.reshape(-1, 3), axis=0)
    center[0], scale[0] = 0.0, 1.0
    x_train = (x_train - center) / scale
    x_test = (x_test - center) / scale
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_len, 3)),
        tf.keras.layers.Conv1D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling1D(2), tf.keras.layers.Conv1D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.GlobalAveragePooling1D(), tf.keras.layers.Dropout(.25),
        tf.keras.layers.Dense(len(encoder.classes_), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=.15, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "flow_cnn.keras")
    (output / "cnn_preprocessing.json").write_text(json.dumps({"max_len": max_len,
        "center": center.tolist(), "scale": scale.tolist(), "classes": encoder.classes_.tolist()}, indent=2), encoding="utf-8")
    (output / "cnn_evaluation.json").write_text(json.dumps({"accuracy": float(accuracy), "loss": float(loss),
        "classes": encoder.classes_.tolist(), "max_sequence_length": max_len}, indent=2), encoding="utf-8")
    return {"accuracy": float(accuracy), "loss": float(loss), "classes": encoder.classes_.tolist()}

