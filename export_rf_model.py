import argparse

import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from process_data import process_all_data


def train_and_export(window_size: int, overlap: int, output_path: str, test_size: float):
    if overlap >= window_size:
        raise ValueError("overlap must be smaller than window_size")

    print("Extracting features from CSV files...")
    X, y, gesture_map = process_all_data(window_size=window_size, overlap=overlap)

    if len(X) == 0:
        raise RuntimeError("No samples were generated. Check your dataset folders and window settings.")

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    rf = RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    print("Training Random Forest...")
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    label_to_name = {int(idx): name for name, idx in gesture_map.items()}
    ordered_names = [label_to_name[i] for i in sorted(label_to_name)]

    print(f"Validation accuracy: {test_acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=ordered_names))

    # Refit on full dataset so exported model uses all available data.
    rf.fit(X, y)

    channels = X.shape[1] // (7 * 2)
    bundle = {
        "model": rf,
        "label_to_name": label_to_name,
        "window_size": int(window_size),
        "overlap": int(overlap),
        "step": int(window_size - overlap),
        "channels": int(channels),
        "bandpass_lowcut": 15.0,
        "bandpass_highcut_cap": 250.0,
        "feature_schema": "time_history_v1",
        "validation_accuracy": float(test_acc),
    }

    dump(bundle, output_path)
    print(f"Saved RF bundle to {output_path}")

    return bundle


def export_c_header(bundle, header_path: str):
    try:
        from micromlgen import port
    except ImportError:
        print("micromlgen is not installed, skipping C header export.")
        return

    classmap = {int(k): v for k, v in bundle["label_to_name"].items()}
    model = bundle["model"]

    c_code = port(model, classmap=classmap)
    with open(header_path, "w", encoding="utf-8") as f:
        f.write(c_code)

    print(f"Saved RF C header to {header_path}")


def main():
    parser = argparse.ArgumentParser(description="Train and export Random Forest model for realtime use.")
    parser.add_argument("--window-size", type=int, default=300, help="Window size in samples.")
    parser.add_argument("--overlap", type=int, default=250, help="Overlap in samples.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument(
        "--output",
        type=str,
        default="rf_realtime_bundle.joblib",
        help="Output path for serialized bundle.",
    )
    parser.add_argument(
        "--header-output",
        type=str,
        default="",
        help="Optional output path for micromlgen C header.",
    )
    args = parser.parse_args()

    bundle = train_and_export(
        window_size=args.window_size,
        overlap=args.overlap,
        output_path=args.output,
        test_size=args.test_size,
    )

    if args.header_output:
        export_c_header(bundle, args.header_output)


if __name__ == "__main__":
    main()