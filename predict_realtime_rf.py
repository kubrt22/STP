import argparse
import time
from collections import deque

import numpy as np
import serial
from joblib import load

from process_data import bandpass_filter, extract_features


def parse_sample(line: str, channels: int):
    parts = line.split(",")
    if len(parts) != channels:
        return None

    try:
        return np.array([float(x) for x in parts], dtype=np.float32)
    except ValueError:
        return None


def collect_calibration(ser, channels: int, seconds: float):
    samples = []
    start = time.perf_counter()

    while (time.perf_counter() - start) < seconds:
        if not ser.in_waiting:
            continue

        line = ser.readline().decode("utf-8", errors="ignore").strip()
        sample = parse_sample(line, channels)
        if sample is not None:
            samples.append(sample)

    elapsed = max(time.perf_counter() - start, 1e-6)
    fs_est = len(samples) / elapsed

    if len(samples) == 0:
        raise RuntimeError("No valid samples received during calibration.")

    arr = np.vstack(samples)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std == 0] = 1.0

    return arr, mean, std, fs_est


def estimate_highcut(fs: float, lowcut: float, cap: float):
    nyquist = fs * 0.5
    highcut = min(cap, nyquist - 1.0)
    if highcut <= lowcut:
        highcut = nyquist * 0.9
    if highcut <= lowcut:
        raise RuntimeError(f"Sampling rate too low for selected bandpass. fs={fs:.2f} Hz")
    return highcut


def main():
    parser = argparse.ArgumentParser(description="Realtime EMG gesture prediction with exported Random Forest model.")
    parser.add_argument("--model", default="rf_realtime_bundle.joblib", help="Path to exported RF bundle.")
    parser.add_argument("--port", default="COM4", help="Serial port (e.g., COM4).")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate.")
    parser.add_argument(
        "--calibration-seconds",
        type=float,
        default=3.0,
        help="Seconds of initial stream used for baseline normalization and fs estimate.",
    )
    parser.add_argument(
        "--smoothing",
        type=int,
        default=5,
        help="Majority vote window size for stable label output.",
    )
    parser.add_argument(
        "--print-every-step",
        action="store_true",
        help="Print prediction for each inference step (otherwise prints only when label changes).",
    )
    args = parser.parse_args()

    bundle = load(args.model)
    model = bundle["model"]
    label_to_name = {int(k): v for k, v in bundle["label_to_name"].items()}

    window_size = int(bundle.get("window_size", 350))
    overlap = int(bundle.get("overlap", 300))
    step = int(bundle.get("step", max(1, window_size - overlap)))
    channels = int(bundle.get("channels", 3))
    lowcut = float(bundle.get("bandpass_lowcut", 15.0))
    highcut_cap = float(bundle.get("bandpass_highcut_cap", 250.0))

    print(f"Loading model from {args.model}")
    print(f"Opening serial: {args.port} @ {args.baud}")

    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(2.0)

    try:
        print(f"Calibrating for {args.calibration_seconds:.1f}s. Keep your arm relaxed...")
        calibration_samples, baseline_mean, baseline_std, fs_est = collect_calibration(
            ser,
            channels=channels,
            seconds=args.calibration_seconds,
        )

        if len(calibration_samples) < window_size:
            print("Calibration window too short, collecting extra samples...")
            extra = []
            while len(calibration_samples) + len(extra) < window_size:
                if not ser.in_waiting:
                    continue
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                sample = parse_sample(line, channels)
                if sample is not None:
                    extra.append(sample)
            calibration_samples = np.vstack([calibration_samples, np.vstack(extra)])

        highcut = estimate_highcut(fs_est, lowcut, highcut_cap)
        print(f"Estimated sampling rate: {fs_est:.1f} Hz")
        print(f"Using window={window_size}, overlap={overlap}, step={step}")
        print("Live inference started. Press Ctrl+C to stop.")

        buffer = deque(calibration_samples[-window_size:], maxlen=window_size)
        vote_history = deque(maxlen=max(1, args.smoothing))
        prev_features = None
        step_counter = 0
        last_printed = None

        while True:
            if not ser.in_waiting:
                continue

            line = ser.readline().decode("utf-8", errors="ignore").strip()
            sample = parse_sample(line, channels)
            if sample is None:
                continue

            buffer.append(sample)
            step_counter += 1

            if len(buffer) < window_size or step_counter < step:
                continue

            step_counter = 0
            window_raw = np.vstack(buffer)
            window_norm = (window_raw - baseline_mean) / baseline_std
            window_filtered = bandpass_filter(
                window_norm,
                lowcut=lowcut,
                highcut=highcut,
                fs=fs_est,
            )

            current_features = extract_features(window_filtered)
            if prev_features is None:
                features = np.concatenate((current_features, current_features))
            else:
                features = np.concatenate((prev_features, current_features))
            prev_features = current_features

            features = features.reshape(1, -1)
            pred = int(model.predict(features)[0])
            pred_name = label_to_name.get(pred, str(pred))

            confidence = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[0]
                class_idx = int(np.where(model.classes_ == pred)[0][0])
                confidence = float(probs[class_idx])

            vote_history.append(pred)
            stable_pred = max(set(vote_history), key=vote_history.count)
            stable_name = label_to_name.get(int(stable_pred), str(stable_pred))

            should_print = args.print_every_step or (stable_name != last_printed)
            if should_print:
                if confidence is None:
                    print(f"Current sign: {stable_name} (raw: {pred_name})")
                else:
                    print(
                        f"Current sign: {stable_name} (raw: {pred_name}, confidence: {confidence * 100:.1f}%)"
                    )
                last_printed = stable_name

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()