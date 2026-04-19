import numpy as np
from scipy import signal
import glob
import os


def bandpass_filter(data, lowcut=20, highcut=450, fs=1000, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    # Safeguard against highcut exceeding Nyquist frequency
    high = min(highcut / nyq, 0.99)
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data, axis=0)


def extract_features(emg_window):
    features = []
    for channel in range(emg_window.shape[1]):
        channel_data = emg_window[:, channel]

        mav = np.mean(np.abs(channel_data))
        rms = np.sqrt(np.mean(channel_data ** 2))
        wl = np.sum(np.abs(np.diff(channel_data)))

        # Zero crossings (with small noise threshold)
        threshold = 0.02 * np.max(np.abs(channel_data))
        zc = np.sum(np.diff(np.sign(channel_data - threshold)) != 0)

        # Slope sign changes (captures muscle firing frequency)
        ssc = np.sum(np.diff(np.sign(np.diff(channel_data))) != 0)

        var = np.var(channel_data)
        
        # Peak-to-peak amplitude (range)
        p2p = np.ptp(channel_data)

        features.extend([mav, rms, wl, zc, ssc, var, p2p])

    return np.array(features)


def process_all_data(window_size=300, overlap=250):
    X_features = []
    y_labels = []

    gesture_map = {}
    gesture_idx = 0

    for data_dir in ['emg_data1', 'emg_data2', 'emg_data3', 'emg_data4']:
        csv_files = sorted(glob.glob(f'{data_dir}/*.csv'))
        if not csv_files:
            continue

        # 1. Calculate session-wide statistics
        session_data_all = []
        for csv_file in csv_files:
            data = np.loadtxt(csv_file, delimiter=',')
            if len(data) > 0:
                session_data_all.append(data)
        
        session_data_combined = np.vstack(session_data_all)
        session_mean = session_data_combined.mean(axis=0)
        session_std = session_data_combined.std(axis=0)
        session_std[session_std == 0] = 1  # avoid division by zero

        # 2. Process each file using session statistics
        for csv_file in csv_files:
            gesture = os.path.basename(csv_file).split('_')[0]

            if gesture not in gesture_map:
                gesture_map[gesture] = gesture_idx
                gesture_idx += 1

            emg_data = np.loadtxt(csv_file, delimiter=',')

            if len(emg_data) < window_size:
                print(f"Skipping {csv_file} — too short ({len(emg_data)} samples)")
                continue

            # Calculate true sampling rate based on 3 seconds of recording
            actual_fs = len(emg_data) / 3.0
            highcut = min(250, (actual_fs / 2) - 10) # Safe highcut below Nyquist

            # Per-session normalization: using stats from the whole folder
            emg_data = (emg_data - session_mean) / session_std

            filtered = bandpass_filter(emg_data, lowcut=15, highcut=highcut, fs=actual_fs)


            n_samples = len(filtered)
            start_idx = int(n_samples * 0.2)
            end_idx = int(n_samples * 0.8)
            stable_data = filtered[start_idx:end_idx]

            step = window_size - overlap
            prev_features = None  # Store the previous window's features for time history
            
            for i in range(0, len(stable_data) - window_size, step):
                window = stable_data[i:i + window_size]
                current_features = extract_features(window)
                
                # Option 2: Add Time History
                # If this is the very first window, we duplicate the current features 
                # to mimic having "no change" from the past. Otherwise, concatenate past + current.
                if prev_features is None:
                    combined_features = np.concatenate((current_features, current_features))
                else:
                    combined_features = np.concatenate((prev_features, current_features))
                
                prev_features = current_features
                
                X_features.append(combined_features)
                y_labels.append(gesture_map[gesture])

    return np.array(X_features), np.array(y_labels), gesture_map


if __name__ == '__main__':
    print("Extracting features...")
    X, y, classmap = process_all_data()

    np.save('features.npy', X)
    np.save('labels.npy', y)

    print(f"Feature vectors: {len(X)}")
    print(f"Feature dimensions: {X.shape[1]}")
    print(f"Classes: {classmap}")

    for name, idx in classmap.items():
        count = np.sum(y == idx)
        print(f"  Class '{name}' (label {idx}): {count} samples")
