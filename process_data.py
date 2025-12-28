import numpy as np
from scipy import signal
import glob
import os

def bandpass_filter(data, lowcut=20, highcut=450, fs=1000, order=4):
    """Apply bandpass filter to EMG signal"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data, axis=0)

def extract_features(emg_window):
    """
    Extract time-domain features from EMG window
    Input: emg_window shape (samples, 3 channels) its the thing later
    Output: feature vector (12 features total)
    """
    features = []
    
    for channel in range(emg_window.shape[1]):
        channel_data = emg_window[:, channel]
        
        mav = np.mean(np.abs(channel_data))
        
        rms = np.sqrt(np.mean(channel_data ** 2))
        
        wl = np.sum(np.abs(np.diff(channel_data)))
        
        threshold = 0.01 * np.max(np.abs(channel_data))
        zc = np.sum(np.diff(np.sign(channel_data - threshold)) != 0)
        
        features.extend([mav, rms, wl, zc])
    
    return np.array(features)

def process_all_data(window_size=200, overlap=100):
    X_features = []
    y_labels = []
    
    gesture_map = {}
    gesture_idx = 0
    
    for csv_file in glob.glob('emg_data/*.csv'):
        gesture = os.path.basename(csv_file)
        
        if gesture not in gesture_map:
            gesture_map[gesture] = gesture_idx
            gesture_idx += 1
        
        emg_data = np.loadtxt(csv_file, delimiter=',')
        
        filtered = bandpass_filter(emg_data)
        
        step = window_size - overlap
        for i in range(0, len(filtered) - window_size, step):
            window = filtered[i:i + window_size] # wtf co se tu deje pomoc pls
            features = extract_features(window)
            
            X_features.append(features)
            y_labels.append(gesture_map[gesture])
    
    return np.array(X_features), np.array(y_labels), gesture_map

if __name__ == '__main__':
    print("Extracting features from collected data...")
    x, y, classmap = process_all_data()
    
    np.save('features.npy', x)
    np.save('labels.npy', y)
    
    print(f"Feature vectors: {len(x)} ")
    print(f"Feature dimensions: {x.shape[1]}")
    print(f"Classes: {classmap}")
