import serial
import time
import numpy as np
import os

PORT = 'COM4'
BAUD_RATE = 115200
GESTURE_NAMES = ['rest', 'fist', 'index', 'peace', 'thumbs_up', 'ok', 'gang_gang']
REPS_PER_GESTURE = 12
DURATION_PER_REP = 3  # seconds
CHANNELS = 3

def collect_gesture_data(gesture_name, rep_num, duration=3):
    """Collect EMG data for one gesture repetition"""
    ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    
    print(f"\nBudes delat: {gesture_name} (Rep {rep_num + 1})")
    print("3")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("1")
    time.sleep(1)
    print("Delej!!!!?!?!?     :P")
    
    data = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8').strip()
            try:
                values = [int(x) for x in line.split(',')]
                if len(values) == CHANNELS:
                    data.append(values)
            except:
                pass
    
    print("cHILL vRO")
    ser.close()
    return np.array(data)

def main():
    os.makedirs('emg_data', exist_ok=True)
    
    for gesture in GESTURE_NAMES:
        print(f"\n=== Collecting data for: {gesture} ===")
        time.sleep(2)
        
        for rep in range(REPS_PER_GESTURE):
            data = collect_gesture_data(gesture, rep, DURATION_PER_REP)
            
            filename = f'emg_data/{gesture}_{rep}.csv'
            np.savetxt(filename, data, delimiter=',', fmt='%d')
            print(f"Saved: {filename} ({len(data)} samples)")
            
            time.sleep(3)

if __name__ == '__main__':
    main()
