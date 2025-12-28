import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from micromlgen import port

def train_and_evaluate():
    x = np.load('features.npy')
    y = np.load('labels.npy')

    print(f"Dataset: {x.shape[0]} samples, {x.shape[1]} features")

    # 80% train, 20% test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalize features IMPORTANT
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Train SVM with RBF kernel
    print("\nTraining SVM...")
    svm = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
    svm.fit(x_train_scaled, y_train)

    # Evaluate
    y_pred = svm.predict(x_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTraining complete!")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Support Vectors: {len(svm.support_)}")
    
    # Confusion matrix
    # Fancy grafika
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Accuracy: {accuracy*100:.1f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    print("Saved confusion_matrix.png")
    
    return svm, scaler

def export_to_esp32(svm, scaler):
    """Export SVM model to C code for ESP32"""
    
    # Define gesture names (adjust to match your data)
    classmap = {
        0: 'rest',
        1: 'fist',
        2: 'index',
        3: 'peace',
        4: 'thumbs_up',
        5: 'ok',
        6: 'gang_gang'
    }
    
    # Export SVM 
    c_code = port(svm, classmap=classmap)
    
    # Save to header file
    with open('svm_model.h', 'w') as f:
        f.write(c_code)
    
    print("\nExported SVM model to svm_model.h")
    
    # Export scaler parameters
    with open('scaler_params.h', 'w') as f:
        f.write("// Feature scaling parameters\n")
        f.write(f"const float SCALER_MEAN[{len(scaler.mean_)}] = {{")
        f.write(', '.join([f'{x:.6f}' for x in scaler.mean_]))
        f.write("};\n\n")
        
        f.write(f"const float SCALER_SCALE[{len(scaler.scale_)}] = {{")
        f.write(', '.join([f'{x:.6f}' for x in scaler.scale_]))
        f.write("};\n")
    
    print("Exported scaler to scaler_params.h")

if __name__ == '__main__':
    svm, scaler = train_and_evaluate()
    export_to_esp32(svm, scaler)
    print("\nReady for ESP32 deployment!")
