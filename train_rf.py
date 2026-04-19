import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time

def train_rf():
    # 1. Load the data
    X = np.load('features.npy')
    y = np.load('labels.npy')

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    gesture_names = ['rest', 'fist', 'index', 'peace', 'thumbs_up', 'ok', 'gang_gang']
    
    # 2. Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Hyper-tune the Random Forest
    print("\nStarting Hyperparameter Search for Random Forest...")
    start_time = time.time()
    
    # Base model
    rf_base = RandomForestClassifier(n_jobs=-1, random_state=42, class_weight='balanced')
    
    # Expanded Parameter grid for tuning (Massive search space)
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500, 600, 800],
        'max_depth': [10, 20, 30, 40, 50, 60, None],
        'min_samples_split': [2, 3, 5, 7, 10],
        'min_samples_leaf': [1, 2, 3, 4, 5],
        'max_features': ['sqrt', 'log2', None, 0.3, 0.5], # 0.3/0.5 means 30%/50% of features
        'criterion': ['gini', 'entropy'],
        'bootstrap': [True, False]
    }
    
    # Randomized Search (faster than checking every single combination)
    rf_search = RandomizedSearchCV(
        estimator=rf_base, 
        param_distributions=param_dist, 
        n_iter=100,          # Test 100 different random configurations instead of 20
        cv=5,                # Increased to 5-fold cross validation for very robust scoring
        verbose=1, 
        random_state=42, 
        n_jobs=-1            # Use all cores
    )
    
    # Fit the random search model
    rf_search.fit(X_train, y_train)
    print(f"\nHyperparameter tuning took {time.time() - start_time:.2f} seconds.")
    
    best_params = rf_search.best_params_
    print(f"Best Parameters Found: {best_params}")
    
    # -----------------------------------------------------------------
    # LOG RESULTS AND EXPORT TO CSV
    # -----------------------------------------------------------------
    print("\nExporting tuning results to spreadsheet...")
    import pandas as pd
    
    # Get the results dictionary from the random search
    cv_results = rf_search.cv_results_
    
    # Convert exactly the columns we care about into a DataFrame
    results_df = pd.DataFrame({
        'Rank': cv_results['rank_test_score'],
        'Mean_CV_Accuracy_%': np.round(cv_results['mean_test_score'] * 100, 2),
        'Std_CV_Acc_%': np.round(cv_results['std_test_score'] * 100, 2),
        'n_estimators': [params.get('n_estimators') for params in cv_results['params']],
        'max_depth': [params.get('max_depth') for params in cv_results['params']],
        'min_samples_split': [params.get('min_samples_split') for params in cv_results['params']],
        'min_samples_leaf': [params.get('min_samples_leaf') for params in cv_results['params']],
        'max_features': [params.get('max_features') for params in cv_results['params']],
        'criterion': [params.get('criterion') for params in cv_results['params']],
        'bootstrap': [params.get('bootstrap') for params in cv_results['params']],
    })
    
    # Sort by rank so the best models are at the top
    results_df = results_df.sort_values(by='Rank')
    
    # Save to CSV
    csv_filename = 'rf_hyperparameter_test_results.csv'
    results_df.to_csv(csv_filename, index=False, sep=';')
    print(f"Saved spreadsheet to: {csv_filename}\n")
    # -----------------------------------------------------------------

    # Use the best model found
    rf_model = rf_search.best_estimator_

    # 4. Evaluate on test set
    y_pred = rf_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print(f"FINAL TEST ACCURACY: {test_acc * 100:.2f}%")
    print("="*50 + "\n")
    
    print("Detailed Report:")
    print(classification_report(y_test, y_pred, target_names=gesture_names))

    # 5. Extract Feature Importances
    # We extracted 7 features per channel (MAV, RMS, WL, ZC, SSC, VAR, P2P)
    # BUT NOW we have history! So each feature exists twice (Past and Present)
    feature_types = ['MAV', 'RMS', 'WL', 'ZC', 'SSC', 'VAR', 'P2P']
    channels = X.shape[1] // (len(feature_types) * 2)
    
    feature_names = []
    
    # 1. First half of the array is the PREVIOUS window (t-1)
    for c in range(channels):
        for f in feature_types:
            feature_names.append(f"PAST_CH{c+1}_{f}")
            
    # 2. Second half of the array is the CURRENT window (t)
    for c in range(channels):
        for f in feature_types:
            feature_names.append(f"NOW_CH{c+1}_{f}")

    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nTop 10 Most Important Features:")
    
    # Also log feature importances to a separate CSV
    feature_ranking_data = []
    
    for i in range(len(indices)):  # Save all features to CSV, but only print top 10
        idx = indices[i]
        feature_name = feature_names[idx]
        score = importances[idx]
        
        if i < 10:
            print(f"{i+1}. {feature_name:<10} (Score: {score:.4f})")
            
        feature_ranking_data.append({
            'Rank': i + 1,
            'Feature_Name': feature_name,
            'Importance_Score': round(score, 6)
        })

    # Save features to CSV
    features_df = pd.DataFrame(feature_ranking_data)
    features_csv = 'rf_feature_importances.csv'
    features_df.to_csv(features_csv, index=False, sep=';')
    print(f"Saved all feature importances to: {features_csv}\n")

    # 6. Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_percent, annot=True, fmt='.1f', cmap='Greens',
        xticklabels=gesture_names, yticklabels=gesture_names,
        vmin=0, vmax=100
    )
    plt.title(f'Random Forest Confusion Matrix (Acc: {test_acc*100:.1f}%)')
    plt.ylabel('True Gesture')
    plt.xlabel('Predicted Gesture')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('rf_confusion_matrix.png', dpi=150)
    print("\nSaved rf_confusion_matrix.png")

if __name__ == '__main__':
    train_rf()
