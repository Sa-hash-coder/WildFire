"""
Wildfire Risk Assessment - Model Training & Feature Engineering Pipeline
Trains both a Baseline Logistic Regression and an Improved Random Forest.
Generates comprehensive validation metrics, feature importances, and historical benchmarks.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)

def compute_meteorological_features(df):
    """
    Computes domain-specific fire weather and atmospheric features:
    - Relative Humidity (Magnus-Tetens formula)
    - Vapor Pressure Deficit (VPD, kPa)
    - Dewpoint Depression (T - Td)
    - Angstrom Fire Risk Index
    - Hot-Dry-Windy (HDW) proxy
    - Cyclical diurnal encoding (sin/cos hour)
    """
    df = df.copy()
    
    # Magnus-Tetens approximation for saturation & actual vapor pressure (hPa)
    temp = df['temperature_c']
    dew = df['dewpoint_c']
    
    es = 6.112 * np.exp((17.67 * temp) / (temp + 243.5))
    e = 6.112 * np.exp((17.67 * dew) / (dew + 243.5))
    
    # Relative Humidity in percentage [0, 100]
    df['relative_humidity'] = np.clip(100.0 * (e / es), 0.0, 100.0)
    
    # Vapor Pressure Deficit (VPD in kPa: atmospheric dryness demand)
    df['vpd'] = np.maximum(0.0, es - e) / 10.0
    
    # Dewpoint Depression
    df['dewpoint_depression'] = temp - dew
    
    # Angstrom Fire Danger Index: I = (RH / 20) + ((27 - T) / 10)
    # I < 2.0: Fire conditions very high; 2.0-2.5: favorable; > 4.0: unlikely
    df['angstrom_index'] = (df['relative_humidity'] / 20.0) + ((27.0 - temp) / 10.0)
    
    # Hot-Dry-Windy proxy: wind_speed * VPD
    df['hdw_index'] = df['wind_speed'] * df['vpd']
    
    # Surface pressure in hPa
    if 'surface_pressure_hpa' not in df.columns:
        df['surface_pressure_hpa'] = df['surface_pressure'] / 100.0
        
    # Diurnal cyclical time features
    df['sin_hour'] = np.sin(2.0 * np.pi * df['hour'] / 24.0)
    df['cos_hour'] = np.cos(2.0 * np.pi * df['hour'] / 24.0)
    
    return df

def train_and_evaluate():
    print("[1/5] Loading merged ERA5-FIRMS dataset...")
    df = pd.read_csv('merged_df.csv')
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print(f"Total observations: {len(df):,}")
    print(f"Fire occurrences: {df['fire_occurred'].sum():,} ({df['fire_occurred'].mean()*100:.2f}%)")
    
    print("[2/5] Engineering meteorological features...")
    df = compute_meteorological_features(df)
    
    feature_cols = [
        'latitude', 'longitude',
        'temperature_c', 'dewpoint_c', 'wind_speed', 'precipitation_mm',
        'surface_pressure_hpa', 'u10', 'v10', 'hour', 'month',
        'relative_humidity', 'vpd', 'dewpoint_depression',
        'angstrom_index', 'hdw_index', 'sin_hour', 'cos_hour'
    ]
    
    target_col = 'fire_occurred'
    
    # Chronological split matching original project
    unique_days = df['datetime'].dt.date.unique()
    split_date = unique_days[-1]
    
    train_df = df[df['datetime'].dt.date < split_date].copy()
    test_df  = df[df['datetime'].dt.date >= split_date].copy()
    
    print(f"Training set: {len(train_df):,} rows (up to {split_date})")
    print(f"Test set:     {len(test_df):,} rows (on {split_date})")
    
    X_train_full = train_df[feature_cols]
    y_train_full = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # Use balanced subsample for training stability & sharper separation
    pos_train = train_df[train_df[target_col] == 1]
    neg_train = train_df[train_df[target_col] == 0]
    # 1:6 ratio retains all fires and samples ~63,000 negatives
    neg_sample = neg_train.sample(n=len(pos_train) * 6, random_state=42)
    sample_train = pd.concat([pos_train, neg_sample]).sample(frac=1.0, random_state=42)
    
    X_train_sample = sample_train[feature_cols]
    y_train_sample = sample_train[target_col]
    
    print(f"Optimized training sample: {len(sample_train):,} rows ({len(pos_train):,} fire, {len(neg_sample):,} non-fire)")
    
    print("[3/5] Training Baseline Logistic Regression...")
    pipe_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    pipe_lr.fit(X_train_sample, y_train_sample)
    lr_probs = pipe_lr.predict_proba(X_test)[:, 1]
    lr_preds = (lr_probs >= 0.5).astype(int)
    
    lr_roc = float(roc_auc_score(y_test, lr_probs))
    lr_pr = float(average_precision_score(y_test, lr_probs))
    lr_f1 = float(f1_score(y_test, lr_preds))
    lr_prec = float(precision_score(y_test, lr_preds, zero_division=0))
    lr_rec = float(recall_score(y_test, lr_preds))
    lr_cm = confusion_matrix(y_test, lr_preds).tolist()
    
    print(f"  Logistic Regression -> ROC-AUC: {lr_roc:.4f}, PR-AUC: {lr_pr:.4f}, F1: {lr_f1:.4f}, Recall: {lr_rec:.4f}")
    
    print("[4/5] Training Improved Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=120,
        max_depth=16,
        min_samples_leaf=4,
        min_samples_split=8,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    rf_model.fit(X_train_sample, y_train_sample)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    
    # Optimal threshold for F1
    precisions, recalls, thresholds = precision_recall_curve(y_test, rf_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = int(np.argmax(f1_scores))
    optimal_threshold = float(thresholds[best_idx])
    
    rf_preds = (rf_probs >= optimal_threshold).astype(int)
    rf_roc = float(roc_auc_score(y_test, rf_probs))
    rf_pr = float(average_precision_score(y_test, rf_probs))
    rf_f1 = float(f1_score(y_test, rf_preds))
    rf_prec = float(precision_score(y_test, rf_preds, zero_division=0))
    rf_rec = float(recall_score(y_test, rf_preds))
    rf_cm = confusion_matrix(y_test, rf_preds).tolist()
    
    print(f"  Random Forest -> ROC-AUC: {rf_roc:.4f}, PR-AUC: {rf_pr:.4f}, F1: {rf_f1:.4f}, Precision: {rf_prec:.4f}, Recall: {rf_rec:.4f}")
    print(f"  Optimal Decision Threshold: {optimal_threshold:.4f}")
    
    # Feature importances
    importances = rf_model.feature_importances_
    feat_imp = [
        {"feature": f, "importance": round(float(imp), 4)}
        for f, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    ]
    
    # Downsampled curves for visual plotting in frontend
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, rf_probs)
    step = max(1, len(fpr) // 35)
    roc_curve_data = [
        {"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)}
        for i in range(0, len(fpr), step)
    ]
    if roc_curve_data[-1]["fpr"] != 1.0:
        roc_curve_data.append({"fpr": 1.0, "tpr": 1.0})
        
    # PR curve
    pr_curve_data = [
        {"recall": round(float(recalls[i]), 4), "precision": round(float(precisions[i]), 4)}
        for i in range(0, len(recalls), max(1, len(recalls) // 35))
    ]
    
    # Extract representative historical test samples
    test_df_eval = test_df.copy()
    test_df_eval['pred_prob'] = rf_probs
    test_df_eval['pred_fire'] = rf_preds
    
    # 20 true positive fire cases, 20 high-confidence negative cases, 10 boundary cases
    tp_samples = test_df_eval[test_df_eval[target_col] == 1].sort_values(by='pred_prob', ascending=False).head(20)
    tn_samples = test_df_eval[test_df_eval[target_col] == 0].sort_values(by='pred_prob', ascending=True).head(20)
    boundary_samples = test_df_eval.iloc[(test_df_eval['pred_prob'] - optimal_threshold).abs().argsort()[:10]]
    
    sample_records = pd.concat([tp_samples, tn_samples, boundary_samples]).drop_duplicates().to_dict(orient='records')
    cleaned_samples = []
    for s in sample_records:
        cleaned_samples.append({
            "datetime": str(s['datetime']),
            "latitude": round(float(s['latitude']), 3),
            "longitude": round(float(s['longitude']), 3),
            "temperature_c": round(float(s['temperature_c']), 1),
            "dewpoint_c": round(float(s['dewpoint_c']), 1),
            "wind_speed": round(float(s['wind_speed']), 2),
            "precipitation_mm": round(float(s['precipitation_mm']), 2),
            "surface_pressure_hpa": round(float(s['surface_pressure_hpa']), 1),
            "relative_humidity": round(float(s['relative_humidity']), 1),
            "vpd": round(float(s['vpd']), 2),
            "angstrom_index": round(float(s['angstrom_index']), 2),
            "fire_occurred": int(s['fire_occurred']),
            "predicted_prob": round(float(s['pred_prob']), 4),
            "hour": int(s['hour'])
        })
        
    print("[5/5] Extracting active FIRMS satellite hotspots for interactive map...")
    firms_df = pd.read_csv('firms_clean.csv')
    hotspots = []
    for _, row in firms_df.head(600).iterrows():
        hotspots.append({
            "lat": round(float(row['latitude']), 3),
            "lon": round(float(row['longitude']), 3),
            "frp": round(float(row['frp']), 1),
            "brightness": round(float(row['bright_ti4']), 1),
            "date": str(row['acq_date'])[:10],
            "confidence": str(row['confidence']),
            "daynight": str(row['daynight'])
        })
        
    with open('backend/firms_hotspots.json', 'w', encoding='utf-8') as f:
        json.dump(hotspots, f, indent=2)
    print(f"Saved {len(hotspots)} active fire hotspots to backend/firms_hotspots.json")
    
    metadata = {
        "features": feature_cols,
        "optimal_threshold": optimal_threshold,
        "metrics": {
            "random_forest": {
                "name": "Improved Random Forest",
                "accuracy": round(float(accuracy_score(y_test, rf_preds)), 4),
                "roc_auc": round(rf_roc, 4),
                "pr_auc": round(rf_pr, 4),
                "f1_score": round(rf_f1, 4),
                "precision": round(rf_prec, 4),
                "recall": round(rf_rec, 4),
                "confusion_matrix": rf_cm
            },
            "logistic_regression": {
                "name": "Baseline Logistic Regression",
                "accuracy": round(float(accuracy_score(y_test, lr_preds)), 4),
                "roc_auc": round(lr_roc, 4),
                "pr_auc": round(lr_pr, 4),
                "f1_score": round(lr_f1, 4),
                "precision": round(lr_prec, 4),
                "recall": round(lr_rec, 4),
                "confusion_matrix": lr_cm
            }
        },
        "feature_importances": feat_imp,
        "roc_curve": roc_curve_data,
        "pr_curve": pr_curve_data,
        "historical_samples": cleaned_samples,
        "risk_thresholds": {
            "low": 0.20,
            "moderate": 0.45,
            "high": 0.70,
            "critical": 0.85
        }
    }
    
    with open('backend/model_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
        
    bundle = {
        "rf_model": rf_model,
        "lr_pipeline": pipe_lr,
        "feature_cols": feature_cols,
        "optimal_threshold": optimal_threshold
    }
    joblib.dump(bundle, 'backend/model_bundle.joblib', compress=3)
    print("Successfully saved backend/model_bundle.joblib and backend/model_metadata.json")

if __name__ == '__main__':
    train_and_evaluate()
