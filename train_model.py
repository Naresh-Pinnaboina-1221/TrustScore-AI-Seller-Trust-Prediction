"""
train_model.py
==============
Trains TWO models on your real dataset:
  1. RandomForest Regressor  → predicts trust_score (0-200)
  2. GradientBoosting Classifier → predicts trust_label (0/1)

Run this ONCE before starting the Flask app:
    python train_model.py

Outputs (saved to ./models/):
  - trust_score_regressor.pkl
  - trust_label_classifier.pkl
  - scaler.pkl
  - feature_importance.json
  - model_metrics.json
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    classification_report, accuracy_score
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
CSV_PATH   = "enhanced_trust_score_marketplace_dataset.csv"
MODEL_DIR  = "models"
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"   → {len(df)} rows, {df.shape[1]} columns")

# ── FEATURE ENGINEERING (mirrors app.py logic) ────────────────────────────────
print("⚙️  Engineering features...")

df['complaint_ratio']      = df['complaint_count'] / (df['total_transactions'] + 1)
df['weighted_rating']      = df['avg_rating'] * df['rating_count']
df['risk_score']           = df['refund_rate'] + df['chargeback_rate'] + df['late_delivery_rate']
df['activity_score']       = df['account_age_days'] * df['inventory_count']
df['response_efficiency']  = 1 / (df['average_response_time_hours'] + 1)
df['log_transactions']     = np.log1p(df['total_transactions'])
df['log_inventory']        = np.log1p(df['inventory_count'])
df['rating_x_sentiment']   = df['avg_rating'] * df['review_sentiment_score']
df['fraud_x_chargeback']   = df['fraud_flag_history'] * df['chargeback_rate']

# ── FEATURE LIST ──────────────────────────────────────────────────────────────
FEATURES = [
    # Raw inputs
    'total_transactions', 'avg_rating', 'rating_count', 'refund_rate',
    'complaint_count', 'account_age_days', 'late_delivery_rate',
    'average_response_time_hours', 'inventory_count', 'price_variance_index',
    'review_sentiment_score', 'fraud_flag_history', 'chargeback_rate',
    'verification_status',
    # Engineered
    'complaint_ratio', 'weighted_rating', 'risk_score',
    'activity_score', 'response_efficiency',
    'log_transactions', 'log_inventory',
    'rating_x_sentiment', 'fraud_x_chargeback',
]

X = df[FEATURES].values
y_score = df['trust_score'].values
y_label = df['trust_label'].values

# ── TRAIN / TEST SPLIT ────────────────────────────────────────────────────────
X_train, X_test, ys_train, ys_test, yl_train, yl_test = train_test_split(
    X, y_score, y_label, test_size=0.2, random_state=RANDOM_STATE
)

# ── SCALE ─────────────────────────────────────────────────────────────────────
print("📏 Scaling features...")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── TRAIN REGRESSOR ───────────────────────────────────────────────────────────
print("🌲 Training RandomForest Regressor (trust_score)...")
reg = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
reg.fit(X_train_s, ys_train)
ys_pred = reg.predict(X_test_s)
mae  = mean_absolute_error(ys_test, ys_pred)
r2   = r2_score(ys_test, ys_pred)
print(f"   → MAE: {mae:.3f}  |  R²: {r2:.4f}")

# ── TRAIN CLASSIFIER ──────────────────────────────────────────────────────────
print("🚀 Training GradientBoosting Classifier (trust_label)...")
clf = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=RANDOM_STATE
)
clf.fit(X_train_s, yl_train)
yl_pred = clf.predict(X_test_s)
acc = accuracy_score(yl_test, yl_pred)
print(f"   → Accuracy: {acc*100:.2f}%")
print(classification_report(yl_test, yl_pred, target_names=["Untrusted","Trusted"]))

# ── FEATURE IMPORTANCE ────────────────────────────────────────────────────────
importance = dict(zip(FEATURES, reg.feature_importances_.tolist()))
importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

# ── SAVE ARTEFACTS ────────────────────────────────────────────────────────────
print("💾 Saving models to ./models/ ...")
with open(f"{MODEL_DIR}/trust_score_regressor.pkl",  "wb") as f: pickle.dump(reg,    f)
with open(f"{MODEL_DIR}/trust_label_classifier.pkl", "wb") as f: pickle.dump(clf,    f)
with open(f"{MODEL_DIR}/scaler.pkl",                 "wb") as f: pickle.dump(scaler, f)

metrics = {
    "regressor":  {"MAE": round(mae, 4), "R2": round(r2, 4)},
    "classifier": {"Accuracy": round(acc * 100, 2)},
    "n_train": len(X_train),
    "n_test":  len(X_test),
    "features": FEATURES,
}
with open(f"{MODEL_DIR}/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

with open(f"{MODEL_DIR}/feature_importance.json", "w") as f:
    json.dump(importance_sorted, f, indent=2)

# Save feature list for app.py to read
with open(f"{MODEL_DIR}/features.json", "w") as f:
    json.dump(FEATURES, f)

print("\n✅ Done! Files saved:")
for fname in os.listdir(MODEL_DIR):
    print(f"   models/{fname}")
print("\n▶  Now run:  python app.py")
