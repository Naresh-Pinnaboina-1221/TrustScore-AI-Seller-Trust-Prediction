#  TrustScore AI — Seller Trust Prediction App

A production-ready Flask web app that predicts **marketplace seller trustworthiness**
using two trained ML models:
- **RandomForest Regressor** → predicts `trust_score` (0–200)
- **GradientBoosting Classifier** → predicts `trust_label` (Trusted / Untrusted)

---

##  Project Structure

```
trust_model_app/
│
├── app.py                  ← Flask web server (main entry point)
├── train_model.py          ← ML training script (run this first!)
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
│
├── enhanced_trust_score_marketplace_dataset.csv   ← Your training data
│
├── models/                 ← Auto-generated after training
│   ├── trust_score_regressor.pkl
│   ├── trust_label_classifier.pkl
│   ├── scaler.pkl
│   ├── features.json
│   ├── feature_importance.json
│   └── model_metrics.json
│
└── templates/
    ├── index.html          ← Input form
    ├── result.html         ← Prediction result
    ├── metrics.html        ← Model performance page
    └── error.html          ← Error page
```

---

##  Quick Start (5 minutes)

### Step 1 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Place your CSV file

Make sure `enhanced_trust_score_marketplace_dataset.csv` is in the same folder as `train_model.py`.

### Step 3 — Train the models

```bash
python train_model.py
```

You'll see output like:
```
Loading dataset...
→ 5000 rows, 20 columns
Engineering features...
Scaling features...
Training RandomForest Regressor (trust_score)...
→ MAE: 5.111  |  R²: 0.8583
Training GradientBoosting Classifier (trust_label)...
Accuracy: 99.50%
Done!
```

This creates all `.pkl` files in the `models/` folder.

### Step 4 — Start the web app

```bash
python app.py
```

### Step 5 — Open browser

Go to: **http://127.0.0.1:5000**

---

##  Model Performance

| Model | Algorithm | Metric | Value |
|-------|-----------|--------|-------|
| Trust Score Regressor | RandomForest (300 trees) | R² | 0.8583 |
| Trust Score Regressor | RandomForest (300 trees) | MAE | 5.11 pts |
| Trust Label Classifier | GradientBoosting (200 trees) | Accuracy | 99.5% |

Training set: **4,000 sellers**  
Test set: **1,000 sellers**

---

##  Features Used

### Raw Inputs (14)
| Feature | Description |
|---------|-------------|
| `total_transactions` | Total number of completed transactions |
| `avg_rating` | Average customer rating (0–5) |
| `rating_count` | Number of ratings received |
| `refund_rate` | Proportion of transactions refunded (0–1) |
| `complaint_count` | Number of complaints filed |
| `account_age_days` | Days since account creation |
| `late_delivery_rate` | Proportion of late deliveries (0–1) |
| `average_response_time_hours` | Avg hours to respond to customers |
| `inventory_count` | Number of items in inventory |
| `price_variance_index` | Price stability metric (0=stable, 1=volatile) |
| `review_sentiment_score` | NLP sentiment of reviews (−1 to +1) |
| `fraud_flag_history` | Fraud flag level (0=none, 4=critical) |
| `chargeback_rate` | Chargeback proportion (0–1) |
| `verification_status` | Whether seller is verified (0/1) |

### Engineered Features (9 additional)
| Feature | Formula |
|---------|---------|
| `complaint_ratio` | complaint_count / (total_transactions + 1) |
| `weighted_rating` | avg_rating × rating_count |
| `risk_score` | refund_rate + chargeback_rate + late_delivery_rate |
| `activity_score` | account_age_days × inventory_count |
| `response_efficiency` | 1 / (response_time + 1) |
| `log_transactions` | log(1 + total_transactions) |
| `log_inventory` | log(1 + inventory_count) |
| `rating_x_sentiment` | avg_rating × review_sentiment_score |
| `fraud_x_chargeback` | fraud_flag_history × chargeback_rate |

---

##  Pages

| URL | Description |
|-----|-------------|
| `http://localhost:5000/` | Prediction form |
| `http://localhost:5000/metrics` | Model performance & feature importance |
| `http://localhost:5000/api/predict` | JSON API (POST) |

---

##  JSON API Usage

Send a POST request to `/api/predict` with JSON body:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "total_transactions": 1500,
    "avg_rating": 4.7,
    "rating_count": 900,
    "refund_rate": 0.02,
    "complaint_count": 10,
    "account_age_days": 1200,
    "late_delivery_rate": 0.05,
    "average_response_time_hours": 3.5,
    "inventory_count": 5000,
    "price_variance_index": 0.3,
    "review_sentiment_score": 0.85,
    "fraud_flag_history": 0,
    "chargeback_rate": 0.01,
    "verification_status": 1
  }'
```

Response:
```json
{
  "trust_score": 148.23,
  "trust_label": 0,
  "trusted": false
}
```

---

##  Score Interpretation

| Score Range | Tier | Meaning |
|-------------|------|---------|
| 145 – 200 | 🟢 Elite | Exceptional seller, maximum trust |
| 120 – 144 | 🔵 Good | Reliable seller, above average |
| 90 – 119 | 🟡 Moderate | Average risk, monitor closely |
| 0 – 89 | 🔴 High Risk | Significant trust concerns |

> **Note:** The `trust_label` threshold in this dataset is very strict (score ≈ 200).
> Only 13 of 5,000 sellers (0.26%) have `trust_label = 1`.
> The score (0–200) is more actionable than the binary label.

---

##  Retraining on New Data

To retrain on a new dataset:

1. Replace `enhanced_trust_score_marketplace_dataset.csv` with your new file
2. Make sure it has the same column names
3. Run `python train_model.py` again
4. Restart `python app.py`

The app will automatically pick up the new model files.

---

##  Production Deployment

For production use, change `app.run(debug=True)` to:

```python
app.run(debug=False, host="0.0.0.0", port=5000)
```

Or use gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 app:app
```

---

##  Dependencies

- **Flask** — Web framework
- **scikit-learn** — ML models
- **pandas** — Data handling
- **numpy** — Numerical computing

---

