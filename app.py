"""
app.py — TrustScore AI v2.0
"""
import os, io, json, pickle, traceback
import numpy as np
import pandas as pd
from flask import (Flask, render_template, request,
                   jsonify, send_file, redirect, url_for, session)
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "trustscore-ai-2024"
os.makedirs("uploads", exist_ok=True)

MODEL_DIR = "models"

def load_models():
    with open(f"{MODEL_DIR}/trust_score_regressor.pkl",  "rb") as f: reg    = pickle.load(f)
    with open(f"{MODEL_DIR}/trust_label_classifier.pkl", "rb") as f: clf    = pickle.load(f)
    with open(f"{MODEL_DIR}/scaler.pkl",                 "rb") as f: scaler = pickle.load(f)
    with open(f"{MODEL_DIR}/features.json",              "r")  as f: feats  = json.load(f)
    with open(f"{MODEL_DIR}/model_metrics.json",         "r")  as f: metrics = json.load(f)
    with open(f"{MODEL_DIR}/feature_importance.json",    "r")  as f: importance = json.load(f)
    return reg, clf, scaler, feats, metrics, importance

try:
    reg_model, cls_model, scaler, FEATURES, METRICS, IMPORTANCE = load_models()
    print("Models loaded.")
except Exception as e:
    print(f"Model load error: {e}")
    reg_model = cls_model = scaler = FEATURES = METRICS = IMPORTANCE = None

RAW_COLS = [
    "total_transactions","avg_rating","rating_count","refund_rate",
    "complaint_count","account_age_days","late_delivery_rate",
    "average_response_time_hours","inventory_count","price_variance_index",
    "review_sentiment_score","fraud_flag_history","chargeback_rate",
    "verification_status",
]

def engineer(df):
    d = df.copy()
    d["complaint_ratio"]     = d["complaint_count"] / (d["total_transactions"] + 1)
    d["weighted_rating"]     = d["avg_rating"] * d["rating_count"]
    d["risk_score"]          = d["refund_rate"] + d["chargeback_rate"] + d["late_delivery_rate"]
    d["activity_score"]      = d["account_age_days"] * d["inventory_count"]
    d["response_efficiency"] = 1 / (d["average_response_time_hours"] + 1)
    d["log_transactions"]    = np.log1p(d["total_transactions"])
    d["log_inventory"]       = np.log1p(d["inventory_count"])
    d["rating_x_sentiment"]  = d["avg_rating"] * d["review_sentiment_score"]
    d["fraud_x_chargeback"]  = d["fraud_flag_history"] * d["chargeback_rate"]
    return d[FEATURES]

def tier_info(score):
    if score >= 145: return "Elite",    "#4ade80", 4
    if score >= 120: return "Good",     "#3de8c8", 3
    if score >= 90:  return "Moderate", "#f59e0b", 2
    return                  "High Risk","#f4607a", 1

def signals(row):
    out = []
    if row["avg_rating"] < 3.0:           out.append({"f":"Low Rating",         "t":"neg","v":f"{row['avg_rating']:.1f}/5"})
    if row["fraud_flag_history"] >= 2:    out.append({"f":"Fraud History",      "t":"neg","v":f"Level {int(row['fraud_flag_history'])}"})
    if row["refund_rate"] > 0.25:         out.append({"f":"High Refund Rate",   "t":"neg","v":f"{row['refund_rate']*100:.1f}%"})
    if row["chargeback_rate"] > 0.15:     out.append({"f":"High Chargebacks",   "t":"neg","v":f"{row['chargeback_rate']*100:.1f}%"})
    if row["late_delivery_rate"] > 0.3:   out.append({"f":"Late Deliveries",    "t":"neg","v":f"{row['late_delivery_rate']*100:.1f}%"})
    if row["verification_status"] == 1:   out.append({"f":"Verified Seller",    "t":"pos","v":"Yes"})
    if row["avg_rating"] >= 4.5:          out.append({"f":"Excellent Rating",   "t":"pos","v":f"{row['avg_rating']:.1f}/5"})
    if row["review_sentiment_score"]>0.5: out.append({"f":"Positive Sentiment", "t":"pos","v":f"{row['review_sentiment_score']:.2f}"})
    return out

@app.route("/")
def index():
    imp = list(IMPORTANCE.items())[:8] if IMPORTANCE else []
    return render_template("index.html", metrics=METRICS, top_features=imp, raw_cols=RAW_COLS)

@app.route("/predict/manual", methods=["POST"])
def predict_manual():
    if reg_model is None:
        return render_template("error.html", error="Models not loaded. Run train_model.py first.")
    try:
        raw = {c: float(request.form[c]) for c in RAW_COLS}
        df  = pd.DataFrame([raw])
        X   = scaler.transform(engineer(df).values)
        score = round(float(reg_model.predict(X)[0]), 2)
        label = int(cls_model.predict(X)[0])
        conf  = round(float(max(cls_model.predict_proba(X)[0]))*100, 1)
        t, tc, tl = tier_info(score)
        return render_template("result_single.html",
            score=score, label="Trusted" if label else "Untrusted",
            trusted=bool(label), tier=t, tier_color=tc, tier_level=tl,
            confidence=conf, sigs=signals(raw), raw=raw,
            bar_width=min(score/2,100), diff=round(score-116.82,2))
    except Exception as e:
        return render_template("error.html", error=str(e))

@app.route("/predict/bulk", methods=["POST"])
def predict_bulk():
    if reg_model is None:
        return render_template("error.html", error="Models not loaded.")
    if "file" not in request.files or request.files["file"].filename == "":
        return render_template("error.html", error="No file selected.")
    f = request.files["file"]
    fname = secure_filename(f.filename)
    ext = fname.rsplit(".",1)[-1].lower()
    try:
        df = pd.read_csv(f) if ext=="csv" else pd.read_excel(f) if ext in ("xlsx","xls") else None
        if df is None:
            return render_template("error.html", error="Only CSV or Excel files are supported.")
        missing = [c for c in RAW_COLS if c not in df.columns]
        if missing:
            return render_template("error.html", error=f"Missing columns: {', '.join(missing)}")
        work   = df[RAW_COLS].copy().fillna(0)
        X      = scaler.transform(engineer(work).values)
        scores = reg_model.predict(X).round(2)
        labels = cls_model.predict(X)
        probas = cls_model.predict_proba(X).max(axis=1)
        results = []
        for i in range(len(work)):
            sc = float(scores[i]); lb = int(labels[i])
            t, tc, tl = tier_info(sc)
            results.append({
                "row": i+1,
                "seller_id": str(df["seller_id"].iloc[i]) if "seller_id" in df.columns else str(i+1),
                "score": sc, "label":"Trusted" if lb else "Untrusted",
                "trusted":bool(lb), "tier":t, "tier_color":tc, "tier_level":tl,
                "confidence": round(float(probas[i])*100,1),
                "avg_rating": round(float(work["avg_rating"].iloc[i]),2),
                "fraud": int(work["fraud_flag_history"].iloc[i]),
                "verified": int(work["verification_status"].iloc[i]),
                "refund": round(float(work["refund_rate"].iloc[i]),4),
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        all_s = [r["score"] for r in results]
        summary = {
            "total":len(results),"mean":round(sum(all_s)/len(all_s),2),
            "max":round(max(all_s),2),"min":round(min(all_s),2),
            "elite":    sum(1 for r in results if r["tier"]=="Elite"),
            "good":     sum(1 for r in results if r["tier"]=="Good"),
            "moderate": sum(1 for r in results if r["tier"]=="Moderate"),
            "highrisk": sum(1 for r in results if r["tier"]=="High Risk"),
        }
        session["bulk_results"] = results
        return render_template("result_bulk.html", results=results, summary=summary, filename=fname)
    except Exception as e:
        return render_template("error.html", error=f"{str(e)}\n\n{traceback.format_exc()}")

@app.route("/download/results")
def download_results():
    results = session.get("bulk_results")
    if not results: return redirect(url_for("index"))
    rows = [{"Seller ID":r["seller_id"],"Trust Score":r["score"],"Label":r["label"],
             "Tier":r["tier"],"Confidence (%)":r["confidence"],"Avg Rating":r["avg_rating"],
             "Fraud Flag":r["fraud"],"Verified":"Yes" if r["verified"] else "No",
             "Refund Rate":r["refund"]} for r in results]
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode()), mimetype="text/csv",
                     as_attachment=True, download_name="trust_score_predictions.csv")

@app.route("/download/template")
def download_template():
    example = {"total_transactions":1500,"avg_rating":4.2,"rating_count":900,"refund_rate":0.05,
               "complaint_count":12,"account_age_days":1200,"late_delivery_rate":0.08,
               "average_response_time_hours":4.5,"inventory_count":5000,"price_variance_index":0.3,
               "review_sentiment_score":0.75,"fraud_flag_history":0,"chargeback_rate":0.01,"verification_status":1}
    buf = io.StringIO()
    pd.DataFrame([example]).to_csv(buf, index=False)
    buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode()), mimetype="text/csv",
                     as_attachment=True, download_name="upload_template.csv")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if reg_model is None: return jsonify({"error":"Models not loaded"}), 500
    try:
        data = request.get_json()
        X = scaler.transform(engineer(pd.DataFrame([data])).values)
        sc = round(float(reg_model.predict(X)[0]),2)
        lb = int(cls_model.predict(X)[0])
        t, tc, _ = tier_info(sc)
        return jsonify({"trust_score":sc,"trust_label":lb,"trusted":bool(lb),"tier":t})
    except Exception as e:
        return jsonify({"error":str(e)}), 400

@app.route("/metrics")
def metrics_page():
    top = list(IMPORTANCE.items())[:12] if IMPORTANCE else []
    return render_template("metrics.html", metrics=METRICS, top_features=top)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
