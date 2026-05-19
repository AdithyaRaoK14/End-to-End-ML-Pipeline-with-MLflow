# dashboard/app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from mlflow.tracking import MlflowClient
import time

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

MLFLOW_URI = "http://localhost:5000"
API_URL = "http://localhost:8000"

mlflow.set_tracking_uri(MLFLOW_URI)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 Credit Card Fraud Detection")
st.caption("End-to-End ML Pipeline · MLflow · FastAPI · Random Forest + SMOTE")
st.divider()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        ["📊 Experiment Results", "🔮 Live Prediction", "📈 Model Info"],
        label_visibility="collapsed"
    )

    st.divider()

    # API status indicator
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.status_code == 200:
            st.success("API: Online ✅")
        else:
            st.error("API: Error ❌")
    except Exception:
        st.error("API: Offline ❌")

    # MLflow status
    try:
        client = MlflowClient()
        client.search_experiments()
        st.success("MLflow: Online ✅")
    except Exception:
        st.error("MLflow: Offline ❌")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Experiment Results
# ════════════════════════════════════════════════════════════════════════════════
if page == "📊 Experiment Results":
    st.header("📊 MLflow Experiment Results")
    st.markdown(
        "Comparing 5 experiments across different imbalance handling strategies.")

    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(
            "credit-card-fraud-detection")

        if experiment is None:
            st.warning("No experiment found. Run train.py first.")
        else:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.recall_fraud DESC"]
            )

            if not runs:
                st.warning("No runs found.")
            else:
                # ── Build results dataframe ────────────────────────────────
                rows = []
                for run in runs:
                    m = run.data.metrics
                    p = run.data.params
                    rows.append({
                        "Run":           run.info.run_name,
                        "Sampling":      p.get("sampling_strategy", "-"),
                        "Model":         p.get("model_type", "-"),
                        "Recall":        round(m.get("recall_fraud", 0), 4),
                        "Precision":     round(m.get("precision_fraud", 0), 4),
                        "ROC-AUC":       round(m.get("roc_auc", 0), 4),
                        "PR-AUC":        round(m.get("pr_auc", 0), 4),
                        "Fraud Missed":  int(m.get("fraud_missed", 0)),
                        "False Alarms":  int(m.get("false_alarms", 0)),
                    })

                df = pd.DataFrame(rows)

                # ── Key insight callout ────────────────────────────────────
                best = df.loc[df["Recall"].idxmax()]
                worst = df.loc[df["Recall"].idxmin()]

                col1, col2, col3, col4 = st.columns(4)
                col1.metric(
                    "Best Recall",
                    f"{best['Recall']:.4f}",
                    f"+{best['Recall']-worst['Recall']:.4f} vs baseline"
                )
                col2.metric(
                    "Best ROC-AUC",
                    f"{df['ROC-AUC'].max():.4f}"
                )
                col3.metric(
                    "Min Fraud Missed",
                    f"{int(df['Fraud Missed'].min())} / 98"
                )
                col4.metric(
                    "Total Experiments",
                    len(df)
                )

                st.divider()

                # ── Results table ──────────────────────────────────────────
                st.subheader("All Runs")

                # Highlight best recall row
                def highlight_best(row):
                    if row["Recall"] == df["Recall"].max():
                        return ["background-color: #1a3a1a"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    df.style.apply(highlight_best, axis=1).format({
                        "Recall":    "{:.4f}",
                        "Precision": "{:.4f}",
                        "ROC-AUC":   "{:.4f}",
                        "PR-AUC":    "{:.4f}",
                    }),
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "🟢 Highlighted row = best recall (most fraud caught)")

                st.divider()

                # ── Charts ────────────────────────────────────────────────
                col_left, col_right = st.columns(2)

                with col_left:
                    st.subheader("Recall by Experiment")
                    fig, ax = plt.subplots(figsize=(7, 4))
                    colors = [
                        "#2ecc71" if r == df["Recall"].max()
                        else "#e74c3c" if r == df["Recall"].min()
                        else "#3498db"
                        for r in df["Recall"]
                    ]
                    bars = ax.barh(df["Run"], df["Recall"],
                                   color=colors, alpha=0.85)
                    ax.set_xlabel("Recall (Fraud)")
                    ax.set_xlim(0.7, 1.0)
                    ax.axvline(x=0.85, color='white', linestyle='--',
                               alpha=0.5, label='Target (0.85)')
                    ax.legend()
                    ax.set_facecolor('#0e1117')
                    fig.patch.set_facecolor('#0e1117')
                    ax.tick_params(colors='white')
                    ax.xaxis.label.set_color('white')
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#333')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                with col_right:
                    st.subheader("Fraud Missed vs False Alarms")
                    fig, ax = plt.subplots(figsize=(7, 4))
                    x = np.arange(len(df))
                    width = 0.35
                    ax.bar(x - width/2, df["Fraud Missed"],
                           width, label="Fraud Missed", color="#e74c3c", alpha=0.85)
                    ax.bar(x + width/2, df["False Alarms"],
                           width, label="False Alarms", color="#f39c12", alpha=0.85)
                    ax.set_xticks(x)
                    ax.set_xticklabels(
                        [r.replace("RF-", "").replace("-smote", "+S")
                         for r in df["Run"]],
                        rotation=20, ha='right', fontsize=8
                    )
                    ax.legend()
                    ax.set_facecolor('#0e1117')
                    fig.patch.set_facecolor('#0e1117')
                    ax.tick_params(colors='white')
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#333')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                st.divider()
                st.subheader("Key Finding")
                st.info(
                    f"**SMOTE oversampling reduced missed fraud from "
                    f"{int(worst['Fraud Missed'])} to {int(best['Fraud Missed'])} cases** "
                    f"out of 98 fraud transactions in the test set. "
                    f"This represents a {((worst['Fraud Missed']-best['Fraud Missed'])/worst['Fraud Missed']*100):.0f}% "
                    f"reduction in undetected fraud — directly translating to financial loss prevention."
                )

    except Exception as e:
        st.error(f"Could not load MLflow data: {e}")
        st.info("Make sure MLflow is running at localhost:5000")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Live Prediction
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Live Prediction":
    st.header("🔮 Live Transaction Prediction")
    st.markdown("Test the deployed model with real transaction data.")

    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.subheader("Transaction Input")

        # Quick fill buttons
        st.markdown("**Quick fill:**")
        qcol1, qcol2 = st.columns(2)

        # Real fraud transaction from dataset
        FRAUD_EXAMPLE = {
            "features": [
                -2.3122, 1.952, -1.6099, 3.9979, -0.5222,
                -1.4265, -2.5374, 1.3917, -2.7701, -2.7723,
                3.202, -2.8999, -0.5952, -4.2893, 0.3897,
                -1.1407, -2.8301, -0.0168, 0.417, 0.1269,
                0.5172, -0.035, -0.4652, 0.3202, 0.0445,
                0.1778, 0.2611, -0.1433
            ],
            "amount": 0.0,
            "time": 406.0
        }

        # Normal transaction
        NORMAL_EXAMPLE = {
            "features": [1.2, 0.5, -0.3, 0.8, 0.2,
                         -0.1, 0.4, 0.3, -0.2, 0.1,
                         0.5, -0.3, 0.2, 0.1, -0.4,
                         0.3, 0.2, -0.1, 0.4, 0.2,
                         -0.1, 0.3, 0.1, -0.2, 0.4,
                         0.1, -0.3, 0.2],
            "amount": 50.0,
            "time": 1000.0
        }

        if qcol1.button("Load Normal Transaction", use_container_width=True):
            st.session_state["example"] = NORMAL_EXAMPLE
        if qcol2.button("Load Suspicious Transaction", use_container_width=True):
            st.session_state["example"] = FRAUD_EXAMPLE

        example = st.session_state.get("example", NORMAL_EXAMPLE)

        amount = st.number_input("Amount ($)",
                                 min_value=0.0, max_value=50000.0,
                                 value=float(example["amount"]), step=0.01)
        time_val = st.number_input("Time (seconds since first transaction)",
                                   min_value=0.0,
                                   value=float(example["time"]), step=1.0)

        st.markdown("**V1–V28 Features** (PCA-transformed)")
        features_str = st.text_area(
            "Features (28 comma-separated values)",
            value=", ".join([str(round(f, 4)) for f in example["features"]]),
            height=120,
            label_visibility="collapsed"
        )

        predict_btn = st.button(
            "🔍 Predict", type="primary", use_container_width=True)

    with col_result:
        st.subheader("Prediction Result")

        if predict_btn:
            try:
                features = [float(x.strip()) for x in features_str.split(",")]

                if len(features) != 28:
                    st.error(f"Need exactly 28 features, got {len(features)}")
                else:
                    payload = {
                        "features": features,
                        "amount": amount,
                        "time": time_val
                    }

                    with st.spinner("Calling API..."):
                        start = time.time()
                        response = requests.post(
                            f"{API_URL}/predict",
                            json=payload,
                            timeout=10
                        )
                        elapsed = (time.time() - start) * 1000

                    if response.status_code == 200:
                        result = response.json()

                        # ── Big verdict ────────────────────────────────────
                        if result["is_fraud"]:
                            st.error("## 🚨 FRAUD DETECTED")
                        else:
                            st.success("## ✅ LEGITIMATE TRANSACTION")

                        st.divider()

                        # ── Metrics ────────────────────────────────────────
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Fraud Probability",
                                  f"{result['fraud_probability']*100:.2f}%")
                        m2.metric("Risk Level", result["risk_level"])
                        m3.metric("Latency", f"{result['latency_ms']:.1f}ms")

                        st.divider()

                        # ── Probability gauge ──────────────────────────────
                        proba = result["fraud_probability"]
                        fig, ax = plt.subplots(figsize=(6, 1.5))
                        ax.barh([""], [proba],
                                color="#e74c3c" if proba > 0.5 else "#2ecc71",
                                alpha=0.85, height=0.5)
                        ax.barh([""], [1 - proba], left=[proba],
                                color="#333", alpha=0.5, height=0.5)
                        ax.axvline(x=0.5, color='white',
                                   linestyle='--', alpha=0.7)
                        ax.set_xlim(0, 1)
                        ax.set_xlabel("Fraud Probability")
                        ax.set_facecolor('#0e1117')
                        fig.patch.set_facecolor('#0e1117')
                        ax.tick_params(colors='white')
                        ax.xaxis.label.set_color('white')
                        for spine in ax.spines.values():
                            spine.set_edgecolor('#333')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                        st.caption(
                            f"Transaction ID: `{result['transaction_id']}` · "
                            f"Model: `{result['model_version']}`"
                        )
                    else:
                        st.error(f"API error: {response.status_code}")
                        st.json(response.json())

            except ValueError:
                st.error("Invalid features format. Use comma-separated numbers.")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure uvicorn is running.")
        else:
            st.info("Fill in the transaction details and click Predict.")
            st.markdown("""
            **How to use:**
            1. Click **Load Normal** or **Load Suspicious** to prefill
            2. Adjust Amount and Time if you want
            3. Click **Predict** to call the live API

            **What the features are:**
            - V1–V28: PCA-transformed transaction features (anonymized for privacy)
            - Amount: Transaction amount in USD
            - Time: Seconds elapsed since first transaction in dataset
            """)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Info
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Info":
    st.header("📈 Production Model Info")

    try:
        response = requests.get(f"{API_URL}/model-info", timeout=5)

        if response.status_code == 200:
            info = response.json()

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Model Details")
                st.markdown(
                    f"**Model:** `{info.get('model_name', 'fraud-detector')}`")
                st.markdown(f"**Version:** `v{info.get('version', '?')}`")
                st.markdown(
                    f"**Type:** `{info.get('model_type', 'RandomForest')}`")
                st.markdown(
                    f"**Sampling:** `{info.get('sampling_strategy', 'smote')}`")

                st.divider()
                st.subheader("Threshold Analysis")
                threshold_r = requests.get(
                    f"{API_URL}/threshold-info", timeout=5)
                if threshold_r.status_code == 200:
                    t = threshold_r.json()
                    st.markdown(
                        f"**Current threshold:** `{t['current_threshold']}`")
                    st.info(t['explanation'])
                    for thresh, desc in t['tradeoffs'].items():
                        st.markdown(f"- **{thresh}**: {desc}")

            with col2:
                st.subheader("Production Metrics")
                metrics = info.get("metrics", {})

                # Metric cards
                st.metric("Recall (Fraud)",
                          f"{metrics.get('recall_fraud', 0):.4f}",
                          help="Fraction of actual fraud cases caught")
                st.metric("Precision",
                          f"{metrics.get('precision_fraud', 0):.4f}",
                          help="Fraction of fraud predictions that are correct")
                st.metric("ROC-AUC",
                          f"{metrics.get('roc_auc', 0):.4f}")
                st.metric("Fraud Caught",
                          f"{int(metrics.get('fraud_caught', 0))} / 98")
                st.metric("Fraud Missed",
                          f"{int(metrics.get('fraud_missed', 0))} cases")

        else:
            st.error("Could not fetch model info from API")

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure uvicorn is running at port 8000.")
    except Exception as e:
        st.error(f"Error: {e}")
