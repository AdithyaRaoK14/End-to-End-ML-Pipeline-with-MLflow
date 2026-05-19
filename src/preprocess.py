# src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle
import os


def preprocess(data_path, sampling_strategy="smote"):
    """
    Load and preprocess the credit card fraud dataset.

    sampling_strategy options:
        "smote"        — oversample minority class with SMOTE
        "class_weight" — no resampling, handle in model
        "none"         — raw imbalanced data (baseline)

    Returns: X_train, X_test, y_train, y_test
    """

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.4f}%)")

    # ── Feature Engineering ──────────────────────────────────────────────────
    # V1-V28 are already PCA-transformed — leave them as-is
    # Amount and Time need scaling — they're on completely different scales

    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled'] = scaler.fit_transform(df[['Time']])

    feature_cols = [f'V{i}' for i in range(
        1, 29)] + ['Amount_scaled', 'Time_scaled']
    X = df[feature_cols].values
    y = df['Class'].values

    print(f"Features used: {len(feature_cols)}")

    # ── Train/Test Split ─────────────────────────────────────────────────────
    # stratify=y is critical here — without it you might get 0 fraud in test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(
        f"\nTrain size: {X_train.shape[0]:,} | Test size: {X_test.shape[0]:,}")
    print(f"Train fraud: {sum(y_train == 1)} | Test fraud: {sum(y_test == 1)}")

    # ── Imbalance Handling ───────────────────────────────────────────────────
    if sampling_strategy == "smote":
        print("\nApplying SMOTE...")
        # sampling_strategy=0.1 means fraud will be 10% of legit count
        # not 50/50 — that's too aggressive and hurts precision
        sm = SMOTE(random_state=42, sampling_strategy=0.1)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(
            f"After SMOTE — Fraud: {sum(y_train == 1):,} | Legit: {sum(y_train == 0):,}")

    elif sampling_strategy == "class_weight":
        # No resampling here — class_weight="balanced" goes in the model
        print("\nUsing class_weight balancing (no resampling)")

    elif sampling_strategy == "none":
        print("\nNo imbalance handling — baseline experiment")

    # ── Save Scaler ──────────────────────────────────────────────────────────
    # This needs to be saved so the API uses the same scaling at inference
    scaler_path = "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\nScaler saved to: {scaler_path}")

    return X_train, X_test, y_train, y_test


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(
        __file__), "..", "data", "creditcard.csv")

    print("=" * 50)
    print("TEST 1: SMOTE")
    print("=" * 50)
    X_train, X_test, y_train, y_test = preprocess(
        data_path, sampling_strategy="smote")
    print(f"Final X_train shape: {X_train.shape}")
    print(
        f"Final y_train fraud ratio: {sum(y_train == 1)/len(y_train)*100:.2f}%")

    print("\n" + "=" * 50)
    print("TEST 2: No Sampling (baseline)")
    print("=" * 50)
    X_train2, X_test2, y_train2, y_test2 = preprocess(
        data_path, sampling_strategy="none")
    print(f"Final X_train shape: {X_train2.shape}")
    print(
        f"Final y_train fraud ratio: {sum(y_train2 == 1)/len(y_train2)*100:.2f}%")

    print("\n✅ preprocess.py working correctly")
