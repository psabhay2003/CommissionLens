import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

def train_all_models(data_path="data/fund_dataset.csv", model_dir="models", report_dir="reports"):
    print("\n🔹 STAGE 4 / 6 — MODEL TRAINING (MLP & STACKING)")
    
    # 1. Load Data
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Drop non-feature columns
    drop_cols = ['date', 'fund_id', 'target', 'target_regression']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df['target']
    
    # 2. Temporal Train-Test Split (Approximating your previous split)
    split_index = int(len(df) * 0.77) # roughly 439 train / 132 test
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print(f"  → Temporal split: train {len(X_train)} rows, test {len(X_test)} rows")

    # Neural Networks require scaled data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==========================================
    # APPROACH 1: SHALLOW MLP
    # ==========================================
    print("\n  ── Training Shallow MLP ──")
    # 1 hidden layer with 16 neurons, L2 regularization (alpha=0.01) to prevent overfitting
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(16,), 
        activation='relu',
        solver='adam',
        alpha=0.01, 
        max_iter=1000,
        early_stopping=True,
        random_state=42
    )
    
    mlp_model.fit(X_train_scaled, y_train)
    mlp_preds = mlp_model.predict(X_test_scaled)
    mlp_probs = mlp_model.predict_proba(X_test_scaled)[:, 1]
    
    mlp_accuracy = accuracy_score(y_test, mlp_preds)
    print(f"    MLP Accuracy : {mlp_accuracy:.4f}")
    
    # Check if MLP meets the target
    if mlp_accuracy >= 0.60:
        print("    ✅ MLP achieved >= 0.60 accuracy. Proceeding with MLP.")
        final_model = mlp_model
        final_preds = mlp_preds
        final_probs = mlp_probs
        model_name = "shallow_mlp"
        
    else:
        # ==========================================
        # FALLBACK APPROACH: STACKING CLASSIFIER
        # ==========================================
        print("    ⚠️ MLP accuracy below 0.60. Falling back to Stacking Classifier...")
        print("\n  ── Training Stacking Ensemble ──")
        
        # Define Base Learners
        estimators = [
            ('xgb', XGBClassifier(max_depth=3, learning_rate=0.05, n_estimators=100, random_state=42)),
            ('cat', CatBoostClassifier(iterations=100, depth=4, learning_rate=0.05, verbose=0, random_state=42)),
            ('lr_elastic', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000, random_state=42))
        ]
        
        # Meta-Learner (Logistic Regression)
        stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=5 # Cross-validation to prevent leakage in meta-learner
        )
        
        stacking_model.fit(X_train_scaled, y_train)
        stacking_preds = stacking_model.predict(X_test_scaled)
        stacking_probs = stacking_model.predict_proba(X_test_scaled)[:, 1]
        
        final_model = stacking_model
        final_preds = stacking_preds
        final_probs = stacking_probs
        model_name = "stacking_ensemble"
        
        stacking_accuracy = accuracy_score(y_test, final_preds)
        print(f"    Stacking Accuracy : {stacking_accuracy:.4f}")

    # ==========================================
    # METRICS CALCULATION & SAVING
    # ==========================================
    final_auc = roc_auc_score(y_test, final_probs)
    final_f1 = f1_score(y_test, final_preds)
    
    print(f"\n  ── Final Selected Model: {model_name} ──")
    print(f"    AUC-ROC : {final_auc:.4f}")
    print(f"    F1 Score: {final_f1:.4f}")
    print(f"    Accuracy: {accuracy_score(y_test, final_preds):.4f}")
    print("\n    Classification Report:\n")
    print(classification_report(y_test, final_preds))

    # Save metrics
    metrics = {
        "classification": {
            "model_used": model_name,
            "auc_roc": final_auc,
            "f1": final_f1,
            "accuracy": accuracy_score(y_test, final_preds)
        }
    }
    
    os.makedirs(report_dir, exist_ok=True)
    with open(f"{report_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save model and scaler for later use (e.g., SIP Simulation & SHAP)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(final_model, f"{model_dir}/best_classifier.pkl")
    joblib.dump(scaler, f"{model_dir}/scaler.pkl") 
    
    print("  ✓ Final model, scaler, & metrics saved")

if __name__ == "__main__":
    train_all_models()
