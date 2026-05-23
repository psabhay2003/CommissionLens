import pandas as pd
import joblib
import os
# ... (keep your existing imports for plotting/simulation) ...

def run_sip_simulation(data_path="data/fund_dataset.csv", model_dir="models", report_dir="reports"):
    print("\n🔹 STAGE 6 / 6 — SIP BACK-VALIDATION")
    
    # 1. Load the dataset
    df = pd.read_csv(data_path)
    # ... (keep your existing data prep logic here) ...
    
    # 2. Load BOTH the model and the scaler
    try:
        model = joblib.load(f"{model_dir}/best_classifier.pkl")
        scaler = joblib.load(f"{model_dir}/scaler.pkl")
    except FileNotFoundError:
        print("  ❌ Model or scaler not found. Run training first.")
        return

    # 3. Prepare the features for prediction
    drop_cols = ['date', 'fund_id', 'target', 'target_regression']
    features = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # 4. CRITICAL FIX: Scale the features before predicting
    features_scaled = scaler.transform(features)
    
    # 5. Generate Predictions
    df['model_signal'] = model.predict(features_scaled)
    
    # ... (Keep the rest of your existing SIP XIRR calculation and plotting code exactly the same) ...
    # print("  ── SIP XIRR Comparison ──")
    # ...
