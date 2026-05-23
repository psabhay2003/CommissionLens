import pandas as pd
import numpy as np
import shap
import joblib
import os
import matplotlib.pyplot as plt

def run_shap_analysis(data_path="data/fund_dataset.csv", model_dir="models", report_dir="reports"):
    print("\n🔹 STAGE 5 / 6 — SHAP ANALYSIS")
    
    # Load data
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    drop_cols = ['date', 'fund_id', 'target', 'target_regression']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    split_index = int(len(df) * 0.77)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    
    # Load the model and the scaler
    print("  → Loading model and scaler...")
    model = joblib.load(f"{model_dir}/best_classifier.pkl")
    scaler = joblib.load(f"{model_dir}/scaler.pkl")
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("  → Computing SHAP values (KernelExplainer)...")
    # Use K-means to summarize the background data and speed up KernelExplainer
    background = shap.kmeans(X_train_scaled, 10)
    
    # KernelExplainer is model-agnostic
    explainer = shap.KernelExplainer(model.predict_proba, background)
    
    # Calculate SHAP values for the test set
    shap_values = explainer.shap_values(X_test_scaled)
    
    # For binary classification, KernelExplainer returns a list of arrays [negative_class, positive_class]
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
    else:
        shap_values_pos = shap_values

    os.makedirs(report_dir, exist_ok=True)
    
    print("  → Generating SHAP beeswarm plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_pos, X_test, show=False) # Pass unscaled X_test for readable labels
    plt.tight_layout()
    plt.savefig(f"{report_dir}/shap_summary.png")
    plt.close()
    
    print("  → Generating SHAP bar plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_pos, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{report_dir}/shap_bar.png")
    plt.close()
    
    print(f"  ✓ SHAP plots saved to {report_dir}")

if __name__ == "__main__":
    run_shap_analysis()
