# -*- coding: utf-8 -*-
"""
Script 3: Dynamic Cost Estimation via Weighted DBSCAN
Paper Section: 4.2 & 4.3 Empirical Analysis and Anomaly Detection

Description: 
Loads the feature matrix, applies Weighted DBSCAN to identify transaction clusters,
and calculates the reasonable cost intervals.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# --- Configuration ---
# Please rename your extracted feature file to 'feature_matrix.xlsx' and place it in 'data' folder
INPUT_FILE = r"../data/feature_matrix.xlsx"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Data file not found at {INPUT_FILE}")
        print("Please upload your feature matrix Excel file.")
        return

    print("Loading data...")
    df = pd.read_excel(INPUT_FILE)
    
    # Pre-processing
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df = df.dropna(subset=['Price'])
    
    # 1. Feature Preparation
    # Columns: CPU, Memory, Acceleration(GPU), Xinchuang(XC), Service
    cols = ['v_cpu', 'v_mem', 'v_acc', 'v_xc', 'v_srv']
    df[cols] = df[cols].fillna(0)
    X_raw = df[cols].values
    
    # 2. Normalization
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # 3. Weighted Distance (Key Innovation)
    # Weights: XC=3.0, GPU=3.0 (Hard constraints), Others=1.0, Service=0.5
    weights = np.array([1.0, 1.0, 3.0, 3.0, 0.5])
    X_weighted = X_scaled * weights
    
    # 4. DBSCAN Clustering
    print("Running Weighted DBSCAN...")
    db = DBSCAN(eps=0.5, min_samples=3, metric='euclidean')
    df['Cluster'] = db.fit_predict(X_weighted)
    
    # 5. Result Analysis
    n_clusters = len(set(df['Cluster'])) - (1 if -1 in df['Cluster'] else 0)
    print(f"\nModel Result: Identified {n_clusters} valid pricing clusters.")
    
    # Calculate Reasonable Intervals (Q1 - 1.5IQR ~ Q3 + 1.5IQR)
    print("\n--- Reasonable Cost Intervals (Unit: 10k CNY) ---")
    for cls in sorted(df['Cluster'].unique()):
        if cls == -1: continue
        sub = df[df['Cluster']==cls]['Price'] / 10000.0
        
        q1, q3 = sub.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = max(0, q1 - 1.5*iqr)
        upper = q3 + 1.5*iqr
        
        print(f"  Cluster {cls}: [{lower:.2f}, {upper:.2f}] (Median: {sub.median():.2f})")

if __name__ == "__main__":
    main()