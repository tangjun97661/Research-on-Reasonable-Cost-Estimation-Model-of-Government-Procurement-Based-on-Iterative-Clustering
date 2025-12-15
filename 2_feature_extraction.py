# -*- coding: utf-8 -*-
"""
Script 2: Iterative Feature Clustering (Human-in-the-loop)
Paper Section: 3.2 Generation of Reasonable Cost Evaluation Indicator System

Description: 
Demonstrates the logic of the 3-round iterative clustering to extract 
technical indicators (CPU, Memory, Xinchuang) from noisy text.
"""

import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering

# --- Configuration ---
# For demonstration, we use a sample corpus. 
# In the full experiment, this list is populated by reading the files in '../dataset/'.
SAMPLE_CORPUS = [
    "供应 服务器 CPU Intel Xeon Gold 6248R 内存 32G 硬盘 4T",
    "国产化 替代 方案 海光 3号 处理器 统信 操作系统 麒麟",
    "招标文件 编号 2024-001 投标 截止 时间 盖章", # Noise sample
    "NVIDIA A800 GPU 显卡 深度学习 智算 中心",
    "提供 3年 原厂 质保 驻场 运维 服务 工程师",
    "采购 打印机 墨盒 A4纸 办公 用品 耗材", # Heterogeneous sample
]

# Stopwords & Blocklist (Results from Iteration 1 & 2)
STOP_WORDS = {'招标文件', '编号', '投标', '截止', '时间', '盖章', '公司', '采购'}
BLOCK_LIST = {'打印机', '墨盒', '耗材', '空调'}

def iterative_clustering_demo(corpus):
    print(">>> Starting Iterative Clustering Process...")
    
    clean_docs = []
    
    # --- Iteration 1 & 2: Cleaning ---
    for doc in corpus:
        # Filter heterogeneous devices
        if any(bad in doc for bad in BLOCK_LIST):
            continue
        
        # Remove administrative noise
        words = [w for w in jieba.lcut(doc) if w not in STOP_WORDS and len(w) > 1]
        clean_docs.append(" ".join(words))
    
    if not clean_docs:
        print("No valid documents after cleaning.")
        return

    # --- Iteration 3: Feature Emergence ---
    # Convert text to TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=50)
    X = vectorizer.fit_transform(clean_docs).toarray()
    feature_names = vectorizer.get_feature_names_out()
    
    # Run Hierarchical Clustering
    model = AgglomerativeClustering(n_clusters=3, linkage='ward')
    labels = model.fit_predict(X)
    
    # Show Results
    print("\n[Result] Emerged Feature Clusters:")
    df = pd.DataFrame(X, columns=feature_names)
    df['Cluster_Label'] = labels
    
    for i in range(3):
        if i not in df['Cluster_Label'].values: continue
        # Find top keywords for each cluster
        mean_tfidf = df[df['Cluster_Label']==i].mean().drop('Cluster_Label')
        top_keywords = mean_tfidf.sort_values(ascending=False).head(4).index.tolist()
        print(f"  Cluster {i}: {top_keywords}")

if __name__ == "__main__":
    iterative_clustering_demo(SAMPLE_CORPUS)