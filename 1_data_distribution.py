# -*- coding: utf-8 -*-
"""
Script 1: Data Distribution Analysis
Paper Section: 3.1 Data Source and Sample Statistics

Description: 
This script iterates through the raw procurement documents (PDF/Word) in the './dataset/' folder,
extracts metadata (Procurement Mode, Industry, Region), and prints statistical summaries.
"""

import os
import glob
import pandas as pd
import pdfplumber
from docx import Document

# --- Configuration ---
# Path to the raw dataset folder (Relative path)
DATASET_PATH = r"../dataset"

def get_file_content(filepath):
    """Extract text content from PDF or DOCX files."""
    text = ""
    try:
        if filepath.endswith('.docx'):
            doc = Document(filepath)
            # Read first 50 paragraphs to find metadata
            text = " ".join([p.text for p in doc.paragraphs[:50]])
        elif filepath.endswith('.pdf'):
            with pdfplumber.open(filepath) as pdf:
                # Read first page
                text = pdf.pages[0].extract_text() or ""
    except Exception as e:
        print(f"Warning: Could not read {os.path.basename(filepath)} - {e}")
    return text

def analyze_dataset(folder_path):
    if not os.path.exists(folder_path):
        print(f"Error: Dataset folder '{folder_path}' not found.")
        return

    files = glob.glob(os.path.join(folder_path, "*.*"))
    print(f"Found {len(files)} documents. Analyzing metadata...")

    stats = []
    
    for file in files:
        if not (file.endswith('.pdf') or file.endswith('.docx')): continue
        
        fname = os.path.basename(file)
        text = get_file_content(file)
        
        # Simple rule-based extraction for demonstration
        # 1. Procurement Mode
        mode = "Public Bidding" # Default
        if "竞争性磋商" in text or "磋商" in text: mode = "Competitive Consultation"
        elif "竞争性谈判" in text or "谈判" in text: mode = "Competitive Negotiation"
        elif "询价" in text: mode = "Inquiry"
        
        # 2. Industry Classification (Simplified)
        industry = "Government"
        if any(k in text for k in ['医院', '医疗', '卫生']): industry = "Healthcare"
        elif any(k in text for k in ['学校', '学院', '大学', '教育']): industry = "Education"
        elif any(k in text for k in ['公安', '警']): industry = "Public Security"
        
        stats.append({"Filename": fname, "Mode": mode, "Industry": industry})
    
    # Output Statistics
    df = pd.DataFrame(stats)
    print("\n=== Data Distribution Summary ===")
    print(f"Total Documents: {len(df)}")
    print("\n[By Procurement Mode]")
    print(df['Mode'].value_counts())
    print("\n[By Industry]")
    print(df['Industry'].value_counts())

if __name__ == "__main__":
    analyze_dataset(DATASET_PATH)