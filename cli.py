"""
CLI Debugging Tool

This script allows developers to run and debug the analysis and machine learning
modules locally without starting the web server or database.

It loads data directly from the CSV file, runs all analysis functions, and
saves the resulting artifacts (plots, reports) to a 'debug_output' directory.

Usage:
    python cli.py
"""

import os
import sys
import base64
import pandas as pd
import analysis  # Imports your analysis module

# Configuration
DATA_PATH = os.path.join('data', 'iris_dataset.csv')
OUTPUT_DIR = 'debug_output'

def setup_environment():
    """Ensures input data exists and output directories are ready."""
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Data file not found at: {DATA_PATH}")
        print("Please ensure 'data/iris_dataset.csv' exists.")
        sys.exit(1)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[INFO] Created output directory: {OUTPUT_DIR}/")

def load_data():
    """Loads the dataset into a Pandas DataFrame."""
    print(f"[INFO] Loading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"[INFO] Successfully loaded {len(df)} rows.")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        sys.exit(1)

def debug_statistics(df):
    """Runs and verifies summary statistics generation."""
    print("\n--- Debugging Statistical Analysis ---")
    try:
        html_stats = analysis.generate_summary_stats(df)
        print("[PASS] Summary statistics HTML generated.")
        
        # Save raw HTML for inspection
        with open(f"{OUTPUT_DIR}/stats.html", "w") as f:
            f.write(html_stats)
        print(f"[INFO] Saved HTML stats to {OUTPUT_DIR}/stats.html")
    except Exception as e:
        print(f"[FAIL] Statistics generation failed: {e}")

def debug_plotting(df):
    """Runs plot generation and saves images to disk."""
    print("\n--- Debugging Visualization Generation ---")
    try:
        plots = analysis.generate_plots(df)
        
        for name, b64_data in plots.items():
            # Decode base64 and save as PNG
            image_data = base64.b64decode(b64_data)
            output_path = f"{OUTPUT_DIR}/{name}.png"
            with open(output_path, "wb") as f:
                f.write(image_data)
            print(f"[PASS] Generated {name} plot -> saved to {output_path}")
            
    except Exception as e:
        print(f"[FAIL] Plot generation failed: {e}")

def debug_machine_learning(df):
    """Runs the model training pipeline and prints metrics."""
    print("\n--- Debugging Machine Learning Model ---")
    try:
        results = analysis.train_and_evaluate_model(df)
        
        print(f"[PASS] Model Trained Successfully")
        print(f"       Accuracy: {results.get('accuracy')}")
        print(f"       Training Set: {results.get('train_size')} samples")
        print(f"       Testing Set:  {results.get('test_size')} samples")
        
        # Save Confusion Matrix
        if 'confusion_matrix' in results:
            img_data = base64.b64decode(results['confusion_matrix'])
            with open(f"{OUTPUT_DIR}/confusion_matrix.png", "wb") as f:
                f.write(img_data)
            print(f"[INFO] Saved confusion matrix to {OUTPUT_DIR}/confusion_matrix.png")
            
        # Save Classification Report
        if 'report_html' in results:
            with open(f"{OUTPUT_DIR}/model_report.html", "w") as f:
                f.write(results['report_html'])
            print(f"[INFO] Saved classification report to {OUTPUT_DIR}/model_report.html")

    except Exception as e:
        print(f"[FAIL] Model training failed: {e}")
        import traceback
        traceback.print_exc()

def debug_pdf_report(df):
    """Runs PDF generation and moves the file to debug folder."""
    print("\n--- Debugging PDF Report Generation ---")
    try:
        # analysis.py saves to /tmp/iris_report.pdf by default
        pdf_path = analysis.generate_pdf_report(df)
        
        if os.path.exists(pdf_path):
            # Move/Copy to debug folder
            dest_path = f"{OUTPUT_DIR}/final_report.pdf"
            with open(pdf_path, 'rb') as src, open(dest_path, 'wb') as dst:
                dst.write(src.read())
            print(f"[PASS] PDF Report generated -> saved to {dest_path}")
        else:
            print("[FAIL] PDF generation function returned success, but file was not found.")
            
    except Exception as e:
        print(f"[FAIL] PDF generation failed: {e}")

def main():
    print("=== Starting Local Analysis Debugger ===")
    setup_environment()
    
    df = load_data()
    
    if df.empty:
        print("[ERROR] DataFrame is empty. Cannot proceed.")
        sys.exit(1)

    debug_statistics(df)
    debug_plotting(df)
    debug_machine_learning(df)
    debug_pdf_report(df)
    
    print("\n=== Debugging Complete ===")
    print(f"Check the '{OUTPUT_DIR}' folder for generated artifacts.")

if __name__ == "__main__":
    main()