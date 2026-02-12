"""
Analysis Module

Handles data visualization, statistical summaries, report generation,
and machine learning model training for the Iris dataset.
"""

import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend for server environments

import base64
import io
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from fpdf import FPDF
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def generate_summary_stats(df):
    """
    Generates an HTML table of descriptive statistics for the dataframe.
    """
    if df.empty:
        return "<p class='text-muted'>No data available for statistics.</p>"
    
    return df.describe().to_html(
        classes="table table-striped table-hover",
        float_format="%.2f",
        justify="center"
    )

def generate_plots(df):
    """
    Creates distribution and relationship plots from the dataset.

    Returns:
        dict: Base64 encoded strings of the generated plots.
    """
    plots = {}
    
    if df.empty:
        return plots

    # Normalize target column name for plotting consistency
    plot_df = df.copy()
    if 'species' not in plot_df.columns:
        object_cols = plot_df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            plot_df.rename(columns={object_cols[0]: 'species'}, inplace=True)
        else:
            plot_df['species'] = 'Unknown'

    # 1. Boxplot: Feature distributions by species
    plt.figure(figsize=(10, 6))
    numeric_cols = plot_df.select_dtypes(include=['float64', 'int64']).columns
    df_melted = pd.melt(
        plot_df, 
        id_vars=['species'], 
        value_vars=numeric_cols, 
        var_name="Feature", 
        value_name="Measurement"
    )
    
    sns.boxplot(x="Feature", y="Measurement", hue="species", data=df_melted)
    plt.title("Feature Distributions by Species")
    plots['boxplot'] = _fig_to_base64(plt)

    # 2. Pairplot: Pairwise relationships
    plt.figure()
    pp = sns.pairplot(plot_df, hue='species', markers=["o", "s", "D"])
    plots['pairplot'] = _fig_to_base64(pp.figure)
    
    return plots

def train_and_evaluate_model(df):
    """
    Trains a Logistic Regression model on the dataset and evaluates performance.

    Returns:
        dict: Metrics, confusion matrix visualization, and classification report.
    """
    results = {}
    
    # Identify target column
    target_col = 'species'
    if target_col not in df.columns:
        obj_cols = df.select_dtypes(include=['object']).columns
        if len(obj_cols) > 0:
            target_col = obj_cols[0]
        else:
            raise ValueError("Could not identify target classification column.")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split: 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    results['accuracy'] = f"{acc * 100:.2f}%"
    results['train_size'] = len(X_train)
    results['test_size'] = len(X_test)
    
    # Visualization: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', 
        xticklabels=model.classes_, yticklabels=model.classes_
    )
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (Test Set)')
    results['confusion_matrix'] = _fig_to_base64(plt)
    
    # Report Table
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    results['report_html'] = report_df.to_html(
        classes="table table-bordered table-striped", 
        float_format="%.2f"
    )
    
    return results

def generate_pdf_report(df):
    """
    Generates a PDF summary report using FPDF.
    
    Returns:
        str: Filesystem path to the generated PDF.
    """
    class ReportPDF(FPDF):
        def header(self):
            self.set_font('Helvetica', 'B', 15)
            self.cell(0, 10, 'Iris Dataset Analysis Report', 0, 1, 'C')
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = ReportPDF()
    pdf.add_page()
    
    # Section: Statistics
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, txt="Summary Statistics:", ln=True)
    
    stats = df.describe().round(2)
    
    # Render dataframe as simple text block
    pdf.set_font("Courier", size=10)
    line_height = 6
    for line in stats.to_string().split('\n'):
        pdf.cell(0, line_height, txt=line, ln=True)
        
    pdf.ln(10)
    
    # Section: Metadata
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, txt="Dataset Metadata:", ln=True)
    pdf.cell(0, 10, txt=f"Total Rows: {len(df)}", ln=True)
    pdf.cell(0, 10, txt=f"Total Columns: {len(df.columns)}", ln=True)
    
    output_path = '/tmp/iris_report.pdf'
    pdf.output(output_path)
    return output_path

def _fig_to_base64(fig):
    """Internal helper to convert Matplotlib figure to Base64 string."""
    buf = io.BytesIO()
    
    # Handle both Figure objects and pyplot state-machine
    if hasattr(fig, 'savefig'):
        fig.savefig(buf, format='png', bbox_inches='tight')
    else:
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close() 
        
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')