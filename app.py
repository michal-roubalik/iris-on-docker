"""
Main Application Entry Point

Handles Flask routes, database connections, and orchestrates
analysis tasks via the analysis module.
"""

import os
import time
import logging
import pandas as pd
from flask import Flask, jsonify, render_template, send_file
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
import analysis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def get_db_engine(max_retries=5, delay=5):
    """
    Establishes a database connection using SQLAlchemy.
    Includes retry logic to handle container startup timing.
    """
    connection_string = os.getenv('DATABASE_URL')
    
    if not connection_string:
        logger.error("DATABASE_URL environment variable is missing.")
        raise ValueError("DATABASE_URL must be set.")

    for attempt in range(max_retries):
        try:
            engine = create_engine(connection_string)
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return engine
        except OperationalError:
            logger.warning(f"Database not ready. Retrying in {delay}s (Attempt {attempt + 1}/{max_retries})...")
            time.sleep(delay)
    
    raise ConnectionError("Failed to connect to the database after multiple attempts.")

def get_data_frame():
    """Fetches full dataset from the database."""
    try:
        engine = get_db_engine(max_retries=1)
        # Select all data; ensure table name matches init_db
        return pd.read_sql("SELECT * FROM iris_data", engine)
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def init_db():
    """
    Idempotent database initialization.
    Loads CSV data if the table is empty or needs to be reset.
    """
    try:
        logger.info("Starting database initialization...")
        engine = get_db_engine()
        
        csv_path = os.path.join('data', 'iris_dataset.csv')
        
        if os.path.exists(csv_path):
            logger.info(f"Loading data from {csv_path}...")
            df = pd.read_csv(csv_path)
            df.to_sql('iris_data', engine, if_exists='replace', index=False)
            logger.info(f"Successfully loaded {len(df)} rows.")
        else:
            logger.warning(f"Data file not found at {csv_path}. Skipping import.")
            
    except Exception as e:
        logger.critical(f"Database initialization failed: {e}")

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    df = get_data_frame()
    
    if df.empty:
        return render_template('index.html', error="No data available. Please initialize the database.")

    stats_html = analysis.generate_summary_stats(df)
    plots = analysis.generate_plots(df)
    
    return render_template('dashboard.html', stats_html=stats_html, plots=plots)

@app.route('/train')
def train_model():
    try:
        df = get_data_frame()
        if df.empty:
            return "No data available for training.", 404
            
        results = analysis.train_and_evaluate_model(df)
        return render_template('model.html', results=results)
    except Exception as e:
        logger.error(f"Training error: {e}")
        return f"Error during training: {str(e)}", 500

@app.route('/download-report')
def download_report():
    try:
        df = get_data_frame()
        if df.empty:
            return "No data available.", 404
            
        pdf_path = analysis.generate_pdf_report(df)
        return send_file(
            pdf_path, 
            as_attachment=True, 
            download_name='Iris_Analysis_Report.pdf'
        )
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        return f"Error generating report: {e}", 500

@app.route('/api/health')
def health_check():
    """API endpoint for frontend connectivity checks."""
    try:
        engine = get_db_engine(max_retries=1)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT count(*) FROM iris_data"))
            count = result.scalar()
        return jsonify({
            "status": "success", 
            "message": f"Connected. Table contains {count} rows."
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Local development entry point
    app.run(host='0.0.0.0', port=5000, debug=True)