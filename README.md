# Iris Analysis Tool & ML Pipeline

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://iris-app.onrender.com)
[![Docker](https://img.shields.io/badge/docker-automated-blue)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/python-3.11-yellow)](https://www.python.org/)

**[View Live Deployment](https://iris-app.onrender.com)** *(Note: As this runs on a free-tier server, it may take ~50 seconds to wake up on the first visit.)*

---

## 📖 Project Overview

This repository hosts a complete, end-to-end data science pipeline containerized with Docker.

**The primary goal of this project is experimental:** to explore the challenges and patterns involved in deploying a data science application—from database initialization to model training—in a production-like environment.

The frontend is intentionally kept minimal. It serves strictly as an interface to trigger backend processes, visualize statistical outputs, and demonstrate the underlying architecture.

### Key Objectives
* **Containerization:** Fully dockerized environment separating the application logic (`web`) from the persistence layer (`db`).
* **Pipeline Automation:** Automated database initialization, data loading, and connection handling with retry logic.
* **Machine Learning Integration:** On-demand training of a Logistic Regression model to identify Iris species.
* **Reporting:** Dynamic generation of PDF analysis reports and visualization dashboards.

---

## 🔬 Data Science Pipeline

This project implements a classic supervised learning workflow to classify Iris flower species based on morphological measurements.

### 1. Input Data
* **Source:** The standard [Iris flower dataset](https://archive.ics.uci.edu/ml/datasets/iris).
* **Features (X):** Sepal Length, Sepal Width, Petal Length, Petal Width (all in cm).
* **Target (y):** Species class (`setosa`, `versicolor`, `virginica`).
* **Ingestion:** Data is loaded from a CSV file into a PostgreSQL database upon container startup to simulate a production data warehouse.

### 2. Analysis & Processing Methods
* **Exploratory Data Analysis (EDA):**
    * **Descriptive Statistics:** Calculation of mean, median, std dev, min/max for all numerical features.
    * **Visualization:**
        * **Boxplots:** To visualize distribution and outliers across species.
        * **Pairplots:** To observe pairwise relationships and separability of classes.
* **Machine Learning (Identification Tool):**
    * **Preprocessing:** Data is fetched from the database and cleaned (handling missing values/types).
    * **Splitting:** The dataset is stratified and split into **80% Training** and **20% Testing** sets using `train_test_split`.
    * **Model:** A **Logistic Regression** classifier (`sklearn`) is trained on the 80% training subset. This model was chosen for its interpretability and efficiency on small datasets.

### 3. Output & Evaluation
* **Performance Metrics:**
    * **Accuracy Score:** Overall percentage of correct predictions on the test set.
    * **Confusion Matrix:** A heatmap visualization showing true positives vs. false positives/negatives for each class.
    * **Classification Report:** Precision, Recall, and F1-Score for each species.
* **Artifacts:**
    * **Interactive Dashboard:** HTML rendering of the plots and stats.
    * **PDF Report:** A downloadable document compiling the statistical summary and dataset metadata.

---

## 🚀 Features

* **Database Connectivity:** Robust connection handling to PostgreSQL with automated health checks.
* **Interactive Dashboard:** Visualizes data distributions (Boxplots) and feature relationships (Pairplots).
* **ML Identification Tool:** Trains a model on the dataset (80/20 split) and evaluates performance (Confusion Matrix, Precision/Recall).
* **PDF Export:** Generates downloadable reports of the analysis on the fly.
* **CLI Debugger:** Includes a command-line tool for testing data science logic locally without spinning up the full web server.

---

## 🛠️ Tech Stack

* **Infrastructure:** Docker, Docker Compose, PostgreSQL
* **Backend:** Python 3.11, Flask, Gunicorn, SQLAlchemy
* **Data Science:** Pandas, Scikit-learn, Matplotlib, Seaborn
* **Deployment:** Render (Web Service + Managed PostgreSQL)