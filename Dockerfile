# Base image: Official Python slim version for smaller image size
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies required for psycopg2 and other binary compilations
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create data directory to ensure it exists
RUN mkdir -p data

# Copy the rest of the application code
COPY . .

# Expose port 5000 (Local default)
EXPOSE 5000

# Command to run the application
# 1. Initialize the database logic
# 2. Start Gunicorn server (binds to $PORT for Render, defaults to 5000 locally)
CMD python -c "from app import init_db; init_db()" && \
    gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --timeout 120 app:app