# Use the official lightweight Python image optimized for Linux
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for proper CatBoost operation on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependency file and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all remaining project files (code, data cache, and model weights)
COPY . .

# Expose the port that Streamlit uses by default
EXPOSE 8501

# Command to launch the web interface when the container starts
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
