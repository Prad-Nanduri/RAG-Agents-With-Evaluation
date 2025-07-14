FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install the package
RUN pip install -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Optional: Expose port for any Streamlit visualization
EXPOSE 8501

# Entry point: run the full pipeline
ENTRYPOINT ["python"]
CMD ["run_all.py"]
# Optional: If you want to run a specific script, you can change the CMD line
# CMD ["streamlit", "run", "app.py"]