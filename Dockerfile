FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git-lfs \
    curl \
    gcc \
    g++ \
    && git lfs install \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download en_core_web_lg (needed as base for your custom model)
RUN python -m spacy download en_core_web_lg

# Copy application code including your trained model
COPY . .

# Verify your custom model can load
RUN python -c "import spacy; nlp = spacy.load('output/model-best'); print('Custom model loaded successfully')"

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 10000

# Start application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]