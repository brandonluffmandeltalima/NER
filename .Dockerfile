FROM python:3.12-slim

# Install git-lfs
RUN apt-get update && apt-get install -y git-lfs && git lfs install

# Copy project
WORKDIR /app
COPY . .

# Pull LFS model files
RUN git lfs pull

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Start API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
