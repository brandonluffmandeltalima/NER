FROM python:3.13.9

# Install Git LFS
RUN apt-get update && \
    apt-get install -y git git-lfs && \
    git lfs install && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy your repo (Render will clone it, but this ensures files are there)
COPY . .

# Fetch LFS files
RUN git lfs pull || echo "No LFS files or already fetched"

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (adjust as needed)
EXPOSE 10000

# Run your application (adjust to your start command)
CMD ["python", "main.py"]