FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for numerical computing
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (Docker layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY requirements-dev.txt /app/requirements-dev.txt
RUN pip install --no-cache-dir -r /app/requirements-dev.txt

# Copy source code (overridden by volume mount in development)
COPY . /app

CMD ["python", "-m", "tsd.main"]
