# syntax=docker/dockerfile:1

# Builder Stage
FROM python:3.9-slim as builder
WORKDIR /app

RUN pip install --upgrade pip

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Final Stage
FROM python:3.9-slim
WORKDIR /app

# Install production system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpython3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Copy the project source
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Create non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:80/ || exit 1

# Start command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]