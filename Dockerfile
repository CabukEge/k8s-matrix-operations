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
   curl \
   && rm -rf /var/lib/apt/lists/*

# Copy the installed packages and executables from the builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the project source
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 80

# Health check using Python
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
   CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:80/')" || exit 1

# Start command (using uvicorn)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]