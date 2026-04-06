# Stage 1: Builder - Install Dependencies
FROM python:3.10-slim as builder

# Set the working directory
WORKDIR /app

# Install system dependencies needed by opencv and ultralytics
# Note: Installing these packages can take a few minutes.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    libxi6 \
    libstdc++6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime - Create the final minimal image
FROM python:3.10-slim as runtime

# Set the working directory
WORKDIR /app

# Copy necessary system dependencies from the builder stage
COPY --from=builder /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libgomp.so.1 /usr/lib/x86_64-linux-gnu/

# Copy Python packages (only the installed ones)
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy the application code and the trained model
COPY app.py .
COPY car_detection_model.pt .

# Expose the port (FastAPI default)
EXPOSE 8013

# Command to run the application using Uvicorn
# We use 'app:app' to point Uvicorn to the FastAPI instance named 'app' 
# inside the file 'app.py'.
# The --host 0.0.0.0 is crucial for Docker to expose the port correctly.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8013"]