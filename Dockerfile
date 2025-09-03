FROM python:3.11-slim

# Install dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libx11-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set working directory
WORKDIR /app

# Copy application files
COPY . /app

CMD ["python", "app.py"]
