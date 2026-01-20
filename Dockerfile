FROM python:3.10-slim

# Install system dependencies for LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
# Upgrade pip first to ensure NumPy 2.0 installs correctly
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy necessary artifacts and code
COPY src/components/data_processor.py src/components/
COPY data/model/ data/model/
COPY app.py .

ENV PYTHONPATH="${PYTHONPATH}:/app/src/components"

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]