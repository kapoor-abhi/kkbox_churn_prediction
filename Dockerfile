# Stage 1: Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# --- FIX: Install the missing system dependency for LightGBM ---
RUN apt-get update && apt-get install -y libgomp1

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Expose port 8000 to allow communication to/from the app
EXPOSE 8000

# Define the command to run your app using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]