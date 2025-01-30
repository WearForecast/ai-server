# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /src

# Install build tools if needed (for dependencies requiring compilation)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install only the dependencies first
COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the application port
EXPOSE 8080

# Default command to run the application (use --reload only for development)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]