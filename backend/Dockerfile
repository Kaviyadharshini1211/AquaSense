# Use official Python image
FROM python:3.11-slim



# Set working directory inside container
WORKDIR /app

# Install system dependencies (optional but helpful for certain packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your app files
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Launch FastAPI app using uvicorn (app is the filename, app is the FastAPI instance)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
