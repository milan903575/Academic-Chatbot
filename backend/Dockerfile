FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip first
RUN pip install --upgrade pip>=25.0.0

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create NLTK data directory with proper permissions
RUN mkdir -p /root/nltk_data /usr/local/nltk_data /app/nltk_data
RUN chmod -R 755 /root/nltk_data /usr/local/nltk_data /app/nltk_data

# Download essential NLTK data WITHOUT wordnet
RUN python -c "import nltk; import ssl; ssl._create_default_https_context = ssl._create_unverified_context; nltk.download('punkt_tab', download_dir='/root/nltk_data', quiet=True)" || \
    python -c "import nltk; import ssl; ssl._create_default_https_context = ssl._create_unverified_context; nltk.download('punkt', download_dir='/root/nltk_data', quiet=True)" || true

RUN python -c "import nltk; import ssl; ssl._create_default_https_context = ssl._create_unverified_context; nltk.download('stopwords', download_dir='/root/nltk_data', quiet=True)" || true

# Set NLTK data path environment variables
ENV NLTK_DATA=/root/nltk_data:/usr/local/nltk_data:/app/nltk_data

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads vectorstore temp academic_cache user_sessions 

# Set permissions
RUN chmod 755 uploads vectorstore temp academic_cache user_sessions 

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=app.py

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Run the application
CMD ["python", "app.py"]
