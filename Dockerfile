FROM python:3.11-slim

WORKDIR /app

# System deps for some ML libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

COPY . .

EXPOSE 5000

ENV FLASK_APP=app/app.py
ENV PYTHONUNBUFFERED=1

CMD ["python", "app/app.py"]
