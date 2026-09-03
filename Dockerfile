FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 git unzip && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# artifacts/ and data/ are expected to be baked in or mounted:
#   docker run -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts sentinel
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
