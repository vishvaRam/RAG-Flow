FROM python:3.11.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gunicorn \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV RAG_PORT=4545
ENV RAG_GUNICORN_WORKERS=1

# for quick debugging run the ASGI app directly with uvicorn instead of gunicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "4545", "--log-level", "debug"]
