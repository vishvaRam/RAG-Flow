FROM python:3.11.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gunicorn \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV RAG_PORT=4545
ENV RAG_GUNICORN_WORKERS=2

CMD ["gunicorn", "api:app", "-b", "0.0.0.0:${RAG_PORT}", "-w", "${RAG_GUNICORN_WORKERS}","-k","uvicorn.workers.UvicornWorker","--timeout","120"]