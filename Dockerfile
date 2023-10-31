FROM --platform=arm64 python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV OPENAI_API_KEY="None"
ENV OPENAI_API_BASE="http://localhost:8000"




