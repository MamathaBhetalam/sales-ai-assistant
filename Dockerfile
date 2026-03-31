# Azure Container Apps / Web App for Containers
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "exec python -m streamlit run app.py --server.port=${PORT:-8000} --server.address=0.0.0.0 --server.headless=true"]
