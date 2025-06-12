FROM python:3.9-slim

WORKDIR /app

COPY api_service/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY churn_model.pickle churn_model.pickle
COPY api_service ./api_service

ENV PYTHONPATH=/app

CMD ["uvicorn", "api_service.main:app", "--host", "0.0.0.0", "--port", "8000"] 