FROM python:3.9-slim

WORKDIR /app

COPY batch_processor/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY batch_processor ./batch_processor
COPY churn_model.pickle churn_model.pickle
COPY batch_processor/input_data/ input_data/

ENV PYTHONPATH=/app

# CMD will be set by GitHub Actions or a local script to run the batch job
# For example: CMD ["python", "-m", "batch_processor.main"]
CMD ["python", "-m", "batch_processor.main"]