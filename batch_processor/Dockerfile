FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY churn_model.pickle churn_model.pickle
# Copy all CSVs from input_data folder
COPY input_data/ input_data/

COPY . .

ENV PYTHONPATH=/app

# CMD will be set by GitHub Actions or a local script to run the batch job
# For example: CMD ["python", "main.py"]
CMD ["python", "-m", "main"] 