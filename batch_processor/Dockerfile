FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY churn_model.pickle churn_model.pickle
# Copy CSVs from the current directory
COPY database_input.csv database_input.csv
COPY database_input.csv database_input.csv
COPY database_input2.csv database_input2.csv
COPY database_input3.csv database_input3.csv

COPY . .

ENV PYTHONPATH=/app

# CMD will be set by GitHub Actions or a local script to run the batch job
# For example: CMD ["python", "main.py"]
CMD ["python", "-m", "main"] 