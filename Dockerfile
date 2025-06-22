FROM python:3.11-slim

WORKDIR /app

# install libgomp and other dependencies
RUN apt-get update && apt-get install -y libgomp1 && apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY data/ ./data/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
