FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# build normal model & copy to this dockerfile (not recommended)

RUN pip install -r requirements.txt

COPY scripts/ scripts/

ENV PYTHONPATH=/app

CMD ["python3", "scripts/session_3/api.py"]
