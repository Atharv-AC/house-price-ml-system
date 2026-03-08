FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml /app
COPY src /app/src

RUN pip install -e .

COPY models /app/models


EXPOSE 8000

CMD ["uvicorn", "house_price_prediction.api:app", "--host", "0.0.0.0", "--port", "8000"]