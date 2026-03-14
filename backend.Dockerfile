FROM python:3.13

WORKDIR /app

COPY pyproject.toml .

RUN pip install --no-cache-dir .

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.rag_chatbot.main:app", "--host", "0.0.0.0", "--port", "8000"]