FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY app ./app
COPY ui ./ui

# Includes the embeddings extra (sentence-transformers/torch) and the ui
# extra (streamlit) so the same image can run either the API or the
# Streamlit UI, selected via docker-compose's `command:` override.
RUN pip install --no-cache-dir ".[embeddings,ui]"

EXPOSE 8000 8501

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
