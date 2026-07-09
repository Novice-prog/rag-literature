FROM python:3.11-slim

# Не писать .pyc, не буферизовать stdout, кэш HF-моделей внутри контейнера
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache

WORKDIR /app

# Сначала зависимости — чтобы слой кэшировался при неизменном requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код приложения
COPY main.py app.py build_index.py evaluate.py ./

# FAISS-индексы (vectorization/) и ключ GROQ_API_KEY пробрасываются при запуске,
# в образ не зашиваются. См. README.
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
