"""Настройки проекта: модели, пути, параметры ретрива, чтение ключа."""

import os

from dotenv import load_dotenv

load_dotenv()

# --- Модели ---
LLM_MODEL = "llama3-70b-8192"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# --- Пути ---
VECTOR_DIR = "vectorization"
BOOKS_DIR = "books"
DATA_DIR = "data"
QUESTIONS_CSV = os.path.join(DATA_DIR, "LR2.csv")
ANSWERS_CSV = os.path.join(DATA_DIR, "LR2_answer.csv")
OUTPUT_CSV = "answers.csv"

# --- Ретрив ---
TOP_K = 10
SCORE_THRESHOLD = 0.70

# --- Индексация ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Книги, разбитые на несколько частей при векторизации (ищем по всем частям).
MULTIPART_BOOKS = {"Война и мир": ["Война и мир 1", "Война и мир 2"]}


def get_groq_api_key() -> str:
    """Возвращает ключ Groq API или бросает понятную ошибку."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Не задан GROQ_API_KEY. Скопируй .env.example в .env и впиши свой ключ "
            "(получить: https://console.groq.com/keys)."
        )
    return api_key
