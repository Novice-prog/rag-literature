"""Загрузка FAISS-индексов и поиск релевантного контекста по книге."""

import os

from langchain_community.vectorstores import FAISS

from . import config
from .prompts import keyword_query_prompt
from .providers import get_embeddings, get_llm


def load_stores(book: str) -> list:
    """Возвращает список FAISS-хранилищ для книги (учитывая многотомники)."""
    folders = config.MULTIPART_BOOKS.get(book, [book])
    return [
        FAISS.load_local(
            folder_path=os.path.join(config.VECTOR_DIR, folder),
            embeddings=get_embeddings(),
            allow_dangerous_deserialization=True,
        )
        for folder in folders
    ]


def get_context(book: str, question: str, answers: dict, k: int = config.TOP_K) -> str:
    """Ищет в индексе книги фрагменты, релевантные вопросу и вариантам ответа."""
    query = get_llm().invoke(keyword_query_prompt(question)).content

    search_text = f"{query}, {answers['a']}, {answers['b']}, {answers['c']}, {answers['d']}"
    context_docs = []
    for store in load_stores(book):
        context_docs += store.similarity_search(
            search_text, k=k, search_kwargs={"score_threshold": config.SCORE_THRESHOLD}
        )
    return "\n".join(doc.page_content for doc in context_docs)
