"""Ленивые синглтоны тяжёлых объектов: LLM-клиент и модель эмбеддингов.

lru_cache гарантирует единственный экземпляр и создаёт его только при первом
обращении — поэтому import пакета дешёвый и не требует ключа (важно для тестов).
"""

from functools import lru_cache

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from . import config


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    return ChatGroq(api_key=config.get_groq_api_key(), model=config.LLM_MODEL)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
