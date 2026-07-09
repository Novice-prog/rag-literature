"""Построение FAISS-индексов по текстам книг из BOOKS_DIR."""

import os

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .providers import get_embeddings


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
    )


def build_index(books_dir: str = config.BOOKS_DIR, vector_dir: str = config.VECTOR_DIR):
    """Для каждого .txt в books_dir строит FAISS-индекс vector_dir/<имя книги>/."""
    os.makedirs(vector_dir, exist_ok=True)
    splitter = _splitter()
    embeddings = get_embeddings()

    for root, _dirs, files in os.walk(books_dir):
        for filename in files:
            if not filename.endswith(".txt"):
                continue
            path = os.path.join(root, filename)
            book_name = filename[:-4]
            print(f"Индексирую: {book_name}")

            documents = TextLoader(path, encoding="utf-8").load()
            text_content = [doc.page_content for doc in documents]
            split_documents = splitter.create_documents(text_content)

            store = FAISS.from_documents(split_documents, embeddings)
            store.save_local(os.path.join(vector_dir, book_name))

    print("Готово.")
