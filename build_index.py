"""
Построение FAISS-индексов по текстам книг.

Кладёт .txt-файлы книг в папку books/ (имя файла = название книги, например
"Мертвые души.txt") и запусти:  python build_index.py

Для каждой книги создаётся папка vectorization/<название книги>/ с index.faiss
и index.pkl, которые затем использует main.py при поиске контекста.
"""

import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BOOKS_DIR = "books"
VECTOR_DIR = "vectorization"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

emb_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)


def build():
    os.makedirs(VECTOR_DIR, exist_ok=True)
    for root, _dirs, files in os.walk(BOOKS_DIR):
        for filename in files:
            if not filename.endswith(".txt"):
                continue
            path = os.path.join(root, filename)
            book_name = filename[:-4]
            print(f"Индексирую: {book_name}")

            documents = TextLoader(path, encoding="utf-8").load()
            text_content = [doc.page_content for doc in documents]
            split_documents = splitter.create_documents(text_content)

            store = FAISS.from_documents(split_documents, emb_model)
            store.save_local(os.path.join(VECTOR_DIR, book_name))

    print("Готово.")


if __name__ == "__main__":
    build()
