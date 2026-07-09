"""RAG QA по русской литературе — публичный интерфейс пакета."""

from .parsing import extract_answer
from .pipeline import answer_dataset, answer_question
from .evaluation import evaluate
from .indexing import build_index

__all__ = [
    "answer_question",
    "answer_dataset",
    "extract_answer",
    "evaluate",
    "build_index",
]
