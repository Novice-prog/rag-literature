"""CLI пакета.

Примеры:
    python -m rag_literature build-index
    python -m rag_literature answer
    python -m rag_literature evaluate
"""

import argparse

from . import config


def main():
    parser = argparse.ArgumentParser(prog="rag_literature", description="RAG QA по русской литературе")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build-index", help="построить FAISS-индексы по книгам из books/")

    p_answer = sub.add_parser("answer", help="ответить на вопросы из датасета")
    p_answer.add_argument("--questions", default=config.QUESTIONS_CSV)
    p_answer.add_argument("--output", default=config.OUTPUT_CSV)

    p_eval = sub.add_parser("evaluate", help="сравнить ответы с эталоном")
    p_eval.add_argument("--predictions", default=config.OUTPUT_CSV)
    p_eval.add_argument("--answers", default=config.ANSWERS_CSV)

    args = parser.parse_args()

    if args.command == "build-index":
        from .indexing import build_index

        build_index()
    elif args.command == "answer":
        from .pipeline import answer_dataset

        answer_dataset(args.questions, args.output)
    elif args.command == "evaluate":
        from .evaluation import evaluate

        evaluate(args.predictions, args.answers)


if __name__ == "__main__":
    main()
