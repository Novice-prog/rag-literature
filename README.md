# RAG QA по русской литературе

Система отвечает на тестовые вопросы с вариантами (A/B/C/D) по произведениям
русской литературы, используя **RAG** (Retrieval-Augmented Generation): сначала
находит релевантные фрагменты книги в векторном индексе, затем LLM рассуждает по
этому контексту и выбирает единственный правильный вариант.

## Как это работает

```
вопрос ──► LLM выделяет ключевые слова
                   │
                   ▼
        поиск в FAISS-индексе книги (top-k фрагментов)
                   │
                   ▼
   LLM + контекст ──► пошаговое рассуждение ──► [ОТВЕТ: X]
```

1. **Извлечение ключевых слов.** По формулировке вопроса LLM формирует поисковый
   запрос.
2. **Поиск контекста.** Запрос вместе с вариантами ответа прогоняется через
   similarity search по FAISS-индексу нужной книги (многотомники, например
   «Война и мир», ищутся по всем частям).
3. **Рассуждение.** LLM получает вопрос, найденный контекст и варианты ответа,
   последовательно их сравнивает и выдаёт ответ в формате `[ОТВЕТ: X]`.
4. **Оценка.** Ответы сравниваются с эталонными (`evaluate.py`).

## Стек

- **LLM:** [Groq](https://groq.com/) — `llama3-70b-8192`
- **Эмбеддинги:** `intfloat/multilingual-e5-small` (HuggingFace)
- **Векторный индекс:** FAISS
- **Оркестрация:** LangChain
- **API:** FastAPI + Uvicorn
- **Данные:** pandas

## Установка

```bash
git clone <URL-репозитория>
cd laba2_GPT_useRAG

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

Задай ключ Groq API:

```bash
cp .env.example .env
# впиши в .env свой GROQ_API_KEY (получить: https://console.groq.com/keys)
```

## Данные

Тексты книг и построенные FAISS-индексы **не хранятся в репозитории** (большой
размер + авторские права на часть произведений). Добавь их сам:

1. Положи `.txt`-файлы книг в папку `books/`. Имя файла = название книги, как оно
   указано в колонке `book` датасета (например `Мертвые души.txt`).
2. Построй индексы:

   ```bash
   python build_index.py
   ```

   Для каждой книги появится папка `vectorization/<название>/` с `index.faiss` и
   `index.pkl`.

Формат датасета вопросов (`LR2.csv`, `LR2_dev.csv`):

| колонка | значение |
|---|---|
| `question` | текст вопроса |
| `answer a`…`answer d` | варианты ответа |
| `book` | название книги |

Эталонные ответы (`LR2_answer.csv`) — код правильного варианта: `1=a, 2=b, 3=c, 4=d`.

## Запуск

```bash
python main.py        # прогнать вопросы из LR2.csv, ответы → answers.csv
python evaluate.py    # сравнить answers.csv с эталоном LR2_answer.csv
```

## API-сервис (FastAPI)

```bash
uvicorn app:app --reload
```

Документация Swagger UI: http://127.0.0.1:8000/docs

**`GET /health`** — проверка живости.

**`POST /ask`** — ответить на вопрос по книге:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "book": "Роковые яйца",
    "question": "Что украли у профессора Персикова?",
    "a": "Его красный луч",
    "b": "Драгоценности",
    "c": "Лабораторные записи",
    "d": "Очки"
  }'
```

Ответ:

```json
{
  "code": 1,
  "letter": "A",
  "reasoning": "Подробный анализ вариантов... [ОТВЕТ: A]"
}
```

`code`: `1=A, 2=B, 3=C, 4=D`, `0` — ответ не распознан. Если для указанной книги
нет индекса в `vectorization/`, вернётся `404`.

## Docker

```bash
docker build -t rag-literature .

docker run --rm -p 8000:8000 \
  -e GROQ_API_KEY=your_key \
  -v "$(pwd)/vectorization:/app/vectorization" \
  rag-literature
```

FAISS-индексы (`vectorization/`) и ключ в образ не зашиваются: индексы монтируются
volume'ом, ключ передаётся через `-e GROQ_API_KEY`. Модель эмбеддингов
скачивается при первом запросе (кэшируется в контейнере).

Или одной командой через Docker Compose (нужен заполненный `.env`):

```bash
docker compose up --build
```

## Тесты

```bash
pip install -r requirements-dev.txt
pytest
```

Тесты не требуют Groq-ключа: разбор ответа (`extract_answer`) проверяется напрямую,
а эндпоинты — с замоканным пайплайном (тяжёлые LLM/эмбеддинги инициализируются
лениво, поэтому `import main` дешёвый).

## Структура

```
main.py           — RAG-пайплайн: поиск контекста + ответ LLM (answer_question)
app.py            — FastAPI-сервис: POST /ask, GET /health
build_index.py    — построение FAISS-индексов по книгам из books/
evaluate.py       — подсчёт точности по эталонным ответам
tests/            — юнит-тесты (extract_answer, эндпоинты /ask и /health)
Dockerfile        — образ сервиса
docker-compose.yml — запуск сервиса одной командой
requirements.txt  — зависимости приложения
requirements-dev.txt — зависимости для тестов
LR2*.csv          — датасеты вопросов и эталонных ответов
```

## Возможные улучшения

- Порог отсечения по score в similarity search и подбор `k`
- Few-shot примеры в промпте, self-consistency (несколько прогонов + голосование)
- Кэширование ответов LLM
- Сравнение разных моделей эмбеддингов и LLM
