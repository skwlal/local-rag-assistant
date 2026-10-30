"""Terminal interface to the existing local RAG pipeline."""

import argparse

import httpx
from ollama import ResponseError

from app.config import Settings
from app.rag_engine import LocalRAGPipeline
from app.schemas import IngestRequest, RAGRequest

HELP = """/models           список установленных моделей
/model ИМЯ        выбрать модель для следующих ответов
/ingest           обработать документы из data/raw и перестроить индекс
/sources          показать/скрыть полные фрагменты источников
/help             команды
/exit             выход
Каждый вопрос независим: история переписки не передаётся модели.
"""
EXPECTED_ERRORS = (httpx.HTTPError, ResponseError, FileNotFoundError, ValueError, RuntimeError)


def select_model(pipeline: LocalRAGPipeline, name: str) -> str:
    available = pipeline.models().ollama_models
    if name not in available:
        raise ValueError(f"Модель {name!r} не установлена. Доступны: {', '.join(available)}")
    # The hybrid retriever currently consults DEFAULT_LLM_MODEL too.
    pipeline.settings.DEFAULT_LLM_MODEL = name
    return name


def answer(pipeline: LocalRAGPipeline, model: str, question: str, sources: bool) -> None:
    result = pipeline.query(RAGRequest(question=question, llm_model=model))
    print(f"\n{result.answer}\n")
    for number, source in enumerate(result.sources, 1):
        print(f"[S{number}] {source.doc_name}")
        if sources:
            print(source.text, "\n")
    print(f"Время: {result.metrics.total_latency_s:.1f} с")
    for warning in result.citation_warnings:
        print(f"Проверка ссылок: {warning}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Локальный RAG в терминале")
    parser.add_argument("--model", help="Точное имя установленной модели Ollama")
    parser.add_argument(
        "--ollama-url", help="Адрес локальной Ollama, например http://127.0.0.1:18480"
    )
    parser.add_argument("--models", action="store_true", help="Показать модели и выйти")
    parser.add_argument("--ingest", action="store_true", help="Перестроить индекс перед общением")
    parser.add_argument("--question", help="Ответить на один вопрос и выйти")
    parser.add_argument("--sources", action="store_true", help="Показывать текст источников")
    args = parser.parse_args(argv)
    settings = Settings(**({"OLLAMA_BASE_URL": args.ollama_url} if args.ollama_url else {}))
    pipeline = LocalRAGPipeline(settings)
    try:
        if args.models:
            print("\n".join(pipeline.models().ollama_models) or "Установленных моделей нет.")
            return 0
        model = select_model(pipeline, args.model or settings.DEFAULT_LLM_MODEL)
        if args.ingest:
            print("Обработка документов и построение индекса...", flush=True)
            print(pipeline.ingest(IngestRequest()))
        if args.question:
            answer(pipeline, model, args.question, args.sources)
            return 0
        print(f"Модель: {model}\nДокументы: {settings.RAW_DATA_DIR}\n{HELP}")
        print("Если индекс ещё не создан или документы изменились, выполните /ingest.")
        show_sources = args.sources
        while True:
            try:
                question = input("\nВы: ").strip()
            except EOFError:
                break
            if not question:
                continue
            command, _, argument = question.partition(" ")
            try:
                if command == "/exit":
                    break
                if command == "/help":
                    print(HELP)
                elif command == "/models":
                    print("\n".join(pipeline.models().ollama_models))
                elif command == "/model":
                    model = select_model(pipeline, argument.strip())
                    print(f"Выбрана модель: {model}")
                elif command == "/ingest":
                    print("Обработка документов и построение индекса...", flush=True)
                    print(pipeline.ingest(IngestRequest()))
                elif command == "/sources":
                    show_sources = not show_sources
                    print(f"Полные источники: {'включены' if show_sources else 'выключены'}")
                elif question.startswith("/"):
                    print("Неизвестная команда. Список команд: /help")
                else:
                    answer(pipeline, model, question, show_sources)
            except EXPECTED_ERRORS as exc:
                print(f"Ошибка: {exc}")
        return 0
    finally:
        pipeline.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nЧат закрыт.")
        raise SystemExit(130) from None
    except EXPECTED_ERRORS as exc:
        print(f"Ошибка: {exc}")
        raise SystemExit(1) from None
