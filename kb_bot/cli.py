"""Command line interface: `python -m kb_bot` (interactive) or `python -m kb_bot -q "..."`."""

from __future__ import annotations

import argparse
import json
import logging
import sys

import openai
from dotenv import load_dotenv

from .bot import Answer, KnowledgeBase
from .config import Settings


def _print_answer(answer: Answer) -> None:
    print(f"\n{answer.text}")
    if answer.sources:
        print("\nSources:")
        for i, s in enumerate(answer.sources, 1):
            print(f"  [{i}] {s.citation}  (similarity {s.score:.2f})")


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return number


def _openai_message(exc: Exception) -> str:
    """Short, key-free explanation of an OpenAI failure."""
    if isinstance(exc, openai.AuthenticationError):
        return "OpenAI rejected the API key. Check OPENAI_API_KEY, or use --offline."
    if isinstance(exc, openai.RateLimitError):
        return "OpenAI rate limit or quota reached. Try again later or check your plan."
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return "Could not reach OpenAI. Check your network, or use --offline."
    return f"OpenAI request failed ({type(exc).__name__}). Use --offline to run without it."


def _fail(message: str) -> int:
    print(f"Error: {message}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kb_bot", description="Ask questions about your documents.")
    parser.add_argument("-q", "--question", help="ask one question and exit")
    parser.add_argument("--data-dir", help="folder with .txt/.md/.csv/.pdf files (default: data)")
    parser.add_argument("--top-k", type=_positive_int, help="passages to retrieve (default: 4)")
    parser.add_argument("--rebuild", action="store_true", help="ignore the saved index and re-embed")
    parser.add_argument("--offline", action="store_true", help="never call OpenAI, even if a key is set")
    parser.add_argument("--json", action="store_true", help="with -q: print the result as JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    load_dotenv()
    try:
        settings = Settings.from_env()
    except ValueError as exc:
        return _fail(f"invalid setting in environment or .env ({exc})")
    if args.offline:
        settings = settings.with_(api_key="")
    if args.data_dir:
        settings = settings.with_(data_dir=args.data_dir)
    if args.top_k:
        settings = settings.with_(top_k=args.top_k)

    if args.question is not None and not args.question.strip():
        return _fail("the question is empty")

    try:
        kb = KnowledgeBase(settings)
        count = kb.load(force_rebuild=args.rebuild)
    except (FileNotFoundError, ValueError) as exc:
        return _fail(str(exc))
    except openai.OpenAIError as exc:
        return _fail(_openai_message(exc))
    if not args.json:
        note = "" if kb.mode == "openai" else " (offline mode: set OPENAI_API_KEY for LLM answers)"
        print(f"Indexed {count} chunks from {len(kb.stats()['files'])} files{note}.")

    if args.question is not None:
        try:
            answer = kb.ask(args.question)
        except openai.OpenAIError as exc:
            return _fail(_openai_message(exc))
        if args.json:
            print(json.dumps({"question": answer.question, "answer": answer.text,
                              "sources": [vars(s) for s in answer.sources]}, indent=2))
        else:
            _print_answer(answer)
        return 0

    print("Ask a question. Commands: stats, exit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"exit", "quit", "q"}:
            return 0
        if question.lower() == "stats":
            print(json.dumps(kb.stats(), indent=2))
        elif question:
            try:
                _print_answer(kb.ask(question))
            except openai.OpenAIError as exc:
                print(f"Error: {_openai_message(exc)}", file=sys.stderr)
