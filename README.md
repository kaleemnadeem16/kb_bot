# kb_bot

Ask questions about your own documents and get answers with citations. A small, readable retrieval-augmented generation (RAG) bot: it splits your files into chunks, embeds them, stores them in a local FAISS index and answers from the best-matching passages.

- **Reads:** `.txt`, `.md`, `.csv`, `.pdf` (PDF citations include the page number)
- **Cites sources:** each answer lists the files (and PDF pages) its passages came from, with a similarity score. If nothing relevant is found it says so and lists no sources.
- **Runs without an API key:** offline mode does keyword-level retrieval and shows the matching passages. Add an OpenAI key to get LLM-written answers.
- **Local index:** vectors are saved in `vector_store/` and rebuilt automatically when your documents, chunk settings or embedding model change. A damaged or incompatible index is ignored and rebuilt.
- **Tested:** 31 offline tests (`pytest`, no network). CI is configured for Python 3.10 and 3.12 (`.github/workflows/ci.yml`); the suite has so far only been run locally on 3.14.

**Status:** working CLI, not a hosted product. There is no web UI yet.

## Quick start

```bash
git clone https://github.com/kaleemnadeem16/kb_bot.git
cd kb_bot
pip install -r requirements.txt

python -m kb_bot -q "Which AI technologies are used in healthcare?"   # one question
python -m kb_bot                                                      # interactive
```

The `data/` folder ships with three small sample files so it works immediately. To use your own documents, put them in a folder and pass `--data-dir path/to/docs`.

### With OpenAI answers

```bash
cp .env.example .env     # then set OPENAI_API_KEY=...
python -m kb_bot -q "What is supervised learning?"
```

Defaults are `gpt-4.1-nano` for answers and `text-embedding-3-small` for embeddings (both configurable). Without a key, or with `--offline`, nothing is sent anywhere.

### Docker

```bash
docker build -t kb_bot .
docker run --rm -it -v "$PWD/my_docs:/app/data" -e OPENAI_API_KEY kb_bot -q "your question"
```

## Example (offline mode, abbreviated real output)

```
$ python -m kb_bot -q "Which AI technologies are used in healthcare?"
Indexed 17 chunks from 3 files (offline mode: set OPENAI_API_KEY for LLM answers).

(offline mode, no LLM: best matching passages)

[1] ...## Where AI is used

1. **Healthcare**: helping read medical scans, suggesting candidate drugs and supporting treatment planning. ...

Sources:
  [1] ai_knowledge.txt  (similarity 0.22)

$ python -m kb_bot -q "chocolate cake recipe"
I could not find anything relevant in the documents.
```

`--json` prints the same result as JSON, for scripting.

## Options

| Flag | Meaning |
|---|---|
| `-q "..."` | ask one question and exit (otherwise interactive; type `stats` or `exit`) |
| `--data-dir DIR` | folder of documents (default `data`) |
| `--top-k N` | passages to retrieve (default 4) |
| `--rebuild` | ignore the saved index and re-embed everything |
| `--offline` | never call OpenAI, even if a key is set |
| `--json` | with `-q`, machine-readable output |

Environment variables (see `.env.example`): `OPENAI_API_KEY`, `OPENAI_MODEL`, `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `MIN_SCORE`, `TEMPERATURE`, `DATA_DIR`, `INDEX_DIR`. `MIN_SCORE` is an inclusive cosine-similarity floor (default 0, meaning only positive matches). Offline mode always applies its own floor of 0.15 to hide noise. Invalid values produce an error message, not a crash.

## How it works

```
data/*.txt|md|csv|pdf
   -> documents.py     load files (one document per PDF page, one line per CSV row)
   -> chunking.py      ~500-char chunks with overlap, cut at paragraph / sentence boundaries
   -> embeddings.py    OpenAI embeddings (retries with backoff) or the offline hashing embedder
   -> vector_store.py  FAISS cosine index + meta.json, keyed by a hash of text + settings
   -> answering.py     OpenAI chat with numbered, cited context, or the offline extractive answer
   -> bot.py           KnowledgeBase.load() / .ask()
```

Use it from Python:

```python
from kb_bot import KnowledgeBase, Settings

kb = KnowledgeBase(Settings.from_env())
kb.load()
result = kb.ask("What is deep learning?")
print(result.text, [s.citation for s in result.sources])
```

## Limitations

- Offline mode matches shared words, not meaning. Use an OpenAI key for semantic search.
- The index is rebuilt in full when any document changes (fine for hundreds of files, not for huge corpora).
- PDFs must contain selectable text; scanned PDFs need OCR first, which is not included.
- The OpenAI code path (embeddings, chat, retries, error messages) is unit-tested with stubbed clients only. It has not been exercised against the live API in this repo's test suite.
- Chunks can begin at a sentence fragment (they never begin mid-word, except inside a single very long unbroken string).
- The Docker image has not been built in CI.
- No authentication, web UI or multi-user support.

## Development

```bash
pip install -r requirements-dev.txt
pytest
```

## License

MIT, see [LICENSE](LICENSE). The sample files in `data/` are short generic texts about AI and machine learning.
