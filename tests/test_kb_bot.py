"""Offline tests: no network, no API key."""

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kb_bot import KnowledgeBase, Settings
from kb_bot.chunking import split_text
from kb_bot.documents import load_documents
from kb_bot.embeddings import OfflineEmbedder, OpenAIEmbedder
from kb_bot.vector_store import VectorStore

SAMPLE_DATA = Path(__file__).parent.parent / "data"


def make_settings(tmp_path, data_dir=SAMPLE_DATA):
    return Settings(data_dir=data_dir, index_dir=tmp_path / "index")


# --- chunking ---------------------------------------------------------------

def test_split_respects_size_and_covers_text():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    pieces = split_text(text, size=300, overlap=40)
    assert len(pieces) > 5
    assert all(len(p) <= 300 for p in pieces)
    assert pieces[0].startswith("Sentence number 0")
    assert pieces[-1].endswith("Sentence number 199 is here.")


def test_split_prefers_sentence_boundaries():
    text = ("Alpha beta gamma delta. " * 30).strip()
    pieces = split_text(text, size=200, overlap=20)
    assert all(p.endswith(".") for p in pieces)


def test_chunks_never_start_mid_word():
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    text = " ".join(words[i % 6] for i in range(400))
    for piece in split_text(text, size=120, overlap=30):
        assert piece.split()[0] in words and piece.split()[-1] in words


def test_split_always_terminates_on_unbroken_text():
    pieces = split_text("x" * 5000, size=100, overlap=99)
    assert len(pieces) > 1


def test_split_rejects_bad_overlap():
    with pytest.raises(ValueError):
        split_text("hello", size=10, overlap=10)


def test_split_empty():
    assert split_text("   \n\n  ") == []


# --- loading ----------------------------------------------------------------

def test_load_txt_csv_and_skip_unknown(tmp_path):
    (tmp_path / "a.txt").write_text("Hello world", encoding="utf-8")
    (tmp_path / "b.csv").write_text("name,role\nAda,engineer\nLin,\n", encoding="utf-8")
    (tmp_path / "c.bin").write_bytes(b"\x00\x01")
    docs = {d.source: d for d in load_documents(tmp_path)}
    assert set(docs) == {"a.txt", "b.csv"}
    assert "name: Ada | role: engineer" in docs["b.csv"].text
    assert "Lin" in docs["b.csv"].text and "role: |" not in docs["b.csv"].text


def test_load_non_utf8_text(tmp_path):
    (tmp_path / "a.txt").write_bytes("caf\xe9".encode("cp1252"))
    assert load_documents(tmp_path)[0].text == "café"


def test_load_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nope")


def test_load_pdf_pages(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(200, 200)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 100 Td (Pelicans migrate south) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    with open(tmp_path / "birds.pdf", "wb") as fh:
        writer.write(fh)

    docs = load_documents(tmp_path)
    assert len(docs) == 1 and docs[0].page == 1
    assert "Pelicans migrate south" in docs[0].text
    assert pypdf  # silence unused warning


# --- embeddings and store ---------------------------------------------------

def test_offline_embedder_is_deterministic_and_discriminates():
    emb = OfflineEmbedder()
    a, b = emb.embed(["vector search with FAISS"]), emb.embed(["vector search with FAISS"])
    assert np.array_equal(a, b)
    vecs = emb.embed(["vector search with FAISS", "baking sourdough bread"])
    assert float(vecs[0] @ vecs[1]) == 0.0


def test_vector_store_ranks_by_cosine_and_roundtrips(tmp_path):
    from kb_bot.chunking import Chunk

    chunks = [Chunk("one", "a.txt", ".txt", 0), Chunk("two", "b.txt", ".txt", 0)]
    vectors = np.array([[1, 0, 0], [0, 5, 0]], dtype=np.float32)  # different lengths
    store = VectorStore.build(vectors, chunks, key="k1")
    top = store.search(np.array([0, 1, 0], dtype=np.float32), k=2)
    assert top[0][0].source == "b.txt" and top[0][1] == pytest.approx(1.0)

    store.save(tmp_path)
    assert VectorStore.load(tmp_path, "k1").chunks == chunks
    assert VectorStore.load(tmp_path, "other-key") is None
    assert VectorStore.load(tmp_path / "missing", "k1") is None


def test_openai_embedder_batches_and_retries():
    import openai

    calls = {"n": 0}

    class FakeEmbeddings:
        def create(self, model, input):
            calls["n"] += 1
            if calls["n"] == 1:
                raise openai.APIConnectionError(request=SimpleNamespace())
            return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 2.0]) for _ in input])

    import kb_bot.embeddings as mod
    mod.time.sleep = lambda s: None  # no real waiting in tests
    emb = OpenAIEmbedder(SimpleNamespace(embeddings=FakeEmbeddings()), "m", batch_size=2)
    out = emb.embed(["a", "b", "c"])
    assert out.shape == (3, 2)
    assert calls["n"] == 3  # 1 failure + 2 batches


# --- end to end -------------------------------------------------------------

def test_end_to_end_offline_answers_with_correct_source(tmp_path):
    kb = KnowledgeBase(make_settings(tmp_path))
    assert kb.load() > 0 and kb.mode == "offline"
    result = kb.ask("What is supervised learning?")
    assert result.sources
    assert result.sources[0].citation == "ml_fundamentals.txt"
    assert result.sources[0].score > 0.2


def test_unrelated_question_returns_nothing(tmp_path):
    kb = KnowledgeBase(make_settings(tmp_path))
    kb.load()
    result = kb.ask("zzzz qqqq xxxx")
    assert result.sources == [] and "could not find" in result.text


def test_index_reused_then_rebuilt_when_data_changes(tmp_path):
    data = tmp_path / "data"
    shutil.copytree(SAMPLE_DATA, data)
    settings = make_settings(tmp_path, data)

    first = KnowledgeBase(settings)
    first.load()
    assert first.rebuilt

    second = KnowledgeBase(settings)
    second.load()
    assert not second.rebuilt  # cache hit

    (data / "new.txt").write_text("Quokkas are small marsupials from Western Australia.")
    third = KnowledgeBase(settings)
    third.load()
    assert third.rebuilt
    assert third.ask("Where do quokkas live?").sources[0].citation == "new.txt"


def test_empty_data_folder_is_a_clear_error(tmp_path):
    (tmp_path / "data").mkdir()
    kb = KnowledgeBase(make_settings(tmp_path, tmp_path / "data"))
    with pytest.raises(ValueError, match="No readable text"):
        kb.load()


def test_ask_before_load_and_empty_question(tmp_path):
    kb = KnowledgeBase(make_settings(tmp_path))
    with pytest.raises(RuntimeError):
        kb.ask("hi")
    kb.load()
    with pytest.raises(ValueError):
        kb.ask("   ")


def test_cli_one_shot_json(tmp_path, capsys, monkeypatch):
    import json

    from kb_bot.cli import main

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "idx"))
    code = main(["--data-dir", str(SAMPLE_DATA), "-q", "What is deep learning?", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sources"] and payload["question"] == "What is deep learning?"


# --- regressions from the independent audit ---------------------------------

def run_cli(tmp_path, monkeypatch, *args):
    from kb_bot.cli import main

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "idx"))
    return main(["--data-dir", str(SAMPLE_DATA), *args])


@pytest.mark.parametrize("question", ["", "   "])
def test_cli_rejects_empty_question(tmp_path, monkeypatch, capsys, question):
    assert run_cli(tmp_path, monkeypatch, "-q", question) == 1
    assert "question is empty" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-1"])
def test_cli_rejects_bad_top_k(tmp_path, monkeypatch, value):
    with pytest.raises(SystemExit):
        run_cli(tmp_path, monkeypatch, "--top-k", value, "-q", "learning")


def test_cli_reports_bad_env_values_without_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MIN_SCORE", "oops")
    assert run_cli(tmp_path, monkeypatch, "-q", "learning") == 1
    assert "invalid setting" in capsys.readouterr().err


def test_settings_validation():
    for bad in ({"chunk_size": 0}, {"chunk_overlap": 500}, {"top_k": 0},
                {"min_score": float("nan")}, {"temperature": 5}):
        with pytest.raises(ValueError):
            Settings(**bad).validate()
    Settings().validate()


def test_cli_openai_failures_are_clean_errors(tmp_path, monkeypatch, capsys):
    import openai

    import kb_bot.bot as bot_module

    response = SimpleNamespace(status_code=401, headers={}, request=None)

    class FailingEmbedder:
        name = "failing"

        def embed(self, texts):
            raise openai.AuthenticationError("bad key sk-secret", response=response, body=None)

    monkeypatch.setattr(bot_module.KnowledgeBase, "_default_backends",
                        staticmethod(lambda s: (FailingEmbedder(), bot_module.ExtractiveAnswerer())))
    assert run_cli(tmp_path, monkeypatch, "-q", "learning") == 1
    err = capsys.readouterr().err
    assert "rejected the API key" in err and "sk-secret" not in err


def test_openai_answerer_with_stub_client():
    from kb_bot.answering import OpenAIAnswerer, build_prompt
    from kb_bot.chunking import Chunk

    seen = {}

    class FakeChat:
        def create(self, **kwargs):
            seen.update(kwargs)
            message = SimpleNamespace(content="  Otters eat shellfish [1].  ")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeChat()))
    chunks = [Chunk("Otters eat shellfish.", "animals.pdf", ".pdf", 0, 2)]
    out = OpenAIAnswerer(client, "m").answer("What do otters eat?", chunks)
    assert out == "Otters eat shellfish [1]."
    assert seen["model"] == "m" and "animals.pdf, p.2" in build_prompt("q", chunks)
    assert "using only the numbered context" in seen["messages"][0]["content"]


def test_cache_with_wrong_dimension_is_a_miss(tmp_path):
    import faiss

    from kb_bot.chunking import Chunk

    chunks = [Chunk("one", "a.txt", ".txt", 0)]
    VectorStore.build(np.array([[1, 0, 0]], dtype=np.float32), chunks, "k").save(tmp_path)
    bad = faiss.IndexFlatIP(5)
    bad.add(np.ones((1, 5), dtype=np.float32))
    faiss.write_index(bad, str(tmp_path / "index.faiss"))
    assert VectorStore.load(tmp_path, "k") is None


def test_offline_sources_match_displayed_passages(tmp_path):
    kb = KnowledgeBase(make_settings(tmp_path))
    kb.load()
    result = kb.ask("What is supervised learning?")
    assert len(result.sources) <= 2
    assert result.text.count("[2]") == (1 if len(result.sources) == 2 else 0)


def test_changing_chunk_settings_rebuilds_index(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "short.txt").write_text("Only one short chunk of text lives here.")
    base = make_settings(tmp_path, data)
    KnowledgeBase(base).load()
    other = KnowledgeBase(base.with_(chunk_size=600))
    other.load()
    assert other.rebuilt


def test_min_score_is_inclusive(tmp_path):
    kb = KnowledgeBase(make_settings(tmp_path))
    kb.load()
    top = kb.ask("What is supervised learning?").sources[0].score
    kb.settings = kb.settings.with_(min_score=top)
    assert kb.ask("What is supervised learning?").sources[0].score == pytest.approx(top)
