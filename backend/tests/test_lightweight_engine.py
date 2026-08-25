import asyncio
import json
from pathlib import Path

from app.engine import RAGEngine


def test_render_free_automatically_selects_lightweight_mode(monkeypatch):
    monkeypatch.delenv("LIGHTWEIGHT_MODE", raising=False)
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("RENDER_CPU_COUNT", "0.1")

    assert RAGEngine().lightweight is True


def test_lightweight_index_answers_exact_and_rejects_partial_entity_match(tmp_path, monkeypatch):
    rows = {
        "en": [
            {
                "id": "photosynthesis",
                "text": "Plants use sunlight and water to produce glucose and oxygen.",
                "language": "en",
                "strategy": "native_passage",
                "source_query": "how does photosynthesis work",
            },
            {
                "id": "british-pm",
                "text": "Winston Churchill was prime minister of the United Kingdom.",
                "language": "en",
                "strategy": "native_passage",
                "source_query": "who was prime minister of the united kingdom",
            },
        ],
        "hi": [{"id": "hi", "text": "हिंदी प्रमाण", "language": "hi", "source_query": "हिंदी प्रश्न"}],
        "te": [{"id": "te", "text": "తెలుగు ఆధారం", "language": "te", "source_query": "తెలుగు ప్రశ్న"}],
    }
    for language, language_rows in rows.items():
        path = tmp_path / f"{language}.jsonl"
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in language_rows), encoding="utf-8")

    monkeypatch.setenv("LIGHTWEIGHT_MODE", "true")
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("CURATED_INDEX_DIR", str(tmp_path / "no-curated-data"))
    engine = RAGEngine()
    engine.load()

    exact = asyncio.run(engine.ask("How does photosynthesis work?", "en", "fast"))
    unsupported = asyncio.run(engine.ask("Who is the prime minister of India?", "en", "fast"))

    assert exact["status"] == "answered"
    assert exact["confidence"] == 1.0
    assert unsupported["status"] == "insufficient_context"
    assert all(language in engine.metadata for language in ("en", "hi", "te"))


def test_curated_goa_and_gk_sources_are_loaded_and_answered(monkeypatch):
    project_backend = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("LIGHTWEIGHT_MODE", "true")
    monkeypatch.setenv("INDEX_DIR", str(project_backend / "data" / "indexes"))
    monkeypatch.setenv("CURATED_INDEX_DIR", str(project_backend / "data" / "curated"))

    engine = RAGEngine()
    engine.load()

    goa = asyncio.run(engine.ask("What is the capital of Goa?", "en", "fast"))
    gk = asyncio.run(engine.ask("सौरमंडल में कितने ग्रह हैं?", "hi", "fast"))
    telugu = asyncio.run(engine.ask("గోవా ఎప్పుడు విముక్తి పొందింది?", "te", "fast"))

    for answer in (goa, gk, telugu):
        assert answer["status"] == "answered"
        assert answer["sources"]
        assert answer["sources"][0]["strategy"] == "curated_source"
        assert answer["sources"][0]["source_url"].startswith("https://")
