import asyncio
import json

from app.engine import RAGEngine


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
    engine = RAGEngine()
    engine.load()

    exact = asyncio.run(engine.ask("How does photosynthesis work?", "en", "fast"))
    unsupported = asyncio.run(engine.ask("Who is the prime minister of India?", "en", "fast"))

    assert exact["status"] == "answered"
    assert exact["confidence"] == 1.0
    assert unsupported["status"] == "insufficient_context"
    assert all(language in engine.metadata for language in ("en", "hi", "te"))
