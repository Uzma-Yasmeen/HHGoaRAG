import hashlib
import re
from dataclasses import dataclass, asdict

SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")

@dataclass
class Chunk:
    id: str
    text: str
    language: str
    query_id: int | str
    strategy: str
    selected: bool
    source_query: str | None = None

    def json(self):
        return asdict(self)

def _make(text: str, language: str, query_id, strategy: str, selected: bool, source_query: str | None) -> Chunk:
    cleaned = " ".join(text.split())
    digest = hashlib.sha1(f"{language}|{query_id}|{strategy}|{cleaned}".encode()).hexdigest()[:16]
    return Chunk(digest, cleaned, language, query_id, strategy, selected, source_query)

def multi_strategy_chunks(text: str, language: str, query_id, selected: bool, source_query: str | None = None) -> list[Chunk]:
    """Native, semantic sentence, overlapping word, and answer-centred chunks."""
    text = " ".join((text or "").split())
    if len(text) < 35:
        return []
    chunks = [_make(text, language, query_id, "native_passage", selected, source_query)]
    sentences = [s.strip() for s in SENTENCE_END.split(text) if s.strip()]
    if len(sentences) > 2:
        for i in range(0, len(sentences), 1):
            window = " ".join(sentences[i:i + 2])
            if len(window) >= 45:
                chunks.append(_make(window, language, query_id, "sentence_window_2_overlap_1", selected, source_query))
    words = text.split()
    if len(words) > 140:
        for start in range(0, len(words), 90):
            window = " ".join(words[start:start + 120])
            if len(window.split()) >= 40:
                chunks.append(_make(window, language, query_id, "word_120_overlap_30", selected, source_query))
    if selected:
        answer_window = " ".join(words[:min(80, len(words))])
        chunks.append(_make(answer_window, language, query_id, "answer_centred", True, source_query))
    unique = {}
    for chunk in chunks:
        unique.setdefault(chunk.text.casefold(), chunk)
    return list(unique.values())
