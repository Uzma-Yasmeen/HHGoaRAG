"""Native adapter for the rag-local-eval-loop target interface.

The evaluator supplies its own temporary passage index.  This module exposes
GoaVaani's real E5 embedding model and its grounded, extractive answer path so
the evaluator can exercise those components without going through GoaVaani's
production index.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
from dotenv import load_dotenv

from .engine import RAGEngine


load_dotenv(Path(__file__).resolve().parents[1] / ".env")

_engine = RAGEngine()
_model_lock = Lock()


def get_model():
    """Load and return the same multilingual E5 model used by GoaVaani."""
    if _engine.model is None:
        with _model_lock:
            if _engine.model is None:
                from sentence_transformers import SentenceTransformer

                _engine.model = SentenceTransformer(
                    _engine.model_name,
                    backend="onnx",
                    model_kwargs={"file_name": "onnx/model_qint8_avx512_vnni.onnx"},
                )
                _engine.model.max_seq_length = 256
                _engine.model.encode(
                    ["query: warm up"],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
    return _engine.model


def embed(texts: list[str]) -> np.ndarray:
    """Embed evaluator passages with the production model's E5 prefix."""
    model = get_model()
    return model.encode(
        [f"passage: {text}" for text in texts],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")


def embed_one(text: str) -> np.ndarray:
    """Embed one evaluator query with the production model's E5 prefix."""
    model = get_model()
    return model.encode(
        [f"query: {text}"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].astype("float32")


@dataclass(frozen=True)
class EvalAnswer:
    text: str
    grounded: bool
    generation_ms: float
    model: str = "GoaVaani fast extractive"


def generate_answer(query: str, results: list) -> EvalAnswer:
    """Run GoaVaani's fast grounded synthesis over evaluator-provided hits."""
    started = time.perf_counter()
    sources = [
        {
            "id": result.source,
            "text": result.text,
            "score": float(result.score),
            "lexical_score": _engine._lexical_score(query, result.text),
        }
        for result in results
    ]

    # Match RAGEngine.ask(): the refusal gate is based on the highest-ranked
    # passage, not on whichever lower-ranked passage happens to share a token.
    best_score = sources[0]["score"] if sources else 0.0
    best_alignment = sources[0]["lexical_score"] if sources else 0.0
    if not sources or best_score < _engine.threshold or best_alignment < 0.45:
        text = (
            "I couldn't find enough supporting evidence in the indexed corpus "
            "to answer that reliably."
        )
        grounded = False
    else:
        eligible = [source for source in sources if source["score"] >= _engine.threshold]
        text = _engine.extractive_answer(query, eligible)
        grounded = bool(text)

    return EvalAnswer(
        text=text,
        grounded=grounded,
        generation_ms=(time.perf_counter() - started) * 1000,
    )
