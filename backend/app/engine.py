from __future__ import annotations

import asyncio
import heapq
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

UNSAFE = re.compile(r"\b(make|build|instructions?|steps?)\b.{0,30}\b(bomb|explosive|weapon|poison|malware)\b", re.I)
INJECTION = re.compile(r"ignore (all|the|any) (previous|above)|system prompt|developer message", re.I)
TOKEN = re.compile(r"[\w\u0900-\u097F\u0C00-\u0C7F]+", re.UNICODE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "the", "to", "was",
    "were", "what", "when", "where", "which", "who", "why", "with",
}

DEMO = {
    "en": [
        {"id":"demo-en-1","text":"Photosynthesis is the process by which green plants use sunlight, carbon dioxide and water to create glucose and oxygen.","language":"en","strategy":"native_passage"},
        {"id":"demo-en-2","text":"The Manhattan Project produced the first atomic weapons. Their use in 1945 contributed to Japan's surrender and transformed international security.","language":"en","strategy":"native_passage"},
        {"id":"demo-en-3","text":"ARPANET and TCP/IP contributed to the modern internet. Tim Berners-Lee proposed the World Wide Web in 1989.","language":"en","strategy":"sentence_window_2_overlap_1"},
    ],
    "hi": [{"id":"demo-hi-1","text":"प्रकाश संश्लेषण वह प्रक्रिया है जिसमें पौधे सूर्य के प्रकाश, कार्बन डाइऑक्साइड और पानी से ग्लूकोज और ऑक्सीजन बनाते हैं।","language":"hi","strategy":"native_passage"}],
    "te": [{"id":"demo-te-1","text":"కిరణజన్య సంయోగక్రియలో మొక్కలు సూర్యరశ్మి, కార్బన్ డయాక్సైడ్ మరియు నీటిని ఉపయోగించి గ్లూకోజ్‌ను తయారు చేసి ఆక్సిజన్‌ను విడుదల చేస్తాయి.","language":"te","strategy":"native_passage"}],
}

class RAGEngine:
    def __init__(self):
        self.model = None
        self.faiss = None
        self.indexes: dict[str, Any] = {}
        self.metadata: dict[str, list[dict]] = {}
        self.query_lookup: dict[str, dict[str, list[int]]] = {}
        self.index_dir = Path(os.getenv("INDEX_DIR", "data/indexes"))
        self.curated_dir = Path(os.getenv("CURATED_INDEX_DIR", "data/curated"))
        self.curated_metadata: dict[str, list[dict]] = {}
        self.threshold = float(os.getenv("ANSWERABILITY_THRESHOLD", "0.48"))
        self.top_k = int(os.getenv("RETRIEVAL_TOP_K", "6"))
        self.model_name = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-small")
        configured_lightweight = os.getenv("LIGHTWEIGHT_MODE")
        if configured_lightweight is None:
            try:
                render_cpu_count = float(os.getenv("RENDER_CPU_COUNT", "1"))
            except ValueError:
                render_cpu_count = 1.0
            self.lightweight = os.getenv("RENDER", "").lower() == "true" and render_cpu_count <= 0.1
        else:
            self.lightweight = configured_lightweight.strip().lower() in {"1", "true", "yes", "on"}
        self.ready = False
        self.demo_mode = True

    def _load_metadata(self):
        for lang in ("en", "hi", "te"):
            meta_path = self.index_dir / f"{lang}.jsonl"
            if not meta_path.exists():
                continue
            rows = []
            lookup: dict[str, list[int]] = {}
            with meta_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    index = len(rows)
                    rows.append(row)
                    key = self._normalized_query(row.get("source_query", ""))
                    if key:
                        lookup.setdefault(key, []).append(index)
            self.metadata[lang] = rows
            self.query_lookup[lang] = lookup

    def _load_curated(self):
        """Load a small sourced corpus kept separate from benchmark indexes."""
        for lang in ("en", "hi", "te"):
            path = self.curated_dir / f"{lang}.jsonl"
            if not path.exists():
                continue
            rows = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row.setdefault("language", lang)
                    row.setdefault("strategy", "curated_source")
                    rows.append(row)
            self.curated_metadata[lang] = rows

    def load(self):
        # Curated records do not depend on the embedding model, so keep them
        # available even when an offline/model startup falls back safely.
        self._load_curated()
        if self.lightweight:
            try:
                self._load_metadata()
                self.demo_mode = not bool(self.metadata or self.curated_metadata)
                self.ready = True
                print(f"[startup] lightweight lexical indexes ready: {', '.join(sorted(set(self.metadata) | set(self.curated_metadata)))}")
            except Exception as exc:
                print(f"[startup] lightweight index unavailable; safe demo fallback active: {exc}")
                self.ready = True
                self.demo_mode = True
            return
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            self.faiss = faiss
            self.model = SentenceTransformer(self.model_name, backend="onnx", model_kwargs={"file_name":os.getenv("MODEL_FILE","onnx/model_qint8_avx512_vnni.onnx")})
            self.model.max_seq_length = int(os.getenv("MODEL_MAX_LENGTH", "256"))
            self.model.encode(["query: warm up"], normalize_embeddings=True, show_progress_bar=False)
            self._load_metadata()
            for lang in self.metadata:
                index_path = self.index_dir / f"{lang}.faiss"
                if index_path.exists():
                    self.indexes[lang] = faiss.read_index(str(index_path))
            self.demo_mode = not bool(self.indexes or self.curated_metadata)
            self.ready = True
        except Exception as exc:
            print(f"[startup] optimized index unavailable; safe demo fallback active: {exc}")
            self.ready = True
            self.demo_mode = True

    @staticmethod
    def _lexical_score(query: str, text: str) -> float:
        q = {token for token in TOKEN.findall(query.casefold()) if token not in STOPWORDS}
        t = {token for token in TOKEN.findall(text.casefold()) if token not in STOPWORDS}
        return len(q & t) / max(1, len(q))

    @staticmethod
    def _normalized_query(text: str) -> str:
        return " ".join(TOKEN.findall(text.casefold()))

    def _curated_candidates(self, question: str, language: str) -> list[dict]:
        normalized = self._normalized_query(question)
        candidates = []
        for stored in self.curated_metadata.get(language, []):
            item = dict(stored)
            questions = [item.get("source_query", ""), *item.get("aliases", [])]
            normalized_questions = [self._normalized_query(value) for value in questions if value]
            exact = normalized in normalized_questions
            query_score = max((self._lexical_score(question, value) for value in questions if value), default=0.0)
            lexical_score = self._lexical_score(question, item.get("text", ""))
            score = 1.0 if exact else min(0.99, 0.72 * query_score + 0.28 * lexical_score)
            if score <= 0:
                continue
            item.update({
                "dense_score": 0.0,
                "lexical_score": lexical_score,
                "query_score": query_score,
                "score": score,
            })
            candidates.append(item)
        return candidates

    @staticmethod
    def _merge_candidates(*groups: list[dict]) -> list[dict]:
        merged = {}
        for group in groups:
            for item in group:
                existing = merged.get(item["id"])
                if existing is None or item.get("score", 0.0) > existing.get("score", 0.0):
                    merged[item["id"]] = item
        return sorted(
            merged.values(),
            key=lambda item: (
                item.get("score", 0.0),
                item.get("strategy") == "curated_source",
            ),
            reverse=True,
        )

    def retrieve(self, question: str, language: str):
        start = time.perf_counter()
        if self.model is not None and language in self.indexes:
            embed_start = time.perf_counter()
            vector = self.model.encode([f"query: {question}"], normalize_embeddings=True, show_progress_bar=False).astype("float32")
            embed_ms = (time.perf_counter() - embed_start) * 1000
            search_start = time.perf_counter()
            scores, ids = self.indexes[language].search(vector, min(self.top_k * 2, self.indexes[language].ntotal))
            search_ms = (time.perf_counter() - search_start) * 1000
            candidates = []
            for score, idx in zip(scores[0], ids[0]):
                if idx < 0 or idx >= len(self.metadata[language]): continue
                item = dict(self.metadata[language][idx]); item["dense_score"] = float(score)
                item["lexical_score"] = self._lexical_score(question, item["text"])
                item["query_score"] = self._lexical_score(question, item.get("source_query", ""))
                item["score"] = .76 * item["dense_score"] + .14 * item["lexical_score"] + .10 * item["query_score"]
                candidates.append(item)
            # Dataset suggestions and exact spoken variants should deterministically
            # reach their answer-bearing passage even when ANN ranking is crowded.
            exact_ids = self.query_lookup.get(language, {}).get(self._normalized_query(question), [])[:4]
            existing = {item["id"] for item in candidates}
            for idx in exact_ids:
                item = dict(self.metadata[language][idx])
                if item["id"] in existing: continue
                item.update({"dense_score": 1.0, "lexical_score": self._lexical_score(question, item["text"]), "query_score": 1.0, "score": 1.0})
                candidates.append(item); existing.add(item["id"])
            candidates = self._merge_candidates(candidates, self._curated_candidates(question, language))
            return candidates[:self.top_k], {"embedding": embed_ms, "retrieval": search_ms}
        if self.lightweight and language in self.metadata:
            normalized = self._normalized_query(question)
            lookup = self.query_lookup.get(language, {})
            if normalized in lookup:
                matched_queries = [(1.0, normalized)]
            else:
                matched_queries = heapq.nlargest(
                    self.top_k,
                    ((self._lexical_score(question, source_query), source_query) for source_query in lookup),
                    key=lambda match: match[0],
                )
            candidates = []
            seen = set()
            for query_score, source_query in matched_queries:
                if query_score <= 0:
                    continue
                for idx in lookup[source_query][:4]:
                    item = dict(self.metadata[language][idx])
                    if item["id"] in seen:
                        continue
                    item.update({
                        "dense_score": 0.0,
                        "lexical_score": self._lexical_score(question, item["text"]),
                        "query_score": query_score,
                        "score": query_score,
                    })
                    candidates.append(item)
                    seen.add(item["id"])
            candidates = self._merge_candidates(candidates, self._curated_candidates(question, language))
            return candidates[:self.top_k], {"embedding": 0.0, "retrieval": (time.perf_counter() - start) * 1000}
        if language in self.curated_metadata:
            candidates = self._curated_candidates(question, language)
            candidates.sort(key=lambda item: item["score"], reverse=True)
            return candidates[:self.top_k], {"embedding": 0.0, "retrieval": (time.perf_counter() - start) * 1000}
        candidates = [dict(item, score=self._lexical_score(question, item["text"])) for item in DEMO.get(language, DEMO["en"])]
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:self.top_k], {"embedding": 0.0, "retrieval": (time.perf_counter() - start) * 1000}

    @staticmethod
    def extractive_answer(question: str, sources: list[dict]) -> str:
        sentences = []
        q_tokens = set(TOKEN.findall(question.casefold()))
        for number, source in enumerate(sources[:3], 1):
            for sentence in re.split(r"(?<=[.!?।])\s+", source["text"]):
                tokens = set(TOKEN.findall(sentence.casefold()))
                score = len(q_tokens & tokens) / max(1, len(q_tokens))
                sentences.append((score, len(sentence), sentence.strip(), number))
        chosen, seen = [], set()
        for _, _, sentence, number in sorted(sentences, reverse=True):
            if sentence and sentence.casefold() not in seen:
                chosen.append(f"{sentence} [{number}]"); seen.add(sentence.casefold())
            if len(chosen) == 2: break
        return " ".join(chosen)

    async def groq_answer(self, question: str, sources: list[dict], language: str) -> str:
        from groq import AsyncGroq
        client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
        context = "\n".join(f"[{i}] {s['text']}" for i, s in enumerate(sources[:4], 1))
        prompt = f"Answer only from the evidence. Use at most 2 short sentences in language '{language}'. Cite every claim as [1], [2], etc. If evidence is insufficient, output exactly INSUFFICIENT_EVIDENCE.\nQuestion: {question}\nEvidence:\n{context}"
        response = await client.chat.completions.create(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"), messages=[{"role":"system","content":"You are a strictly grounded retrieval answer synthesizer. Never use outside knowledge."},{"role":"user","content":prompt}], temperature=0, max_completion_tokens=100)
        return (response.choices[0].message.content or "").strip()

    @staticmethod
    def grounding_score(answer: str, sources: list[dict]) -> float:
        answer_tokens = set(TOKEN.findall(re.sub(r"\[\d+\]", "", answer).casefold()))
        evidence_tokens = set(TOKEN.findall(" ".join(s["text"] for s in sources).casefold()))
        return len(answer_tokens & evidence_tokens) / max(1, len(answer_tokens))

    async def ask(self, question: str, language: str, mode: str):
        total_start = time.perf_counter(); timings = {}; request_id = str(uuid.uuid4())
        using_demo = language not in self.metadata and language not in self.curated_metadata
        guard_start = time.perf_counter()
        if UNSAFE.search(question):
            timings["guardrail"] = (time.perf_counter() - guard_start) * 1000; timings["total"] = (time.perf_counter() - total_start) * 1000
            return {"answer":"I can’t help with instructions that could cause harm.","status":"refused","mode":mode,"grounded":False,"confidence":1.0,"timings_ms":timings,"sources":[],"request_id":request_id}
        if INJECTION.search(question):
            timings["guardrail"] = (time.perf_counter() - guard_start) * 1000; timings["total"] = (time.perf_counter() - total_start) * 1000
            return {"answer":"I can answer questions from the indexed corpus, but I won’t follow instructions that attempt to override the retrieval policy.","status":"refused","mode":mode,"grounded":False,"confidence":1.0,"timings_ms":timings,"sources":[],"request_id":request_id}
        timings["guardrail"] = (time.perf_counter() - guard_start) * 1000
        sources, retrieval_timings = await asyncio.to_thread(self.retrieve, question, language); timings.update(retrieval_timings)
        best = sources[0]["score"] if sources else 0.0
        best_evidence_alignment = max(sources[0].get("query_score", 0.0), sources[0].get("lexical_score", 0.0)) if sources else 0.0
        # The tiny showcase corpus is lexical-only. A low gate lets an unrelated
        # question match a passage through one incidental token, so keep the demo
        # at least as strict as the real index.
        active_threshold = max(.50, self.threshold) if using_demo else max(.72, self.threshold) if self.lightweight else self.threshold
        if not sources or best < active_threshold or (not using_demo and best_evidence_alignment < .45):
            timings["total"] = (time.perf_counter() - total_start) * 1000
            return {"answer":"I couldn’t find enough supporting evidence in the indexed corpus to answer that reliably.","status":"insufficient_context","mode":mode,"grounded":False,"confidence":round(best,4),"timings_ms":timings,"sources":[],"request_id":request_id}
        # Low-scoring tail candidates must never become citations or synthesis input.
        sources = [source for source in sources if source["score"] >= active_threshold]
        synth_start = time.perf_counter(); answer_mode = "fast"
        answer = self.extractive_answer(question, sources)
        if mode == "enhanced" and os.getenv("GROQ_API_KEY"):
            try:
                groq_start = time.perf_counter(); candidate = await asyncio.wait_for(self.groq_answer(question, sources, language), timeout=8); timings["groq_generation"] = (time.perf_counter() - groq_start) * 1000
                if candidate != "INSUFFICIENT_EVIDENCE" and self.grounding_score(candidate, sources) >= .55 and re.search(r"\[\d+\]", candidate): answer, answer_mode = candidate, "enhanced"
                else: timings["grounding_fallback"] = .01
            except Exception: timings["groq_fallback"] = .01
        timings["synthesis"] = (time.perf_counter() - synth_start) * 1000
        timings["total"] = (time.perf_counter() - total_start) * 1000
        visible_sources = [{"id":s["id"],"text":s["text"],"language":s.get("language",language),"score":round(float(s["score"]),4),"strategy":s.get("strategy"),"title":s.get("title"),"source_url":s.get("source_url")} for s in sources[:4]]
        return {"answer":answer,"status":"answered","mode":answer_mode,"grounded":True,"confidence":round(min(1.0,best),4),"timings_ms":{k:round(v,3) for k,v in timings.items()},"sources":visible_sources,"request_id":request_id,"note":f"Demo corpus active for {language}; build or load that MSMARCO-XI index." if using_demo else None}
