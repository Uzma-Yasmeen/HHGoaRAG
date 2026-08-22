"""Build focused English/Hindi/Telugu FAISS indexes from MSMARCO-XI."""
import argparse
import json
import os
import sys
from pathlib import Path

import faiss
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.chunking import multi_strategy_chunks

LOCAL_PARQUET: dict[str, str] = {}

def rows_for(language: str, include_negatives: bool = False):
    # The upstream repository was migrated from language configs to Parquet files.
    # Its validation files contain enough answer-bearing rows for the configured
    # focused index and are an order of magnitude smaller than the train shards.
    # Hindi also contains the paired original English passages.
    parquet_files = {
        "en": "validation/hinval.parquet",
        "hi": "validation/hinval.parquet",
        "te": "validation/telval.parquet",
    }
    filename = parquet_files[language]
    if filename not in LOCAL_PARQUET:
        print(f"[{language}] caching {filename} locally", flush=True)
        LOCAL_PARQUET[filename] = hf_hub_download(repo_id="ai4bharat/MSMARCO-XI", filename=filename, repo_type="dataset")
    dataset = load_dataset("parquet", data_files={"train": LOCAL_PARQUET[filename]}, split="train", streaming=True)
    for row in dataset:
        passages = row.get("passages") or {}
        texts = passages.get("English_passages" if language == "en" else "Translated_passages") or []
        selected = passages.get("is_selected") or [0] * len(texts)
        source_query = row.get("Eng_Query") if language == "en" else row.get("query")
        for text, chosen in zip(texts,selected):
            if chosen or include_negatives:
                yield row.get("query_id","unknown"), text, bool(chosen), source_query

def build(language: str, limit: int, output: Path, model, include_negatives: bool = False):
    chunks, seen = [], set()
    for query_id,text,selected,source_query in rows_for(language, include_negatives):
        for chunk in multi_strategy_chunks(text,language,query_id,selected,source_query):
            if chunk.text.casefold() not in seen:
                chunks.append(chunk); seen.add(chunk.text.casefold())
                if len(chunks) % 5000 == 0: print(f"[{language}] collected {len(chunks)}/{limit} unique chunks", flush=True)
                if len(chunks) >= limit: break
        if len(chunks) >= limit: break
    if not chunks: raise RuntimeError(f"No chunks produced for {language}")
    embeddings = model.encode([f"passage: {c.text}" for c in chunks],batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE","256")),normalize_embeddings=True,show_progress_bar=True).astype("float32")
    index = faiss.IndexHNSWFlat(embeddings.shape[1],32,faiss.METRIC_INNER_PRODUCT); index.hnsw.efConstruction=80; index.hnsw.efSearch=64; index.add(embeddings)
    output.mkdir(parents=True,exist_ok=True); faiss.write_index(index,str(output/f"{language}.faiss"))
    with (output/f"{language}.jsonl").open("w",encoding="utf-8") as handle:
        for chunk in chunks: handle.write(json.dumps(chunk.json(),ensure_ascii=False)+"\n")
    print(json.dumps({"language":language,"chunks":len(chunks),"dimension":embeddings.shape[1],"strategies":sorted({c.strategy for c in chunks})}))

if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--languages",nargs="+",default=["en","hi","te"]);parser.add_argument("--limit-per-language",type=int,default=25000);parser.add_argument("--output",type=Path,default=Path("data/indexes"));parser.add_argument("--include-negatives",action="store_true",help="Also index unselected distractor passages; off by default for broader positive-answer coverage");args=parser.parse_args()
    model=SentenceTransformer(os.getenv("MODEL_NAME","intfloat/multilingual-e5-small"),backend="onnx",model_kwargs={"file_name":os.getenv("MODEL_FILE","onnx/model_qint8_avx512_vnni.onnx")});model.max_seq_length=int(os.getenv("MODEL_MAX_LENGTH","256"));model.encode(["query: warm up"],normalize_embeddings=True)
    for lang in args.languages: build(lang,args.limit_per_language,args.output,model,args.include_negatives)
