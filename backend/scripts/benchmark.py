"""Honest text-pipeline benchmark. Groq is included when --mode enhanced is used."""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

import httpx

QUERIES=["How does photosynthesis work?","What was the impact of the Manhattan Project?","Who invented the World Wide Web?","Why is the sky blue?","How does vaccination protect people?","What causes ocean tides?","How do solar panels generate electricity?","What is machine learning?","How does inflation affect prices?","Why do leaves change colour?"]
def percentile(values,p):
    ordered=sorted(values);index=(len(ordered)-1)*p/100;lower=int(index);upper=min(lower+1,len(ordered)-1);return ordered[lower]+(ordered[upper]-ordered[lower])*(index-lower)
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--url",default="http://localhost:8000");parser.add_argument("--runs",type=int,default=100);parser.add_argument("--warmup",type=int,default=10);parser.add_argument("--mode",choices=["fast","enhanced"],default="fast");parser.add_argument("--output",default="benchmark-results.json");args=parser.parse_args()
    records=[]
    with httpx.Client(timeout=20) as client:
        health=client.get(f"{args.url}/health");health.raise_for_status();health=health.json()
        if not health.get("model_ready"):
            raise SystemExit("Backend model is still warming. Wait for /health to report model_ready=true, then run the benchmark again.")
        for i in range(args.warmup): client.post(f"{args.url}/api/ask",json={"question":QUERIES[i%len(QUERIES)]+f" warmup {i}","language":"en","mode":args.mode})
        for i in range(args.runs):
            question=random.choice(QUERIES)+f" (test case {i})";wall=time.perf_counter();response=client.post(f"{args.url}/api/ask",json={"question":question,"language":"en","mode":args.mode});response.raise_for_status();payload=response.json();payload["wall_ms"]=(time.perf_counter()-wall)*1000;records.append(payload)
    stage_names=sorted({key for row in records for key in row["timings_ms"]}|{"wall"});summary={}
    for stage in stage_names:
        values=[row["wall_ms"] if stage=="wall" else row["timings_ms"].get(stage) for row in records];values=[v for v in values if v is not None]
        summary[stage]={"p50_ms":round(percentile(values,50),3),"p70_ms":round(percentile(values,70),3),"p100_ms":round(max(values),3),"mean_ms":round(statistics.mean(values),3),"samples":len(values)}
    result={"mode":args.mode,"runs":args.runs,"warmup_excluded":args.warmup,"scope":"text-to-final-answer; speech-to-text not included","index_ready":bool(health.get("index_ready")),"corpus":"MSMARCO-XI index" if health.get("index_ready") else "small demonstration corpus","summary":summary}
    Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result,indent=2))
if __name__=="__main__": main()
