"""Benchmark the configured STT provider -> retrieval -> final answer endpoint."""
import argparse
import json
import mimetypes
from pathlib import Path

import httpx

def percentile(values,p):
    ordered=sorted(values);index=(len(ordered)-1)*p/100;lower=int(index);upper=min(lower+1,len(ordered)-1);return ordered[lower]+(ordered[upper]-ordered[lower])*(index-lower)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("audio_dir",type=Path);parser.add_argument("--url",default="http://localhost:8000");parser.add_argument("--mode",choices=["fast","enhanced"],default="fast");parser.add_argument("--language",choices=["en","hi","te"],default="en");parser.add_argument("--output",default="benchmark-results-voice.json");args=parser.parse_args()
    files=[p for p in args.audio_dir.iterdir() if p.suffix.lower() in {".wav",".mp3",".m4a",".webm",".ogg"}]
    if len(files)<10: raise SystemExit("Provide at least 10 audio query files for a reasonable voice benchmark.")
    records=[]
    with httpx.Client(timeout=40) as client:
        for path in files:
            with path.open("rb") as handle: response=client.post(f"{args.url}/api/voice-ask",params={"language":args.language,"mode":args.mode},files={"file":(path.name,handle,mimetypes.guess_type(path.name)[0] or "application/octet-stream")})
            response.raise_for_status();records.append(response.json())
    stages=sorted({stage for row in records for stage in row["timings_ms"]});summary={}
    for stage in stages:
        values=[row["timings_ms"][stage] for row in records if stage in row["timings_ms"]]
        summary[stage]={"p50_ms":round(percentile(values,50),3),"p70_ms":round(percentile(values,70),3),"p100_ms":round(max(values),3),"samples":len(values)}
    result={"scope":"audio-upload-to-final-answer, including configured STT provider and Groq when enhanced","mode":args.mode,"language":args.language,"runs":len(records),"summary":summary}
    Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result,indent=2))
if __name__=="__main__": main()
