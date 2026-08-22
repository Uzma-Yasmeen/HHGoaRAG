# GoaVaani — Voice-enabled grounded RAG

GoaVaani is a submission-ready foundation for HH Goa 2026. It combines an animated Goa product site with a real FastAPI retrieval backend, ElevenLabs Scribe v2 speech-to-text, focused MSMARCO-XI indexes and an optional Groq answer synthesizer.

## What is honest about this version

- The interface never displays invented benchmark numbers.
- **Fast mode** measures preprocessing, guardrails, ONNX embedding, FAISS search and extractive grounded synthesis.
- **Enhanced mode** uses Groq (`openai/gpt-oss-20b` by default) and includes Groq generation in the measured total.
- Speech-to-text latency is returned by `/api/transcribe` and is not silently mixed into a text-only benchmark.
- The full voice demo is real only after `ELEVENLABS_API_KEY` is configured.
- When the real index is absent, the backend and website explicitly identify the small demonstration corpus.

## Architecture

1. ElevenLabs Scribe v2 transcribes English, Hindi or Telugu audio.
2. A pre-warmed `intfloat/multilingual-e5-small` ONNX encoder embeds the final query.
3. A language-specific FAISS HNSW index retrieves candidates.
4. Dense similarity and lexical overlap are combined.
5. Unsafe, injected or low-evidence questions are refused.
6. Fast mode extracts two cited evidence sentences. Enhanced mode asks Groq to rewrite only from retrieved evidence and falls back if citation/grounding validation fails.

The dataset is chunked offline using four strategies: native passage, two-sentence semantic windows with one-sentence overlap, 120-word windows with 30-word overlap, and answer-centred selected-passage chunks. Exact text is deduplicated before indexing.

## Windows: simplest setup

Requirements: Node.js 22+, Python 3.11 or 3.12, and about 5 GB free for packages and a small index.

1. Double-click `setup-windows.bat` once.
2. Open `backend\.env` and add:

   ```env
   ELEVENLABS_API_KEY=your_key
   GROQ_API_KEY=your_key
   ```

3. Double-click `run-local.bat`.
4. Open `http://localhost:5173` if it does not open automatically.

Typed showcase questions work even if the backend is unavailable. Voice requires ElevenLabs Scribe v2. Enhanced mode requires Groq.

### If speech-to-text is not working

1. Confirm the file is named exactly `backend\.env` and not `.env.txt`.
2. Confirm it contains `ELEVENLABS_API_KEY=...` without quotation marks or spaces around `=`.
3. Close both command windows and run `run-local.bat` again; environment changes are read only when the backend starts.
4. Double-click `check-connection.bat`.
5. Its JSON should show `"stt_ready": true`. If the backend is still warming, `"model_state": "warming"` is normal and voice transcription can already run.
6. Open `http://localhost:8000/health` in the browser. If it does not open, run `start-backend.bat` and keep that window open so you can see the exact error.
7. Use the website through `http://localhost:5173` rather than opening an HTML file directly, and allow microphone access in the browser.

The website now retries the backend connection every five seconds and displays separate readiness states for Backend, Voice, Knowledge and Groq.

## Build the real focused index

Start with `build-index-small.bat`. It builds 20,000 answer-bearing chunks for each language. Restart the backend afterwards. For a larger index:

```bash
cd backend
.venv\Scripts\activate
python scripts\build_index.py --languages en hi te --limit-per-language 50000
```

The full dataset is about 55.6 GB, so this project deliberately streams and indexes a disclosed language subset instead of downloading every language.

## Languages and question coverage

This application currently supports exactly three complete language paths: English (`en`), Hindi (`hi`) and Telugu (`te`). ElevenLabs can transcribe more languages, but they are intentionally not exposed until a matching retrieval index and evaluation set are added.

Showcase mode contains only a tiny local corpus and deliberately offers one supported question for each selected language. After the MSMARCO-XI index is built, the application can answer other factual questions when relevant evidence exists in the indexed subset. It will not reliably answer live news, opinions, personal advice or facts absent from the corpus. The question chips then come from the real index so the user can choose known-answerable examples.

Good query forms include “What is…?”, “Who…?”, “When…?”, “Where…?”, “Why…?” and “How…?” across science, technology, history, general health concepts, people and places.

## Latency interpretation

No universal latency value is hard-coded because it depends on your laptop, network and API region. Run `benchmark-fast.bat` to obtain the actual post-transcription text-RAG P50, P70 and P100. Run `scripts\benchmark_voice.py` with recorded audio files to measure complete speech-to-answer latency. Fast mode targets under 200 ms after transcription; complete voice-to-answer and Groq-enhanced requests are reported separately and may exceed 200 ms.

## Benchmark correctly

Keep the backend running, then use `benchmark-fast.bat`. It performs 10 excluded warmups and 100 uncached requests, then writes `backend\benchmark-results-fast.json`. It reports single P50, P70 and P100 values for every executed stage and wall-clock HTTP time.

To include Groq generation:

```bash
cd backend
.venv\Scripts\activate
python scripts\benchmark.py --runs 50 --warmup 5 --mode enhanced --output benchmark-results-enhanced.json
```

Do not call the fast benchmark “voice-to-answer latency.” For the final submission, show:

- text RAG P50/P70/P100;
- Groq-enhanced P50/P70/P100;
- ElevenLabs transcription timing from actual audio;
- complete user-observed voice-to-answer timing from the browser/network test.

## Useful endpoints

- `GET /health` — readiness and configured services
- `POST /api/ask` — guarded retrieval and answer generation
- `POST /api/transcribe` — ElevenLabs Scribe v2 transcription
- `POST /api/voice-ask` — one fully measured voice-to-final-answer request
- `GET /docs` — interactive FastAPI documentation

## Production notes

Keep API keys only on the backend. Deploy the website and API in the same region where possible, keep at least one backend instance warm, build the index before deployment and mount or copy it into the service image. Do not use a host that sleeps during judging if P100 latency matters.

See `DEPLOYMENT.md` for the recommended order: validate locally, deploy the always-warm backend, connect the website URL, and only then run the final benchmark.

Before submission, replace the interface’s benchmark placeholders only with the real JSON results, test at least 100 varied questions across all three languages, record refusals, and preserve the raw result files in the GitHub repository.

For a genuine voice benchmark, record at least ten `.wav`, `.mp3`, `.m4a`, `.webm` or `.ogg` questions into one folder and run:

```bash
cd backend
.venv\Scripts\activate
python scripts\benchmark_voice.py path\to\audio-queries --language en --mode fast
```
