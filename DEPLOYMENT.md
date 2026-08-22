# GoaVaani deployment order

## Recommended sequence

1. Run `setup-windows.bat` once.
2. Put `ELEVENLABS_API_KEY=...` in `backend\.env`.
3. Run `run-local.bat`, open `http://localhost:8000/health`, and confirm `"stt_ready": true` and `"stt_provider": "elevenlabs"`.
4. Test one typed and one spoken question in English, Hindi and Telugu.
5. Run `build-index-small.bat`, restart the backend, and confirm `"index_ready": true`.
6. Deploy the backend container to an always-warm container host in an India-near region. The backend must receive `ELEVENLABS_API_KEY`, `ALLOWED_ORIGINS`, and optionally `GROQ_API_KEY` as secrets/environment variables.
7. Build the website with `NEXT_PUBLIC_RAG_API_URL=https://YOUR-BACKEND-URL` and deploy the static/frontend output.
8. Run both fast-text and voice benchmarks against the deployed URL. Report them separately.

## Why not deploy the backend as a Netlify Function?

The FastAPI service keeps the embedding model and FAISS indexes warm in memory. A short-lived serverless function can cold-start and reload those assets, making a strict P100 latency claim unreliable. Netlify is acceptable for the website, but the Python retrieval backend should run as an always-on container or VPS process.

## Cloud Run container example

From the repository root, build the `backend` directory as the container context. Configure at least one minimum instance for judging so the embedding model and index stay loaded. Use enough memory for the ONNX model plus all three FAISS indexes. Start with 2 GiB for the small index and increase if startup logs show memory pressure.

## VPS example

Install Docker, build `backend/Dockerfile`, run the container with `--restart unless-stopped`, and place Caddy or Nginx in front for HTTPS. Keep the frontend and backend in nearby regions and set `ALLOWED_ORIGINS` to the exact deployed frontend origin.

## Latency claim

- **Text fast path:** target under 200 ms after warm-up; report measured P50/P70/P100.
- **Voice path:** report STT and text-RAG separately, plus total microphone-upload-to-answer time.
- **Enhanced mode:** Groq is optional and its generation time must be included. Do not call it a sub-200 ms non-LLM result.
