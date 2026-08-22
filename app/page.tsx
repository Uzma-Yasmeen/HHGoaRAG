"use client";

import { FormEvent, useEffect, useRef, useState } from "react";

type Source = { id: string; text: string; language: string; score: number; strategy?: string };
type Answer = { answer: string; status: "answered" | "refused" | "insufficient_context"; mode: "fast" | "enhanced" | "demo"; grounded: boolean; confidence: number; timings_ms: Record<string, number>; sources: Source[]; request_id?: string; note?: string };
type Health = { status: string; model_ready: boolean; index_ready: boolean; stt_ready: boolean; stt_provider?: "elevenlabs" | null; groq_ready: boolean; languages: string[] };
type BenchmarkStage = { p50_ms: number; p70_ms: number; p100_ms: number; mean_ms: number; samples: number };
type Benchmark = { mode: string; runs: number; warmup_excluded: number; scope: string; index_ready?: boolean; corpus?: string; summary: Record<string, BenchmarkStage> };
const API = process.env.NEXT_PUBLIC_RAG_API_URL
  || (process.env.NODE_ENV === "production"
    ? "https://goavaani-api.onrender.com"
    : "http://localhost:8000");
const sampleQuestions: Record<string, string[]> = {
  en: ["How does photosynthesis work?", "Who invented the World Wide Web?", "Why do ocean tides occur?"],
  hi: ["प्रकाश संश्लेषण कैसे काम करता है?", "विश्व व्यापी वेब का आविष्कार किसने किया?", "समुद्र में ज्वार क्यों आता है?"],
  te: ["కిరణజన్య సంయోగక్రియ ఎలా పనిచేస్తుంది?", "వరల్డ్ వైడ్ వెబ్‌ను ఎవరు కనుగొన్నారు?", "సముద్రంలో అలలు ఎందుకు వస్తాయి?"],
};
const demoQuestions: Record<string, string> = { en: "How does photosynthesis work?", hi: "प्रकाश संश्लेषण कैसे काम करता है?", te: "కిరణజన్య సంయోగక్రియ ఎలా పనిచేస్తుంది?" };
const demos = [
  { keys: ["manhattan", "project", "impact"], answer: "The immediate impact of the Manhattan Project’s success was the creation and wartime use of atomic weapons, which accelerated the end of World War II and began the nuclear age. [1]", source: "The Manhattan Project produced the first atomic weapons. Their use in 1945 contributed to Japan’s surrender and transformed international security.", language: "en" },
  { keys: ["photosynthesis", "plant", "sunlight"], answer: "Photosynthesis converts light energy into chemical energy. Plants use carbon dioxide and water to produce glucose, releasing oxygen as a by-product. [1]", source: "Photosynthesis is the process by which green plants transform light energy, carbon dioxide and water into glucose and oxygen.", language: "en" },
  { keys: ["internet", "invented", "web"], answer: "The internet developed through several research networks rather than being invented by one person. The World Wide Web was later created by Tim Berners-Lee. [1]", source: "ARPANET and TCP/IP contributed to the modern internet. Tim Berners-Lee proposed the World Wide Web in 1989.", language: "en" },
  { keys: ["प्रकाश", "संश्लेषण", "पौधे"], answer: "प्रकाश संश्लेषण में पौधे सूर्य के प्रकाश, कार्बन डाइऑक्साइड और पानी का उपयोग करके ग्लूकोज बनाते हैं और ऑक्सीजन छोड़ते हैं। [1]", source: "प्रकाश संश्लेषण वह प्रक्रिया है जिसमें पौधे सूर्य के प्रकाश, कार्बन डाइऑक्साइड और पानी से ग्लूकोज और ऑक्सीजन बनाते हैं।", language: "hi" },
  { keys: ["కిరణజన్య", "మొక్కలు", "సూర్యరశ్మి"], answer: "కిరణజన్య సంయోగక్రియలో మొక్కలు సూర్యరశ్మి, కార్బన్ డయాక్సైడ్ మరియు నీటిని ఉపయోగించి గ్లూకోజ్‌ను తయారు చేసి ఆక్సిజన్‌ను విడుదల చేస్తాయి. [1]", source: "కిరణజన్య సంయోగక్రియలో మొక్కలు సూర్యరశ్మి, కార్బన్ డయాక్సైడ్ మరియు నీటిని ఉపయోగించి గ్లూకోజ్‌ను తయారు చేసి ఆక్సిజన్‌ను విడుదల చేస్తాయి.", language: "te" },
];

function demoAnswer(question: string): Answer {
  const normalized = question.toLowerCase();
  if (/\b(bomb|weapon|kill|suicide|explosive)\b/i.test(question) && /how|build|make|instructions/i.test(question)) return { answer: "I can’t help with instructions that could cause harm.", status: "refused", mode: "demo", grounded: false, confidence: 1, timings_ms: {}, sources: [], note: "Local showcase result — start the backend for a measured pipeline." };
  const hit = demos.map(item => ({ item, score: item.keys.filter(key => normalized.includes(key)).length })).sort((a, b) => b.score - a.score)[0];
  if (!hit || hit.score === 0) return { answer: "I couldn’t find enough supporting evidence in the local showcase corpus. Start the backend and build the MSMARCO-XI index for broad coverage.", status: "insufficient_context", mode: "demo", grounded: false, confidence: .12, timings_ms: {}, sources: [], note: "This is an intentional grounded refusal, not a generated guess." };
  return { answer: hit.item.answer, status: "answered", mode: "demo", grounded: true, confidence: Math.min(.94, .58 + hit.score * .12), timings_ms: {}, sources: [{ id: "demo-1", text: hit.item.source, language: hit.item.language, score: .91, strategy: "native_passage" }], note: "Local showcase result — start the backend for real measured timings." };
}
const fmt = (value?: number) => value === undefined ? "—" : `${value.toFixed(value < 10 ? 2 : 1)} ms`;

export default function Home() {
  const [question, setQuestion] = useState("");
  const [language, setLanguage] = useState("en");
  const [mode, setMode] = useState<"fast" | "enhanced">("fast");
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [working, setWorking] = useState(false);
  const [recording, setRecording] = useState(false);
  const [backend, setBackend] = useState<"checking" | "ready" | "demo">("checking");
  const [health, setHealth] = useState<Health | null>(null);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [benchmark, setBenchmark] = useState<Benchmark | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);

  useEffect(() => {
    let active = true;
    const check = async () => {
      try {
        const response = await fetch(`${API}/health`, { signal: AbortSignal.timeout(3500) });
        if (!response.ok) throw new Error();
        const payload = await response.json();
        if (active) { setHealth(payload); setBackend("ready"); }
      } catch { if (active) { setHealth(null); setBackend("demo"); } }
    };
    check(); const timer = window.setInterval(check, 5000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);
  useEffect(() => {
    fetch("/benchmark-results-fast.json", { cache: "no-store" })
      .then(response => response.ok ? response.json() : Promise.reject())
      .then(setBenchmark)
      .catch(() => setBenchmark(null));
  }, []);
  useEffect(() => {
    if (!answer) return;
    const frame = window.requestAnimationFrame(() => document.querySelector(".result-section")?.scrollIntoView({ behavior: "smooth", block: "start" }));
    return () => window.cancelAnimationFrame(frame);
  }, [answer]);
  useEffect(() => {
    if (!health?.index_ready) return;
    fetch(`${API}/api/suggestions?language=${language}`, { signal: AbortSignal.timeout(3000) })
      .then(response => response.ok ? response.json() : Promise.reject())
      .then(payload => setSuggestions(payload.questions || []))
      .catch(() => setSuggestions([]));
  }, [language, health?.index_ready]);
  async function ask(raw = question) {
    const q = raw.trim(); if (!q || working) return;
    setQuestion(q); setWorking(true); setAnswer(null);
    try {
      const response = await fetch(`${API}/api/ask`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: q, language, mode }), signal: AbortSignal.timeout(mode === "fast" ? 5000 : 15000) });
      if (!response.ok) throw new Error(); setAnswer(await response.json()); setBackend("ready");
    } catch { setBackend("demo"); setAnswer(demoAnswer(q)); } finally { setWorking(false); }
  }
  async function submit(event: FormEvent) { event.preventDefault(); await ask(); }
  async function toggleRecording() {
    if (recording && recorder.current) { recorder.current.stop(); setRecording(false); return; }
    if (health && !health.stt_ready) {
      setAnswer({ answer: "Voice transcription is not configured. Add ELEVENLABS_API_KEY to backend\\.env, restart the backend, and confirm that Voice shows ElevenLabs ready.", status: "insufficient_context", mode: "demo", grounded: false, confidence: 0, timings_ms: {}, sources: [] });
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const preferredType = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"].find(type => MediaRecorder.isTypeSupported(type));
      const mediaRecorder = new MediaRecorder(stream, preferredType ? { mimeType: preferredType } : undefined); recorder.current = mediaRecorder; chunks.current = [];
      mediaRecorder.ondataavailable = event => { if (event.data.size) chunks.current.push(event.data); };
      mediaRecorder.onstop = async () => {
        const voiceRequestStart = performance.now();
        stream.getTracks().forEach(track => track.stop()); setWorking(true); setAnswer(null);
        try {
          const audioType = mediaRecorder.mimeType || chunks.current[0]?.type || "audio/webm";
          const audio = new Blob(chunks.current, { type: audioType });
          if (audio.size < 1000) throw new Error("The recording was too short. Speak for at least one second before stopping");
          const extension = audioType.includes("mp4") ? "m4a" : "webm";
          const form = new FormData(); form.append("file", audio, `question.${extension}`);
          const transcriptionResponse = await fetch(`${API}/api/transcribe?language=${language}`, { method: "POST", body: form, signal: AbortSignal.timeout(35000) });
          if (!transcriptionResponse.ok) { const error = await transcriptionResponse.json().catch(() => ({})); throw new Error(error.detail || `Voice transcription failed (${transcriptionResponse.status})`); }
          const transcription = await transcriptionResponse.json();
          const transcript = String(transcription.text || "").trim();
          if (!transcript) throw new Error("No speech was recognized. Try again in a quieter place");
          setQuestion(transcript);
          const answerResponse = await fetch(`${API}/api/ask`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: transcript, language, mode }), signal: AbortSignal.timeout(mode === "fast" ? 10000 : 20000) });
          if (!answerResponse.ok) { const error = await answerResponse.json().catch(() => ({})); throw new Error(error.detail || `Answer request failed (${answerResponse.status})`); }
          const payload = await answerResponse.json();
          const ragTimings = { ...(payload.timings_ms || {}) };
          const ragTotal = ragTimings.total;
          delete ragTimings.total;
          payload.timings_ms = {
            speech_to_text: transcription.timings_ms?.speech_to_text,
            ...ragTimings,
            rag_total: ragTotal,
            total: Math.round((performance.now() - voiceRequestStart) * 1000) / 1000,
          };
          setAnswer(payload); setBackend("ready");
        }
        catch (error) { const message = error instanceof Error ? error.message : "Voice transcription failed"; setAnswer({ answer: `${message}. Check the connection panel, then try again.`, status: "insufficient_context", mode: "demo", grounded: false, confidence: 0, timings_ms: {}, sources: [] }); }
        finally { setWorking(false); }
      };
      mediaRecorder.start(250); setRecording(true);
    } catch { setAnswer({ answer: "Microphone permission was not granted. Type your question below instead.", status: "refused", mode: "demo", grounded: false, confidence: 0, timings_ms: {}, sources: [] }); }
  }

  return <main>
    <nav className="nav hero-nav shell"><a className="brand" href="#top"><span className="brand-mark">G</span><span>GoaVaani</span></a><div className="nav-links"><a href="#ask">Ask</a><a href="#architecture">Architecture</a><a href="#metrics">Metrics</a></div><div className={`status ${backend}`}><i />{backend === "ready" ? "Pipeline ready" : backend === "checking" ? "Checking backend" : "Showcase mode"}</div></nav>
    <section className="hero hero-full" id="top"><img className="hero-background" src="/goa-cinematic-hero.png" alt="Goa coastline with a Portuguese-style resort, coconut palms and the Arabian Sea at golden hour"/><div className="hero-shade"/><div className="hero-content shell"><div className="hero-copy"><p className="hero-name"><strong>GoaVaani</strong><span>Multilingual Voice RAG</span></p><p className="eyebrow"><span>VOICE-NATIVE RAG</span> · HH GOA 2026</p><h1>Ask in your language.<br/><em>Answer with evidence.</em></h1><p className="lede">Search trusted passages in English, हिन्दी or తెలుగు—and receive a fast, cited answer that knows when to say “I don’t know.”</p><div className="hero-actions"><a className="primary" href="#ask">Start asking <span>↗</span></a><a className="secondary" href="#architecture">See how it works</a></div><div className="hero-stats"><div><strong>3</strong><span>Language indexes</span></div><div><strong>&lt;200</strong><span>ms text target</span></div><div><strong>100%</strong><span>Source-linked</span></div></div></div><div className="hero-float"><div className="live-line"><span className="pulse">●</span><div><b>Ready across three languages</b><small>English · हिन्दी · తెలుగు</small></div></div><div className="hero-float-rule"/><p>Grounded retrieval across science, history, technology, people and places.</p><div className="coast-label-inline"><span>GOA · INDIA</span><b>15.2993° N</b></div></div></div></section>
    <section className="ask-section" id="ask"><div className="shell ask-grid"><div className="ask-intro"><p className="eyebrow dark">TRY THE PIPELINE</p><h2>One question.<br/>A traceable answer.</h2><p>Use voice or text. Every response carries its evidence, confidence and real stage timings.</p><div className="privacy"><span>✓</span><div><b>Grounded by design</b><small>Weak evidence produces a refusal—not a hallucination.</small></div></div><div className="ask-guide"><b>What can I ask?</b><p>Ask any factual “what”, “who”, “when”, “where”, “why” or “how” question whose evidence exists in the index. The buttons are examples—not the only supported questions.</p><div className="scope-chips"><span>Science</span><span>History</span><span>Technology</span><span>Geography</span><span>People</span><span>Health concepts</span></div><small>Not designed for live news, opinions, creative writing, personal advice or facts absent from the corpus.</small></div></div><div className="console"><div className="console-top"><div className="lang-tabs">{[["en","English"],["hi","हिन्दी"],["te","తెలుగు"]].map(([code,label]) => <button key={code} className={language === code ? "active" : ""} onClick={() => setLanguage(code)}>{label}</button>)}</div><select value={mode} onChange={e => setMode(e.target.value as "fast" | "enhanced")}><option value="fast">⚡ Fast grounded</option><option value="enhanced">✦ Groq enhanced</option></select></div><div className="voice-zone"><button className={`mic ${recording ? "recording" : ""}`} onClick={toggleRecording} aria-label="Toggle recording"><span>{recording ? "■" : "●"}</span><i/><i/><i/></button><h3>{recording ? "Listening… tap to stop" : working ? "Finding grounded evidence…" : "Tap to speak"}</h3><p>or type a question below</p></div><form onSubmit={submit} className="question-form"><input value={question} onChange={e => setQuestion(e.target.value)} placeholder="Ask a factual question from the indexed corpus…"/><button disabled={!question.trim() || working}>{working ? "···" : "Ask →"}</button></form><div className="sample-label">{health?.index_ready ? "Suggestions from your index" : "Available showcase question"}</div><div className="samples">{(health?.index_ready ? (suggestions.length ? suggestions : sampleQuestions[language]) : [demoQuestions[language]]).map(sample => <button key={sample} onClick={() => ask(sample)}>{sample}</button>)}</div><div className="connection-panel"><div><i className={backend === "ready" ? "ok" : "bad"}/><span>Backend</span><b>{backend === "ready" ? "Connected" : "Not connected"}</b></div><div><i className={health?.stt_ready ? "ok" : "bad"}/><span>Voice</span><b>{health?.stt_ready ? `ElevenLabs ready` : "Key missing / offline"}</b></div><div><i className={health?.index_ready ? "ok" : "warn"}/><span>Knowledge</span><b>{health?.index_ready ? "MSMARCO index" : "Sample corpus"}</b></div><div><i className={health?.groq_ready ? "ok" : "warn"}/><span>Enhanced</span><b>{health?.groq_ready ? "Groq ready" : "Fast mode only"}</b></div></div></div></div></section>
    {answer && <section className="result-section"><div className="shell result-grid"><article className="answer-card"><div className="answer-meta"><span className={`result-state ${answer.status}`}>{answer.status.replace("_", " ")}</span><span>{answer.mode === "enhanced" ? "Groq-enhanced" : answer.mode === "fast" ? "Fast extractive" : "Local showcase"}</span><span>{Math.round(answer.confidence * 100)}% confidence</span></div><h3>Answer</h3><p className="answer-text">{answer.answer}</p>{answer.note && <p className="answer-note">{answer.note}</p>}<div className="sources"><h4>Supporting evidence</h4>{answer.sources.length ? answer.sources.map((source,index) => <div className="source" key={source.id}><b>[{index+1}]</b><p>{source.text}</p><span>{source.language.toUpperCase()} · {Math.round(source.score*100)}% · {source.strategy || "passage"}</span></div>) : <p className="muted">No source passed the evidence threshold.</p>}</div></article><aside className="timing-card"><p className="eyebrow dark">REQUEST TRACE</p><div className="total-time"><strong>{fmt(answer.timings_ms.total)}</strong><span>{answer.timings_ms.speech_to_text !== undefined ? "voice-to-answer total" : "text-to-answer total"}</span></div>{Object.entries(answer.timings_ms).filter(([key]) => key !== "total").map(([key,value]) => <div className="timing-row" key={key}><span>{key.replaceAll("_", " ")}</span><b>{fmt(value)}</b></div>)}<p className="timing-foot">Stages may be nested, so their values are diagnostic and should not be added together. Only executed stages are shown.</p></aside></div></section>}
    <section className="architecture shell" id="architecture"><div className="section-head"><p className="eyebrow dark">ENGINEERING, NOT THEATRE</p><h2>The latency-aware architecture</h2><p>Heavy work happens before users arrive. Models and indexes stay warm in memory; every live request follows a small, observable path.</p></div><div className="pipeline"><div className="pipe-card coral"><span>01</span><b>ElevenLabs Scribe v2</b><small>Multilingual speech transcription</small></div><i>→</i><div className="pipe-card sand"><span>02</span><b>ONNX E5-small</b><small>Pre-warmed embedding</small></div><i>→</i><div className="pipe-card aqua"><span>03</span><b>FAISS HNSW</b><small>Hybrid evidence search</small></div><i>→</i><div className="pipe-card green"><span>04</span><b>Guarded answer</b><small>Non-LLM fast path</small></div></div><div className="architecture-notes"><article><b>Four-way chunking</b><p>Native passages, sentence windows, overlapping word windows and answer-centred chunks are deduplicated before indexing.</p></article><article><b>Two honest modes</b><p>Fast mode is non-LLM extractive synthesis. Enhanced mode is optional and includes Groq latency in the trace.</p></article><article><b>Three focused indexes</b><p>English, Hindi and Telugu are sharded for lower memory use and predictable retrieval.</p></article><article><b>Grounding gate</b><p>Low similarity, unsafe intent, prompt injection or unsupported synthesis produces a structured refusal.</p></article></div></section>
    <section className="metrics" id="metrics"><div className="shell metrics-grid"><div><p className="eyebrow">BENCHMARK CONTRACT</p><h2>Numbers earned,<br/>never decorated.</h2><p>The benchmark tool warms the model, uses uncached questions and reports stage-level P50, P70 and P100. Speech and Groq are reported whenever used.</p>{benchmark && <p><small>{benchmark.runs} measured text requests after {benchmark.warmup_excluded} excluded warmups · {benchmark.corpus || (benchmark.index_ready ? "indexed corpus" : "demo corpus")} · wall-clock HTTP latency</small></p>}</div><div className="metric-board"><div><span>P50</span><strong>{benchmark?.summary.wall ? fmt(benchmark.summary.wall.p50_ms) : "Run benchmark"}</strong><small>{benchmark ? "Median observed run" : "No result artifact yet"}</small></div><div><span>P70</span><strong>{benchmark?.summary.wall ? fmt(benchmark.summary.wall.p70_ms) : "Run benchmark"}</strong><small>{benchmark ? "70th percentile" : "Use benchmark-fast.bat"}</small></div><div><span>P100</span><strong>{benchmark?.summary.wall ? fmt(benchmark.summary.wall.p100_ms) : "Run benchmark"}</strong><small>{benchmark ? "Worst observed run" : "No placeholder claim"}</small></div></div></div></section>
    <footer className="shell"><a className="brand" href="#top"><span className="brand-mark">G</span><span>GoaVaani</span></a><p>Built for HH Goa 2026 · Evidence before eloquence.</p><span>#RAGInGoa</span></footer>
  </main>;
}
