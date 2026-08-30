"""LLM-as-a-judge: the technique from the CampusX "LLM Eval Methods" video
this suite follows -- prompting an LLM to score another model's output
against a stated rubric, rather than exact/fuzzy string matching.

Two judge calls, matching that video's reference-based vs. reference-free
split exactly:

  judge_faithfulness()  -- REFERENCE-FREE. No ground-truth answer is given
                            to the judge at all -- only the retrieved
                            context and the generated answer. Scores
                            whether every claim in the answer is actually
                            supported by that context. This is the
                            hallucination check: a reference-free judge is
                            required here specifically because hallucination
                            is a property of the answer's relationship to
                            its *own* context, not to some external ground
                            truth -- an answer can be faithful to bad
                            context, or unfaithful even when the context
                            happens to be the same topic as a correct
                            reference answer.

  judge_correctness()   -- REFERENCE-BASED. Given the MSMARCO-XI ground-
                            truth answer (Eng_Answer) as the reference, and
                            the target system's generated answer, scores
                            whether they convey the same information. This
                            is what "correctness" means here -- e.g. is the
                            model right, not just non-hallucinatory (a
                            model can be faithful to its context and still
                            wrong, if the retrieved context itself doesn't
                            contain the correct answer).

Deliberately a *separate* call from whatever GENERATION_BACKEND produced
the answer under test (see eval/target.py) -- judging a model with itself,
using the same call that produced the answer, is a known bias risk (a
model is more likely to rate its own output favorably).

PROVIDER-AGNOSTIC ON PURPOSE: this suite is public, and whoever runs it
against their own RAG project won't necessarily have an OpenAI key --
they might have an Anthropic key instead, or a local-only setup with no
hosted API key at all. The judge picks whichever real, working credential
is actually present rather than assuming OpenAI:

  EVAL_JUDGE_PROVIDER=openai      force OpenAI (needs OPENAI_API_KEY)
  EVAL_JUDGE_PROVIDER=openrouter  force OpenRouter (needs OPENROUTER_API_KEY)
  EVAL_JUDGE_PROVIDER=anthropic   force Anthropic (needs ANTHROPIC_API_KEY,
                                   or any credential `ant auth status` reports --
                                   see the Anthropic SDK's own auth resolution)
  EVAL_JUDGE_PROVIDER=auto        (default) OpenRouter, OpenAI, then Anthropic
                                   when the corresponding key is present

Both providers are called with a strict JSON output contract (OpenAI:
`response_format={"type": "json_object"}`, verified working against
JUDGE_MODEL_OPENAI before use; Anthropic: `output_config.format` with an
explicit json_schema, which per Anthropic's own docs guarantees the first
content block is valid JSON matching the schema). The same tolerant
fallback parser backs both anyway, in case a provider ever returns
something unexpected -- fail closed (verdict=False) rather than crash the
whole run over one bad example.

No live Anthropic key was available in the environment this suite was
built and tested in -- the OpenAI path has been run end-to-end repeatedly
(see this repo's README for real output); the Anthropic path is written
directly from Anthropic's own current API documentation (verified
`output_config` schema shape, current exception classes), not guessed, but
has NOT been exercised against a live response here. If you're the first
to run it with an Anthropic key, and something's off, that's the part to
check first.
"""
import json
import os
import time
from collections import deque
from dataclasses import dataclass
from threading import Lock

from eval import target

JUDGE_MODEL_OPENAI = os.environ.get("EVAL_JUDGE_MODEL_OPENAI", "gpt-5.4-mini")
JUDGE_MODEL_ANTHROPIC = os.environ.get("EVAL_JUDGE_MODEL_ANTHROPIC", "claude-opus-5")
JUDGE_MODEL_OPENROUTER = os.environ.get(
    "EVAL_JUDGE_MODEL_OPENROUTER", "openrouter/free"
)
JUDGE_MODEL_GROQ = os.environ.get("EVAL_JUDGE_MODEL_GROQ", "openai/gpt-oss-20b")

_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}

_openai_client = None
_anthropic_client = None
_openrouter_client = None
_groq_client = None
_groq_rate_lock = Lock()
_groq_last_request = 0.0
_groq_token_window: deque[tuple[float, int]] = deque()


class JudgeNotConfigured(RuntimeError):
    """No usable judge credential available."""


@dataclass
class JudgeVerdict:
    verdict: bool          # True = faithful / correct, False = hallucinated / incorrect
    reason: str
    judge_ms: float
    provider: str
    raw: str                # raw judge output, kept for debugging/audit


def _resolve_provider() -> str:
    # Best-effort: if the target has an app.config that loads a .env (this
    # suite's original target project does, via python-dotenv), importing
    # it here guarantees that's happened before the env var checks below --
    # relying on some other module having imported it first is fragile to
    # call order. Not every target does this, or even has an app.config at
    # all (it's OPTIONAL per eval/target.py's interface contract), so this
    # is silently skipped rather than required -- either way, the actual
    # judge credential can just be set in the shell environment directly.
    target.load_target()
    try:
        import app.config  # noqa: F401 -- imported for its load_dotenv() side effect, if any
    except ImportError:
        pass

    forced = os.environ.get("EVAL_JUDGE_PROVIDER", "auto").lower()
    if forced not in ("openai", "openrouter", "groq", "anthropic", "auto"):
        raise JudgeNotConfigured(
            f'EVAL_JUDGE_PROVIDER={forced!r} is not "openai", "openrouter", "groq", '
            '"anthropic", or "auto".'
        )

    if forced in ("groq", "auto") and os.environ.get("GROQ_API_KEY"):
        return "groq"
    if forced in ("openrouter", "auto") and os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    if forced in ("openai", "auto") and os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if forced in ("anthropic", "auto") and (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return "anthropic"
    if forced == "anthropic":
        # Don't hard-fail here: the Anthropic SDK also resolves credentials from
        # an `ant auth login` profile or Workload Identity Federation env vars,
        # neither of which show up as a plain env var check -- let the actual
        # client call in _call_anthropic() be the final word (it has its own
        # broad failure handling for the "truly nothing configured" case).
        return "anthropic"
    raise JudgeNotConfigured(
        "The judge needs a real LLM credential and found neither. Set one of:\n"
        "  GROQ_API_KEY        (Groq-compatible judge)\n"
        "  OPENROUTER_API_KEY  (OpenRouter-compatible judge)\n"
        "  OPENAI_API_KEY      (loaded via the target project's .env, see eval/target.py)\n"
        "  ANTHROPIC_API_KEY   (or ANTHROPIC_AUTH_TOKEN, or `ant auth login` -- see `ant auth status`)\n"
        "...or set EVAL_JUDGE_PROVIDER=anthropic explicitly if you're using a credential source "
        "that doesn't show up as one of the env vars above.\n"
        "Judge-based checks (faithfulness, correctness) can't run without one; retrieval, "
        "reliability, and latency checks don't need it."
    )


def _wait_for_groq_slot(estimated_tokens: int) -> None:
    """Stay below Groq's free-plan request and token throughput limits.

    Judge prompts include several retrieved chunks, so the token/minute limit
    is reached well before the nominal 30-request/minute limit. Track a
    conservative rolling estimate and stay below 6,500 tokens/minute, leaving
    headroom under Groq's documented 8,000-token free-plan limit.
    """
    global _groq_last_request
    budget = 6_500
    estimated_tokens = min(max(estimated_tokens, 1), budget)
    with _groq_rate_lock:
        while True:
            now = time.monotonic()
            while _groq_token_window and now - _groq_token_window[0][0] >= 60:
                _groq_token_window.popleft()
            used = sum(tokens for _, tokens in _groq_token_window)
            request_wait = max(0.0, 2.1 - (now - _groq_last_request))
            if used + estimated_tokens <= budget and request_wait <= 0:
                _groq_last_request = now
                _groq_token_window.append((now, estimated_tokens))
                return
            token_wait = (
                max(0.1, 60 - (now - _groq_token_window[0][0]))
                if used + estimated_tokens > budget and _groq_token_window
                else 0.0
            )
            time.sleep(max(request_wait, token_wait, 0.1))


def _call_groq(system_prompt: str, user_content: str) -> JudgeVerdict:
    """Use Groq's OpenAI-compatible endpoint with strict structured output."""
    global _groq_client
    import openai

    if _groq_client is None:
        _groq_client = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
            timeout=45.0,
            max_retries=0,
        )

    t0 = time.perf_counter()
    raw = ""
    max_attempts = 4
    # Roughly three characters/token is intentionally conservative for the
    # mixed English/Hindi evaluation text; reserve room for the verdict too.
    estimated_tokens = (len(system_prompt) + len(user_content)) // 3 + 200
    for attempt in range(1, max_attempts + 1):
        _wait_for_groq_slot(estimated_tokens)
        try:
            response = _groq_client.chat.completions.create(
                model=JUDGE_MODEL_GROQ,
                max_completion_tokens=200,
                reasoning_effort="low",
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "judge_verdict",
                        "strict": True,
                        "schema": _VERDICT_SCHEMA,
                    },
                },
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            raw = (response.choices[0].message.content or "").strip()
            if not raw:
                raise RuntimeError("Groq returned an empty final answer")
            break
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status not in (429, 500, 502, 503, 504) or attempt == max_attempts:
                raise
            response = getattr(exc, "response", None)
            headers = getattr(response, "headers", {}) or {}
            try:
                delay = max(2.0, float(headers.get("retry-after", 5)))
            except (TypeError, ValueError):
                delay = 5.0
            print(
                f"[judge/groq] transient response failure ({type(exc).__name__}); "
                f"retry {attempt}/{max_attempts} in {delay:.1f}s"
            )
            time.sleep(delay)

    judge_ms = (time.perf_counter() - t0) * 1000
    verdict, reason = _parse_verdict(raw)
    return JudgeVerdict(
        verdict=verdict,
        reason=reason,
        judge_ms=judge_ms,
        provider="groq",
        raw=raw,
    )


def _call_openrouter(system_prompt: str, user_content: str) -> JudgeVerdict:
    """Use OpenRouter's OpenAI-compatible chat-completions endpoint."""
    global _openrouter_client
    import openai

    if _openrouter_client is None:
        _openrouter_client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            timeout=45.0,
            max_retries=0,
        )

    t0 = time.perf_counter()
    response = None
    raw = ""
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            response = _openrouter_client.chat.completions.create(
                model=JUDGE_MODEL_OPENROUTER,
                max_completion_tokens=2000,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            choices = getattr(response, "choices", None)
            if not choices:
                raise RuntimeError("OpenRouter returned no choices")
            raw = (choices[0].message.content or "").strip()
            if not raw:
                raise RuntimeError("OpenRouter returned an empty final answer")
            break
        except Exception as exc:  # provider free-tier failures can be transient
            status = getattr(exc, "status_code", None)
            # Authentication, payment, permission, and missing-model errors
            # are permanent for this request. Retrying them only hides the
            # configuration problem behind several minutes of backoff.
            if status in (401, 402, 403, 404) or attempt == max_attempts:
                raise
            delay = attempt
            print(
                f"[judge/openrouter] transient response failure "
                f"({type(exc).__name__}: {exc}); "
                f"retry {attempt}/{max_attempts} in {delay}s"
            )
            time.sleep(delay)

    judge_ms = (time.perf_counter() - t0) * 1000
    verdict, reason = _parse_verdict(raw)
    return JudgeVerdict(
        verdict=verdict,
        reason=reason,
        judge_ms=judge_ms,
        provider="openrouter",
        raw=raw,
    )


def _parse_verdict(raw: str) -> tuple[bool, str]:
    try:
        parsed = json.loads(raw)
        return bool(parsed["verdict"]), str(parsed.get("reason", ""))
    except (json.JSONDecodeError, KeyError, TypeError):
        # Judge didn't follow the JSON contract -- fail closed (treat as a
        # negative verdict) rather than silently dropping the example, and
        # keep the raw text so it's auditable in the saved report.
        return False, f"[judge output did not parse as expected JSON: {raw[:200]!r}]"


def _call_openai(system_prompt: str, user_content: str) -> JudgeVerdict:
    global _openai_client
    import openai

    if _openai_client is None:
        _openai_client = openai.OpenAI()

    t0 = time.perf_counter()
    response = _openai_client.chat.completions.create(
        model=JUDGE_MODEL_OPENAI,
        max_completion_tokens=200,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    judge_ms = (time.perf_counter() - t0) * 1000
    raw = (response.choices[0].message.content or "").strip()
    verdict, reason = _parse_verdict(raw)
    return JudgeVerdict(verdict=verdict, reason=reason, judge_ms=judge_ms, provider="openai", raw=raw)


def _call_anthropic(system_prompt: str, user_content: str) -> JudgeVerdict:
    global _anthropic_client
    try:
        import anthropic
    except ImportError as e:
        raise JudgeNotConfigured(
            "EVAL_JUDGE_PROVIDER=anthropic needs the `anthropic` package: pip install anthropic"
        ) from e

    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY / ant auth profile

    t0 = time.perf_counter()
    try:
        response = _anthropic_client.messages.create(
            model=JUDGE_MODEL_ANTHROPIC,
            max_tokens=300,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
            output_config={"format": {"type": "json_schema", "schema": _VERDICT_SCHEMA}},
        )
    except anthropic.AuthenticationError as e:
        raise JudgeNotConfigured(f"Invalid Anthropic credentials: {e}") from e
    except TypeError as e:
        # Verified in the target RAG project's own earlier history (see its
        # app/generator.py git log): when NO Anthropic credentials exist at
        # all -- not even a resolvable `ant auth login` profile -- the SDK
        # fails locally in request construction with a bare TypeError, not
        # AuthenticationError (that one only fires for a real 401 from the
        # API). Without this, "truly unconfigured" surfaces as a confusing
        # crash instead of the same clear JudgeNotConfigured message the
        # OpenAI path already gives.
        raise JudgeNotConfigured(
            f"Anthropic credentials could not be resolved (`ant auth status` may help diagnose): {e}"
        ) from e
    judge_ms = (time.perf_counter() - t0) * 1000
    raw = next((b.text for b in response.content if b.type == "text"), "").strip()
    verdict, reason = _parse_verdict(raw)
    return JudgeVerdict(verdict=verdict, reason=reason, judge_ms=judge_ms, provider="anthropic", raw=raw)


def _call_judge(system_prompt: str, user_content: str) -> JudgeVerdict:
    provider = _resolve_provider()
    if provider == "groq":
        return _call_groq(system_prompt, user_content)
    if provider == "openrouter":
        return _call_openrouter(system_prompt, user_content)
    if provider == "anthropic":
        return _call_anthropic(system_prompt, user_content)
    return _call_openai(system_prompt, user_content)


_FAITHFULNESS_SYSTEM = """You are a strict fact-checking judge for a retrieval-augmented \
generation system. You will be given CONTEXT (retrieved document chunks) and an ANSWER a \
model produced from that context. Judge ONLY whether every factual claim in the ANSWER is \
directly supported by the CONTEXT -- do not judge whether the answer is true in general, \
only whether the CONTEXT supports it. An answer that correctly says the context doesn't \
cover the question is faithful (verdict: true). An answer that states anything not \
present in or directly implied by the CONTEXT is unfaithful (verdict: false), even if that \
claim happens to be true in reality.

Respond ONLY with a JSON object: {"verdict": true or false, "reason": "one short sentence"}"""


def judge_faithfulness(answer: str, context: str) -> JudgeVerdict:
    user_content = f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
    return _call_judge(_FAITHFULNESS_SYSTEM, user_content)


_CORRECTNESS_SYSTEM = """You are a grading judge comparing a model's ANSWER to a QUESTION \
against a REFERENCE ANSWER known to be correct. Judge whether the ANSWER conveys the same \
core information as the REFERENCE ANSWER -- wording, length, and extra (correct) detail \
don't matter, only whether the key fact(s) match. If the ANSWER says the documents don't \
contain the information, or refuses to answer, that is INCORRECT (verdict: false) -- the \
REFERENCE ANSWER proves the information was answerable.

Respond ONLY with a JSON object: {"verdict": true or false, "reason": "one short sentence"}"""


def judge_correctness(query: str, answer: str, reference_answer: str) -> JudgeVerdict:
    user_content = f"QUESTION:\n{query}\n\nREFERENCE ANSWER:\n{reference_answer}\n\nANSWER:\n{answer}"
    return _call_judge(_CORRECTNESS_SYSTEM, user_content)
