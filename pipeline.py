import contextvars
import json
import os
import random
import re
import time

import groq
import httpx
from groq import Groq
from retrieval import Retriever

MODEL = "openai/gpt-oss-20b"

if os.environ.get("USE_FAKE") == "1":
    client = Groq(api_key="fake", base_url="http://127.0.0.1:8000", max_retries=0)
else:
    client = Groq(max_retries=0)   # we do our own retries so every one is visible

MIN_SCORE = 6.0
REFUSAL = "The papers do not say."
GIVE_UP_TEXT = "Sorry, I could not finish this answer in time. Please try again in a minute."
DAILY_TEXT = ("The daily limit for this assistant has been reached, so I cannot answer right now. "
              "No answer was guessed and nothing was lost. The allowance refills gradually, "
              "so please try again later today.")
DEADLINE_S = 20.0      # a person waiting longer than this has given up
MAX_TRIES = 5          # tries per AI call

CTX = contextvars.ContextVar("ctx")   # each request keeps its own clocks here


class GiveUp(Exception):
    def __init__(self, kind):
        super().__init__(kind)
        self.kind = kind   # "timeout", "rate_limited", "daily_limit" or "error"


def retry_mode():
    return os.environ.get("RETRY_MODE", "jitter")   # jitter, sync or none


def stream_once(messages, max_tokens, deadline):
    """One try of one AI call. Streams the reply, notes when the first word arrives."""
    start = time.perf_counter()
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        reasoning_effort="low",
        max_completion_tokens=max_tokens,
        stream=True,
        timeout=max(1.0, deadline - start),
    )
    text, first_at, usage = "", None, None
    for chunk in stream:
        now = time.perf_counter()
        if now > deadline:
            raise GiveUp("timeout")
        if chunk.choices and chunk.choices[0].delta.content:
            if first_at is None:
                first_at = now
            text += chunk.choices[0].delta.content
        u = getattr(chunk, "usage", None)
        if u is None:
            xg = getattr(chunk, "x_groq", None)
            u = getattr(xg, "usage", None) if xg else None
        if u:
            usage = u
    end = time.perf_counter()
    return {
        "text": text,
        "first_at": first_at,
        "first_ms": (first_at - start) * 1000 if first_at else None,
        "total_ms": (end - start) * 1000,
        "tokens_in": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "tokens_out": getattr(usage, "completion_tokens", 0) if usage else 0,
    }


def call_model(messages, max_tokens):
    """One AI call with our retry policy."""
    ctx = CTX.get()
    attempt = 0
    while True:
        t = time.perf_counter()
        if t >= ctx["deadline"]:
            raise GiveUp("timeout")
        kind, hint = None, 1.0
        try:
            result = stream_once(messages, max_tokens, ctx["deadline"])
            ctx["work_ms"] += (time.perf_counter() - t) * 1000
            return result
        except GiveUp:
            ctx["retry_ms"] += (time.perf_counter() - t) * 1000
            raise
        except groq.RateLimitError as e:
            kind = "rate_limited"
            ctx["rate_limited"] += 1
            try:
                hint = float(e.response.headers.get("retry-after", 1))
            except ValueError:
                hint = 1.0
            if "per day" in str(e):
                ctx["retry_ms"] += (time.perf_counter() - t) * 1000
                raise GiveUp("daily_limit")       # waiting seconds cannot refill a daily budget
            if retry_mode() == "none":
                ctx["retry_ms"] += (time.perf_counter() - t) * 1000
                raise GiveUp("rate_limited")      # experiment mode: never retry
        except groq.InternalServerError:
            kind = "error"
            hint = 0.5 * (2 ** attempt)
        except groq.APIStatusError:
            raise GiveUp("error")                 # 400, 401, 404 ... retrying cannot help
        except (groq.APIConnectionError, httpx.HTTPError):
            kind = "error"
            hint = 0.5 * (2 ** attempt)
        except groq.APIError:
            kind = "error"
            hint = 0.5 * (2 ** attempt)

        ctx["retry_ms"] += (time.perf_counter() - t) * 1000
        attempt += 1
        if retry_mode() == "sync":
            wait_s = hint                                          # everyone waits exactly as told
        else:
            wait_s = hint + random.uniform(0, 0.5 * hint + 0.5)    # jitter
        if time.perf_counter() + wait_s >= ctx["deadline"]:
            raise GiveUp("rate_limited" if kind == "rate_limited" else "timeout")
        if attempt >= MAX_TRIES:
            raise GiveUp(kind)
        ctx["retries"] += 1
        time.sleep(wait_s)
        ctx["wait_ms"] += wait_s * 1000


def parse_json(text):
    try:
        return json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    except Exception:
        return {}


def format_passages(hits):
    return "\n\n".join(
        f"[{h['paper']} p.{h['page']} | {h['section']}]\n{h['text']}" for h in hits
    )


# ---------- the 5 steps ----------

def step_understand(question):
    messages = [
        {"role": "system", "content":
            "You help search three research papers: 'Attention Is All You Need' (the Transformer), "
            "'Moshi' (a speech-text dialogue model) and 'Dynamic Chunking' (H-Net). "
            'Reply with JSON only: {"search_query": "5 to 10 keywords likely to appear in the paper text"}'},
        {"role": "user", "content": question},
    ]
    r = call_model(messages, 300)
    q = parse_json(r["text"]).get("search_query")
    if isinstance(q, list):
        q = " ".join(str(x) for x in q)
    return (q if isinstance(q, str) and q.strip() else question), r


def step_draft(question, hits):
    messages = [
        {"role": "system", "content":
            "Answer the question using ONLY the passages given. After each claim, cite its source "
            "like [paper p.N]. If the passages do not contain the answer, reply exactly: "
            + REFUSAL + " Be concise: at most 4 sentences."},
        {"role": "user", "content": f"Passages:\n{format_passages(hits)}\n\nQuestion: {question}"},
    ]
    return call_model(messages, 600)


def step_verify(answer, hits):
    messages = [
        {"role": "system", "content":
            "You are a strict fact checker. Given passages and an answer, reply with JSON only: "
            '{"supported": true} if every claim in the answer is stated in the passages, '
            'otherwise {"supported": false}.'},
        {"role": "user", "content": f"Passages:\n{format_passages(hits)}\n\nAnswer: {answer}"},
    ]
    r = call_model(messages, 300)
    return parse_json(r["text"]).get("supported") is True, r


# ---------- the pipeline ----------

def answer_question(question, retriever, deadline_s=DEADLINE_S):
    t0 = time.perf_counter()
    ctx = {"deadline": t0 + deadline_s, "retries": 0, "rate_limited": 0,
           "wait_ms": 0.0, "retry_ms": 0.0, "work_ms": 0.0}
    CTX.set(ctx)
    calls, steps, hits, query = [], {}, [], question
    status, outcome, final, draft_first_at, reason = "ok", "answered", REFUSAL, None, ""

    try:
        t = time.perf_counter()
        query, r = step_understand(question)
        calls.append(r)
        steps["understand_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        hits = retriever.search(query)                      # the tool call
        if hits[0]["score"] < MIN_SCORE:                    # branch: weak result, try again
            hits = retriever.search(question)
        steps["retrieve_ms"] = (time.perf_counter() - t) * 1000

        if hits[0]["score"] < MIN_SCORE:
            outcome = "refused_no_evidence"
        else:
            t = time.perf_counter()
            d = step_draft(question, hits)
            calls.append(d)
            draft_first_at = d["first_at"]
            steps["draft_ms"] = (time.perf_counter() - t) * 1000
            final = d["text"].strip()

            if REFUSAL in final:
                outcome = "refused_by_model"
            else:
                t = time.perf_counter()
                ok, v = step_verify(final, hits)
                calls.append(v)
                steps["verify_ms"] = (time.perf_counter() - t) * 1000
                if not ok:
                    outcome = "failed_verification"
                    final = "I could not verify an answer from the papers."
    except GiveUp as g:
        if g.kind == "daily_limit":
            status, outcome, final, reason = "rate_limited", "gave_up", DAILY_TEXT, "daily_limit"
        else:
            status, outcome, final, reason = g.kind, "gave_up", GIVE_UP_TEXT, g.kind

    return {
        "question": question,
        "answer": final, "search_query": query,
        "status": status,            # ok, rate_limited, timeout or error (goes in results.csv)
        "outcome": outcome,          # what kind of answer the person got
        "gave_up_reason": reason,
        "total_ms": (time.perf_counter() - t0) * 1000,
        "draft_first_token_ms": (draft_first_at - t0) * 1000 if draft_first_at else None,
        "steps": steps,
        "retries": ctx["retries"],
        "rate_limited_replies": ctx["rate_limited"],
        "working_ms": ctx["work_ms"],
        "waiting_ms": ctx["wait_ms"],
        "failed_attempts_ms": ctx["retry_ms"],
        "tokens_in": sum(c["tokens_in"] for c in calls),
        "tokens_out": sum(c["tokens_out"] for c in calls),
        "ai_calls": len(calls),
        "top_score": hits[0]["score"] if hits else 0.0,
        "sources": [f"{h['paper']} p.{h['page']} | {h['section']}" for h in hits],
    }


if __name__ == "__main__":
    retriever = Retriever(250)
    for q in ["How many attention heads does the base Transformer model use?",
              "How many dollars did it cost to train the H-Net model?"]:
        r = answer_question(q, retriever)
        print("\nQ:", q)
        print("A:", r["answer"])
        print("status:", r["status"], "| outcome:", r["outcome"], "| total ms:", round(r["total_ms"]))