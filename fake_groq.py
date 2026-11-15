import asyncio
import json
import math
import random
import re
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

# ---- numbers measured on the real service (from calibrate.py) ----
cal = json.load(open("calibration.json"))
SAMPLES = cal["first_token_ms_samples"]
MS_PER_TOKEN = cal["ms_per_output_token_median"] or 0.7
LOAD_K = 0.1   # 30 in flight -> about 4x slower, as measured
REFUSAL_BASE_MS = (cal["refusal_ms_median"] or 260) / (1 + LOAD_K * 39)

CONFIG = {
    "rpm": 30, "tpm": 8000, "rpd": 1000, "tpd": 200000,
    "rpd_start": None, "tpd_start": None,   # set to 0 to start with the daily budget gone
    "fail_rate": 0.0,    # chance of an HTTP 500 error
    "hang_rate": 0.0,    # chance of a reply that succeeds very slowly
    "hang_ms": 20000,
    "latency_scale": 1.0,
    "load_k": LOAD_K,
}


class Bucket:
    """Refills steadily, like the real limits do."""
    def __init__(self, capacity, per_seconds, start=None):
        self.capacity = capacity
        self.rate = capacity / per_seconds
        self.level = capacity if start is None else start
        self.t = time.monotonic()

    def _refill(self):
        now = time.monotonic()
        self.level = min(self.capacity, self.level + (now - self.t) * self.rate)
        self.t = now

    def available(self):
        self._refill()
        return self.level

    def take(self, n):
        self._refill()
        self.level -= n

    def wait_for(self, n):
        self._refill()
        return max(0.0, (n - self.level) / self.rate)


def build_buckets():
    global rpm_b, tpm_b, rpd_b, tpd_b
    rpm_b = Bucket(CONFIG["rpm"], 60)
    tpm_b = Bucket(CONFIG["tpm"], 60)
    rpd_b = Bucket(CONFIG["rpd"], 86400, CONFIG["rpd_start"])
    tpd_b = Bucket(CONFIG["tpd"], 86400, CONFIG["tpd_start"])


build_buckets()
INFLIGHT = 0
STATS = {"served": 0, "limited": 0, "failed": 0, "peak_inflight": 0}


def rl_headers():
    return {
        "x-ratelimit-limit-requests": str(CONFIG["rpd"]),
        "x-ratelimit-limit-tokens": str(CONFIG["tpm"]),
        "x-ratelimit-remaining-requests": str(max(0, int(rpd_b.available()))),
        "x-ratelimit-remaining-tokens": str(max(0, int(tpm_b.available()))),
        "x-ratelimit-reset-requests": f"{(rpd_b.capacity - rpd_b.level) / rpd_b.rate:.1f}s",
        "x-ratelimit-reset-tokens": f"{(tpm_b.capacity - tpm_b.level) / tpm_b.rate:.2f}s",
    }


def make_reply(messages):
    """Fake text, but shaped like what each pipeline step expects."""
    system = messages[0]["content"] if messages else ""
    user = messages[-1]["content"] if messages else ""
    if "search_query" in system:
        return json.dumps({"search_query": user[:120]}), 45
    if "fact checker" in system:
        return '{"supported": true}', 40
    m = re.search(r"\[([a-z_]+ p\.\d+)", user)
    cite = m.group(1) if m else "attention p.1"
    return f"Stand-in answer based on the passages. [{cite}]", 130


@app.post("/openai/v1/chat/completions")
async def chat(request: Request):
    global INFLIGHT
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "fake")
    prompt_tokens = 70 + sum(len(m.get("content", "")) for m in messages) // 4

    INFLIGHT += 1
    STATS["peak_inflight"] = max(STATS["peak_inflight"], INFLIGHT)
    factor = 1 + CONFIG["load_k"] * (INFLIGHT - 1)

    # 1) rate limits: if any bucket is short, refuse
    waits = {
        "requests per minute (RPM)": rpm_b.wait_for(1),
        "requests per day (RPD)": rpd_b.wait_for(1),
        "tokens per minute (TPM)": tpm_b.wait_for(prompt_tokens),
        "tokens per day (TPD)": tpd_b.wait_for(prompt_tokens),
    }
    kind, wait = max(waits.items(), key=lambda kv: kv[1])
    if wait > 0:
        STATS["limited"] += 1
        await asyncio.sleep(REFUSAL_BASE_MS * factor / 1000)
        INFLIGHT -= 1
        retry = max(1, math.ceil(wait))
        headers = rl_headers()
        headers["retry-after"] = str(retry)
        return JSONResponse(status_code=429, headers=headers, content={"error": {
            "message": f"Rate limit reached for model `{model}` on {kind}. "
                       f"Please try again in {retry}s.",
            "type": "requests" if "requests" in kind else "tokens",
            "code": "rate_limit_exceeded"}})

    rpm_b.take(1)
    rpd_b.take(1)
    tpm_b.take(prompt_tokens)
    tpd_b.take(prompt_tokens)
    headers = rl_headers()

    # 2) random failure
    if random.random() < CONFIG["fail_rate"]:
        STATS["failed"] += 1
        await asyncio.sleep(0.2 * factor)
        INFLIGHT -= 1
        return JSONResponse(status_code=500, headers=headers,
                            content={"error": {"message": "stand-in internal error", "type": "server_error"}})

    # 3) work out the delay and the reply
    first_delay = random.choice(SAMPLES) * factor * CONFIG["latency_scale"]
    if random.random() < CONFIG["hang_rate"]:
        first_delay += CONFIG["hang_ms"]       # succeeds, but slowly
    text, completion_tokens = make_reply(messages)
    completion_tokens = min(completion_tokens, body.get("max_completion_tokens") or 10**6)
    completion_tokens = max(1, int(completion_tokens * random.uniform(0.8, 1.2)))
    pieces = re.findall(r"\S+\s*", text) or [text]
    stream_ms = MS_PER_TOKEN * completion_tokens * factor
    per_piece = stream_ms / len(pieces) / 1000

    async def stream():
        global INFLIGHT
        try:
            await asyncio.sleep(first_delay / 1000)
            base = {"id": "fake", "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": model}
            for p in pieces:
                chunk = dict(base, choices=[{"index": 0, "delta": {"content": p}, "finish_reason": None}])
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(per_piece)
            tpm_b.take(completion_tokens)
            tpd_b.take(completion_tokens)
            usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                     "total_tokens": prompt_tokens + completion_tokens}
            final = dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
                         x_groq={"usage": usage})
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"
            STATS["served"] += 1
        finally:
            INFLIGHT -= 1

    return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)


@app.post("/admin/config")
async def set_config(request: Request):
    """Change settings and refill the buckets. Example: {"rpm": 15}"""
    CONFIG.update(await request.json())
    build_buckets()
    return CONFIG


@app.get("/admin/stats")
async def stats():
    return dict(STATS, inflight=INFLIGHT, rpm_left=rpm_b.available(), tpm_left=tpm_b.available(),
                rpd_left=rpd_b.available(), tpd_left=tpd_b.available(), config=CONFIG)