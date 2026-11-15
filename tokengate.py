import threading
import time
from collections import deque

import pipeline

BUSY_TEXT = ("I am answering many questions at the moment and cannot take yours right now. "
             "Nothing was guessed. Please try again in about half a minute.")


class Cheap:
    """Same search, fewer pieces: a cheaper and weaker answer."""

    def __init__(self, base, top_k):
        self.base = base
        self.top_k = top_k

    def search(self, question):
        return self.base.search(question, top_k=self.top_k)


def make_token_gate(tpm, est_normal=2300, est_cheap=1600, degrade_at=0.6, shed_at=0.95):
    lock = threading.Lock()
    finished = deque()            # (time, tokens) of requests finished in the last 60 s
    reserved = {"tokens": 0}      # estimated tokens of requests still running

    def load_now(now):
        while finished and finished[0][0] < now - 60:
            finished.popleft()
        return sum(t for _, t in finished) + reserved["tokens"]

    def gated(question, retriever, **kw):
        t0 = time.perf_counter()
        with lock:
            load = load_now(time.monotonic())
            if load + est_cheap > shed_at * tpm:
                mode, est = "shed", 0
            elif load + est_normal > degrade_at * tpm:
                mode, est = "cheap", est_cheap
            else:
                mode, est = "normal", est_normal
            reserved["tokens"] += est
        if mode == "shed":
            return {"question": question, "answer": BUSY_TEXT, "search_query": question,
                    "status": "rate_limited", "outcome": "shed_busy", "gave_up_reason": "busy",
                    "total_ms": (time.perf_counter() - t0) * 1000, "draft_first_token_ms": None,
                    "steps": {}, "retries": 0, "rate_limited_replies": 0, "working_ms": 0.0,
                    "waiting_ms": 0.0, "failed_attempts_ms": 0.0, "tokens_in": 0, "tokens_out": 0,
                    "ai_calls": 0, "top_score": 0.0, "sources": []}
        r = None
        try:
            search_with = retriever if mode == "normal" else Cheap(retriever, 2)
            r = pipeline.answer_question(question, search_with, **kw)
        finally:
            with lock:
                reserved["tokens"] -= est
                if r is not None:
                    finished.append((time.monotonic(), r["tokens_in"] + r["tokens_out"]))
        if mode == "cheap" and r["outcome"] == "answered":
            r["outcome"] = "answered_cheap"
        return r

    return gated