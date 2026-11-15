import threading
import time

import pipeline

BUSY_TEXT = ("I am answering many questions at the moment and cannot take yours right now. "
             "Nothing was guessed. Please try again in about half a minute.")


def make_gated(max_inflight):
    lock = threading.Lock()
    state = {"inflight": 0}

    def gated(question, retriever, **kw):
        t0 = time.perf_counter()
        with lock:
            admitted = state["inflight"] < max_inflight
            if admitted:
                state["inflight"] += 1
        if not admitted:
            return {"question": question, "answer": BUSY_TEXT, "search_query": question,
                    "status": "rate_limited", "outcome": "shed_busy", "gave_up_reason": "busy",
                    "total_ms": (time.perf_counter() - t0) * 1000, "draft_first_token_ms": None,
                    "steps": {}, "retries": 0, "rate_limited_replies": 0, "working_ms": 0.0,
                    "waiting_ms": 0.0, "failed_attempts_ms": 0.0, "tokens_in": 0, "tokens_out": 0,
                    "ai_calls": 0, "top_score": 0.0, "sources": []}
        try:
            return pipeline.answer_question(question, retriever, **kw)
        finally:
            with lock:
                state["inflight"] -= 1

    return gated