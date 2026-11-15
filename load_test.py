import os
os.environ["USE_FAKE"] = "1"      # this script can only ever talk to the stand-in

import csv
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import httpx

from pipeline import answer_question
from retrieval import Retriever

SERVER = "http://127.0.0.1:8000"
SETTING = "w120-k4-keyword"
SEED = 7

# the stand-in here is a plan 10x bigger than the real free tier, so we can send hundreds of requests
BASE = {"rpm": 300, "tpm": 80000, "rpd": 10000, "tpd": 2000000,
        "rpd_start": None, "tpd_start": None,
        "fail_rate": 0.0, "hang_rate": 0.0, "hang_ms": 20000,
        "latency_scale": 1.0, "load_k": 0.1}


def evenly(n, seconds):
    return [i * seconds / n for i in range(n)]


SCENARIOS = [
    {"name": "steady", "seconds": 180, "offsets": evenly(60, 180), "config": {}},
    {"name": "burst", "seconds": 5, "offsets": evenly(100, 5), "config": {}},
    {"name": "sustained_overload", "seconds": 180, "offsets": evenly(180, 180), "config": {}},
    {"name": "per_minute_limit_halved", "seconds": 180, "offsets": evenly(60, 180),
     "config": {"rpm": 150, "tpm": 40000}},
    {"name": "daily_budget_exhausted", "seconds": 60, "offsets": evenly(30, 60),
     "config": {"rpd_start": 0, "tpd_start": 0}},
]

RESULT_COLS = ["request_id", "scenario", "source", "start_time", "first_token_ms",
               "total_ms", "status", "retries", "tokens_in", "tokens_out",
               "retrieval_setting", "answer_grade"]
DETAIL_COLS = ["request_id", "scenario", "total_ms", "draft_first_token_ms", "working_ms",
               "waiting_ms", "failed_attempts_ms", "rate_limited_replies", "retries", "outcome",
               "understand_ms", "retrieve_ms", "draft_ms", "verify_ms"]
META_COLS = ["scenario", "source", "requests_sent", "window_s", "duration_s",
             "offered_rate_per_min", "peak_inflight", "config"]


def append_csv(path, cols, row):
    is_new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if is_new:
            w.writeheader()
        w.writerow(row)


def load_questions(path="questions.txt"):
    out = []
    for line in open(path, encoding="utf-8"):
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 4 and parts[0] in ("answerable", "unanswerable", "tricky"):
            out.append(parts[1])
    return out


def pct(values, p):
    values = sorted(values)
    if not values:
        return None
    return values[max(0, -(-p * len(values) // 100) - 1)]    # nearest-rank


def r1(x):
    return round(x, 1) if x is not None else ""


def run_scenario(sc, retriever, questions):
    cfg = dict(BASE)
    cfg.update(sc["config"])
    httpx.post(f"{SERVER}/admin/config", json=cfg, timeout=10).raise_for_status()

    rng = random.Random(SEED)
    qs = [rng.choice(questions) for _ in sc["offsets"]]
    lock = threading.Lock()
    state = {"inflight": 0, "peak": 0}
    results = []

    def one(i, question):
        t_req = time.perf_counter()
        started = datetime.now().isoformat(timespec="milliseconds")
        with lock:
            state["inflight"] += 1
            state["peak"] = max(state["peak"], state["inflight"])
        try:
            r = answer_question(question, retriever)
        except Exception as e:
            print("   crash:", e)
            r = {"status": "error", "outcome": "crash", "total_ms": (time.perf_counter() - t_req) * 1000,
                 "draft_first_token_ms": None, "retries": 0, "tokens_in": 0, "tokens_out": 0,
                 "working_ms": 0, "waiting_ms": 0, "failed_attempts_ms": 0,
                 "rate_limited_replies": 0, "steps": {}}
        rid = f"{sc['name']}-{i:03d}"
        with lock:
            state["inflight"] -= 1
            append_csv("results.csv", RESULT_COLS, {
                "request_id": rid, "scenario": sc["name"], "source": "stand-in",
                "start_time": started, "first_token_ms": r1(r["draft_first_token_ms"]),
                "total_ms": r1(r["total_ms"]), "status": r["status"], "retries": r["retries"],
                "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"],
                "retrieval_setting": SETTING, "answer_grade": ""})
            st = r.get("steps", {})
            append_csv("load_details.csv", DETAIL_COLS, {
                "request_id": rid, "scenario": sc["name"], "total_ms": r1(r["total_ms"]),
                "draft_first_token_ms": r1(r["draft_first_token_ms"]),
                "working_ms": r1(r["working_ms"]), "waiting_ms": r1(r["waiting_ms"]),
                "failed_attempts_ms": r1(r["failed_attempts_ms"]),
                "rate_limited_replies": r["rate_limited_replies"], "retries": r["retries"],
                "outcome": r["outcome"], "understand_ms": r1(st.get("understand_ms")),
                "retrieve_ms": r1(st.get("retrieve_ms")), "draft_ms": r1(st.get("draft_ms")),
                "verify_ms": r1(st.get("verify_ms"))})
            results.append(r)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(sc["offsets"])) as pool:
        for i, (off, q) in enumerate(zip(sc["offsets"], qs)):
            delay = t0 + off - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            pool.submit(one, i, q)
    duration = time.perf_counter() - t0          # the pool waits for every request to finish

    n = len(results)
    append_csv("scenarios_meta.csv", META_COLS, {
        "scenario": sc["name"], "source": "stand-in", "requests_sent": n,
        "window_s": sc["seconds"], "duration_s": round(duration, 1),
        "offered_rate_per_min": round(n / sc["seconds"] * 60, 1),
        "peak_inflight": state["peak"], "config": str(cfg)})

    statuses = [r["status"] for r in results]
    totals = [r["total_ms"] for r in results if r["status"] == "ok"]
    print(f"{sc['name']}: {n} requests in {duration:.0f} s | "
          f"{ {s: statuses.count(s) for s in sorted(set(statuses))} } | "
          f"peak in flight {state['peak']} | ok total p50/p95: {r1(pct(totals, 50))} / {r1(pct(totals, 95))} ms")


if __name__ == "__main__":
    try:
        httpx.get(f"{SERVER}/admin/stats", timeout=5).raise_for_status()
    except Exception:
        raise SystemExit("The stand-in server is not running. Start it in the server window first.")

    retriever = Retriever(120)
    retriever.search("warm up")
    questions = load_questions()
    print(len(questions), "questions loaded. This takes about 12 minutes. Do not touch the server window.\n")
    for sc in SCENARIOS:
        run_scenario(sc, retriever, questions)
        time.sleep(8)                              # let the server settle between scenarios
    print("\nDone. Raw rows were added to results.csv. Details are in load_details.csv and scenarios_meta.csv.")