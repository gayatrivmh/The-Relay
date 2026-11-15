import os
import time

import httpx

import load_test as lt        # this also forces stand-in mode
from retrieval import Retriever

MODES = ["none", "sync", "jitter"]     # no retry, wait exactly retry-after, wait retry-after plus a random extra


def counters():
    s = httpx.get(f"{lt.SERVER}/admin/stats", timeout=5).json()
    return s["served"], s["limited"]


if __name__ == "__main__":
    try:
        httpx.get(f"{lt.SERVER}/admin/stats", timeout=5).raise_for_status()
    except Exception:
        raise SystemExit("The stand-in server is not running. Start it in the server window first.")
    retriever = Retriever(120)
    retriever.search("warm up")
    questions = lt.load_questions()

    plan = []
    for shape, seconds, count in [("burst", 5, 100), ("overload", 180, 180)]:
        for mode in MODES:
            plan.append((mode, {"name": f"retry_{mode}_{shape}", "seconds": seconds,
                                "offsets": lt.evenly(count, seconds), "config": {}}))
    plan.append(("jitter", {"name": "daily_budget_exhausted_failfast", "seconds": 60,
                            "offsets": lt.evenly(30, 60),
                            "config": {"rpd_start": 0, "tpd_start": 0}}))

    for mode, sc in plan:
        os.environ["RETRY_MODE"] = mode
        s0, l0 = counters()
        lt.run_scenario(sc, retriever, questions)
        s1, l1 = counters()
        print(f"   server: {s1 - s0} calls completed, {l1 - l0} calls refused (retry mode: {mode})")
        time.sleep(8)
    print("\nDone.")