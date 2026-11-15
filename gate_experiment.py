import os
import time

import httpx

import load_test as lt          # also forces stand-in mode
from gate import make_gated
from retrieval import Retriever

CAPS = [6, 12, 24]

if __name__ == "__main__":
    try:
        httpx.get(f"{lt.SERVER}/admin/stats", timeout=5).raise_for_status()
    except Exception:
        raise SystemExit("The stand-in server is not running. Start it in the server window first.")
    os.environ["RETRY_MODE"] = "jitter"
    retriever = Retriever(120)
    retriever.search("warm up")
    questions = lt.load_questions()
    for cap in CAPS:
        lt.answer_question = make_gated(cap)
        for shape, seconds, count in [("burst", 5, 100), ("overload", 180, 180)]:
            sc = {"name": f"gate{cap}_{shape}", "seconds": seconds,
                  "offsets": lt.evenly(count, seconds), "config": {}}
            lt.run_scenario(sc, retriever, questions)
            time.sleep(8)
    print("\nDone.")