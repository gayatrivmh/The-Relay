import os
import time

import httpx

import load_test as lt          # also forces stand-in mode
from retrieval import Retriever
from tokengate import make_token_gate

# name, seconds, requests, stand-in limits, limit the gate is told
PLAN = [
    ("tokengate_burst", 5, 100, {}, None),
    ("tokengate_overload", 180, 180, {}, None),
    ("tokengate_halved_overload", 180, 180, {"rpm": 150, "tpm": 40000}, 40000),
]

if __name__ == "__main__":
    try:
        httpx.get(f"{lt.SERVER}/admin/stats", timeout=5).raise_for_status()
    except Exception:
        raise SystemExit("The stand-in server is not running. Start it in the server window first.")
    os.environ["RETRY_MODE"] = "jitter"
    retriever = Retriever(120)
    retriever.search("warm up")
    questions = lt.load_questions()
    for name, seconds, count, config, tpm in PLAN:
        lt.answer_question = make_token_gate(tpm or lt.BASE["tpm"])
        sc = {"name": name, "seconds": seconds, "offsets": lt.evenly(count, seconds), "config": config}
        lt.run_scenario(sc, retriever, questions)
        time.sleep(8)
    print("\nDone.")