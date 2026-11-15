import csv
import functools
import os
import re
import sys
import time
from datetime import datetime

from pipeline import answer_question
from retrieval import Retriever

SOURCE = "stand-in" if os.environ.get("USE_FAKE") == "1" else "real"
SCENARIO = "quality"
SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 250
RERANK = len(sys.argv) > 2 and sys.argv[2] == "hybrid"
TOPK = int(sys.argv[3]) if len(sys.argv) > 3 else 4
SETTING = f"w{SIZE}-k{TOPK}-{'hybrid' if RERANK else 'keyword'}" + os.environ.get("SETTING_SUFFIX", "")
REVIEW = f"quality_review_{SETTING}.csv"

RESULT_COLS = ["request_id", "scenario", "source", "start_time", "first_token_ms",
               "total_ms", "status", "retries", "tokens_in", "tokens_out",
               "retrieval_setting", "answer_grade"]
REVIEW_COLS = ["id", "kind", "question", "key", "answer", "auto_grade", "outcome", "sources", "search_query"]
STOP = set("the and with that this its from into than then which are was were for has have".split())


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
            out.append({"kind": parts[0], "question": parts[1], "key": parts[2], "pages": parts[3]})
    return out


def content_ok(key, answer):
    """First guess only: do the key's numbers (or main words) appear in the answer?"""
    a = answer.lower().replace(",", "")
    need = re.findall(r"\d+(?:\.\d+)?", key)
    if not need:
        need = [w[:6] for w in re.findall(r"[a-z]+", key.lower()) if len(w) >= 5 and w not in STOP]
    if not need:
        return False
    hit = sum(1 for w in need if re.search(r"\b" + re.escape(w), a))
    return hit / len(need) >= 0.6


def citation_ok(pages, answer):
    """Is at least one cited paper and page within 1 page of the key?"""
    key_pages = [(p, int(n)) for p, n in re.findall(r"([a-z_]+) p\.(\d+)", pages)]
    cited = [(p, int(n)) for p, n in re.findall(r"([a-z_]+) p\.(\d+)", answer)]
    return any(p == kp and abs(n - kn) <= 1 for p, n in cited for kp, kn in key_pages)


def grade(q, r):
    if r["outcome"] == "gave_up":
        return ""                                   # no answer was given at all
    refused = r["outcome"] in ("refused_no_evidence", "refused_by_model", "failed_verification")
    if refused:
        return "refused_correctly" if q["kind"] == "unanswerable" else "refused_wrongly"
    if q["kind"] == "unanswerable":
        return "wrong"                              # it answered something that has no answer
    ok = content_ok(q["key"], r["answer"]) and citation_ok(q["pages"], r["answer"])
    return "correct" if ok else "wrong"


if __name__ == "__main__":
    print("source:", SOURCE, "| setting:", SETTING)
    retriever = Retriever(SIZE, rerank=RERANK)
    retriever.search = functools.partial(retriever.search, top_k=TOPK)
    t = time.perf_counter()
    retriever.search("warm up")
    print(f"search warm-up took {round((time.perf_counter() - t) * 1000)} ms (not counted in results)")

    questions = load_questions()
    print(len(questions), "questions loaded\n")
    if os.path.exists(REVIEW):
        os.remove(REVIEW)

    review = []
    for i, q in enumerate(questions, start=1):
        started = datetime.now().isoformat(timespec="milliseconds")
        t0 = time.perf_counter()
        try:
            r = answer_question(q["question"], retriever)
        except Exception as e:                      # one bad question must not stop the run
            r = {"answer": f"CRASH: {e}", "outcome": "gave_up", "status": "error",
                 "total_ms": (time.perf_counter() - t0) * 1000, "draft_first_token_ms": None,
                 "retries": 0, "tokens_in": 0, "tokens_out": 0, "sources": [], "steps": {}}
        g = grade(q, r)
        first = r["draft_first_token_ms"]
        append_csv("results.csv", RESULT_COLS, {
            "request_id": f"quality-{SETTING}-{i:02d}", "scenario": SCENARIO, "source": SOURCE,
            "start_time": started,
            "first_token_ms": round(first, 1) if first is not None else "",
            "total_ms": round(r["total_ms"], 1), "status": r["status"],
            "retries": r["retries"], "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"],
            "retrieval_setting": SETTING, "answer_grade": g})
        row = {"id": f"{SETTING}-{i:02d}", "kind": q["kind"], "question": q["question"],
               "key": q["key"], "answer": r["answer"], "auto_grade": g,
               "outcome": r["outcome"], "sources": " ; ".join(r["sources"]), "search_query": r.get("search_query", "")}
        append_csv(REVIEW, REVIEW_COLS, row)        # saved right away
        review.append(row)
        search_ms = round(r.get("steps", {}).get("retrieve_ms", 0))
        print(f"{i:02d} [{q['kind']}] grade={g or 'none'} outcome={r['outcome']} "
              f"tokens={r['tokens_in']}/{r['tokens_out']} total={round(r['total_ms'])} ms search={search_ms} ms")
        # wait as long as the tokens we just used need to refill (8,000 per minute, 20% spare)
        time.sleep(max(5.0, (r["tokens_in"] + r["tokens_out"]) / 8000 * 60 * 1.2))

    print("\nSUMMARY (auto grades, still to be checked by hand)")
    for kind in ("answerable", "unanswerable", "tricky"):
        grades = [x["auto_grade"] or "none" for x in review if x["kind"] == kind]
        print(" ", kind, {g: grades.count(g) for g in sorted(set(grades))})
    print("\nSaved", REVIEW, ". Open it and read every answer.")