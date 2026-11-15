import json
import re

import numpy as np
from rank_bm25 import BM25Okapi

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _embedder


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def normalize(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


class Retriever:
    def __init__(self, chunk_size, rerank=False):
        with open(f"chunks_{chunk_size}.json", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.rerank = rerank
        self.vectors = None
        if rerank:
            path = f"vectors_{chunk_size}.npy"
            try:
                self.vectors = np.load(path)
            except FileNotFoundError:
                print(f"building meaning-vectors for {len(self.chunks)} pieces (one time only)...")
                emb = get_embedder()
                self.vectors = normalize(np.array(list(emb.embed([c["text"] for c in self.chunks]))))
                np.save(path, self.vectors)

    def search(self, question, top_k=4):
        kw = np.array(self.bm25.get_scores(tokenize(question)))
        kw_order = np.argsort(-kw)
        if not self.rerank:
            return [dict(self.chunks[i], score=float(kw[i])) for i in kw_order[:top_k]]

        q = normalize(np.array(list(get_embedder().query_embed(question)))[0])
        sim = self.vectors @ q
        sim_order = np.argsort(-sim)

        fused = {}                                  # reciprocal rank fusion
        for rank, i in enumerate(kw_order[:30]):
            fused[int(i)] = fused.get(int(i), 0) + 1 / (60 + rank)
        for rank, i in enumerate(sim_order[:30]):
            fused[int(i)] = fused.get(int(i), 0) + 1 / (60 + rank)
        best = sorted(fused, key=fused.get, reverse=True)[:top_k]
        # "score" stays the keyword score, so the pipeline's weak-result check works as before
        return [dict(self.chunks[i], score=float(kw[i]), similarity=float(sim[i])) for i in best]


if __name__ == "__main__":
    r = Retriever(250, rerank=True)
    for hit in r.search("How many sound slices per second does Moshi's audio compressor produce?", top_k=3):
        print(f"[{hit['paper']} p.{hit['page']} | {hit['section']}]", hit["text"][:120], "...")