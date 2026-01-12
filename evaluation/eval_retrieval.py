import json, faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("models/structbert")
index = faiss.read_index("retrieval/index.faiss")

corpus = []
for l in open("data/contrastive/pairs.jsonl"):
    corpus.append(json.loads(l)["positive"])

def recall_at_k(ranks, k):
    return any(r < k for r in ranks)

def mrr(ranks):
    return min([1/(r+1) for r in ranks], default=0)

def evaluate(k=5):
    recalls, mrrs = [], []

    for l in open("evaluation/retrieval_eval.jsonl"):
        ex = json.loads(l)
        q_emb = model.encode([ex["query"]], normalize_embeddings=True)
        _, ids = index.search(q_emb, 20)

        ranks = []
        for gold in ex["gold"]:
            for r, idx in enumerate(ids[0]):
                if gold.lower() in corpus[idx].lower():
                    ranks.append(r)

        recalls.append(recall_at_k(ranks, k))
        mrrs.append(mrr(ranks))

    print(f"Recall@{k}: {np.mean(recalls):.3f}")
    print(f"MRR: {np.mean(mrrs):.3f}")

evaluate()
