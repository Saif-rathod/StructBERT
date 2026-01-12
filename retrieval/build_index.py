import faiss, json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("models/structbert")

texts = []
for l in open("data/contrastive/pairs.jsonl"):
    j = json.loads(l)
    texts.append(j["positive"])

embeddings = model.encode(texts, normalize_embeddings=True)

index = faiss.IndexFlatIP(768)
index.add(embeddings)

faiss.write_index(index, "retrieval/index.faiss")
