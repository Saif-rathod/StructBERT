import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("models/structbert")
index = faiss.read_index("retrieval/index.faiss")

def retrieve(query, k=3):
    q = model.encode([query], normalize_embeddings=True)
    _, ids = index.search(q, k)
    return ids[0]
