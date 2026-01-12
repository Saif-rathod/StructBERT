import json, torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, losses

class ContrastiveDataset(Dataset):
    def __init__(self, path):
        self.data = [json.loads(l) for l in open(path)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]["anchor"], self.data[idx]["positive"]

model = SentenceTransformer(
    "models/structbert_mlm",
    device="cuda"
)

dataset = ContrastiveDataset("data/contrastive/pairs.jsonl")
loader = DataLoader(dataset, shuffle=True, batch_size=32)

loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(loader, loss)],
    epochs=3,
    warmup_steps=100,
    output_path="models/structbert"
)
