import os, json

from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

PROMPTS = [
    "Explain AVL Tree",
    "AVL Tree rotations",
    "Difference between AVL Tree and Red Black Tree",
    "Time complexity of AVL Tree operations",
]

out = []

for p in PROMPTS:
    res = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3.2",
        messages=[{"role": "user", "content": p}],
    )
    out.append({
        "prompt": p,
        "answer": res.choices[0].message.content
    })

with open("data/raw/deepseek_outputs.json", "w") as f:
    json.dump(out, f, indent=2)
