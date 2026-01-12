from retrieval.retrieve import retrieve
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"]
)

def answer(query, context):
    prompt = f"""
Use the following algorithmic context:

{context}

Question:
{query}
"""
    res = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3.2",
        messages=[{"role": "user", "content": prompt}],
    )
    return res.choices[0].message.content
