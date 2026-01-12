# import os
# os.environ['HF_TOKEN'] = 'REDACTED'

# from openai import OpenAI

# client = OpenAI(
#     base_url="https://router.huggingface.co/v1",
#     api_key=os.environ["HF_TOKEN"],
# )

# completion = client.chat.completions.create(
#     model="deepseek-ai/DeepSeek-V3.2",
#     messages=[
#         {
#             "role": "user",
#             "content": "Give me code for AVL tree?"
#         }
#     ],
# )

# print(completion.choices[0].message)
