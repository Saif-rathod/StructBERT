from transformers import (
    BertTokenizerFast,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments
)
from datasets import load_dataset

MODEL = "bert-base-uncased"

tokenizer = BertTokenizerFast.from_pretrained(MODEL)
model = BertForMaskedLM.from_pretrained(MODEL)

dataset = load_dataset(
    "text",
    data_files={"train": "data/mlm/dsa_corpus.txt"}
)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

args = TrainingArguments(
    output_dir="models/structbert_mlm",
    per_device_train_batch_size=32,
    num_train_epochs=3,
    save_steps=1000,
    learning_rate=2e-5,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    data_collator=collator,
)

trainer.train()
trainer.save_model("models/structbert_mlm")
