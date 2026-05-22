# StructBERT

StructBERT is a lightweight project for training a BERT-based masked language model on a custom text corpus. It uses Hugging Face Transformers and Datasets to fine-tune `bert-base-uncased` for masked language modeling.

## Features

- Fine-tunes `bert-base-uncased` for masked language modeling
- Uses Hugging Face `Trainer` for simple training
- Supports custom text corpora
- Saves the trained model for later use

## Project Structure

```text
StructBERT/
├── data/
│   └── mlm/
│       └── dsa_corpus.txt
├── models/
│   └── structbert_mlm/
└── training/
    └── train_mlm.py
```

## Requirements

- Python 3.8+
- `transformers`
- `datasets`
- `torch`

Install dependencies:

```bash
pip install transformers datasets torch
```

## Dataset

The training script expects a plain text file at:

```text
data/mlm/dsa_corpus.txt
```

Each line can contain a sentence, paragraph, or training sample text.

## Training

Run the MLM training script with:

```bash
python training/train_mlm.py
```

### What the script does

1. Loads the tokenizer and model from `bert-base-uncased`
2. Loads the text dataset from `data/mlm/dsa_corpus.txt`
3. Tokenizes each sample with truncation to 128 tokens
4. Applies masked language modeling with a 15% masking probability
5. Trains the model for 3 epochs
6. Saves the final model to:

```text
models/structbert_mlm
```

## Output

After training, you should find the trained model files in:

```text
models/structbert_mlm
```

You can later reload the model with Hugging Face APIs.

## Script Overview

The main training logic is in:

```python
training/train_mlm.py
```

## Notes

- The current script uses a pretrained base BERT model as a starting point.
- If you want to train on a larger corpus, consider adjusting:
  - `max_length`
  - `per_device_train_batch_size`
  - `num_train_epochs`
  - `learning_rate`

## Future Improvements

- Add evaluation and validation splits
- Add support for dynamic dataset paths
- Add configuration via command-line arguments
- Add inference script for testing the trained model
- Add requirements file or environment setup instructions

## License

Add your project license here.
