import pdfplumber
from pathlib import Path

PDF_PATH = "/home/saif/Desktop/StructBERT/data/sedwick_corpus.pdf"
OUTPUT_FILE = "data/mlm/sedgewick_corpus.txt"

all_lines = []

with pdfplumber.open(PDF_PATH) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            # Split into sentences using '.', '!', '?'
            sentences = [s.strip() for s in text.replace("\n", " ").split('.') if len(s.strip()) > 30]
            all_lines.extend(sentences)

# Save extracted text
Path(OUTPUT_FILE).parent.mkdir(exist_ok=True, parents=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for line in all_lines:
        f.write(line + "\n")

print(f"✅ Extracted {len(all_lines)} lines from PDF to {OUTPUT_FILE}")

