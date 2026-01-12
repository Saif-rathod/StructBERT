import ast
from pathlib import Path
import re

REPO_PATH = "repos/TheAlgorithms"
OUTPUT_FILE = "data/mlm/corpus.txt"
MIN_LINE_LEN = 10  # lower threshold

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_comments_and_docstrings(path):
    lines = []

    # Read file
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
    except:
        return []

    # 1️⃣ Extract all docstrings
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                doc = ast.get_docstring(node)
                if doc:
                    for l in doc.splitlines():
                        l = normalize(l)
                        if len(l) >= MIN_LINE_LEN:
                            lines.append(l)
    except Exception:
        pass

    # 2️⃣ Extract inline comments anywhere in line
    for l in source.splitlines():
        l = l.strip()
        if "#" in l:
            comment = l.split("#", 1)[1]
            comment = normalize(comment)
            if len(comment) >= MIN_LINE_LEN:
                lines.append(comment)

    return lines

# Walk through all .py files
all_lines = []
for path in Path(REPO_PATH).rglob("*.py"):
    all_lines.extend(extract_comments_and_docstrings(path))

# Save corpus
Path(OUTPUT_FILE).parent.mkdir(exist_ok=True, parents=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for line in all_lines:
        f.write(line + "\n")

print(f"✅ Corpus built at {OUTPUT_FILE} with {len(all_lines)} lines")
