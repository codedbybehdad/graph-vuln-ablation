import os
import re
from gensim.models import Word2Vec
from tqdm import tqdm


def tokenize_c_code(code: str):
    """
    Clean C code and split into tokens.

    - Removes /* */ comments
    - Removes // comments
    - Keeps identifiers, numbers, operators, punctuation
    """

    # Remove multiline comments
    code = re.sub(r'/\*[\s\S]*?\*/', '', code)

    # Remove single-line comments
    code = re.sub(r'//.*', '', code)

    # Tokenize identifiers, numbers, operators
    tokens = re.findall(r'[a-zA-Z_]\w*|\d+|[^\w\s]', code)

    return tokens


def collect_c_files(source_dir):
    """Recursively collect all .c files."""
    c_files = []

    for root, _, files in os.walk(source_dir):
        for f in files:
            if f.endswith(".c"):
                c_files.append(os.path.join(root, f))

    return c_files


def train_codebase_w2v(source_dir, output_path="models/code_w2v.model"):
    print(f"📂 Scanning {source_dir} for C source files...")

    c_files = collect_c_files(source_dir)

    if not c_files:
        print("❌ No .c files found! Check your directory path.")
        return

    print(f"✅ Found {len(c_files)} C files")

    sentences = []

    for file_path in tqdm(c_files, desc="Processing Source Code"):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()

            tokens = tokenize_c_code(code)

            if tokens:
                sentences.append(tokens)

        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")

    print("\n🧠 Training Word2Vec Model...")
    print(f"   Total token sequences: {len(sentences)}")

    model = Word2Vec(
        sentences=sentences,
        vector_size=100,     # Devign uses 100-dim embeddings
        window=5,            # context window
        min_count=1,         # keep rare tokens
        workers=os.cpu_count(),
        sg=1,                # Skip-gram (better for semantic relationships)
        epochs=10
    )

    print(f"✅ Vocabulary size: {len(model.wv)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model.save(output_path)

    print(f"\n✅ Word2Vec model saved to: {output_path}")


if __name__ == "__main__":
    # Directory containing Devign .c files
    SOURCE_DIR = "data/intermediate/devign_code"

    # Output Word2Vec model
    OUTPUT_MODEL = "models/code_w2v.model"

    train_codebase_w2v(SOURCE_DIR, OUTPUT_MODEL)
