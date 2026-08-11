"""
train_w2v.py

Purpose
-------
This script trains a Word2Vec model on the extracted C source
files from the Devign dataset.

The trained model is used to convert code tokens into semantic
vector representations for the GGNN input features.

The model is saved to:

    models/code_w2v.model

This script should be run AFTER:

    splitIntoFiles.py

because it relies on the generated .c source files located in:

    data/intermediate/

Usage
-----

Activate virtual environment first:

    source venv/bin/activate

Then run:

    python scripts/preprocessing/train_w2v.py
"""

import os
import re
from gensim.models import Word2Vec


# =========================================================
# BASE DIRECTORY
# =========================================================
# Compute absolute project root directory so this script
# works regardless of where it is executed from.

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)


# =========================================================
# DATA DIRECTORY
# =========================================================
# Directory containing extracted .c files from both datasets:
#
#   data/intermediate/qemu_code/
#   data/intermediate/ffmpeg_code/

DATA_DIR = os.path.join(
    BASE_DIR,
    "data/intermediate"
)


# =========================================================
# MODEL OUTPUT PATH
# =========================================================

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "code_w2v.model"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# =========================================================
# TOKENIZER REGEX
# =========================================================
# This regex extracts meaningful C tokens:
#
# - identifiers
# - operators
# - numeric constants
#
# Keeping it consistent with build_dataset.py ensures
# embedding compatibility.

TOKEN_RE = re.compile(
    r"""
    [A-Za-z_]\w+     |
    ==|!=|<=|>=      |
    ->               |
    \+\+|--          |
    &&|\|\|          |
    <<|>>            |
    [+\-*/%=<>&|^~!] |
    \d+
    """,
    re.VERBOSE
)


# =========================================================
# TOKENIZATION FUNCTION
# =========================================================

def tokenize_code(code):
    """
    Tokenizes raw C source code into a list of tokens.
    """
    return TOKEN_RE.findall(code)


# =========================================================
# COLLECT TRAINING SENTENCES
# =========================================================
# Word2Vec expects a list of token lists:
#
#   [
#       ["int", "main", "(", ")", "{", ...],
#       ["if", "(", "x", ">", "0", ")", ...],
#       ...
#   ]

print("🔍 Scanning C source files for Word2Vec training...\n")

sentences = []
total_files = 0
total_tokens = 0

for root, dirs, files in os.walk(DATA_DIR):

    for file in files:

        if not file.endswith(".c"):
            continue

        total_files += 1

        path = os.path.join(
            root,
            file
        )

        try:
            with open(
                path,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as f:

                code = f.read()

                tokens = tokenize_code(code)

                if len(tokens) > 0:
                    sentences.append(tokens)
                    total_tokens += len(tokens)

        except Exception as e:
            print(f"⚠ Failed reading {file}")
            print(str(e))

print("✅ Scanning complete.\n")
print(f"📄 Files processed: {total_files}")
print(f"🧠 Total token sequences: {len(sentences)}")
print(f"🔢 Total tokens: {total_tokens}")


# =========================================================
# TRAIN WORD2VEC
# =========================================================
# Key parameters:
#
# vector_size = 100   → matches EMBED_SIZE in build_dataset.py
# window = 5          → context size
# min_count = 2       → ignore very rare tokens
# workers = 4         → parallel training
# sg = 1              → skip-gram (better for code semantics)

print("\n🚀 Training Word2Vec model...\n")

w2v = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1
)

print("✅ Training complete.")


# =========================================================
# SAVE MODEL
# =========================================================

w2v.save(MODEL_PATH)

print("\n💾 Model saved to:")
print(MODEL_PATH)

print("\n🎉 Word2Vec training finished successfully.")
