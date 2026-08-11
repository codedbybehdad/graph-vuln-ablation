import os
import re
import html
import argparse
import numpy as np
import pandas as pd
import torch

from gensim.models import Word2Vec
from torch_geometric.data import Data

# =========================================================
# ARGUMENTS
# =========================================================

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, choices=["qemu", "ffmpeg"])
args = parser.parse_args()

DATASET = args.dataset

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

# Joern export directory (AST + CFG graphs)
GRAPH_DIR = os.path.join(BASE_DIR, "data/intermediate/graphs")

if not os.path.exists(GRAPH_DIR):
    raise FileNotFoundError(
        f"\n❌ Graph directory not found:\n  {GRAPH_DIR}\n"
        "Run Joern graph export first."
    )

LABEL_FILE = os.path.join(
    BASE_DIR,
    f"data/intermediate/{DATASET}_labels.csv"
)

if not os.path.exists(LABEL_FILE):
    raise FileNotFoundError(
        f"\n❌ Label file not found:\n  {LABEL_FILE}\n"
    )

W2V_MODEL = os.path.join(BASE_DIR, "models/code_w2v.model")

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    f"data/processed/{DATASET}_graphs"
)

META_FILE = os.path.join(
    BASE_DIR,
    f"data/processed/{DATASET}_dataset_index.pt"
)


# =========================================================
# FEATURE SETTINGS
# =========================================================

EMBED_SIZE = 100
NODE_TYPE_EMBED_DIM = 32
MAX_NODE_TYPES = 256
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# =========================================================
# LOAD LABELS
# =========================================================

print(f"🚀 Loading labels for dataset: {DATASET}...")
labels = pd.read_csv(LABEL_FILE)
label_map = dict(zip(labels.id, labels.target))

# =========================================================
# LOAD WORD2VEC
# =========================================================

print("🧠 Loading Word2Vec...")
w2v = Word2Vec.load(W2V_MODEL)

# =========================================================
# REGEX
# =========================================================

NODE_RE = re.compile(
    r'"(\d+)"\s+\[label\s*=\s*<(.*?),\s*\d+<BR/>(.*?)>\s*\]'
)

EDGE_RE = re.compile(
    r'"(\d+)"\s*->\s*"(\d+)"\s*\[\s*label\s*=\s*"(AST|CFG|DFG):'
)

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
# OUTPUT DIR
# =========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# FIND DOT FILES RECURSIVELY
# =========================================================

print("🔍 Collecting graph files...")

graph_files = []

for root, _, files in os.walk(GRAPH_DIR):
    for f in files:
        if f.endswith(".dot"):
            graph_files.append(os.path.join(root, f))

graph_files = sorted(graph_files)

if len(graph_files) == 0:
    raise RuntimeError(
        f"\n❌ No .dot files found in:\n  {GRAPH_DIR}\n"
    )

print(f"📦 Found {len(graph_files)} graph files")

# =========================================================
# COLLECT NODE TYPES
# =========================================================

print("🔍 Collecting node types...")

node_types_set = set()

for idx, path in enumerate(graph_files):

    if idx % 1000 == 0:
        print(f"📦 Scanning node types: {idx}")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = NODE_RE.search(line)
            if m:
                node_types_set.add(m.group(2).strip())

node_types = sorted(list(node_types_set))[:MAX_NODE_TYPES]
node_type_map = {t: i for i, t in enumerate(node_types)}

NUM_NODE_TYPES = len(node_type_map)

print(f"📦 Node types kept: {NUM_NODE_TYPES}")

# =========================================================
# NODE TYPE EMBEDDINGS
# =========================================================

print("🎲 Creating dense node type embeddings...")

node_type_embeddings = np.random.normal(
    0.0,
    0.1,
    (NUM_NODE_TYPES, NODE_TYPE_EMBED_DIM)
).astype(np.float32)

# =========================================================
# TOKENIZATION
# =========================================================

def tokenize_code(code):
    code = html.unescape(code)
    code = re.sub(r"<.*?>", " ", code)
    return TOKEN_RE.findall(code)

def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm

def build_embedding(tokens):
    vectors = [w2v.wv[t] for t in tokens if t in w2v.wv]
    if not vectors:
        return np.zeros(EMBED_SIZE, dtype=np.float32)
    emb = np.mean(vectors, axis=0).astype(np.float32)
    return normalize_vector(emb)

VULN_KEYWORDS = {"strcpy","memcpy","malloc","free","gets","scanf","sprintf","strcat","realloc","memset"}
ARITH_OPS = {"+","-","*","/","%","++","--"}
COMPARE_OPS = {"==","!=","<",">","<=",">="}

def build_handcrafted_features(tokens,code):
    return np.array([
        min(len(tokens)/50.0,1.0),
        min(sum(t in VULN_KEYWORDS for t in tokens)/5.0,1.0),
        min(sum(t in ARITH_OPS for t in tokens)/10.0,1.0),
        min(sum(t in COMPARE_OPS for t in tokens)/10.0,1.0),
        float("*" in code or "->" in code),
        float("[" in code and "]" in code)
    ], dtype=np.float32)

# =========================================================
# BUILD DATASET
# =========================================================

total_nodes = 0
zero_embed_nodes = 0
saved_graphs = 0
failed_graphs = 0
dataset_index = []
feature_variances = []

print("\n🏗️ Building dataset...\n")

for idx, path in enumerate(graph_files):

    if idx % 250 == 0:
        print(f"📊 [{idx}/{len(graph_files)}] saved={saved_graphs} failed={failed_graphs}")

    try:
        fname = os.path.splitext(os.path.basename(path))[0]
        file_id = fname.split("-")[0]

        if not file_id.isdigit():
            continue

        file_id = int(file_id)

        if file_id not in label_map:
            continue

        label = int(label_map[file_id])

        node_info = {}
        edges = []
        edge_types = []

        with open(path,"r",encoding="utf-8",errors="ignore") as f:
            for line in f:

                nm = NODE_RE.search(line)
                if nm:
                    nid = int(nm.group(1))
                    ntype = nm.group(2).strip()
                    code = html.unescape(nm.group(3))
                    code = re.sub(r"<.*?>"," ",code)
                    node_info[nid] = (ntype,code)
                    continue

                em = EDGE_RE.search(line)
                if em:
                    src = int(em.group(1))
                    dst = int(em.group(2))
                    etype = em.group(3)
                    edges.append((src,dst))

                    if etype == "AST":
                        edge_types.append(0)
                    elif etype == "CFG":
                        edge_types.append(1)
                    else:
                        edge_types.append(2)

        if len(edges) == 0:
            continue

        used_nodes = sorted({n for e in edges for n in e})
        node_map = {n:i for i,n in enumerate(used_nodes)}

        x = []

        for nid in used_nodes:
            total_nodes += 1

            ntype,code = node_info.get(nid,("UNKNOWN",""))
            tokens = tokenize_code(code)

            emb = build_embedding(tokens)

            if np.abs(emb).sum() == 0:
                zero_embed_nodes += 1

            type_idx = node_type_map.get(ntype,0)
            type_emb = node_type_embeddings[type_idx]

            handcrafted = build_handcrafted_features(tokens,code)

            feature = normalize_vector(
                np.concatenate([emb,type_emb,handcrafted])
            )

            x.append(feature)

        if len(x) == 0:
            continue

        x_np = np.array(x,dtype=np.float32)
        feature_variances.append(np.var(x_np))
        x = torch.tensor(x_np,dtype=torch.float)

        remapped_edges = []
        valid_edge_types = []

        for i,(s,t) in enumerate(edges):
            if s not in node_map or t not in node_map:
                continue
            remapped_edges.append([node_map[s],node_map[t]])
            valid_edge_types.append(edge_types[i])

        if len(remapped_edges) == 0:
            continue

        edge_index = torch.tensor(remapped_edges,dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(valid_edge_types,dtype=torch.long)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            y=torch.tensor([label],dtype=torch.long)
        )

        graph_path = os.path.join(OUTPUT_DIR,f"{file_id}.pt")
        torch.save(data,graph_path)

        dataset_index.append({
            "file_id":file_id,
            "path":graph_path,
            "label":label
        })

        saved_graphs += 1

    except Exception as e:
        failed_graphs += 1
        print(f"\n❌ Failed on: {path}")
        print(str(e))

FINAL_FEATURE_DIM = EMBED_SIZE + NODE_TYPE_EMBED_DIM + 6

print("\n💾 Saving dataset index...")

torch.save(
    {
        "dataset":DATASET,
        "graphs":dataset_index,
        "num_node_types":NUM_NODE_TYPES,
        "feature_dim":FINAL_FEATURE_DIM
    },
    META_FILE
)

print("\n✅ Graphs saved:", saved_graphs)
print("❌ Graphs failed:", failed_graphs)
print("🧩 Total nodes:", total_nodes)
print("🧠 Final feature dim:", FINAL_FEATURE_DIM)

print("\n🎉 Done.")
