import os
import re
import html
import argparse
from collections import defaultdict

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
# PATHS
# =========================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# joern-export --repr cpg14 writes one or more .dot files under
# per-source-file directories. PDG is exported separately and merged here.
GRAPH_DIR = os.path.join(BASE_DIR, "data", "intermediate", "graphs")
PDG_DIR = os.path.join(BASE_DIR, "data", "intermediate", "pdg")

for required_dir in (GRAPH_DIR, PDG_DIR):
    if not os.path.exists(required_dir):
        raise FileNotFoundError(
            f"\n❌ Required graph directory not found:\n  {required_dir}\n"
            "Run Joern graph extraction first."
        )

LABEL_FILE = os.path.join(BASE_DIR, f"data/intermediate/{DATASET}_labels.csv")
if not os.path.exists(LABEL_FILE):
    raise FileNotFoundError(f"\n❌ Label file not found:\n  {LABEL_FILE}\n")

W2V_MODEL = os.path.join(BASE_DIR, f"models/code_w2v_{DATASET}.model")
if not os.path.exists(W2V_MODEL):
    raise FileNotFoundError(f"\n❌ Word2Vec model not found:\n  {W2V_MODEL}\n")

OUTPUT_DIR = os.path.join(BASE_DIR, f"data/processed/{DATASET}_graphs")
META_FILE = os.path.join(BASE_DIR, f"data/processed/{DATASET}_dataset_index.pt")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# FEATURE SETTINGS
# =========================================================

EMBED_SIZE = 100
NODE_TYPE_EMBED_DIM = 32
MAX_NODE_TYPES = 256
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# =========================================================
# LOAD LABELS / WORD2VEC
# =========================================================

print(f"🚀 Loading labels for dataset: {DATASET}...")
labels = pd.read_csv(LABEL_FILE)
label_map = dict(zip(labels.id.astype(int), labels.target.astype(int)))

print("🧠 Loading Word2Vec...")
w2v = Word2Vec.load(W2V_MODEL)

# =========================================================
# GRAPHVIZ PARSING
# =========================================================

NODE_ID_RE = re.compile(r'^\s*"?(\d+)"?\s*\[')
EDGE_RE = re.compile(
    r'^\s*"?(\d+)"?\s*->\s*"?(\d+)"?\s*\[(.*?)\]\s*;?\s*$'
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
    re.VERBOSE,
)


def source_id_from_path(path, root):
    """Recover the numeric sample ID from Joern's nested export path."""
    rel_parts = os.path.relpath(path, root).split(os.sep)

    for part in rel_parts[:-1]:
        match = re.fullmatch(r"(\d+)\.c", part)
        if match:
            return int(match.group(1))

    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.fullmatch(r"(\d+)(?:\.c)?", stem)
    if match:
        return int(match.group(1))

    # Also support legacy flat names such as 123-main.dot.
    match = re.match(r"(\d+)(?:[-_.].*)?$", stem)
    if match:
        return int(match.group(1))

    return None


def extract_attribute(text, attribute):
    """Extract a Graphviz attribute value, supporting quoted/escaped values."""
    match = re.search(rf"\b{re.escape(attribute)}\s*=\s*", text)
    if not match:
        return ""

    i = match.end()
    if i >= len(text):
        return ""

    if text[i] != '"':
        end = i
        while end < len(text) and text[end] not in " ]":
            end += 1
        return text[i:end]

    i += 1
    chars = []
    escaped = False

    while i < len(text):
        ch = text[i]
        if escaped:
            # Keep common escaped characters readable.
            if ch == 'n':
                chars.append("\n")
            elif ch == 'r':
                chars.append("\r")
            elif ch == 't':
                chars.append("\t")
            else:
                chars.append(ch)
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            break
        else:
            chars.append(ch)
        i += 1

    return "".join(chars)


def parse_node_line(line):
    match = NODE_ID_RE.match(line)
    if not match:
        return None

    # Ignore edge/graph declarations that happen to contain brackets.
    if "->" in line.split("[", 1)[0]:
        return None

    node_id = int(match.group(1))

    # Support the legacy HTML-label format used by older Joern/custom exports.
    html_label = re.search(r"label\s*=\s*<\s*([^,<\s]+).*?<BR/>(.*?)>\s*\]\s*;?\s*$", line, re.IGNORECASE)
    if html_label:
        node_type = html_label.group(1).strip()
        code = html.unescape(html_label.group(2)).strip()
        code = re.sub(r"<.*?>", " ", code)
        return node_id, node_type, code

    label_text = extract_attribute(line, "label")
    if not label_text:
        return node_id, "UNKNOWN", ""

    node_type = label_text.strip().split()[0] if label_text.strip() else "UNKNOWN"
    code = extract_attribute(label_text, "CODE")
    code = html.unescape(code)
    code = re.sub(r"<.*?>", " ", code).strip()
    return node_id, node_type, code


def parse_edge_line(line, forced_type=None):
    match = EDGE_RE.match(line)
    if not match:
        return None

    src = int(match.group(1))
    dst = int(match.group(2))
    attrs = match.group(3)

    if forced_type is not None:
        return src, dst, forced_type

    label = extract_attribute(attrs, "label").upper()

    # cpg14 contains many CPG edge labels. For this experiment we retain
    # exactly AST and CFG from cpg14; PDG edges are read separately.
    if label.startswith("AST"):
        return src, dst, 0
    if label.startswith("CFG"):
        return src, dst, 1

    return None


def collect_dot_files(root):
    files = []
    for current_root, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".dot"):
                files.append(os.path.join(current_root, filename))
    return sorted(files)


def load_graph_exports(root, forced_type=None):
    """Group Joern dot files by source sample ID and merge their nodes/edges."""
    grouped = defaultdict(lambda: {"nodes": {}, "edges": set()})
    dot_files = collect_dot_files(root)

    for path in dot_files:
        source_id = source_id_from_path(path, root)
        if source_id is None:
            continue

        graph = grouped[source_id]

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                node = parse_node_line(line)
                if node is not None:
                    nid, ntype, code = node
                    # Prefer a populated node record over UNKNOWN/empty data.
                    old = graph["nodes"].get(nid)
                    if old is None or (old[0] == "UNKNOWN" and ntype != "UNKNOWN") or (not old[1] and code):
                        graph["nodes"][nid] = (ntype, code)
                    continue

                edge = parse_edge_line(line, forced_type=forced_type)
                if edge is not None:
                    graph["edges"].add(edge)

    return grouped, len(dot_files)


# =========================================================
# TOKEN / FEATURE BUILDING
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


VULN_KEYWORDS = {
    "strcpy", "memcpy", "malloc", "free", "gets", "scanf",
    "sprintf", "strcat", "realloc", "memset"
}
ARITH_OPS = {"+", "-", "*", "/", "%", "++", "--"}
COMPARE_OPS = {"==", "!=", "<", ">", "<=", ">="}


def build_handcrafted_features(tokens, code):
    return np.array([
        min(len(tokens) / 50.0, 1.0),
        min(sum(t in VULN_KEYWORDS for t in tokens) / 5.0, 1.0),
        min(sum(t in ARITH_OPS for t in tokens) / 10.0, 1.0),
        min(sum(t in COMPARE_OPS for t in tokens) / 10.0, 1.0),
        float("*" in code or "->" in code),
        float("[" in code and "]" in code),
    ], dtype=np.float32)


# =========================================================
# LOAD GRAPH EXPORTS
# =========================================================

print("🔍 Collecting CPG14 (AST + CFG) graph exports...")
cpg_groups, cpg_dot_count = load_graph_exports(GRAPH_DIR)

print("🔍 Collecting PDG graph exports...")
pdg_groups, pdg_dot_count = load_graph_exports(PDG_DIR, forced_type=2)

all_source_ids = sorted(set(cpg_groups) | set(pdg_groups))

if not all_source_ids:
    raise RuntimeError(
        f"\n❌ No usable Joern .dot exports were found in:\n"
        f"  {GRAPH_DIR}\n  {PDG_DIR}\n"
    )

print(f"📦 CPG14 .dot files found: {cpg_dot_count}")
print(f"📦 PDG .dot files found: {pdg_dot_count}")
print(f"📦 Source samples discovered: {len(all_source_ids)}")

# =========================================================
# COLLECT NODE TYPES
# =========================================================

print("🔍 Collecting node types...")
node_types_set = set()

for source_id in all_source_ids:
    for node_type, _ in cpg_groups.get(source_id, {}).get("nodes", {}).values():
        node_types_set.add(node_type)
    for node_type, _ in pdg_groups.get(source_id, {}).get("nodes", {}).values():
        node_types_set.add(node_type)

node_types = sorted(node_types_set)[:MAX_NODE_TYPES]
node_type_map = {t: i for i, t in enumerate(node_types)}
NUM_NODE_TYPES = len(node_type_map)

print(f"📦 Node types kept: {NUM_NODE_TYPES}")

node_type_embeddings = np.random.normal(
    0.0, 0.1, (NUM_NODE_TYPES, NODE_TYPE_EMBED_DIM)
).astype(np.float32)

# =========================================================
# BUILD DATASET
# =========================================================

total_nodes = 0
zero_embed_nodes = 0
saved_graphs = 0
failed_graphs = 0
skipped_no_edges = 0
dataset_index = []
edge_type_counts = {0: 0, 1: 0, 2: 0}
graph_type_presence = {0: 0, 1: 0, 2: 0}

print("\n🏗️ Building dataset...\n")

for idx, file_id in enumerate(all_source_ids):

    if idx % 250 == 0:
        print(f"📊 [{idx}/{len(all_source_ids)}] saved={saved_graphs} failed={failed_graphs}")

    try:
        if file_id not in label_map:
            continue

        label = int(label_map[file_id])

        cpg_graph = cpg_groups.get(file_id, {"nodes": {}, "edges": set()})
        pdg_graph = pdg_groups.get(file_id, {"nodes": {}, "edges": set()})

        node_info = dict(cpg_graph["nodes"])
        node_info.update({
            nid: value
            for nid, value in pdg_graph["nodes"].items()
            if nid not in node_info or node_info[nid][0] == "UNKNOWN"
        })

        edges = set(cpg_graph["edges"]) | set(pdg_graph["edges"])

        if not edges:
            skipped_no_edges += 1
            continue

        used_nodes = sorted({n for edge in edges for n in edge})
        node_map = {n: i for i, n in enumerate(used_nodes)}

        x = []

        for nid in used_nodes:
            total_nodes += 1

            ntype, code = node_info.get(nid, ("UNKNOWN", ""))
            tokens = tokenize_code(code)

            emb = build_embedding(tokens)
            if np.abs(emb).sum() == 0:
                zero_embed_nodes += 1

            type_idx = node_type_map.get(ntype, 0)
            type_emb = node_type_embeddings[type_idx]
            handcrafted = build_handcrafted_features(tokens, code)

            feature = np.concatenate([emb, type_emb, handcrafted]).astype(np.float32)
            x.append(feature)

        remapped_edges = []
        valid_edge_types = []

        for src, dst, edge_type in sorted(edges):
            if src not in node_map or dst not in node_map:
                continue

            remapped_edges.append([node_map[src], node_map[dst]])
            valid_edge_types.append(edge_type)
            edge_type_counts[edge_type] += 1

        if not remapped_edges:
            skipped_no_edges += 1
            continue

        edge_index = torch.tensor(remapped_edges, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(valid_edge_types, dtype=torch.long)

        for edge_type_id in (0, 1, 2):
            if edge_type_id in valid_edge_types:
                graph_type_presence[edge_type_id] += 1

        data = Data(
            x=torch.tensor(np.asarray(x), dtype=torch.float32),
            edge_index=edge_index,
            edge_type=edge_type,
            y=torch.tensor([label], dtype=torch.long),
        )

        graph_path = os.path.join(OUTPUT_DIR, f"{file_id}.pt")
        torch.save(data, graph_path)

        dataset_index.append({
            "file_id": file_id,
            "path": graph_path,
            "label": label,
        })

        saved_graphs += 1

    except Exception as exc:
        failed_graphs += 1
        print(f"\n❌ Failed on source ID {file_id}")
        print(str(exc))

FINAL_FEATURE_DIM = EMBED_SIZE + NODE_TYPE_EMBED_DIM + 6
coverage = 0.0 if total_nodes == 0 else 100.0 * (total_nodes - zero_embed_nodes) / total_nodes

print(f"🧠 Word2Vec node coverage: {coverage:.2f}% ({total_nodes - zero_embed_nodes}/{total_nodes})")
print("\n📌 Edge-type extraction summary:")
print(f"   AST edges: {edge_type_counts[0]}")
print(f"   CFG edges: {edge_type_counts[1]}")
print(f"   PDG edges: {edge_type_counts[2]}")
print(f"   Graphs containing AST: {graph_type_presence[0]}")
print(f"   Graphs containing CFG: {graph_type_presence[1]}")
print(f"   Graphs containing PDG: {graph_type_presence[2]}")

if edge_type_counts[0] == 0:
    raise RuntimeError("❌ No AST edges were extracted. Aborting to avoid a misleading experiment.")
if edge_type_counts[1] == 0:
    raise RuntimeError("❌ No CFG edges were extracted. Aborting to avoid a misleading experiment.")
if edge_type_counts[2] == 0:
    raise RuntimeError("❌ No PDG edges were extracted. Aborting to avoid a misleading experiment.")

if saved_graphs == 0:
    raise RuntimeError("❌ No graphs were saved.")

print("\n💾 Saving dataset index...")
torch.save(
    {
        "dataset": DATASET,
        "graphs": dataset_index,
        "num_node_types": NUM_NODE_TYPES,
        "feature_dim": FINAL_FEATURE_DIM,
    },
    META_FILE,
)

print(f"\n✅ Graphs saved: {saved_graphs}")
print(f"❌ Graphs failed: {failed_graphs}")
print(f"⚠️ Graphs skipped (no selected edges): {skipped_no_edges}")
print(f"🧩 Total nodes: {total_nodes}")
print(f"🧠 Final feature dim: {FINAL_FEATURE_DIM}")
print("\n🎉 Done.")
