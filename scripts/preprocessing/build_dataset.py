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
# PATHS
# =========================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
GRAPH_DIR = os.path.join(BASE_DIR, "data/intermediate/graphs")
PDG_DIR = os.path.join(BASE_DIR, "data/intermediate/pdg")
LABEL_FILE = os.path.join(BASE_DIR, f"data/intermediate/{DATASET}_labels.csv")
W2V_MODEL = os.path.join(BASE_DIR, f"models/code_w2v_{DATASET}.model")
OUTPUT_DIR = os.path.join(BASE_DIR, f"data/processed/{DATASET}_graphs")
META_FILE = os.path.join(BASE_DIR, f"data/processed/{DATASET}_dataset_index.pt")

if not os.path.exists(GRAPH_DIR):
    raise FileNotFoundError(
        f"\n❌ Graph directory not found:\n  {GRAPH_DIR}\n"
        "Run Joern graph export first."
    )
if not os.path.exists(PDG_DIR):
    raise FileNotFoundError(
        f"\n❌ PDG directory not found:\n  {PDG_DIR}\n"
        "Run Joern PDG export first."
    )
if not os.path.exists(LABEL_FILE):
    raise FileNotFoundError(f"\n❌ Label file not found:\n  {LABEL_FILE}\n")
if not os.path.exists(W2V_MODEL):
    raise FileNotFoundError(f"\n❌ Word2Vec model not found:\n  {W2V_MODEL}\n")

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
# PARSERS
# =========================================================

# Old/current cpg14 style emitted by Joern.
NODE_RE_HTML = re.compile(
    r'^\s*"(\d+)"\s+\[label\s*=\s*<(.*?),\s*\d+<BR/>(.*?)>\s*\]'
)

# Newer quoted-label style emitted by Joern versions where DOT labels
# contain properties such as CODE="...".
NODE_RE_QUOTED = re.compile(r'^\s*"(\d+)"\s*\[label\s*=\s*"(.*?)"\s*\]')

EDGE_RE = re.compile(
    r'"(\d+)"\s*->\s*"(\d+)"\s*\[([^\]]*)\]'
)

TOKEN_RE = re.compile(
    r"""
    [A-Za-z_]\w+     |
    ==|!=|<=|>=      |
    ->               |
    \+\+|--         |
    &&|\|\|         |
    <<|>>            |
    [+\-*/%=<>&|^~!] |
    \d+
    """,
    re.VERBOSE,
)


def extract_file_id(path):
    """Extract the numeric sample id from Joern export paths."""
    candidates = [os.path.splitext(os.path.basename(path))[0]]
    candidates.extend(reversed(os.path.normpath(path).split(os.sep)))
    for value in candidates:
        match = re.match(r"^(\d+)(?:[-_.].*)?$", value)
        if match:
            return int(match.group(1))
    return None


def unescape_dot_text(value):
    value = value.replace(r'\"', '"')
    value = value.replace(r'\\', '\\')
    value = value.replace(r'\n', '\n')
    return html.unescape(value)


def parse_node_line(line):
    match = NODE_RE_HTML.search(line)
    if match:
        return int(match.group(1)), match.group(2).strip(), unescape_dot_text(match.group(3))

    match = NODE_RE_QUOTED.search(line)
    if not match:
        return None

    node_id = int(match.group(1))
    payload = unescape_dot_text(match.group(2))

    # Try the most useful CPG properties first.
    code_match = re.search(r'CODE\s*=\s*"((?:\\.|[^"\\])*)"', payload)
    code = unescape_dot_text(code_match.group(1)) if code_match else ""

    # Joern labels normally start with the node type, followed by properties.
    node_type = payload.split()[0].strip() if payload.strip() else "UNKNOWN"
    return node_id, node_type, code


def classify_edge_label(attributes):
    label_match = re.search(r'label\s*=\s*(?:"([^"]*)"|<([^>]*)>)', attributes)
    label = (label_match.group(1) if label_match and label_match.group(1) is not None else
             label_match.group(2) if label_match else attributes)
    label_upper = label.upper()

    if "AST" in label_upper:
        return 0
    if "CFG" in label_upper or "FLOWS_TO" in label_upper or "CONTROLS" in label_upper:
        return 1
    if any(token in label_upper for token in ("DFG", "DDG", "CDG", "REACHING_DEF", "REACHES")):
        return 2
    return None


def parse_dot(path, force_pdg=False):
    node_info = {}
    edges = []
    edge_types = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            node = parse_node_line(line)
            if node is not None:
                nid, ntype, code = node
                node_info[nid] = (ntype, code)
                continue

            match = EDGE_RE.search(line)
            if not match:
                continue

            src = int(match.group(1))
            dst = int(match.group(2))
            etype = 2 if force_pdg else classify_edge_label(match.group(3))
            if etype is None:
                continue

            edges.append((src, dst))
            edge_types.append(etype)

    return node_info, edges, edge_types


def collect_graph_files(directory):
    by_id = {}
    for root, _, files in os.walk(directory):
        for filename in files:
            if not filename.endswith(".dot"):
                continue
            path = os.path.join(root, filename)
            file_id = extract_file_id(path)
            if file_id is not None:
                by_id.setdefault(file_id, []).append(path)
    return by_id


# =========================================================
# COLLECT GRAPH EXPORTS
# =========================================================

print("🔍 Collecting AST/CFG graph files...")
graph_files = collect_graph_files(GRAPH_DIR)
print(f"📦 AST/CFG sample ids: {len(graph_files)}")

print("🔍 Collecting PDG graph files...")
pdg_files = collect_graph_files(PDG_DIR)
print(f"📦 PDG sample ids: {len(pdg_files)}")

if not graph_files:
    raise RuntimeError(f"\n❌ No graph DOT files found in:\n  {GRAPH_DIR}\n")
if not pdg_files:
    raise RuntimeError(f"\n❌ No PDG DOT files found in:\n  {PDG_DIR}\n")

# =========================================================
# NODE TYPE EMBEDDINGS
# =========================================================

print("🔍 Collecting node types...")
node_types_set = set()

for idx, file_id in enumerate(sorted(graph_files)):
    if idx % 1000 == 0:
        print(f"📦 Scanning node types: {idx}")
    for path in graph_files[file_id]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                node = parse_node_line(line)
                if node is not None:
                    node_types_set.add(node[1])

# Include node types only from AST/CFG exports because they define the shared
# node universe used by the feature representation.
node_types = sorted(node_types_set)[:MAX_NODE_TYPES]
node_type_map = {t: i for i, t in enumerate(node_types)}
NUM_NODE_TYPES = len(node_type_map)
print(f"📦 Node types kept: {NUM_NODE_TYPES}")

print("🎲 Creating dense node type embeddings...")
rng = np.random.default_rng(RANDOM_SEED)
node_type_embeddings = rng.normal(
    0.0, 0.1, (max(NUM_NODE_TYPES, 1), NODE_TYPE_EMBED_DIM)
).astype(np.float32)

# =========================================================
# TOKEN / FEATURE FUNCTIONS
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
    return normalize_vector(np.mean(vectors, axis=0).astype(np.float32))


VULN_KEYWORDS = {
    "strcpy", "memcpy", "malloc", "free", "gets", "scanf",
    "sprintf", "strcat", "realloc", "memset",
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
# BUILD DATASET
# =========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_IDS = set(label_map.keys())
dataset_index = []
total_nodes = 0
zero_embed_nodes = 0
saved_graphs = 0
failed_graphs = 0
edge_counts = {0: 0, 1: 0, 2: 0}
missing_pdg = 0

print("\n🏗️ Building relation-aware dataset...\n")

for idx, file_id in enumerate(sorted(graph_files)):
    if idx % 250 == 0:
        print(
            f"📊 [{idx}/{len(graph_files)}] saved={saved_graphs} "
            f"failed={failed_graphs}"
        )

    if file_id not in LABEL_IDS:
        continue

    try:
        # A sample can have more than one export file in future Joern layouts;
        # merge them deterministically by sample id.
        node_info = {}
        edges = []
        edge_types = []

        for path in sorted(graph_files[file_id]):
            nodes, parsed_edges, parsed_types = parse_dot(path, force_pdg=False)
            node_info.update(nodes)
            edges.extend(parsed_edges)
            edge_types.extend(parsed_types)

        if file_id in pdg_files:
            for path in sorted(pdg_files[file_id]):
                pdg_nodes, pdg_edges, pdg_types = parse_dot(path, force_pdg=True)
                # Do not replace richer node information from cpg14 when it exists.
                for nid, value in pdg_nodes.items():
                    if nid not in node_info or not node_info[nid][1]:
                        node_info[nid] = value
                edges.extend(pdg_edges)
                edge_types.extend(pdg_types)
        else:
            missing_pdg += 1

        if not edges or not node_info:
            continue

        # Preserve the complete union of nodes appearing in AST/CFG/PDG edges.
        used_nodes = sorted({n for edge in edges for n in edge})
        node_map = {nid: i for i, nid in enumerate(used_nodes)}

        x = []
        for nid in used_nodes:
            total_nodes += 1
            ntype, code = node_info.get(nid, ("UNKNOWN", ""))
            tokens = tokenize_code(code)
            emb = build_embedding(tokens)
            if not np.any(emb):
                zero_embed_nodes += 1

            type_idx = node_type_map.get(ntype, 0)
            type_emb = node_type_embeddings[type_idx]
            handcrafted = build_handcrafted_features(tokens, code)
            x.append(np.concatenate([emb, type_emb, handcrafted]).astype(np.float32))

        remapped_edges = []
        valid_edge_types = []
        seen_edges = set()

        for (src, dst), etype in zip(edges, edge_types):
            if src not in node_map or dst not in node_map:
                continue
            key = (node_map[src], node_map[dst], int(etype))
            if key in seen_edges:
                continue
            seen_edges.add(key)
            remapped_edges.append([node_map[src], node_map[dst]])
            valid_edge_types.append(int(etype))
            edge_counts[int(etype)] += 1

        if not remapped_edges:
            continue

        data = Data(
            x=torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32),
            edge_index=torch.tensor(remapped_edges, dtype=torch.long).t().contiguous(),
            edge_type=torch.tensor(valid_edge_types, dtype=torch.long),
            y=torch.tensor([label_map[file_id]], dtype=torch.long),
        )

        graph_path = os.path.join(OUTPUT_DIR, f"{file_id}.pt")
        torch.save(data, graph_path)
        dataset_index.append({
            "file_id": int(file_id),
            "path": graph_path,
            "label": int(label_map[file_id]),
        })
        saved_graphs += 1

    except Exception as exc:
        failed_graphs += 1
        print(f"\n❌ Failed on sample id: {file_id}")
        print(str(exc))

FINAL_FEATURE_DIM = EMBED_SIZE + NODE_TYPE_EMBED_DIM + 6
coverage = 0.0 if total_nodes == 0 else 100.0 * (total_nodes - zero_embed_nodes) / total_nodes

print(f"\n🧠 Word2Vec node coverage: {coverage:.2f}% ({total_nodes - zero_embed_nodes}/{total_nodes})")
print(f"🔗 AST edges: {edge_counts[0]:,}")
print(f"🔗 CFG edges: {edge_counts[1]:,}")
print(f"🔗 PDG edges: {edge_counts[2]:,}")
print(f"⚠ Samples without PDG export: {missing_pdg}")

if pdg_files and edge_counts[2] == 0:
    raise RuntimeError(
        "PDG exports were found, but zero PDG edges were parsed. "
        "Refusing to save a silently broken PDG dataset; inspect the Joern PDG DOT format."
    )

print("\n💾 Saving dataset index...")

torch.save(
    {
        "dataset": DATASET,
        "graphs": dataset_index,
        "num_node_types": NUM_NODE_TYPES,
        "feature_dim": FINAL_FEATURE_DIM,
        "edge_type_map": {"AST": 0, "CFG": 1, "PDG": 2},
        "graph_stats": {
            "saved_graphs": saved_graphs,
            "failed_graphs": failed_graphs,
            "edge_counts": {"AST": edge_counts[0], "CFG": edge_counts[1], "PDG": edge_counts[2]},
            "graphs_without_pdg": missing_pdg,
        },
    },
    META_FILE,
)

print("\n✅ Graphs saved:", saved_graphs)
print("❌ Graphs failed:", failed_graphs)
print("🧩 Total nodes:", total_nodes)
print("🧠 Final feature dim:", FINAL_FEATURE_DIM)
print("\n🎉 Done.")
