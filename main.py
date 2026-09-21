"""
main.py
Main CLI for the graph vulnerability detection pipeline.
"""

import subprocess
import sys
import os
import shutil
import argparse


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Keep the original repository layout, but resolve paths relative to this file
# so the CLI works even when it is launched from another working directory.
LOCAL_JOERN_DIR = os.path.join(BASE_DIR, "joern", "joern-cli")
# Optional override for an existing Joern installation.
# Example: export JOERN_HOME=/path/to/joern/joern-cli
JOERN_DIR = os.environ.get("JOERN_HOME", LOCAL_JOERN_DIR)

JOERN_PARSE = os.path.join(JOERN_DIR, "joern-parse")
JOERN_EXPORT = os.path.join(JOERN_DIR, "joern-export")


def run_command(command_list):
    try:
        print("\n▶ Running:")
        if isinstance(command_list, str):
            print(command_list, "\n")
            subprocess.run(command_list, shell=True, check=True, cwd=BASE_DIR)
        else:
            print(" ".join(command_list), "\n")
            subprocess.run(command_list, check=True, cwd=BASE_DIR)

        print("✅ Completed successfully.\n")

    except subprocess.CalledProcessError as e:
        print("\n❌ Error while running command:")
        if isinstance(command_list, list):
            print(" ".join(command_list))
        else:
            print(command_list)
        print(e)
        sys.exit(1)


def split_dataset(dataset):
    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "scripts/preprocessing/splitIntoFiles.py"),
        "--project",
        dataset,
    ]
    run_command(cmd)


def train_w2v_if_needed(dataset):
    # Train and cache a separate embedding model for each project.
    # This prevents a QEMU-only Word2Vec model from being reused for FFmpeg.
    model_path = f"models/code_w2v_{dataset}.model"

    if os.path.exists(model_path):
        print(f"✅ Word2Vec model for {dataset.upper()} already exists. Skipping.\n")
        return

    print(f"🧠 Training Word2Vec for {dataset.upper()}...\n")

    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "scripts/preprocessing/train_w2v.py"),
        "--dataset",
        dataset,
    ]
    run_command(cmd)


def run_joern(dataset):
    code_dir = os.path.join(BASE_DIR, "data", "intermediate", f"{dataset}_code")

    missing = [p for p in (JOERN_PARSE, JOERN_EXPORT) if not os.path.isfile(p) or not os.access(p, os.X_OK)]
    if missing:
        print("\n❌ Joern executable(s) not found or not executable:")
        for p in missing:
            print(f"   {p}")
        print("\nPlace the Joern executables in joern/joern-cli/ or set JOERN_HOME to an existing Joern installation.")
        sys.exit(1)
    cpg_file = os.path.join(BASE_DIR, "data", "intermediate", "devign.cpg")

    graph_dir = os.path.join(BASE_DIR, "data", "intermediate", "graphs")
    pdg_dir = os.path.join(BASE_DIR, "data", "intermediate", "pdg")

    if not os.path.exists(code_dir):
        print(f"\n❌ Code directory not found: {code_dir}")
        sys.exit(1)

    os.makedirs("data/intermediate", exist_ok=True)

    if os.path.exists(graph_dir):
        shutil.rmtree(graph_dir)

    if os.path.exists(pdg_dir):
        shutil.rmtree(pdg_dir)

    if os.path.exists(cpg_file):
        os.remove(cpg_file)

    print("\n⚙ Building CPG with joern-parse...\n")

    run_command([
        JOERN_PARSE,
        code_dir,
        "--output",
        cpg_file
    ])

    print("\n⚙ Exporting CPG graphs (AST + CFG)...\n")

    run_command([
        JOERN_EXPORT,
        "--repr",
        "cpg14",
        "--out",
        graph_dir,
        cpg_file
    ])

    print("\n⚙ Exporting PDG graphs...\n")

    run_command([
        JOERN_EXPORT,
        "--repr",
        "pdg",
        "--out",
        pdg_dir,
        cpg_file
    ])

    cpg_dot_files = []
    pdg_dot_files = []
    for root, _, files in os.walk(graph_dir):
        cpg_dot_files.extend(os.path.join(root, f) for f in files if f.endswith(".dot"))
    for root, _, files in os.walk(pdg_dir):
        pdg_dot_files.extend(os.path.join(root, f) for f in files if f.endswith(".dot"))

    if not cpg_dot_files:
        print(f"\n❌ Joern produced no CPG14 .dot files in: {graph_dir}")
        sys.exit(1)
    if not pdg_dot_files:
        print(f"\n❌ Joern produced no PDG .dot files in: {pdg_dir}")
        sys.exit(1)

    print(f"\n✅ CPG14 dot files: {len(cpg_dot_files)}")
    print(f"✅ PDG dot files: {len(pdg_dot_files)}")
    print("\n✅ Joern graph extraction finished.\n")


def build_dataset(dataset):

    print(f"\n⚙ Building dataset for {dataset}...\n")

    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "scripts/preprocessing/build_dataset.py"),
        "--dataset",
        dataset
    ]

    run_command(cmd)


def train_model(dataset, edge_types="ALL", folds=5):

    if isinstance(edge_types, list):
        edge_types = ",".join(edge_types)

    edges_arg = edge_types.lower().replace(",", "+")

    print(f"\n⚙ Training model for {dataset} with edges: {edges_arg}...\n")

    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "scripts/training/train_ggnn.py"),
        "--dataset",
        dataset,
        "--edges",
        edges_arg,
        "--folds",
        str(folds),
    ]

    run_command(cmd)


def clean_workspace():
    paths = [
        "data/intermediate",
        "data/processed",
        "models",
        "results",
        "workspace",
    ]

    print("\n⚠ Cleaning workspace\n")

    for p in paths:
        if os.path.exists(p):
            print("Removing", p)
            if os.path.isfile(p):
                os.remove(p)
            else:
                shutil.rmtree(p)

    print("\n✅ Workspace cleaned\n")


def get_all_edge_combinations():
    return [
        "AST",
        "CFG",
        "PDG",
        "AST,CFG",
        "AST,PDG",
        "CFG,PDG",
        "AST,CFG,PDG",
    ]


def normalize_edge_combo(edge_string):

    parts = [p.strip().upper() for p in edge_string.split(",") if p.strip()]

    valid_types = {"AST", "CFG", "PDG"}

    invalid = [p for p in parts if p not in valid_types]

    if invalid:
        raise ValueError(f"Invalid edge type(s): {', '.join(invalid)}")

    canonical = ["AST", "CFG", "PDG"]

    return ",".join([e for e in canonical if e in parts])


def parse_edge_combinations(edge_input):

    if edge_input is None or not edge_input.strip():
        return ["AST,CFG"]

    edge_input = edge_input.strip().upper()

    if edge_input == "ALL":
        return get_all_edge_combinations()

    return [normalize_edge_combo(edge_input)]


def run_pipeline(dataset, edge_combinations=None):

    if edge_combinations is None:
        edge_combinations = ["AST,CFG"]

    print("\n=====================================")
    print(f" Running Pipeline for {dataset.upper()}")
    print("=====================================\n")

    split_dataset(dataset)
    train_w2v_if_needed(dataset)
    run_joern(dataset)

    print("\n--- Building dataset (once) ---")

    build_dataset(dataset)

    # When running the complete seven-configuration experiment, call the
    # training script once with --edges all. This guarantees that one set of
    # reproducible 5-fold partitions is created and reused for every ablation
    # inside the same process, and produces a combined summary JSON.
    normalized = {normalize_edge_combo(edges) for edges in edge_combinations}
    all_combos = {normalize_edge_combo(edges) for edges in get_all_edge_combinations()}

    if normalized == all_combos:
        print("\n--- Training all seven edge configurations with 5-fold CV ---")
        train_model(dataset, "ALL", folds=5)
    else:
        for edges in edge_combinations:
            print(f"\n--- Training for edge combination: {edges} ---")
            train_model(dataset, edges, folds=5)

    print(f"\n✅ Pipeline finished for {dataset.upper()}.")


def run_full_experiment():

    datasets = ["qemu", "ffmpeg"]

    combos = get_all_edge_combinations()

    for dataset in datasets:
        run_pipeline(dataset, combos)

    print("\n🎉 Full experiment finished.\n")


def menu():

    parser = argparse.ArgumentParser(description="Graph Vulnerability Detection Pipeline CLI")

    parser.add_argument("--dataset", choices=["qemu", "ffmpeg"])

    parser.add_argument("--edge-types")

    parser.add_argument("--full-experiment", action="store_true")

    parser.add_argument("--clean", action="store_true")

    args = parser.parse_args()

    if args.clean:
        clean_workspace()
        return

    if args.full_experiment:
        run_full_experiment()
        return

    if args.dataset:

        combos = parse_edge_combinations(args.edge_types)

        run_pipeline(args.dataset, combos)

        return

    while True:

        print("\n=================================================")
        print("Graph Vulnerability Detection CLI")
        print("=================================================")
        print("1. Split dataset (QEMU)")
        print("2. Split dataset (FFmpeg)")
        print("3. Train Word2Vec")
        print("4. Run Joern (QEMU)")
        print("5. Run Joern (FFmpeg)")
        print("6. Build dataset (QEMU)")
        print("7. Build dataset (FFmpeg)")
        print("8. Train model (QEMU)")
        print("9. Train model (FFmpeg)")
        print("10. Run full pipeline (QEMU)")
        print("11. Run full pipeline (FFmpeg)")
        print("12. Run FULL experiment")
        print("13. Clean workspace")
        print("14. Exit")

        choice = input("\nSelect option: ").strip()

        if choice == "1":
            split_dataset("qemu")

        elif choice == "2":
            split_dataset("ffmpeg")

        elif choice == "3":
            dataset = input("Dataset for Word2Vec (qemu/ffmpeg): ").strip().lower()
            train_w2v_if_needed(dataset)

        elif choice == "4":
            run_joern("qemu")

        elif choice == "5":
            run_joern("ffmpeg")

        elif choice == "6":
            build_dataset("qemu")

        elif choice == "7":
            build_dataset("ffmpeg")

        elif choice == "8":

            edges = input("Edge types (AST,CFG etc or 'all'): ")

            combos = parse_edge_combinations(edges)

            for c in combos:
                train_model("qemu", c)

        elif choice == "9":

            edges = input("Edge types (AST,CFG etc or 'all'): ")

            combos = parse_edge_combinations(edges)

            for c in combos:
                train_model("ffmpeg", c)

        elif choice == "10":
            run_pipeline("qemu")

        elif choice == "11":
            run_pipeline("ffmpeg")

        elif choice == "12":
            run_full_experiment()

        elif choice == "13":
            clean_workspace()

        elif choice == "14":
            break

        else:
            print("Invalid choice")


if __name__ == "__main__":
    menu()
