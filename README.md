# Software Vulnerability Detection Using Graph Neural Networks

This repository contains the implementation and experimental pipeline developed for a master's thesis on **software vulnerability detection using static program analysis and machine learning**.

The project represents C functions as graphs extracted from source code and uses a **Gated Graph Neural Network (GGNN)** to classify functions as vulnerable or non-vulnerable. The experiments investigate how different program-graph representations affect vulnerability detection performance.

The implementation uses the **Devign dataset** and **Joern** for program graph extraction.

---

## Overview

The overall pipeline is:

```text
Devign Dataset
      │
      ▼
Function Extraction
      │
      ▼
Individual C Source Files
      │
      ├──────────────► Word2Vec Training
      │                       │
      │                       ▼
      │                Code Embeddings
      │
      ▼
Joern
      │
      ├── AST
      ├── CFG
      └── PDG
      │
      ▼
Graph Dataset Construction
      │
      ▼
Node Feature Construction
      │
      ▼
GGNN Training
      │
      ▼
Vulnerability Classification
      │
      ▼
Accuracy / Precision / Recall / F1 / AUC
```

The main pipeline is implemented in `main.py`, which can execute dataset preprocessing, Word2Vec training, Joern graph extraction, graph dataset construction, and GGNN training.

---

## Research Dataset

The project uses the **Devign dataset**, which contains vulnerable and non-vulnerable C functions collected from real-world open-source projects.

The experiments in this repository support:

* **QEMU**
* **FFmpeg**

The preprocessing script reads `dataset.json`, extracts the functions belonging to the selected project, writes each function to an individual `.c` file, and creates a CSV file containing the corresponding vulnerability labels and metadata.

### Devign

The original Devign implementation and dataset resources are available at:

**Devign repository:**
https://github.com/epicosy/devign

The Devign repository provides the original implementation and processing resources for vulnerability identification using graph neural networks.

---

## Program Graph Extraction

**Joern** is used to analyze the extracted C source files and generate graph representations of the programs.

The pipeline first creates a Code Property Graph (CPG) using `joern-parse`, executes the project's data-flow analysis script, and then exports graph representations for:

* Abstract Syntax Graph / AST
* Control Flow Graph / CFG
* Program Dependence Graph / PDG

This process is implemented directly in the project pipeline.

### Joern

The Joern project is available at:

**Joern repository:** 
https://github.com/joernio/joern Version: 4.0.485

Joern is an open-source code analysis platform based on Code Property Graphs and supports analysis of languages including C/C++.

---

## Graph Representations

The experiments evaluate seven graph configurations:

| Configuration     | Description                          |
| ----------------- | ------------------------------------ |
| `AST`             | Abstract Syntax Tree                 |
| `CFG`             | Control Flow Graph                   |
| `PDG`             | Program Dependence Graph             |
| `AST + CFG`       | Abstract Syntax + Control Flow       |
| `AST + PDG`       | Abstract Syntax + Program Dependence |
| `CFG + PDG`       | Control Flow + Program Dependence    |
| `AST + CFG + PDG` | Combination of all three graph types |

The training code explicitly supports all seven configurations. The dataset builder now combines the separate Joern PDG export with the AST/CFG export, so `PDG`, `AST+PDG`, `CFG+PDG`, and `AST+CFG+PDG` use real PDG edges rather than silently ignoring the PDG export.

These configurations allow the experiment to compare the contribution of different types of structural and semantic program information to vulnerability detection.

---

## Node Feature Construction

Each graph node is represented using a combination of several feature types.

### 1. Word2Vec Code Embeddings

The source code contained in graph nodes is tokenized and used to train a Word2Vec model.

The Word2Vec configuration is:

```text
Vector size : 100
Window      : 5
Min count   : 2
Workers     : 4
Architecture: Skip-gram
```

The resulting model is saved per project as:

```text
models/code_w2v_qemu.model
models/code_w2v_ffmpeg.model
```

The implementation uses the same tokenizer during Word2Vec training and graph feature construction to maintain compatibility between the embeddings and graph features.

### 2. Node-Type Embeddings

Each graph node type is mapped to a dense 32-dimensional embedding.

The implementation considers up to 256 node types.

### 3. Handcrafted Features

Six additional handcrafted features are extracted from each node:

* Token count
* Presence/frequency of selected vulnerability-related functions
* Arithmetic operators
* Comparison operators
* Pointer-related syntax
* Array-related syntax

These features are concatenated with the Word2Vec and node-type representations.

The final node representation therefore consists of:

```text
100-dimensional Word2Vec embedding
+
32-dimensional node-type embedding
+
6 handcrafted features
```

for a total of **138 input features per graph node**.

---

## GGNN Model

The graph classifier follows the Devign design while extending the edge vocabulary to the three representations studied in this thesis. Each node starts from a 138-dimensional representation (100-dimensional Word2Vec embedding, 32-dimensional node-type embedding, and 6 handcrafted features).

The model then applies a **6-step relation-aware GGNN**. AST, CFG, and PDG edges are represented by separate relation weights inside an `RGCNConv`, followed by a `GRUCell` update. The final graph readout uses the Devign-style dual Conv1d pathways: one processes `[initial features, final hidden state]`, the other processes the final hidden state, and their sequence-level outputs are multiplied before graph-level reduction.

This matters for the ablation: selecting `AST`, `CFG`, or `PDG` actually filters the corresponding relation edges before message passing, and combinations retain all selected relation types.


## Training Configuration

The default training configuration is:

| Parameter          |             Value |
| ------------------ | ----------------: |
| Epochs             |                60 |
| Batch size         |               128 |
| Optimizer          |             AdamW |
| Learning rate      |          1 × 10⁻⁴ |
| Weight decay       |        1.3 × 10⁻⁶ |
| LR scheduler       | ReduceLROnPlateau |
| Scheduler factor   |               0.5 |
| Scheduler patience |                 4 |
| Random seed        |                42 |
| Cross-validation   |          5-fold stratified |
| Mixed precision    |             CUDA AMP |
| GGNN hidden dim    |                200 |
| GGNN steps         |                  6 |

The implementation uses a `WeightedRandomSampler` during training so each fold sees approximately balanced class sampling; the loss itself is not additionally positive-weighted, avoiding double class reweighting.

For each fold, the checkpoint is selected by validation F1 (AUC and accuracy are tie-breakers), the classification threshold is selected on that fold's validation partition, and the final reported score is the fixed-threshold score of the selected checkpoint. Results are aggregated as mean ± standard deviation across the 5 folds and saved under `results/`.

---

## Runtime and GPU optimization

For Kaggle GPU runs, the trainer is configured to avoid the two main throughput bottlenecks in the original implementation:

* Graph `.pt` files are preloaded once into RAM instead of being reopened on every training sample and epoch.
* The Devign Conv readout is vectorized with `to_dense_batch`, so graphs in a mini-batch are processed by the Conv1d layers in GPU batches rather than with a Python loop over graphs.

CUDA AMP is enabled by default, pinned host memory is used for CUDA transfers, and worker processes use Linux `fork` so the read-only graph cache can be shared. The trainer prints peak allocated VRAM after every fold.

On a Kaggle Tesla T4, start with `--batch-size 128`. Increase it only after confirming the reported peak VRAM and runtime; filling VRAM is not itself a correctness objective.


## Evaluation Metrics

The trained models are evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **ROC-AUC**

The classification threshold is selected separately on each validation fold by evaluating multiple thresholds and choosing the threshold that provides the highest validation F1. That fold-specific threshold is then fixed when reporting the fold result.

---

## Project Structure

The main repository structure is:

```text
.
├── data/
│   ├── raw/
│   │   └── dataset.json
│   ├── intermediate/
│   │   ├── qemu_code/
│   │   ├── ffmpeg_code/
│   │   ├── qemu_labels.csv
│   │   ├── ffmpeg_labels.csv
│   │   ├── graphs/
│   │   └── pdg/
│   └── processed/
│       ├── qemu_graphs/
│       └── ffmpeg_graphs/
│
├── docs/
│
├── joern/
│   ├── joern-cli/
│   └── run_dataflow.sc
│
├── models/
│
├── results/
│
├── scripts/
│   ├── experiments/
│   ├── preprocessing/
│   │   ├── splitIntoFiles.py
│   │   ├── train_w2v.py
│   │   └── build_dataset.py
│   ├── training/
│   │   └── train_ggnn.py
│   └── utils/
│
├── main.py
└── README.md
```

The repository contains separate preprocessing, training, experiment, and utility directories, while intermediate and processed graph data are stored under `data/`.

---

## Kaggle GPU setup

The project is designed to run on a Kaggle GPU with approximately **14 GiB of VRAM**. CUDA is detected automatically and CUDA AMP is enabled by default. The training script also reports the GPU name, total VRAM, and peak VRAM allocated for each fold.

### 1. Create the Kaggle notebook

In Kaggle:

1. Create a new **Notebook**.
2. In **Settings**, set the accelerator to a **GPU**.
3. Upload this repository ZIP as a Kaggle Dataset, or upload/unzip it into the notebook working directory.
4. Make sure the repository root contains `main.py`, `scripts/`, `joern/`, and `data/`.

A typical Kaggle working directory is:

```text
/kaggle/working/graph-vuln-ablation-main-kfold-kaggle/
```

Then change into that directory:

```bash
cd /kaggle/working/graph-vuln-ablation-main-kfold-kaggle
```

### 2. Install Python dependencies

Run:

```bash
pip install -r requirements-kaggle.txt
```

The repository does **not** include the Joern binaries in the ZIP. You therefore need either to add a Kaggle Dataset containing a Joern installation under `joern/joern-cli/`, or point the code to an existing installation with `JOERN_HOME`. For example:

```bash
export JOERN_HOME=/kaggle/input/your-joern-dataset/joern-cli
```

The installation must provide executable `joern`, `joern-parse`, and `joern-export` files.

### 3. Add the Devign dataset

Place the Devign `dataset.json` at:

```text
data/raw/dataset.json
```

The Kaggle notebook can obtain this file either from an attached Kaggle Dataset or by downloading the dataset before running the pipeline. The code expects exactly this relative path.

### 4. Run a small validation experiment first

Before starting the complete 2-dataset × 7-configuration experiment, verify the environment with one configuration:

```bash
python main.py --dataset qemu --edge-types AST --folds 5 --epochs 60 --batch-size 128 --hidden-dim 200 --steps 6 --workers 4
```

This checks CUDA, Joern, preprocessing, graph loading, the GGNN, and 5-fold training without committing to the full experiment.

### 5. Run the complete experiment

After the smoke test succeeds:

```bash
python main.py \
  --full-experiment \
  --folds 5 \
  --epochs 60 \
  --batch-size 128 \
  --hidden-dim 200 \
  --steps 6 \
  --workers 4
```

This runs both **QEMU** and **FFmpeg** and evaluates all seven graph configurations:

```text
AST
CFG
PDG
AST + CFG
AST + PDG
CFG + PDG
AST + CFG + PDG
```

### 6. VRAM tuning for Kaggle

`--batch-size` is the main control for GPU memory usage. Start with `128`, then use the peak-VRAM value printed after each fold:

```text
Fold 1 peak VRAM allocated: ... GiB
```

If the run runs out of memory, reduce the batch size:

```bash
--batch-size 96
```

or:

```bash
--batch-size 64
```

If substantial VRAM remains unused and the run is stable, increase it (for example to `160` or `192`). The goal is **high utilization without out-of-memory errors**; PyTorch does not need to reserve all 14 GiB for every workload. AMP is enabled automatically on CUDA unless `--no-amp` is supplied.

### 7. Important Kaggle note about preprocessing

The complete pipeline is compute-heavy because Joern graph extraction is performed before training. The generated intermediate and processed files are stored inside the repository tree and can be reused in the same notebook session. Word2Vec models are cached separately for QEMU and FFmpeg.

For repeated model experiments, it is therefore faster to keep:

```text
data/intermediate/
data/processed/
models/
```

and rerun only the training commands.

### Recommended Kaggle training settings

| Parameter | Recommended value |
| --- | ---: |
| GPU | Kaggle GPU (~14 GiB VRAM) |
| Folds | 5 |
| Epochs | 60 |
| Batch size | 128 (tune using peak VRAM) |
| Hidden dimension | 200 |
| GGNN steps | 6 |
| Learning rate | 1 × 10⁻⁴ |
| Weight decay | 1.3 × 10⁻⁶ |
| AMP | Enabled |
| Workers | 4 |
| Seed | 42 |

## Installation

### Requirements

The project requires Python and the following major Python packages:

```text
numpy
pandas
gensim
torch
torch-geometric
scikit-learn
matplotlib
```

Joern must also be available because it is required for program graph extraction.

The original Devign repository similarly uses Joern, PyTorch, PyTorch Geometric, Gensim, Pandas, and Scikit-learn as core dependencies.

### Virtual Environment

Create and activate a Python virtual environment:

```bash
python -m venv venv
```

Linux/macOS:

```bash
source venv/bin/activate
```

Windows:

```powershell
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

> The exact PyTorch and PyTorch Geometric installation command may depend on the available CPU/CUDA configuration.

---

## Dataset Preparation

Place the Devign dataset file at:

```text
data/raw/dataset.json
```

The preprocessing script supports both QEMU and FFmpeg:

```bash
python scripts/preprocessing/splitIntoFiles.py --project qemu
```

or:

```bash
python scripts/preprocessing/splitIntoFiles.py --project ffmpeg
```

This produces:

```text
data/intermediate/qemu_code/
data/intermediate/qemu_labels.csv
```

or:

```text
data/intermediate/ffmpeg_code/
data/intermediate/ffmpeg_labels.csv
```

Each source function is stored as an individual C file and associated with its vulnerability label.

---

## Training Word2Vec

After extracting the source files, train the Word2Vec model:

```bash
python scripts/preprocessing/train_w2v.py
```

The model will be saved as:

```text
models/code_w2v.model
```

Word2Vec is trained separately for each project using that project's extracted C source files. This prevents the embedding vocabulary learned for one project from being reused as the sole embedding model for the other project.

---

## Running Joern

The pipeline expects Joern to be available under:

```text
joern/joern-cli/
```

The main pipeline invokes:

```text
joern-parse
joern
joern-export
```

to create the CPG, perform data-flow analysis, and export the required graph representations.

---

## Building the Graph Dataset

Once the Joern graphs have been generated:

```bash
python scripts/preprocessing/build_dataset.py --dataset qemu
```

or:

```bash
python scripts/preprocessing/build_dataset.py --dataset ffmpeg
```

The resulting graph objects are stored under:

```text
data/processed/qemu_graphs/
data/processed/ffmpeg_graphs/
```

along with a dataset index.

The graph-building stage loads the Word2Vec model, parses graph nodes and edge types, creates node features, and stores each graph as a PyTorch Geometric `Data` object.

---

## Training the GGNN

The training script performs **stratified k-fold cross-validation**. With the default `--folds 5`, the dataset is split into five stratified folds; each fold has its own model checkpoint, validation threshold, and metrics. The final result is reported as mean ± standard deviation across folds.

A single graph configuration can be trained using:

```bash
python scripts/training/train_ggnn.py --dataset qemu --edges ast
```

Examples:

```bash
python scripts/training/train_ggnn.py --dataset qemu --edges cfg
```

```bash
python scripts/training/train_ggnn.py --dataset qemu --edges pdg
```

```bash
python scripts/training/train_ggnn.py --dataset qemu --edges ast+cfg
```

```bash
python scripts/training/train_ggnn.py --dataset qemu --edges ast+pdg
```

```bash
python scripts/training/train_ggnn.py --dataset qemu --edges cfg+pdg
```

```bash
python scripts/training/train_ggnn.py --dataset qemu --edges ast+cfg+pdg
```

The same commands can be used for FFmpeg by replacing `qemu` with `ffmpeg`.

---

## Running the Complete Experiment

The easiest way to reproduce the complete experiment is:

```bash
python main.py --full-experiment
```

This runs the complete pipeline for both:

```text
QEMU
FFmpeg
```

and evaluates all seven graph configurations:

```text
AST
CFG
PDG
AST + CFG
AST + PDG
CFG + PDG
AST + CFG + PDG
```

The main CLI explicitly defines this full experiment workflow.

---

## Running a Single Dataset

For QEMU:

```bash
python main.py --dataset qemu
```

For FFmpeg:

```bash
python main.py --dataset ffmpeg
```

A specific graph configuration can also be selected:

```bash
python main.py --dataset qemu --edge-types AST,CFG
```

To run all configurations for one dataset:

```bash
python main.py --dataset qemu --edge-types ALL
```

---

## Interactive CLI

Running:

```bash
python main.py
```

opens an interactive menu that provides options for:

1. Splitting the QEMU dataset
2. Splitting the FFmpeg dataset
3. Training Word2Vec
4. Running Joern on QEMU
5. Running Joern on FFmpeg
6. Building the QEMU graph dataset
7. Building the FFmpeg graph dataset
8. Training the QEMU model
9. Training the FFmpeg model
10. Running the complete QEMU pipeline
11. Running the complete FFmpeg pipeline
12. Running the full experiment
13. Cleaning the workspace
14. Exiting

These options are implemented in `main.py`.

---

## Results

Experiment results are stored as JSON files under:

```text
results/
```

For example:

```text
results/qemu_ast_metrics.json
results/qemu_cfg_metrics.json
results/qemu_pdg_metrics.json
results/qemu_ast+cfg_metrics.json
results/qemu_ast+pdg_metrics.json
results/qemu_cfg+pdg_metrics.json
results/qemu_ast+cfg+pdg_metrics.json
```

Each result contains:

```text
Accuracy
Precision
Recall
F1
AUC
Threshold
```

as well as additional prediction statistics.

---

## Reproducibility

The implementation sets the random seed to `42` for Python, NumPy, and PyTorch and enables deterministic cuDNN behavior.

For reproducible experiments, use the same:

* Dataset version
* Python environment
* PyTorch/PyTorch Geometric versions
* Joern version
* Random seed
* Training configuration

---

## Cleaning Generated Files

The project provides a cleanup option:

```bash
python main.py --clean
```

This removes generated intermediate data, processed datasets, models, results, and workspace files.

---

## References and External Projects

This research implementation builds upon and uses resources from the following projects:

### Devign

**Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks**

https://github.com/epicosy/devign

The Devign repository provides the dataset and the original graph-based vulnerability detection implementation that motivated this work.

### Joern

**Joern — Open-source code analysis platform based on Code Property Graphs**

https://github.com/joernio/joern

Joern is used in this project for generating and exporting program graph representations from C source code.

---

## Citation

If you use this repository or its implementation in academic work, please cite the associated thesis and the original Devign work.

The external Devign and Joern projects should also be acknowledged when using their dataset, methodology, or software.

---

## License

This repository contains original research code developed for academic purposes. Components derived from or distributed with external projects remain subject to their respective licenses.

Please consult the licenses of:

* Devign
* Joern
* PyTorch
* PyTorch Geometric
* Gensim
* Other third-party dependencies

before redistributing the repository or its generated artifacts.

## Devign-paper reporting mode

The training script supports a reporting configuration based on the settings explicitly stated in the Devign paper (Zhou et al., NeurIPS 2019): random stratified 75% training / 25% validation split, a standard 0.5 classification threshold, and Accuracy and F1 as the primary reported metrics. The paper states 100-epoch patience for early stopping. The script therefore defaults to 100 epochs and patience 100.

The paper does not specify the exact metric used internally to choose an early-stopping checkpoint. This implementation uses validation Accuracy, with F1 as a tie-breaker, and records that choice explicitly in the results JSON. It does **not** present a separate test-set result for this reporting mode, because the main paper results are described as a 75% training / 25% validation evaluation.

The reported Accuracy/F1 should therefore be compared with the paper's Accuracy/F1 table only as a reporting-protocol comparison; the architecture and preprocessing in this thesis implementation are not identical to the original Devign implementation.
