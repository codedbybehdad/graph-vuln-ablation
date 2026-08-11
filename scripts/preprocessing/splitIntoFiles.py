"""
splitIntoFiles.py

Purpose
-------
This script reads the Devign dataset (dataset.json) and extracts the C source
code of each function into separate .c files.

Each function becomes one file so later steps (graph extraction and model
training) can process them individually.

The script also generates a CSV file containing the labels and metadata
for each function.

Supported projects
------------------
The Devign dataset contains functions from multiple projects such as:
- qemu
- ffmpeg

This script allows selecting which project to export so experiments can
be run separately for each dataset.

Output structure
----------------
Example for QEMU:

data/
 ├─ intermediate/
 │   ├─ qemu_code/          (individual .c files)
 │   └─ qemu_labels.csv     (labels and metadata)

Example for FFmpeg:

data/
 ├─ intermediate/
 │   ├─ ffmpeg_code/
 │   └─ ffmpeg_labels.csv

Usage
-----
python splitIntoFiles.py --project qemu
python splitIntoFiles.py --project ffmpeg
"""

import json
import os
import csv
import argparse


# -----------------------------
# Parse command line arguments
# -----------------------------
# Allows selecting which project to extract from the Devign dataset
parser = argparse.ArgumentParser(description="Split Devign dataset into individual source files")
parser.add_argument(
    "--project",
    type=str,
    required=True,
    choices=["qemu", "ffmpeg"],
    help="Project to extract from dataset (qemu or ffmpeg)"
)

args = parser.parse_args()
project_name = args.project.lower()


# -----------------------------
# File paths
# -----------------------------
# Devign dataset JSON input
input_file = "data/raw/dataset.json"

# Directory where extracted code files will be stored
code_dir = f"data/intermediate/{project_name}_code"

# CSV file storing labels and metadata
labels_file = f"data/intermediate/{project_name}_labels.csv"


# Create output directory if it does not exist
os.makedirs(code_dir, exist_ok=True)


# -----------------------------
# Load dataset JSON
# -----------------------------
# The dataset is a list of entries where each entry represents a function
with open(input_file, "r") as f:
    data = json.load(f)


# Counter used to assign file IDs
count = 0


# -----------------------------
# Open labels CSV for writing
# -----------------------------
with open(labels_file, "w", newline="") as csvfile:

    writer = csv.writer(csvfile)

    # CSV header
    writer.writerow(["id", "target", "project", "commit_id"])

    # Iterate through all dataset samples
    for item in data:

        # Identify which project the sample belongs to
        project = item.get("project", "").strip().lower()

        # Skip samples from other projects
        if project != project_name:
            continue

        # Extract function source code
        code = item.get("func", "").strip()

        # Skip empty functions
        if not code:
            continue

        # -----------------------------
        # Write C source file
        # -----------------------------
        # Each function becomes a separate file:
        # 0.c, 1.c, 2.c, ...
        output_path = os.path.join(code_dir, f"{count}.c")

        with open(output_path, "w") as out:
            out.write(code)

        # -----------------------------
        # Write metadata to CSV
        # -----------------------------
        writer.writerow([
            count,                       # unique ID for this sample
            item.get("target", 0),       # vulnerability label (0 = safe, 1 = vulnerable)
            project_name,                # project name (qemu / ffmpeg)
            item.get("commit_id", "")    # commit hash
        ])

        count += 1


# -----------------------------
# Final summary
# -----------------------------
print(f"✅ Done. Exported {count} samples from project '{project_name}'.")
print(f"Source files saved to: {code_dir}")
print(f"Labels saved to: {labels_file}")
