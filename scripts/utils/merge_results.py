import json
from pathlib import Path

results_dir = Path("results")
output_file = results_dir / "combined_metrics.json"

combined = {}

for json_file in results_dir.glob("*.json"):
    with open(json_file, "r") as f:
        data = json.load(f)

    # use filename without .json as key
    key = json_file.stem
    combined[key] = data

with open(output_file, "w") as f:
    json.dump(combined, f, indent=4)

print(f"Combined metrics written to {output_file}")
