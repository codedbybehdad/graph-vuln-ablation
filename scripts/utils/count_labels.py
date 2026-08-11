import json

# Load the dataset
with open("data/raw/dataset.json", "r") as f:
    data = json.load(f)

# Count per project
counts = {}

for item in data:
    project = item.get("project", "Unknown")
    target = item.get("target", -1)  # 0 = safe, 1 = vulnerable

    if project not in counts:
        counts[project] = {"vuln": 0, "safe": 0}

    if target == 1:
        counts[project]["vuln"] += 1
    elif target == 0:
        counts[project]["safe"] += 1

# Print results
print("=" * 50)
print("Vulnerability Distribution by Project")
print("=" * 50)

for project, stats in counts.items():
    total = stats["vuln"] + stats["safe"]
    vuln_pct = (stats["vuln"] / total) * 100 if total > 0 else 0
    safe_pct = (stats["safe"] / total) * 100 if total > 0 else 0

    print(f"\n{project}:")
    print(f"  Vulnerable: {stats['vuln']:>6} ({vuln_pct:>5.1f}%)")
    print(f"  Safe:       {stats['safe']:>6} ({safe_pct:>5.1f}%)")
    print(f"  Total:      {total:>6}")

# Overall totals
total_vuln = sum(s["vuln"] for s in counts.values())
total_safe = sum(s["safe"] for s in counts.values())
total_all = total_vuln + total_safe

print("\n" + "=" * 50)
print("OVERALL")
print("=" * 50)
print(f"  Vulnerable: {total_vuln:>6} ({(total_vuln/total_all)*100:>5.1f}%)")
print(f"  Safe:       {total_safe:>6} ({(total_safe/total_all)*100:>5.1f}%)")
print(f"  Total:      {total_all:>6}")