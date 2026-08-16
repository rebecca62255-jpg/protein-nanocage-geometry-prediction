import csv

INPUT_CSV = "/nobackup/rmgl20/dissertation/scripts/viperdb_filtered_3.5A.csv"

with open(INPUT_CSV) as f:
    reader = csv.DictReader(f)
    entries = list(reader)

print(f"Total entries: {len(entries)}")
print(f"Columns: {list(entries[0].keys())}\n")

fields_to_check = ["pdb_id", "t_number", "resolution", "outer_diameter"]

for field in fields_to_check:
    if field not in entries[0]:
        print(f"WARNING: column '{field}' does not exist in CSV")
        continue
    missing = [e["pdb_id"] for e in entries if not e[field].strip()]
    print(f"{field}: {len(missing)} missing")
    if missing[:5]:
        print(f"  examples: {missing[:5]}")
