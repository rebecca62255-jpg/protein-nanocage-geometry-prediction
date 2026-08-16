"""
Read viperdb_entries.csv and print summary statistics:
  - Total entries
  - T-number distribution
  - Symmetry type distribution (derived from T-number notation)
  - Outer diameter range
"""

import csv
import statistics
from collections import Counter
from pathlib import Path

CSV_PATH = Path(__file__).parent / "viperdb_entries.csv"


def classify_symmetry(t: str) -> str:
    """Infer symmetry type from VIPERdb T-number notation."""
    if not t or t == "":
        return "Unknown"
    if t == "multi":
        return "Multi-component"
    if t.startswith("pT"):
        return "Pseudo-T"
    if t.endswith("d"):
        return "Dextro (Td)"
    if t.endswith("l"):
        return "Laevo (Tl)"
    try:
        int(t)
        return "Standard T"
    except ValueError:
        return "Other"


def print_distribution(counter: Counter, title: str, unit: str = "", top_n: int = None):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")
    items = counter.most_common(top_n) if top_n else sorted(counter.items(), key=lambda x: (str(x[0]).zfill(10)))
    for key, count in items:
        bar = "█" * min(count // 5, 40)
        print(f"  {str(key) + unit:<15}  {count:>5}  {bar}")


def main():
    with CSV_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)

    # T-number distribution
    t_counter = Counter()
    for r in rows:
        t_counter[r["t_number"] if r["t_number"] else "N/A"] += 1

    # Symmetry type distribution
    sym_counter = Counter(classify_symmetry(r["t_number"]) for r in rows)

    # Outer diameter
    diameters = []
    for r in rows:
        try:
            diameters.append(float(r["outer_diameter_A"]))
        except (ValueError, TypeError):
            pass

    # ── Output ──────────────────────────────────────────
    print(f"\n{'═' * 50}")
    print(f"  VIPERdb Summary  ({CSV_PATH.name})")
    print(f"{'═' * 50}")
    print(f"\n  Total entries : {total}")
    print(f"  Entries with diameter data : {len(diameters)}")

    print_distribution(t_counter, "T-number distribution")

    print_distribution(sym_counter, "Symmetry type distribution", top_n=None)

    print(f"\n{'─' * 50}")
    print(f"  Outer diameter range (Å)")
    print(f"{'─' * 50}")
    print(f"  Min    : {min(diameters):.0f} Å")
    print(f"  Max    : {max(diameters):.0f} Å")
    print(f"  Mean   : {statistics.mean(diameters):.1f} Å")
    print(f"  Median : {statistics.median(diameters):.0f} Å")
    print(f"  Stdev  : {statistics.stdev(diameters):.1f} Å")

    print(f"\n{'═' * 50}\n")


if __name__ == "__main__":
    main()
