"""
Download all VIPERdb entries and save to viperdb_entries.csv.
Fields: pdb_id, name, t_number, outer_diameter (Å), num_subunits, chain_ids
"""

import csv
import json
import ssl
import time
import urllib.request
from pathlib import Path

BASE_URL = "https://viperdb.org/services"
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_PATH = DATA_DIR / "viperdb_entries.csv"
DELAY = 0.1  # seconds between requests to be polite

_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE


def fetch_json(url: str, retries: int = 3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, context=_ssl_ctx, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(1)


def get_all_entries() -> list:
    return fetch_json(f"{BASE_URL}/aa_info.php?serviceName=viruses")


def get_layers(pdb_id: str):
    try:
        data = fetch_json(f"{BASE_URL}/biodata.php?serviceName=layers&VDB={pdb_id}")
        return data[0] if data else None
    except Exception:
        return None


def get_num_subunits(pdb_id: str):
    try:
        return fetch_json(f"{BASE_URL}/biodata.php?serviceName=numSubunits&VDB={pdb_id}")
    except Exception:
        return None


def get_chain_ids(pdb_id: str) -> str:
    try:
        data = fetch_json(f"{BASE_URL}/oligomer_muli.php?serviceName=lai&VDB={pdb_id}")
        return ";".join(item["label_asym_id"] for item in data if "label_asym_id" in item)
    except Exception:
        return ""


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching entry list from VIPERdb...")
    entries = get_all_entries()
    total = len(entries)
    print(f"Found {total} entries. Downloading details...\n")

    fieldnames = ["pdb_id", "name", "t_number", "outer_diameter", "num_subunits", "chain_ids"]

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, entry in enumerate(entries, 1):
            pdb_id = entry.get("entry_id", "").strip()
            name = entry.get("name", "").strip()

            layers = get_layers(pdb_id)
            time.sleep(DELAY)

            num_subunits = get_num_subunits(pdb_id)
            time.sleep(DELAY)

            chain_ids = get_chain_ids(pdb_id)
            time.sleep(DELAY)

            t_number = layers.get("tnumber") if layers else None
            outer_diameter = layers.get("max_diameter") if layers else None

            writer.writerow({
                "pdb_id": pdb_id,
                "name": name,
                "t_number": t_number,
                "outer_diameter": outer_diameter,
                "num_subunits": num_subunits,
                "chain_ids": chain_ids,
            })

            if i % 50 == 0 or i == total:
                print(f"  [{i}/{total}] {pdb_id} done")

    print(f"\nSaved {total} entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
