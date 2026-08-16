"""
Add a 'resolution' column to viperdb_entries.csv using the VIPERdb biodata API.
Uses concurrent requests (10 workers) to speed up the process.
Overwrites the original CSV in place.
"""

import csv
import json
import ssl
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CSV_PATH = Path(__file__).parent / "viperdb_entries.csv"
BASE_URL = "https://viperdb.org/services/biodata.php?serviceName=biodata_values&VDB={pdb_id}"
MAX_WORKERS = 10
RETRIES = 3

_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE


def fetch_resolution(pdb_id: str) -> tuple[str, object]:
    url = BASE_URL.format(pdb_id=pdb_id)
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url, context=_ssl_ctx, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            val = data.get("resolution")
            return pdb_id, float(val) if val not in (None, "", "None") else None
        except Exception:
            if attempt == RETRIES - 1:
                return pdb_id, None
            time.sleep(1)


def main():
    with CSV_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    total = len(rows)
    print(f"Loaded {total} entries from {CSV_PATH.name}")
    print(f"Querying VIPERdb with {MAX_WORKERS} workers...\n")

    pdb_ids = [r["pdb_id"] for r in rows]
    resolution_map: dict[str, object] = {}
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_resolution, pid): pid for pid in pdb_ids}
        for future in as_completed(futures):
            pdb_id, resolution = future.result()
            resolution_map[pdb_id] = resolution
            done += 1
            if done % 100 == 0 or done == total:
                found = sum(1 for v in resolution_map.values() if v is not None)
                print(f"  [{done}/{total}]  resolved: {found}")

    fieldnames = list(rows[0].keys())
    if "resolution" not in fieldnames:
        fieldnames.append("resolution")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row["resolution"] = resolution_map.get(row["pdb_id"])
            writer.writerow(row)

    found = sum(1 for v in resolution_map.values() if v is not None)
    print(f"\nDone. Resolution found: {found}/{total}")
    print(f"CSV updated: {CSV_PATH}")


if __name__ == "__main__":
    main()
