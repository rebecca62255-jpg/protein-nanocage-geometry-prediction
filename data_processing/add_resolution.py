"""
Add structural resolution information to VIPERdb entries
and filter structures by resolution.

Input:
    data/viperdb_entries.csv

Outputs:
    data/viperdb_entries.csv
        - original entries with an added resolution column

    data/viperdb_filtered_3.5A.csv
        - entries with resolution <= 3.5 Å
"""

import csv
import json
import ssl
import time
import urllib.request

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ============================================================
# 1. File paths and settings
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = ROOT / "data" / "viperdb_entries.csv"

FILTERED_OUTPUT = (
    ROOT /
    "data" /
    "viperdb_filtered_3.5A.csv"
)

BASE_URL = (
    "https://viperdb.org/services/biodata.php"
    "?serviceName=biodata_values&VDB={pdb_id}"
)

MAX_WORKERS = 10
RETRIES = 3

RESOLUTION_THRESHOLD = 3.5


# ============================================================
# 2. SSL configuration
# ============================================================

_ssl_ctx = ssl.create_default_context()

_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE


# ============================================================
# 3. Retrieve resolution from VIPERdb
# ============================================================

def fetch_resolution(pdb_id: str):

    url = BASE_URL.format(
        pdb_id=pdb_id
    )

    for attempt in range(RETRIES):

        try:

            with urllib.request.urlopen(
                url,
                context=_ssl_ctx,
                timeout=15
            ) as response:

                data = json.loads(
                    response.read().decode()
                )


            value = data.get(
                "resolution"
            )


            if value in (
                None,
                "",
                "None"
            ):

                return pdb_id, None


            return (
                pdb_id,
                float(value)
            )


        except Exception:

            if attempt == RETRIES - 1:

                return (
                    pdb_id,
                    None
                )

            time.sleep(1)


# ============================================================
# 4. Main processing
# ============================================================

def main():

    # --------------------------------------------------------
    # Load VIPERdb entries
    # --------------------------------------------------------

    with CSV_PATH.open(
        encoding="utf-8"
    ) as f:

        rows = list(
            csv.DictReader(f)
        )


    total = len(rows)


    print(
        f"Loaded {total} entries "
        f"from {CSV_PATH.name}"
    )

    print(
        f"Querying VIPERdb with "
        f"{MAX_WORKERS} workers...\n"
    )


    # --------------------------------------------------------
    # Retrieve resolution values
    # --------------------------------------------------------

    pdb_ids = [
        row["pdb_id"]
        for row in rows
    ]


    resolution_map = {}

    done = 0


    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {

            executor.submit(
                fetch_resolution,
                pdb_id
            ): pdb_id

            for pdb_id in pdb_ids
        }


        for future in as_completed(
            futures
        ):

            pdb_id, resolution = (
                future.result()
            )

            resolution_map[pdb_id] = (
                resolution
            )

            done += 1


            if (
                done % 100 == 0
                or
                done == total
            ):

                found = sum(
                    1
                    for value
                    in resolution_map.values()
                    if value is not None
                )

                print(
                    f"[{done}/{total}] "
                    f"resolved: {found}"
                )


    # --------------------------------------------------------
    # Add resolution column
    # --------------------------------------------------------

    fieldnames = list(
        rows[0].keys()
    )


    if "resolution" not in fieldnames:

        fieldnames.append(
            "resolution"
        )


    for row in rows:

        row["resolution"] = (
            resolution_map.get(
                row["pdb_id"]
            )
        )


    # --------------------------------------------------------
    # Overwrite original CSV with resolution column
    # --------------------------------------------------------

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    # --------------------------------------------------------
    # Filter entries with resolution <= 3.5 Å
    # --------------------------------------------------------

    filtered_rows = []

    missing_resolution = 0


    for row in rows:

        value = row.get(
            "resolution"
        )


        if value in (
            None,
            "",
            "None"
        ):

            missing_resolution += 1
            continue


        try:

            resolution = float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            missing_resolution += 1
            continue


        if (
            resolution
            <= RESOLUTION_THRESHOLD
        ):

            filtered_rows.append(
                row
            )


    # --------------------------------------------------------
    # Save filtered dataset
    # --------------------------------------------------------

    FILTERED_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with FILTERED_OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            filtered_rows
        )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    found = sum(
        1
        for value
        in resolution_map.values()
        if value is not None
    )


    print(
        f"\nResolution found: "
        f"{found}/{total}"
    )

    print(
        f"CSV updated:\n"
        f"{CSV_PATH}"
    )

    print(
        f"\nResolution <= "
        f"{RESOLUTION_THRESHOLD} Å: "
        f"{len(filtered_rows)} entries"
    )

    print(
        f"Missing/unusable resolution: "
        f"{missing_resolution}"
    )

    print(
        f"Filtered dataset saved to:\n"
        f"{FILTERED_OUTPUT}"
    )


# ============================================================
# 5. Run
# ============================================================

if __name__ == "__main__":
    main()
