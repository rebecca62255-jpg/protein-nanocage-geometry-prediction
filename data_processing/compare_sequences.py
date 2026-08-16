"""
For 5 selected VIPERdb entries:
  1. Download PDB file via BioPython
  2. Extract protein sequences per asymmetric-unit chain
  3. Fetch sequences from VIPERdb ss_info API
  4. Align and compare the two sources
"""

import json
import ssl
import time
import urllib.request
from pathlib import Path

from Bio import Align
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder

# ── Config ──────────────────────────────────────────────────────────────────
ENTRIES = [
    {"pdb_id": "1rb8", "t_number": "1",    "chain_ids": "F;G;J;X"},
    {"pdb_id": "8g1r", "t_number": "4",    "chain_ids": "A;B;C;D;E"},
    {"pdb_id": "8e8l", "t_number": "pT3",  "chain_ids": "A;B;C;D;H;L"},
    {"pdb_id": "3j4u", "t_number": "7l",   "chain_ids": "A;B;C;D;E;F;G;H;I;J;K;L;M;N"},
    {"pdb_id": "7xr2", "t_number": "13",   "chain_ids": "A;B;C;D;E;F;G;H;I;J;K;L;M;N;O;X;Y"},
]

PDB_DIR   = Path(__file__).parent / "pdb_files"
BASE_URL  = "https://viperdb.org/services/ss_info.php"
DELAY     = 0.15

_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode    = ssl.CERT_NONE


# ── Helpers ──────────────────────────────────────────────────────────────────
def fetch(url: str):
    with urllib.request.urlopen(url, context=_ssl_ctx, timeout=15) as r:
        return json.loads(r.read().decode())


def download_pdb(pdb_id: str, dest_dir: Path) -> Path:
    """Download PDB file from RCSB using urllib (SSL bypass)."""
    dest = dest_dir / f"{pdb_id}.pdb"
    if dest.exists():
        return dest
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    with urllib.request.urlopen(url, context=_ssl_ctx, timeout=30) as r:
        dest.write_bytes(r.read())
    return dest


def viper_chain_info(pdb_id: str) -> list[dict]:
    """Return [{label_asym_id, MINlsi, MAXlsi}, ...] from VIPERdb."""
    url = f"{BASE_URL}?serviceName=chain&VDB={pdb_id}"
    try:
        return fetch(url)
    except Exception:
        return []


def viper_sequence(pdb_id: str, asym: str, start: int, end: int) -> str:
    """Return single-letter sequence string from VIPERdb for one chain."""
    url = f"{BASE_URL}?serviceName=sequence&VDB={pdb_id}&asym={asym}&asymstart={start}&asymend={end}"
    try:
        data = fetch(url)
        # Filter protein residues: single uppercase letter name, not X/- gap
        return "".join(
            r["name"] for r in data
            if len(r.get("name", "")) == 1 and r["name"].isupper()
        )
    except Exception:
        return ""


def pdb_sequence(structure, chain_id: str) -> str:
    """Extract sequence from a BioPython chain by iterating residues."""
    ppb = PPBuilder()
    model = structure[0]
    if chain_id not in [c.id for c in model]:
        return ""
    chain = model[chain_id]
    segs = ppb.build_peptides(chain, aa_only=True)
    return "".join(str(pp.get_sequence()) for pp in segs)


def align_summary(seq_pdb: str, seq_viper: str) -> dict:
    """Global pairwise alignment; return score, identity%, lengths."""
    if not seq_pdb or not seq_viper:
        return {"identity": None, "score": None, "len_pdb": len(seq_pdb), "len_viper": len(seq_viper)}
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score    =  1
    aligner.mismatch_score = -1
    aligner.open_gap_score    = -2
    aligner.extend_gap_score  = -0.5
    alignments = aligner.align(seq_pdb, seq_viper)
    best = next(iter(alignments))
    aligned_pdb, aligned_viper = best[0], best[1]
    matches = sum(a == b for a, b in zip(aligned_pdb, aligned_viper))
    aln_len = max(len(aligned_pdb), len(aligned_viper))
    identity = 100.0 * matches / aln_len if aln_len else 0.0
    return {
        "identity": round(identity, 1),
        "score": best.score,
        "len_pdb": len(seq_pdb),
        "len_viper": len(seq_viper),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = PDBParser(QUIET=True)

    for entry in ENTRIES:
        pdb_id  = entry["pdb_id"]
        t_num   = entry["t_number"]
        asu_chains = entry["chain_ids"].split(";")

        print(f"\n{'═' * 60}")
        print(f"  {pdb_id.upper()}  (T={t_num})  ASU chains: {', '.join(asu_chains)}")
        print(f"{'═' * 60}")

        # 1. Download PDB
        pdb_path = download_pdb(pdb_id, PDB_DIR)
        print(f"  PDB file : {pdb_path.name}")

        structure = parser.get_structure(pdb_id, pdb_path)
        pdb_chains = {c.id for c in structure[0]}

        # 2. VIPERdb chain info
        time.sleep(DELAY)
        viper_chains = viper_chain_info(pdb_id)
        viper_chain_map = {c["label_asym_id"]: c for c in viper_chains}

        print(f"  VIPERdb chains: {[c['label_asym_id'] for c in viper_chains]}")
        print(f"  PDB chains    : {sorted(pdb_chains)}")
        print()

        # 3. Compare per chain
        for chain_id in asu_chains:
            seq_pdb = pdb_sequence(structure, chain_id)

            vinfo = viper_chain_map.get(chain_id)
            if vinfo:
                time.sleep(DELAY)
                seq_viper = viper_sequence(
                    pdb_id, chain_id, vinfo["MINlsi"], vinfo["MAXlsi"]
                )
            else:
                seq_viper = ""

            result = align_summary(seq_pdb, seq_viper)

            status = ""
            if not seq_pdb:
                status = " [PDB: no protein residues]"
            if not seq_viper:
                status += " [VIPERdb: no sequence]"

            if result["identity"] is not None:
                print(
                    f"  chain {chain_id} | "
                    f"PDB {result['len_pdb']:>4} aa | "
                    f"VIPERdb {result['len_viper']:>4} aa | "
                    f"identity {result['identity']:>5.1f}%"
                )
            else:
                print(f"  chain {chain_id} |{status}")


if __name__ == "__main__":
    main()
