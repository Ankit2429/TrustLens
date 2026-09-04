"""Standalone independent cryptographic re-verification against Polygon Amoy and IPFS.

Reads the anchored proof from the deployed ProofRegistry smart contract, fetches the
canonical evidence manifest from IPFS, recomputes the SHA-256 hash, and verifies
tamper-evident integrity.

Usage:
    python verify.py <record_hash_hex>
"""
import datetime
import sys
from dotenv import load_dotenv

from pipeline.chain import get_proof
from pipeline.fingerprint import sha256_of_json
from pipeline.ipfs_store import fetch_json

load_dotenv()


def verify(data_hash_hex: str) -> bool:
    """Perform independent blockchain and IPFS cryptographic re-verification.

    Args:
        data_hash_hex: 64-character SHA-256 hex string.

    Returns:
        True if the recomputed hash matches the blockchain hash, False otherwise.
    """
    clean_hash = data_hash_hex.strip()
    if clean_hash.startswith("0x"):
        clean_hash = clean_hash[2:]

    print("=" * 70)
    print("  INDEPENDENT PROOF RE-VERIFICATION (POLYGON AMOY + IPFS)")
    print("=" * 70)
    print(f"\n[1/3] Querying ProofRegistry smart contract on Polygon Amoy...")
    try:
        onchain = get_proof(clean_hash)
    except Exception as e:
        print(f"[-] Smart contract query failed: {e}")
        print("    Ensure CONTRACT_ADDRESS and AMOY_RPC_URL are valid.")
        return False

    submitter = onchain["submitter"]
    ipfs_cid = onchain["ipfs_cid"]
    ts = onchain["timestamp"]
    date_str = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print(f"      -> Status     : FOUND ON-CHAIN")
    print(f"      -> Submitter  : {submitter}")
    print(f"      -> IPFS CID   : {ipfs_cid}")
    print(f"      -> Timestamp  : {ts} ({date_str})")

    print(f"\n[2/3] Retrieving canonical evidence manifest from IPFS...")
    try:
        record = fetch_json(ipfs_cid)
        print(f"      -> Successfully retrieved record ({len(str(record))} chars)")
    except Exception as e:
        print(f"[-] Failed to fetch record from IPFS: {e}")
        return False

    print(f"\n[3/3] Canonicalizing manifest and recomputing SHA-256 fingerprint...")
    recomputed_hash = sha256_of_json(record)
    is_valid = recomputed_hash.lower() == clean_hash.lower()

    print("\n" + "=" * 70)
    print("  VERIFICATION RESULT")
    print("=" * 70)
    print(f"  BLOCKCHAIN HASH : {clean_hash}")
    print(f"  LOCAL HASH      : {recomputed_hash}")
    print(f"  IPFS CID        : {ipfs_cid}")
    print(f"  NETWORK         : Polygon Amoy (Chain ID 80002)")
    print(f"  SUBMITTER       : {submitter}")
    print(f"  TIMESTAMP       : {date_str}")
    print(f"  RESULT          : {'VALID (Bit-for-bit cryptographic match)' if is_valid else 'TAMPER DETECTED'}")
    print("=" * 70)

    # Display provenance if manifest has standard schema
    cand = record.get("candidate", {})
    verif = record.get("verification", {})
    face = record.get("face", {})

    print("\n  Provenanced Evidence Metadata:")
    print(f"  - Source URL          : {cand.get('source_url', 'N/A')}")
    print(f"  - Platform / Domain   : {cand.get('platform', 'N/A')} ({cand.get('domain', 'N/A')})")
    print(f"  - Face Similarity     : {verif.get('face_similarity_score', 'N/A')}")
    if verif.get("separation_margin") is not None:
        print(f"  - Separation Margin   : {verif.get('separation_margin')} ({verif.get('margin_interpretation', '')})")
    print(f"  - Decision Status     : [{verif.get('decision', 'N/A')}] - {verif.get('decision_reason', '')}")
    if verif.get("candidate_image_quality") is not None:
        print(f"  - Candidate Quality   : {verif.get('candidate_image_quality')}")
    print(f"  - Face Algorithm      : {face.get('model', 'ArcFace')} ({face.get('algorithm', 'buffalo_l')})")
    print(f"  - Embedding Hash      : {face.get('embedding_hash', 'N/A')[:24]}...")
    print("=" * 70 + "\n")

    return is_valid


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify.py <evidence_hash_hex>")
        sys.exit(1)
    success = verify(sys.argv[1])
    sys.exit(0 if success else 1)
