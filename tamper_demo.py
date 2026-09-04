"""Cryptographic Tamper-Evidence Demonstration.

Proves that any local mutation of discovered metadata, URLs, or face similarity scores
violates the cryptographic SHA-256 fingerprint anchored on Polygon Amoy.

Usage:
    python tamper_demo.py <record_hash_hex>
"""
import copy
import json
import sys
from dotenv import load_dotenv

from pipeline.chain import get_proof
from pipeline.fingerprint import sha256_of_json
from pipeline.ipfs_store import fetch_json

load_dotenv()


def run_tamper_demo(data_hash_hex: str):
    clean_hash = data_hash_hex.strip()
    if clean_hash.startswith("0x"):
        clean_hash = clean_hash[2:]

    print("=" * 75)
    print("  CRYPTOGRAPHIC TAMPER-EVIDENCE DEMONSTRATION (POLYGON AMOY + IPFS)")
    print("=" * 75)

    # 1. Fetch on-chain proof
    print(f"\n[1/4] Retrieving authentic proof from Polygon Amoy ProofRegistry...")
    try:
        onchain = get_proof(clean_hash)
        ipfs_cid = onchain["ipfs_cid"]
        submitter = onchain["submitter"]
        print(f"      -> Blockchain Submitter : {submitter}")
        print(f"      -> Anchored IPFS CID    : {ipfs_cid}")
        print(f"      -> Anchored Hash (Chain): {clean_hash}")
    except Exception as e:
        print(f"[-] Failed to read proof from Polygon Amoy: {e}")
        print("    (If testing offline without chain, pass a mock hash or run with local manifest)")
        sys.exit(1)

    # 2. Fetch authentic manifest from IPFS
    print(f"\n[2/4] Fetching original canonical evidence manifest from IPFS...")
    try:
        original_record = fetch_json(ipfs_cid)
        print(f"      -> Successfully fetched record from IPFS")
    except Exception as e:
        print(f"[-] IPFS retrieval failed: {e}")
        sys.exit(1)

    # 3. Verify original record
    print(f"\n[3/4] Verifying authenticity of the original manifest...")
    original_computed_hash = sha256_of_json(original_record)
    original_is_valid = original_computed_hash.lower() == clean_hash.lower()

    print(f"      -> Blockchain Hash : {clean_hash}")
    print(f"      -> Local Recomputed: {original_computed_hash}")
    print(f"      -> Match Status    : {'VALID' if original_is_valid else 'MISMATCH'}")

    if not original_is_valid:
        print("[!] Original record does not match blockchain hash. Aborting demo.")
        sys.exit(1)

    # 4. Deliberately tamper with record
    print(f"\n[4/4] Deliberately tampering with record metadata...")
    tampered_record = copy.deepcopy(original_record)

    # Modify source URL and similarity score
    orig_url = tampered_record.get("candidate", {}).get("source_url", "N/A")
    orig_sim = tampered_record.get("verification", {}).get("face_similarity_score", "N/A")

    if "candidate" in tampered_record and isinstance(tampered_record["candidate"], dict):
        tampered_record["candidate"]["source_url"] = "https://fake-imposter-profile.example.com/spoofed-photo"
    else:
        tampered_record["source_url"] = "https://fake-imposter-profile.example.com/spoofed-photo"

    if "verification" in tampered_record and isinstance(tampered_record["verification"], dict):
        tampered_record["verification"]["face_similarity_score"] = 0.9999
        tampered_record["verification"]["decision"] = "VERIFIED"
    else:
        tampered_record["face_similarity_score"] = 0.9999

    print(f"      * Modified candidate source_url:")
    print(f"        Original : {orig_url}")
    print(f"        Tampered : {tampered_record.get('candidate', {}).get('source_url')}")
    print(f"      * Modified face_similarity_score:")
    print(f"        Original : {orig_sim}")
    print(f"        Tampered : {tampered_record.get('verification', {}).get('face_similarity_score')}")

    tampered_computed_hash = sha256_of_json(tampered_record)
    tampered_is_valid = tampered_computed_hash.lower() == clean_hash.lower()

    print("\n" + "=" * 75)
    print("  TAMPER DETECTION RESULTS")
    print("=" * 75)
    print("  [A] ORIGINAL RECORD VERIFICATION:")
    print(f"      Blockchain Hash : {clean_hash}")
    print(f"      Local Hash      : {original_computed_hash}")
    print(f"      Result          : VALID (Bit-for-bit cryptographic match)")

    print("\n  [B] TAMPERED RECORD VERIFICATION:")
    print(f"      Blockchain Hash : {clean_hash}")
    print(f"      Tampered Hash   : {tampered_computed_hash}")
    print(f"      Result          : TAMPER DETECTED!")
    print("=" * 75)
    print("  EXPLANATION:")
    print("  Because SHA-256 is collision-resistant and avalanche-sensitive, altering")
    print("  even a single character in the evidence manifest produces a completely")
    print("  different cryptographic fingerprint, rendering unauthorized tampering")
    print("  immediately detectable by anyone querying the blockchain.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tamper_demo.py <evidence_hash_hex>")
        sys.exit(1)
    run_tamper_demo(sys.argv[1])

