"""Cryptographic Tamper-Evidence Demonstration.

Proves that any mutation of discovered metadata, URLs, face similarity scores,
or quality metrics violates the cryptographic SHA-256 fingerprint anchored on Polygon Amoy.

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

    # 4. Multi-scenario deliberate tampering tests
    print(f"\n[4/4] Executing multi-scenario cryptographic tamper simulations...")
    
    # Scenario A: URL Tampering
    tampered_a = copy.deepcopy(original_record)
    if "candidate" in tampered_a and isinstance(tampered_a["candidate"], dict):
        tampered_a["candidate"]["source_url"] = "https://fake-imposter-profile.example.com/spoofed-photo"
    else:
        tampered_a["source_url"] = "https://fake-imposter-profile.example.com/spoofed-photo"
    hash_a = sha256_of_json(tampered_a)
    is_valid_a = hash_a.lower() == clean_hash.lower()

    # Scenario B: Similarity Score Tampering
    tampered_b = copy.deepcopy(original_record)
    if "verification" in tampered_b and isinstance(tampered_b["verification"], dict):
        tampered_b["verification"]["face_similarity_score"] = 0.9999
    else:
        tampered_b["face_similarity_score"] = 0.9999
    hash_b = sha256_of_json(tampered_b)
    is_valid_b = hash_b.lower() == clean_hash.lower()

    # Scenario C: Platform / Quality Metadata Tampering
    tampered_c = copy.deepcopy(original_record)
    if "candidate" in tampered_c and isinstance(tampered_c["candidate"], dict):
        tampered_c["candidate"]["platform"] = "Manipulated Platform"
    if "verification" in tampered_c and isinstance(tampered_c["verification"], dict):
        tampered_c["verification"]["decision"] = "FORGED_STATUS"
    hash_c = sha256_of_json(tampered_c)
    is_valid_c = hash_c.lower() == clean_hash.lower()

    print("\n" + "=" * 75)
    print("  TAMPER DETECTION RESULTS (MULTI-SCENARIO)")
    print("=" * 75)
    print("  [AUTHENTIC] ORIGINAL RECORD VERIFICATION:")
    print(f"      Blockchain Hash : {clean_hash}")
    print(f"      Local Hash      : {original_computed_hash}")
    print(f"      Result          : VALID (Bit-for-bit cryptographic match)")

    print("\n  [SCENARIO 1] SOURCE URL MODIFICATION:")
    print(f"      Blockchain Hash : {clean_hash}")
    print(f"      Tampered Hash   : {hash_a}")
    print(f"      Result          : {'TAMPER DETECTED!' if not is_valid_a else 'FAIL'}")

    print("\n  [SCENARIO 2] FACE SIMILARITY SCORE MODIFICATION:")
    print(f"      Blockchain Hash : {clean_hash}")
    print(f"      Tampered Hash   : {hash_b}")
    print(f"      Result          : {'TAMPER DETECTED!' if not is_valid_b else 'FAIL'}")

    print("\n  [SCENARIO 3] PLATFORM & DECISION METADATA FORGERY:")
    print(f"      Blockchain Hash : {clean_hash}")
    print(f"      Tampered Hash   : {hash_c}")
    print(f"      Result          : {'TAMPER DETECTED!' if not is_valid_c else 'FAIL'}")
    print("=" * 75)
    print("  EXPLANATION:")
    print("  Because SHA-256 is collision-resistant and avalanche-sensitive, altering")
    print("  even a single bit or character in the evidence manifest produces a")
    print("  completely different cryptographic hash, rendering unauthorized tampering")
    print("  immediately detectable by anyone querying the blockchain.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tamper_demo.py <evidence_hash_hex>")
        sys.exit(1)
    run_tamper_demo(sys.argv[1])
