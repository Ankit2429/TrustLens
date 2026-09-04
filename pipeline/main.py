"""End-to-end pipeline: face scan -> multi-platform web discovery -> independent face verification -> IPFS -> Polygon Amoy proof.

Usage:
    python -m pipeline.main path/to/face.jpg
"""
import argparse
import hashlib
import os
import sys
import tempfile
from typing import Any, Optional

import requests
from dotenv import load_dotenv

from pipeline.face_id import (
    analyze_face,
    cosine_similarity,
    get_embedding_metadata,
    hash_embedding,
    normalize_embedding,
)
from pipeline.search import reverse_image_search, filter_social_matches
from pipeline.fingerprint import build_evidence_manifest, sha256_of_json, sha256_of_bytes
from pipeline.ipfs_store import pin_file, pin_json, gateway_url
from pipeline.chain import register_proof, get_proof

load_dotenv()

DEFAULT_VERIFIED_THRESHOLD = float(os.environ.get("VERIFIED_THRESHOLD", os.environ.get("MATCH_SIMILARITY_THRESHOLD", "0.40")))
DEFAULT_REVIEW_THRESHOLD = float(os.environ.get("REVIEW_THRESHOLD", "0.30"))


def _download_to_temp(url: str, suffix: str = ".jpg") -> tuple[str, bytes]:
    """Download remote image URL to a temporary file and return (temp_path, raw_bytes)."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    raw = resp.content

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(raw)
    tmp.close()
    return tmp.name, raw


def classify_decision(
    similarity: float,
    verified_threshold: float,
    review_threshold: float,
) -> tuple[str, str]:
    """Determine verification decision and rationale code.

    Returns:
        (decision, reason_str) where decision is 'VERIFIED', 'REVIEW', or 'REJECTED'.
    """
    if similarity >= verified_threshold:
        return (
            "VERIFIED",
            f"High ArcFace embedding cosine similarity ({similarity:.4f} >= {verified_threshold:.2f})",
        )
    elif similarity >= review_threshold:
        return (
            "REVIEW",
            f"Moderate ArcFace cosine similarity ({similarity:.4f} >= {review_threshold:.2f}); warrants manual inspection",
        )
    else:
        return (
            "REJECTED",
            f"Low ArcFace cosine similarity ({similarity:.4f} < {review_threshold:.2f}) below threshold",
        )


def run_pipeline(
    image_path: str,
    verified_threshold: float = DEFAULT_VERIFIED_THRESHOLD,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    skip_blockchain: bool = False,
):
    """Execute the full 9-step Task 3 pipeline."""
    print("=" * 70)
    print("  FACE IDENTIFICATION & BLOCKCHAIN VERIFICATION PIPELINE")
    print("=" * 70)

    # ---------------------------------------------------------
    # [1/9] Face Detection & Quality Assessment
    # ---------------------------------------------------------
    print(f"\n[1/9] Detecting face in query image: {image_path}")
    if not os.path.exists(image_path):
        print(f"[-] Error: File not found at '{image_path}'")
        sys.exit(1)

    try:
        query_analysis = analyze_face(image_path)
    except Exception as e:
        print(f"[-] Face detection failed: {e}")
        sys.exit(1)

    print(f"      -> Face detected: YES (Total faces in image: {query_analysis['face_count']})")
    print(f"      -> Detection confidence: {query_analysis['det_score']:.4f}")
    print(f"      -> Selected bounding box: {query_analysis['bbox']}")

    # ---------------------------------------------------------
    # [2/9] Full 512-d Face Embedding & Deterministic Hashing
    # ---------------------------------------------------------
    print("\n[2/9] Generating full 512-d ArcFace embedding & deterministic SHA-256 hash")
    query_emb = query_analysis["normalized_embedding"]
    query_face_hash = query_analysis["embedding_hash"]
    query_face_meta = query_analysis["metadata"]

    print(f"      -> Model: {query_face_meta['model']} ({query_face_meta['algorithm']})")
    print(f"      -> Dimensions: {query_face_meta['dimension']}-d (dtype: {query_face_meta['dtype']})")
    print(f"      -> Normalization: {query_face_meta['normalization_status']}")
    print(f"      -> Full Embedding SHA-256: {query_face_hash}")

    # ---------------------------------------------------------
    # [3/9] Query Image IPFS Pinning (Pinata Gateway)
    # ---------------------------------------------------------
    print("\n[3/9] Pinning query image to IPFS for public reverse search access")
    query_cid = None
    public_search_url = None
    try:
        query_cid = pin_file(image_path, name=f"query_face_{os.path.basename(image_path)}")
        public_search_url = gateway_url(query_cid)
        print(f"      -> IPFS CID: {query_cid}")
        print(f"      -> Gateway URL: {public_search_url}")
    except Exception as e:
        print(f"[!] IPFS pinning failed ({e}).")
        print("    Ensure PINATA_JWT is configured in .env.")
        sys.exit(1)

    # ---------------------------------------------------------
    # [4/9] Multi-Source & Multi-Platform Web Discovery
    # ---------------------------------------------------------
    print("\n[4/9] Running multi-source visual discovery across indexed web & social platforms")
    print("      (Querying SerpApi Google Lens engine...)")
    try:
        candidates = reverse_image_search(public_search_url)
    except Exception as e:
        print(f"[-] Search operation failed: {e}")
        print("    Check SERPAPI_KEY in .env and your network connection.")
        sys.exit(1)

    if not candidates:
        print("[-] No visual matches discovered. Try a more distinctive public photo.")
        sys.exit(1)

    print(f"      -> Discovered {len(candidates)} candidate result(s)")
    platform_counts: dict[str, int] = {}
    for c in candidates:
        plat = c.get("platform", "General Web")
        platform_counts[plat] = platform_counts.get(plat, 0) + 1

    print("      -> Platform distribution:")
    for plat, count in sorted(platform_counts.items(), key=lambda x: -x[1]):
        print(f"         - {plat}: {count} candidate(s)")

    # ---------------------------------------------------------
    # [5/9] Independent Candidate Face Verification
    # ---------------------------------------------------------
    print("\n[5/9] Performing independent face detection & ArcFace verification on candidate images")
    verified_results = []
    usable_images_count = 0
    faces_detected_count = 0

    for idx, c in enumerate(candidates, start=1):
        thumb_url = c.get("thumbnail")
        if not thumb_url:
            c["evaluation_status"] = "NO_IMAGE"
            continue

        temp_path = None
        try:
            temp_path, raw_bytes = _download_to_temp(thumb_url)
            usable_images_count += 1
            thumb_hash = sha256_of_bytes(raw_bytes)
            c["thumbnail_sha256"] = thumb_hash

            cand_analysis = analyze_face(temp_path)
            faces_detected_count += 1
            cand_emb = cand_analysis["normalized_embedding"]
            sim = cosine_similarity(query_emb, cand_emb)

            decision, reason = classify_decision(sim, verified_threshold, review_threshold)

            result_entry = {
                "candidate": c,
                "similarity": sim,
                "decision": decision,
                "reason": reason,
                "face_count": cand_analysis["face_count"],
                "det_confidence": cand_analysis["det_score"],
                "thumbnail_sha256": thumb_hash,
            }
            verified_results.append(result_entry)
        except Exception:
            # Download failed, corrupt image, or no face detected in thumbnail
            continue
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    # ---------------------------------------------------------
    # [6/9] Candidate Ranking & Decision Analysis
    # ---------------------------------------------------------
    print("\n[6/9] Ranking candidates and evaluating decision classifications")
    decision_priority = {"VERIFIED": 3, "REVIEW": 2, "REJECTED": 1}
    verified_results.sort(
        key=lambda r: (
            decision_priority.get(r["decision"], 0),
            r["similarity"],
            r["det_confidence"],
        ),
        reverse=True,
    )

    verified_count = sum(1 for r in verified_results if r["decision"] == "VERIFIED")
    review_count = sum(1 for r in verified_results if r["decision"] == "REVIEW")
    rejected_count = sum(1 for r in verified_results if r["decision"] == "REJECTED")

    print("\n  --- DISCOVERY & VERIFICATION SUMMARY ---")
    print(f"  Candidates Discovered    : {len(candidates)}")
    print(f"  Usable Images Evaluated  : {usable_images_count}")
    print(f"  Faces Detected           : {faces_detected_count}")
    print(f"  VERIFIED Matches         : {verified_count}")
    print(f"  REVIEW Matches           : {review_count}")
    print(f"  REJECTED Matches         : {rejected_count}")
    print("  ----------------------------------------\n")

    if verified_results:
        print("  Top Evaluated Candidates:")
        for rank_idx, r in enumerate(verified_results[:5], start=1):
            cand = r["candidate"]
            print(f"  Candidate #{rank_idx}")
            print(f"    Platform       : {cand.get('platform')}")
            print(f"    Source URL     : {cand.get('link')}")
            print(f"    Title          : {cand.get('title')[:60]}")
            print(f"    Face Similarity: {r['similarity']:.4f}")
            print(f"    Decision       : [{r['decision']}] - {r['reason']}")
            print()

    # Select primary match for evidence anchoring
    usable_matches = [r for r in verified_results if r["decision"] in ("VERIFIED", "REVIEW")]
    if not usable_matches:
        if verified_results:
            top_cand = verified_results[0]
            print(f"[!] Warning: Highest candidate similarity was {top_cand['similarity']:.4f}, which is below review threshold ({review_threshold:.2f}).")
            print("    Creating REJECTED baseline evidence record for demonstration provenance.")
            chosen_match = top_cand
        else:
            print("[-] No candidate images could be parsed or contained detectable faces.")
            sys.exit(1)
    else:
        chosen_match = usable_matches[0]

    cand_data = chosen_match["candidate"]
    cand_sim = chosen_match["similarity"]
    cand_decision = chosen_match["decision"]
    cand_reason = chosen_match["reason"]

    # ---------------------------------------------------------
    # [7/9] Canonical Evidence Manifest Generation & SHA-256
    # ---------------------------------------------------------
    print("\n[7/9] Generating RFC-8785 canonical evidence manifest & cryptographic SHA-256 fingerprint")
    manifest, manifest_hash = build_evidence_manifest(
        query_face_metadata=query_face_meta,
        candidate=cand_data,
        similarity_score=cand_sim,
        decision=cand_decision,
        verified_threshold=verified_threshold,
        review_threshold=review_threshold,
        decision_reason=cand_reason,
        query_image_cid=query_cid,
        total_candidates=len(candidates),
        usable_images_count=usable_images_count,
        candidate_face_count=chosen_match.get("face_count", 1),
        candidate_det_confidence=chosen_match.get("det_confidence", 1.0),
        thumbnail_sha256=chosen_match.get("thumbnail_sha256"),
    )

    print(f"      -> Manifest Schema: {manifest['schema_version']}")
    print(f"      -> Target Source: {cand_data.get('link')}")
    print(f"      -> Canonical SHA-256 Hash: {manifest_hash}")

    # ---------------------------------------------------------
    # [8/9] Evidence Manifest IPFS Storage
    # ---------------------------------------------------------
    print("\n[8/9] Pinning evidence manifest to IPFS via Pinata")
    try:
        manifest_cid = pin_json(manifest, name=f"evidence_proof_{manifest_hash[:12]}")
        manifest_gateway_url = gateway_url(manifest_cid)
        print(f"      -> Evidence IPFS CID: {manifest_cid}")
        print(f"      -> Gateway Reference: {manifest_gateway_url}")
    except Exception as e:
        print(f"[!] IPFS Manifest Pinning failed: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # [9/9] Polygon Amoy Proof Anchoring & Immediate Verification
    # ---------------------------------------------------------
    print("\n[9/9] Anchoring proof to Polygon Amoy Testnet (ProofRegistry Smart Contract)")
    if skip_blockchain:
        print("      -> [SKIPPED] Blockchain anchoring skipped via --skip-blockchain flag.")
        receipt = {"tx_hash": "N/A (offline mode)", "block": 0, "status": "SKIPPED"}
    else:
        try:
            receipt = register_proof(manifest_hash, manifest_cid)
            print(f"      -> Transaction Hash: {receipt['tx_hash']}")
            print(f"      -> Block Number   : {receipt['block']}")
            print(f"      -> Status         : {receipt['status']}")
            print(f"      -> Contract       : {receipt['contract_address']}")
            print(f"      -> Submitter      : {receipt['submitter']}")
            print(f"      -> Network        : Polygon Amoy (Chain ID 80002)")
        except Exception as e:
            print(f"[-] Blockchain transaction failed: {e}")
            print("    Check PRIVATE_KEY, CONTRACT_ADDRESS, and AMOY_RPC_URL in .env.")
            sys.exit(1)

        # Immediate independent on-chain read verification
        print("\n  [+] Performing immediate read verification against Polygon Amoy...")
        try:
            onchain_proof = get_proof(manifest_hash)
            print(f"      -> On-chain verification: CONFIRMED")
            print(f"      -> On-chain Submitter   : {onchain_proof['submitter']}")
            print(f"      -> On-chain IPFS CID    : {onchain_proof['ipfs_cid']}")
            print(f"      -> On-chain Timestamp   : {onchain_proof['timestamp']}")
        except Exception as e:
            print(f"[!] On-chain proof lookup failed: {e}")

    # ---------------------------------------------------------
    # Final Output Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PIPELINE EXECUTION COMPLETE")
    print("=" * 70)
    print(f"  Evidence SHA-256 Hash : {manifest_hash}")
    print(f"  Evidence IPFS CID     : {manifest_cid}")
    if not skip_blockchain:
        print(f"  Polygon Tx Hash       : {receipt['tx_hash']}")
    print(f"  Verified Match        : {cand_data.get('link')} [{cand_decision}]")
    print("\n  To re-verify this proof independently anytime:")
    print(f"    python verify.py {manifest_hash}")
    print("\n  To run the cryptographic tamper demonstration:")
    print(f"    python tamper_demo.py {manifest_hash}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Task 3: Face Identification & Blockchain Verification Pipeline"
    )
    parser.add_argument("image", help="Path to input face-scan image (e.g. demo/sample_face.jpg)")
    parser.add_argument(
        "--verified-threshold",
        type=float,
        default=DEFAULT_VERIFIED_THRESHOLD,
        help=f"Cosine similarity threshold for VERIFIED status (default: {DEFAULT_VERIFIED_THRESHOLD})",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=DEFAULT_REVIEW_THRESHOLD,
        help=f"Cosine similarity threshold for REVIEW status (default: {DEFAULT_REVIEW_THRESHOLD})",
    )
    parser.add_argument(
        "--skip-blockchain",
        action="store_true",
        help="Skip writing transaction to Polygon Amoy (useful for dry runs / offline testing)",
    )
    args = parser.parse_args()

    run_pipeline(
        image_path=args.image,
        verified_threshold=args.verified_threshold,
        review_threshold=args.review_threshold,
        skip_blockchain=args.skip_blockchain,
    )

