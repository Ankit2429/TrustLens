"""End-to-end pipeline: face scan -> multi-platform web discovery -> independent face verification -> IPFS -> Polygon Amoy proof.

Usage:
    python -m pipeline.main path/to/face.jpg [--face-index 0] [--verified-threshold 0.40] [--review-threshold 0.30] [--min-quality 0.25] [--skip-blockchain]
"""
import argparse
import os
import sys
import tempfile
from typing import Any, Optional

import requests
from dotenv import load_dotenv

from pipeline.face_id import (
    analyze_face,
    compare_group_faces,
    cosine_similarity,
    detect_all_faces,
    get_embedding_metadata,
    hash_embedding,
    normalize_embedding,
)
from pipeline.search import reverse_image_search, filter_social_matches
from pipeline.fingerprint import (
    build_evidence_manifest,
    calculate_separation_margin,
    sha256_of_json,
    sha256_of_bytes,
)
from pipeline.ipfs_store import pin_file, pin_json, gateway_url
from pipeline.chain import register_proof, get_proof

load_dotenv()

DEFAULT_VERIFIED_THRESHOLD = float(os.environ.get("VERIFIED_THRESHOLD", "0.40"))
DEFAULT_REVIEW_THRESHOLD = float(os.environ.get("REVIEW_THRESHOLD", "0.30"))
DEFAULT_MIN_QUALITY = float(os.environ.get("MIN_QUALITY_THRESHOLD", "0.20"))


def _download_to_temp(url: str, suffix: str = ".jpg", max_size_bytes: int = 10 * 1024 * 1024) -> tuple[str, bytes]:
    """Safely download remote image URL to a temporary file with size/timeout limits.

    Args:
        url: Remote HTTP(S) image URL.
        suffix: File extension.
        max_size_bytes: Maximum allowed byte size (default 10 MB).

    Returns:
        (temp_file_path, raw_bytes).
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    resp = requests.get(url, headers=headers, timeout=15, stream=True)
    resp.raise_for_status()

    content_len = resp.headers.get("Content-Length")
    if content_len and int(content_len) > max_size_bytes:
        raise ValueError(f"Image exceeds maximum size limit ({content_len} > {max_size_bytes} bytes)")

    raw = bytearray()
    for chunk in resp.iter_content(chunk_size=65536):
        raw.extend(chunk)
        if len(raw) > max_size_bytes:
            raise ValueError(f"Image download exceeded maximum size limit of {max_size_bytes} bytes")

    raw_bytes = bytes(raw)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(raw_bytes)
    tmp.close()
    return tmp.name, raw_bytes


def classify_decision(
    similarity: float,
    verified_threshold: float,
    review_threshold: float,
    is_quality_pass: bool = True,
) -> tuple[str, str]:
    """Determine verification decision and explainable rationale code.

    Returns:
        (decision, reason_str) where decision is 'VERIFIED', 'REVIEW', or 'REJECTED'.
    """
    if not is_quality_pass:
        return (
            "REVIEW",
            f"Face similarity ({similarity:.4f}) meets baseline, but image quality warrants manual inspection",
        )

    if similarity >= verified_threshold:
        return (
            "VERIFIED",
            f"Face similarity ({similarity:.4f} >= {verified_threshold:.2f}) exceeds verified threshold and candidate passed quality checks",
        )
    elif similarity >= review_threshold:
        return (
            "REVIEW",
            f"Similarity ({similarity:.4f} >= {review_threshold:.2f}) is near decision boundary; warrants manual inspection",
        )
    else:
        return (
            "REJECTED",
            f"Face similarity ({similarity:.4f} < {review_threshold:.2f}) below review threshold",
        )


def run_pipeline(
    image_path: str,
    face_index: int = 0,
    verified_threshold: float = DEFAULT_VERIFIED_THRESHOLD,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    min_quality: float = DEFAULT_MIN_QUALITY,
    skip_blockchain: bool = False,
):
    """Execute the hardened 9-step Task 3 pipeline."""
    print("=" * 70)
    print("  TRUSTLENS — FACE IDENTIFICATION & BLOCKCHAIN VERIFICATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # [1/9] Robust Face Detection, Quality Assessment & Selection
    # ---------------------------------------------------------
    print(f"\n[1/9] Detecting and analyzing face(s) in query image: {image_path}")
    if not os.path.exists(image_path):
        print(f"[-] Error: File not found at '{image_path}'")
        sys.exit(1)

    try:
        query_analysis = analyze_face(image_path, face_index=face_index, min_quality=min_quality)
    except Exception as e:
        print(f"[-] Face analysis failed: {e}")
        sys.exit(1)

    total_faces = query_analysis["face_count"]
    print(f"      -> Total Faces Detected : {total_faces}")
    print(f"      -> Selected Face Index  : {face_index} of {total_faces}")
    print(f"      -> Detection Confidence : {query_analysis['det_score']:.4f}")
    print(f"      -> Overall Quality Score: {query_analysis['quality_score']:.4f} (Usable: {query_analysis['is_usable']})")
    print(f"         * Sharpness  : {query_analysis['quality_breakdown']['sharpness']:.4f}")
    print(f"         * Exposure   : {query_analysis['quality_breakdown']['exposure']:.4f}")
    print(f"         * Resolution : {query_analysis['quality_breakdown']['resolution']:.4f}")
    print(f"         * Frontality : {query_analysis['quality_breakdown']['frontality']:.4f}")
    print(f"      -> Bounding Box Coordinates: {query_analysis['bbox']}")

    if total_faces > 1:
        print("\n      [!] Multi-Face Notice: Multiple faces detected in input.")
        for f_info in query_analysis["all_faces_summary"]:
            selected_marker = " [SELECTED TARGET]" if f_info["index"] == face_index else ""
            print(f"          - Face #{f_info['index']}: Det={f_info['det_score']:.2f}, Quality={f_info['quality_score']:.2f}, Box={f_info['bbox']}{selected_marker}")
        print("          (Use `--face-index <N>` to target a different subject).\n")

    # ---------------------------------------------------------
    # [2/9] Full 512-d Face Embedding & Deterministic Hashing
    # ---------------------------------------------------------
    print("\n[2/9] Extracting 512-d ArcFace embedding & computing deterministic SHA-256 fingerprint")
    query_emb = query_analysis["normalized_embedding"]
    query_face_hash = query_analysis["embedding_hash"]
    query_face_meta = query_analysis["metadata"]

    print(f"      -> Model Architecture   : {query_face_meta['model']} ({query_face_meta['algorithm']})")
    print(f"      -> Vector Dimension     : {query_face_meta['dimension']}-d ({query_face_meta['dtype']})")
    print(f"      -> Normalization Status : {query_face_meta['normalization_status']}")
    print(f"      -> Full Embedding SHA256: {query_face_hash}")

    # ---------------------------------------------------------
    # [3/9] Query Image IPFS Pinning (Pinata Gateway)
    # ---------------------------------------------------------
    print("\n[3/9] Pinning query image to IPFS for decentralized visual discovery")
    try:
        query_cid = pin_file(image_path, name=f"query_face_{os.path.basename(image_path)}")
        public_search_url = gateway_url(query_cid)
        print(f"      -> Query IPFS CID       : {query_cid}")
        print(f"      -> Gateway Search URL   : {public_search_url}")
    except Exception as e:
        print(f"[!] IPFS pinning failed ({e}). Check PINATA_JWT in .env.")
        sys.exit(1)

    # ---------------------------------------------------------
    # [4/9] Multi-Source & Multi-Platform Web Discovery
    # ---------------------------------------------------------
    print("\n[4/9] Performing multi-source visual discovery across indexed web & social platforms")
    print("      (Querying SerpApi Google Lens engine...)")
    try:
        candidates = reverse_image_search(public_search_url)
    except Exception as e:
        print(f"[-] Discovery failed: {e}. Check SERPAPI_KEY in .env.")
        sys.exit(1)

    if not candidates:
        print("[-] No visual matches discovered. Try a more distinctive public photo.")
        sys.exit(1)

    print(f"      -> Discovered {len(candidates)} candidate result(s)")
    platform_counts: dict[str, int] = {}
    for c in candidates:
        plat = c.get("platform", "General Web")
        platform_counts[plat] = platform_counts.get(plat, 0) + 1

    print("      -> Multi-source platform breakdown:")
    for plat, count in sorted(platform_counts.items(), key=lambda x: -x[1]):
        print(f"         - {plat:<15}: {count} candidate(s)")

    # ---------------------------------------------------------
    # [5/9] Independent Candidate Multi-Face Verification
    # ---------------------------------------------------------
    print("\n[5/9] Independently evaluating candidate image thumbnails with ArcFace")
    verified_results = []
    usable_images_count = 0
    all_evaluated_similarities: list[float] = []

    for idx, c in enumerate(candidates, start=1):
        thumb_url = c.get("thumbnail")
        if not thumb_url:
            continue

        temp_path = None
        try:
            temp_path, raw_bytes = _download_to_temp(thumb_url)
            usable_images_count += 1
            thumb_hash = sha256_of_bytes(raw_bytes)
            c["thumbnail_sha256"] = thumb_hash

            cand_faces = detect_all_faces(temp_path, min_quality=0.15)
            if not cand_faces:
                continue

            # Compare query face against ALL detected faces in the candidate thumbnail
            group_eval = compare_group_faces(query_emb, cand_faces)
            best_sim = group_eval["best_similarity"]
            best_cand_face = group_eval["best_candidate_face"]
            cand_quality = best_cand_face["quality"]["overall_quality"]
            is_quality_pass = best_cand_face["quality"]["is_usable"]

            decision, reason = classify_decision(
                best_sim,
                verified_threshold,
                review_threshold,
                is_quality_pass=is_quality_pass,
            )

            all_evaluated_similarities.append(best_sim)
            result_entry = {
                "candidate": c,
                "similarity": best_sim,
                "decision": decision,
                "reason": reason,
                "face_count": group_eval["evaluated_face_count"],
                "best_face_index": group_eval["best_face_index"],
                "det_confidence": best_cand_face["det_score"],
                "image_quality": cand_quality,
                "thumbnail_sha256": thumb_hash,
            }
            verified_results.append(result_entry)
        except Exception:
            continue
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    # ---------------------------------------------------------
    # [6/9] Candidate Ranking & Separation Margin Analysis
    # ---------------------------------------------------------
    print("\n[6/9] Ranking candidates and calculating separation margin")
    decision_priority = {"VERIFIED": 3, "REVIEW": 2, "REJECTED": 1}
    # Deterministic ranking: Decision tier -> Similarity -> Det confidence -> Quality -> Search rank
    verified_results.sort(
        key=lambda r: (
            decision_priority.get(r["decision"], 0),
            r["similarity"],
            r["det_confidence"],
            r["image_quality"],
            -r["candidate"].get("search_rank", 999),
        ),
        reverse=True,
    )

    verified_sims = [r["similarity"] for r in verified_results if r["decision"] == "VERIFIED"]
    rejected_sims = [r["similarity"] for r in verified_results if r["decision"] in ("REVIEW", "REJECTED")]
    margin_data = calculate_separation_margin(verified_sims, rejected_sims)
    sep_margin = margin_data["separation_margin"]

    print("\n  --- DISCOVERY & VERIFICATION SUMMARY ---")
    print(f"  Total Candidates Discovered : {len(candidates)}")
    print(f"  Usable Thumbnails Analyzed  : {usable_images_count}")
    print(f"  Evaluated Matches with Faces: {len(verified_results)}")
    print(f"  VERIFIED Matches            : {len(verified_sims)}")
    print(f"  REVIEW Matches              : {sum(1 for r in verified_results if r['decision'] == 'REVIEW')}")
    print(f"  REJECTED Matches            : {sum(1 for r in verified_results if r['decision'] == 'REJECTED')}")
    if sep_margin is not None:
        print(f"  Best Verified Similarity    : {margin_data['best_verified_similarity']:.4f}")
        print(f"  Best Rejected Similarity    : {margin_data['best_rejected_similarity']:.4f}")
        print(f"  Separation Margin Gap       : {sep_margin:.4f} ({margin_data['margin_interpretation']})")
    print("  ----------------------------------------\n")

    if verified_results:
        print("  Top 5 Evaluated Candidates:")
        for rank_idx, r in enumerate(verified_results[:5], start=1):
            cand = r["candidate"]
            print(f"  [{rank_idx}] Platform: {cand.get('platform')} | Rank #{cand.get('search_rank')}")
            print(f"      URL        : {cand.get('link')}")
            print(f"      Similarity : {r['similarity']:.4f} | Quality: {r['image_quality']:.2f}")
            print(f"      Decision   : [{r['decision']}] - {r['reason']}")

    # Select best candidate
    if not verified_results:
        print("[-] No candidate images could be parsed or contained detectable faces.")
        sys.exit(1)

    # Prioritize verified social media candidates if available
    social_verified = [r for r in verified_results if r["candidate"].get("platform") != "General Web" and r["decision"] == "VERIFIED"]
    if social_verified:
        chosen_match = social_verified[0]
    else:
        chosen_match = verified_results[0]

    cand_data = chosen_match["candidate"]
    cand_sim = chosen_match["similarity"]
    cand_decision = chosen_match["decision"]
    cand_reason = chosen_match["reason"]

    # ---------------------------------------------------------
    # [7/9] Canonical Evidence Manifest Generation (RFC-8785)
    # ---------------------------------------------------------
    print("\n[7/9] Building RFC-8785 canonical evidence manifest and SHA-256 fingerprint")
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
        candidate_image_quality=chosen_match.get("image_quality"),
        query_image_quality=query_analysis["quality_score"],
        selected_face_index=face_index,
        separation_margin=sep_margin,
        margin_interpretation=margin_data.get("margin_interpretation"),
        thumbnail_sha256=chosen_match.get("thumbnail_sha256"),
    )

    print(f"      -> Manifest Schema      : {manifest['schema_version']}")
    print(f"      -> Matched Subject URL  : {cand_data.get('link')}")
    print(f"      -> Platform Category    : {cand_data.get('platform')}")
    print(f"      -> Canonical SHA256 Hash: {manifest_hash}")

    # ---------------------------------------------------------
    # [8/9] Evidence Manifest IPFS Pinning (Pinata)
    # ---------------------------------------------------------
    print("\n[8/9] Pinning evidence manifest to IPFS via Pinata...")
    try:
        manifest_cid = pin_json(manifest, name=f"evidence_proof_{manifest_hash[:12]}")
        manifest_gateway_url = gateway_url(manifest_cid)
        print(f"      -> Evidence IPFS CID    : {manifest_cid}")
        print(f"      -> Gateway URL          : {manifest_gateway_url}")
    except Exception as e:
        print(f"[!] IPFS Manifest Pinning failed: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # [9/9] Polygon Amoy Proof Anchoring & Confirmation
    # ---------------------------------------------------------
    print("\n[9/9] Anchoring proof to Polygon Amoy Testnet (ProofRegistry.sol)...")
    if skip_blockchain:
        print("      -> [SKIPPED] Blockchain anchoring skipped via --skip-blockchain flag.")
        receipt = {"tx_hash": "N/A (offline mode)", "block": 0, "status": "SKIPPED"}
    else:
        try:
            receipt = register_proof(manifest_hash, manifest_cid)
            print(f"      -> Transaction Hash     : {receipt['tx_hash']}")
            print(f"      -> Block Number         : {receipt['block']}")
            print(f"      -> Status               : {receipt['status']}")
            print(f"      -> Contract Address     : {receipt['contract_address']}")
            print(f"      -> Submitter Address    : {receipt['submitter']}")
            print(f"      -> Network              : Polygon Amoy (Chain ID 80002)")
        except Exception as e:
            print(f"[-] Blockchain transaction failed: {e}")
            sys.exit(1)

        print("\n  [+] Querying on-chain proof from Polygon Amoy to verify registry state...")
        try:
            onchain_proof = get_proof(manifest_hash)
            print(f"      -> On-Chain Verification: CONFIRMED")
            print(f"      -> On-Chain Submitter   : {onchain_proof['submitter']}")
            print(f"      -> On-Chain IPFS CID    : {onchain_proof['ipfs_cid']}")
            print(f"      -> On-Chain Timestamp   : {onchain_proof['timestamp']}")
        except Exception as e:
            print(f"[!] On-chain proof lookup failed: {e}")

    # ---------------------------------------------------------
    # Final Output Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("  TRUSTLENS PIPELINE EXECUTION COMPLETE")
    print("=" * 70)
    print(f"  Evidence SHA-256 Hash : {manifest_hash}")
    print(f"  Evidence IPFS CID     : {manifest_cid}")
    if not skip_blockchain:
        print(f"  Polygon Tx Hash       : {receipt['tx_hash']}")
        print(f"  Block Number          : {receipt['block']}")
    print(f"  Selected Match        : {cand_data.get('link')}")
    print(f"  Platform              : {cand_data.get('platform')}")
    print(f"  Face Similarity       : {cand_sim:.4f}")
    print(f"  Decision Status       : [{cand_decision}]")
    if sep_margin is not None:
        print(f"  Separation Margin     : {sep_margin:.4f}")
    print("\n  To re-verify this proof independently anytime:")
    print(f"    python verify.py {manifest_hash}")
    print("\n  To run the cryptographic tamper demonstration:")
    print(f"    python tamper_demo.py {manifest_hash}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TrustLens: Face Identification & Blockchain Verification Pipeline"
    )
    parser.add_argument("image", help="Path to input face-scan image (e.g. demo/sample_face.jpg)")
    parser.add_argument(
        "--face-index",
        type=int,
        default=0,
        help="Index of face to select if input contains multiple faces (default: 0)",
    )
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
        "--min-quality",
        type=float,
        default=DEFAULT_MIN_QUALITY,
        help=f"Minimum quality score threshold for face input (default: {DEFAULT_MIN_QUALITY})",
    )
    parser.add_argument(
        "--skip-blockchain",
        action="store_true",
        help="Skip writing transaction to Polygon Amoy (useful for dry runs / offline testing)",
    )
    args = parser.parse_args()

    run_pipeline(
        image_path=args.image,
        face_index=args.face_index,
        verified_threshold=args.verified_threshold,
        review_threshold=args.review_threshold,
        min_quality=args.min_quality,
        skip_blockchain=args.skip_blockchain,
    )
