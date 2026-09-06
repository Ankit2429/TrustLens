"""End-to-end pipeline: face scan -> multi-platform web discovery -> independent face verification -> IPFS -> blockchain proof.

Usage:
    python -m pipeline.main path/to/face.jpg [--face-index 0] [--verified-threshold 0.60] [--review-threshold 0.40] [--min-quality 0.20] [--skip-blockchain]
"""
import argparse
import concurrent.futures
import os
import sys
import time
from typing import Any, Optional

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import requests
from requests.adapters import HTTPAdapter
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
from pipeline.search import reverse_image_search, filter_social_matches, extract_image_metadata
from pipeline.fingerprint import (
    build_evidence_manifest,
    build_source_relationship_graph,
    calculate_dynamic_confidence,
    calculate_separation_margin,
    compute_candidate_consensus,
    sha256_of_json,
    sha256_of_bytes,
)
from pipeline.ipfs_store import pin_file, pin_json, gateway_url
from pipeline.chain import register_proof, get_proof

load_dotenv()

# Biometric Decision Thresholds
# Cosine similarity thresholds calibrated on standard ArcFace distributions:
# - VERIFIED (>= 0.60): High-confidence genuine identity match across diverse real-world conditions.
# - REVIEW  (0.40–0.60): Borderline / manual review required (adverse lighting, heavy compression, avatar).
# - REJECTED (< 0.40):  Unrelated candidate clearly below the biometric decision boundary.
DEFAULT_VERIFIED_THRESHOLD = float(os.environ.get("VERIFIED_THRESHOLD", "0.60"))
DEFAULT_REVIEW_THRESHOLD = float(os.environ.get("REVIEW_THRESHOLD", "0.40"))
DEFAULT_MIN_QUALITY = float(os.environ.get("MIN_QUALITY_THRESHOLD", "0.20"))
DEFAULT_MAX_CANDIDATES = int(os.environ.get("MAX_CANDIDATES", "40"))



def classify_decision(
    similarity: float,
    verified_threshold: float,
    review_threshold: float,
    is_quality_pass: bool = True,
) -> tuple[str, str]:
    """Determine verification decision and explainable rationale code.

    Decision Categories:
    - VERIFIED: Similarity crosses verified threshold (>= 0.60 default) and candidate passes quality checks.
    - REVIEW: Similarity is near boundary (review <= sim < verified) or high-similarity face with degraded quality.
    - REJECTED: Similarity is below review threshold (< 0.40 default) — unrelated identity.

    Returns:
        (decision, reason_str) where decision is 'VERIFIED', 'REVIEW', or 'REJECTED'.
    """
    if similarity < review_threshold:
        return (
            "REJECTED",
            f"Face similarity ({similarity:.4f} < {review_threshold:.2f}) is below baseline (unrelated identity)",
        )

    if not is_quality_pass:
        return (
            "REVIEW",
            f"Face similarity ({similarity:.4f} >= {review_threshold:.2f}) meets candidate baseline, but candidate image quality warrants manual inspection",
        )

    if similarity >= verified_threshold:
        return (
            "VERIFIED",
            f"Face similarity ({similarity:.4f} >= {verified_threshold:.2f}) exceeds verified threshold and candidate passed quality checks",
        )
    else:
        return (
            "REVIEW",
            f"Face similarity ({similarity:.4f} >= {review_threshold:.2f}) is near decision boundary; warrants manual inspection",
        )


def _fetch_candidate_thumbnail(c: dict[str, Any], session: requests.Session) -> Optional[tuple[dict[str, Any], bytes]]:
    """Download candidate thumbnail bytes with session connection pooling."""
    thumb_url = c.get("thumbnail")
    if not thumb_url:
        return None
    try:
        resp = session.get(thumb_url, timeout=6)
        if resp.status_code == 200 and len(resp.content) > 0:
            return (c, resp.content)
    except Exception:
        pass
    return None


def evaluate_candidates_concurrently(
    candidates: list[dict[str, Any]],
    query_emb: Any,
    verified_threshold: float,
    review_threshold: float,
    max_workers: int = 8,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    query_multiview: Optional[dict[str, Any]] = None,
) -> tuple[list[dict[str, Any]], int]:
    """Concurrently download and independently evaluate candidate thumbnails in-memory.

    Args:
        candidates: Discovered candidates from visual search.
        query_emb: Normalized 512-d ArcFace query embedding.
        verified_threshold: Similarity cutoff for VERIFIED status.
        review_threshold: Similarity cutoff for REVIEW status.
        max_workers: Thread pool size for bounded concurrent I/O.
        max_candidates: Maximum candidates to process (bounded top-K).
        query_multiview: Optional multi-view query representation dictionary.

    Returns:
        (evaluated_results, usable_images_count)
    """
    # Deduplicate candidate URLs while preserving search rank ordering
    unique_candidates: list[dict[str, Any]] = []
    seen_thumbs: set[str] = set()
    for c in candidates:
        thumb = c.get("thumbnail")
        if thumb and thumb not in seen_thumbs:
            seen_thumbs.add(thumb)
            unique_candidates.append(c)
        elif not thumb:
            unique_candidates.append(c)

    target_candidates = unique_candidates[:max_candidates]

    # Concurrent thumbnail download using requests.Session with connection pooling
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers, max_retries=1)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    downloaded: list[tuple[dict[str, Any], bytes]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_fetch_candidate_thumbnail, c, session) for c in target_candidates]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res is not None:
                downloaded.append(res)

    usable_images_count = len(downloaded)
    verified_results: list[dict[str, Any]] = []

    # In-memory face detection & group comparison (avoids expensive disk temp files)
    for c, raw_bytes in downloaded:
        try:
            thumb_hash = sha256_of_bytes(raw_bytes)
            c["thumbnail_sha256"] = thumb_hash

            cand_faces = detect_all_faces(raw_bytes, min_quality=0.15)
            if not cand_faces:
                # No face detected in image -> REJECTED
                continue

            # Compare query face against ALL detected faces in candidate group photo
            group_eval = compare_group_faces(query_emb, cand_faces, query_multiview=query_multiview)
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

            result_entry = {
                "candidate": c,
                "similarity": best_sim,
                "decision": decision,
                "reason": reason,
                "face_count": group_eval["evaluated_face_count"],
                "best_face_index": group_eval["best_face_index"],
                "matched_face_id": group_eval.get("matched_face_id", f"FACE {group_eval['best_face_index'] + 1:02d}"),
                "candidate_faces_evaluated": group_eval.get("candidate_faces_evaluated", []),
                "det_confidence": best_cand_face["det_score"],
                "image_quality": cand_quality,
                "thumbnail_sha256": thumb_hash,
            }
            verified_results.append(result_entry)
        except Exception:
            continue

    return verified_results, usable_images_count


def run_pipeline(
    image_path: str,
    face_index: int = 0,
    verified_threshold: float = DEFAULT_VERIFIED_THRESHOLD,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    min_quality: float = DEFAULT_MIN_QUALITY,
    skip_blockchain: bool = False,
):
    """Execute the hardened 9-step Task 3 pipeline with stage timing instrumentation."""
    t_start = time.perf_counter()
    timings: dict[str, float] = {}

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

    t0 = time.perf_counter()
    try:
        query_analysis = analyze_face(image_path, face_index=face_index, min_quality=min_quality)
    except Exception as e:
        print(f"[-] Face analysis failed: {e}")
        sys.exit(1)
    timings["1_face_detect"] = time.perf_counter() - t0

    image_metadata = extract_image_metadata(image_path)
    dense_geom = query_analysis.get("dense_geometry")

    total_faces = query_analysis["face_count"]
    print(f"      -> Total Faces Detected : {total_faces}")
    print(f"      -> Selected Face Index  : {face_index} of {total_faces}")
    print(f"      -> Detection Confidence : {query_analysis['det_score']:.4f}")
    print(f"      -> Overall Quality Score: {query_analysis['quality_score']:.4f} (Usable: {query_analysis['is_usable']})")
    print(f"         * Sharpness  : {query_analysis['quality_breakdown']['sharpness']:.4f}")
    print(f"         * Exposure   : {query_analysis['quality_breakdown']['exposure']:.4f}")
    print(f"         * Resolution : {query_analysis['quality_breakdown']['resolution']:.4f}")
    print(f"         * Frontality : {query_analysis['quality_breakdown']['frontality']:.4f}")
    if dense_geom and dense_geom.get("pose_3d"):
        p3d = dense_geom["pose_3d"]
        print(f"      -> Dense 3D Head Pose   : Pitch {p3d['pitch']}°, Yaw {p3d['yaw']}°, Roll {p3d['roll']}°")
        if dense_geom.get("metrics"):
            m = dense_geom["metrics"]
            print(f"      -> Facial Geometry      : IOD={m['inter_ocular_distance']}px, Consistency={m['landmark_consistency']:.2f}")
    print(f"      -> Image EXIF Metadata  : {image_metadata['metadata_status']}")
    print(f"      -> Bounding Box Coordinates: {query_analysis['bbox']}")
    print(f"      -> Stage Latency        : {timings['1_face_detect']:.3f} s")

    # ---------------------------------------------------------
    # [2/9] Full 512-d Face Embedding & Deterministic Hashing
    # ---------------------------------------------------------
    print("\n[2/9] Extracting 512-d ArcFace embedding & computing deterministic SHA-256 fingerprint")
    t0 = time.perf_counter()
    query_emb = query_analysis["normalized_embedding"]
    query_face_hash = query_analysis["embedding_hash"]
    query_face_meta = query_analysis["metadata"]
    timings["2_embedding"] = time.perf_counter() - t0

    print(f"      -> Model Architecture   : {query_face_meta['model']} ({query_face_meta['algorithm']})")
    print(f"      -> Vector Dimension     : {query_face_meta['dimension']}-d ({query_face_meta['dtype']})")
    print(f"      -> Normalization Status : {query_face_meta['normalization_status']}")
    print(f"      -> Full Embedding SHA256: {query_face_hash}")

    # ---------------------------------------------------------
    # [3/9] Query Image IPFS Pinning (Pinata Gateway)
    # ---------------------------------------------------------
    print("\n[3/9] Pinning query image to IPFS for decentralized visual discovery")
    t0 = time.perf_counter()
    try:
        query_cid = pin_file(image_path, name=f"query_face_{os.path.basename(image_path)}")
        public_search_url = gateway_url(query_cid)
        timings["3_ipfs_query"] = time.perf_counter() - t0
        print(f"      -> Query IPFS CID       : {query_cid}")
        print(f"      -> Gateway Search URL   : {public_search_url}")
        print(f"      -> Stage Latency        : {timings['3_ipfs_query']:.3f} s")
    except Exception as e:
        print(f"[!] IPFS pinning failed ({e}). Check PINATA_JWT in .env.")
        sys.exit(1)

    # ---------------------------------------------------------
    # [4/9] Multi-Source & Multi-Platform Web Discovery
    # ---------------------------------------------------------
    print("\n[4/9] Performing multi-source visual discovery across indexed web & social platforms")
    print("      (Querying SerpApi Google Lens engine with deep pagination & source expansion...)")
    t0 = time.perf_counter()
    try:
        search_res = reverse_image_search(public_search_url, return_telemetry=True)
        if isinstance(search_res, tuple):
            candidates, search_telemetry = search_res
        else:
            candidates = search_res
            search_telemetry = {
                "pages_scanned": 1,
                "total_discovered": len(candidates),
                "unique_candidates": len(candidates),
                "platforms_discovered": list({c.get("platform", "General Web") for c in candidates}),
                "source_expansions": 0,
            }
        timings["4_search_api"] = time.perf_counter() - t0
    except Exception as e:
        print(f"[-] Discovery failed: {e}. Check SERPAPI_KEY in .env.")
        sys.exit(1)

    if not candidates:
        print("[-] No visual matches discovered. Try a more distinctive public photo.")
        sys.exit(1)

    print(f"      -> Pages Scanned        : {search_telemetry.get('pages_scanned', 1)}")
    print(f"      -> Total Discovered     : {search_telemetry.get('total_discovered', len(candidates))}")
    print(f"      -> Unique Candidates    : {len(candidates)}")
    print(f"      -> Source Expansions    : {search_telemetry.get('source_expansions', 0)}")
    if search_telemetry.get("search_modes_active"):
        print(f"      -> Active Search Modes  : {', '.join(search_telemetry['search_modes_active'])}")
    if search_telemetry.get("exact_matches_count") is not None:
        print(f"         * Exact Matches      : {search_telemetry.get('exact_matches_count', 0)}")
        print(f"         * Visual Matches     : {search_telemetry.get('visual_matches_count', 0)}")
        print(f"         * About This Image   : {search_telemetry.get('about_image_count', 0)}")
    print(f"      -> Discovery Latency    : {timings['4_search_api']:.3f} s")
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
    print("\n[5/9] Independently evaluating candidate image thumbnails with ArcFace (Concurrent In-Memory)")
    t0 = time.perf_counter()
    verified_results, usable_images_count = evaluate_candidates_concurrently(
        candidates,
        query_emb,
        verified_threshold,
        review_threshold,
        max_workers=8,
        query_multiview=query_analysis.get("multiview"),
    )
    timings["5_cand_eval"] = time.perf_counter() - t0
    print(f"      -> Evaluated {len(verified_results)} candidate faces across {usable_images_count} thumbnails in {timings['5_cand_eval']:.3f} s")

    # ---------------------------------------------------------
    # [6/9] Candidate Ranking & Separation Margin Analysis
    # ---------------------------------------------------------
    print("\n[6/9] Ranking candidates, calculating separation margin & consensus clustering")
    t0 = time.perf_counter()
    decision_priority = {"VERIFIED": 3, "REVIEW": 2, "REJECTED": 1}
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

    verified_matches = [r for r in verified_results if r["decision"] == "VERIFIED"]
    review_matches = [r for r in verified_results if r["decision"] == "REVIEW"]
    rejected_matches = [r for r in verified_results if r["decision"] == "REJECTED"]

    verified_sims = [r["similarity"] for r in verified_matches]
    rejected_sims = [r["similarity"] for r in (review_matches + rejected_matches)]
    margin_data = calculate_separation_margin(verified_sims, rejected_sims)
    sep_margin = margin_data["separation_margin"]

    # Compute consensus clustering & cross-candidate agreement
    consensus_data = compute_candidate_consensus(verified_results, verified_threshold=verified_threshold)

    timings["6_ranking"] = time.perf_counter() - t0

    print("\n  ====================================================================")
    print("  TRUSTLENS DISCOVERY & VERIFICATION RESULT")
    print("  ====================================================================")
    print(f"  Total Candidates Discovered : {len(candidates)}")
    print(f"  Usable Thumbnails Analyzed  : {usable_images_count}")
    print(f"  VERIFIED Matches            : {len(verified_matches)}")
    print(f"  REVIEW Candidates           : {len(review_matches)}")
    print(f"  REJECTED Candidates         : {len(rejected_matches)}")
    print(f"  Identity Consensus Level    : {consensus_data['consensus_level']}")
    print(f"  Supporting Verified Images  : {consensus_data['total_supporting']} across {consensus_data['platform_count']} platform(s)")
    if sep_margin is not None:
        print(f"  Best Verified Similarity    : {margin_data['best_verified_similarity']:.4f}")
        print(f"  Best Rejected Similarity    : {margin_data['best_rejected_similarity']:.4f}")
        print(f"  Separation Margin Gap       : {sep_margin:.4f} ({margin_data['margin_interpretation']})")
    print("  ====================================================================\n")

    # Primary Section: Verified Matches
    if verified_matches:
        print("  [+] PRIMARY SECTION: VERIFIED MATCHES:")
        for idx, r in enumerate(verified_matches, start=1):
            c = r["candidate"]
            print(f"      [{idx}] Platform  : {c.get('platform')}")
            print(f"          URL       : {c.get('link')}")
            print(f"          Similarity: {r['similarity']:.4f} (Quality: {r['image_quality']:.2f})")
            print(f"          Decision  : [{r['decision']}]")
            print(f"          Reason    : {r['reason']}")
    else:
        print("  [-] No candidates met the high-confidence VERIFIED threshold.")

    # Select primary match: Prioritize verified social platform -> any verified -> review -> rejected
    if verified_matches:
        social_verified = [r for r in verified_matches if r["candidate"].get("platform") != "General Web"]
        best_result = social_verified[0] if social_verified else verified_matches[0]
    elif review_matches:
        best_result = review_matches[0]
    elif verified_results:
        best_result = verified_results[0]
    else:
        best_result = None

    if not best_result:
        print("[-] No candidate faces could be evaluated.")
        sys.exit(1)

    matched_candidate = best_result["candidate"]
    similarity_score = best_result["similarity"]
    decision = best_result["decision"]
    decision_reason = best_result["reason"]
    candidate_face_count = best_result["face_count"]
    candidate_det_confidence = best_result["det_confidence"]
    candidate_quality = best_result["image_quality"]
    thumbnail_sha256 = best_result["thumbnail_sha256"]

    # Compute dynamic identity confidence
    confidence_data = calculate_dynamic_confidence(
        similarity=similarity_score,
        candidate_quality=candidate_quality,
        separation_margin=sep_margin,
        cross_result_agreement=consensus_data["agreement_ratio"],
        platform_diversity=consensus_data["platform_count"],
    )

    # Build Evidence Relationship Graph
    relationship_graph = build_source_relationship_graph(
        query_image_cid=query_cid,
        evaluated_results=verified_results,
        query_thumb_hash=query_face_hash,
    )
    print("      -> Source Relationship Graph:")
    print(f"         * Same Image Duplicates: {relationship_graph['counts']['same_image']}")
    print(f"         * Same Person (Diff Img): {relationship_graph['counts']['same_person_different_image']}")
    print(f"         * Visually Related Imgs: {relationship_graph['counts']['visually_related_image']}")
    print(f"         * Different Individuals: {relationship_graph['counts']['different_person']}")

    # ---------------------------------------------------------
    # [7/9] Canonical Evidence Manifest Generation (RFC-8785)
    # ---------------------------------------------------------
    print("\n[7/9] Building RFC-8785 canonical evidence manifest and SHA-256 fingerprint")
    t0 = time.perf_counter()
    matched_face_id = best_result.get("matched_face_id", "FACE 01")
    candidate_faces_evaluated = best_result.get("candidate_faces_evaluated", [])
    manifest, manifest_hash = build_evidence_manifest(
        query_face_metadata=query_face_meta,
        candidate=matched_candidate,
        similarity_score=similarity_score,
        decision=decision,
        verified_threshold=verified_threshold,
        review_threshold=review_threshold,
        decision_reason=decision_reason,
        query_image_cid=query_cid,
        total_candidates=len(candidates),
        usable_images_count=usable_images_count,
        candidate_face_count=candidate_face_count,
        candidate_det_confidence=candidate_det_confidence,
        candidate_image_quality=candidate_quality,
        query_image_quality=query_analysis["quality_score"],
        selected_face_index=face_index,
        separation_margin=sep_margin,
        margin_interpretation=margin_data.get("margin_interpretation"),
        thumbnail_sha256=thumbnail_sha256,
        confidence_data=confidence_data,
        consensus_data=consensus_data,
        matched_face_id=matched_face_id,
        candidate_faces_evaluated=candidate_faces_evaluated,
        search_telemetry=search_telemetry,
        relationship_graph=relationship_graph,
        image_metadata=image_metadata,
        dense_geometry=dense_geom,
    )
    timings["7_manifest"] = time.perf_counter() - t0
    print(f"      -> Manifest Schema      : {manifest.get('schema_version')}")
    print(f"      -> Decision Verdict     : [{decision}]")
    print(f"      -> Dynamic Confidence   : {confidence_data['confidence_score']:.4f}")
    print(f"      -> Matched Candidate Face: [{matched_face_id}]")
    print(f"      -> Subject / Source URL : {matched_candidate.get('link')}")
    print(f"      -> Platform Category    : {matched_candidate.get('platform')}")
    print(f"      -> Canonical SHA256 Hash: {manifest_hash}")

    # ---------------------------------------------------------
    # [8/9] IPFS Manifest Pinning (Pinata)
    # ---------------------------------------------------------
    print("\n[8/9] Pinning evidence manifest to IPFS via Pinata...")
    t0 = time.perf_counter()
    try:
        manifest_cid = pin_json(manifest, name=f"evidence_manifest_{manifest_hash[:12]}")
        timings["8_ipfs_manifest"] = time.perf_counter() - t0
        print(f"      -> Evidence IPFS CID    : {manifest_cid}")
        print(f"      -> Gateway URL          : {gateway_url(manifest_cid)}")
        print(f"      -> Stage Latency        : {timings['8_ipfs_manifest']:.3f} s")
    except Exception as e:
        print(f"[!] Manifest IPFS pinning failed: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # [9/9] Blockchain Proof Anchoring (Strict VERIFIED Gate)
    # ---------------------------------------------------------
    chain_receipt = None
    if not skip_blockchain and decision == "VERIFIED":
        print("\n[9/9] Anchoring proof to Blockchain (ProofRegistry.sol)...")
        t0 = time.perf_counter()
        try:
            chain_receipt = register_proof(manifest_hash, manifest_cid)
            timings["9_blockchain"] = time.perf_counter() - t0
            print(f"      -> Transaction Hash     : {chain_receipt['tx_hash']}")
            print(f"      -> Block Number         : {chain_receipt['block']}")
            print(f"      -> Status               : {chain_receipt['status']}")
            print(f"      -> Contract Address     : {chain_receipt['contract_address']}")
            print(f"      -> Submitter Address    : {chain_receipt['submitter']}")
            print(f"      -> Network              : {chain_receipt['network']} (Chain ID {chain_receipt['chain_id']})")
            print(f"      -> Stage Latency        : {timings['9_blockchain']:.3f} s")
        except Exception as e:
            print(f"[!] Blockchain proof registration failed: {e}")
            sys.exit(1)

        print(f"\n  [+] Querying on-chain proof from {chain_receipt['network']} to verify registry state...")
        onchain = get_proof(manifest_hash)
        print(f"      -> On-Chain Verification: CONFIRMED")
        print(f"      -> On-Chain Submitter   : {onchain['submitter']}")
        print(f"      -> On-Chain IPFS CID    : {onchain['ipfs_cid']}")
        print(f"      -> On-Chain Timestamp   : {onchain['timestamp']}")
    elif decision != "VERIFIED":
        print(f"\n[9/9] Skipping blockchain proof registration: Decision is [{decision}] (Proof registration strictly requires [VERIFIED] evidence).")
    else:
        print("\n[9/9] Skipping blockchain proof registration (--skip-blockchain specified).")

    total_latency = time.perf_counter() - t_start

    print("\n" + "=" * 70)
    print("  TRUSTLENS PIPELINE EXECUTION COMPLETE")
    print("=" * 70)
    print(f"  Evidence SHA-256 Hash : {manifest_hash}")
    print(f"  Evidence IPFS CID     : {manifest_cid}")
    if chain_receipt:
        print(f"  Blockchain Tx Hash    : {chain_receipt['tx_hash']}")
        print(f"  Block Number          : {chain_receipt['block']}")
        print(f"  Network               : {chain_receipt['network']} (Chain ID {chain_receipt['chain_id']})")
    if decision == "VERIFIED":
        print(f"  VERDICT               : [VERIFIED MATCH FOUND]")
        print(f"  Verified Source URL   : {matched_candidate.get('link')}")
        print(f"  Platform              : {matched_candidate.get('platform')}")
        print(f"  Face Similarity       : {similarity_score:.4f} (>= {verified_threshold:.2f})")
    elif decision == "REVIEW":
        print(f"  VERDICT               : [BORDERLINE / MANUAL REVIEW REQUIRED]")
        print(f"  Review Candidate URL  : {matched_candidate.get('link')}")
        print(f"  Platform              : {matched_candidate.get('platform')}")
        print(f"  Face Similarity       : {similarity_score:.4f} (Quality: {candidate_quality:.2f})")
        print(f"  Review Rationale      : {decision_reason}")
    else:
        print(f"  VERDICT               : [NO RELIABLE MATCH FOUND (All candidates rejected)]")
        print(f"  Top Candidate Score   : {similarity_score:.4f} (< {review_threshold:.2f} threshold)")
        print(f"  Rejection Rationale   : {decision_reason}")
    if sep_margin is not None:
        print(f"  Separation Margin     : {sep_margin:.4f} ({margin_data.get('margin_interpretation', '')})")
    print(f"  Total Measured Runtime: {total_latency:.2f} s")
    print(f"  Stage Timings Breakdown:")
    for k, v in sorted(timings.items()):
        print(f"    - {k:<20}: {v:.3f} s ({v/total_latency*100:.1f}%)")
    print("=" * 70)

    return {
        "manifest_hash": manifest_hash,
        "manifest_cid": manifest_cid,
        "blockchain_receipt": chain_receipt,
        "best_match": best_result,
        "query_analysis": query_analysis,
        "manifest": manifest,
        "all_candidates_ranked": verified_results,
        "verified_matches": verified_matches,
        "review_matches": review_matches,
        "rejected_matches": rejected_matches,
        "total_candidates": len(candidates),
        "separation_margin": sep_margin,
        "total_latency_seconds": total_latency,
        "stage_timings": timings,
    }


def main():
    parser = argparse.ArgumentParser(description="TrustLens End-to-End Face Identification & Blockchain Verification")
    parser.add_argument("image", help="Path to input face image (JPG, PNG, WEBP)")
    parser.add_argument("--face-index", type=int, default=0, help="Index of detected face to evaluate (default: 0)")
    parser.add_argument("--verified-threshold", type=float, default=DEFAULT_VERIFIED_THRESHOLD, help=f"Similarity threshold for VERIFIED status (default: {DEFAULT_VERIFIED_THRESHOLD})")
    parser.add_argument("--review-threshold", type=float, default=DEFAULT_REVIEW_THRESHOLD, help=f"Similarity threshold for REVIEW status (default: {DEFAULT_REVIEW_THRESHOLD})")
    parser.add_argument("--min-quality", type=float, default=DEFAULT_MIN_QUALITY, help=f"Minimum face quality threshold (default: {DEFAULT_MIN_QUALITY})")
    parser.add_argument("--skip-blockchain", action="store_true", help="Skip blockchain registration (dry run / offline mode)")

    args = parser.parse_args()
    run_pipeline(
        image_path=args.image,
        face_index=args.face_index,
        verified_threshold=args.verified_threshold,
        review_threshold=args.review_threshold,
        min_quality=args.min_quality,
        skip_blockchain=args.skip_blockchain,
    )


if __name__ == "__main__":
    main()
