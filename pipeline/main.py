"""End-to-end pipeline: face scan -> multi-platform web discovery -> independent face verification -> IPFS -> blockchain proof.

Usage:
    python -m pipeline.main path/to/face.jpg [--face-index 0] [--verified-threshold 0.60] [--review-threshold 0.40] [--min-quality 0.20] [--skip-blockchain]
"""
import argparse
import collections
import concurrent.futures
import os
import sys
import time
from typing import Any, Optional, Union

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import cv2
import numpy as np
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



# Image I/O & Validation Limits
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB maximum thumbnail buffer
MIN_IMAGE_BYTES = 100               # 100 bytes minimum
MIN_IMAGE_DIM = 32                  # 32x32 minimum dimension for usable face detection
MAX_IMAGE_DIM = 4096                # 4096x4096 maximum dimension
DOWNLOAD_CONNECT_TIMEOUT = 3.0      # Fast connect timeout
DOWNLOAD_READ_TIMEOUT = 4.0         # Fast read timeout


def classify_decision(
    similarity: float,
    verified_threshold: float,
    review_threshold: float,
    is_quality_pass: bool = True,
    det_confidence: float = 1.0,
) -> tuple[str, str]:
    """Determine verification decision and explainable rationale code.

    Decision Categories:
    - VERIFIED: Similarity crosses verified threshold (>= 0.60 default), quality passed, and det_confidence >= 0.40.
    - REVIEW: Similarity is near boundary (review <= sim < verified) or high-similarity face with degraded quality/confidence.
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

    if det_confidence < 0.40:
        return (
            "REVIEW",
            f"Face similarity ({similarity:.4f} >= {review_threshold:.2f}) meets candidate baseline, but face detector confidence ({det_confidence:.2f} < 0.40) warrants manual verification",
        )

    if similarity >= verified_threshold:
        return (
            "VERIFIED",
            f"Face similarity ({similarity:.4f} >= {verified_threshold:.2f}) exceeds verified threshold with confirmed quality and detection confidence",
        )
    else:
        return (
            "REVIEW",
            f"Face similarity ({similarity:.4f} >= {review_threshold:.2f}) is near decision boundary; warrants manual inspection",
        )


def _validate_image_bytes(raw_bytes: bytes) -> tuple[bool, Optional[np.ndarray], str]:
    """Validate and decode image byte buffer prior to expensive InsightFace inference."""
    if not raw_bytes or len(raw_bytes) < MIN_IMAGE_BYTES:
        return False, None, "Buffer empty or below minimum size threshold (100 bytes)"
    if len(raw_bytes) > MAX_IMAGE_BYTES:
        return False, None, f"Buffer exceeds 10MB maximum limit ({len(raw_bytes)} bytes)"

    try:
        nparr = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False, None, "Failed to decode image from byte buffer (corrupted or unsupported format)"
        h, w = img.shape[:2]
        if h < MIN_IMAGE_DIM or w < MIN_IMAGE_DIM:
            return False, None, f"Image dimensions ({w}x{h}) below minimum {MIN_IMAGE_DIM}x{MIN_IMAGE_DIM} px threshold"
        if h > MAX_IMAGE_DIM or w > MAX_IMAGE_DIM:
            return False, None, f"Image dimensions ({w}x{h}) exceed maximum {MAX_IMAGE_DIM}x{MAX_IMAGE_DIM} px threshold"
        return True, img, "Valid image"
    except Exception as e:
        return False, None, f"Image validation exception: {str(e)}"


def _fetch_thumbnail_by_url(
    url: str,
    session: requests.Session,
) -> tuple[str, Optional[bytes], bool, str, float]:
    """Download single thumbnail URL using pooled session and early validation.

    Returns:
        (url, raw_bytes, is_valid, status_or_error, latency_seconds)
    """
    t0 = time.perf_counter()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    }
    try:
        resp = session.get(url, headers=headers, timeout=(DOWNLOAD_CONNECT_TIMEOUT, DOWNLOAD_READ_TIMEOUT), stream=False)
        latency = time.perf_counter() - t0
        if resp.status_code != 200:
            return url, None, False, f"HTTP {resp.status_code}", latency
        raw = resp.content
        valid, _, reason = _validate_image_bytes(raw)
        if not valid:
            return url, None, False, reason, latency
        return url, raw, True, "OK", latency
    except requests.exceptions.Timeout:
        return url, None, False, "Connection/read timeout", time.perf_counter() - t0
    except requests.exceptions.RequestException as e:
        return url, None, False, f"Network error: {type(e).__name__}", time.perf_counter() - t0
    except Exception as e:
        return url, None, False, f"Download error: {str(e)}", time.perf_counter() - t0


def evaluate_candidates_concurrently(
    candidates: list[dict[str, Any]],
    query_emb: Any,
    verified_threshold: float,
    review_threshold: float,
    max_workers: int = 8,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    query_multiview: Optional[dict[str, Any]] = None,
    return_telemetry: bool = False,
) -> Union[tuple[list[dict[str, Any]], int], tuple[list[dict[str, Any]], int, dict[str, Any]]]:
    """Concurrently download, deduplicate by content hash, and evaluate candidate images.

    High-Efficiency Architecture:
    1. Early image validation rejects corrupted/empty/sub-dimensional images before InsightFace.
    2. Concurrent downloading uses requests connection pooling with bounded worker pool & fast timeouts.
    3. Content Hash Deduplication: Multiple candidate sources referencing identical image bytes are
       downloaded and processed through InsightFace EXACTLY ONCE.
    4. Provenance Preservation: All candidate source records pointing to a shared image hash retain their
       distinct platform, URL, title, and metadata, while sharing the biometric evaluation.
    5. Two-Stage Verification: Stage A fast pass (face detection + ArcFace cosine similarity),
       Stage B deep checks on top/borderline candidates.
    6. Non-retrievable candidates are classified as UNRETRIEVABLE and retained in the evidence set.
    """
    target_candidates = candidates[:max_candidates]

    url_to_candidates: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    candidates_without_thumb: list[dict[str, Any]] = []

    for c in target_candidates:
        thumb = c.get("thumbnail") or c.get("image")
        if thumb and str(thumb).startswith(("http://", "https://")):
            url_to_candidates[str(thumb).strip()].append(c)
        else:
            candidates_without_thumb.append(c)

    unique_urls = list(url_to_candidates.keys())

    # Concurrent thumbnail download using connection-pooled requests.Session
    t_down_start = time.perf_counter()
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers, max_retries=1)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    url_to_bytes: dict[str, bytes] = {}
    url_to_error: dict[str, str] = {}
    download_latencies: list[float] = []

    if unique_urls:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(_fetch_thumbnail_by_url, url, session): url for url in unique_urls}
            for f in concurrent.futures.as_completed(future_to_url):
                try:
                    url, raw, is_valid, status, lat = f.result()
                    download_latencies.append(lat)
                    if is_valid and raw:
                        url_to_bytes[url] = raw
                    else:
                        url_to_error[url] = status
                except Exception as e:
                    u = future_to_url[f]
                    url_to_error[u] = f"Fetch exception: {str(e)}"

    total_download_time = time.perf_counter() - t_down_start
    download_attempted = len(unique_urls)
    download_successful = len(url_to_bytes)
    download_failed = len(url_to_error)

    # Content-hash based deduplication: Map unique image bytes SHA-256 to candidate records
    hash_to_bytes: dict[str, bytes] = {}
    hash_to_candidates: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)

    for url, raw in url_to_bytes.items():
        h = sha256_of_bytes(raw)
        hash_to_bytes[h] = raw
        for cand in url_to_candidates[url]:
            cand["thumbnail_sha256"] = h
            hash_to_candidates[h].append(cand)

    # Face Analysis — Runs EXACTLY ONCE per unique image content hash
    t_face_start = time.perf_counter()
    hash_to_eval: dict[str, dict[str, Any]] = {}
    total_faces_analyzed = 0

    for h, raw_bytes in hash_to_bytes.items():
        try:
            cand_faces = detect_all_faces(raw_bytes, min_quality=0.15)
            if not cand_faces:
                hash_to_eval[h] = {
                    "similarity": 0.0,
                    "decision": "REJECTED",
                    "reason": "No faces detected in candidate image",
                    "face_count": 0,
                    "best_face_index": 0,
                    "matched_face_id": "NO FACE",
                    "candidate_faces_evaluated": [],
                    "det_confidence": 0.0,
                    "image_quality": 0.0,
                    "best_cand_face": None,
                }
                continue

            total_faces_analyzed += len(cand_faces)

            # Stage A — Fast Pass: Group comparison against query
            group_eval = compare_group_faces(
                query_emb,
                cand_faces,
                query_multiview=query_multiview,
                verified_threshold=verified_threshold,
                review_threshold=review_threshold,
            )
            best_sim = group_eval["best_similarity"]
            best_cand_face = group_eval["best_candidate_face"]
            cand_quality = best_cand_face["quality"]["overall_quality"]
            is_quality_pass = best_cand_face["quality"]["is_usable"]
            det_conf = float(best_cand_face.get("det_score", 1.0))

            # Stage B — Deep Check on boundary and top candidates
            if best_sim >= review_threshold:
                dense = best_cand_face.get("dense_geometry")
                if dense and dense.get("landmark_consistency") == "DEGRADED":
                    if best_sim >= verified_threshold:
                        is_quality_pass = False

            decision, reason = classify_decision(
                best_sim,
                verified_threshold,
                review_threshold,
                is_quality_pass=is_quality_pass,
                det_confidence=det_conf,
            )

            hash_to_eval[h] = {
                "similarity": best_sim,
                "decision": decision,
                "reason": reason,
                "face_count": group_eval["evaluated_face_count"],
                "best_face_index": group_eval["best_face_index"],
                "matched_face_id": group_eval.get("matched_face_id", f"FACE {group_eval['best_face_index'] + 1:02d}"),
                "candidate_faces_evaluated": group_eval.get("candidate_faces_evaluated", []),
                "det_confidence": det_conf,
                "image_quality": cand_quality,
                "best_cand_face": best_cand_face,
            }
        except Exception as e:
            hash_to_eval[h] = {
                "similarity": 0.0,
                "decision": "REJECTED",
                "reason": f"Face inference error: {str(e)}",
                "face_count": 0,
                "best_face_index": 0,
                "matched_face_id": "ERROR",
                "candidate_faces_evaluated": [],
                "det_confidence": 0.0,
                "image_quality": 0.0,
                "best_cand_face": None,
            }

    face_inference_time = time.perf_counter() - t_face_start

    # Replicate evaluated face results across all candidate records pointing to that image
    verified_results: list[dict[str, Any]] = []

    for h, cands in hash_to_candidates.items():
        ev = hash_to_eval.get(h, {
            "similarity": 0.0,
            "decision": "REJECTED",
            "reason": "Unevaluated image",
            "face_count": 0,
            "best_face_index": 0,
            "matched_face_id": "NONE",
            "candidate_faces_evaluated": [],
            "det_confidence": 0.0,
            "image_quality": 0.0,
        })
        for c in cands:
            verified_results.append({
                "candidate": c,
                "similarity": ev["similarity"],
                "decision": ev["decision"],
                "reason": ev["reason"],
                "face_count": ev["face_count"],
                "best_face_index": ev["best_face_index"],
                "matched_face_id": ev["matched_face_id"],
                "candidate_faces_evaluated": ev["candidate_faces_evaluated"],
                "det_confidence": ev["det_confidence"],
                "image_quality": ev["image_quality"],
                "thumbnail_sha256": h,
            })

    # Retain failed download candidates as UNRETRIEVABLE
    for url, err in url_to_error.items():
        for c in url_to_candidates[url]:
            verified_results.append({
                "candidate": c,
                "similarity": 0.0,
                "decision": "REJECTED",
                "reason": f"UNRETRIEVABLE: {err}",
                "face_count": 0,
                "best_face_index": 0,
                "matched_face_id": "UNRETRIEVABLE",
                "candidate_faces_evaluated": [],
                "det_confidence": 0.0,
                "image_quality": 0.0,
                "thumbnail_sha256": None,
            })

    # Retain candidates without thumbnails as UNRETRIEVABLE
    for c in candidates_without_thumb:
        verified_results.append({
            "candidate": c,
            "similarity": 0.0,
            "decision": "REJECTED",
            "reason": "UNRETRIEVABLE: Missing image thumbnail URL",
            "face_count": 0,
            "best_face_index": 0,
            "matched_face_id": "NO IMAGE",
            "candidate_faces_evaluated": [],
            "det_confidence": 0.0,
            "image_quality": 0.0,
            "thumbnail_sha256": None,
        })

    # Retain any remaining candidates beyond max_candidates
    for c in candidates[max_candidates:]:
        verified_results.append({
            "candidate": c,
            "similarity": 0.0,
            "decision": "REJECTED",
            "reason": "EXCEEDED BUDGET: Candidate deferred past maximum evaluation limit",
            "face_count": 0,
            "best_face_index": 0,
            "matched_face_id": "BUDGET LIMIT",
            "candidate_faces_evaluated": [],
            "det_confidence": 0.0,
            "image_quality": 0.0,
            "thumbnail_sha256": None,
        })

    usable_images_count = download_successful

    telemetry = {
        "download_attempted": download_attempted,
        "download_successful": download_successful,
        "download_failed": download_failed,
        "download_time": round(total_download_time, 3),
        "download_avg_latency": round(float(np.mean(download_latencies)) if download_latencies else 0.0, 3),
        "unique_images_analyzed": len(hash_to_bytes),
        "total_faces_analyzed": total_faces_analyzed,
        "face_inference_time": round(face_inference_time, 3),
        "candidates_evaluated": len(verified_results),
        "usable_images_count": usable_images_count,
    }

    if return_telemetry:
        return verified_results, usable_images_count, telemetry
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
    eval_res = evaluate_candidates_concurrently(
        candidates,
        query_emb,
        verified_threshold,
        review_threshold,
        max_workers=8,
        query_multiview=query_analysis.get("multiview"),
        return_telemetry=True,
    )
    verified_results, usable_images_count, eval_telemetry = eval_res
    timings["5_cand_eval"] = time.perf_counter() - t0
    timings["5a_download"] = eval_telemetry["download_time"]
    timings["5b_face_analysis"] = eval_telemetry["face_inference_time"]
    print(f"      -> Concurrent Download  : {eval_telemetry['download_time']:.3f} s ({eval_telemetry['download_successful']}/{eval_telemetry['download_attempted']} successful)")
    print(f"      -> Unique Images Analyzed: {eval_telemetry['unique_images_analyzed']} (Deduplicated from {usable_images_count} downloads)")
    print(f"      -> Total Faces Evaluated: {eval_telemetry['total_faces_analyzed']} in {eval_telemetry['face_inference_time']:.3f} s")
    print(f"      -> Total Candidates Evaluated: {len(verified_results)}")

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

    # Candidate Breakdown Sections
    if verified_matches:
        print("  [+] VERIFIED MATCHES:")
        for idx, r in enumerate(verified_matches, start=1):
            c = r["candidate"]
            print(f"      [{idx}] Platform  : {c.get('platform')}")
            print(f"          URL       : {c.get('link')}")
            print(f"          Similarity: {r['similarity']:.4f} (Quality: {r['image_quality']:.2f})")
            print(f"          Decision  : [{r['decision']}]")
            print(f"          Reason    : {r['reason']}")
    else:
        print("  [-] No candidates met the high-confidence VERIFIED threshold.")

    if review_matches:
        print("\n  [?] REVIEW CANDIDATES (BORDERLINE / AMBIGUOUS):")
        for idx, r in enumerate(review_matches, start=1):
            c = r["candidate"]
            print(f"      [{idx}] Platform  : {c.get('platform')}")
            print(f"          URL       : {c.get('link')}")
            print(f"          Similarity: {r['similarity']:.4f} (Quality: {r['image_quality']:.2f})")
            print(f"          Decision  : [{r['decision']}]")
            print(f"          Reason    : {r['reason']}")

    if rejected_matches:
        print(f"\n  [x] REJECTED CANDIDATES ({len(rejected_matches)} NON-MATCHING EVIDENCE):")
        for idx, r in enumerate(rejected_matches[:8], start=1):
            c = r["candidate"]
            print(f"      [{idx}] Platform  : {c.get('platform')}")
            print(f"          URL       : {c.get('link')}")
            print(f"          Similarity: {r['similarity']:.4f}")
            print(f"          Decision  : [{r['decision']}]")
        if len(rejected_matches) > 8:
            print(f"      ... and {len(rejected_matches) - 8} more rejected candidate(s)")

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

    # Final Search Summary (Requirement 19)
    print("\n" + "=" * 60)
    print("  DISCOVERY COMPLETE")
    print("=" * 60)
    print(f"  Search modes            : {', '.join(search_telemetry.get('search_modes_active', ['visual_matches']))}")
    print(f"  Result sets             : {search_telemetry.get('pages_scanned', 1)}")
    print(f"  Unique sources          : {len(consensus_data.get('distinct_domains', []))}")
    print(f"  Candidate images        : {usable_images_count}")
    print(f"  Faces analyzed          : {len(verified_results)}")
    print(f"  Strongest verified match: {similarity_score:.4f} [{decision}] ({matched_candidate.get('title', 'Discovered Web Identity')})")
    print(f"  Supporting sources      : {consensus_data.get('total_supporting', 0)}")
    print("=" * 60)

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
    t_disc = timings.get("4_search_api", 0.0)
    t_down = timings.get("5a_download", 0.0)
    t_face = timings.get("5b_face_analysis", 0.0)
    t_evid = timings.get("6_ranking", 0.0) + timings.get("7_manifest", 0.0)
    print(f"\n  RUNTIME TELEMETRY:")
    print(f"  DISCOVERY: {t_disc:.2f}s | DOWNLOAD: {t_down:.2f}s | FACE ANALYSIS: {t_face:.2f}s | EVIDENCE: {t_evid:.2f}s | TOTAL: {total_latency:.2f}s")
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
        "evaluation_telemetry": eval_telemetry,
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
