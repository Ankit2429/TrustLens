"""Build canonical evidence manifests and calculate cryptographic SHA-256 fingerprints.

Follows RFC-8785 canonical JSON representation (lexicographically sorted keys,
compact separators, deterministic UTF-8 byte encoding) to ensure identical records
always produce bit-for-bit identical hashes across all environments.
"""
import hashlib
import json
import time
from typing import Any, Optional


def _normalize_numbers(obj: Any) -> Any:
    """Normalize numeric types in accordance with RFC-8785 Section 3.2.2.3.

    Integer values (including floats with no fractional part like 1.0 or 0.0)
    must be formatted without a fractional part (e.g. 1 instead of 1.0).
    """
    if isinstance(obj, float) and obj.is_integer():
        return int(obj)
    if isinstance(obj, dict):
        return {k: _normalize_numbers(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize_numbers(v) for v in obj]
    return obj


def canonical_json_bytes(obj: Any) -> bytes:
    """Serialize any Python dictionary or data structure into canonical JSON UTF-8 bytes.

    - Lexicographically sorted keys
    - RFC-8785 integer-float canonical representation
    - Compact separators (no extra whitespace: `(`,`:`)`)
    - Deterministic UTF-8 encoding
    """
    return json.dumps(
        _normalize_numbers(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_of_json(obj: Any) -> str:
    """Compute deterministic SHA-256 hex digest of a canonical JSON object."""
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def sha256_of_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest of raw binary bytes."""
    return hashlib.sha256(data).hexdigest()


def calculate_separation_margin(
    verified_similarities: list[float],
    rejected_similarities: list[float],
) -> dict[str, Any]:
    """Calculate the separation margin between the best verified match and the strongest rejected/non-verified candidate.

    Args:
        verified_similarities: List of similarity scores for candidates meeting verified threshold.
        rejected_similarities: List of similarity scores for non-verified / rejected candidates.

    Returns:
        Dictionary with best_verified, best_nonverified, separation_margin, is_suspicious_margin, and interpretation.
    """
    best_ver = max(verified_similarities) if verified_similarities else 0.0
    best_rej = max(rejected_similarities) if rejected_similarities else 0.0
    margin = round(best_ver - best_rej, 4) if (verified_similarities and rejected_similarities) else None
    is_suspicious = False

    if margin is not None:
        if margin >= 0.30:
            interp = f"Clear separation ({margin:.4f} >= 0.30): Strong differentiation between matching subject and non-matching candidates"
        elif margin >= 0.10:
            interp = f"Moderate separation ({margin:.4f} >= 0.10): Meaningful differentiation observed across evaluated candidates"
        else:
            is_suspicious = True
            interp = f"Suspicious / Narrow margin ({margin:.4f} < 0.10): Boundary risk between verified match ({best_ver:.4f}) and non-verified candidate ({best_rej:.4f})"
    else:
        interp = "Single-tier distribution: Margin calculation requires both verified and non-matching candidates"

    return {
        "best_verified_similarity": round(best_ver, 4) if verified_similarities else None,
        "best_rejected_similarity": round(best_rej, 4) if rejected_similarities else None,
        "best_nonverified_similarity": round(best_rej, 4) if rejected_similarities else None,
        "separation_margin": margin,
        "is_suspicious_margin": is_suspicious,
        "margin_interpretation": interp,
    }


def calculate_dynamic_confidence(
    similarity: float,
    candidate_quality: float = 0.80,
    separation_margin: Optional[float] = None,
    cross_result_agreement: float = 1.0,
    platform_diversity: int = 1,
) -> dict[str, Any]:
    """Compute a multi-signal identity confidence score.

    Formula:
        Confidence = 0.45 * similarity_norm
                   + 0.15 * quality_score
                   + 0.15 * margin_norm
                   + 0.15 * cross_result_agreement
                   + 0.10 * platform_diversity_factor

    Where:
        - similarity_norm: max(0, min(1, similarity / 0.90))
        - quality_score: candidate image quality [0.0, 1.0]
        - margin_norm: max(0, min(1, (margin or 0.20) / 0.40))
        - cross_result_agreement: fraction of top supporting matches [0.0, 1.0]
        - platform_diversity_factor: min(1.0, platform_diversity / 2.0)

    Returns:
        Dictionary containing overall confidence in [0.0, 1.0] and detailed breakdown.
    """
    sim_norm = max(0.0, min(1.0, similarity / 0.90))
    qual_norm = max(0.0, min(1.0, candidate_quality))
    eff_margin = 0.20 if separation_margin is None else max(0.0, separation_margin)
    margin_norm = max(0.0, min(1.0, eff_margin / 0.40))
    agree_norm = max(0.0, min(1.0, cross_result_agreement))
    plat_norm = min(1.0, max(0.5, platform_diversity / 2.0))

    score = (
        0.45 * sim_norm
        + 0.15 * qual_norm
        + 0.15 * margin_norm
        + 0.15 * agree_norm
        + 0.10 * plat_norm
    )
    confidence = round(float(max(0.0, min(1.0, score))), 4)

    return {
        "confidence_score": confidence,
        "breakdown": {
            "similarity_norm": round(sim_norm, 4),
            "quality_factor": round(qual_norm, 4),
            "separation_factor": round(margin_norm, 4),
            "agreement_factor": round(agree_norm, 4),
            "platform_diversity_factor": round(plat_norm, 4),
        },
        "formula": "0.45*sim_norm + 0.15*qual + 0.15*margin + 0.15*agreement + 0.10*platform_diversity",
    }


def compute_candidate_consensus(
    evaluated_results: list[dict[str, Any]],
    verified_threshold: float = 0.60,
) -> dict[str, Any]:
    """Analyze cluster convergence and multi-source consensus across evaluated candidate results.

    Evaluates:
    - total_supporting_verified: Count of candidate images with similarity >= verified_threshold.
    - distinct_supporting_platforms: Set of unique platforms confirming the subject.
    - agreement_ratio: Ratio of verified candidate images to total evaluated images.
    - consensus_level: 'STRONG_CONSENSUS', 'MODERATE_CONSENSUS', or 'ISOLATED_MATCH'.
    - mean_verified_similarity: Average similarity among verified matches.
    - median_verified_similarity: Median similarity among verified matches.
    - score_variance: Variance across verified scores (stability metric).
    - consensus_strength: 'STRONG', 'MODERATE', 'WEAK', or 'NONE'.
    """
    if not evaluated_results:
        return {
            "total_supporting": 0,
            "distinct_platforms": [],
            "distinct_domains": [],
            "agreement_ratio": 0.0,
            "consensus_level": "NO_EVALUATED_CANDIDATES",
            "consensus_strength": "NONE",
            "mean_verified_similarity": 0.0,
            "median_verified_similarity": 0.0,
            "score_variance": 0.0,
        }

    verified_candidates = [r for r in evaluated_results if r.get("similarity", 0.0) >= verified_threshold]
    total_eval = len(evaluated_results)
    supp_count = len(verified_candidates)
    platforms = sorted(list({r.get("candidate", {}).get("platform", "General Web") for r in verified_candidates}))
    domains = sorted(list({r.get("candidate", {}).get("domain", "") for r in verified_candidates if r.get("candidate", {}).get("domain")}))

    ratio = round(supp_count / float(max(1, total_eval)), 4)

    verified_sims = [float(r.get("similarity", 0.0)) for r in verified_candidates]
    if verified_sims:
        mean_sim = round(float(sum(verified_sims) / len(verified_sims)), 4)
        median_sim = round(float(sorted(verified_sims)[len(verified_sims) // 2]), 4)
        variance = round(float(sum((s - mean_sim) ** 2 for s in verified_sims) / len(verified_sims)), 6)
    else:
        mean_sim = 0.0
        median_sim = 0.0
        variance = 0.0

    if supp_count >= 3 and len(domains) >= 2:
        level = "STRONG_MULTI_PLATFORM_CONSENSUS"
        strength = "STRONG"
    elif supp_count >= 2:
        level = "MODERATE_REPEATED_CONSENSUS"
        strength = "MODERATE"
    elif supp_count == 1:
        level = "ISOLATED_SINGLE_MATCH"
        strength = "WEAK"
    else:
        level = "NO_VERIFIED_CONSENSUS"
        strength = "NONE"

    return {
        "total_supporting": supp_count,
        "distinct_platforms": platforms,
        "platform_count": len(platforms),
        "distinct_domains": domains,
        "domain_count": len(domains),
        "agreement_ratio": ratio,
        "consensus_level": level,
        "consensus_strength": strength,
        "mean_verified_similarity": mean_sim,
        "median_verified_similarity": median_sim,
        "score_variance": variance,
    }


def classify_source_relationship(
    similarity: float,
    is_exact_match: bool = False,
    is_same_image_hash: bool = False,
    is_quality_pass: bool = True,
) -> str:
    """Classify the relationship between query photo and discovered candidate.

    Categories:
    - SAME_IMAGE: Exact image match or identical SHA-256 fingerprint.
    - SAME_PERSON_DIFFERENT_IMAGE: Distinct image with strong biometric ArcFace match (>= 0.60).
    - VISUALLY_RELATED_IMAGE: Visual match with intermediate similarity (0.40 <= sim < 0.60).
    - DIFFERENT_PERSON: Biometric similarity clearly rejected (< 0.40).
    - UNCERTAIN: Degraded quality or boundary condition.
    """
    if not is_quality_pass:
        return "UNCERTAIN"
    if is_same_image_hash or (is_exact_match and similarity >= 0.95):
        return "SAME_IMAGE"
    if similarity >= 0.60:
        return "SAME_PERSON_DIFFERENT_IMAGE"
    if similarity >= 0.40:
        return "VISUALLY_RELATED_IMAGE"
    return "DIFFERENT_PERSON"


def build_source_relationship_graph(
    query_image_cid: Optional[str],
    evaluated_results: list[dict[str, Any]],
    query_thumb_hash: Optional[str] = None,
) -> dict[str, Any]:
    """Construct an internal evidence relationship graph across discovered sources.

    Maps:
    QUERY PHOTO
       ├── SAME_IMAGE (exact duplicates, identical image hashes)
       ├── SAME_PERSON_DIFFERENT_IMAGE (independent photo corroboration)
       ├── VISUALLY_RELATED_IMAGE (contextual search results)
       └── DIFFERENT_PERSON (competing/distractor faces)

    Returns:
        Structured dictionary with categorized relationships, edge nodes, and distribution counts.
    """
    same_images: list[dict[str, Any]] = []
    same_person: list[dict[str, Any]] = []
    visually_related: list[dict[str, Any]] = []
    different_person: list[dict[str, Any]] = []
    uncertain: list[dict[str, Any]] = []

    for r in evaluated_results:
        cand = r.get("candidate", {})
        sim = float(r.get("similarity", 0.0))
        dec = r.get("decision", "REJECTED")
        is_exact = cand.get("category") == "exact_matches"
        cand_hash = r.get("thumbnail_sha256") or cand.get("thumbnail_sha256")
        is_same_hash = bool(query_thumb_hash and cand_hash and query_thumb_hash == cand_hash)
        is_qual = bool(r.get("image_quality", 1.0) >= 0.20)

        rel = classify_source_relationship(
            similarity=sim,
            is_exact_match=is_exact,
            is_same_image_hash=is_same_hash,
            is_quality_pass=is_qual,
        )

        entry = {
            "title": cand.get("title", ""),
            "link": cand.get("link", ""),
            "domain": cand.get("domain", ""),
            "platform": cand.get("platform", "General Web"),
            "similarity": round(sim, 4),
            "decision": dec,
            "relationship": rel,
            "matched_face_id": r.get("matched_face_id", "FACE 01"),
        }

        if rel == "SAME_IMAGE":
            same_images.append(entry)
        elif rel == "SAME_PERSON_DIFFERENT_IMAGE":
            same_person.append(entry)
        elif rel == "VISUALLY_RELATED_IMAGE":
            visually_related.append(entry)
        elif rel == "DIFFERENT_PERSON":
            different_person.append(entry)
        else:
            uncertain.append(entry)

    return {
        "query_node": {
            "type": "QUERY_PHOTO",
            "cid": query_image_cid or "LOCAL_UPLOAD",
            "sha256": query_thumb_hash,
        },
        "counts": {
            "same_image": len(same_images),
            "same_person_different_image": len(same_person),
            "visually_related_image": len(visually_related),
            "different_person": len(different_person),
            "uncertain": len(uncertain),
            "total_nodes": len(evaluated_results),
        },
        "relationships": {
            "same_image": same_images,
            "same_images": same_images,
            "same_person_different_image": same_person,
            "same_person_different_images": same_person,
            "visually_related_image": visually_related,
            "visually_related_images": visually_related,
            "different_person": different_person,
            "different_persons": different_person,
            "uncertain": uncertain,
        },
    }


def build_evidence_manifest(
    query_face_metadata: dict[str, Any],
    candidate: dict[str, Any],
    similarity_score: float,
    decision: str,
    verified_threshold: float,
    review_threshold: float,
    decision_reason: str,
    query_image_cid: Optional[str] = None,
    total_candidates: int = 1,
    usable_images_count: int = 1,
    candidate_face_count: int = 1,
    candidate_det_confidence: float = 1.0,
    candidate_image_quality: Optional[float] = None,
    query_image_quality: Optional[float] = None,
    selected_face_index: int = 0,
    separation_margin: Optional[float] = None,
    margin_interpretation: Optional[str] = None,
    thumbnail_sha256: Optional[str] = None,
    discovery_timestamp: Optional[int] = None,
    blockchain_info: Optional[dict[str, Any]] = None,
    confidence_data: Optional[dict[str, Any]] = None,
    consensus_data: Optional[dict[str, Any]] = None,
    matched_face_id: Optional[str] = None,
    candidate_faces_evaluated: Optional[list[dict[str, Any]]] = None,
    search_telemetry: Optional[dict[str, Any]] = None,
    relationship_graph: Optional[dict[str, Any]] = None,
    image_metadata: Optional[dict[str, Any]] = None,
    dense_geometry: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], str]:
    """Construct a complete, tamper-evident evidence manifest for a verified discovery.

    Args:
        query_face_metadata: Dictionary with model, dimension, normalization, and embedding_hash.
        candidate: Dictionary with link, title, domain, platform, search_rank, thumbnail.
        similarity_score: Float cosine similarity between query and candidate face embeddings.
        decision: Verification status ('VERIFIED', 'REVIEW', or 'REJECTED').
        verified_threshold: Configured cutoff for VERIFIED status.
        review_threshold: Configured cutoff for REVIEW status.
        decision_reason: Human-readable rationale for the decision.
        query_image_cid: IPFS CID of the query image (if pinned).
        total_candidates: Total number of candidates discovered by search.
        usable_images_count: Total candidate images successfully downloaded & analyzed.
        candidate_face_count: Number of faces detected in candidate image.
        candidate_det_confidence: Detection confidence score for the candidate face.
        candidate_image_quality: Quality score of the candidate face image.
        query_image_quality: Quality score of the query face image.
        selected_face_index: Index of the selected query face (for multi-face inputs).
        separation_margin: Score delta between best verified and best non-matching candidate.
        margin_interpretation: Qualitative interpretation note for the separation margin.
        thumbnail_sha256: SHA-256 hash of the downloaded candidate thumbnail bytes.
        discovery_timestamp: Unix timestamp when discovery occurred (defaults to now).
        blockchain_info: Optional on-chain anchor details (network, chain_id, contract).
        confidence_data: Multi-signal dynamic confidence breakdown.
        consensus_data: Identity consensus and multi-source agreement metrics.
        matched_face_id: Specific identifier of matching candidate face in group photos (e.g. 'FACE 02').
        candidate_faces_evaluated: List of all evaluated faces in candidate image with scores.
        search_telemetry: Dictionary containing deep search pages, results, and platform metrics.
        relationship_graph: Optional source relationship graph across candidates.
        image_metadata: Optional non-sensitive camera and EXIF metadata.
        dense_geometry: Optional dense facial geometry metrics.

    Returns:
        Tuple of (manifest_dict, canonical_sha256_hex).
    """
    ts = int(time.time()) if discovery_timestamp is None else int(discovery_timestamp)

    # Clean candidate fields
    source_url = candidate.get("link", "")
    source_title = candidate.get("title", "")
    domain = candidate.get("domain", "")
    platform = candidate.get("platform", "General Web")
    search_rank = int(candidate.get("search_rank", 1))
    thumbnail_url = candidate.get("thumbnail")

    # Compute default confidence if not passed
    if confidence_data is None:
        confidence_data = calculate_dynamic_confidence(
            similarity=similarity_score,
            candidate_quality=candidate_image_quality or 0.80,
            separation_margin=separation_margin,
            cross_result_agreement=consensus_data.get("agreement_ratio", 1.0) if consensus_data else 1.0,
            platform_diversity=consensus_data.get("platform_count", 1) if consensus_data else 1,
        )

    search_info: dict[str, Any] = {
        "provider": "SerpApi",
        "engine": "google_lens",
        "query_image_cid": query_image_cid,
        "discovery_timestamp": ts,
        "total_candidates_discovered": int(total_candidates),
        "usable_images_evaluated": int(usable_images_count),
    }
    if search_telemetry:
        search_info["pages_scanned"] = int(search_telemetry.get("pages_scanned", 1))
        search_info["platforms_discovered"] = search_telemetry.get("platforms_discovered", [])
        if "search_modes_active" in search_telemetry:
            search_info["search_modes_active"] = search_telemetry["search_modes_active"]
        if "search_modes_queried" in search_telemetry:
            search_info["search_modes_queried"] = search_telemetry["search_modes_queried"]
        if "exact_matches_count" in search_telemetry:
            search_info["exact_matches_count"] = int(search_telemetry["exact_matches_count"])
        if "visual_matches_count" in search_telemetry:
            search_info["visual_matches_count"] = int(search_telemetry["visual_matches_count"])
        if "about_image_count" in search_telemetry:
            search_info["about_image_count"] = int(search_telemetry["about_image_count"])
        if "search_request_id" in search_telemetry and search_telemetry["search_request_id"]:
            search_info["search_request_id"] = str(search_telemetry["search_request_id"])

    verification_info: dict[str, Any] = {
        "face_similarity_score": round(float(similarity_score), 4),
        "identity_confidence": confidence_data.get("confidence_score", round(float(similarity_score), 4)),
        "confidence_breakdown": confidence_data.get("breakdown"),
        "verified_threshold": round(float(verified_threshold), 4),
        "review_threshold": round(float(review_threshold), 4),
        "decision": decision,
        "decision_reason": decision_reason,
        "face_detection_confidence": round(float(candidate_det_confidence), 4),
        "candidate_face_count": int(candidate_face_count),
        "matched_candidate_face_id": matched_face_id or ("FACE 01" if candidate_face_count >= 1 else "NONE"),
        "candidate_image_quality": round(float(candidate_image_quality), 4) if candidate_image_quality is not None else None,
        "separation_margin": round(float(separation_margin), 4) if separation_margin is not None else None,
        "margin_interpretation": margin_interpretation,
        "consensus_metrics": consensus_data,
    }
    if candidate_faces_evaluated:
        verification_info["candidate_faces_breakdown"] = [
            {
                "face_id": str(cf.get("face_id", f"FACE {idx+1:02d}")),
                "similarity": round(float(cf.get("similarity", 0.0)), 4),
                "quality": round(float(cf.get("quality", 0.0)), 4),
                "decision": str(cf.get("decision", "REJECTED")),
                "is_matched": bool(cf.get("is_matched", False)),
            }
            for idx, cf in enumerate(candidate_faces_evaluated)
        ]

    face_info: dict[str, Any] = {
        "algorithm": query_face_metadata.get("algorithm", "InsightFace buffalo_l"),
        "model": query_face_metadata.get("model", "ArcFace"),
        "dimension": int(query_face_metadata.get("dimension", 512)),
        "dtype": query_face_metadata.get("dtype", "float32"),
        "normalized": bool(query_face_metadata.get("normalization_status") == "L2_normalized"),
        "embedding_hash": query_face_metadata.get("embedding_hash", ""),
        "selected_face_index": int(selected_face_index),
        "query_image_quality": round(float(query_image_quality), 4) if query_image_quality is not None else None,
    }
    if dense_geometry:
        face_info["dense_geometry"] = {
            "pose_3d": dense_geometry.get("pose_3d"),
            "metrics": dense_geometry.get("metrics"),
        }

    manifest_payload: dict[str, Any] = {
        "schema_version": "1.2.0",
        "record_timestamp": ts,
        "face": face_info,
        "search": search_info,
        "candidate": {
            "source_url": source_url,
            "source_title": source_title,
            "domain": domain,
            "platform": platform,
            "search_rank": search_rank,
            "thumbnail_url": thumbnail_url,
            "thumbnail_sha256": thumbnail_sha256,
        },
        "verification": verification_info,
        "integrity": {
            "canonicalization_method": "RFC-8785 canonical JSON (sorted keys, compact separators, UTF-8)",
        },
    }

    if relationship_graph:
        manifest_payload["evidence_graph"] = relationship_graph
    if image_metadata:
        manifest_payload["image_metadata"] = image_metadata

    if blockchain_info:
        manifest_payload["blockchain"] = {
            "network": blockchain_info.get("network", "Polygon Amoy"),
            "chain_id": int(blockchain_info.get("chain_id", 80002)),
            "contract_address": blockchain_info.get("contract_address", ""),
        }

    # Compute deterministic hash over the payload
    manifest_hash = sha256_of_json(manifest_payload)
    manifest_payload["integrity"]["sha256_hash"] = manifest_hash

    # Final root hash over the entire complete manifest
    final_root_hash = sha256_of_json(manifest_payload)
    return manifest_payload, final_root_hash


# Backward compatibility alias
def build_record(query_face_hash: str, match: dict, similarity: float) -> tuple[dict, str]:
    """Legacy helper for backward compatibility."""
    return build_evidence_manifest(
        query_face_metadata={"embedding_hash": query_face_hash},
        candidate=match,
        similarity_score=similarity,
        decision="VERIFIED" if similarity >= 0.35 else "REJECTED",
        verified_threshold=0.35,
        review_threshold=0.25,
        decision_reason="Automated threshold evaluation",
    )
