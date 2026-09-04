"""Build canonical evidence manifests and calculate cryptographic SHA-256 fingerprints.

Follows RFC-8785 canonical JSON representation (lexicographically sorted keys,
compact separators, deterministic UTF-8 byte encoding) to ensure identical records
always produce bit-for-bit identical hashes across all environments.
"""
import hashlib
import json
import time
from typing import Any, Optional


def canonical_json_bytes(obj: Any) -> bytes:
    """Serialize any Python dictionary or data structure into canonical JSON UTF-8 bytes.

    - Lexicographically sorted keys
    - Compact separators (no extra whitespace: `(`,`:`)`)
    - Deterministic UTF-8 encoding
    """
    return json.dumps(
        obj,
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
    """Calculate the separation margin between the best verified match and the strongest rejected candidate.

    Args:
        verified_similarities: List of similarity scores for candidates meeting verified threshold.
        rejected_similarities: List of similarity scores for non-verified / rejected candidates.

    Returns:
        Dictionary with best_verified, best_rejected, separation_margin, and interpretation.
    """
    best_ver = max(verified_similarities) if verified_similarities else 0.0
    best_rej = max(rejected_similarities) if rejected_similarities else 0.0
    margin = round(best_ver - best_rej, 4) if (verified_similarities and rejected_similarities) else None

    if margin is not None:
        if margin >= 0.30:
            interp = "Clear separation: Strong differentiation between matching subject and non-matching candidates"
        elif margin >= 0.10:
            interp = "Moderate separation: Meaningful differentiation observed across evaluated candidates"
        else:
            interp = "Narrow margin: Evaluation warrants closer inspection of boundary candidates"
    else:
        interp = "Single-tier distribution: Margin calculation requires both verified and non-matching candidates"

    return {
        "best_verified_similarity": round(best_ver, 4) if verified_similarities else None,
        "best_rejected_similarity": round(best_rej, 4) if rejected_similarities else None,
        "separation_margin": margin,
        "margin_interpretation": interp,
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

    manifest_payload: dict[str, Any] = {
        "schema_version": "1.1.0",
        "record_timestamp": ts,
        "face": {
            "algorithm": query_face_metadata.get("algorithm", "InsightFace buffalo_l"),
            "model": query_face_metadata.get("model", "ArcFace"),
            "dimension": int(query_face_metadata.get("dimension", 512)),
            "dtype": query_face_metadata.get("dtype", "float32"),
            "normalized": bool(query_face_metadata.get("normalization_status") == "L2_normalized"),
            "embedding_hash": query_face_metadata.get("embedding_hash", ""),
            "selected_face_index": int(selected_face_index),
            "query_image_quality": round(float(query_image_quality), 4) if query_image_quality is not None else None,
        },
        "search": {
            "provider": "SerpApi",
            "engine": "google_lens",
            "query_image_cid": query_image_cid,
            "discovery_timestamp": ts,
            "total_candidates_discovered": int(total_candidates),
            "usable_images_evaluated": int(usable_images_count),
        },
        "candidate": {
            "source_url": source_url,
            "source_title": source_title,
            "domain": domain,
            "platform": platform,
            "search_rank": search_rank,
            "thumbnail_url": thumbnail_url,
            "thumbnail_sha256": thumbnail_sha256,
        },
        "verification": {
            "face_similarity_score": round(float(similarity_score), 4),
            "verified_threshold": round(float(verified_threshold), 4),
            "review_threshold": round(float(review_threshold), 4),
            "decision": decision,
            "decision_reason": decision_reason,
            "face_detection_confidence": round(float(candidate_det_confidence), 4),
            "candidate_face_count": int(candidate_face_count),
            "candidate_image_quality": round(float(candidate_image_quality), 4) if candidate_image_quality is not None else None,
            "separation_margin": round(float(separation_margin), 4) if separation_margin is not None else None,
            "margin_interpretation": margin_interpretation,
        },
        "integrity": {
            "canonicalization_method": "RFC-8785 canonical JSON (sorted keys, compact separators, UTF-8)",
        },
    }

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
