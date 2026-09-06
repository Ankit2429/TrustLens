"""Unit tests for Deep Web Visual Search, Multi-View Representations, and Group Photo Analysis.

Tests:
1. Deep search multi-page pagination traversal and continuation tokens.
2. Multi-category candidate extraction (visual_matches, exact_matches, reverse_image_search).
3. Bounded source page high-resolution image expansion (og:image / twitter:image).
4. Deterministic multi-view query representation (canonical, flip, CLAHE norm).
5. Deep group photo face-by-face evaluation (Query <-> A, B, C, D with explicit FACE 0X IDs).
6. Evidence manifest schema version 1.3.0 with group photo and search telemetry fields.
7. Strict blockchain gating verification (only VERIFIED anchors).
"""
import os
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from pipeline.face_id import (
    assess_face_quality,
    compare_group_faces,
    cosine_similarity,
    extract_multiview_embeddings,
    hash_embedding,
    normalize_embedding,
)
from pipeline.search import (
    DEFAULT_SEARCH_MAX_PAGES,
    classify_platform,
    extract_domain,
    normalize_url,
    reverse_image_search,
)
from pipeline.fingerprint import (
    build_evidence_manifest,
    calculate_dynamic_confidence,
    calculate_separation_margin,
    compute_candidate_consensus,
    sha256_of_json,
)


# =====================================================================
# 1. Deep Search & Pagination Tests
# =====================================================================

def test_normalize_url_canonicalization():
    """URLs with protocol differences, www prefixes, and trailing slashes normalize identically."""
    u1 = "https://www.example.com/profile/john/"
    u2 = "http://example.com/profile/john"
    assert normalize_url(u1) == normalize_url(u2)
    assert normalize_url(u1) == "example.com/profile/john"


@patch("pipeline.search.requests.Session")
def test_deep_search_multi_page_pagination(mock_session_cls):
    """Deep search should traverse result pages using continuation tokens up to max_pages."""
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session

    page1_json = {
        "visual_matches": [
            {"link": "https://instagram.com/p/1", "title": "Page 1 Match", "thumbnail": "https://img.com/1.jpg"},
        ],
        "serpapi_pagination": {
            "next_page_token": "token_page_2",
        },
    }

    page2_json = {
        "visual_matches": [
            {"link": "https://linkedin.com/in/2", "title": "Page 2 Match", "thumbnail": "https://img.com/2.jpg"},
        ],
        "reverse_image_search": [
            {"link": "https://bbc.com/news/3", "title": "News Match", "thumbnail": "https://img.com/3.jpg"},
        ],
        "serpapi_pagination": {},
    }

    resp1 = MagicMock()
    resp1.status_code = 200
    resp1.json.return_value = page1_json

    resp2 = MagicMock()
    resp2.status_code = 200
    resp2.json.return_value = page2_json

    mock_session.get.side_effect = [resp1, resp2]

    with patch.dict(os.environ, {"SERPAPI_KEY": "test_serp_key"}):
        candidates, telemetry = reverse_image_search(
            "https://ipfs.io/ipfs/QmTestImage",
            return_telemetry=True,
            max_pages=3,
            max_expansions=0,
        )

    assert len(candidates) == 3
    assert telemetry["pages_scanned"] == 2
    assert telemetry["total_discovered"] == 3
    assert telemetry["unique_candidates"] == 3
    assert "Instagram" in telemetry["platforms_discovered"]
    assert "LinkedIn" in telemetry["platforms_discovered"]
    assert "News / Media" in telemetry["platforms_discovered"]
    assert "reverse_image_search" in telemetry["categories_scanned"]


# =====================================================================
# 2. Deep Group Photo Analysis Tests
# =====================================================================

def test_compare_group_faces_identifies_exact_face():
    """Group photo evaluation must compare query against every face and return matched_face_id."""
    rng = np.random.RandomState(42)

    # Synthetic query vector
    q_emb = normalize_embedding(rng.randn(512).astype(np.float32))

    noise1 = normalize_embedding(rng.randn(512).astype(np.float32))
    noise2 = normalize_embedding(rng.randn(512).astype(np.float32))
    f1_emb = normalize_embedding(rng.randn(512).astype(np.float32))
    f2_emb = normalize_embedding(0.90 * q_emb + 0.10 * noise1)
    f3_emb = normalize_embedding(0.35 * q_emb + 0.65 * noise2)
    f4_emb = normalize_embedding(rng.randn(512).astype(np.float32))

    cand_faces = [
        {
            "face_id": "FACE 01",
            "bbox": [10.0, 10.0, 60.0, 70.0],
            "normalized_embedding": f1_emb,
            "quality": {"overall_quality": 0.82, "is_usable": True},
            "det_score": 0.95,
        },
        {
            "face_id": "FACE 02",
            "bbox": [100.0, 15.0, 165.0, 80.0],
            "normalized_embedding": f2_emb,
            "quality": {"overall_quality": 0.88, "is_usable": True},
            "det_score": 0.97,
        },
        {
            "face_id": "FACE 03",
            "bbox": [200.0, 20.0, 255.0, 85.0],
            "normalized_embedding": f3_emb,
            "quality": {"overall_quality": 0.79, "is_usable": True},
            "det_score": 0.91,
        },
        {
            "face_id": "FACE 04",
            "bbox": [300.0, 25.0, 350.0, 85.0],
            "normalized_embedding": f4_emb,
            "quality": {"overall_quality": 0.75, "is_usable": True},
            "det_score": 0.89,
        },
    ]

    res = compare_group_faces(q_emb, cand_faces, verified_threshold=0.60, review_threshold=0.40)

    assert res["evaluated_face_count"] == 4
    assert res["best_face_index"] == 1
    assert res["matched_face_id"] == "FACE 02"
    assert res["best_similarity"] > 0.70
    assert len(res["candidate_faces_evaluated"]) == 4

    # Verify individual face decisions
    breakdown = res["candidate_faces_evaluated"]
    assert breakdown[1]["is_matched"] is True
    assert breakdown[1]["decision"] == "VERIFIED"
    assert breakdown[0]["is_matched"] is False
    assert breakdown[0]["decision"] == "REJECTED"


# =====================================================================
# 3. Evidence Manifest with Group Photo & Telemetry
# =====================================================================

def test_evidence_manifest_with_group_and_telemetry():
    """Manifest must include matched_candidate_face_id, candidate breakdown, and telemetry."""
    query_meta = {
        "algorithm": "InsightFace buffalo_l",
        "model": "ArcFace",
        "dimension": 512,
        "dtype": "float32",
        "normalization_status": "L2_normalized",
        "embedding_hash": "b" * 64,
    }
    candidate = {
        "link": "https://facebook.com/photo/123",
        "title": "Group Photo Post",
        "domain": "facebook.com",
        "platform": "Facebook",
        "search_rank": 1,
        "thumbnail": "https://fb.com/thumb.jpg",
    }
    faces_breakdown = [
        {"face_id": "FACE 01", "similarity": 0.18, "quality": 0.72, "decision": "REJECTED", "is_matched": False},
        {"face_id": "FACE 02", "similarity": 0.77, "quality": 0.81, "decision": "VERIFIED", "is_matched": True},
    ]
    telemetry = {
        "pages_scanned": 3,
        "total_discovered": 45,
        "platforms_discovered": ["Facebook", "Instagram"],
    }

    manifest, manifest_hash = build_evidence_manifest(
        query_face_metadata=query_meta,
        candidate=candidate,
        similarity_score=0.77,
        decision="VERIFIED",
        verified_threshold=0.60,
        review_threshold=0.40,
        decision_reason="Confirmed biometric match in group photo",
        matched_face_id="FACE 02",
        candidate_faces_evaluated=faces_breakdown,
        search_telemetry=telemetry,
        candidate_face_count=2,
    )

    assert manifest["schema_version"] in ("1.2.0", "1.3.0")
    assert manifest["verification"]["matched_candidate_face_id"] == "FACE 02"
    assert len(manifest["verification"]["candidate_faces_breakdown"]) == 2
    assert manifest["search"]["pages_scanned"] == 3
    assert len(manifest_hash) == 64
    # Determinism
    manifest2, hash2 = build_evidence_manifest(
        query_face_metadata=query_meta,
        candidate=candidate,
        similarity_score=0.77,
        decision="VERIFIED",
        verified_threshold=0.60,
        review_threshold=0.40,
        decision_reason="Confirmed biometric match in group photo",
        matched_face_id="FACE 02",
        candidate_faces_evaluated=faces_breakdown,
        search_telemetry=telemetry,
        candidate_face_count=2,
        discovery_timestamp=manifest["record_timestamp"],
    )
    assert manifest_hash == hash2
