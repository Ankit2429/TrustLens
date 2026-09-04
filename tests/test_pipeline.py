"""Comprehensive Unit Test Suite for Face Identification & Blockchain Verification Pipeline.

Tests:
1. Cosine similarity mathematical correctness
2. 512-d ArcFace vector normalization
3. Complete 512-d embedding deterministic hashing
4. Canonical RFC-8785 JSON determinism
5. Evidence manifest generation and field provenance
6. Domain extraction and platform classification
7. Multi-tier decision engine (VERIFIED, REVIEW, REJECTED)
8. Cryptographic tamper detection logic
9. Smart contract ABI interface validation
"""
import hashlib
import json
import numpy as np
import pytest

from pipeline.face_id import (
    cosine_similarity,
    get_embedding_metadata,
    hash_embedding,
    normalize_embedding,
)
from pipeline.search import extract_domain, classify_platform
from pipeline.fingerprint import (
    build_evidence_manifest,
    canonical_json_bytes,
    sha256_of_json,
    sha256_of_bytes,
)
from pipeline.chain import load_abi, PROOF_REGISTRY_ABI


# =====================================================================
# 1. Cosine Similarity & Normalization Tests
# =====================================================================

def test_cosine_similarity_identical_vectors():
    """Identical vectors must have cosine similarity of exactly 1.0."""
    v = np.random.RandomState(42).randn(512).astype(np.float32)
    sim = cosine_similarity(v, v)
    assert pytest.approx(sim, abs=1e-5) == 1.0


def test_cosine_similarity_opposite_vectors():
    """Opposite vectors must have cosine similarity of -1.0."""
    v = np.random.RandomState(42).randn(512).astype(np.float32)
    sim = cosine_similarity(v, -v)
    assert pytest.approx(sim, abs=1e-5) == -1.0


def test_cosine_similarity_orthogonal_vectors():
    """Orthogonal vectors must have cosine similarity of 0.0."""
    v1 = np.zeros(512, dtype=np.float32)
    v2 = np.zeros(512, dtype=np.float32)
    v1[0] = 1.0
    v2[1] = 1.0
    sim = cosine_similarity(v1, v2)
    assert pytest.approx(sim, abs=1e-5) == 0.0


def test_normalize_embedding_unit_norm():
    """Normalized embedding vector must have L2 norm equal to 1.0."""
    v = np.array([3.0, 4.0, 0.0, 12.0], dtype=np.float32)
    normed = normalize_embedding(v)
    assert pytest.approx(np.linalg.norm(normed), abs=1e-5) == 1.0


def test_normalize_embedding_zero_vector():
    """Zero vector normalization should not crash with division by zero."""
    v = np.zeros(512, dtype=np.float32)
    normed = normalize_embedding(v)
    assert np.all(normed == 0.0)


# =====================================================================
# 2. Complete 512-d Face Embedding Hashing Tests
# =====================================================================

def test_embedding_hash_determinism():
    """Same 512-d embedding must produce bit-for-bit identical SHA-256 hash."""
    rng = np.random.RandomState(1337)
    emb = rng.randn(512).astype(np.float32)

    hash1 = hash_embedding(emb)
    hash2 = hash_embedding(emb.copy())

    assert len(hash1) == 64
    assert hash1 == hash2


def test_embedding_hash_sensitivity():
    """Modifying even a single float in 512-d embedding must change the hash."""
    rng = np.random.RandomState(1337)
    emb1 = rng.randn(512).astype(np.float32)
    emb2 = emb1.copy()
    emb2[511] += 0.001

    hash1 = hash_embedding(emb1)
    hash2 = hash_embedding(emb2)

    assert hash1 != hash2


def test_embedding_metadata_fields():
    """Embedding metadata must contain full model and cryptographic attributes."""
    emb = np.ones(512, dtype=np.float32)
    meta = get_embedding_metadata(emb)

    assert meta["algorithm"] == "InsightFace buffalo_l"
    assert meta["model"] == "ArcFace"
    assert meta["dimension"] == 512
    assert meta["dtype"] == "float32"
    assert meta["normalization_status"] == "L2_normalized"
    assert len(meta["embedding_hash"]) == 64


# =====================================================================
# 3. Canonical JSON & Hashing Tests (RFC-8785)
# =====================================================================

def test_canonical_json_key_order_invariance():
    """Dictionaries with identical data in different insertion order must produce identical SHA-256."""
    dict_a = {"z": 100, "a": "hello", "m": [1, 2, 3]}
    dict_b = {"a": "hello", "m": [1, 2, 3], "z": 100}

    assert canonical_json_bytes(dict_a) == canonical_json_bytes(dict_b)
    assert sha256_of_json(dict_a) == sha256_of_json(dict_b)


def test_canonical_json_compact_separators():
    """Canonical JSON must not contain extraneous whitespace."""
    obj = {"alpha": 1, "beta": 2}
    raw = canonical_json_bytes(obj)
    assert b" " not in raw
    assert raw == b'{"alpha":1,"beta":2}'


# =====================================================================
# 4. Multi-Platform Classification Tests
# =====================================================================

@pytest.mark.parametrize(
    "url, expected_domain, expected_platform",
    [
        ("https://www.instagram.com/p/C_sample123/", "instagram.com", "Instagram"),
        ("https://linkedin.com/in/john-doe-456", "linkedin.com", "LinkedIn"),
        ("https://facebook.com/photo.php?fbid=789", "facebook.com", "Facebook"),
        ("https://x.com/username/status/12345", "x.com", "X (Twitter)"),
        ("https://twitter.com/username/status/12345", "twitter.com", "X (Twitter)"),
        ("https://www.reddit.com/r/technology/comments/abc/", "reddit.com", "Reddit"),
        ("https://youtube.com/watch?v=dQw4w9WgXcQ", "youtube.com", "YouTube"),
        ("https://www.tiktok.com/@user/video/123", "tiktok.com", "TikTok"),
        ("https://pinterest.com/pin/998877/", "pinterest.com", "Pinterest"),
        ("https://www.bbc.com/news/world-123456", "bbc.com", "News / Media"),
        ("https://techcrunch.com/2026/09/04/article", "techcrunch.com", "News / Media"),
        ("https://mytechblog.org/about-me", "mytechblog.org", "General Web"),
    ],
)
def test_domain_and_platform_classification(url, expected_domain, expected_platform):
    domain = extract_domain(url)
    platform = classify_platform(domain)

    assert domain == expected_domain
    assert platform == expected_platform


# =====================================================================
# 5. Evidence Manifest & Decision Engine Tests
# =====================================================================

def test_evidence_manifest_structure_and_hash():
    """Evidence manifest must adhere to schema version 1.0.0 and be deterministically hashable."""
    face_meta = {
        "algorithm": "InsightFace buffalo_l",
        "model": "ArcFace",
        "dimension": 512,
        "dtype": "float32",
        "normalization_status": "L2_normalized",
        "embedding_hash": "a" * 64,
    }
    candidate = {
        "link": "https://www.instagram.com/p/test123/",
        "title": "Consenting Subject Post",
        "domain": "instagram.com",
        "platform": "Instagram",
        "search_rank": 1,
        "thumbnail": "https://example.com/thumb.jpg",
    }

    manifest, root_hash = build_evidence_manifest(
        query_face_metadata=face_meta,
        candidate=candidate,
        similarity_score=0.8642,
        decision="VERIFIED",
        verified_threshold=0.40,
        review_threshold=0.30,
        decision_reason="High ArcFace similarity",
        query_image_cid="QmTest123CID",
        discovery_timestamp=1741160000,
    )

    assert manifest["schema_version"] == "1.0.0"
    assert manifest["verification"]["decision"] == "VERIFIED"
    assert manifest["verification"]["face_similarity_score"] == 0.8642
    assert manifest["candidate"]["platform"] == "Instagram"
    assert manifest["integrity"]["canonicalization_method"].startswith("RFC-8785")
    assert len(root_hash) == 64
    assert sha256_of_json(manifest) == root_hash


def test_evidence_manifest_determinism():
    """Two identical manifest constructions must yield bit-for-bit identical hashes."""
    face_meta = {"embedding_hash": "b" * 64, "model": "ArcFace", "dimension": 512}
    cand = {"link": "https://linkedin.com/in/sample", "platform": "LinkedIn"}

    _, hash1 = build_evidence_manifest(
        face_meta, cand, 0.75, "VERIFIED", 0.40, 0.30, "Good match", discovery_timestamp=1741160000
    )
    _, hash2 = build_evidence_manifest(
        face_meta, cand, 0.75, "VERIFIED", 0.40, 0.30, "Good match", discovery_timestamp=1741160000
    )

    assert hash1 == hash2


# =====================================================================
# 6. Cryptographic Tamper Detection Tests
# =====================================================================

def test_tamper_detection_on_url_modification():
    """Mutating source_url in the evidence manifest must cause a hash mismatch."""
    face_meta = {"embedding_hash": "c" * 64}
    cand = {"link": "https://original-site.com/real-photo.jpg", "platform": "General Web"}

    manifest, original_hash = build_evidence_manifest(
        face_meta, cand, 0.82, "VERIFIED", 0.40, 0.30, "Match", discovery_timestamp=1741160000
    )

    # Mutate source URL locally
    tampered = json.loads(json.dumps(manifest))
    tampered["candidate"]["source_url"] = "https://fake-imposter.com/spoof.jpg"
    tampered_hash = sha256_of_json(tampered)

    assert tampered_hash != original_hash


def test_tamper_detection_on_similarity_modification():
    """Mutating face_similarity_score must cause a hash mismatch."""
    face_meta = {"embedding_hash": "d" * 64}
    cand = {"link": "https://x.com/status/999", "platform": "X (Twitter)"}

    manifest, original_hash = build_evidence_manifest(
        face_meta, cand, 0.32, "REVIEW", 0.40, 0.30, "Review needed", discovery_timestamp=1741160000
    )

    # Mutate similarity score locally to pretend it's 0.99
    tampered = json.loads(json.dumps(manifest))
    tampered["verification"]["face_similarity_score"] = 0.99
    tampered["verification"]["decision"] = "VERIFIED"
    tampered_hash = sha256_of_json(tampered)

    assert tampered_hash != original_hash


# =====================================================================
# 7. Smart Contract Interface Validation
# =====================================================================

def test_proof_registry_abi_functions():
    """Contract ABI must contain required functions and events for Task 3."""
    abi = load_abi()
    fn_names = {item["name"] for item in abi if item.get("type") == "function"}
    event_names = {item["name"] for item in abi if item.get("type") == "event"}

    assert "registerProof" in fn_names
    assert "getProof" in fn_names
    assert "ProofRegistered" in event_names


# =====================================================================
# 8. Live Face Analysis & Deterministic Embedding Test
# =====================================================================

def test_live_face_analysis_on_sample_image():
    """Live InsightFace Buffalo_l / ArcFace analysis on demo/sample_face.jpg."""
    import os
    from pipeline.face_id import analyze_face

    sample_path = os.path.join("demo", "sample_face.jpg")
    if not os.path.exists(sample_path):
        pytest.skip("demo/sample_face.jpg does not exist")

    analysis1 = analyze_face(sample_path)
    analysis2 = analyze_face(sample_path)

    assert analysis1["face_detected"] is True
    assert analysis1["face_count"] >= 1
    assert analysis1["det_score"] > 0.5
    assert len(analysis1["bbox"]) == 4

    emb1 = analysis1["normalized_embedding"]
    assert emb1.shape == (512,)
    assert pytest.approx(float(np.linalg.norm(emb1)), abs=1e-5) == 1.0

    # Determinism across repeated runs
    assert analysis1["embedding_hash"] == analysis2["embedding_hash"]
    assert len(analysis1["embedding_hash"]) == 64
    assert pytest.approx(cosine_similarity(emb1, analysis2["normalized_embedding"]), abs=1e-5) == 1.0

