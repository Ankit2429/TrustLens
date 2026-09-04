"""Stage-by-stage pipeline validation test."""
import copy
import json
import numpy as np
from pipeline.face_id import analyze_face, cosine_similarity, hash_embedding
from pipeline.fingerprint import build_evidence_manifest, sha256_of_json
from pipeline.search import extract_domain, classify_platform

print("=" * 70)
print("  STAGE-BY-STAGE INTEGRATION VALIDATION")
print("=" * 70)

print("\n[1] REAL FACE DETECTION & 512-D ARCFACE EMBEDDING")
analysis = analyze_face("demo/sample_face.jpg")
print(f"    - Face Detected    : {analysis['face_detected']}")
print(f"    - Face Count       : {analysis['face_count']}")
print(f"    - Detection Score  : {analysis['det_score']:.4f}")
print(f"    - Bounding Box     : {analysis['bbox']}")
print(f"    - Vector Dimension : {analysis['metadata']['dimension']}")
print(f"    - Vector L2 Norm   : {np.linalg.norm(analysis['normalized_embedding']):.4f}")
print(f"    - Embedding SHA-256: {analysis['embedding_hash']}")

print("\n[2] MULTI-PLATFORM DOMAIN CLASSIFICATION")
test_urls = [
    "https://www.instagram.com/p/C_abc123/",
    "https://linkedin.com/in/researcher",
    "https://x.com/tech_lead/status/9988",
    "https://youtube.com/watch?v=sample",
    "https://reddit.com/r/security/post1",
    "https://www.tiktok.com/@creator/video/1",
    "https://pinterest.com/pin/123",
    "https://nytimes.com/tech/article",
    "https://independent-blog.org/post",
]
for u in test_urls:
    dom = extract_domain(u)
    plat = classify_platform(dom)
    print(f"    - {u[:38]:<38} -> {dom:<22} [{plat}]")

print("\n[3] DETERMINISTIC EVIDENCE MANIFEST GENERATION (RFC-8785)")
candidate = {
    "link": "https://www.instagram.com/p/C_abc123/",
    "title": "Public Profile Photo - Consenting Subject",
    "domain": "instagram.com",
    "platform": "Instagram",
    "search_rank": 1,
    "thumbnail": "https://example.com/thumb.jpg",
    "thumbnail_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}
sim_score = 0.8752
manifest, manifest_hash = build_evidence_manifest(
    query_face_metadata=analysis["metadata"],
    candidate=candidate,
    similarity_score=sim_score,
    decision="VERIFIED",
    verified_threshold=0.40,
    review_threshold=0.30,
    decision_reason="High ArcFace embedding cosine similarity",
    query_image_cid="QmSampleQueryCID123456789",
    total_candidates=14,
    usable_images_count=8,
    candidate_face_count=1,
    candidate_det_confidence=0.9850,
    thumbnail_sha256=candidate["thumbnail_sha256"],
    discovery_timestamp=1741160000,
)
print(f"    - Schema Version   : {manifest['schema_version']}")
print(f"    - Verification     : [{manifest['verification']['decision']}] (similarity: {manifest['verification']['face_similarity_score']})")
print(f"    - Canonical Hash   : {manifest_hash}")

print("\n[4] INDEPENDENT RE-VERIFICATION")
recomputed_hash = sha256_of_json(manifest)
print(f"    - Stored Hash      : {manifest_hash}")
print(f"    - Recomputed Hash  : {recomputed_hash}")
print(f"    - Validation Result: {'VALID (Bit-for-bit cryptographic match)' if recomputed_hash == manifest_hash else 'MISMATCH'}")

print("\n[5] CRYPTOGRAPHIC TAMPER DETECTION")
tampered = copy.deepcopy(manifest)
tampered["candidate"]["source_url"] = "https://fake-imposter-site.com/spoofed"
tampered_hash = sha256_of_json(tampered)
print(f"    - Original Hash    : {manifest_hash}")
print(f"    - Tampered Hash    : {tampered_hash}")
print(f"    - Tamper Result    : {'TAMPER DETECTED!' if tampered_hash != manifest_hash else 'FAILED'}")
print("=" * 70)
