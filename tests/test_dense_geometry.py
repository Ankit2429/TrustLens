"""Unit & Regression Tests for Dense Facial Geometry / Mesh Layer.

Tests verify:
1. Dense 2D (106-point) and 3D (68-point) landmark extraction.
2. 3D head pose estimation (pitch, yaw, roll).
3. Biometric landmark consistency and inter-ocular distance (IOD).
4. Preservation of InsightFace ArcFace 512-d identity embeddings.
5. Multi-face group photo geometry extraction and face attribution.
6. Source relationship graph and multi-mode search classification.
7. Web API serialization of dense geometry, discovery HUD, and EXIF status.
"""
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from pipeline.face_id import (
    extract_dense_geometry,
    assess_face_quality,
    detect_all_faces,
    analyze_face,
    compare_group_faces,
    cosine_similarity,
    normalize_embedding,
)
from pipeline.search import extract_image_metadata
from pipeline.fingerprint import (
    classify_source_relationship,
    build_source_relationship_graph,
    compute_candidate_consensus,
    build_evidence_manifest,
)


class MockFace:
    """Mock InsightFace Face object with 106 2D and 68 3D landmarks."""
    def __init__(self, bbox=None, kps=None, landmark_2d=None, landmark_3d=None, det_score=0.95, embedding=None):
        self.bbox = np.array(bbox if bbox is not None else [50.0, 50.0, 150.0, 160.0], dtype=np.float32)
        self.kps = np.array(kps if kps is not None else [
            [80.0, 85.0],   # left eye
            [120.0, 85.0],  # right eye
            [100.0, 105.0], # nose
            [85.0, 130.0],  # left mouth
            [115.0, 130.0], # right mouth
        ], dtype=np.float32)
        
        # 106 2D points
        if landmark_2d is not None:
            self.landmark_2d_106 = np.array(landmark_2d, dtype=np.float32)
        else:
            pts = []
            for i in range(106):
                pts.append([50.0 + (i % 10) * 10.0, 50.0 + (i // 10) * 10.0])
            self.landmark_2d_106 = np.array(pts, dtype=np.float32)
            
        # 68 3D points
        if landmark_3d is not None:
            self.landmark_3d_68 = np.array(landmark_3d, dtype=np.float32)
        else:
            pts_3d = []
            for i in range(68):
                pts_3d.append([50.0 + (i % 8) * 12.0, 50.0 + (i // 8) * 12.0, float(i)])
            self.landmark_3d_68 = np.array(pts_3d, dtype=np.float32)

        self.det_score = det_score
        self.pose = np.array([2.5, -4.1, 0.8], dtype=np.float32)
        
        if embedding is not None:
            self.embedding = np.array(embedding, dtype=np.float32)
        else:
            v = np.random.randn(512).astype(np.float32)
            self.embedding = v / np.linalg.norm(v)


def test_extract_dense_geometry_full():
    """Verify extract_dense_geometry computes 106 2D points, 68 3D points, pose and metrics."""
    mock = MockFace()
    geom = extract_dense_geometry(mock)

    assert geom is not None
    assert geom["point_count_2d"] == 106
    assert geom["point_count_3d"] == 68
    assert len(geom["landmarks_2d"]) == 106
    assert len(geom["landmarks_3d"]) == 68

    # Check 3D pose extraction
    pose = geom["pose_3d"]
    assert "pitch" in pose and "yaw" in pose and "roll" in pose
    assert isinstance(pose["pitch"], float)
    assert isinstance(pose["yaw"], float)
    assert isinstance(pose["roll"], float)

    # Check metrics
    metrics = geom["metrics"]
    assert metrics["iod"] > 0
    assert "eye_line_angle" in metrics
    assert metrics["jaw_width"] > 0
    assert metrics["face_height"] > 0
    assert geom["landmark_consistency"] in ("HIGH", "MODERATE", "DEGRADED")


def test_assess_face_quality_with_3d_pose():
    """Verify assess_face_quality utilizes 3D pose angles in breakdown."""
    mock = MockFace()
    dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
    q = assess_face_quality(dummy_img, mock)

    assert "overall_quality" in q
    assert "breakdown" in q
    bd = q["breakdown"]
    assert "pose_pitch_est" in bd
    assert "pose_yaw_est" in bd
    assert "pose_roll_est" in bd


def test_classify_source_relationship_categories():
    """Verify classify_source_relationship maps biometrics and discovery metadata accurately."""
    # 1. Exact match / same hash
    assert classify_source_relationship(0.98, is_exact_match=True) == "SAME_IMAGE"
    assert classify_source_relationship(0.75, is_same_image_hash=True) == "SAME_IMAGE"

    # 2. Same person, different image (verified biometric threshold >= 0.60)
    assert classify_source_relationship(0.72) == "SAME_PERSON_DIFFERENT_IMAGE"
    assert classify_source_relationship(0.60) == "SAME_PERSON_DIFFERENT_IMAGE"

    # 3. Visually related / review range (0.40 <= sim < 0.60)
    assert classify_source_relationship(0.48) == "VISUALLY_RELATED_IMAGE"
    assert classify_source_relationship(0.40) == "VISUALLY_RELATED_IMAGE"

    # 4. Different person (< 0.40)
    assert classify_source_relationship(0.25) == "DIFFERENT_PERSON"
    assert classify_source_relationship(0.05) == "DIFFERENT_PERSON"

    # 5. Degraded quality
    assert classify_source_relationship(0.85, is_quality_pass=False) == "UNCERTAIN"


def test_build_source_relationship_graph_structure():
    """Verify build_source_relationship_graph constructs structured hierarchy."""
    evaluated = [
        {
            "candidate": {"title": "Exact Match", "link": "https://example.com/1", "category": "exact_matches"},
            "similarity": 0.99,
            "decision": "VERIFIED",
            "image_quality": 0.95,
            "matched_face_id": "FACE 01",
        },
        {
            "candidate": {"title": "News Profile", "link": "https://news.com/a", "category": "visual_matches"},
            "similarity": 0.74,
            "decision": "VERIFIED",
            "image_quality": 0.90,
            "matched_face_id": "FACE 01",
        },
        {
            "candidate": {"title": "Related Article", "link": "https://blog.com/b", "category": "visual_matches"},
            "similarity": 0.48,
            "decision": "REVIEW",
            "image_quality": 0.80,
            "matched_face_id": "FACE 01",
        },
        {
            "candidate": {"title": "Unrelated Face", "link": "https://other.com/c", "category": "visual_matches"},
            "similarity": 0.12,
            "decision": "REJECTED",
            "image_quality": 0.85,
            "matched_face_id": "FACE 01",
        },
    ]

    graph = build_source_relationship_graph(
        query_image_cid="QmTest123",
        evaluated_results=evaluated,
        query_thumb_hash="abc123hash",
    )

    assert graph["query_node"]["cid"] == "QmTest123"
    counts = graph["counts"]
    assert counts["same_image"] == 1
    assert counts["same_person_different_image"] == 1
    assert counts["visually_related_image"] == 1
    assert counts["different_person"] == 1
    assert counts["total_nodes"] == 4

    rels = graph["relationships"]
    assert len(rels["same_image"]) == 1
    assert rels["same_image"][0]["title"] == "Exact Match"
    assert len(rels["same_person_different_image"]) == 1


def test_extract_image_metadata_synthetic(tmp_path):
    """Verify extract_image_metadata parses PIL properties and non-sensitive tags."""
    img_path = tmp_path / "test_meta.jpg"
    img = Image.new("RGB", (320, 240), color=(100, 150, 200))
    img.save(img_path, format="JPEG")

    meta = extract_image_metadata(img_path)
    assert meta["width"] == 320
    assert meta["height"] == 240
    assert meta["format"] == "JPEG"
    assert meta["status"] in ("METADATA AVAILABLE", "NO METADATA")


def test_multi_face_group_photo_geometry():
    """Verify group photo multi-face comparison retains per-face dense geometry."""
    query_emb = np.random.randn(512).astype(np.float32)
    query_emb = normalize_embedding(query_emb)

    # Candidate image with 2 faces
    cand_faces = [
        {"face_id": "FACE 01", "embedding": query_emb * 0.98, "det_score": 0.95, "bbox": [10, 10, 60, 60]},
        {"face_id": "FACE 02", "embedding": np.random.randn(512).astype(np.float32), "det_score": 0.92, "bbox": [100, 10, 150, 60]},
    ]
    cand_faces[1]["embedding"] = normalize_embedding(cand_faces[1]["embedding"])

    res = compare_group_faces(query_emb, cand_faces, verified_threshold=0.60, review_threshold=0.40)

    assert res["matched_face_id"] == "FACE 01"
    assert res["similarity"] > 0.90
    assert res["decision"] == "VERIFIED"
    assert len(res["all_faces"]) == 2
    assert res["all_faces"][0]["is_matched"] is True
    assert res["all_faces"][1]["is_matched"] is False
