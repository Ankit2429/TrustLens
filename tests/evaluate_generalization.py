"""Comprehensive Generalization and Robustness Evaluation Runner for TrustLens.

Runs face detection, multi-factor quality scoring, embedding extraction, and multi-candidate
biometric evaluation across diverse test cases.
"""
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from pipeline.face_id import (
    analyze_face,
    cosine_similarity,
    detect_all_faces,
    hash_embedding,
    normalize_embedding,
)
from pipeline.main import classify_decision
from pipeline.fingerprint import calculate_separation_margin

TEST_DIR = os.path.join("demo", "test_cases")


def evaluate_all():
    print("=" * 80)
    print("  TRUSTLENS COMPREHENSIVE ACCURACY & GENERALIZATION AUDIT")
    print("=" * 80)

    test_files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".jpg")])

    results = []
    for filename in test_files:
        filepath = os.path.join(TEST_DIR, filename)
        print(f"\nEvaluating: {filename}...")
        try:
            faces = detect_all_faces(filepath, min_quality=0.10)
            if not faces:
                print(f"[-] No faces detected in {filename}")
                results.append({
                    "file": filename,
                    "face_detected": False,
                    "face_count": 0,
                    "confidence": 0.0,
                    "quality": 0.0,
                    "emb_success": False,
                    "decision": "NO_FACE_DETECTED",
                    "reason": "RetinaFace / SCRFD detector found no face in image",
                })
                continue

            primary_face = faces[0]
            quality = primary_face["quality"]
            emb = primary_face["normalized_embedding"]
            emb_hash = primary_face["embedding_hash"]

            results.append({
                "file": filename,
                "face_detected": True,
                "face_count": len(faces),
                "confidence": primary_face["det_score"],
                "quality": quality["overall_quality"],
                "quality_usable": quality["is_usable"],
                "sharpness": quality["breakdown"]["sharpness"],
                "exposure": quality["breakdown"]["exposure"],
                "resolution": quality["breakdown"]["resolution"],
                "frontality": quality["breakdown"]["frontality"],
                "emb_success": True,
                "emb_hash": emb_hash,
                "embedding": emb,
            })
            print(f"  -> Detected: {len(faces)} face(s)")
            print(f"  -> Confidence: {primary_face['det_score']:.4f}")
            print(f"  -> Overall Quality: {quality['overall_quality']:.4f} (Usable: {quality['is_usable']})")
            print(f"  -> Embedding SHA-256: {emb_hash[:20]}...")

        except Exception as e:
            print(f"[-] Error evaluating {filename}: {e}")

    # Pairwise Cosine Similarity Matrix between test cases
    valid_cases = [r for r in results if r.get("emb_success")]
    print("\n" + "=" * 80)
    print("  CROSS-VARIATION BIOMETRIC COSINE SIMILARITY MATRIX")
    print("=" * 80)

    # Compare Tom Hanks variations against each other (genuine pairs) and against group/other (imposter pairs)
    if len(valid_cases) >= 2:
        base_hanks = valid_cases[0]["embedding"]
        for r in valid_cases:
            sim = cosine_similarity(base_hanks, r["embedding"])
            r["sim_vs_base"] = sim
            dec, reason = classify_decision(sim, verified_threshold=0.45, review_threshold=0.35, is_quality_pass=r.get("quality_usable", True))
            r["decision_vs_base"] = dec
            r["reason_vs_base"] = reason

    print(f"{'Test Case File':<28} | {'Det Conf':<8} | {'Quality':<7} | {'Sim vs Base':<11} | {'Decision':<10} | {'Status'}")
    print("-" * 80)
    for r in valid_cases:
        sim_str = f"{r.get('sim_vs_base', 1.0):.4f}" if "sim_vs_base" in r else "N/A"
        dec_str = r.get("decision_vs_base", "N/A")
        print(f"{r['file']:<28} | {r['confidence']:<8.4f} | {r['quality']:<7.4f} | {sim_str:<11} | {dec_str:<10} | {r.get('reason_vs_base', '')[:30]}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    evaluate_all()
