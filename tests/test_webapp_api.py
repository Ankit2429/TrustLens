"""Integration test script for TrustLens Web Application API on localhost."""
import json
import os
import sys
import requests

BASE_URL = "http://127.0.0.1:8000"

def test_webapp_integration():
    print("[1/7] Testing GET /api/health...")
    r = requests.get(f"{BASE_URL}/api/health", timeout=10)
    assert r.status_code == 200, f"Health check failed: {r.text}"
    health = r.json()
    print(f"      Health OK. Network: {health.get('network')}, Block: #{health.get('latest_block')}, Contract: {health.get('contract_address')}")

    print("\n[2/7] Testing GET /api/demo-image...")
    r = requests.get(f"{BASE_URL}/api/demo-image", timeout=10)
    assert r.status_code == 200, f"Demo image endpoint failed: {r.status_code}"
    demo_bytes = r.content
    print(f"      Demo image retrieved ({len(demo_bytes)} bytes)")

    print("\n[3/7] Testing POST /api/detect-faces...")
    files = {"image": ("public_face_demo.jpg", demo_bytes, "image/jpeg")}
    r = requests.post(f"{BASE_URL}/api/detect-faces", files=files, timeout=30)
    assert r.status_code == 200, f"Detect faces failed: {r.text}"
    detect_res = r.json()
    assert detect_res["face_detected"] is True, "No face detected in demo image"
    print(f"      Faces detected: {detect_res['face_count']}, Quality: {detect_res['faces'][0]['quality']['overall_quality']:.4f}")

    print("\n[4/7] Testing POST /api/analyze (Full Real Task 3 Pipeline)...")
    files = {"image": ("public_face_demo.jpg", demo_bytes, "image/jpeg")}
    data = {
        "face_index": 0,
        "verified_threshold": 0.40,
        "review_threshold": 0.30,
        "min_quality": 0.20,
        "skip_blockchain": False,
    }
    r = requests.post(f"{BASE_URL}/api/analyze", files=files, data=data, timeout=180)
    assert r.status_code == 200, f"Analyze pipeline failed: {r.text}"
    pipeline_res = r.json()
    assert pipeline_res["success"] is True, "Pipeline did not succeed"
    proof_hash = pipeline_res["manifest_hash"]
    proof_cid = pipeline_res["manifest_cid"]
    chain_receipt = pipeline_res.get("blockchain_receipt")
    best_match = pipeline_res.get("best_match")

    print(f"      Pipeline Succeeded!")
    print(f"      - Best Match Platform: {best_match.get('platform')}")
    print(f"      - Best Match URL: {best_match.get('link')}")
    print(f"      - Similarity Score: {best_match.get('similarity')}")
    print(f"      - Decision: {best_match.get('decision')}")
    print(f"      - Manifest SHA-256: {proof_hash}")
    print(f"      - Manifest IPFS CID: {proof_cid}")
    if chain_receipt:
        print(f"      - Polygon Amoy Tx: {chain_receipt.get('tx_hash')} (Block #{chain_receipt.get('block')})")

    print(f"\n[5/7] Testing POST /api/verify for hash: {proof_hash}...")
    r = requests.post(f"{BASE_URL}/api/verify", json={"proof_hash": proof_hash}, timeout=30)
    assert r.status_code == 200, f"Verify endpoint failed: {r.text}"
    verif_res = r.json()
    assert verif_res["is_valid"] is True, f"Verification returned invalid: {verif_res}"
    print(f"      On-Chain Verification: VALID (Match: {verif_res['is_valid']})")

    print(f"\n[6/7] Testing POST /api/tamper-test for hash: {proof_hash}...")
    r = requests.post(f"{BASE_URL}/api/tamper-test", json={"proof_hash": proof_hash, "tamper_field": "url"}, timeout=30)
    assert r.status_code == 200, f"Tamper test failed: {r.text}"
    tamper_res = r.json()
    assert tamper_res["tamper_detected"] is True, "Tamper was not detected"
    print(f"      Tamper Test Result: {tamper_res['status']} (Detected: {tamper_res['tamper_detected']})")

    print("\n[7/7] Testing GET / (Static Web UI)...")
    r = requests.get(f"{BASE_URL}/", timeout=10)
    assert r.status_code == 200, f"Frontend index.html failed: {r.status_code}"
    assert "TRUSTLENS" in r.text, "Index HTML missing TRUSTLENS title"
    print("      Static Frontend served correctly.")

    print("\n=======================================================")
    print("ALL WEBAPP API & INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=======================================================")

if __name__ == "__main__":
    test_webapp_integration()
