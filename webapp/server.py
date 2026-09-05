"""TrustLens FastAPI Web Application Server.

Exposes REST endpoints that directly invoke the real TrustLens pipeline,
verification logic, and tamper detection against Polygon Amoy and IPFS.
"""
import copy
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import cv2
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.face_id import (
    analyze_face,
    assess_face_quality,
    compare_group_faces,
    cosine_similarity,
    detect_all_faces,
    get_embedding_metadata,
    hash_embedding,
    normalize_embedding,
)
from pipeline.search import reverse_image_search, filter_social_matches
from pipeline.fingerprint import (
    build_evidence_manifest,
    calculate_dynamic_confidence,
    calculate_separation_margin,
    compute_candidate_consensus,
    sha256_of_json,
    sha256_of_bytes,
)
from pipeline.ipfs_store import pin_file, pin_json, gateway_url, fetch_json
from pipeline.chain import (
    register_proof,
    get_proof,
    get_web3_and_contract,
    get_blockchain_config,
)
from pipeline.main import (
    DEFAULT_VERIFIED_THRESHOLD,
    DEFAULT_REVIEW_THRESHOLD,
    DEFAULT_MIN_QUALITY,
    classify_decision,
    evaluate_candidates_concurrently,
)

app = FastAPI(
    title="TrustLens",
    description="Face Identification & Blockchain Verification Web API",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


class VerifyRequest(BaseModel):
    proof_hash: str


class TamperRequest(BaseModel):
    proof_hash: str
    tamper_field: str = "url"  # "url", "similarity", "platform"


@app.get("/api/health")
def get_health():
    """Health check returning network, RPC and contract connectivity status."""
    cfg = get_blockchain_config()
    try:
        w3, contract = get_web3_and_contract()
        connected = w3.is_connected()
        block = w3.eth.block_number
        contract_addr = contract.address
    except Exception as e:
        connected = False
        block = 0
        contract_addr = cfg.get("contract_address") or "Not configured"

    has_serp = bool(os.environ.get("SERPAPI_KEY") and os.environ.get("SERPAPI_KEY") != "your_serpapi_key")
    has_pinata = bool(os.environ.get("PINATA_JWT") and os.environ.get("PINATA_JWT") != "your_pinata_jwt")
    has_wallet = bool(cfg.get("private_key") and cfg.get("private_key") not in ("your_testnet_wallet_private_key", "your_anvil_private_key"))

    explorer_url = ""
    if not cfg["is_local"] and contract_addr:
        explorer_url = f"https://amoy.polygonscan.com/address/{contract_addr}"

    return {
        "status": "ok" if connected else "degraded",
        "network": cfg["network_label"],
        "chain_id": cfg["chain_id"],
        "is_local": cfg["is_local"],
        "rpc_connected": connected,
        "latest_block": block,
        "contract_address": contract_addr,
        "explorer_url": explorer_url,
        "thresholds": {
            "verified_threshold": DEFAULT_VERIFIED_THRESHOLD,
            "review_threshold": DEFAULT_REVIEW_THRESHOLD,
            "min_quality": DEFAULT_MIN_QUALITY,
        },
        "services": {
            "serpapi": has_serp,
            "pinata_ipfs": has_pinata,
            "wallet": has_wallet,
        },
    }


@app.get("/api/demo-image")
def get_demo_image():
    """Return the consenting demo image path or binary."""
    demo_path = PROJECT_ROOT / "demo" / "public_face_demo.jpg"
    if not demo_path.exists():
        demo_path = PROJECT_ROOT / "demo" / "sample_face.jpg"
    if not demo_path.exists():
        raise HTTPException(status_code=404, detail="Demo image file not found")
    return FileResponse(str(demo_path), media_type="image/jpeg", filename=demo_path.name)


@app.post("/api/detect-faces")
async def detect_faces_endpoint(image: UploadFile = File(...)):
    """Upload image and detect all faces with landmarks and quality scores before search."""
    suffix = Path(image.filename).suffix or ".jpg"
    contents = await image.read()
    if len(contents) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image exceeds 15 MB limit")

    try:
        faces = detect_all_faces(contents, min_quality=0.10)
        if not faces:
            return {"face_detected": False, "face_count": 0, "faces": []}

        serializable_faces = []
        for f in faces:
            serializable_faces.append({
                "face_index": f["face_index"],
                "det_score": round(float(f["det_score"]), 4),
                "bbox": [round(float(x), 2) for x in f["bbox"]],
                "landmarks": f.get("landmarks"),
                "quality": f["quality"],
                "embedding_hash": f["embedding_hash"],
            })

        return {
            "face_detected": True,
            "face_count": len(faces),
            "faces": serializable_faces,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze")
async def analyze_and_execute_pipeline(
    image: UploadFile = File(...),
    face_index: int = Form(0),
    verified_threshold: float = Form(DEFAULT_VERIFIED_THRESHOLD),
    review_threshold: float = Form(DEFAULT_REVIEW_THRESHOLD),
    min_quality: float = Form(DEFAULT_MIN_QUALITY),
    skip_blockchain: bool = Form(False),
):
    """Execute the full real 9-stage TrustLens pipeline with in-memory candidate concurrency."""
    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    suffix = Path(image.filename).suffix or ".jpg"
    contents = await image.read()
    if len(contents) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image file exceeds 15 MB limit")

    tmp_path = None
    stages_log: list[dict[str, Any]] = []

    try:
        # [1/9] Face Analysis
        t0 = time.perf_counter()
        stages_log.append({"stage": 1, "name": "Face Detection & Quality Assessment", "status": "RUNNING"})
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(contents)
            tmp_path = f.name

        query_analysis = analyze_face(tmp_path, face_index=face_index, min_quality=min_quality)
        query_emb = query_analysis["normalized_embedding"]
        query_face_meta = query_analysis["metadata"]
        timings["1_face_detect"] = time.perf_counter() - t0
        stages_log[-1]["status"] = "SUCCESS"
        stages_log[-1]["detail"] = f"Detected {query_analysis['face_count']} face(s). Selected #{face_index} (Quality: {query_analysis['quality_score']:.2f}) [{timings['1_face_detect']:.2f}s]"

        # [2/9] Embedding Fingerprint
        t0 = time.perf_counter()
        stages_log.append({"stage": 2, "name": "ArcFace 512-d Embedding Hashing", "status": "SUCCESS", "detail": f"Embedding SHA-256: {query_analysis['embedding_hash'][:16]}..."})
        timings["2_embedding"] = time.perf_counter() - t0

        # [3/9] Pin Query Image
        t0 = time.perf_counter()
        stages_log.append({"stage": 3, "name": "IPFS Query Image Pinning", "status": "RUNNING"})
        query_cid = pin_file(tmp_path, name=f"trustlens_query_{os.path.basename(image.filename or 'face.jpg')}")
        public_url = gateway_url(query_cid)
        timings["3_ipfs_query"] = time.perf_counter() - t0
        stages_log[-1]["status"] = "SUCCESS"
        stages_log[-1]["detail"] = f"Pinned to IPFS CID: {query_cid} [{timings['3_ipfs_query']:.2f}s]"

        # [4/9] Multi-Source Search
        t0 = time.perf_counter()
        stages_log.append({"stage": 4, "name": "Multi-Source Visual Search", "status": "RUNNING"})
        candidates = reverse_image_search(public_url)
        timings["4_search_api"] = time.perf_counter() - t0
        stages_log[-1]["status"] = "SUCCESS"
        stages_log[-1]["detail"] = f"Discovered {len(candidates)} candidates across web & social platforms [{timings['4_search_api']:.2f}s]"

        # [5/9] Candidate Multi-Face Verification (Concurrent In-Memory)
        t0 = time.perf_counter()
        stages_log.append({"stage": 5, "name": "Independent Candidate Multi-Face Verification", "status": "RUNNING"})
        
        raw_eval_results, usable_count = evaluate_candidates_concurrently(
            candidates=candidates,
            query_emb=query_emb,
            verified_threshold=verified_threshold,
            review_threshold=review_threshold,
            max_workers=8,
            query_multiview=query_analysis.get("multiview"),
        )

        evaluated_candidates = []
        for r in raw_eval_results:
            c = r["candidate"]
            evaluated_candidates.append({
                "rank": c.get("search_rank", 1),
                "platform": c.get("platform", "General Web"),
                "title": c.get("title", ""),
                "link": c.get("link", ""),
                "domain": c.get("domain", ""),
                "thumbnail": c.get("thumbnail"),
                "thumbnail_sha256": r.get("thumbnail_sha256"),
                "similarity": round(float(r["similarity"]), 4),
                "quality": round(float(r["image_quality"]), 4),
                "det_confidence": round(float(r["det_confidence"]), 4),
                "face_count": r["face_count"],
                "best_face_index": r.get("best_face_index", 0),
                "decision": r["decision"],
                "reason": r["reason"],
            })

        decision_prio = {"VERIFIED": 3, "REVIEW": 2, "REJECTED": 1}
        evaluated_candidates.sort(
            key=lambda x: (decision_prio.get(x["decision"], 0), x["similarity"], x["quality"], -x.get("rank", 999)),
            reverse=True,
        )
        timings["5_cand_eval"] = time.perf_counter() - t0
        stages_log[-1]["status"] = "SUCCESS"
        stages_log[-1]["detail"] = f"Evaluated {len(evaluated_candidates)} candidate faces across {usable_count} images in {timings['5_cand_eval']:.2f}s"

        # [6/9] Candidate Ranking & Separation Margin
        t0 = time.perf_counter()
        stages_log.append({"stage": 6, "name": "Candidate Ranking & Separation Margin", "status": "RUNNING"})
        verified_matches = [x for x in evaluated_candidates if x["decision"] == "VERIFIED"]
        review_candidates = [x for x in evaluated_candidates if x["decision"] == "REVIEW"]
        rejected_candidates = [x for x in evaluated_candidates if x["decision"] == "REJECTED"]

        verified_sims = [x["similarity"] for x in verified_matches]
        rejected_sims = [x["similarity"] for x in (review_candidates + rejected_candidates)]
        margin_info = calculate_separation_margin(verified_sims, rejected_sims)
        sep_margin = margin_info["separation_margin"]

        # Compute consensus clustering & cross-candidate agreement
        consensus_data = compute_candidate_consensus(raw_eval_results, verified_threshold=verified_threshold)

        timings["6_ranking"] = time.perf_counter() - t0
        stages_log[-1]["status"] = "SUCCESS"
        stages_log[-1]["detail"] = f"Separation Margin: {sep_margin if sep_margin is not None else 'N/A'} (Verified: {len(verified_matches)}, Review: {len(review_candidates)}, Rejected: {len(rejected_candidates)})"

        # Select primary match: Prioritize verified social -> any verified -> review -> rejected
        if verified_matches:
            social_ver = [x for x in verified_matches if x["platform"] != "General Web"]
            best_match = social_ver[0] if social_ver else verified_matches[0]
        elif review_candidates:
            best_match = review_candidates[0]
        elif evaluated_candidates:
            best_match = evaluated_candidates[0]
        else:
            best_match = None

        if not best_match:
            raise HTTPException(status_code=404, detail="No faces detected in discovered candidate images")

        # Compute dynamic identity confidence
        confidence_data = calculate_dynamic_confidence(
            similarity=best_match["similarity"],
            candidate_quality=best_match["quality"],
            separation_margin=sep_margin,
            cross_result_agreement=consensus_data["agreement_ratio"],
            platform_diversity=consensus_data["platform_count"],
        )

        # [7/9] Evidence Manifest Generation (RFC-8785)
        t0 = time.perf_counter()
        stages_log.append({"stage": 7, "name": "Canonical Evidence Manifest (RFC-8785)", "status": "RUNNING"})
        matched_candidate_dict = {
            "link": best_match["link"],
            "title": best_match["title"],
            "domain": best_match["domain"],
            "platform": best_match["platform"],
            "search_rank": best_match["rank"],
            "thumbnail": best_match["thumbnail"],
        }
        manifest, manifest_hash = build_evidence_manifest(
            query_face_metadata=query_face_meta,
            candidate=matched_candidate_dict,
            similarity_score=best_match["similarity"],
            decision=best_match["decision"],
            verified_threshold=verified_threshold,
            review_threshold=review_threshold,
            decision_reason=f"Multi-source facial verification ({best_match['decision']}) against {best_match['platform']}",
            query_image_cid=query_cid,
            total_candidates=len(candidates),
            usable_images_count=usable_count,
            candidate_face_count=best_match["face_count"],
            candidate_det_confidence=best_match["det_confidence"],
            candidate_image_quality=best_match["quality"],
            query_image_quality=query_analysis["quality_score"],
            selected_face_index=face_index,
            separation_margin=sep_margin,
            margin_interpretation=margin_info.get("margin_interpretation"),
            thumbnail_sha256=best_match.get("thumbnail_sha256"),
            confidence_data=confidence_data,
            consensus_data=consensus_data,
        )
        timings["7_manifest"] = time.perf_counter() - t0
        stages_log[-1]["status"] = "SUCCESS"
        stages_log[-1]["detail"] = f"Canonical SHA-256: {manifest_hash}"

        # [8/9] IPFS Manifest Pinning
        t0 = time.perf_counter()
        stages_log.append({"stage": 8, "name": "Evidence IPFS Storage", "status": "RUNNING"})
        manifest_cid = pin_json(manifest, name=f"evidence_{manifest_hash[:12]}")
        timings["8_ipfs_manifest"] = time.perf_counter() - t0
        stages_log[-1]["status"] = "SUCCESS"
        stages_log[-1]["detail"] = f"Pinned Manifest CID: {manifest_cid} [{timings['8_ipfs_manifest']:.2f}s]"

        # [9/9] Blockchain Proof Anchoring (Strict VERIFIED Gate)
        cfg = get_blockchain_config()
        t0 = time.perf_counter()
        stages_log.append({"stage": 9, "name": f"{cfg['network_label']} Proof Anchoring", "status": "RUNNING"})
        chain_receipt = None
        if not skip_blockchain and best_match["decision"] == "VERIFIED":
            chain_receipt = register_proof(manifest_hash, manifest_cid)
            timings["9_blockchain"] = time.perf_counter() - t0
            stages_log[-1]["status"] = "SUCCESS"
            stages_log[-1]["detail"] = f"Tx Hash: {chain_receipt['tx_hash'][:16]}... (Block #{chain_receipt['block']}) [{timings['9_blockchain']:.2f}s]"
        elif best_match["decision"] != "VERIFIED":
            stages_log[-1]["status"] = "SKIPPED"
            stages_log[-1]["detail"] = f"Proof registration skipped: Decision is [{best_match['decision']}] (only VERIFIED proofs are anchored on-chain)"
        else:
            stages_log[-1]["status"] = "SKIPPED"
            stages_log[-1]["detail"] = "Offline mode requested (dry run)"

        total_latency = time.perf_counter() - t_start

        return {
            "success": True,
            "stages_log": stages_log,
            "query": {
                "face_count": query_analysis["face_count"],
                "selected_face_index": face_index,
                "det_score": query_analysis["det_score"],
                "quality_score": query_analysis["quality_score"],
                "quality_breakdown": query_analysis["quality_breakdown"],
                "embedding_hash": query_analysis["embedding_hash"],
                "bbox": query_analysis["bbox"],
                "query_cid": query_cid,
            },
            "search_summary": {
                "total_discovered": len(candidates),
                "usable_evaluated": usable_count,
                "verified_count": len(verified_matches),
                "review_count": len(review_candidates),
                "rejected_count": len(rejected_candidates),
                "separation_margin": sep_margin,
                "margin_interpretation": margin_info.get("margin_interpretation"),
                "consensus": consensus_data,
                "confidence": confidence_data,
            },
            "best_match": best_match,
            "confidence_data": confidence_data,
            "consensus_data": consensus_data,
            "verified_matches": verified_matches,
            "review_candidates": review_candidates,
            "rejected_candidates": rejected_candidates,
            "all_candidates": evaluated_candidates,
            "manifest": manifest,
            "manifest_hash": manifest_hash,
            "manifest_cid": manifest_cid,
            "blockchain_receipt": chain_receipt,
            "performance": {
                "total_latency_seconds": round(total_latency, 2),
                "timings_seconds": {k: round(v, 3) for k, v in timings.items()},
            },
        }
    except Exception as e:
        stages_log.append({"stage": len(stages_log) + 1, "name": "Pipeline Error", "status": "FAILED", "detail": str(e)})
        raise HTTPException(status_code=500, detail={"error": str(e), "stages_log": stages_log})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.get("/api/result/{proof_hash}")
def get_result_by_hash(proof_hash: str):
    """Retrieve and verify a proof directly from Polygon Amoy and IPFS."""
    clean_hash = proof_hash.strip().lower()
    if clean_hash.startswith("0x"):
        clean_hash = clean_hash[2:]

    try:
        onchain = get_proof(clean_hash)
        cid = onchain["ipfs_cid"]
        submitter = onchain["submitter"]
        ts = onchain["timestamp"]

        record = fetch_json(cid)
        recomputed = sha256_of_json(record)
        is_valid = recomputed.lower() == clean_hash.lower()

        return {
            "is_valid": is_valid,
            "blockchain_hash": clean_hash,
            "recomputed_hash": recomputed,
            "ipfs_cid": cid,
            "submitter": submitter,
            "timestamp": ts,
            "network": f"{onchain.get('network', 'Blockchain')} (Chain ID {onchain.get('chain_id', 'N/A')})",
            "chain_id": onchain.get("chain_id"),
            "contract_address": onchain.get("contract_address", ""),
            "manifest": record,
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to query proof: {e}")


@app.post("/api/verify")
def verify_proof_endpoint(req: VerifyRequest):
    """Independent on-chain cryptographic re-verification."""
    return get_result_by_hash(req.proof_hash)


@app.post("/api/tamper-test")
def tamper_test_endpoint(req: TamperRequest):
    """Simulate deliberate tampering on authentic IPFS record and test on-chain mismatch."""
    clean_hash = req.proof_hash.strip().lower()
    if clean_hash.startswith("0x"):
        clean_hash = clean_hash[2:]

    try:
        onchain = get_proof(clean_hash)
        cid = onchain["ipfs_cid"]
        original_record = fetch_json(cid)
        orig_hash = sha256_of_json(original_record)

        tampered_record = copy.deepcopy(original_record)

        if req.tamper_field == "url":
            if "candidate" in tampered_record:
                tampered_record["candidate"]["source_url"] = "https://fake-imposter-profile.example.com/spoofed"
        elif req.tamper_field == "similarity":
            if "verification" in tampered_record:
                tampered_record["verification"]["face_similarity_score"] = 0.9999
        elif req.tamper_field == "platform":
            if "candidate" in tampered_record:
                tampered_record["candidate"]["platform"] = "Forged / Manipulated Platform"

        tampered_hash = sha256_of_json(tampered_record)
        is_valid = tampered_hash.lower() == clean_hash.lower()

        return {
            "onchain_hash": clean_hash,
            "original_computed_hash": orig_hash,
            "tampered_computed_hash": tampered_hash,
            "tamper_field": req.tamper_field,
            "tamper_detected": not is_valid,
            "status": "TAMPER DETECTED (INVALID)" if not is_valid else "VALID",
            "original_manifest": original_record,
            "tampered_manifest": tampered_record,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mount static files
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def main():
    import uvicorn
    uvicorn.run("webapp.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
