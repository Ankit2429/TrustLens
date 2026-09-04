"""TrustLens FastAPI Web Application Server.

Exposes REST endpoints that directly invoke the real TrustLens pipeline,
verification logic, and tamper detection against Polygon Amoy and IPFS.
"""
import copy
import os
import sys
import tempfile
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
    calculate_separation_margin,
    sha256_of_json,
    sha256_of_bytes,
)
from pipeline.ipfs_store import pin_file, pin_json, gateway_url, fetch_json
from pipeline.chain import register_proof, get_proof, get_web3_and_contract

app = FastAPI(
    title="TrustLens",
    description="Face Identification & Blockchain Verification Web API",
    version="1.1.0",
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
    try:
        w3, contract = get_web3_and_contract()
        connected = w3.is_connected()
        block = w3.eth.block_number
        contract_addr = contract.address
    except Exception as e:
        connected = False
        block = 0
        contract_addr = os.environ.get("CONTRACT_ADDRESS", "Not configured")

    has_serp = bool(os.environ.get("SERPAPI_KEY") and os.environ.get("SERPAPI_KEY") != "your_serpapi_key")
    has_pinata = bool(os.environ.get("PINATA_JWT") and os.environ.get("PINATA_JWT") != "your_pinata_jwt")
    has_wallet = bool(os.environ.get("PRIVATE_KEY") and os.environ.get("PRIVATE_KEY") != "your_testnet_wallet_private_key")

    return {
        "status": "ok" if connected else "degraded",
        "network": "Polygon Amoy",
        "chain_id": 80002,
        "rpc_connected": connected,
        "latest_block": block,
        "contract_address": contract_addr,
        "explorer_url": f"https://amoy.polygonscan.com/address/{contract_addr}" if contract_addr else "",
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

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(contents)
            tmp_path = f.name

        faces = detect_all_faces(tmp_path, min_quality=0.10)
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
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.post("/api/analyze")
async def analyze_and_execute_pipeline(
    image: UploadFile = File(...),
    face_index: int = Form(0),
    verified_threshold: float = Form(0.40),
    review_threshold: float = Form(0.30),
    min_quality: float = Form(0.20),
    skip_blockchain: bool = Form(False),
):
    """Execute the full real 9-stage TrustLens pipeline for an uploaded face image."""
    suffix = Path(image.filename).suffix or ".jpg"
    contents = await image.read()
    if len(contents) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image file exceeds 15 MB limit")

    tmp_path = None
    stages_log: list[dict[str, Any]] = []

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(contents)
            tmp_path = f.name

        # [1/9] Face Analysis
        stages_log.append({"stage": 1, "name": "Face Detection & Quality Assessment", "status": "RUNNING"})
        query_analysis = analyze_face(tmp_path, face_index=face_index, min_quality=min_quality)
        query_emb = query_analysis["normalized_embedding"]
        query_face_meta = query_analysis["metadata"]
        stages_log[-1]["status"] = "SUCCESS"
        stages_log[-1]["detail"] = f"Detected {query_analysis['face_count']} face(s). Selected #{face_index} (Quality: {query_analysis['quality_score']:.2f})"

        # [2/9] Embedding Fingerprint
        stages_log.append({"stage": 2, "name": "ArcFace 512-d Embedding Hashing", "status": "SUCCESS", "detail": f"Embedding SHA-256: {query_analysis['embedding_hash'][:16]}..."})

        # [3/9] Pin Query Image
        stages_log.append({"stage": 3, "name": "IPFS Query Image Pinning", "status": "RUNNING"})
        query_cid = pin_file(tmp_path, name=f"trustlens_query_{os.path.basename(image.filename or 'face.jpg')}")
        public_url = gateway_url(query_cid)
        stages_log[-1]["status"] = "SUCCESS"
        stages_log[-1]["detail"] = f"Pinned to IPFS CID: {query_cid}"

        # [4/9] Multi-Source Search
        stages_log.append({"stage": 4, "name": "Multi-Source Visual Search", "status": "RUNNING"})
        candidates = reverse_image_search(public_url)
        stages_log[-1]["status"] = "SUCCESS"
        stages_log[-1]["detail"] = f"Discovered {len(candidates)} candidates across web & social platforms"

        # [5/9] Candidate Multi-Face Verification
        stages_log.append({"stage": 5, "name": "Independent Candidate Multi-Face Verification", "status": "RUNNING"})
        evaluated_candidates = []
        usable_count = 0
        for idx, c in enumerate(candidates, start=1):
            thumb = c.get("thumbnail")
            if not thumb:
                continue
            cand_tmp = None
            try:
                import requests
                r = requests.get(thumb, timeout=12)
                if r.status_code == 200:
                    usable_count += 1
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf:
                        tf.write(r.content)
                        cand_tmp = tf.name

                    cand_faces = detect_all_faces(cand_tmp, min_quality=0.15)
                    if cand_faces:
                        group_eval = compare_group_faces(query_emb, cand_faces)
                        best_sim = group_eval["best_similarity"]
                        best_cand_face = group_eval["best_candidate_face"]
                        cand_q = best_cand_face["quality"]["overall_quality"]

                        decision = "VERIFIED" if best_sim >= verified_threshold else ("REVIEW" if best_sim >= review_threshold else "REJECTED")

                        evaluated_candidates.append({
                            "rank": c.get("search_rank", idx),
                            "platform": c.get("platform", "General Web"),
                            "title": c.get("title", ""),
                            "link": c.get("link", ""),
                            "domain": c.get("domain", ""),
                            "thumbnail": thumb,
                            "thumbnail_sha256": sha256_of_bytes(r.content),
                            "similarity": round(best_sim, 4),
                            "quality": round(cand_q, 4),
                            "det_confidence": round(float(best_cand_face["det_score"]), 4),
                            "face_count": group_eval["evaluated_face_count"],
                            "decision": decision,
                        })
            except Exception:
                pass
            finally:
                if cand_tmp and os.path.exists(cand_tmp):
                    try:
                        os.remove(cand_tmp)
                    except Exception:
                        pass

        decision_prio = {"VERIFIED": 3, "REVIEW": 2, "REJECTED": 1}
        evaluated_candidates.sort(
            key=lambda x: (decision_prio.get(x["decision"], 0), x["similarity"], x["quality"]),
            reverse=True,
        )
        stages_log[-1]["status"] = "SUCCESS"
        stages_log[-1]["detail"] = f"Independently evaluated {len(evaluated_candidates)} candidate thumbnail faces"

        # [6/9] Candidate Ranking & Separation Margin
        stages_log.append({"stage": 6, "name": "Candidate Ranking & Separation Margin", "status": "RUNNING"})
        verified_sims = [x["similarity"] for x in evaluated_candidates if x["decision"] == "VERIFIED"]
        rejected_sims = [x["similarity"] for x in evaluated_candidates if x["decision"] in ("REVIEW", "REJECTED")]
        margin_info = calculate_separation_margin(verified_sims, rejected_sims)
        sep_margin = margin_info["separation_margin"]
        stages_log[-1]["status"] = "SUCCESS"
        stages_log[-1]["detail"] = f"Separation Margin: {sep_margin if sep_margin is not None else 'N/A'}"

        # Select primary match (prioritize verified social platforms if found)
        social_ver = [x for x in evaluated_candidates if x["platform"] != "General Web" and x["decision"] == "VERIFIED"]
        best_match = social_ver[0] if social_ver else (evaluated_candidates[0] if evaluated_candidates else None)

        if not best_match:
            raise HTTPException(status_code=404, detail="No faces detected in discovered candidate images")

        # [7/9] Evidence Manifest Generation (RFC-8785)
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
            decision_reason=f"Multi-source facial verification against {best_match['platform']}",
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
        )
        stages_log[-1]["status"] = "SUCCESS"
        stages_log[-1]["detail"] = f"Canonical SHA-256: {manifest_hash}"

        # [8/9] IPFS Manifest Pinning
        stages_log.append({"stage": 8, "name": "Evidence IPFS Storage", "status": "RUNNING"})
        manifest_cid = pin_json(manifest, name=f"evidence_{manifest_hash[:12]}")
        stages_log[-1]["status"] = "SUCCESS"
        stages_log[-1]["detail"] = f"Pinned Manifest CID: {manifest_cid}"

        # [9/9] Polygon Amoy Proof Anchoring
        stages_log.append({"stage": 9, "name": "Polygon Amoy Proof Anchoring", "status": "RUNNING"})
        chain_receipt = None
        if not skip_blockchain:
            chain_receipt = register_proof(manifest_hash, manifest_cid)
            stages_log[-1]["status"] = "SUCCESS"
            stages_log[-1]["detail"] = f"Tx Hash: {chain_receipt['tx_hash'][:16]}... (Block #{chain_receipt['block']})"
        else:
            stages_log[-1]["status"] = "SKIPPED"
            stages_log[-1]["detail"] = "Offline mode requested (dry run)"

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
                "verified_count": len(verified_sims),
                "rejected_count": len(rejected_sims),
                "separation_margin": sep_margin,
                "margin_interpretation": margin_info.get("margin_interpretation"),
            },
            "best_match": best_match,
            "all_candidates": evaluated_candidates[:30],
            "manifest": manifest,
            "manifest_hash": manifest_hash,
            "manifest_cid": manifest_cid,
            "blockchain_receipt": chain_receipt,
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
            "network": "Polygon Amoy (Chain ID 80002)",
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
