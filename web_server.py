"""TrustLens Web Application Server.

Provides a modern web UI and JSON API endpoints for live face analysis,
multi-platform candidate discovery, evidence manifest generation,
Polygon Amoy blockchain anchoring, standalone verification, and tamper detection.

Usage:
    python web_server.py [--port 8080]
"""
import argparse
import base64
import cgi
import io
import json
import os
import sys
import tempfile
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

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

WEB_DIR = Path(__file__).resolve().parent / "web"


class TrustLensHTTPRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/status":
            self._handle_status()
        elif parsed.path == "/api/verify":
            query = urllib.parse.parse_qs(parsed.query)
            evidence_hash = query.get("hash", [""])[0]
            self._handle_verify(evidence_hash)
        elif parsed.path == "/api/demo-image":
            self._handle_demo_image()
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/analyze":
            self._handle_analyze()
        elif parsed.path == "/api/pipeline":
            self._handle_full_pipeline()
        elif parsed.path == "/api/tamper":
            self._handle_tamper()
        else:
            self._send_json({"error": "Endpoint not found"}, status=404)

    def _handle_status(self):
        """Return system and blockchain connectivity status."""
        try:
            w3, contract = get_web3_and_contract()
            block = w3.eth.block_number
            connected = w3.is_connected()
            contract_addr = contract.address
        except Exception as e:
            connected = False
            block = 0
            contract_addr = os.environ.get("CONTRACT_ADDRESS", "Not configured")

        has_serp = bool(os.environ.get("SERPAPI_KEY") and os.environ.get("SERPAPI_KEY") != "your_serpapi_key")
        has_pinata = bool(os.environ.get("PINATA_JWT") and os.environ.get("PINATA_JWT") != "your_pinata_jwt")
        has_wallet = bool(os.environ.get("PRIVATE_KEY") and os.environ.get("PRIVATE_KEY") != "your_testnet_wallet_private_key")

        self._send_json({
            "polygon_amoy": {
                "connected": connected,
                "chain_id": 80002,
                "latest_block": block,
                "contract_address": contract_addr,
                "network_name": "Polygon Amoy Testnet",
                "explorer_url": f"https://amoy.polygonscan.com/address/{contract_addr}" if contract_addr else "",
            },
            "credentials": {
                "serpapi_ready": has_serp,
                "pinata_ready": has_pinata,
                "wallet_ready": has_wallet,
            },
            "thresholds": {
                "verified": float(os.environ.get("VERIFIED_THRESHOLD", "0.40")),
                "review": float(os.environ.get("REVIEW_THRESHOLD", "0.30")),
                "min_quality": float(os.environ.get("MIN_QUALITY_THRESHOLD", "0.20")),
            },
        })

    def _handle_demo_image(self):
        """Return the public demo face image as base64."""
        demo_path = Path("demo/public_face_demo.jpg")
        if not demo_path.exists():
            demo_path = Path("demo/sample_face.jpg")
        if demo_path.exists():
            with open(demo_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            self._send_json({"image_base64": f"data:image/jpeg;base64,{b64}", "filename": demo_path.name})
        else:
            self._send_json({"error": "Demo image not found"}, status=404)

    def _read_post_body(self) -> dict:
        content_len = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_len)
        try:
            return json.loads(post_data.decode("utf-8"))
        except Exception:
            return {}

    def _handle_analyze(self):
        """Analyze an uploaded base64 image or path."""
        data = self._read_post_body()
        img_b64 = data.get("image_base64")
        if not img_b64:
            self._send_json({"error": "No image provided"}, status=400)
            return

        # Decode base64
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]
        raw_bytes = base64.b64decode(img_b64)

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
                f.write(raw_bytes)
                tmp_path = f.name

            faces = detect_all_faces(tmp_path, min_quality=0.15)
            if not faces:
                self._send_json({"face_detected": False, "face_count": 0, "faces": []})
                return

            serializable_faces = []
            for f in faces:
                serializable_faces.append({
                    "face_index": f["face_index"],
                    "det_score": round(float(f["det_score"]), 4),
                    "bbox": [round(float(x), 2) for x in f["bbox"]],
                    "landmarks": f.get("landmarks"),
                    "quality": f["quality"],
                    "embedding_hash": f["embedding_hash"],
                    "metadata": f["metadata"],
                })

            self._send_json({
                "face_detected": True,
                "face_count": len(faces),
                "faces": serializable_faces,
            })
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _handle_full_pipeline(self):
        """Execute full pipeline for uploaded image."""
        data = self._read_post_body()
        img_b64 = data.get("image_base64")
        face_index = int(data.get("face_index", 0))
        verified_threshold = float(data.get("verified_threshold", os.environ.get("VERIFIED_THRESHOLD", "0.40")))
        review_threshold = float(data.get("review_threshold", os.environ.get("REVIEW_THRESHOLD", "0.30")))
        min_quality = float(data.get("min_quality", os.environ.get("MIN_QUALITY_THRESHOLD", "0.20")))
        skip_chain = bool(data.get("skip_blockchain", False))

        if not img_b64:
            self._send_json({"error": "No image provided"}, status=400)
            return

        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]
        raw_bytes = base64.b64decode(img_b64)

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
                f.write(raw_bytes)
                tmp_path = f.name

            # 1. Face analysis
            query_analysis = analyze_face(tmp_path, face_index=face_index, min_quality=min_quality)
            query_emb = query_analysis["normalized_embedding"]
            query_meta = query_analysis["metadata"]

            # 2. Pin query image
            query_cid = pin_file(tmp_path, name="trustlens_query_face.jpg")
            public_url = gateway_url(query_cid)

            # 3. Search
            candidates = reverse_image_search(public_url)

            # 4. Evaluate candidates
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

            verified_sims = [x["similarity"] for x in evaluated_candidates if x["decision"] == "VERIFIED"]
            rejected_sims = [x["similarity"] for x in evaluated_candidates if x["decision"] in ("REVIEW", "REJECTED")]
            margin_info = calculate_separation_margin(verified_sims, rejected_sims)
            sep_margin = margin_info["separation_margin"]

            # Select primary match (prioritize social platforms if verified)
            social_ver = [x for x in evaluated_candidates if x["platform"] != "General Web" and x["decision"] == "VERIFIED"]
            best_match = social_ver[0] if social_ver else (evaluated_candidates[0] if evaluated_candidates else None)

            if not best_match:
                self._send_json({"error": "No faces detected in candidates"}, status=400)
                return

            # Build Manifest
            matched_candidate_dict = {
                "link": best_match["link"],
                "title": best_match["title"],
                "domain": best_match["domain"],
                "platform": best_match["platform"],
                "search_rank": best_match["rank"],
                "thumbnail": best_match["thumbnail"],
            }
            manifest, manifest_hash = build_evidence_manifest(
                query_face_metadata=query_meta,
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

            # Pin manifest
            manifest_cid = pin_json(manifest, name=f"evidence_{manifest_hash[:12]}")

            # Blockchain proof
            chain_receipt = None
            if not skip_chain:
                chain_receipt = register_proof(manifest_hash, manifest_cid)

            self._send_json({
                "query": {
                    "face_count": query_analysis["face_count"],
                    "selected_face_index": face_index,
                    "quality": query_analysis["quality_score"],
                    "quality_breakdown": query_analysis["quality_breakdown"],
                    "embedding_hash": query_analysis["embedding_hash"],
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
                "all_candidates": evaluated_candidates[:25],
                "manifest": manifest,
                "manifest_hash": manifest_hash,
                "manifest_cid": manifest_cid,
                "blockchain_receipt": chain_receipt,
            })
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _handle_verify(self, evidence_hash: str):
        """Perform on-chain and IPFS verification for a hash."""
        if not evidence_hash:
            self._send_json({"error": "Hash parameter is required"}, status=400)
            return

        clean_hash = evidence_hash.strip().lower()
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

            self._send_json({
                "is_valid": is_valid,
                "blockchain_hash": clean_hash,
                "recomputed_hash": recomputed,
                "ipfs_cid": cid,
                "submitter": submitter,
                "timestamp": ts,
                "network": "Polygon Amoy (Chain ID 80002)",
                "manifest": record,
            })
        except Exception as e:
            self._send_json({"error": str(e), "is_valid": False}, status=404)

    def _handle_tamper(self):
        """Simulate tampering against an on-chain proof."""
        data = self._read_post_body()
        evidence_hash = data.get("hash", "").strip().lower()
        tamper_type = data.get("tamper_type", "url")  # url, similarity, platform

        if evidence_hash.startswith("0x"):
            evidence_hash = evidence_hash[2:]

        try:
            onchain = get_proof(evidence_hash)
            cid = onchain["ipfs_cid"]
            original_record = fetch_json(cid)
            orig_hash = sha256_of_json(original_record)

            import copy
            tampered_record = copy.deepcopy(original_record)

            if tamper_type == "url":
                if "candidate" in tampered_record:
                    tampered_record["candidate"]["source_url"] = "https://fake-imposter-profile.example.com/spoofed"
            elif tamper_type == "similarity":
                if "verification" in tampered_record:
                    tampered_record["verification"]["face_similarity_score"] = 0.9999
            elif tamper_type == "platform":
                if "candidate" in tampered_record:
                    tampered_record["candidate"]["platform"] = "Manipulated Platform"

            tampered_hash = sha256_of_json(tampered_record)
            is_tampered_valid = tampered_hash.lower() == evidence_hash.lower()

            self._send_json({
                "original_hash": orig_hash,
                "tampered_hash": tampered_hash,
                "onchain_hash": evidence_hash,
                "tamper_type": tamper_type,
                "tamper_detected": not is_tampered_valid,
                "original_record": original_record,
                "tampered_record": tampered_record,
            })
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)


def run_server(port: int = 8080):
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    server_address = ("", port)
    httpd = HTTPServer(server_address, TrustLensHTTPRequestHandler)
    print("=" * 70)
    print(f"  TRUSTLENS WEB APPLICATION SERVER RUNNING")
    print(f"  URL: http://localhost:{port}")
    print("=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutting down...")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrustLens Web Application Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to run web server on (default: 8080)")
    args = parser.parse_args()
    run_server(port=args.port)
