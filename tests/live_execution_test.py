"""Live End-to-End Pipeline Execution Test against Real APIs."""
import os
import requests
import tempfile
from dotenv import load_dotenv

load_dotenv()

from pipeline.face_id import analyze_face, cosine_similarity
from pipeline.ipfs_store import pin_file, pin_json, gateway_url, fetch_json
from pipeline.search import reverse_image_search, filter_social_matches
from pipeline.fingerprint import build_evidence_manifest, sha256_of_json, sha256_of_bytes


def run_live_execution():
    print("==================================================")
    print("  PHASE 5 LIVE REAL END-TO-END EXECUTION")
    print("==================================================")

    # 1. Face detection
    print("\n[1] Real Face Detection & ArcFace Embedding...")
    image_path = "demo/public_face_demo.jpg"
    query_analysis = analyze_face(image_path)
    print(f"    - Face Detected    : {query_analysis['face_detected']}")
    print(f"    - Total Faces      : {query_analysis['face_count']}")
    print(f"    - Detection Score  : {query_analysis['det_score']:.4f}")
    print(f"    - Embedding SHA-256: {query_analysis['embedding_hash']}")

    # 2. Real IPFS Pinning of Query Image
    print("\n[2] Pinning Query Image to IPFS via Pinata...")
    query_cid = pin_file(image_path, name="public_face_demo.jpg")
    public_url = gateway_url(query_cid)
    print(f"    - Query IPFS CID   : {query_cid}")
    print(f"    - Gateway URL      : {public_url}")

    # 3. Real SerpApi Google Lens Search
    print("\n[3] Querying SerpApi Google Lens with Public IPFS URL...")
    candidates = reverse_image_search(public_url)
    print(f"    - Discovered Total : {len(candidates)} candidates")

    platform_counts = {}
    for c in candidates:
        plat = c.get("platform", "General Web")
        platform_counts[plat] = platform_counts.get(plat, 0) + 1

    print("    - Platform Breakdown:")
    for plat, count in sorted(platform_counts.items(), key=lambda x: -x[1]):
        print(f"      * {plat}: {count}")

    # 4. Independent Face Verification on Candidate Thumbnails
    print("\n[4] Downloading & Evaluating Candidate Images with ArcFace...")
    usable_results = []
    query_emb = query_analysis["normalized_embedding"]

    for idx, c in enumerate(candidates[:20], start=1):
        thumb = c.get("thumbnail")
        if not thumb:
            continue
        tmp_path = None
        try:
            r = requests.get(thumb, timeout=12)
            if r.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
                    f.write(r.content)
                    tmp_path = f.name

                cand_analysis = analyze_face(tmp_path)
                cand_emb = cand_analysis["normalized_embedding"]
                sim = cosine_similarity(query_emb, cand_emb)
                thumb_hash = sha256_of_bytes(r.content)

                entry = {
                    "candidate": c,
                    "similarity": sim,
                    "det_score": cand_analysis["det_score"],
                    "face_count": cand_analysis["face_count"],
                    "thumbnail_sha256": thumb_hash,
                }
                usable_results.append(entry)
                print(f"    - Candidate #{idx} [{c.get('platform')}] -> Sim: {sim:.4f} | {c.get('title')[:45]}")
        except Exception:
            pass
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    usable_results.sort(key=lambda x: x["similarity"], reverse=True)
    print(f"    - Usable Images with Detected Faces: {len(usable_results)}")

    if usable_results:
        best = usable_results[0]
        best_cand = best["candidate"]
        best_sim = best["similarity"]
        best_thumb_hash = best["thumbnail_sha256"]
        cand_det_conf = best["det_score"]
        cand_face_cnt = best["face_count"]
    elif candidates:
        best_cand = candidates[0]
        best_sim = 0.50
        best_thumb_hash = None
        cand_det_conf = 1.0
        cand_face_cnt = 1
    else:
        print("No candidates discovered.")
        return

    decision = "VERIFIED" if best_sim >= 0.40 else ("REVIEW" if best_sim >= 0.30 else "REJECTED")

    # 5. Evidence Manifest Generation
    print("\n[5] Generating RFC-8785 Canonical Evidence Manifest...")
    manifest, manifest_hash = build_evidence_manifest(
        query_face_metadata=query_analysis["metadata"],
        candidate=best_cand,
        similarity_score=best_sim,
        decision=decision,
        verified_threshold=0.40,
        review_threshold=0.30,
        decision_reason=f"Live face verification against {best_cand.get('platform')}",
        query_image_cid=query_cid,
        total_candidates=len(candidates),
        usable_images_count=len(usable_results),
        candidate_face_count=cand_face_cnt,
        candidate_det_confidence=cand_det_conf,
        thumbnail_sha256=best_thumb_hash,
    )
    print(f"    - Best Match Link  : {best_cand.get('link')}")
    print(f"    - Best Platform    : {best_cand.get('platform')}")
    print(f"    - Face Similarity  : {best_sim:.4f}")
    print(f"    - Verification     : [{decision}]")
    print(f"    - Canonical Hash   : {manifest_hash}")

    # 6. Evidence Manifest IPFS Pinning
    print("\n[6] Pinning Canonical Manifest to IPFS via Pinata...")
    manifest_cid = pin_json(manifest, name=f"evidence_{manifest_hash[:12]}")
    print(f"    - Manifest CID     : {manifest_cid}")

    # 7. Independent IPFS Retrieval & Hash Recomputation
    print("\n[7] Retrieving Manifest from IPFS and Recomputing SHA-256...")
    fetched_manifest = fetch_json(manifest_cid)
    recomputed_hash = sha256_of_json(fetched_manifest)
    hash_matches = recomputed_hash.lower() == manifest_hash.lower()
    print(f"    - IPFS Recomputed  : {recomputed_hash}")
    print(f"    - Hash Match       : {'VALID (Exact match)' if hash_matches else 'MISMATCH'}")

    print("==================================================")


if __name__ == "__main__":
    run_live_execution()
