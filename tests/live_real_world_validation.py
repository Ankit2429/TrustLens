"""Live Real-World Validation Suite for Hacker House Goa Task 3.

Executes genuine end-to-end pipeline runs against diverse, unseen real-world images:
- Positive cases (Publicly indexed figures)
- Negative/No-match cases (Unindexed synthetic identities)
- Group photo / multi-person images
- Difficult lighting / action pose images

Captures all empirical metrics without hardcoding or simulation.
"""
import json
import os
import sys
import time

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.main import run_pipeline


def evaluate_live_suite():
    test_files = [
        {
            "id": "POS_01",
            "name": "Satya Nadella (Public Figure)",
            "file": "demo/live_validation/01_public_figure_satya.jpg",
            "expected_type": "POSITIVE"
        },
        {
            "id": "POS_02",
            "name": "Elon Musk (Public Figure)",
            "file": "demo/live_validation/Elon_Musk_Royal_Society.jpg",
            "expected_type": "POSITIVE"
        },
        {
            "id": "DIFF_01",
            "name": "Lionel Messi (Action Pose / Match Lighting)",
            "file": "demo/live_validation/05_side_pose_diff_lighting.jpg",
            "expected_type": "POSITIVE_OR_REVIEW"
        },
        {
            "id": "GROUP_01",
            "name": "Steve Jobs & Bill Gates (Group Photo)",
            "file": "demo/live_validation/Steve_Jobs_and_Bill_Gates_(522695099).jpg",
            "expected_type": "POSITIVE"
        },
        {
            "id": "NEG_01",
            "name": "Novel Procedural Face 1 (Unindexed)",
            "file": "demo/live_validation/03_synthetic_novel_face_1.jpg",
            "expected_type": "NEGATIVE"
        },
        {
            "id": "NEG_02",
            "name": "Novel Procedural Face 2 (Unindexed)",
            "file": "demo/live_validation/03_synthetic_novel_face_2.jpg",
            "expected_type": "NEGATIVE"
        },
    ]

    results = []

    print("=" * 80)
    print("  TRUSTLENS — LIVE REAL-WORLD DEMO VALIDATION (HH GOA TASK 3)")
    print("=" * 80)
    print(f"Total Test Cases: {len(test_files)}\n")

    for tc in test_files:
        print(f"\n>>> Running Test Case [{tc['id']}] : {tc['name']}")
        print(f"    Target File: {tc['file']}")
        
        t0 = time.time()
        try:
            res = run_pipeline(tc["file"])
            elapsed = time.time() - t0
            
            best = res.get("best_match", {})
            cand = best.get("candidate", {})
            query_analysis = res.get("query_analysis", {})
            chain_rcpt = res.get("blockchain_receipt") or {}
            all_ranked = res.get("all_candidates_ranked", [])
            verified_matches = res.get("verified_matches", [])
            review_matches = res.get("review_matches", [])
            rejected_matches = res.get("rejected_matches", [])

            top_sims = [f"{c.get('similarity', 0.0):.4f}" for c in all_ranked[:5]]
            
            record = {
                "id": tc["id"],
                "name": tc["name"],
                "expected_type": tc["expected_type"],
                "face_count": query_analysis.get("face_count", 1),
                "det_score": round(float(query_analysis.get("det_score", 0.0)), 4),
                "quality_score": round(float(query_analysis.get("quality_score", 0.0)), 4),
                "search_results": res.get("total_candidates", len(all_ranked)),
                "usable_thumbnails": len(all_ranked),
                "verified_matches_count": len(verified_matches),
                "review_matches_count": len(review_matches),
                "rejected_matches_count": len(rejected_matches),
                "top_5_similarities": top_sims,
                "top_similarity": round(float(best.get("similarity", 0.0)), 4),
                "top_platform": cand.get("platform", "N/A"),
                "top_url": cand.get("link", "N/A"),
                "separation_margin": res.get("separation_margin"),
                "decision": best.get("decision", "UNKNOWN"),
                "decision_reason": best.get("reason", ""),
                "manifest_hash": res.get("manifest_hash", "N/A"),
                "ipfs_cid": res.get("manifest_cid", "N/A"),
                "onchain_tx": chain_rcpt.get("tx_hash", "N/A"),
                "onchain_confirmed": bool(chain_rcpt.get("tx_hash")),
                "elapsed_sec": round(elapsed, 2)
            }
            results.append(record)
            print(f"    -> Verdict: [{record['decision']}] | Top Sim: {record['top_similarity']} | Margin: {record['separation_margin']}")
            print(f"    -> Platform: {record['top_platform']} | URL: {record['top_url'][:60]}...")
            print(f"    -> Proof Tx: {record['onchain_tx'][:20]}... (On-Chain Confirmed: {record['onchain_confirmed']})")

        except Exception as e:
            print(f"    [-] Execution Error: {e}")
            results.append({
                "id": tc["id"],
                "name": tc["name"],
                "expected_type": tc["expected_type"],
                "decision": "ERROR",
                "decision_reason": str(e)
            })

    os.makedirs("demo/live_validation", exist_ok=True)
    with open("demo/live_validation/live_validation_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("  LIVE VALIDATION SUMMARY REPORT")
    print("=" * 80)
    for r in results:
        print(f"[{r['id']}] {r['name']}")
        print(f"  - Faces in Query : {r.get('face_count')} (Det Conf: {r.get('det_score')}, Quality: {r.get('quality_score')})")
        print(f"  - Candidates     : {r.get('usable_thumbnails')} analyzed ({r.get('verified_matches_count')} Verified, {r.get('review_matches_count')} Review, {r.get('rejected_matches_count')} Rejected)")
        print(f"  - Top-5 Sim      : {r.get('top_5_similarities')}")
        print(f"  - Top Candidate  : {r.get('top_platform')} -> {r.get('top_url')}")
        print(f"  - Margin Gap     : {r.get('separation_margin')}")
        print(f"  - Final Decision : [{r.get('decision')}]")
        print(f"  - Reason         : {r.get('decision_reason')}")
        print(f"  - Blockchain     : Confirmed={r.get('onchain_confirmed')} (Tx: {r.get('onchain_tx')})")
        print("-" * 80)

    return results


if __name__ == "__main__":
    evaluate_live_suite()
