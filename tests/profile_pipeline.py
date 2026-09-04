"""Profile each pipeline stage to find exact latency bottlenecks."""
import os
import sys
import time
from pathlib import Path
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.face_id import analyze_face, detect_all_faces, compare_group_faces
from pipeline.search import reverse_image_search
from pipeline.ipfs_store import pin_file, pin_json, gateway_url
from pipeline.chain import register_proof, get_proof

load_dotenv()

image_path = "demo/public_face_demo.jpg"

print("--- PROFILING TRUSTLENS PIPELINE STAGES ---")
t0 = time.perf_counter()

# Stage 1
s1_t0 = time.perf_counter()
query_analysis = analyze_face(image_path)
query_emb = query_analysis["normalized_embedding"]
s1_dur = time.perf_counter() - s1_t0
print(f"Stage 1: Face Detection & Quality: {s1_dur:.3f} s")

# Stage 2 (pin query image)
s2_t0 = time.perf_counter()
query_cid = pin_file(image_path, name="profile_query.jpg")
public_url = gateway_url(query_cid)
s2_dur = time.perf_counter() - s2_t0
print(f"Stage 2: IPFS Pin Query Image: {s2_dur:.3f} s")

# Stage 3 (SerpApi search)
s3_t0 = time.perf_counter()
candidates = reverse_image_search(public_url)
s3_dur = time.perf_counter() - s3_t0
print(f"Stage 3: SerpApi Google Lens Search: {s3_dur:.3f} s ({len(candidates)} candidates)")

# Stage 4 (Candidate downloads: sequential vs concurrent)
s4_seq_t0 = time.perf_counter()
usable_seq = 0
for idx, c in enumerate(candidates[:25]):
    thumb = c.get("thumbnail")
    if not thumb:
        continue
    try:
        r = requests.get(thumb, timeout=5)
        if r.status_code == 200:
            usable_seq += 1
    except Exception:
        pass
s4_seq_dur = time.perf_counter() - s4_seq_t0
print(f"Stage 4 (25 Sequential Downloads): {s4_seq_dur:.3f} s (Usable: {usable_seq})")

# Concurrent download test with ThreadPoolExecutor & Session
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter

s4_par_t0 = time.perf_counter()
session = requests.Session()
adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=1)
session.mount("https://", adapter)
session.mount("http://", adapter)

def fetch_thumb(c):
    thumb = c.get("thumbnail")
    if not thumb:
        return None
    try:
        r = session.get(thumb, timeout=5)
        if r.status_code == 200:
            return (c, r.content)
    except Exception:
        pass
    return None

with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(fetch_thumb, candidates[:25]))
usable_par = sum(1 for r in results if r is not None)
s4_par_dur = time.perf_counter() - s4_par_t0
print(f"Stage 4 (25 Concurrent Downloads with Session Pool): {s4_par_dur:.3f} s (Usable: {usable_par})")
print(f"Speedup for downloads: {s4_seq_dur / s4_par_dur:.2f}x faster!")

total_dur = time.perf_counter() - t0
print(f"Total profiling run time: {total_dur:.3f} s")
