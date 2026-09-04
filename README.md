# TrustLens — Face Identification & Blockchain Verification

[![Tests](https://img.shields.io/badge/pytest-28%20passed-brightgreen.svg)](tests/)
[![Network](https://img.shields.io/badge/Polygon-Amoy%20(80002)-8247E5.svg)](https://amoy.polygonscan.com/)
[![Storage](https://img.shields.io/badge/IPFS-Pinata-E11D48.svg)](https://pinata.cloud/)
[![Model](https://img.shields.io/badge/InsightFace-buffalo__l%20ArcFace%20512--d-blue.svg)](https://github.com/deepinsight/insightface)

---

## 1. Overview & Problem Statement

In an era of digital media proliferation, establishing verifiable cryptographic provenance for discovered public web/social content is critical. When searching for public visual matches of a face image, conventional search engines cannot be treated as trusted arbiters of identity, and central web hosts can modify or delete evidence without record.

**This project implements a decentralized, tamper-evident verification pipeline:**
1. Generates complete, normalized 512-dimensional ArcFace face embeddings and deterministic cryptographic fingerprints.
2. Discovers matching public web and social media candidates via multi-source reverse visual discovery (Google Lens via SerpApi).
3. Independently downloads and executes local face detection and ArcFace cosine similarity verification on candidate images (never blindly trusting search engine ranking).
4. Classifies matches using a configurable multi-tier decision engine (`VERIFIED`, `REVIEW`, `REJECTED`).
5. Generates an RFC-8785 canonical JSON evidence manifest capturing full discovery and verification provenance.
6. Pins the canonical manifest to IPFS for decentralized content-addressed storage.
7. Anchors the cryptographic SHA-256 hash to the **Polygon Amoy testnet** using a Solidity smart contract (`ProofRegistry.sol`).
8. Provides standalone independent re-verification (`verify.py`) and a cryptographic tamper demonstration (`tamper_demo.py`).

---

## 2. System Architecture & Pipeline

```
[ QUERY IMAGE ]
       │
       ▼
[1] Face Detection & Quality Assessment (SCRFD / RetinaFace)
       │
       ▼
[2] Full 512-d ArcFace Embedding & Normalized SHA-256 Hashing
       │
       ▼
[3] Query Image IPFS Pinning (Pinata Gateway for Search)
       │
       ▼
[4] Multi-Source & Multi-Platform Web Discovery (Google Lens via SerpApi)
       │  (Instagram, LinkedIn, Facebook, X, Reddit, YouTube, TikTok, Pinterest, News/Media, General Web)
       ▼
[5] Independent Candidate Face Verification (InsightFace ArcFace on CPU)
       │
       ▼
[6] Candidate Ranking & Multi-Tier Decision Engine (VERIFIED / REVIEW / REJECTED)
       │
       ▼
[7] Canonical Evidence Manifest Generation (RFC-8785 Deterministic JSON)
       │
       ▼
[8] SHA-256 Cryptographic Fingerprint & IPFS Storage
       │
       ▼
[9] Polygon Amoy Smart Contract Proof Anchoring (`registerProof`)
       │
       ▼
[ INDEPENDENT RE-VERIFICATION & TAMPER DETECTION ] (`verify.py` / `tamper_demo.py`)
```

---

## 3. Technology Stack

- **Face Recognition Engine**: [InsightFace](https://github.com/deepinsight/insightface) `buffalo_l` (SCRFD detector + ArcFace 512-dimensional embedding extractor) running locally on CPU with ONNX Runtime.
- **Reverse Visual Discovery**: SerpApi Google Lens engine (discovers indexed public web pages, blogs, and social platforms).
- **Decentralized Storage**: IPFS via [Pinata](https://pinata.cloud/) with multi-gateway fallback resolution (Pinata, Cloudflare, IPFS.io, dweb.link).
- **Blockchain**: Polygon Amoy Testnet (Chain ID: `80002`, Sepolia-anchored EVM testnet).
- **Smart Contract & Tooling**: Solidity `0.8.20`, Hardhat, and `web3.py` (with dynamic EIP-1559 gas fee estimation).
- **Testing**: `pytest` (28 unit tests covering vector math, deterministic hashing, manifest schema, and contract logic).

---

## 4. Setup & Installation

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm

### 1. Clone & Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
npm install
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and configure your credentials:
```bash
cp .env.example .env
```

Edit `.env`:
```env
# Search Provider (SerpApi)
SERPAPI_KEY=your_serpapi_key

# IPFS Pinning (Pinata)
PINATA_JWT=your_pinata_jwt

# Blockchain: Polygon Amoy Testnet (Chain ID 80002)
AMOY_RPC_URL=https://rpc-amoy.polygon.technology/
PRIVATE_KEY=your_testnet_wallet_private_key
CONTRACT_ADDRESS=your_deployed_contract_address

# Configurable Decision Thresholds
VERIFIED_THRESHOLD=0.40
REVIEW_THRESHOLD=0.30
```

### 3. Deploy Smart Contract to Polygon Amoy
1. Obtain free Polygon Amoy testnet tokens (POL) from the [Polygon Faucet](https://faucet.polygon.technology/).
2. Compile and deploy the `ProofRegistry` contract:
```bash
npm run compile
npm run deploy:amoy
```
3. Copy the outputted contract address into `.env` under `CONTRACT_ADDRESS`.

---

## 5. Usage & Demonstration Commands

### A. Run Full Pipeline
To run the complete 9-stage pipeline on a face image:
```bash
python -m pipeline.main demo/sample_face.jpg
```

#### Optional CLI Arguments:
- `--verified-threshold <float>`: Custom similarity cutoff for `VERIFIED` status (default: `0.40`).
- `--review-threshold <float>`: Custom similarity cutoff for `REVIEW` status (default: `0.30`).
- `--skip-blockchain`: Run visual search, face verification, and IPFS pinning without submitting an on-chain transaction (dry run / offline test mode).

### B. Independent On-Chain Re-Verification
Anyone can re-verify the authenticity and integrity of a previously registered proof directly from Polygon Amoy and IPFS without reusing memory from the main pipeline:
```bash
python verify.py <evidence_sha256_hash>
```

**Example output:**
```text
======================================================================
  INDEPENDENT PROOF RE-VERIFICATION (POLYGON AMOY + IPFS)
======================================================================
  BLOCKCHAIN HASH : a3f48c909e235be8...
  LOCAL HASH      : a3f48c909e235be8...
  IPFS CID        : QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG
  NETWORK         : Polygon Amoy (Chain ID 80002)
  SUBMITTER       : 0x71C...
  TIMESTAMP       : 2026-09-04 08:30:00 UTC
  RESULT          : VALID
======================================================================
```

### C. Cryptographic Tamper-Evidence Demo
Demonstrate that modifying even a single character in the candidate source URL or face similarity score breaks cryptographic validation against Polygon Amoy:
```bash
python tamper_demo.py <evidence_sha256_hash>
```

**Example output:**
```text
===========================================================================
  TAMPER DETECTION RESULTS
===========================================================================
  [A] ORIGINAL RECORD VERIFICATION:
      Blockchain Hash : a3f48c909e235be8...
      Local Hash      : a3f48c909e235be8...
      Result          : VALID (Bit-for-bit cryptographic match)

  [B] TAMPERED RECORD VERIFICATION:
      Blockchain Hash : a3f48c909e235be8...
      Tampered Hash   : 7d1b329cf551e4aa...
      Result          : TAMPER DETECTED!
===========================================================================
```

### D. Run Automated Test Suite
Run all unit tests offline without requiring external API keys:
```bash
pytest tests/ -v
```

---

## 6. Decision Engine & Threshold Calibration

ArcFace cosine similarities for Buffalo_l embeddings typically exhibit:
- **Same individual under varied pose/lighting**: `0.40` to `0.85+`
- **Different individuals / random pairs**: `< 0.25`
- **Ambiguous / partial match**: `0.30` to `0.40`

### Multi-Tier Decision System:
| Status | Condition | Meaning |
| :--- | :--- | :--- |
| **`VERIFIED`** | `similarity >= VERIFIED_THRESHOLD` (default 0.40) | High-confidence face similarity confirming discovery match. |
| **`REVIEW`** | `0.30 <= similarity < 0.40` | Moderate face similarity; flagged for manual inspection. |
| **`REJECTED`** | `similarity < REVIEW_THRESHOLD` (default 0.30) | Insufficient facial similarity; rejected as non-matching. |

*Note: Thresholds are configurable heuristics tuned for demonstration sensitivity and should not be treated as absolute biometric identity certifications.*

---

## 7. Deterministic Evidence Manifest Schema (RFC-8785)

The evidence manifest captures complete provenance:
```json
{
  "schema_version": "1.0.0",
  "record_timestamp": 1741160000,
  "face": {
    "algorithm": "InsightFace buffalo_l",
    "model": "ArcFace",
    "dimension": 512,
    "dtype": "float32",
    "normalized": true,
    "embedding_hash": "64_hex_character_sha256_of_512d_float32_bytes"
  },
  "search": {
    "provider": "SerpApi",
    "engine": "google_lens",
    "query_image_cid": "QmQueryCID...",
    "discovery_timestamp": 1741160000,
    "total_candidates_discovered": 18,
    "usable_images_evaluated": 12
  },
  "candidate": {
    "source_url": "https://www.instagram.com/p/...",
    "source_title": "Post Title / Caption",
    "domain": "instagram.com",
    "platform": "Instagram",
    "search_rank": 1,
    "thumbnail_url": "https://...",
    "thumbnail_sha256": "64_hex_thumbnail_hash"
  },
  "verification": {
    "face_similarity_score": 0.8642,
    "verified_threshold": 0.40,
    "review_threshold": 0.30,
    "decision": "VERIFIED",
    "decision_reason": "High ArcFace embedding cosine similarity (0.8642 >= 0.40)",
    "face_detection_confidence": 0.9821,
    "candidate_face_count": 1
  },
  "integrity": {
    "canonicalization_method": "RFC-8785 canonical JSON (sorted keys, compact separators, UTF-8)",
    "sha256_hash": "manifest_root_sha256"
  }
}
```

---

## 8. Smart Contract Design (`ProofRegistry.sol`)

The `ProofRegistry` contract maintains an immutable registry of cryptographic evidence:
```solidity
struct Proof {
    address submitter;
    string ipfsCID;
    uint256 timestamp;
}
mapping(bytes32 => Proof) public proofs;
```
- **Anti-Overwrite Protection**: Prevents overwriting previously anchored evidence proofs (`require(proofs[dataHash].timestamp == 0, "Already registered")`).
- **Audit Event**: Emits `event ProofRegistered(bytes32 indexed dataHash, string ipfsCID, address indexed submitter, uint256 timestamp)`.
- **Minimal Footprint**: Only cryptographic hashes and CIDs are stored on-chain to minimize gas costs and avoid placing unnecessary personal data on a public ledger.

---

## 9. Privacy, Ethical Scope & Known Limitations

- **Public / Indexed Content Only**: The pipeline exclusively evaluates publicly discoverable web results indexed by visual search engines.
- **No Access Control Bypasses**: Does not attempt to bypass authentication, paywalls, CAPTCHAs, or private social profiles.
- **Demonstration Scope**: Tested strictly against consenting subjects (team members' public photos).
- **Proof-of-Existence / Integrity**: Demonstrates cryptographic integrity and proof-of-existence of public evidence, **not legal identity certification**.
- **Rate Caps**: Subject to standard public rate limits for SerpApi and Pinata testnet tiers.

---

## 10. Repository Structure

```
├── contracts/
│   └── ProofRegistry.sol       # Solidity smart contract for proof registry
├── scripts/
│   └── deploy.js               # Hardhat deployment script for Polygon Amoy
├── pipeline/
│   ├── __init__.py
│   ├── face_id.py              # InsightFace ArcFace 512-d embeddings & normalization
│   ├── search.py               # SerpApi Google Lens discovery & platform classifier
│   ├── fingerprint.py          # RFC-8785 canonical evidence manifest & SHA-256
│   ├── ipfs_store.py           # Pinata IPFS integration with multi-gateway fallback
│   ├── chain.py                # Web3.py Polygon Amoy interaction & EIP-1559 gas
│   └── main.py                 # 9-stage CLI orchestrator
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py        # 28-test automated pytest suite
├── demo/
│   ├── README.md
│   └── sample_face.jpg         # Sample test image for live verification
├── verify.py                   # Standalone independent proof re-verification
├── tamper_demo.py              # Cryptographic tamper detection demonstration
├── hardhat.config.js           # Hardhat network configuration (Polygon Amoy)
├── package.json                # Node dependencies & compile/deploy scripts
├── requirements.txt            # Python dependencies
└── README.md                   # System documentation
```

