# TrustLens — Face Identification & Blockchain Verification

[![Tests](https://img.shields.io/badge/pytest-31%20passed-brightgreen.svg)](tests/)
[![Network](https://img.shields.io/badge/Polygon-Amoy%20(80002)-8247E5.svg)](https://amoy.polygonscan.com/)
[![Storage](https://img.shields.io/badge/IPFS-Pinata-E11D48.svg)](https://pinata.cloud/)
[![Model](https://img.shields.io/badge/InsightFace-buffalo__l%20ArcFace%20512--d-blue.svg)](https://github.com/deepinsight/insightface)

---

## 1. Overview & Problem Statement

In an era of digital media proliferation, establishing verifiable cryptographic provenance for discovered public web and social content is critical. When searching for public visual matches of a face image, conventional search engines cannot be treated as trusted arbiters of identity, and centralized web hosts can modify or delete evidence without an immutable record.

**TrustLens implements a decentralized, tamper-evident verification pipeline:**
1. **Explainable Face Analysis**: Detects faces, evaluates multi-factor image quality (sharpness, exposure, resolution, frontality, confidence), supports multi-face selection, and extracts complete normalized 512-dimensional ArcFace embeddings with deterministic SHA-256 fingerprints.
2. **Multi-Source Visual Discovery**: Discovers matching public web and social media candidates via Google Lens (SerpApi).
3. **Independent Multi-Face Candidate Verification**: Independently downloads and executes local ArcFace cosine similarity comparisons on all candidate faces in group photos without assuming the largest face is the target subject.
4. **Candidate Ranking & Separation Margin**: Ranks candidates deterministically and calculates the separation margin gap between the best verified match and the strongest rejected candidate.
5. **Multi-Tier Decision Engine**: Categorizes matches using configurable empirical thresholds (`VERIFIED`, `REVIEW`, `REJECTED`) with explicit decision reason codes.
6. **Canonical Evidence Manifest (RFC-8785)**: Generates a deterministic JSON evidence manifest capturing complete provenance, quality scores, and separation metrics.
7. **Decentralized Storage & On-Chain Anchoring**: Pins the manifest to IPFS via Pinata and anchors the cryptographic SHA-256 hash to **Polygon Amoy Testnet** (`Chain ID: 80002`) via `ProofRegistry.sol`.
8. **Independent Re-Verification & Tamper Detection**: Provides standalone public verification (`verify.py`) and multi-scenario cryptographic tamper detection (`tamper_demo.py`).

---

## 2. System Architecture & Pipeline

```
[ INPUT IMAGE ]
       │
       ▼
[1] Face Detection, Multi-Factor Quality Gate & Face Selection (SCRFD / RetinaFace)
       │  (Quality: Sharpness, Exposure, Resolution, Frontality, Confidence)
       ▼
[2] Full 512-d ArcFace Embedding & Normalized SHA-256 Hashing
       │
       ▼
[3] Query Image IPFS Pinning (Pinata Gateway for Visual Search)
       │
       ▼
[4] Multi-Source & Multi-Platform Web Discovery (Google Lens via SerpApi)
       │  (Instagram, LinkedIn, Facebook, X, Reddit, YouTube, TikTok, Pinterest, News/Media, General Web)
       ▼
[5] Independent Multi-Face Candidate Verification (InsightFace ArcFace on CPU)
       │  (Compares query against ALL faces detected in candidate group images)
       ▼
[6] Candidate Ranking & Separation Margin Analysis
       │  (Calculates best verified vs best rejected delta & qualitative interpretation)
       ▼
[7] Canonical Evidence Manifest Generation (RFC-8785 Deterministic JSON)
       │
       ▼
[8] SHA-256 Cryptographic Fingerprint & IPFS Storage
       │
       ▼
[9] Polygon Amoy Smart Contract Proof Anchoring (`ProofRegistry.sol`)
       │
       ▼
[ INDEPENDENT RE-VERIFICATION & MULTI-SCENARIO TAMPER DETECTION ]
```

---

## 3. Technology Stack

- **Face Recognition Engine**: [InsightFace](https://github.com/deepinsight/insightface) `buffalo_l` (SCRFD detector + ArcFace 512-dimensional embedding extractor) running locally on CPU with ONNX Runtime.
- **Reverse Visual Discovery**: SerpApi Google Lens engine (discovers indexed public web pages, blogs, and social platforms).
- **Decentralized Storage**: IPFS via [Pinata](https://pinata.cloud/) with multi-gateway fallback resolution (Pinata, Cloudflare, IPFS.io, dweb.link).
- **Blockchain**: Polygon Amoy Testnet (Chain ID: `80002`, Sepolia-anchored EVM testnet).
- **Smart Contract & Tooling**: Solidity `0.8.20`, Hardhat, and `web3.py` (with dynamic EIP-1559 gas fee estimation).
- **Testing**: `pytest` (31 unit tests covering vector math, deterministic hashing, manifest schema, separation margins, quality gates, and contract logic).

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
AMOY_RPC_URL=https://polygon-amoy.drpc.org
PRIVATE_KEY=your_testnet_wallet_private_key
CONTRACT_ADDRESS=0x6D03eE0515FeeA663D6fe79F8e12dDA4C24B3c5F

# Configurable Decision Thresholds
VERIFIED_THRESHOLD=0.40
REVIEW_THRESHOLD=0.30
MIN_QUALITY_THRESHOLD=0.20
```

### 3. Deploy Smart Contract to Polygon Amoy *(Optional / Pre-deployed)*
The contract is already deployed on Polygon Amoy at `0x6D03eE0515FeeA663D6fe79F8e12dDA4C24B3c5F`. To deploy your own instance:
```bash
npm run compile
npm run deploy:amoy
```

---

## 5. Usage & CLI Commands

### A. Run Full TrustLens Pipeline
To run the complete 9-stage pipeline on a face image:
```bash
python -m pipeline.main demo/public_face_demo.jpg
```

#### Optional CLI Arguments:
- `--face-index <int>`: Index of face to select if input contains multiple detected subjects (default: `0`).
- `--verified-threshold <float>`: Custom similarity cutoff for `VERIFIED` status (default: `0.40`).
- `--review-threshold <float>`: Custom similarity cutoff for `REVIEW` status (default: `0.30`).
- `--min-quality <float>`: Minimum acceptable face quality score (default: `0.20`).
- `--skip-blockchain`: Run visual search, face verification, and IPFS pinning without submitting an on-chain transaction (dry run / offline mode).

### B. Independent On-Chain Re-Verification
Anyone can re-verify the authenticity and integrity of a registered proof directly from Polygon Amoy and IPFS:
```bash
python verify.py <evidence_sha256_hash>
```

**Live Verified Example:**
```bash
python verify.py 378b160a46709a6174b39d5dfaf9cd2760a6f88816054318b2910b816fb8798e
```

**Output:**
```text
======================================================================
  INDEPENDENT PROOF RE-VERIFICATION (POLYGON AMOY + IPFS)
======================================================================
  BLOCKCHAIN HASH : 378b160a46709a6174b39d5dfaf9cd2760a6f88816054318b2910b816fb8798e
  LOCAL HASH      : 378b160a46709a6174b39d5dfaf9cd2760a6f88816054318b2910b816fb8798e
  IPFS CID        : QmT6LL9h8RXk41Lnta49RY2oLib99krTfNfJyY3RRtae2b
  NETWORK         : Polygon Amoy (Chain ID 80002)
  SUBMITTER       : 0xDDc96a25A85849a3803d8cd690c0B8D8971aacbf
  RESULT          : VALID (Bit-for-bit cryptographic match)
======================================================================
```

### C. Multi-Scenario Tamper-Evidence Demonstration
Demonstrates that altering any metadata (URL, similarity score, platform) breaks cryptographic validation against the blockchain anchor:
```bash
python tamper_demo.py 378b160a46709a6174b39d5dfaf9cd2760a6f88816054318b2910b816fb8798e
```

### D. Run Automated Test Suite
Run all unit tests offline without requiring external API keys:
```bash
pytest tests/ -v
```

---

## 6. Decision Engine & Separation Metrics

### Multi-Tier Classification:
| Status | Condition | Meaning |
| :--- | :--- | :--- |
| **`VERIFIED`** | `similarity >= VERIFIED_THRESHOLD` (0.40) & quality pass | High-confidence biometric similarity confirming candidate match. |
| **`REVIEW`** | `0.30 <= similarity < 0.40` or degraded quality | Moderate similarity; flagged for manual inspection. |
| **`REJECTED`** | `similarity < REVIEW_THRESHOLD` (0.30) | Insufficient facial similarity; rejected as non-matching. |

### Separation Margin Analysis:
$$\text{Separation Margin} = \text{Similarity}_{\text{Best Verified}} - \text{Similarity}_{\text{Best Rejected}}$$
- A wide separation margin (e.g. $> 0.30$) demonstrates that the selected subject clearly stands apart from non-matching candidates.
- A narrow margin indicates that the decision boundary warrants manual review.

*Note: Thresholds are configurable application heuristics and do not constitute absolute biometric identity guarantees.*

---

## 7. Deterministic Evidence Manifest Schema (RFC-8785)

The canonical evidence manifest captures complete discovery and verification provenance:
```json
{
  "schema_version": "1.1.0",
  "record_timestamp": 1788542113,
  "face": {
    "algorithm": "InsightFace buffalo_l",
    "model": "ArcFace",
    "dimension": 512,
    "dtype": "float32",
    "normalized": true,
    "embedding_hash": "892007cb66327f837bafe0bca6fa9dfa3dcde86b3bfd1849b8506b1cbabfcf15",
    "selected_face_index": 0,
    "query_image_quality": 0.8812
  },
  "search": {
    "provider": "SerpApi",
    "engine": "google_lens",
    "query_image_cid": "QmW8je2RUShfuHnEeqy5Rdgadg73vC1WS2uGUfPmdb3aSt",
    "discovery_timestamp": 1788542113,
    "total_candidates_discovered": 59,
    "usable_images_evaluated": 59
  },
  "candidate": {
    "source_url": "https://www.pinterest.com/savedbyhimalway/tom-hanks/",
    "source_title": "Tom Hanks Pins - Public Social Content",
    "domain": "pinterest.com",
    "platform": "Pinterest",
    "search_rank": 22,
    "thumbnail_url": "https://...",
    "thumbnail_sha256": "64_hex_thumbnail_hash"
  },
  "verification": {
    "face_similarity_score": 0.9471,
    "verified_threshold": 0.40,
    "review_threshold": 0.30,
    "decision": "VERIFIED",
    "decision_reason": "Face similarity (0.9471 >= 0.40) exceeds verified threshold and candidate passed quality checks",
    "face_detection_confidence": 0.9124,
    "candidate_face_count": 1,
    "candidate_image_quality": 0.8361,
    "separation_margin": 0.7144,
    "margin_interpretation": "Clear separation: Strong differentiation between matching subject and non-matching candidates"
  },
  "integrity": {
    "canonicalization_method": "RFC-8785 canonical JSON (sorted keys, compact separators, UTF-8)",
    "sha256_hash": "378b160a46709a6174b39d5dfaf9cd2760a6f88816054318b2910b816fb8798e"
  }
}
```

---

## 8. Smart Contract Design (`ProofRegistry.sol`)

The `ProofRegistry` contract maintains an immutable registry of cryptographic evidence on Polygon Amoy:
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
- **Minimal Footprint**: Only cryptographic hashes and IPFS CIDs are stored on-chain to minimize gas costs and prevent placing personal data on a public ledger.

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
│   ├── face_id.py              # InsightFace ArcFace 512-d embeddings, quality scoring & multi-face
│   ├── search.py               # SerpApi Google Lens discovery & platform classifier
│   ├── fingerprint.py          # RFC-8785 canonical evidence manifest, separation margins & SHA-256
│   ├── ipfs_store.py           # Pinata IPFS integration with multi-gateway fallback
│   ├── chain.py                # Web3.py Polygon Amoy interaction & EIP-1559 gas
│   └── main.py                 # 9-stage CLI orchestrator with multi-face & quality gates
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py        # 31-test automated pytest suite
│   ├── preflight_check.py      # Live preflight credential & connectivity validation
│   └── validate_stages.py      # Stage-by-stage pipeline integration validation
├── demo/
│   ├── README.md
│   ├── public_face_demo.jpg    # Consenting test image for live verification
│   └── sample_face.jpg         # Sample test image for offline unit tests
├── verify.py                   # Standalone independent proof re-verification
├── tamper_demo.py              # Cryptographic tamper detection demonstration (multi-scenario)
├── hardhat.config.js           # Hardhat network configuration (Polygon Amoy)
├── package.json                # Node dependencies & compile/deploy scripts
├── requirements.txt            # Python dependencies
└── README.md                   # System documentation
```
