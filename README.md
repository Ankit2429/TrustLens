# IDENTITY // VERIFY — Face Identification & Blockchain Verification

[![Tests](https://img.shields.io/badge/pytest-49%20passed-brightgreen.svg)](tests/)
[![Blockchain](https://img.shields.io/badge/Blockchain-Anvil%20%7C%20Polygon%20Amoy-blueviolet.svg)](https://book.getfoundry.sh/anvil/)
[![Local EVM](https://img.shields.io/badge/Local%20Node-Anvil%20(31337)-3B82F6.svg)](http://127.0.0.1:8545)
[![Public Testnet](https://img.shields.io/badge/Polygon-Amoy%20(80002)-8247E5.svg)](https://amoy.polygonscan.com/)
[![Storage](https://img.shields.io/badge/IPFS-Pinata-E11D48.svg)](https://pinata.cloud/)
[![Model](https://img.shields.io/badge/InsightFace-buffalo__l%20ArcFace%20512--d-blue.svg)](https://github.com/deepinsight/insightface)

---

## 1. Overview & Problem Statement

In an era of digital media proliferation, establishing verifiable cryptographic provenance for discovered public web and social content is critical. When searching for public visual matches of a face image, conventional search engines cannot be treated as trusted arbiters of identity, and centralized web hosts can modify or delete evidence without an immutable record.

**This system implements a decentralized, tamper-evident verification pipeline:**
1. **Explainable Face Analysis**: Detects faces, evaluates multi-factor image quality (sharpness, exposure, resolution, frontality, confidence), supports multi-face selection, landmark-aligned ArcFace embedding with 5-point geometric normalization, and extracts 512-dimensional embeddings with deterministic SHA-256 fingerprints.
2. **Multi-Source Visual Discovery**: Discovers matching public web and social media candidates via Google Lens (SerpApi) with honest pagination — follows `serpapi_pagination.next` continuation tokens until exhausted or a configurable maximum is reached. Reports exact pages scanned, raw results, unique candidates, and faces evaluated.
3. **Independent Multi-Face Candidate Verification**: Independently downloads and executes local ArcFace cosine similarity comparisons against **all faces** detected in candidate images (including group photos), without assuming the largest or most prominent face is the target.
4. **Candidate Ranking & Separation Margin**: Ranks candidates deterministically by ArcFace similarity and calculates the separation margin gap between the best verified match and the strongest rejected candidate.
5. **Multi-Tier Decision Engine**: Categorizes matches using configurable empirical thresholds (`VERIFIED`, `REVIEW`, `REJECTED`) with explicit decision reason codes.
6. **Canonical Evidence Manifest (RFC-8785)**: Generates a deterministic JSON evidence manifest capturing complete provenance, quality scores, and separation metrics.
7. **Strictly-Gated Blockchain Proof Anchoring**: Pins the manifest to IPFS via Pinata and anchors the cryptographic SHA-256 hash to a **local Anvil blockchain** (`Chain ID: 31337`) or **Polygon Amoy Testnet** (`Chain ID: 80002`) via `ProofRegistry.sol` **strictly when evidence reaches the high-confidence `VERIFIED` state**. `REVIEW`, `REJECTED`, and `NO RELIABLE MATCH` outcomes do **not** anchor to the ledger (PROOF NOT ANCHORED).
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
[2] 5-Point Landmark Alignment & Full 512-d ArcFace Embedding & SHA-256 Hashing
       │
       ▼
[3] Query Image IPFS Pinning (Pinata Gateway for Visual Search)
       │
       ▼
[4] Multi-Source & Multi-Platform Web Discovery (Google Lens via SerpApi)
       │  (type=visual_matches · Paginates via serpapi_pagination.next · Honest telemetry)
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
[9] Blockchain Smart Contract Proof Anchoring (VERIFIED only — ProofRegistry.sol)
       │  (REVIEW / REJECTED / NO MATCH → PROOF NOT ANCHORED)
       ▼
[ INDEPENDENT RE-VERIFICATION & MULTI-SCENARIO TAMPER DETECTION ]
```

---

## 3. Technology Stack

- **Face Recognition Engine**: [InsightFace](https://github.com/deepinsight/insightface) `buffalo_l` (SCRFD detector + ArcFace 512-dimensional embedding extractor) running locally on CPU with ONNX Runtime. 5-point geometric landmark alignment applied before each embedding.
- **Reverse Visual Discovery**: SerpApi Google Lens engine (`type=visual_matches`) with honest `serpapi_pagination.next` continuation. Discovers indexed public web pages, blogs, and social platforms.
- **Decentralized Storage**: IPFS via [Pinata](https://pinata.cloud/) with multi-gateway fallback resolution (Pinata, Cloudflare, IPFS.io, dweb.link).
- **Blockchain Adapter Layer**:
  - **Local Development Node**: [Foundry Anvil](https://book.getfoundry.sh/anvil/) (Chain ID `31337`, fast instant local blocks, pre-funded development accounts).
  - **Public Testnet**: Polygon Amoy Testnet (Chain ID `80002`, Sepolia-anchored EVM testnet with dynamic EIP-1559 tip calculation).
- **Smart Contract & Tooling**: Solidity `0.8.20`, Hardhat, and `web3.py`.
- **Web Interface**: `IDENTITY // VERIFY` — near-black photo-centric UI with 3-column layout (face analysis, photo workspace, discovery panel), 6-stage pipeline stepper, and interactive tamper demo.
- **Testing**: `pytest` (49 automated unit and integration tests covering vector math, deterministic hashing, manifest schema, separation margins, quality gates, Anvil blockchain adapter, multi-face group analysis, and contract logic).

---

## 4. Anvil Local Blockchain Workflow

The system uses **Foundry Anvil** as its primary local EVM development blockchain. Anvil provides:
- Instant local transaction confirmations (< 0.2s)
- 10 pre-funded test accounts with 10,000 ETH each
- Zero faucet wait times and zero testnet rate limits
- Full standard EVM compatibility with `ProofRegistry.sol`

> [!NOTE]
> Polygon Amoy remains fully supported and configurable. Switch between Anvil and Polygon Amoy at any time by toggling `BLOCKCHAIN_NETWORK` in `.env`.

### Step 1: Start Anvil Local Node
```bash
anvil
# Or with explicit port:
anvil --port 8545
```
Anvil will start listening on `http://127.0.0.1:8545` with Chain ID `31337`.

### Step 2: Compile and Deploy `ProofRegistry.sol` to Anvil
```bash
npm install
npm run compile
npm run deploy:anvil
```
This deploys the `ProofRegistry` contract to your local Anvil node and automatically updates `BLOCKCHAIN_CONTRACT_ADDRESS` and `BLOCKCHAIN_NETWORK=anvil` in your `.env` file.

### Step 3: Run the Pipeline
```bash
python -m pipeline.main demo/public_face_demo.jpg
```

### Step 4: Verify the Proof Against Anvil
```bash
python verify.py <evidence_sha256_hash>
```

### Step 5: Run Tamper Demonstration Against Anvil
```bash
python tamper_demo.py <evidence_sha256_hash>
```

---

## 5. Environment Configuration

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

### Active Network Configuration Examples

#### Option A: Foundry Anvil (Default Local Development)
```env
# Search & IPFS
SERPAPI_KEY=your_serpapi_key
PINATA_JWT=your_pinata_jwt

# Blockchain Configuration (Anvil)
BLOCKCHAIN_NETWORK=anvil
BLOCKCHAIN_RPC_URL=http://127.0.0.1:8545
BLOCKCHAIN_CHAIN_ID=31337
BLOCKCHAIN_PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
BLOCKCHAIN_CONTRACT_ADDRESS=0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512

# Decision Thresholds
VERIFIED_THRESHOLD=0.60
REVIEW_THRESHOLD=0.40
MIN_QUALITY_THRESHOLD=0.20

# Search Depth Controls (optional overrides)
SEARCH_MAX_PAGES=3
SEARCH_MAX_RESULTS=60
SEARCH_MAX_CANDIDATE_IMAGES=40
SEARCH_MAX_SOURCE_EXPANSIONS=8
```

#### Option B: Polygon Amoy Public Testnet
```env
BLOCKCHAIN_NETWORK=polygon_amoy
BLOCKCHAIN_RPC_URL=https://polygon-amoy.drpc.org
BLOCKCHAIN_CHAIN_ID=80002
BLOCKCHAIN_PRIVATE_KEY=your_amoy_wallet_private_key
BLOCKCHAIN_CONTRACT_ADDRESS=your_deployed_amoy_contract_address
```

---

## 6. Usage & CLI Commands

### A. Launch Interactive Web Application Dashboard
```bash
python -m webapp
# Or: python web_server.py --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser to:
- Upload face photos and inspect interactive 5-point landmark bounding boxes and multi-factor quality scores.
- Select target subjects in multi-face photos using the face chip selector.
- Run live SerpApi visual discovery with multi-source candidate filtering and separation margin analysis.
- Inspect canonical RFC-8785 JSON manifests and on-chain proof confirmations.
- Run independent blockchain re-verification and interactive cryptographic tamper simulations.

### B. Run Full CLI Pipeline
```bash
python -m pipeline.main demo/public_face_demo.jpg
```

#### Optional CLI Arguments:
- `--face-index <int>`: Index of face to select if input contains multiple detected subjects (default: `0`).
- `--verified-threshold <float>`: Custom similarity cutoff for `VERIFIED` status (default: `0.60`).
- `--review-threshold <float>`: Custom similarity cutoff for `REVIEW` status (default: `0.40`).
- `--min-quality <float>`: Minimum acceptable face quality score (default: `0.20`).
- `--skip-blockchain`: Run visual search, face verification, and IPFS pinning without submitting an on-chain transaction.

### C. Independent On-Chain Re-Verification
Anyone can re-verify the authenticity and integrity of a registered proof directly from the active blockchain and IPFS:
```bash
python verify.py <evidence_sha256_hash>
```

**Example Output:**
```text
======================================================================
  INDEPENDENT PROOF RE-VERIFICATION (BLOCKCHAIN + IPFS)
======================================================================

[1/3] Querying ProofRegistry smart contract on blockchain...
      -> Status     : FOUND ON-CHAIN
      -> Submitter  : 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
      -> IPFS CID   : QmeQmbik7Z2tTPbJmnwwM2ah93PWbJ5hRXmTggXjTpiMrt
      -> Timestamp  : 1788628767 (2026-09-05 17:19:27 UTC)

[2/3] Retrieving canonical evidence manifest from IPFS...
      -> Successfully retrieved record (1741 chars)

[3/3] Canonicalizing manifest and recomputing SHA-256 fingerprint...

======================================================================
  VERIFICATION RESULT
======================================================================
  BLOCKCHAIN HASH : 1af6de16fdfea4447d02924f1e24ef4f0c8e691801ea63f766c57d9de70eedc3
  LOCAL HASH      : 1af6de16fdfea4447d02924f1e24ef4f0c8e691801ea63f766c57d9de70eedc3
  IPFS CID        : QmeQmbik7Z2tTPbJmnwwM2ah93PWbJ5hRXmTggXjTpiMrt
  NETWORK         : Anvil Local Node (Chain ID 31337)
  SUBMITTER       : 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
  TIMESTAMP       : 2026-09-05 17:19:27 UTC
  RESULT          : VALID (Bit-for-bit cryptographic match)
======================================================================
```

### D. Multi-Scenario Tamper-Evidence Demonstration
Demonstrates that altering any metadata (URL, similarity score, platform) breaks cryptographic validation against the blockchain anchor:
```bash
python tamper_demo.py <evidence_sha256_hash>
```

### E. Run Automated Test Suite & Preflight Check
```bash
# Run preflight connectivity and credential check:
python tests/preflight_check.py

# Run comprehensive pytest suite:
pytest -v
```

---

## 7. Decision Engine & Separation Metrics

### Multi-Tier Classification:
| Status | Condition | Blockchain Anchoring | Meaning |
| :--- | :--- | :--- | :--- |
| **`VERIFIED`** | `similarity >= 0.60` & quality pass | **ANCHORED to ledger** | High-confidence biometric similarity confirming candidate match. |
| **`REVIEW`** | `0.40 <= similarity < 0.60` or degraded quality | **PROOF NOT ANCHORED** | Moderate similarity; flagged for manual inspection. |
| **`REJECTED`** | `similarity < 0.40` | **PROOF NOT ANCHORED** | Insufficient facial similarity; rejected as non-matching. |
| **`NO RELIABLE MATCH`** | No candidates exceed review threshold | **PROOF NOT ANCHORED** | No credible match found in indexed public content. |

> [!IMPORTANT]
> Thresholds are configured via environment variables (`VERIFIED_THRESHOLD`, `REVIEW_THRESHOLD`) in `.env`. Defaults are `0.60` and `0.40` respectively. Do **not** lower these thresholds to produce more matches — FALSE POSITIVE > FALSE NEGATIVE.

### Strict VERIFIED-Only Anchoring:
Only results where `decision == 'VERIFIED'` **and** a blockchain receipt is returned will anchor a proof to the ledger. All other outcomes safely skip on-chain anchoring to prevent immutable ledger pollution with unverified evidence.

### Separation Margin Analysis:
$$\text{Separation Margin} = \text{Similarity}_{\text{Best Verified}} - \text{Similarity}_{\text{Best Rejected}}$$
- A wide separation margin demonstrates that the selected subject clearly stands apart from non-matching candidates.
- A narrow margin indicates that the decision boundary warrants manual review.

---

## 8. Search Depth & Honest Telemetry

The pipeline uses SerpApi's Google Lens engine with `type=visual_matches` for the richest result set. Pagination is driven exclusively by the API's own `serpapi_pagination.next` continuation token:

- **Page 1**: Initial request returns up to ~10–20 visual matches.
- **Page 2+**: If `serpapi_pagination.next` is present, follows the pre-constructed continuation URL.
- **Stops when**: No continuation token is returned (API has no further results), the configurable `SEARCH_MAX_PAGES` limit is reached, or `SEARCH_MAX_RESULTS` unique candidates are collected.
- **Does NOT fabricate depth**: If the API returns only 1 page of results, telemetry reports `pages_scanned: 1` — never inflated.

### Telemetry Fields (returned with every search):
```
pages_scanned       : Actual API pages retrieved (honest count)
total_discovered    : Raw result entries processed (with duplicates)
unique_candidates   : Deduplicated candidates retained
platforms_discovered: Platform breakdown (Instagram, LinkedIn, etc.)
source_expansions   : Source pages inspected for high-res og:image
categories_scanned  : Result categories returned (visual_matches, exact_matches, etc.)
```

### Candidate Image Evaluation:
Every candidate thumbnail is independently fetched and all detected faces compared against the query embedding. In group photos, **every face** is evaluated — the system does not assume position or size. Results include `face_count`, `matched_face_id`, and per-face similarity scores.

---

## 9. Deterministic Evidence Manifest Schema (RFC-8785)

```json
{
  "schema_version": "1.1.0",
  "record_timestamp": 1788628767,
  "face": {
    "algorithm": "InsightFace buffalo_l",
    "model": "ArcFace",
    "dimension": 512,
    "dtype": "float32",
    "normalized": true,
    "embedding_hash": "37cfdea14167c19a10fb62df02b426f2b07c512fb23f43f349ea95186863e0bd",
    "selected_face_index": 0,
    "query_image_quality": 0.9146
  },
  "search": {
    "provider": "SerpApi",
    "engine": "google_lens",
    "query_image_cid": "QmfFbgmchw2vi9BawZqhM5bxW5duFu8rmKwcZcwozXbsNR",
    "discovery_timestamp": 1788628767,
    "total_candidates_discovered": 59,
    "usable_images_evaluated": 40
  },
  "candidate": {
    "source_url": "https://example.com/article",
    "source_title": "Discovered page title",
    "domain": "example.com",
    "platform": "General Web",
    "search_rank": 1,
    "thumbnail_url": "https://...",
    "thumbnail_sha256": "64_hex_thumbnail_hash"
  },
  "verification": {
    "face_similarity_score": 0.7435,
    "verified_threshold": 0.60,
    "review_threshold": 0.40,
    "decision": "VERIFIED",
    "decision_reason": "Face similarity (0.7435 >= 0.60) exceeds verified threshold and candidate passed quality checks",
    "face_detection_confidence": 0.9195,
    "candidate_face_count": 6,
    "candidate_image_quality": 0.7934,
    "separation_margin": 0.0193,
    "margin_interpretation": "Narrow margin: Evaluation warrants closer inspection of boundary candidates"
  },
  "integrity": {
    "canonicalization_method": "RFC-8785 canonical JSON (sorted keys, compact separators, UTF-8)",
    "sha256_hash": "1af6de16fdfea4447d02924f1e24ef4f0c8e691801ea63f766c57d9de70eedc3"
  }
}
```

---

## 10. Smart Contract Design (`ProofRegistry.sol`)

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract ProofRegistry {
    struct Proof {
        address submitter;
        string ipfsCID;
        uint256 timestamp;
    }

    mapping(bytes32 => Proof) public proofs;

    event ProofRegistered(
        bytes32 indexed dataHash,
        string ipfsCID,
        address indexed submitter,
        uint256 timestamp
    );

    function registerProof(bytes32 dataHash, string calldata ipfsCID) external {
        require(proofs[dataHash].timestamp == 0, "Already registered");
        proofs[dataHash] = Proof({
            submitter: msg.sender,
            ipfsCID: ipfsCID,
            timestamp: block.timestamp
        });
        emit ProofRegistered(dataHash, ipfsCID, msg.sender, block.timestamp);
    }

    function getProof(bytes32 dataHash)
        external
        view
        returns (address submitter, string memory ipfsCID, uint256 timestamp)
    {
        Proof memory p = proofs[dataHash];
        require(p.timestamp != 0, "Not found");
        return (p.submitter, p.ipfsCID, p.timestamp);
    }
}
```
- **Anti-Overwrite Protection**: Prevents overwriting previously anchored evidence proofs (`require(proofs[dataHash].timestamp == 0, "Already registered")`).
- **Audit Event**: Emits `event ProofRegistered(bytes32 indexed dataHash, string ipfsCID, address indexed submitter, uint256 timestamp)`.
- **Minimal Footprint**: Only cryptographic hashes and IPFS CIDs are stored on-chain.

---

## 11. Privacy, Ethical Scope & Known Limitations

- **Public / Indexed Content Only**: The pipeline exclusively evaluates publicly discoverable web results indexed by visual search engines. No private data or authenticated content is accessed.
- **No Access Control Bypasses**: Does not attempt to bypass authentication, paywalls, CAPTCHAs, or private social profiles.
- **Search Depth is API-Bound**: The pipeline retrieves as many pages as SerpApi returns via `serpapi_pagination.next`. Google Lens typically returns 1–3 pages (10–30 unique visual matches) per query; depth beyond that is not supported by the API.
- **Not Legal Identity Certification**: Demonstrates cryptographic integrity and proof-of-existence of public evidence. This is a tool for discovering indexed public content — not a legal identity verification system.
- **ArcFace Similarity is Not Infallible**: Thresholds are empirical heuristics. Very similar-looking individuals or poor-quality images may produce boundary-case results. The conservative `0.60` VERIFIED threshold and strict blockchain gating minimize false positives.
- **Rate Caps**: Subject to standard public rate limits for SerpApi and Pinata tiers.
- **Local Anvil Node**: The deployed Anvil contract address (`0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512`) is pre-funded with 10,000 ETH on the local dev node. Reset Anvil clears the chain — redeploy before re-testing.

---

## 12. Repository Structure

```
├── contracts/
│   └── ProofRegistry.sol       # Solidity smart contract for proof registry
├── scripts/
│   └── deploy.js               # Multi-network deployment script (Anvil & Polygon Amoy)
├── pipeline/
│   ├── __init__.py
│   ├── face_id.py              # InsightFace ArcFace 512-d embeddings, 5-pt alignment, quality scoring, multi-face
│   ├── search.py               # SerpApi Google Lens discovery, honest pagination, platform classifier
│   ├── fingerprint.py          # RFC-8785 canonical evidence manifest, separation margins & SHA-256
│   ├── ipfs_store.py           # Pinata IPFS integration with multi-gateway fallback
│   ├── chain.py                # Network-agnostic blockchain adapter (Anvil + Polygon Amoy)
│   └── main.py                 # 9-stage CLI orchestrator with multi-face & quality gates
├── tests/
│   ├── __init__.py
│   ├── test_blockchain_anvil.py# Anvil blockchain adapter unit & integration tests
│   ├── test_pipeline.py        # Comprehensive pipeline unit test suite
│   ├── test_webapp_api.py      # End-to-end web server & API integration tests
│   ├── test_accuracy_generalization.py  # Face analysis accuracy & edge-case tests
│   ├── test_deep_search_and_accuracy.py # SerpApi pagination, group-photo & manifest tests
│   ├── eval_blackbox.py        # 9-condition black-box photograph accuracy benchmark
│   ├── preflight_check.py      # Live preflight credential & connectivity validation
│   └── validate_stages.py      # Stage-by-stage pipeline integration validation
├── webapp/                     # IDENTITY // VERIFY interactive web dashboard
│   ├── static/
│   │   ├── index.html          # Near-black photo-centric UI with 3-column layout
│   │   ├── app.js              # Frontend pipeline orchestration & result rendering
│   │   └── style.css           # Design system: Space Grotesk + Inter + JetBrains Mono
│   └── server.py               # FastAPI/Flask web server
├── demo/                       # Consenting demo and sample test images
├── verify.py                   # Standalone independent proof re-verification
├── tamper_demo.py              # Cryptographic tamper detection demonstration (multi-scenario)
├── hardhat.config.js           # Hardhat network configuration (Anvil localhost + Polygon Amoy)
├── package.json                # Node dependencies & compile/deploy scripts
├── requirements.txt            # Python dependencies
└── README.md                   # System documentation
```
