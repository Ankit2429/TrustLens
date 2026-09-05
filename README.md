# TrustLens — Face Identification & Blockchain Verification

[![Tests](https://img.shields.io/badge/pytest-37%20passed-brightgreen.svg)](tests/)
[![Blockchain](https://img.shields.io/badge/Blockchain-Anvil%20%7C%20Polygon%20Amoy-blueviolet.svg)](https://book.getfoundry.sh/anvil/)
[![Local EVM](https://img.shields.io/badge/Local%20Node-Anvil%20(31337)-3B82F6.svg)](http://127.0.0.1:8545)
[![Public Testnet](https://img.shields.io/badge/Polygon-Amoy%20(80002)-8247E5.svg)](https://amoy.polygonscan.com/)
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
7. **Decentralized Storage & On-Chain Anchoring**: Pins the manifest to IPFS via Pinata and anchors the cryptographic SHA-256 hash to a **local Anvil blockchain** (`Chain ID: 31337`) or **Polygon Amoy Testnet** (`Chain ID: 80002`) via `ProofRegistry.sol`.
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
[9] Blockchain Smart Contract Proof Anchoring (`ProofRegistry.sol` on Anvil or Polygon Amoy)
       │
       ▼
[ INDEPENDENT RE-VERIFICATION & MULTI-SCENARIO TAMPER DETECTION ]
```

---

## 3. Technology Stack

- **Face Recognition Engine**: [InsightFace](https://github.com/deepinsight/insightface) `buffalo_l` (SCRFD detector + ArcFace 512-dimensional embedding extractor) running locally on CPU with ONNX Runtime.
- **Reverse Visual Discovery**: SerpApi Google Lens engine (discovers indexed public web pages, blogs, and social platforms).
- **Decentralized Storage**: IPFS via [Pinata](https://pinata.cloud/) with multi-gateway fallback resolution (Pinata, Cloudflare, IPFS.io, dweb.link).
- **Blockchain Adapter Layer**:
  - **Local Development Node**: [Foundry Anvil](https://book.getfoundry.sh/anvil/) (Chain ID `31337`, fast instant local blocks, pre-funded development accounts).
  - **Public Testnet**: Polygon Amoy Testnet (Chain ID `80002`, Sepolia-anchored EVM testnet with dynamic EIP-1559 tip calculation).
- **Smart Contract & Tooling**: Solidity `0.8.20`, Hardhat, and `web3.py`.
- **Testing**: `pytest` (37 automated unit and integration tests covering vector math, deterministic hashing, manifest schema, separation margins, quality gates, Anvil blockchain adapter, and contract logic).

---

## 4. Anvil Local Blockchain Workflow

TrustLens uses **Foundry Anvil** as its primary local EVM development blockchain. Anvil provides:
- Instant local transaction confirmations (< 0.2s)
- 10 pre-funded test accounts with 10,000 ETH each
- Zero faucet wait times and zero testnet rate limits
- Full standard EVM compatibility with `ProofRegistry.sol`

> [!NOTE]
> Polygon Amoy remains fully supported and configurable. You can switch between Anvil and Polygon Amoy at any time by toggling `BLOCKCHAIN_NETWORK` in `.env`.

### Step 1: Start Anvil Local Node
In a separate terminal window, start Anvil:
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

### Step 3: Run the TrustLens Pipeline
```bash
python -m pipeline.main demo/sample_face.jpg
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
VERIFIED_THRESHOLD=0.45
REVIEW_THRESHOLD=0.38
MIN_QUALITY_THRESHOLD=0.20
```

#### Option B: Polygon Amoy Public Testnet
To switch to Polygon Amoy, deploy the contract with `npm run deploy:amoy` or update your `.env`:
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
- Select target subjects in multi-face photos.
- Run live SerpApi visual discovery with multi-source candidate filtering and separation margin analysis.
- Inspect canonical RFC-8785 JSON manifests and on-chain proof confirmations.
- Run independent blockchain re-verification and interactive cryptographic tamper simulations.

### B. Run Full CLI Pipeline
```bash
python -m pipeline.main demo/sample_face.jpg
```

#### Optional CLI Arguments:
- `--face-index <int>`: Index of face to select if input contains multiple detected subjects (default: `0`).
- `--verified-threshold <float>`: Custom similarity cutoff for `VERIFIED` status (default: `0.45`).
- `--review-threshold <float>`: Custom similarity cutoff for `REVIEW` status (default: `0.38`).
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
| Status | Condition | Meaning |
| :--- | :--- | :--- |
| **`VERIFIED`** | `similarity >= VERIFIED_THRESHOLD` (0.45) & quality pass | High-confidence biometric similarity confirming candidate match. |
| **`REVIEW`** | `0.38 <= similarity < 0.45` or degraded quality | Moderate similarity; flagged for manual inspection. |
| **`REJECTED`** | `similarity < REVIEW_THRESHOLD` (0.38) | Insufficient facial similarity; rejected as non-matching. |

### Separation Margin Analysis:
$$\text{Separation Margin} = \text{Similarity}_{\text{Best Verified}} - \text{Similarity}_{\text{Best Rejected}}$$
- A wide separation margin demonstrates that the selected subject clearly stands apart from non-matching candidates.
- A narrow margin indicates that the decision boundary warrants manual review.

---

## 8. Deterministic Evidence Manifest Schema (RFC-8785)

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
    "source_url": "https://images.dawn.com/news/1174640/save-the-date-friends-just-might-have-a-reunion-of-sorts-on-february-21",
    "source_title": "Save the date: Friends just might have a reunion of sorts on February 21",
    "domain": "images.dawn.com",
    "platform": "General Web",
    "search_rank": 1,
    "thumbnail_url": "https://...",
    "thumbnail_sha256": "64_hex_thumbnail_hash"
  },
  "verification": {
    "face_similarity_score": 0.7435,
    "verified_threshold": 0.40,
    "review_threshold": 0.30,
    "decision": "VERIFIED",
    "decision_reason": "Face similarity (0.7435 >= 0.40) exceeds verified threshold and candidate passed quality checks",
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

## 9. Smart Contract Design (`ProofRegistry.sol`)

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

## 10. Privacy, Ethical Scope & Known Limitations

- **Public / Indexed Content Only**: The pipeline exclusively evaluates publicly discoverable web results indexed by visual search engines.
- **No Access Control Bypasses**: Does not attempt to bypass authentication, paywalls, CAPTCHAs, or private social profiles.
- **Demonstration Scope**: Tested strictly against consenting subjects (team members' public photos).
- **Proof-of-Existence / Integrity**: Demonstrates cryptographic integrity and proof-of-existence of public evidence, **not legal identity certification**.
- **Rate Caps**: Subject to standard public rate limits for SerpApi and Pinata tiers.

---

## 11. Repository Structure

```
├── contracts/
│   └── ProofRegistry.sol       # Solidity smart contract for proof registry
├── scripts/
│   └── deploy.js               # Multi-network deployment script (Anvil & Polygon Amoy)
├── pipeline/
│   ├── __init__.py
│   ├── face_id.py              # InsightFace ArcFace 512-d embeddings, quality scoring & multi-face
│   ├── search.py               # SerpApi Google Lens discovery & platform classifier
│   ├── fingerprint.py          # RFC-8785 canonical evidence manifest, separation margins & SHA-256
│   ├── ipfs_store.py           # Pinata IPFS integration with multi-gateway fallback
│   ├── chain.py                # Network-agnostic blockchain adapter (Anvil + Polygon Amoy)
│   └── main.py                 # 9-stage CLI orchestrator with multi-face & quality gates
├── tests/
│   ├── __init__.py
│   ├── test_blockchain_anvil.py# Anvil blockchain adapter unit & integration tests
│   ├── test_pipeline.py        # Comprehensive 31-test pipeline unit test suite
│   ├── test_webapp_api.py      # End-to-end web server & API integration tests
│   ├── preflight_check.py      # Live preflight credential & connectivity validation
│   └── validate_stages.py      # Stage-by-stage pipeline integration validation
├── webapp/                     # Interactive Web Application Dashboard
├── demo/                       # Consenting demo and sample test images
├── verify.py                   # Standalone independent proof re-verification
├── tamper_demo.py              # Cryptographic tamper detection demonstration (multi-scenario)
├── hardhat.config.js           # Hardhat network configuration (Anvil localhost + Polygon Amoy)
├── package.json                # Node dependencies & compile/deploy scripts
├── requirements.txt            # Python dependencies
└── README.md                   # System documentation
```
