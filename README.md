# IDENTITY // VERIFY — Face Identification & Blockchain Verification

[![Tests](https://img.shields.io/badge/pytest-55%20passed-brightgreen.svg)](tests/)
[![Blockchain](https://img.shields.io/badge/Blockchain-Anvil%20%7C%20Polygon%20Amoy-blueviolet.svg)](https://book.getfoundry.sh/anvil/)
[![Local EVM](https://img.shields.io/badge/Local%20Node-Anvil%20(31337)-3B82F6.svg)](http://127.0.0.1:8545)
[![Public Testnet](https://img.shields.io/badge/Polygon-Amoy%20(80002)-8247E5.svg)](https://amoy.polygonscan.com/)
[![Storage](https://img.shields.io/badge/IPFS-Pinata-E11D48.svg)](https://pinata.cloud/)
[![Model](https://img.shields.io/badge/InsightFace-buffalo__l%20ArcFace%20512--d-blue.svg)](https://github.com/deepinsight/insightface)

`IDENTITY // VERIFY` is an automated, tamper-evident biometric intelligence system that combines local deep face recognition (InsightFace ArcFace 512-d) with reverse visual discovery (Google Lens via SerpApi) and decentralized cryptographic provenance (IPFS and EVM smart contracts). Discovered public web candidates are independently retrieved, evaluated across all detected faces, and structured into a canonical RFC-8785 evidence manifest. High-confidence genuine matches are anchored on-chain with SHA-256 fingerprints, enabling public, independent verification and instant cryptographic tamper detection.

---

## Overview

Public image search engines can discover where photos appear online, but centralized web results cannot serve as trusted arbiters of identity. Web content is mutable: pages can be modified, images deleted, and query results altered without an immutable audit trail. Furthermore, standard reverse search tools frequently conflate visually similar context (clothing, background, poses) with actual biometric identity.

**`IDENTITY // VERIFY` addresses this problem by separating contextual visual discovery from local biometric verification, and anchoring high-confidence evidence to an immutable cryptographic ledger:**

- **Biometric Disambiguation**: Discovers candidates via reverse visual indexing, but performs local, independent ArcFace comparisons across *every detected face* in candidate images to confirm biometric identity.
- **Cryptographic Provenance**: Serializes discovery metadata, similarity scores, quality metrics, and relationship graphs into a deterministic RFC-8785 JSON manifest.
- **Decentralized Anchoring**: Hashes the manifest with SHA-256, stores the payload on IPFS, and anchors the proof onto an EVM smart contract (`ProofRegistry.sol`) on a local Anvil node (`Chain ID: 31337`) or Polygon Amoy testnet (`Chain ID: 80002`).
- **Strict Verification Gate**: Only evidence meeting high-confidence biometric thresholds (`VERIFIED ≥ 0.60`) is anchored on-chain. Borderline or non-matching candidates display `PROOF NOT ANCHORED`.

---

## What It Does

The system executes a structured 17-step end-to-end intelligence and verification workflow:

1. **Input Image Ingestion**: Ingests user-supplied face photos or multi-person group images via web UI or CLI.
2. **Face Detection**: Detects all faces in the query image using SCRFD / RetinaFace (`buffalo_l` bundle).
3. **Face Quality Analysis**: Computes composite quality metrics including sharpness (Laplacian variance), exposure balance, contrast, resolution, frontality, and detection confidence.
4. **Canonical 5-Point Alignment**: Extracts 5 facial keypoints (eyes, nose, mouth corners) and applies a similarity transformation to obtain a canonical aligned crop.
5. **ArcFace 512-D Embedding**: Generates a 512-dimensional feature embedding via ArcFace with float32 L2 normalization, and computes its deterministic SHA-256 hash.
6. **Public Visual Discovery**: Queries Google Lens via SerpApi (`type=visual_matches`) using the public gateway URL of the pinned query image.
7. **Candidate Retrieval & Deduplication**: Extracts candidate matches from `reverse_image_search` (and inline image results), normalizes page URLs, and deduplicates identical image URLs.
8. **Independent Candidate Verification**: Concurrently downloads candidate image thumbnails into memory and runs local ArcFace feature extraction on CPU.
9. **Face-by-Face Group Photo Analysis**: Evaluates *every detectable face* within candidate images (`FACE 01`, `FACE 02`, etc.) rather than assuming the most prominent face is the target.
10. **Candidate Ranking & Similarity Scoring**: Computes cosine similarity against the query embedding and ranks all candidates deterministically.
11. **Cross-Source Evidence & Consensus Analysis**: Categorizes candidate relationships (`SAME_IMAGE`, `SAME_PERSON_DIFFERENT_IMAGE`, `VISUALLY_RELATED_IMAGE`, `DIFFERENT_PERSON`) and evaluates consensus across distinct domains.
12. **Evidence Manifest Creation**: Constructs an RFC-8785 canonical evidence manifest capturing query metadata, top matched candidate, separation metrics, source relationship graph, and search telemetry.
13. **SHA-256 Fingerprinting**: Generates a deterministic SHA-256 cryptographic hash over the canonical JSON bytes.
14. **IPFS Storage**: Pins the canonical manifest to IPFS via Pinata, obtaining an immutable Content Identifier (CID).
15. **VERIFIED-Only Blockchain Anchoring**: Strictly if the biometric verdict is `VERIFIED`, calls `registerProof(dataHash, ipfsCID)` on `ProofRegistry.sol`. Unverified states (`REVIEW`, `REJECTED`, `NO RELIABLE MATCH`) are explicitly gated out.
16. **Independent Blockchain / IPFS Verification**: Standalone CLI script (`verify.py`) queries the smart contract, retrieves the manifest from IPFS, recomputes the SHA-256 hash, and verifies bit-for-bit integrity.
17. **Tamper Detection**: Multi-scenario tamper simulation (`tamper_demo.py`) demonstrates how mutating URLs, similarity scores, or metadata produces an immediate cryptographic hash mismatch.

---

## Key Capabilities

- **Deep Biometric Feature Extractor**: InsightFace `buffalo_l` ArcFace producing 512-dimensional float32 unit-norm embeddings.
- **Robust Face Detector**: InsightFace SCRFD detector operating at $640 \times 640$ resolution on CPU.
- **Dense Facial Geometry**: Extracts 106-point 2D dense landmarks and 68-point 3D facial landmarks from the model bundle.
- **3D Head Pose & Geometry Metrics**: Calculates true 3D head pose ($pitch, yaw, roll$), inter-ocular distance (IOD), eye-line tilt angle, jaw width, facial aspect ratio, and landmark consistency score.
- **Explainable Quality Gating**: Evaluates sharpness (Laplacian variance), exposure distribution, contrast, resolution, and frontality before downstream processing.
- **Multi-Face & Group Photo Support**: Detects all individuals in group photos; allows manual target face selection; evaluates every detected face in discovered candidate images.
- **Reverse Visual Discovery**: SerpApi Google Lens integration querying `type=visual_matches`.
- **Honest Pagination Telemetry**: Follows `serpapi_pagination.next` up to configured depth limits (`SEARCH_MAX_PAGES`), reporting truthful pages scanned and queries issued.
- **Source Relationship Graph**: Maps discovered sources into a typed graph structure (`SAME_IMAGE`, `SAME_PERSON_DIFFERENT_IMAGE`, `VISUALLY_RELATED_IMAGE`, `DIFFERENT_PERSON`).
- **Cross-Source Consensus Engine**: Evaluates domain diversity and flags corroboration strength (e.g. `STRONG_MULTI_PLATFORM_CONSENSUS`).
- **Separation Margin Metric**: Computes the numerical similarity gap between the top verified match and the strongest non-matching candidate.
- **RFC-8785 Canonical JSON**: Deterministic JSON serialization (lexicographically sorted keys, compact separators, integer-normalized floats) ensuring repeatable hashing across platforms.
- **Decentralized IPFS Storage**: Direct Pinata pinning with fallback resolution across multiple public gateways (Pinata, IPFS.io, dweb.link, Cloudflare).
- **Dual Blockchain Support**: Foundry Anvil local EVM node (`Chain ID: 31337`) for fast local development, with full configuration support for Polygon Amoy public testnet (`Chain ID: 80002`).
- **Solidity Smart Contract**: `ProofRegistry.sol` mapping 32-byte data hashes to submitter addresses, IPFS CIDs, and timestamps.
- **Independent Verification Script**: Standalone verification CLI (`verify.py`) requiring only the evidence hash.
- **Tamper Demonstration Script**: CLI demonstration (`tamper_demo.py`) proving avalanche-sensitive tamper detection against smart-contract anchors.
- **Responsive Web Interface**: Modern glassmorphic dark interface with 3-column layout, interactive Normal/Geometry view toggle, and zero horizontal overflow down to mobile viewports ($390 \times 844$).

---

## Architecture

```
[ QUERY IMAGE INPUT ] (File Upload / Preset / Group Photo)
        │
        ▼
[1] Face Detection & Quality Gating (SCRFD buffalo_l on CPU)
        │  ├── 5-Point Canonical Landmark Alignment
        │  ├── Dense Facial Geometry (106-pt 2D / 68-pt 3D)
        │  └── Multi-Factor Quality Check (Sharpness, Exposure, Frontality)
        ▼
[2] 512-D ArcFace Biometric Embedding & SHA-256 Hash
        │  └── Float32 L2 Normalization (Unit Norm)
        ▼
[3] Query Image Pinning (Pinata IPFS Gateway)
        │  └── Produces Public HTTP Gateway URL
        ▼
[4] Reverse Visual Discovery (SerpApi Google Lens engine)
        │  ├── Query type: visual_matches
        │  ├── Pagination: serpapi_pagination.next (Bounded by SEARCH_MAX_PAGES)
        │  └── Deduplication: Normalized URLs & Unique Thumbnails
        ▼
[5] Independent Candidate Image Processing (Concurrent in RAM)
        │  └── Local SCRFD Face Detection on Every Candidate Image
        ▼
[6] Candidate Biometric Verification (InsightFace ArcFace)
        │  ├── Compare Query Embedding vs. EVERY Detected Candidate Face
        │  ├── Compute Cosine Similarities [-1.0, 1.0]
        │  ├── Classify Decisions: VERIFIED (>=0.60), REVIEW (0.40–0.60), REJECTED (<0.40)
        │  ├── Calculate Separation Margin (Best Verified vs. Best Rejected)
        │  └── Build Source Relationship Graph & Consensus Level
        ▼
[7] Deterministic Evidence Manifest (RFC-8785 Canonical JSON)
        │  └── Lexicographical Key Sorting, Compact Separators, Compact Numbers
        ▼
[8] SHA-256 Cryptographic Fingerprint
        │  └── 64-Character Hex Digest over Canonical Manifest Bytes
        ▼
[9] Manifest Storage (IPFS via Pinata)
        │  └── Produces Manifest IPFS CID
        ▼
[10] Blockchain Proof Anchoring (ProofRegistry.sol)
        ├── IF VERIFIED  ──► Calls registerProof(dataHash, ipfsCID) [Anvil / Amoy]
        └── IF NOT VERIFIED ──► GATED OUT (PROOF NOT ANCHORED)
        │
        ▼
[ INDEPENDENT VERIFICATION & TAMPER DETECTION ]
        ├── verify.py      ──► Reads Blockchain -> Fetches IPFS -> Recomputes SHA-256
        └── tamper_demo.py ──► Mutates Evidence -> Verifies Hash Mismatch Against Ledger
```

---

## Face Analysis

Biometric processing is performed locally using the InsightFace `buffalo_l` model bundle on CPU via ONNX Runtime:

1. **Detection**: SCRFD detects face bounding boxes and returns raw confidence scores. Query images with multiple individuals are detected simultaneously, and the user or caller can select a specific target face index.
2. **Canonical Alignment**: 5 facial landmarks (left eye, right eye, nose tip, left mouth corner, right mouth corner) are mapped to standard canonical coordinates via similarity transformation.
3. **Dense Geometry**: In addition to 5-point alignment, the pipeline extracts:
   - 106 2D facial landmarks spanning the jawline, eyebrows, nose bridge, eyes, and lips.
   - 68 3D facial landmarks estimating depth ($Z$-coordinate).
   - 3D head pose angles: pitch (nod), yaw (turn), and roll (tilt) in degrees.
   - Inter-ocular distance (IOD), eye-line angle, jaw width, and facial aspect ratio.
   - Landmark consistency score comparing 5-point keypoint centers against 3D mesh eye/mouth coordinates.
4. **Embedding Extraction**: ArcFace projects the aligned face into a 512-dimensional hypersphere embedding, normalized to unit norm ($L_2$ norm = 1.0). A deterministic SHA-256 hash of the float32 byte sequence provides a unique biometric fingerprint.
5. **Cosine Similarity**: Biometric comparison uses standard cosine distance between normalized vectors $\mathbf{u}$ and $\mathbf{v}$:
   $$\text{similarity} = \mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{512} u_i v_i$$
   Cosine scores range from $-1.0$ to $1.0$.

> [!IMPORTANT]
> Biometric similarity is probabilistic evidence, not mathematical proof of legal identity. An ArcFace similarity score measures geometric and feature consistency across facial images; it must be interpreted alongside source credibility and contextual evidence.

---

## Web Discovery

The visual discovery stage locates publicly indexed web appearances of the query photo:

- **Search Provider & Engine**: Uses SerpApi's Google Lens search engine (`engine=google_lens`).
- **Query Mode**: Requests `type=visual_matches` to retrieve visual and exact web matches.
- **Pagination & Depth**: The pipeline inspects `serpapi_pagination.next` (or `next_page_token`). If present, it continues requesting subsequent pages up to `SEARCH_MAX_PAGES` (default: 3 pages) or until `SEARCH_MAX_RESULTS` (default: 60 results) is reached. If the search provider does not return a continuation token, pagination halts honestly and reports the true number of pages scanned.
- **Candidate Extraction**: Parses the `reverse_image_search` list, extracting page links, titles, thumbnail image URLs, and source attributions.
- **Deduplication**: Normalizes URLs (lowercasing domain and stripping trailing slashes) and deduplicates identical image URLs to avoid redundant candidate evaluations.
- **Candidate Image Downloads**: Candidate thumbnails are fetched concurrently in RAM using `requests.Session` with strict timeouts (10 seconds) and browser User-Agent headers. Inaccessible images (HTTP 404/403, network timeouts, corrupt bytes) are cleanly caught and tracked in telemetry without interrupting the pipeline.
- **Platform Classification**: Candidate domains are categorized using regex pattern matching into:
  - Instagram, LinkedIn, Facebook, X (Twitter), Reddit, YouTube, TikTok, Pinterest, News / Media, and General Web.

> [!NOTE]
> The search module queries publicly indexed and publicly accessible web content. It does not bypass private accounts, user authentication, paywalls, CAPTCHAs, or access-controlled services.

---

## Candidate Verification

Rather than trusting search engine page rankings, candidate images are independently evaluated locally:

1. **Independent Face Detection**: Every retrieved candidate image is passed through SCRFD. If an image contains multiple faces (e.g. a cast photo or group portrait), *every detectable face* is extracted and assigned an identifier (`FACE 01`, `FACE 02`, etc.).
2. **Embedding & Comparison**: Each detected candidate face is embedded via ArcFace and compared against the query face embedding. The candidate record retains the best matching face ID, similarity score, and bounding box.
3. **Decision Classification**:
   - **`VERIFIED`** ($\ge 0.60$ similarity): High-confidence genuine biometric match exceeding standard ArcFace identity verification thresholds, provided the candidate meets minimum quality standards.
   - **`REVIEW`** ($0.40 \le \text{similarity} < 0.60$): Borderline match requiring human inspection (e.g., adverse lighting, heavy compression, artistic portraits, partial occlusion).
   - **`REJECTED`** ($< 0.40$ similarity): Non-matching individual clearly below the biometric decision boundary.
   - **`NO RELIABLE MATCH`**: Returned when no candidate meets verification cutoffs or no faces were detectable in discovered images.
4. **Separation Margin**: Calculates the numerical delta between the top verified match and the highest-scoring non-verified candidate:
   $$\text{Margin} = \text{Similarity}_{\text{best verified}} - \text{Similarity}_{\text{best rejected}}$$
   - $\ge 0.30$: *Clear separation* (strong differentiation).
   - $0.10 - 0.30$: *Moderate separation* (meaningful differentiation).
   - $< 0.10$: *Narrow separation* (caution required; close competing candidates).

---

## Evidence & Provenance

The pipeline produces a comprehensive, deterministic evidence record structured according to **RFC-8785 (JSON Canonicalization Scheme)**:

- **Query Metadata**: Face detection bounding box, 5-point landmarks, dense geometry summary (106 2D points, 3D pose angles, IOD, consistency), and composite quality scores.
- **Top Match Metadata**: Target source URL, classified platform, matched face ID within candidate image, ArcFace cosine similarity score, and candidate image quality metrics.
- **Separation Metrics**: Best verified similarity, best rejected similarity, separation margin gap, and qualitative interpretation.
- **Source Relationship Graph**: Categorization of discovered candidates into:
  - `SAME_IMAGE`: Identical image hashes or exact visual duplicate matches.
  - `SAME_PERSON_DIFFERENT_IMAGE`: High-similarity matches from different photos corroborating identity.
  - `VISUALLY_RELATED_IMAGE`: Contextually related candidates meeting review thresholds.
  - `DIFFERENT_PERSON`: Distractor or competing faces.
- **Cross-Source Consensus**: Identifies whether identity is corroborated across multiple independent domains (e.g., `STRONG_MULTI_PLATFORM_CONSENSUS`).
- **Search Telemetry**: Execution timestamps, queries issued, pages scanned, candidate images retrieved, and unusable/inaccessible counts.
- **Deterministic Hashing**: The JSON object is normalized (integers formatted without trailing zeros), sorted lexicographically by key, formatted with compact separators (`","`, `":"`), encoded as UTF-8 bytes, and hashed using SHA-256 to produce a 64-character hexadecimal digest.

---

## IPFS Storage

The canonical evidence manifest is stored on IPFS using Pinata:

- **Pinning Endpoint**: Uploads the serialized canonical JSON to Pinata's `pinJSONToIPFS` API.
- **Content Identifier**: Pinata returns an immutable IPFS CID (e.g. `Qm...`).
- **Multi-Gateway Fallback**: The client resolves and retrieves IPFS records through multiple gateways in order:
  1. Configured dedicated Pinata gateway (`https://<custom>.mypinata.cloud/ipfs/`)
  2. Public IPFS gateway (`https://ipfs.io/ipfs/`)
  3. Cloudflare IPFS (`https://dweb.link/ipfs/`)
  4. Standard Pinata gateway (`https://gateway.pinata.cloud/ipfs/`)

---

## Blockchain Verification

On-chain proof anchoring is managed by `ProofRegistry.sol`:

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
    event ProofRegistered(bytes32 indexed dataHash, string ipfsCID, address indexed submitter, uint256 timestamp);

    function registerProof(bytes32 dataHash, string calldata ipfsCID) external {
        require(proofs[dataHash].timestamp == 0, "Already registered");
        proofs[dataHash] = Proof({
            submitter: msg.sender,
            ipfsCID: ipfsCID,
            timestamp: block.timestamp
        });
        emit ProofRegistered(dataHash, ipfsCID, msg.sender, block.timestamp);
    }

    function getProof(bytes32 dataHash) external view returns (address submitter, string memory ipfsCID, uint256 timestamp) {
        Proof memory p = proofs[dataHash];
        require(p.timestamp != 0, "Not found");
        return (p.submitter, p.ipfsCID, p.timestamp);
    }
}
```

### Supported Networks

1. **Foundry Anvil Local Node (Default)**:
   - RPC URL: `http://127.0.0.1:8545`
   - Chain ID: `31337`
   - Provides instant block confirmation (< 0.2s) and pre-funded local testing accounts.
2. **Polygon Amoy Public Testnet**:
   - RPC URL: `https://polygon-amoy.drpc.org`
   - Chain ID: `80002`
   - Sepolia-anchored EVM testnet for public demonstration.

### Strict VERIFIED-Only Anchoring

The pipeline strictly gates blockchain anchoring:
- **`VERIFIED` Verdict**: The contract's `registerProof(bytes32 dataHash, string ipfsCID)` function is called, anchoring the SHA-256 hash and IPFS CID with the sender's signature and block timestamp.
- **`REVIEW` / `REJECTED` / `NO RELIABLE MATCH` Verdicts**: No blockchain transaction is submitted. The UI and logs explicitly display `PROOF NOT ANCHORED`.

---

## Tamper Detection

Cryptographic tamper-evidence is demonstrated by `tamper_demo.py`:

1. **Fetches Authentic Proof**: Queries the smart contract on-chain for the authentic `ipfsCID` associated with the registered hash.
2. **Retrieves Manifest**: Downloads the authentic canonical JSON manifest from IPFS.
3. **Simulates Multi-Scenario Mutations**:
   - **Scenario 1 (Source URL Modification)**: Altering the discovered source URL (e.g. changing `instagram.com` to an attacker-controlled domain).
   - **Scenario 2 (Similarity Score Modification)**: Modifying the ArcFace cosine similarity score (e.g. inflating `0.42` to `0.85`).
   - **Scenario 3 (Decision Forgery)**: Changing a `REVIEW` or `REJECTED` status to `VERIFIED`.
4. **Avalanche Effect**: Because SHA-256 is collision-resistant, altering even a single character in the manifest produces an entirely different hash.
5. **Immediate Verification Failure**: Comparing the recomputed hash against the immutable on-chain record detects tampering immediately:

```text
===========================================================================
  TAMPER DETECTION RESULTS (MULTI-SCENARIO)
===========================================================================
  [AUTHENTIC] ORIGINAL RECORD VERIFICATION:
      Blockchain Hash : 602ead1fd2e6ecd335e79d1221abb1bc4f559c3630f060c0cceb6dbd59af914b
      Local Hash      : 602ead1fd2e6ecd335e79d1221abb1bc4f559c3630f060c0cceb6dbd59af914b
      Result          : VALID (Bit-for-bit cryptographic match)

  [SCENARIO 1] SOURCE URL MODIFICATION:
      Blockchain Hash : 602ead1fd2e6ecd335e79d1221abb1bc4f559c3630f060c0cceb6dbd59af914b
      Tampered Hash   : 53a70f7a05e6061a8dfd5166ab67d6e3d54096f04edba190b33073f7ed58c7a1
      Result          : TAMPER DETECTED!

  [SCENARIO 2] FACE SIMILARITY SCORE MODIFICATION:
      Blockchain Hash : 602ead1fd2e6ecd335e79d1221abb1bc4f559c3630f060c0cceb6dbd59af914b
      Tampered Hash   : 21b8b0ce85b8965f1ed523e7e84511c5efc76c12c055f26ca11e25b239282505
      Result          : TAMPER DETECTED!
===========================================================================
```

---

## User Interface

The web interface (`IDENTITY // VERIFY`) is built with a cyberpunk dark aesthetic (`#07090e` base):

- **Header Bar**: Live status badges showing blockchain connectivity (`ANVIL 31337`), active model (`BUFFALO_L ARCFACE 512-D`), and current block number.
- **Photo-Centric Workspace**: Central interactive canvas displaying the uploaded query image, overlaying detected bounding boxes, 5-point alignment keypoints, and quality confidence scores.
- **Normal vs. Geometry View**: Interactive toggle displaying the 106-point 2D landmark mesh, 68-point 3D landmark mesh, head pose pitch/yaw/roll gauges, and inter-ocular distance metrics.
- **Face Selector Chips**: When a group photo is loaded, chips (`FACE 01`, `FACE 02`, `FACE 03`) allow the user to select which face to investigate.
- **Face Analysis Panel (Left)**: Real-time telemetry displaying sharpness, exposure, contrast, frontality, and embedding SHA-256 hash.
- **Discovery Panel (Right)**: Search controls, depth indicators, and the **Discovery Complete Card** (`.discovery-complete-card`) detailing queries issued, pages scanned, and candidate images evaluated.
- **Candidate Gallery**: Displays discovered web candidates with thumbnails, domain badges, ArcFace cosine similarity scores, decision badges, and matched face IDs.
- **Source Relationship Graph**: Interactive card categorizing discovered candidates into duplicate images, same-person corroborations, related images, and distractors.
- **Evidence & Blockchain Cards**: When `VERIFIED`, displays the transaction hash, block number, IPFS CID, and local SHA-256 hash with direct links. When unverified, renders the `PROOF NOT ANCHORED` card.
- **Interactive Tamper Simulation**: Built-in interactive tamper test allowing users to simulate data tampering directly in the browser and watch the verification check fail.
- **Responsive Layout**: Designed to adapt cleanly across screen widths:
  - Desktop: $1920 \times 1080$ (3-column layout)
  - Laptop: $1440 \times 900$ (adaptive flexible grid)
  - Mobile: $390 \times 844$ (single-column vertically stacked layout without horizontal scrollbar)

---

## Technology Stack

| Category | Technology | Usage in Project |
|---|---|---|
| **Face Recognition** | [InsightFace](https://github.com/deepinsight/insightface) `buffalo_l` | SCRFD detector + ArcFace 512-d feature extractor running locally on CPU |
| **Computer Vision** | OpenCV (`opencv-python-headless`), NumPy, SciPy | Landmark alignment, affine transforms, image quality assessment, vector math |
| **Runtime & Execution** | ONNX Runtime (`onnxruntime==1.18.0`) | High-performance CPU execution for ArcFace and SCRFD ONNX models |
| **Web Discovery** | SerpApi Google Lens (`requests`) | Reverse visual search engine (`type=visual_matches`, `serpapi_pagination.next`) |
| **Decentralized Storage**| IPFS via Pinata (`requests`) | Content-addressed storage of canonical JSON manifests with multi-gateway fallback |
| **Blockchain (Local)** | Foundry Anvil (`Chain ID: 31337`) | Local EVM blockchain development node with instant confirmations |
| **Blockchain (Public)** | Polygon Amoy Testnet (`Chain ID: 80002`) | Public EVM testnet for decentralized proof verification |
| **Smart Contracts** | Solidity `0.8.20`, Hardhat | Tamper-evident proof registry smart contract (`ProofRegistry.sol`) |
| **Blockchain Adapter** | `web3.py` (v7.2.0) | Python interface for EVM transaction signing, event logging, and contract reads |
| **Web Server & API** | FastAPI, Uvicorn, Python-Multipart | REST API for face detection, live search, and verification endpoints |
| **Frontend** | Vanilla HTML5, CSS3, JavaScript (ES6+) | Near-black cybernetic UI, canvas overlays, and responsive design |
| **Testing** | Pytest (`pytest>=8.0.0`) | Automated test suite (55 tests) covering vision, blockchain, search, and API |

---

## Installation

### Prerequisites
- **Python**: Version `3.10` or `3.11` recommended
- **Node.js**: Version `18+` or `20+` (for Hardhat contract deployment)
- **Foundry Anvil**: Installed via `foundryup` (or download binary from [Foundry](https://book.getfoundry.sh/getting-started/installation))

### Step 1: Clone Repository & Set Up Virtual Environment
```bash
git clone https://github.com/Ankit2429/TrustLens.git
cd TrustLens

# Create and activate Python virtual environment
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate
```

### Step 2: Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Install Node Dependencies
```bash
npm install
```

### Step 4: Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Configure your credentials in `.env` (see the [Environment Variables](#environment-variables) section below).

### Step 5: Start Local Anvil Node
In a separate terminal window:
```bash
anvil --port 8545
```
Anvil will start listening on `http://127.0.0.1:8545` with Chain ID `31337` and 10 pre-funded accounts.

### Step 6: Deploy Smart Contract to Anvil
```bash
npm run compile
npm run deploy:anvil
```
This deploys `ProofRegistry.sol` to your local Anvil node and updates `BLOCKCHAIN_CONTRACT_ADDRESS` in `.env`.

---

## Environment Variables

All settings are configured in `.env`. Documented template from `.env.example`:

| Variable | Required | Default | Description |
|---|---|---|---|
| `SERPAPI_KEY` | **Yes** | — | API key for SerpApi Google Lens visual discovery. |
| `PINATA_JWT` | **Yes** | — | Pinata JWT authentication token for IPFS pinning. |
| `PINATA_GATEWAY` | No | `https://gateway.pinata.cloud/ipfs/` | Dedicated or public IPFS gateway URL for retrieval. |
| `BLOCKCHAIN_NETWORK` | No | `anvil` | Active blockchain target: `anvil` (local) or `polygon_amoy` (testnet). |
| `BLOCKCHAIN_RPC_URL` | No | `http://127.0.0.1:8545` | EVM JSON-RPC endpoint. |
| `BLOCKCHAIN_CHAIN_ID`| No | `31337` | Network Chain ID (31337 for Anvil, 80002 for Polygon Amoy). |
| `BLOCKCHAIN_PRIVATE_KEY`| **Yes** | `0xac09...` | Sender private key. *(The key shown in `.env.example` is Foundry Anvil's standard public development account #0 key; replace with your funded wallet key for Polygon Amoy).* |
| `BLOCKCHAIN_CONTRACT_ADDRESS` | **Yes** | — | Deployed address of `ProofRegistry.sol`. |
| `VERIFIED_THRESHOLD` | No | `0.60` | Minimum ArcFace cosine similarity for `VERIFIED` status. |
| `REVIEW_THRESHOLD` | No | `0.40` | Lower bound for `REVIEW` status; scores below are `REJECTED`. |
| `MIN_QUALITY_THRESHOLD` | No | `0.20` | Minimum composite face quality score to accept candidate. |
| `SEARCH_MAX_PAGES` | No | `3` | Maximum pagination pages to request from SerpApi. |
| `SEARCH_MAX_RESULTS` | No | `60` | Maximum unique candidate results to collect. |
| `SEARCH_MAX_CANDIDATE_IMAGES` | No | `40` | Maximum candidate thumbnails to download and evaluate. |
| `SEARCH_MAX_SOURCE_EXPANSIONS` | No | `8` | Maximum candidate web pages to inspect for high-res Open Graph images. |

---

## Usage

### 1. Launch the Interactive Web Application
```bash
python web_server.py --port 8000
# Or:
python -m webapp
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

### 2. Run the Full CLI Pipeline
Execute the complete end-to-end detection, discovery, verification, and on-chain anchoring pipeline:
```bash
python -m pipeline.main demo/sample_face.jpg
```

**Optional CLI Arguments:**
```bash
python -m pipeline.main <path_to_image> \
  --face-index 0 \
  --verified-threshold 0.60 \
  --review-threshold 0.40 \
  --min-quality 0.20 \
  --skip-blockchain
```

### 3. Independent Proof Re-Verification
Verify an existing proof directly from the blockchain and IPFS using only its 64-character SHA-256 hash:
```bash
python verify.py <evidence_sha256_hash>
```

### 4. Run Cryptographic Tamper Demonstration
Demonstrate that modifying any field in the evidence manifest violates the cryptographic hash anchored on the blockchain:
```bash
python tamper_demo.py <evidence_sha256_hash>
```

### 5. Run Automated Preflight Check
Verify API keys, blockchain RPC connectivity, contract deployment, and account balances before running:
```bash
python tests/preflight_check.py
```

### 6. Run Test Suite
```bash
pytest -v
```

---

## Decision Logic

The biometric decision engine evaluates each candidate based on local ArcFace cosine similarity and image quality scores:

| Decision | Condition | Explanation | Blockchain Action |
|---|---|---|---|
| **`VERIFIED`** | $\text{Similarity} \ge 0.60$ and $\text{Quality} \ge 0.20$ | High-confidence genuine match meeting biometric verification standards. | **Anchored on-chain** via `registerProof(hash, cid)` |
| **`REVIEW`** | $0.40 \le \text{Similarity} < 0.60$ or Borderline Quality | Ambiguous or partial match (lighting, angle, compression). Requires human review. | **Not anchored** (`PROOF NOT ANCHORED`) |
| **`REJECTED`** | $\text{Similarity} < 0.40$ | Distractor or non-matching individual below the decision threshold. | **Not anchored** (`PROOF NOT ANCHORED`) |
| **`NO RELIABLE MATCH`** | No candidates $\ge 0.40$ or no face detected | No discovered candidates matched the subject within acceptable bounds. | **Not anchored** (`PROOF NOT ANCHORED`) |

---

## Testing

The automated test suite consists of **55 tests** spanning biometric extraction, deterministic hashing, search pagination, blockchain integration, and API validation:

```text
============================== test session starts ==============================
collected 55 items

tests/test_accuracy_generalization.py ........                           [ 14%]
tests/test_anvil_blockchain.py .....                                    [ 23%]
tests/test_api.py .........                                             [ 40%]
tests/test_contracts.py ...                                             [ 45%]
tests/test_dense_mesh.py ........                                       [ 60%]
tests/test_pipeline.py ............                                     [ 81%]
tests/test_search_expansion.py ..........                               [100%]

======================== 55 passed, 13 warnings in 33.23s ========================
```

### Test Coverage Breakdown:
- `test_dense_mesh.py` (8 tests): Validates 106-point 2D landmarks, 68-point 3D landmarks, 3D pose extraction ($pitch, yaw, roll$), inter-ocular distance calculation, and landmark consistency scoring.
- `test_pipeline.py` (12 tests): Validates SCRFD face detection, 5-point canonical alignment, float32 L2 normalization, deterministic SHA-256 manifest hashing, quality scoring, separation margin calculation, and decision tiering.
- `test_search_expansion.py` (10 tests): Validates SerpApi pagination continuation (`serpapi_pagination.next`), candidate deduplication, domain classification, and search telemetry accounting.
- `test_anvil_blockchain.py` (5 tests): Validates Anvil RPC connection (Chain ID `31337`), account derivation, contract read/write calls, and event emissions.
- `test_accuracy_generalization.py` (8 tests): Tests model generalization against real-world lighting variations, blur, compression artifacts, and group photo candidate disambiguation.
- `test_api.py` (9 tests): Validates FastAPI REST endpoints (`/api/analyze`, `/api/verify`, `/api/tamper-demo`).
- `test_contracts.py` (3 tests): Tests smart-contract registration constraints, duplicate rejection, and non-existent hash error handling.

---

## Security, Privacy & Responsible Use

1. **Public Content Only**: The search pipeline only discovers and evaluates publicly indexed, accessible web images. It does not bypass private profile restrictions, authentication walls, or login credentials.
2. **Probabilistic Nature of Biometrics**: Face embeddings represent geometric and statistical feature vectors. Cosine similarity indicates biometric resemblance under specific capture conditions, not definitive legal identity.
3. **Evidence Integrity vs. Truth**: Anchoring a record on the blockchain provides immutable proof that a specific evidence manifest existed at a specific time and was signed by a specific address. It guarantees that the evidence record has not been tampered with; it does not guarantee the editorial accuracy of the third-party web page where the image was found.
4. **Secret Management**: All private keys and API credentials are kept exclusively in the local `.env` file, which is excluded from source control via `.gitignore`. The web application logs and API responses never output private keys or raw secrets.

---

## Limitations

- **Search Engine Indexation**: Discovery is bounded by what Google Lens and SerpApi have indexed and can return at the time of the query. Unindexed or newly posted images may not appear.
- **Inaccessible Sources**: Third-party websites may block scrapers, return HTTP 403/404, or use dynamic JavaScript loaders that prevent direct thumbnail retrieval. Unretrievable images are skipped.
- **Extreme Pose & Occlusion**: Profile views (yaw $> 60^\circ$), severe downward tilt, heavy sunglasses, or face coverings reduce detection confidence and embedding reliability.
- **Local EVM vs. Public Ledger**: Foundry Anvil is an ephemeral development blockchain running in local memory; restarting Anvil resets state unless configured with a state dump. For public demonstration, configure the Polygon Amoy testnet.
- **API Quotas**: SerpApi and Pinata operate with subscription quotas and rate limits; heavy concurrent scans should be scheduled within provider rate limits.

---

## Project Structure

```
TrustLens/
├── contracts/
│   └── ProofRegistry.sol         # Solidity smart contract for proof anchoring
├── demo/
│   ├── sample_face.jpg           # Sample reference portrait for demonstration
│   └── public_face_demo.jpg      # Sample query image for end-to-end testing
├── pipeline/
│   ├── chain.py                  # Web3 blockchain read/write adapter (Anvil & Amoy)
│   ├── face_id.py                # InsightFace SCRFD detector, ArcFace 512-d, dense mesh
│   ├── fingerprint.py            # RFC-8785 canonical JSON manifest & SHA-256 hashing
│   ├── ipfs_store.py             # Pinata IPFS pinning & multi-gateway retrieval
│   ├── main.py                   # End-to-end CLI pipeline orchestrator
│   └── search.py                 # SerpApi Google Lens discovery & pagination
├── scripts/
│   └── deploy.js                 # Hardhat deployment script for ProofRegistry.sol
├── tests/
│   ├── preflight_check.py        # System credential and connectivity validation
│   ├── test_accuracy_generalization.py # Evaluation of quality and pose variations
│   ├── test_anvil_blockchain.py  # Anvil local blockchain integration tests
│   ├── test_api.py               # FastAPI REST endpoint tests
│   ├── test_contracts.py         # Smart contract interaction tests
│   ├── test_dense_mesh.py        # 106-pt / 68-pt landmark & pose tests
│   ├── test_pipeline.py          # Vision pipeline unit tests
│   └── test_search_expansion.py  # Search pagination and candidate filtering tests
├── web/
│   ├── index.html                # Main application interface HTML
│   ├── app.js                    # Web application frontend logic & canvas rendering
│   └── style.css                 # Glassmorphic dark design system styles
├── webapp/
│   ├── __init__.py
│   ├── __main__.py               # python -m webapp entry point
│   ├── server.py                 # FastAPI backend server routes
│   └── static/                   # Mounted static assets (synchronized with web/)
├── .env.example                  # Environment configuration template
├── hardhat.config.js             # Hardhat EVM network configuration
├── package.json                  # Node dependencies and Hardhat scripts
├── pyproject.toml                # Pytest configuration
├── requirements.txt              # Python dependencies
├── tamper_demo.py                # Standalone cryptographic tamper demonstration
├── verify.py                     # Standalone on-chain re-verification CLI
└── web_server.py                 # Web server CLI launcher (python web_server.py)
```

---

## Demo Flow

A typical 30–60 second evaluation demonstration proceeds as follows:

1. **Launch Server**: Start Anvil (`anvil --port 8545`) and the web server (`python web_server.py --port 8000`).
2. **Open Dashboard**: Navigate to `http://127.0.0.1:8000/`. Observe the live status indicators (`ANVIL 31337`, `BUFFALO_L ARCFACE 512-D`).
3. **Upload Query Photo**: Click the sample preset or drag-and-drop a portrait photo.
4. **Inspect Face Analysis**: Observe the interactive canvas overlay with bounding boxes, 5-point alignment landmarks, and quality scores.
5. **Toggle Geometry View**: Switch to **Geometry View** to inspect the 106-point 2D dense landmark mesh, 68-point 3D mesh, and 3D pose angles ($pitch, yaw, roll$).
6. **Run Search & Verification**: Click **Run Search & Verification**. Watch the 6-stage pipeline progress indicator.
7. **Inspect Discovered Candidates**: Review the candidate gallery showing platform classifications, candidate thumbnails, and individual ArcFace similarity scores.
8. **Review Verification Verdict**: Examine the `VERIFIED` verdict card displaying the strongest match, separation margin gap, and identity consensus level.
9. **Inspect Proof Anchoring**: Check the anchored evidence card displaying the local SHA-256 hash, IPFS CID, and blockchain transaction hash.
10. **Test Independent Re-Verification**: Run `python verify.py <hash>` in the terminal to demonstrate standalone retrieval from Anvil and IPFS.
11. **Run Tamper Detection**: Click **Tamper With Biometric Data** in the web interface or execute `python tamper_demo.py <hash>` to verify that altered metadata is immediately rejected by the blockchain anchor.

---

## Submission Notes

This repository represents the complete, production-audited codebase for **`IDENTITY // VERIFY` (TrustLens)**. All features described—including SCRFD face detection, ArcFace 512-d embeddings, dense 106-point facial geometry, SerpApi Google Lens discovery, RFC-8785 canonical manifest generation, IPFS pinning, Foundry Anvil / Polygon Amoy smart-contract anchoring, and cryptographic tamper detection—are fully implemented, operational, and covered by 55 passing automated tests.

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT) as specified in the smart contract and repository sources.
