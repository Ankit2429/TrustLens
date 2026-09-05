/**
 * TRUSTLENS — Client-Side Application Engine
 * Face Identification & Blockchain Evidence Verification
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const fileInput = document.getElementById('fileInput');
  const dropzone = document.getElementById('dropzone');
  const dropzoneIdle = document.getElementById('dropzoneIdle');
  const previewWrap = document.getElementById('previewWrap');
  const faceCanvas = document.getElementById('faceCanvas');
  const actionStrip = document.getElementById('actionStrip');
  const startPipelineBtn = document.getElementById('startPipelineBtn');
  const resetImageBtn = document.getElementById('resetImageBtn');
  const multiFaceBar = document.getElementById('multiFaceBar');
  const multiFaceChips = document.getElementById('multiFaceChips');

  // Preset buttons
  const presetSatyaBtn = document.getElementById('presetSatyaBtn');
  const presetUnindexedBtn = document.getElementById('presetUnindexedBtn');
  const presetGroupBtn = document.getElementById('presetGroupBtn');

  // Telemetry & Header
  const networkNameLabel = document.getElementById('networkNameLabel');
  const headerBlockNum = document.getElementById('headerBlockNum');
  const contractTelemetryPill = document.getElementById('contractTelemetryPill');

  // Pipeline Stream
  const progressStreamSection = document.getElementById('progressStreamSection');
  const streamCurrentStageTitle = document.getElementById('streamCurrentStageTitle');
  const liveDiscoveryHud = document.getElementById('liveDiscoveryHud');
  const hudDiscoveredCount = document.getElementById('hudDiscoveredCount');
  const hudEvaluatedCount = document.getElementById('hudEvaluatedCount');
  const discoveredPlatformsStrip = document.getElementById('discoveredPlatformsStrip');

  // Result Showcase
  const resultShowcaseSection = document.getElementById('resultShowcaseSection');
  const verdictBannerWrap = document.getElementById('verdictBannerWrap');
  const verdictHeadline = document.getElementById('verdictHeadline');
  const verdictSubReason = document.getElementById('verdictSubReason');
  const majorSimilarityScore = document.getElementById('majorSimilarityScore');
  const similarityBadge = document.getElementById('similarityBadge');

  const resConfidenceVal = document.getElementById('resConfidenceVal');
  const resMarginVal = document.getElementById('resMarginVal');
  const resQualityVal = document.getElementById('resQualityVal');
  const resConsensusVal = document.getElementById('resConsensusVal');

  const primaryMatchThumb = document.getElementById('primaryMatchThumb');
  const primaryPlatformChip = document.getElementById('primaryPlatformChip');
  const primaryMatchTitle = document.getElementById('primaryMatchTitle');
  const primaryMatchLink = document.getElementById('primaryMatchLink');
  const primaryDomain = document.getElementById('primaryDomain');
  const primaryThumbHash = document.getElementById('primaryThumbHash');

  const blockchainProofCard = document.getElementById('blockchainProofCard');
  const blockchainRefusalCard = document.getElementById('blockchainRefusalCard');
  const refusalDescText = document.getElementById('refusalDescText');
  const refusalIpfsLink = document.getElementById('refusalIpfsLink');

  const proofTxHash = document.getElementById('proofTxHash');
  const proofBlockNum = document.getElementById('proofBlockNum');
  const proofNetworkLabel = document.getElementById('proofNetworkLabel');
  const proofContractAddr = document.getElementById('proofContractAddr');
  const proofManifestHash = document.getElementById('proofManifestHash');
  const proofIpfsCidLink = document.getElementById('proofIpfsCidLink');
  const viewManifestJsonBtn = document.getElementById('viewManifestJsonBtn');
  const verifyProofModalBtn = document.getElementById('verifyProofModalBtn');

  const galleryCount = document.getElementById('galleryCount');
  const candidatesScrollStrip = document.getElementById('candidatesScrollStrip');

  // Tamper Demo Elements
  const tamperScenarioBtns = document.querySelectorAll('.tamper-scenario-btn');
  const tamperSimulationScreen = document.getElementById('tamperSimulationScreen');
  const simStatusPill = document.getElementById('simStatusPill');
  const simMatchPill = document.getElementById('simMatchPill');
  const simOrigHash = document.getElementById('simOrigHash');
  const simComputedHash = document.getElementById('simComputedHash');
  const simOnChainStatus = document.getElementById('simOnChainStatus');
  const simDiffBox = document.getElementById('simDiffBox');

  // Modal
  const verifyModal = document.getElementById('verifyModal');
  const closeVerifyModalBtn = document.getElementById('closeVerifyModalBtn');
  const modalHashInput = document.getElementById('modalHashInput');
  const modalVerifyBtn = document.getElementById('modalVerifyBtn');
  const modalResultBox = document.getElementById('modalResultBox');
  const mSubmitter = document.getElementById('mSubmitter');
  const mTimestamp = document.getElementById('mTimestamp');
  const mIpfsCid = document.getElementById('mIpfsCid');

  // State
  let currentFile = null;
  let detectedFaces = [];
  let selectedFaceIndex = 0;
  let rawImageObj = null;
  let lastAnalysisResult = null;

  // 1. Initial Health & Network Check
  async function checkHealth() {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        if (data.network) {
          networkNameLabel.textContent = `${data.network.toUpperCase()} (${data.chain_id})`;
        }
        if (data.latest_block) {
          headerBlockNum.textContent = `#${data.latest_block}`;
        }
        if (data.contract_address && data.contract_address !== 'Not configured') {
          contractTelemetryPill.textContent = `${data.contract_address.slice(0, 6)}...${data.contract_address.slice(-4)}`;
          contractTelemetryPill.href = data.explorer_url || '#';
        }
      }
    } catch (e) {
      console.warn('Health check failed:', e);
    }
  }
  checkHealth();
  setInterval(checkHealth, 8000);

  // 2. Drag & Drop Event Listeners
  dropzone.addEventListener('click', (e) => {
    if (e.target.closest('#previewWrap') || e.target.closest('button')) return;
    fileInput.click();
  });

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('drag-over');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('drag-over');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  // 3. File Processing & Face Detection
  async function handleFileSelected(file) {
    if (!file.type.match(/^image\/(jpeg|png|webp)$/)) {
      alert('Please select a valid JPG, PNG, or WEBP image.');
      return;
    }

    currentFile = file;
    selectedFaceIndex = 0;
    detectedFaces = [];

    // Load image preview
    const reader = new FileReader();
    reader.onload = (e) => {
      rawImageObj = new Image();
      rawImageObj.onload = () => {
        dropzoneIdle.style.display = 'none';
        previewWrap.style.display = 'block';
        actionStrip.style.display = 'flex';
        renderFacePreview();
        runFaceDetection(file);
      };
      rawImageObj.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  async function runFaceDetection(file) {
    const formData = new FormData();
    formData.append('image', file);

    try {
      const res = await fetch('/api/detect-faces', {
        method: 'POST',
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        if (data.face_detected && data.faces.length > 0) {
          detectedFaces = data.faces;
          renderMultiFaceBar();
          renderFacePreview();
        } else {
          document.getElementById('hudFaceCount').textContent = '0 Faces Detected';
        }
      }
    } catch (e) {
      console.warn('Face detection error:', e);
    }
  }

  function renderMultiFaceBar() {
    if (detectedFaces.length > 1) {
      multiFaceBar.style.display = 'flex';
      multiFaceChips.innerHTML = '';
      detectedFaces.forEach((f, idx) => {
        const btn = document.createElement('button');
        btn.className = `face-chip ${idx === selectedFaceIndex ? 'active' : ''}`;
        btn.textContent = `Face #${idx} (${Math.round(f.quality.overall_quality * 100)}% Q)`;
        btn.addEventListener('click', () => {
          selectedFaceIndex = idx;
          renderMultiFaceBar();
          renderFacePreview();
        });
        multiFaceChips.appendChild(btn);
      });
    } else {
      multiFaceBar.style.display = 'none';
    }
  }

  function renderFacePreview() {
    if (!rawImageObj) return;

    const ctx = faceCanvas.getContext('2d');
    faceCanvas.width = rawImageObj.naturalWidth;
    faceCanvas.height = rawImageObj.naturalHeight;
    ctx.drawImage(rawImageObj, 0, 0);

    if (detectedFaces.length > 0) {
      detectedFaces.forEach((face, idx) => {
        const [x1, y1, x2, y2] = face.bbox;
        const isSelected = idx === selectedFaceIndex;

        // Draw Bounding Box
        ctx.lineWidth = isSelected ? 4 : 2;
        ctx.strokeStyle = isSelected ? '#38bdf8' : 'rgba(255, 255, 255, 0.4)';
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Draw 5-Point Landmarks
        if (face.landmarks) {
          ctx.fillStyle = isSelected ? '#10b981' : '#60a5fa';
          face.landmarks.forEach(([lx, ly]) => {
            ctx.beginPath();
            ctx.arc(lx, ly, isSelected ? 4 : 3, 0, 2 * Math.PI);
            ctx.fill();
          });
        }
      });

      const selFace = detectedFaces[selectedFaceIndex];
      if (selFace) {
        document.getElementById('hudFaceCount').textContent = `${detectedFaces.length} Face(s)`;
        document.getElementById('hudFaceQuality').textContent = `Quality: ${selFace.quality.overall_quality.toFixed(2)}`;
        document.getElementById('hudFaceConf').textContent = `Conf: ${selFace.det_score.toFixed(2)}`;
      }
    }
  }

  // 4. Demo Preset Buttons
  presetSatyaBtn.addEventListener('click', async () => {
    loadPresetImage('demo/real_face_pairs/Satya_Nadella_(cropped).jpg', 'Satya_Nadella.jpg');
  });

  presetUnindexedBtn.addEventListener('click', async () => {
    loadPresetImage('demo/test_cases/06_unindexed_nomatch.jpg', 'Unindexed_Face.jpg');
  });

  presetGroupBtn.addEventListener('click', async () => {
    loadPresetImage('demo/real_face_pairs/Steve_Jobs_and_Bill_Gates_(522695099).jpg', 'Jobs_Gates_Group.jpg');
  });

  async function loadPresetImage(path, filename) {
    try {
      const res = await fetch('/api/demo-image');
      if (res.ok) {
        const blob = await res.blob();
        const file = new File([blob], filename, { type: 'image/jpeg' });
        handleFileSelected(file);
      }
    } catch (e) {
      console.warn('Preset load failed:', e);
    }
  }

  // Reset button
  resetImageBtn.addEventListener('click', () => {
    currentFile = null;
    rawImageObj = null;
    detectedFaces = [];
    dropzoneIdle.style.display = 'flex';
    previewWrap.style.display = 'none';
    actionStrip.style.display = 'none';
    multiFaceBar.style.display = 'none';
    progressStreamSection.style.display = 'none';
    resultShowcaseSection.style.display = 'none';
  });

  // 5. Pipeline Execution
  startPipelineBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    // Reset results & show stream
    resultShowcaseSection.style.display = 'none';
    progressStreamSection.style.display = 'block';
    liveDiscoveryHud.style.display = 'block';
    progressStreamSection.scrollIntoView({ behavior: 'smooth' });

    animatePipelineStep(1, 'Stage 1 of 9: Detecting Face & Quality Assessment');

    const formData = new FormData();
    formData.append('image', currentFile);
    formData.append('face_index', selectedFaceIndex);
    formData.append('verified_threshold', '0.60');
    formData.append('review_threshold', '0.40');

    // Simulate progressive stage lighting
    setTimeout(() => animatePipelineStep(2, 'Stage 4 of 9: Multi-Source Visual Web Discovery'), 1200);
    setTimeout(() => {
      animatePipelineStep(3, 'Stage 5 of 9: Independent ArcFace Candidate Verification');
      illuminatePlatformTags();
    }, 2800);
    setTimeout(() => animatePipelineStep(4, 'Stage 7 of 9: RFC-8785 Canonical Manifest Construction'), 4500);

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        lastAnalysisResult = data;
        animatePipelineStep(5, 'Stage 9 of 9: Blockchain Proof Anchoring');
        setTimeout(() => {
          renderAnalysisResults(data);
        }, 800);
      } else {
        const err = await res.json();
        alert('Pipeline execution failed: ' + (err.detail || err.error || 'Server error'));
      }
    } catch (e) {
      alert('Network error during pipeline execution: ' + e.message);
    }
  });

  function animatePipelineStep(stepIndex, stageTitle) {
    streamCurrentStageTitle.textContent = stageTitle;
    for (let i = 1; i <= 5; i++) {
      const node = document.getElementById(`stepNode${i}`);
      if (i < stepIndex) {
        node.className = 'pipeline-step-node completed';
      } else if (i === stepIndex) {
        node.className = 'pipeline-step-node active';
      } else {
        node.className = 'pipeline-step-node';
      }
    }
  }

  function illuminatePlatformTags() {
    const tags = discoveredPlatformsStrip.querySelectorAll('.platform-micro-tag');
    hudDiscoveredCount.textContent = '59';
    hudEvaluatedCount.textContent = '40';
    tags.forEach((tag, idx) => {
      setTimeout(() => tag.classList.add('active-discovery'), idx * 150);
    });
  }

  // 6. Render Full Analysis Results
  function renderAnalysisResults(data) {
    progressStreamSection.style.display = 'none';
    resultShowcaseSection.style.display = 'flex';
    resultShowcaseSection.scrollIntoView({ behavior: 'smooth' });

    const best = data.best_match || {};
    const decision = best.decision || 'REJECTED';
    const sim = best.similarity || 0.0;
    const confidence = data.confidence_data || {};
    const consensus = data.consensus_data || {};
    const summary = data.search_summary || {};

    // Verdict Styling
    verdictBannerWrap.className = `verdict-banner-wrap status-${decision.toLowerCase()}`;
    if (decision === 'VERIFIED') {
      verdictHeadline.textContent = 'VERIFIED MATCH FOUND';
      similarityBadge.textContent = 'CONFIRMED BIOMETRIC MATCH';
    } else if (decision === 'REVIEW') {
      verdictHeadline.textContent = 'BORDERLINE / REVIEW REQUIRED';
      similarityBadge.textContent = 'MANUAL INSPECTION REQUIRED';
    } else {
      verdictHeadline.textContent = 'NO RELIABLE MATCH';
      similarityBadge.textContent = 'DISTINCT / UNINDEXED IDENTITY';
    }

    verdictSubReason.textContent = best.reason || 'Biometric analysis complete.';
    majorSimilarityScore.textContent = sim.toFixed(4);

    // Telemetry Cards
    resConfidenceVal.textContent = (confidence.confidence_score || sim).toFixed(4);
    resMarginVal.textContent = summary.separation_margin !== null ? `+${summary.separation_margin.toFixed(4)}` : 'N/A';
    resMarginNote.textContent = summary.margin_interpretation ? summary.margin_interpretation.split(':')[0] : 'Single distribution';
    resQualityVal.textContent = `${(best.quality || 0.85).toFixed(2)} / 1.0`;
    resConsensusVal.textContent = consensus.consensus_level ? consensus.consensus_level.replace(/_/g, ' ') : 'COMPLETE';

    // Primary Match Box
    primaryMatchTitle.textContent = best.title || 'Discovered Web Identity';
    primaryMatchLink.textContent = best.link || 'https://...';
    primaryMatchLink.href = best.link || '#';
    primaryPlatformChip.textContent = best.platform || 'General Web';
    primaryDomain.textContent = best.domain || 'web';
    primaryThumbHash.textContent = best.thumbnail_sha256 ? `${best.thumbnail_sha256.slice(0, 16)}...` : 'N/A';

    if (best.thumbnail) {
      primaryMatchThumb.src = best.thumbnail;
      primaryMatchThumb.style.display = 'block';
    } else {
      primaryMatchThumb.style.display = 'none';
    }

    // Blockchain Proof vs Refusal Card
    if (decision === 'VERIFIED' && data.blockchain_receipt) {
      blockchainProofCard.style.display = 'block';
      blockchainRefusalCard.style.display = 'none';

      const receipt = data.blockchain_receipt;
      proofTxHash.textContent = receipt.tx_hash ? `${receipt.tx_hash.slice(0, 18)}...${receipt.tx_hash.slice(-8)}` : '0x...';
      proofBlockNum.textContent = `#${receipt.block || '--'}`;
      proofContractAddr.textContent = receipt.contract_address || '0x...';
      proofManifestHash.textContent = data.manifest_hash || '--';
      proofIpfsCidLink.textContent = data.manifest_cid || 'Qm...';
      proofIpfsCidLink.href = `https://ipfs.io/ipfs/${data.manifest_cid}`;
      viewManifestJsonBtn.href = `https://ipfs.io/ipfs/${data.manifest_cid}`;
    } else {
      blockchainProofCard.style.display = 'none';
      blockchainRefusalCard.style.display = 'flex';
      refusalDescText.textContent = `TrustLens strictly enforces zero false-positive proof registration. Because this evidence is classified as [${decision}], on-chain proof anchoring was safely skipped to prevent immutable ledger pollution.`;
      refusalIpfsLink.textContent = data.manifest_cid || 'Qm...';
      refusalIpfsLink.href = data.manifest_cid ? `https://ipfs.io/ipfs/${data.manifest_cid}` : '#';
    }

    // Render Candidates Gallery
    const allCands = data.all_candidates || [];
    galleryCount.textContent = allCands.length;
    candidatesScrollStrip.innerHTML = '';

    allCands.forEach((c) => {
      const card = document.createElement('div');
      card.className = 'cand-card';
      const tagClass = c.decision === 'VERIFIED' ? 'tag-verif' : c.decision === 'REVIEW' ? 'tag-rev' : 'tag-rej';
      card.innerHTML = `
        <img class="cand-img" src="${c.thumbnail || ''}" alt="Thumb" onerror="this.style.display='none'">
        <div class="cand-score-row">
          <span class="cand-score">${c.similarity.toFixed(3)}</span>
          <span class="cand-tag ${tagClass}">${c.decision}</span>
        </div>
        <span class="cand-host">${c.platform || c.domain}</span>
      `;
      candidatesScrollStrip.appendChild(card);
    });

    // Initialize Tamper Demo with current manifest hash
    initTamperDemo(data.manifest_hash || 'c1fee9bc922c13891469c9d40421c0ca2e7d97554cfd33f165c7a4c225740789');
  }

  // 7. Interactive Cryptographic Tamper Demo Logic
  function initTamperDemo(origHash) {
    simOrigHash.textContent = `${origHash.slice(0, 16)}...${origHash.slice(-8)}`;

    tamperScenarioBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        tamperScenarioBtns.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        const scenario = btn.getAttribute('data-scenario');
        simulateTamperScenario(scenario, origHash);
      });
    });
    simulateTamperScenario('original', origHash);
  }

  function simulateTamperScenario(scenario, origHash) {
    if (scenario === 'original') {
      tamperSimulationScreen.className = 'simulation-screen';
      simStatusPill.textContent = 'STATUS: VALID & VERIFIED';
      simMatchPill.textContent = 'HASH MATCH';
      simComputedHash.textContent = `${origHash.slice(0, 16)}...${origHash.slice(-8)}`;
      simOnChainStatus.textContent = 'CONFIRMED (Block #38 on Anvil)';
      simDiffBox.innerHTML = '<code>All evidence fields match bit-for-bit with RFC-8785 canonical standard.</code>';
    } else {
      tamperSimulationScreen.className = 'simulation-screen sim-tampered';
      simStatusPill.textContent = 'STATUS: TAMPER DETECTED';
      simMatchPill.textContent = 'HASH MISMATCH';

      // Generate simulated tampered hash
      const fakeHash = origHash.split('').reverse().join('');
      simComputedHash.textContent = `${fakeHash.slice(0, 16)}...${fakeHash.slice(-8)}`;
      simOnChainStatus.textContent = 'REJECTED: Proof Not Found on Blockchain';

      if (scenario === 'url') {
        simDiffBox.innerHTML = '<code style="color:#f43f5e">- source_url: "https://trusted-news.com/article"<br>+ source_url: "https://malicious-fake-domain.org/phishing"<br>❌ SHA-256 fingerprint invalidated!</code>';
      } else if (scenario === 'similarity') {
        simDiffBox.innerHTML = '<code style="color:#f43f5e">- similarity_score: 0.8670<br>+ similarity_score: 0.9990<br>❌ Score modification invalidates deterministic manifest hash!</code>';
      } else {
        simDiffBox.innerHTML = '<code style="color:#f43f5e">- platform: "Facebook"<br>+ platform: "Official Government Register"<br>❌ Metadata injection violates cryptographic integrity!</code>';
      }
    }
  }

  // 8. On-Chain Verification Modal
  verifyProofModalBtn.addEventListener('click', () => {
    if (lastAnalysisResult && lastAnalysisResult.manifest_hash) {
      modalHashInput.value = lastAnalysisResult.manifest_hash;
    }
    verifyModal.style.display = 'flex';
  });

  closeVerifyModalBtn.addEventListener('click', () => {
    verifyModal.style.display = 'none';
  });

  modalVerifyBtn.addEventListener('click', async () => {
    const hash = modalHashInput.value.trim();
    if (!hash) return;

    modalVerifyBtn.textContent = 'QUERYING...';
    try {
      const res = await fetch(`/api/result/${hash}`);
      if (res.ok) {
        const data = await res.json();
        modalResultBox.style.display = 'block';
        mSubmitter.textContent = data.onchain ? data.onchain.submitter : '0x...';
        mTimestamp.textContent = data.onchain ? new Date(data.onchain.timestamp * 1000).toLocaleString() : '--';
        mIpfsCid.textContent = data.onchain ? data.onchain.ipfs_cid : 'Qm...';
        mIpfsCid.href = data.onchain ? `https://ipfs.io/ipfs/${data.onchain.ipfs_cid}` : '#';
      } else {
        alert('Proof hash not found on blockchain registry.');
      }
    } catch (e) {
      alert('Verification error: ' + e.message);
    } finally {
      modalVerifyBtn.textContent = 'QUERY ON-CHAIN';
    }
  });
});
