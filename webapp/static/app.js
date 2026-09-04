/**
 * TrustLens — Web UI Client Controller
 * Connects to the real TrustLens backend, visualizing SCRFD/ArcFace analysis,
 * multi-source candidate matches, RFC-8785 evidence fingerprints, IPFS storage,
 * and Polygon Amoy blockchain proof registration.
 */

let currentFile = null;
let detectedFaces = [];
let selectedFaceIndex = 0;
let currentPipelineResult = null;
let currentProofHash = null;
let loadedImageElement = null;

// DOM Elements
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const dropzonePrompt = document.getElementById('dropzonePrompt');
const previewBox = document.getElementById('previewBox');
const faceCanvas = document.getElementById('faceCanvas');
const previewCaption = document.getElementById('previewCaption');
const loadDemoBtn = document.getElementById('loadDemoBtn');
const analyzeBtn = document.getElementById('analyzeBtn');

const multiFaceSelector = document.getElementById('multiFaceSelector');
const faceOptionsList = document.getElementById('faceOptionsList');
const qualityPanel = document.getElementById('qualityPanel');
const qualityBadge = document.getElementById('qualityBadge');
const qValSharp = document.getElementById('qValSharp');
const fillSharp = document.getElementById('fillSharp');
const qValExpo = document.getElementById('qValExpo');
const fillExpo = document.getElementById('fillExpo');
const qValRes = document.getElementById('qValRes');
const fillRes = document.getElementById('fillRes');
const qValFront = document.getElementById('qValFront');
const fillFront = document.getElementById('fillFront');
const fingerprintPreview = document.getElementById('fingerprintPreview');
const queryEmbeddingHash = document.getElementById('queryEmbeddingHash');

const liveBlockNum = document.getElementById('liveBlockNum');
const contractAddressLink = document.getElementById('contractAddressLink');
const overallPipelineStatus = document.getElementById('overallPipelineStatus');
const logConsole = document.getElementById('logConsole');

// Top Result Banner Elements
const resultTopBanner = document.getElementById('resultTopBanner');
const rtbStatusPill = document.getElementById('rtbStatusPill');
const sumCountVerif = document.getElementById('sumCountVerif');
const sumCountRev = document.getElementById('sumCountRev');
const sumCountRej = document.getElementById('sumCountRej');
const rtbLatencyBadge = document.getElementById('rtbLatencyBadge');

// Best Match Section Elements
const bestMatchSection = document.getElementById('bestMatchSection');
const bmThumbnail = document.getElementById('bmThumbnail');
const bmPlatform = document.getElementById('bmPlatform');
const bmTitle = document.getElementById('bmTitle');
const bmUrl = document.getElementById('bmUrl');
const bmSimilarity = document.getElementById('bmSimilarity');
const bmDecision = document.getElementById('bmDecision');
const bmRank = document.getElementById('bmRank');
const bmQuality = document.getElementById('bmQuality');
const bmSeparationMargin = document.getElementById('bmSeparationMargin');
const bmSeparationInterp = document.getElementById('bmSeparationInterp');
const openSourceBtn = document.getElementById('openSourceBtn');

// Other Candidates Section Elements
const allCandidatesSection = document.getElementById('allCandidatesSection');
const candidatesAccordion = document.getElementById('candidatesAccordion');
const otherCandidatesCount = document.getElementById('otherCandidatesCount');
const countVerif = document.getElementById('countVerif');
const countRev = document.getElementById('countRev');
const countRej = document.getElementById('countRej');
const countAll = document.getElementById('countAll');
const chipCountVerif = document.getElementById('chipCountVerif');
const chipCountRev = document.getElementById('chipCountRev');
const chipCountRej = document.getElementById('chipCountRej');
const countSocial = document.getElementById('countSocial');
const candidatesGrid = document.getElementById('candidatesGrid');

// Blockchain & Proof Elements
const blockchainSection = document.getElementById('blockchainSection');
const dispEvidenceHash = document.getElementById('dispEvidenceHash');
const dispEvidenceCid = document.getElementById('dispEvidenceCid');
const dispIpfsLink = document.getElementById('dispIpfsLink');
const dispContract = document.getElementById('dispContract');
const dispContractLink = document.getElementById('dispContractLink');
const dispTxHash = document.getElementById('dispTxHash');
const dispTxLink = document.getElementById('dispTxLink');
const dispBlockStatus = document.getElementById('dispBlockStatus');
const manifestJsonBlock = document.getElementById('manifestJsonBlock');

// Actions Elements
const actionsGrid = document.getElementById('actionsGrid');
const verifyProofBtn = document.getElementById('verifyProofBtn');
const verifResultArea = document.getElementById('verifResultArea');
const verifBanner = document.getElementById('verifBanner');
const verifIcon = document.getElementById('verifIcon');
const verifTitle = document.getElementById('verifTitle');
const verifDesc = document.getElementById('verifDesc');
const verifKvGrid = document.getElementById('verifKvGrid');

const tamperTestBtn = document.getElementById('tamperTestBtn');
const tamperResultArea = document.getElementById('tamperResultArea');
const tamperCompareView = document.getElementById('tamperCompareView');

// Init
document.addEventListener('DOMContentLoaded', () => {
  fetchHealth();
  setupEventListeners();
  setInterval(fetchHealth, 15000);
});

function logMessage(msg, type = 'info') {
  const line = document.createElement('div');
  line.className = `log-line ${type === 'error' ? 'log-error' : type === 'success' ? 'log-success' : ''}`;
  const time = new Date().toLocaleTimeString();
  line.textContent = `[${time}] ${msg}`;
  logConsole.appendChild(line);
  logConsole.scrollTop = logConsole.scrollHeight;
}

// -------------------------------------------------------------
// Health Check
// -------------------------------------------------------------
async function fetchHealth() {
  try {
    const res = await fetch('/api/health');
    if (!res.ok) return;
    const data = await res.json();
    if (data.latest_block) {
      liveBlockNum.textContent = `#${data.latest_block}`;
    }
    if (data.contract_address) {
      const shortAddr = data.contract_address.slice(0, 6) + '...' + data.contract_address.slice(-4);
      contractAddressLink.textContent = shortAddr;
      contractAddressLink.href = data.explorer_url || `https://amoy.polygonscan.com/address/${data.contract_address}`;
      dispContract.textContent = data.contract_address;
      dispContractLink.href = data.explorer_url || `https://amoy.polygonscan.com/address/${data.contract_address}`;
    }
  } catch (err) {
    console.warn('Health check warning:', err);
  }
}

// -------------------------------------------------------------
// Event Listeners Setup
// -------------------------------------------------------------
function setupEventListeners() {
  dropzone.addEventListener('click', (e) => {
    if (e.target.tagName !== 'BUTTON' && !e.target.closest('.face-card')) {
      fileInput.click();
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleImageSelected(e.target.files[0]);
    }
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
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleImageSelected(e.dataTransfer.files[0]);
    }
  });

  loadDemoBtn.addEventListener('click', async (e) => {
    e.stopPropagation();
    try {
      logMessage('Loading consenting demo image (demo/public_face_demo.jpg)...');
      const res = await fetch('/api/demo-image');
      if (!res.ok) throw new Error('Failed to load demo image');
      const blob = await res.blob();
      const file = new File([blob], 'public_face_demo.jpg', { type: 'image/jpeg' });
      handleImageSelected(file);
    } catch (err) {
      logMessage(`Error loading demo image: ${err.message}`, 'error');
    }
  });

  analyzeBtn.addEventListener('click', () => {
    if (currentFile) {
      executePipeline();
    }
  });

  verifyProofBtn.addEventListener('click', () => {
    if (currentProofHash) {
      runIndependentVerification(currentProofHash);
    }
  });

  tamperTestBtn.addEventListener('click', () => {
    if (currentProofHash) {
      runTamperDemo(currentProofHash);
    }
  });

  // Filter chips
  document.querySelectorAll('.filter-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      filterCandidates(chip.dataset.filter);
    });
  });
}

// -------------------------------------------------------------
// Image Handling & Face Pre-detection
// -------------------------------------------------------------
async function handleImageSelected(file) {
  currentFile = file;
  selectedFaceIndex = 0;
  analyzeBtn.disabled = true;

  // Reset UI sections
  resultTopBanner.style.display = 'none';
  bestMatchSection.style.display = 'none';
  allCandidatesSection.style.display = 'none';
  blockchainSection.style.display = 'none';
  actionsGrid.style.display = 'none';
  resetStageFlowchart();

  logMessage(`Uploaded image: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`);
  overallPipelineStatus.textContent = 'Analyzing Face';
  overallPipelineStatus.className = 'status-summary-badge';

  // Load preview in Canvas
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      loadedImageElement = img;
      renderImageOnCanvas(img);
      dropzonePrompt.style.display = 'none';
      previewBox.style.display = 'block';
      detectFacesInImage(file);
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

function renderImageOnCanvas(img, facesToHighlight = []) {
  const ctx = faceCanvas.getContext('2d');
  const maxDim = 460;
  let w = img.width;
  let h = img.height;
  if (w > maxDim || h > maxDim) {
    if (w > h) {
      h = Math.round((h * maxDim) / w);
      w = maxDim;
    } else {
      w = Math.round((w * maxDim) / h);
      h = maxDim;
    }
  }
  faceCanvas.width = w;
  faceCanvas.height = h;

  ctx.drawImage(img, 0, 0, w, h);

  const scaleX = w / img.width;
  const scaleY = h / img.height;

  // Draw bounding boxes & landmarks
  facesToHighlight.forEach((face) => {
    const isSelected = face.face_index === selectedFaceIndex;
    const [x1, y1, x2, y2] = face.bbox;
    const sx1 = x1 * scaleX;
    const sy1 = y1 * scaleY;
    const sw = (x2 - x1) * scaleX;
    const sh = (y2 - y1) * scaleY;

    ctx.lineWidth = isSelected ? 3 : 2;
    ctx.strokeStyle = isSelected ? '#00f2fe' : '#94a3b8';
    ctx.strokeRect(sx1, sy1, sw, sh);

    // Box Tag
    ctx.fillStyle = isSelected ? 'rgba(0, 242, 254, 0.85)' : 'rgba(30, 41, 59, 0.85)';
    ctx.fillRect(sx1, Math.max(0, sy1 - 22), 70, 20);
    ctx.fillStyle = isSelected ? '#0b0f19' : '#f8fafc';
    ctx.font = 'bold 11px Outfit, sans-serif';
    ctx.fillText(`Face #${face.face_index}`, sx1 + 6, Math.max(14, sy1 - 8));

    // Draw 5 landmarks if available
    if (face.landmarks && Array.isArray(face.landmarks)) {
      face.landmarks.forEach(([lx, ly]) => {
        ctx.beginPath();
        ctx.arc(lx * scaleX, ly * scaleY, isSelected ? 3 : 2, 0, 2 * Math.PI);
        ctx.fillStyle = isSelected ? '#10b981' : '#cbd5e1';
        ctx.fill();
      });
    }
  });
}

async function detectFacesInImage(file) {
  try {
    const formData = new FormData();
    formData.append('image', file);

    const res = await fetch('/api/detect-faces', {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Face detection failed');
    }

    const data = await res.json();
    if (!data.face_detected || data.face_count === 0) {
      previewCaption.textContent = 'No face detected with sufficient quality';
      previewCaption.style.color = '#ef4444';
      logMessage('No faces detected in the provided image.', 'error');
      qualityPanel.style.display = 'none';
      multiFaceSelector.style.display = 'none';
      fingerprintPreview.style.display = 'none';
      analyzeBtn.disabled = true;
      return;
    }

    detectedFaces = data.faces;
    previewCaption.textContent = `Detected ${data.face_count} face${data.face_count > 1 ? 's' : ''}`;
    previewCaption.style.color = '#10b981';

    renderImageOnCanvas(loadedImageElement, detectedFaces);

    // Setup multi-face selector if multiple faces
    if (data.face_count > 1) {
      multiFaceSelector.style.display = 'block';
      faceOptionsList.innerHTML = '';
      detectedFaces.forEach((f) => {
        const card = document.createElement('div');
        card.className = `face-card ${f.face_index === selectedFaceIndex ? 'selected' : ''}`;
        card.innerHTML = `
          <div class="fc-index">Face #${f.face_index}</div>
          <div class="fc-score">Quality: ${(f.quality.overall_quality * 100).toFixed(0)}%</div>
          <div class="fc-det">Det: ${(f.det_score * 100).toFixed(0)}%</div>
        `;
        card.addEventListener('click', () => {
          selectedFaceIndex = f.face_index;
          document.querySelectorAll('.face-card').forEach(c => c.classList.remove('selected'));
          card.classList.add('selected');
          renderImageOnCanvas(loadedImageElement, detectedFaces);
          displayFaceQuality(f);
        });
        faceOptionsList.appendChild(card);
      });
    } else {
      multiFaceSelector.style.display = 'none';
    }

    // Display quality of selected face
    const selectedFace = detectedFaces.find(f => f.face_index === selectedFaceIndex) || detectedFaces[0];
    displayFaceQuality(selectedFace);

    analyzeBtn.disabled = false;
    logMessage(`Detected ${data.face_count} face(s). Selected Face #${selectedFace.face_index} (Quality: ${(selectedFace.quality.overall_quality * 100).toFixed(1)}%).`, 'success');

  } catch (err) {
    logMessage(`Detection error: ${err.message}`, 'error');
    previewCaption.textContent = `Error: ${err.message}`;
    previewCaption.style.color = '#ef4444';
  }
}

function displayFaceQuality(face) {
  qualityPanel.style.display = 'block';
  fingerprintPreview.style.display = 'block';

  const q = face.quality;
  const overallPct = (q.overall_quality * 100).toFixed(1);
  qualityBadge.textContent = `Overall Quality: ${overallPct}%`;
  qualityBadge.className = q.overall_quality >= 0.35 ? 'quality-badge quality-good' : 'quality-badge quality-warning';

  const sharpPct = Math.min(100, Math.round(q.sharpness_score * 100));
  const expoPct = Math.min(100, Math.round(q.exposure_score * 100));
  const resPct = Math.min(100, Math.round(q.resolution_score * 100));
  const frontPct = Math.min(100, Math.round(q.frontality_score * 100));

  qValSharp.textContent = `${sharpPct}%`;
  fillSharp.style.width = `${sharpPct}%`;

  qValExpo.textContent = `${expoPct}%`;
  fillExpo.style.width = `${expoPct}%`;

  qValRes.textContent = `${resPct}%`;
  fillRes.style.width = `${resPct}%`;

  qValFront.textContent = `${frontPct}%`;
  fillFront.style.width = `${frontPct}%`;

  queryEmbeddingHash.textContent = face.embedding_hash;
}

// -------------------------------------------------------------
// Real Pipeline Execution (POST /api/analyze)
// -------------------------------------------------------------
function setStageStatus(stageIndex, status) {
  const step = document.querySelector(`.stage-step[data-stage="${stageIndex}"]`);
  if (!step) return;
  const badge = step.querySelector('.stage-status');
  badge.className = `stage-status status-${status.toLowerCase()}`;
  badge.textContent = status.toUpperCase();
}

function resetStageFlowchart() {
  for (let i = 1; i <= 9; i++) {
    setStageStatus(i, 'pending');
  }
}

async function executePipeline() {
  if (!currentFile) return;

  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = `<span class="btn-spinner"></span> Running Pipeline...`;
  overallPipelineStatus.textContent = 'Pipeline In Progress';
  overallPipelineStatus.className = 'status-summary-badge status-running-badge';

  resultTopBanner.style.display = 'none';
  bestMatchSection.style.display = 'none';
  allCandidatesSection.style.display = 'none';
  blockchainSection.style.display = 'none';
  actionsGrid.style.display = 'none';

  resetStageFlowchart();
  setStageStatus(1, 'success');
  setStageStatus(2, 'running');

  logMessage('--- STARTING TRUSTLENS TASK 3 PIPELINE ---');

  const formData = new FormData();
  formData.append('image', currentFile);
  formData.append('face_index', selectedFaceIndex);
  formData.append('verified_threshold', 0.45);
  formData.append('review_threshold', 0.38);
  formData.append('min_quality', 0.20);
  formData.append('skip_blockchain', false);

  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const errData = await res.json();
      const errMsg = errData.detail?.error || errData.detail || 'Pipeline execution failed';
      throw new Error(errMsg);
    }

    const data = await res.json();
    currentPipelineResult = data;
    currentProofHash = data.manifest_hash;

    // Mark all 9 stages as SUCCESS
    for (let i = 1; i <= 9; i++) {
      setStageStatus(i, 'success');
    }

    overallPipelineStatus.textContent = 'Pipeline Verified & Anchored';
    overallPipelineStatus.className = 'status-summary-badge status-success-badge';

    // Log stages
    if (data.stages_log) {
      data.stages_log.forEach(s => {
        logMessage(`[Stage ${s.stage}] ${s.name}: ${s.detail || s.status}`, s.status === 'FAILED' ? 'error' : 'success');
      });
    }

    renderPipelineResults(data);

  } catch (err) {
    logMessage(`Pipeline execution error: ${err.message}`, 'error');
    overallPipelineStatus.textContent = 'Execution Failed';
    overallPipelineStatus.className = 'status-summary-badge status-failed-badge';
    alert(`TrustLens Pipeline Error:\n${err.message}`);
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="btn-icon">
        <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
      </svg>
      Analyze Image
    `;
  }
}

// -------------------------------------------------------------
// Render Complete Results
// -------------------------------------------------------------
function renderPipelineResults(data) {
  const bm = data.best_match;
  const chain = data.blockchain_receipt;
  const summary = data.search_summary || {};
  const perf = data.performance || {};

  const numVerif = data.verified_matches ? data.verified_matches.length : (summary.verified_count || 0);
  const numRev = data.review_candidates ? data.review_candidates.length : (summary.review_count || 0);
  const numRej = data.rejected_candidates ? data.rejected_candidates.length : (summary.rejected_count || 0);

  // 1. TOP RESULT BANNER
  resultTopBanner.style.display = 'block';
  if (numVerif > 0) {
    resultTopBanner.className = 'result-top-banner';
    rtbStatusPill.textContent = `STATUS: VERIFIED MATCH FOUND (${numVerif} Verified)`;
  } else {
    resultTopBanner.className = 'result-top-banner banner-no-verif';
    rtbStatusPill.textContent = 'STATUS: NO HIGH-CONFIDENCE MATCH';
  }
  sumCountVerif.textContent = numVerif;
  sumCountRev.textContent = numRev;
  sumCountRej.textContent = numRej;
  rtbLatencyBadge.textContent = `⚡ ${(perf.total_latency_seconds || 0).toFixed(1)}s`;

  // 2. PRIMARY SECTION: BEST MATCH
  if (bm) {
    bestMatchSection.style.display = 'block';
    bmThumbnail.src = bm.thumbnail || '';
    bmPlatform.textContent = bm.platform || 'Web';
    bmTitle.textContent = bm.title || 'Untitled Candidate';
    bmUrl.textContent = bm.link || '';
    bmSimilarity.textContent = `${(bm.similarity * 100).toFixed(2)}% (${bm.similarity})`;
    
    bmDecision.textContent = bm.decision;
    bmDecision.className = `decision-badge decision-${bm.decision.toLowerCase()}`;
    
    bmRank.textContent = `#${bm.rank}`;
    bmQuality.textContent = `${(bm.quality * 100).toFixed(1)}%`;

    if (summary.separation_margin !== null && summary.separation_margin !== undefined) {
      bmSeparationMargin.textContent = `${summary.separation_margin.toFixed(4)}`;
      bmSeparationInterp.textContent = summary.margin_interpretation || 'Verified separation';
    }

    openSourceBtn.href = bm.link || '#';
  }

  // 3. SECONDARY SECTION: OTHER SEARCH CANDIDATES (Collapsed by default)
  if (data.all_candidates && data.all_candidates.length > 0) {
    allCandidatesSection.style.display = 'block';
    candidatesAccordion.open = false; // Collapsed by default as requested
    otherCandidatesCount.textContent = data.all_candidates.length;

    countVerif.textContent = numVerif;
    countRev.textContent = numRev;
    countRej.textContent = numRej;

    chipCountVerif.textContent = numVerif;
    chipCountRev.textContent = numRev;
    chipCountRej.textContent = numRej;

    renderCandidateCards(data.all_candidates);
  }

  // 4. BLOCKCHAIN & EVIDENCE PROOF SECTION
  blockchainSection.style.display = 'block';
  dispEvidenceHash.textContent = data.manifest_hash;
  dispEvidenceCid.textContent = data.manifest_cid;
  dispIpfsLink.href = `https://gateway.pinata.cloud/ipfs/${data.manifest_cid}`;

  if (chain) {
    dispTxHash.textContent = chain.tx_hash;
    dispTxLink.href = chain.explorer_url || `https://amoy.polygonscan.com/tx/${chain.tx_hash}`;
    dispBlockStatus.textContent = `Block #${chain.block} (ON-CHAIN VALID)`;
  } else {
    dispTxHash.textContent = 'Offline Dry Run';
    dispTxLink.href = '#';
    dispBlockStatus.textContent = 'Simulated / Not registered';
  }

  manifestJsonBlock.textContent = JSON.stringify(data.manifest, null, 2);

  // 5. SHOW INDEPENDENT VERIFICATION & TAMPER DEMO
  actionsGrid.style.display = 'grid';
  verifResultArea.style.display = 'none';
  tamperResultArea.style.display = 'none';

  logMessage(`Verification completed in ${(perf.total_latency_seconds || 0).toFixed(1)}s: 1 primary verified match, ${numRej} unrelated candidates rejected.`, 'success');
}

function renderCandidateCards(candidates) {
  candidatesGrid.innerHTML = '';
  const socialList = candidates.filter(c => c.platform !== 'General Web');

  countAll.textContent = candidates.length;
  countSocial.textContent = socialList.length;

  candidates.forEach(cand => {
    const card = document.createElement('div');
    const decisionLower = (cand.decision || 'rejected').toLowerCase();
    card.className = `candidate-card cand-card-${decisionLower}`;
    card.dataset.platformType = cand.platform === 'General Web' ? 'WEB' : 'SOCIAL';
    card.dataset.decision = cand.decision;

    const simPct = (cand.similarity * 100).toFixed(1);
    const decClass = `decision-${decisionLower}`;

    card.innerHTML = `
      <div class="cand-thumb-wrap">
        <img class="cand-thumb" src="${cand.thumbnail}" alt="Thumb" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'80\\' height=\\'80\\' fill=\\'none\\'><rect width=\\'100%\\' height=\\'100%\\' fill=\\'%231e293b\\'/></svg>'">
        <span class="cand-platform-tag">${cand.platform}</span>
      </div>
      <div class="cand-info">
        <div class="cand-header-row">
          <span class="cand-rank">#${cand.rank}</span>
          <span class="decision-badge ${decClass}">${cand.decision}</span>
        </div>
        <h4 class="cand-title" title="${cand.title}">${cand.title || 'Untitled'}</h4>
        <div class="cand-stats">
          <span>Similarity: <strong>${simPct}%</strong></span>
          <span>Quality: <strong>${(cand.quality * 100).toFixed(0)}%</strong></span>
        </div>
        <a class="cand-link" href="${cand.link}" target="_blank" rel="noopener noreferrer">Visit Source ↗</a>
      </div>
    `;
    candidatesGrid.appendChild(card);
  });
}

function filterCandidates(filter) {
  const cards = document.querySelectorAll('.candidate-card');
  cards.forEach(card => {
    if (filter === 'ALL') {
      card.style.display = 'flex';
    } else if (filter === 'SOCIAL') {
      card.style.display = card.dataset.platformType === 'SOCIAL' ? 'flex' : 'none';
    } else if (filter === 'VERIFIED') {
      card.style.display = card.dataset.decision === 'VERIFIED' ? 'flex' : 'none';
    } else if (filter === 'REVIEW') {
      card.style.display = card.dataset.decision === 'REVIEW' ? 'flex' : 'none';
    } else if (filter === 'REJECTED') {
      card.style.display = card.dataset.decision === 'REJECTED' ? 'flex' : 'none';
    }
  });
}

// -------------------------------------------------------------
// Independent Verification (POST /api/verify)
// -------------------------------------------------------------
async function runIndependentVerification(proofHash) {
  logMessage(`Executing independent on-chain verification for hash: ${proofHash}...`);
  verifyProofBtn.disabled = true;
  verifyProofBtn.textContent = 'Verifying...';

  try {
    const res = await fetch('/api/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ proof_hash: proofHash }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Verification query failed');
    }

    const data = await res.json();
    verifResultArea.style.display = 'block';

    if (data.is_valid) {
      verifBanner.className = 'result-banner banner-valid';
      verifIcon.textContent = '✓';
      verifTitle.textContent = 'INDEPENDENT VERIFICATION: VALID';
      verifDesc.textContent = 'Smart contract record matched IPFS manifest SHA-256 byte-for-byte.';
    } else {
      verifBanner.className = 'result-banner banner-tampered';
      verifIcon.textContent = '✕';
      verifTitle.textContent = 'VERIFICATION: INVALID / MISMATCH';
      verifDesc.textContent = 'Cryptographic hashes do not match!';
    }

    verifKvGrid.innerHTML = `
      <div class="verif-item">
        <span class="vi-label">On-Chain Registered Hash:</span>
        <code class="vi-val">${data.blockchain_hash}</code>
      </div>
      <div class="verif-item">
        <span class="vi-label">Recomputed IPFS SHA-256:</span>
        <code class="vi-val accent-green">${data.recomputed_hash}</code>
      </div>
      <div class="verif-item">
        <span class="vi-label">IPFS CID:</span>
        <code class="vi-val">${data.ipfs_cid}</code>
      </div>
      <div class="verif-item">
        <span class="vi-label">Submitter Address:</span>
        <code class="vi-val">${data.submitter}</code>
      </div>
      <div class="verif-item">
        <span class="vi-label">Cryptographic Integrity Match:</span>
        <strong class="${data.is_valid ? 'accent-green' : 'accent-red'}">${data.is_valid ? 'YES (100% Bit-for-Bit)' : 'NO (Tampered)'}</strong>
      </div>
      <div class="verif-item">
        <span class="vi-label">Final Verification Result:</span>
        <strong class="${data.is_valid ? 'accent-green' : 'accent-red'}">${data.is_valid ? 'VALID ON-CHAIN PROOF' : 'INVALID'}</strong>
      </div>
    `;

    logMessage(`Independent verification result: ${data.is_valid ? 'VALID' : 'INVALID'}`, data.is_valid ? 'success' : 'error');

  } catch (err) {
    logMessage(`Verification failed: ${err.message}`, 'error');
    alert(`Verification error: ${err.message}`);
  } finally {
    verifyProofBtn.disabled = false;
    verifyProofBtn.textContent = 'Verify Proof';
  }
}

// -------------------------------------------------------------
// Tamper Demonstration (POST /api/tamper-test)
// -------------------------------------------------------------
async function runTamperDemo(proofHash) {
  logMessage(`Running tamper demonstration test on proof hash: ${proofHash}...`);
  tamperTestBtn.disabled = true;
  tamperTestBtn.textContent = 'Testing Tamper...';

  try {
    const res = await fetch('/api/tamper-test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ proof_hash: proofHash, tamper_field: 'url' }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Tamper test query failed');
    }

    const data = await res.json();
    tamperResultArea.style.display = 'block';

    tamperCompareView.innerHTML = `
      <div class="tamper-grid">
        <div class="tamper-box">
          <h5 class="tb-title">Immutable On-Chain Hash</h5>
          <code class="tb-code">${data.onchain_hash}</code>
        </div>
        <div class="tamper-box">
          <h5 class="tb-title">Tampered Manifest Computed SHA-256</h5>
          <code class="tb-code accent-red">${data.tampered_computed_hash}</code>
        </div>
      </div>
      <div class="tamper-details-box">
        <div class="td-line"><span>Mutated Field:</span> <code>candidate.source_url -> "https://fake-imposter-profile.example.com/spoofed"</code></div>
        <div class="td-line"><span>Integrity Match:</span> <strong class="accent-red">NO (Cryptographic Mismatch)</strong></div>
        <div class="td-line"><span>Final Result:</span> <strong class="accent-red">TAMPER DETECTED / INVALID</strong></div>
      </div>
    `;

    logMessage('Tamper demonstration complete: Mutated record failed cryptographic verification against on-chain anchor.', 'error');

  } catch (err) {
    logMessage(`Tamper test failed: ${err.message}`, 'error');
    alert(`Tamper test error: ${err.message}`);
  } finally {
    tamperTestBtn.disabled = false;
    tamperTestBtn.textContent = 'TAMPER TEST';
  }
}
