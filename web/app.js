/**
 * TrustLens Interactive Web Application Client Logic
 */

document.addEventListener("DOMContentLoaded", () => {
  // State
  let currentImageBase64 = null;
  let currentImageObj = null;
  let detectedFaces = [];
  let selectedFaceIndex = 0;
  let pipelineResult = null;
  let activeFilter = "ALL";

  // Elements
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const dropzonePrompt = document.getElementById("dropzonePrompt");
  const previewContainer = document.getElementById("previewContainer");
  const faceCanvas = document.getElementById("faceCanvas");
  const previewOverlayInfo = document.getElementById("previewOverlayInfo");
  const loadDemoBtn = document.getElementById("loadDemoBtn");
  const runPipelineBtn = document.getElementById("runPipelineBtn");

  const multiFaceBar = document.getElementById("multiFaceBar");
  const faceChipsContainer = document.getElementById("faceChipsContainer");
  const qualityCard = document.getElementById("qualityCard");
  const overallQualityBadge = document.getElementById("overallQualityBadge");
  const qSharp = document.getElementById("qSharp");
  const qExpo = document.getElementById("qExpo");
  const qRes = document.getElementById("qRes");
  const qFront = document.getElementById("qFront");
  const barSharp = document.getElementById("barSharp");
  const barExpo = document.getElementById("barExpo");
  const barRes = document.getElementById("barRes");
  const barFront = document.getElementById("barFront");

  const fingerprintBox = document.getElementById("fingerprintBox");
  const queryEmbeddingHash = document.getElementById("queryEmbeddingHash");

  const resultsEmptyState = document.getElementById("resultsEmptyState");
  const pipelineLoading = document.getElementById("pipelineLoading");
  const separationBanner = document.getElementById("separationBanner");
  const sepBestVerified = document.getElementById("sepBestVerified");
  const sepBestRejected = document.getElementById("sepBestRejected");
  const sepMarginVal = document.getElementById("sepMarginVal");
  const sepInterp = document.getElementById("sepInterp");
  const candidatesContainer = document.getElementById("candidatesContainer");
  const candidatesGrid = document.getElementById("candidatesGrid");
  const platformFilterGroup = document.getElementById("platformFilterGroup");

  const proofSummaryCard = document.getElementById("proofSummaryCard");
  const manifestHashDisplay = document.getElementById("manifestHashDisplay");
  const manifestCidDisplay = document.getElementById("manifestCidDisplay");
  const txHashDisplay = document.getElementById("txHashDisplay");
  const contractDisplay = document.getElementById("contractDisplay");
  const manifestJsonCode = document.getElementById("manifestJsonCode");

  // Tab Navigation
  const navTabs = document.querySelectorAll(".nav-tab");
  const tabPanes = document.querySelectorAll(".tab-pane");

  navTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetId = tab.getAttribute("data-tab");
      navTabs.forEach(t => t.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById(targetId)?.classList.add("active");
    });
  });

  // Fetch Live Status & Update Header
  async function fetchStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      if (data.polygon_amoy) {
        document.getElementById("amoyBlock").innerText = `#${data.polygon_amoy.latest_block}`;
        const contractLink = document.getElementById("contractLink");
        const addr = data.polygon_amoy.contract_address;
        if (addr && addr.startsWith("0x")) {
          contractLink.innerText = `${addr.slice(0, 6)}...${addr.slice(-4)}`;
          contractLink.href = `https://amoy.polygonscan.com/address/${addr}`;
        }
      }
    } catch (e) {
      console.warn("Status fetch failed:", e);
    }
  }
  fetchStatus();
  setInterval(fetchStatus, 15000);

  // Drag & Drop Handling
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  });

  function handleFile(file) {
    const reader = new FileReader();
    reader.onload = (evt) => {
      currentImageBase64 = evt.target.result;
      loadImageAndAnalyze(currentImageBase64, file.name);
    };
    reader.readAsDataURL(file);
  }

  // Demo Button
  loadDemoBtn.addEventListener("click", async () => {
    try {
      loadDemoBtn.innerText = "Loading...";
      const res = await fetch("/api/demo-image");
      const data = await res.json();
      if (data.image_base64) {
        currentImageBase64 = data.image_base64;
        loadImageAndAnalyze(currentImageBase64, data.filename || "demo_face.jpg");
      }
    } catch (e) {
      alert("Failed to load demo image: " + e.message);
    } finally {
      loadDemoBtn.innerText = "Load Consenting Demo Face";
    }
  });

  // Load Image and Call Face Analysis API
  async function loadImageAndAnalyze(base64Str, filename) {
    dropzonePrompt.style.display = "none";
    previewContainer.style.display = "flex";
    previewOverlayInfo.innerText = "Analyzing facial biometrics with InsightFace (ArcFace)...";
    runPipelineBtn.disabled = true;

    const img = new Image();
    img.onload = async () => {
      currentImageObj = img;
      renderCanvasBase(img);

      try {
        const res = await fetch("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_base64: base64Str }),
        });
        const data = await res.json();

        if (data.face_detected && data.faces.length > 0) {
          detectedFaces = data.faces;
          selectedFaceIndex = 0;
          renderFaceOverlays(img, detectedFaces, selectedFaceIndex);
          updateFaceUI(detectedFaces[selectedFaceIndex]);

          if (detectedFaces.length > 1) {
            renderMultiFaceChips(detectedFaces);
          } else {
            multiFaceBar.style.display = "none";
          }
          runPipelineBtn.disabled = false;
        } else {
          previewOverlayInfo.innerText = "No face detected in image. Please choose another.";
          runPipelineBtn.disabled = true;
          multiFaceBar.style.display = "none";
          qualityCard.style.display = "none";
          fingerprintBox.style.display = "none";
        }
      } catch (err) {
        previewOverlayInfo.innerText = "Analysis error: " + err.message;
      }
    };
    img.src = base64Str;
  }

  function renderCanvasBase(img) {
    const ctx = faceCanvas.getContext("2d");
    faceCanvas.width = img.naturalWidth || img.width;
    faceCanvas.height = img.naturalHeight || img.height;
    ctx.drawImage(img, 0, 0);
  }

  function renderFaceOverlays(img, faces, activeIdx) {
    const ctx = faceCanvas.getContext("2d");
    ctx.clearRect(0, 0, faceCanvas.width, faceCanvas.height);
    ctx.drawImage(img, 0, 0);

    faces.forEach((f, idx) => {
      const bbox = f.bbox;
      const isSelected = idx === activeIdx;
      const strokeColor = isSelected ? "#06b6d4" : "rgba(255, 255, 255, 0.4)";
      const lineWidth = isSelected ? 4 : 2;

      // Draw bounding box
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = lineWidth;
      ctx.strokeRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);

      // Draw corner accents
      if (isSelected) {
        ctx.fillStyle = "#06b6d4";
        const len = 12;
        ctx.fillRect(bbox[0] - 2, bbox[1] - 2, len, 4);
        ctx.fillRect(bbox[0] - 2, bbox[1] - 2, 4, len);
      }

      // Draw landmarks if present
      if (f.landmarks) {
        f.landmarks.forEach(pt => {
          ctx.beginPath();
          ctx.arc(pt[0], pt[1], isSelected ? 3.5 : 2, 0, 2 * Math.PI);
          ctx.fillStyle = isSelected ? "#6366f1" : "rgba(255, 255, 255, 0.6)";
          ctx.fill();
        });
      }

      // Label badge
      ctx.fillStyle = isSelected ? "#06b6d4" : "rgba(15, 23, 42, 0.8)";
      ctx.fillRect(bbox[0], Math.max(0, bbox[1] - 24), 85, 20);
      ctx.fillStyle = isSelected ? "#000" : "#fff";
      ctx.font = "bold 11px Outfit, sans-serif";
      ctx.fillText(`Face #${idx} (${f.det_score})`, bbox[0] + 6, Math.max(14, bbox[1] - 10));
    });

    previewOverlayInfo.innerText = `Detected ${faces.length} face(s). Target: Face #${activeIdx} (Det: ${faces[activeIdx].det_score})`;
  }

  function renderMultiFaceChips(faces) {
    multiFaceBar.style.display = "block";
    faceChipsContainer.innerHTML = "";
    faces.forEach((f, idx) => {
      const chip = document.createElement("button");
      chip.className = `face-chip ${idx === selectedFaceIndex ? "active" : ""}`;
      chip.innerText = `Face #${idx} (Det: ${f.det_score}, Q: ${f.quality.overall_quality})`;
      chip.addEventListener("click", () => {
        selectedFaceIndex = idx;
        document.querySelectorAll(".face-chip").forEach((c, i) => c.classList.toggle("active", i === idx));
        renderFaceOverlays(currentImageObj, detectedFaces, selectedFaceIndex);
        updateFaceUI(detectedFaces[selectedFaceIndex]);
      });
      faceChipsContainer.appendChild(chip);
    });
  }

  function updateFaceUI(faceData) {
    qualityCard.style.display = "block";
    fingerprintBox.style.display = "block";

    const q = faceData.quality;
    overallQualityBadge.innerText = `Quality: ${q.overall_quality.toFixed(2)}`;
    overallQualityBadge.className = `quality-score-badge ${q.is_usable ? "accent-green" : "accent-red"}`;

    const b = q.breakdown;
    qSharp.innerText = b.sharpness.toFixed(2);
    qExpo.innerText = b.exposure.toFixed(2);
    qRes.innerText = b.resolution.toFixed(2);
    qFront.innerText = b.frontality.toFixed(2);

    barSharp.style.width = `${Math.min(100, b.sharpness * 100)}%`;
    barExpo.style.width = `${Math.min(100, b.exposure * 100)}%`;
    barRes.style.width = `${Math.min(100, b.resolution * 100)}%`;
    barFront.style.width = `${Math.min(100, b.frontality * 100)}%`;

    queryEmbeddingHash.innerText = faceData.embedding_hash;
  }

  // Execute Full Pipeline Button
  runPipelineBtn.addEventListener("click", async () => {
    if (!currentImageBase64) return;

    resultsEmptyState.style.display = "none";
    candidatesContainer.style.display = "none";
    separationBanner.style.display = "none";
    proofSummaryCard.style.display = "none";
    platformFilterGroup.style.display = "none";
    pipelineLoading.style.display = "flex";
    runPipelineBtn.disabled = true;

    try {
      const res = await fetch("/api/pipeline", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_base64: currentImageBase64,
          face_index: selectedFaceIndex,
        }),
      });

      const data = await res.json();
      if (data.error) throw new Error(data.error);

      pipelineResult = data;
      renderPipelineResults(data);
    } catch (err) {
      alert("Pipeline error: " + err.message);
    } finally {
      pipelineLoading.style.display = "none";
      runPipelineBtn.disabled = false;
    }
  });

  function renderPipelineResults(data) {
    const summary = data.search_summary;
    separationBanner.style.display = "block";
    platformFilterGroup.style.display = "flex";
    candidatesContainer.style.display = "block";
    proofSummaryCard.style.display = "block";

    // Separation Margin
    if (summary.separation_margin !== null && summary.separation_margin !== undefined) {
      sepBestVerified.innerText = summary.verified_count > 0 ? (data.best_match ? data.best_match.similarity.toFixed(4) : "0.00") : "N/A";
      const rejectedItems = data.all_candidates.filter(c => c.decision !== "VERIFIED");
      const bestRej = rejectedItems.length > 0 ? Math.max(...rejectedItems.map(r => r.similarity)) : 0.0;
      sepBestRejected.innerText = bestRej > 0 ? bestRej.toFixed(4) : "0.00";
      sepMarginVal.innerText = summary.separation_margin.toFixed(4);
      sepInterp.innerText = summary.margin_interpretation || "Clear differentiation observed";
    }

    // Render Candidates Grid
    renderCandidates(data.all_candidates);

    // Proof Summary
    manifestHashDisplay.innerText = data.manifest_hash;
    manifestCidDisplay.innerText = data.manifest_cid;
    if (data.blockchain_receipt) {
      txHashDisplay.innerText = `${data.blockchain_receipt.tx_hash} (Block #${data.blockchain_receipt.block})`;
      contractDisplay.innerText = data.blockchain_receipt.contract_address;
    } else {
      txHashDisplay.innerText = "Polygon Amoy Anchored";
      contractDisplay.innerText = "0x6D03eE0515FeeA663D6fe79F8e12dDA4C24B3c5F";
    }

    manifestJsonCode.innerText = JSON.stringify(data.manifest, null, 2);
  }

  function renderCandidates(candidates) {
    candidatesGrid.innerHTML = "";
    const filtered = activeFilter === "SOCIAL" 
      ? candidates.filter(c => c.platform !== "General Web")
      : candidates;

    if (filtered.length === 0) {
      candidatesGrid.innerHTML = `<p class="empty-subtitle" style="text-align: center; padding: 20px;">No candidates matching active filter.</p>`;
      return;
    }

    filtered.forEach((c) => {
      const card = document.createElement("div");
      card.className = `candidate-card ${c.decision === "VERIFIED" ? "verified-match" : ""}`;

      const badgeClass = c.decision === "VERIFIED" ? "badge-verified" : (c.decision === "REVIEW" ? "badge-review" : "badge-rejected");

      card.innerHTML = `
        <img class="cand-thumb" src="${c.thumbnail}" alt="Thumbnail" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'60\\' height=\\'60\\' fill=\\'%23334155\\'><rect width=\\'60\\' height=\\'60\\'/></svg>'">
        <div class="cand-info">
          <div class="cand-top">
            <span class="cand-platform">${c.platform}</span>
            <span class="cand-rank">#${c.rank}</span>
          </div>
          <p class="cand-title" title="${c.title}">${c.title || "Discovered Visual Match"}</p>
          <a class="cand-link" href="${c.link}" target="_blank" rel="noopener noreferrer">${c.link}</a>
        </div>
        <div class="cand-metrics">
          <span class="sim-badge ${badgeClass}">${c.similarity.toFixed(4)}</span>
          <span class="cand-q-score">Quality: ${c.quality.toFixed(2)}</span>
        </div>
      `;
      candidatesGrid.appendChild(card);
    });
  }

  // Filter Buttons
  document.querySelectorAll(".filter-pill").forEach(pill => {
    pill.addEventListener("click", () => {
      document.querySelectorAll(".filter-pill").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      activeFilter = pill.getAttribute("data-filter");
      if (pipelineResult && pipelineResult.all_candidates) {
        renderCandidates(pipelineResult.all_candidates);
      }
    });
  });

  // ============================================================
  // TAB 2: INDEPENDENT ON-CHAIN RE-VERIFICATION
  // ============================================================
  const verifyHashInput = document.getElementById("verifyHashInput");
  const runVerifyBtn = document.getElementById("runVerifyBtn");
  const quickProofBtn = document.getElementById("quickProofBtn");
  const verifyResultBox = document.getElementById("verifyResultBox");
  const verifyStatusBanner = document.getElementById("verifyStatusBanner");
  const verifyIcon = document.getElementById("verifyIcon");
  const verifyTitle = document.getElementById("verifyTitle");
  const verifySub = document.getElementById("verifySub");
  const verifyDetailsGrid = document.getElementById("verifyDetailsGrid");

  quickProofBtn.addEventListener("click", () => {
    verifyHashInput.value = "5eb2b2ac50d9bd30d9b08c09395e71c3e15fb716c7e51d4843f467df91d9723c";
    executeVerification();
  });

  runVerifyBtn.addEventListener("click", executeVerification);

  async function executeVerification() {
    const hash = verifyHashInput.value.trim();
    if (!hash) {
      alert("Please enter a 64-character SHA-256 evidence hash.");
      return;
    }

    runVerifyBtn.innerText = "Querying Chain...";
    runVerifyBtn.disabled = true;

    try {
      const res = await fetch(`/api/verify?hash=${encodeURIComponent(hash)}`);
      const data = await res.json();

      verifyResultBox.style.display = "block";
      if (data.is_valid) {
        verifyStatusBanner.className = "verify-status-banner";
        verifyIcon.innerText = "✓";
        verifyTitle.innerText = "CRYPTOGRAPHIC INTEGRITY CONFIRMED (VALID)";
        verifySub.innerText = "The canonical IPFS manifest recomputes to the EXACT SHA-256 hash registered on Polygon Amoy.";
      } else {
        verifyStatusBanner.className = "verify-status-banner tampered";
        verifyIcon.innerText = "✕";
        verifyTitle.innerText = "TAMPER DETECTED OR PROOF NOT FOUND";
        verifySub.innerText = data.error || "The recomputed hash does not match the on-chain anchor.";
      }

      const m = data.manifest || {};
      const cand = m.candidate || {};
      const verif = m.verification || {};

      verifyDetailsGrid.innerHTML = `
        <div class="proof-grid">
          <div class="proof-field"><span class="pf-label">Blockchain Anchor</span><code class="pf-val">${data.blockchain_hash}</code></div>
          <div class="proof-field"><span class="pf-label">Local Recomputed Hash</span><code class="pf-val highlight-purple">${data.recomputed_hash || "N/A"}</code></div>
          <div class="proof-field"><span class="pf-label">IPFS CID</span><code class="pf-val highlight-cyan">${data.ipfs_cid}</code></div>
          <div class="proof-field"><span class="pf-label">Submitter</span><code class="pf-val">${data.submitter}</code></div>
          <div class="proof-field"><span class="pf-label">Matched URL</span><code class="pf-val">${cand.source_url || "N/A"}</code></div>
          <div class="proof-field"><span class="pf-label">Face Similarity</span><code class="pf-val">${verif.face_similarity_score || "N/A"} (${verif.decision || "N/A"})</code></div>
        </div>
      `;
    } catch (err) {
      alert("Verification failed: " + err.message);
    } finally {
      runVerifyBtn.innerText = "Verify On-Chain";
      runVerifyBtn.disabled = false;
    }
  }

  // ============================================================
  // TAB 3: CRYPTOGRAPHIC TAMPER SIMULATOR
  // ============================================================
  const tamperHashInput = document.getElementById("tamperHashInput");
  const runTamperSimBtn = document.getElementById("runTamperSimBtn");
  const tamperResults = document.getElementById("tamperResults");

  runTamperSimBtn.addEventListener("click", async () => {
    const hash = tamperHashInput.value.trim();
    const selectedScenario = document.querySelector('input[name="tamperType"]:checked')?.value || "url";

    if (!hash) {
      alert("Please enter a target on-chain proof hash.");
      return;
    }

    runTamperSimBtn.innerText = "Simulating Tampering...";
    runTamperSimBtn.disabled = true;

    try {
      const res = await fetch("/api/tamper", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hash, tamper_type: selectedScenario }),
      });

      const data = await res.json();
      if (data.error) throw new Error(data.error);

      tamperResults.style.display = "block";
      tamperResults.innerHTML = `
        <div class="verify-status-banner tampered">
          <div class="vs-icon">✕</div>
          <div class="vs-text">
            <h3 class="vs-title">CRYPTOGRAPHIC TAMPER DETECTED!</h3>
            <p class="vs-sub">Due to SHA-256 avalanche sensitivity, modifying manifest metadata altered the hash from the immutable blockchain anchor.</p>
          </div>
        </div>

        <div class="tamper-compare-grid">
          <div class="tamper-box">
            <h4 class="tb-title accent-green">[1] Authentic On-Chain Proof</h4>
            <p class="pf-label">On-Chain Hash:</p>
            <code class="pf-val highlight-cyan">${data.onchain_hash}</code>
            <p class="pf-label" style="margin-top: 8px;">Manifest Content (Authentic):</p>
            <pre class="json-code" style="max-height: 200px;">${JSON.stringify(data.original_record, null, 2)}</pre>
          </div>

          <div class="tamper-box">
            <h4 class="tb-title accent-red">[2] Forged / Tampered Record</h4>
            <p class="pf-label">Forged Hash (Mismatch):</p>
            <code class="pf-val accent-red">${data.tampered_hash}</code>
            <p class="pf-label" style="margin-top: 8px;">Manifest Content (Manipulated):</p>
            <pre class="json-code" style="max-height: 200px;">${JSON.stringify(data.tampered_record, null, 2)}</pre>
          </div>
        </div>
      `;
    } catch (err) {
      alert("Tamper simulation error: " + err.message);
    } finally {
      runTamperSimBtn.innerText = "Simulate Tampering & Execute Cryptographic Verification";
      runTamperSimBtn.disabled = false;
    }
  });
});
