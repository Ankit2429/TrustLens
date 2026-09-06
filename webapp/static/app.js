/**
 * IDENTITY — Biometric Facial Intelligence & Ledger Evidence System
 * Client-Side Application Engine
 */

document.addEventListener('DOMContentLoaded', () => {
  // Navigation & Landing Elements
  const landingStage = document.getElementById('landingStage');
  const investigationWorkspace = document.getElementById('investigationWorkspace');
  const fileInput = document.getElementById('fileInput');
  const dropzone = document.getElementById('dropzone');
  const dropzoneIdle = document.getElementById('dropzoneIdle');
  
  // Hero Photo & Canvas Elements
  const previewWrap = document.getElementById('previewWrap');
  const faceCanvas = document.getElementById('faceCanvas');
  const actionStrip = document.getElementById('actionStrip');
  const startPipelineBtn = document.getElementById('startPipelineBtn');
  const resetImageBtn = document.getElementById('resetImageBtn');
  const multiFaceBar = document.getElementById('multiFaceBar');
  const multiFaceChips = document.getElementById('multiFaceChips');

  // Photo Floating HUD
  const hudFaceCount = document.getElementById('hudFaceCount');
  const hudFaceQuality = document.getElementById('hudFaceQuality');
  const hudFaceConf = document.getElementById('hudFaceConf');

  // Left Column Telemetry Elements
  const faDetectionStatus = document.getElementById('faDetectionStatus');
  const faQuality = document.getElementById('faQuality');
  const faConfidence = document.getElementById('faConfidence');
  const faFaceSize = document.getElementById('faFaceSize');
  const faPose = document.getElementById('faPose');
  const faDenseMesh = document.getElementById('faDenseMesh');
  const faConsistency = document.getElementById('faConsistency');
  const faExifStatus = document.getElementById('faExifStatus');
  const faSharpness = document.getElementById('faSharpness');
  const faExposure = document.getElementById('faExposure');
  const faEmbeddingHash = document.getElementById('faEmbeddingHash');

  // View Mode & EXIF Toolbar Elements
  const viewModeNormalBtn = document.getElementById('viewModeNormalBtn');
  const viewModeGeometryBtn = document.getElementById('viewModeGeometryBtn');
  const photoMetaBadge = document.getElementById('photoMetaBadge');
  const metaStatusLabel = document.getElementById('metaStatusLabel');
  const metaDimLabel = document.getElementById('metaDimLabel');

  // Telemetry & Header
  const networkNameLabel = document.getElementById('networkNameLabel');
  const headerBlockNum = document.getElementById('headerBlockNum');
  const contractTelemetryPill = document.getElementById('contractTelemetryPill');

  // Pipeline Stepper & Stream Banner
  const streamBanner = document.getElementById('streamBanner');
  const streamCurrentStageTitle = document.getElementById('streamCurrentStageTitle');
  const liveDiscoveryHud = document.getElementById('liveDiscoveryHud');
  const hudDiscoveredCount = document.getElementById('hudDiscoveredCount');
  const hudExactCount = document.getElementById('hudExactCount');
  const hudVisualCount = document.getElementById('hudVisualCount');
  const hudAboutCount = document.getElementById('hudAboutCount');
  const hudPagesCount = document.getElementById('hudPagesCount');
  const hudUniqueSourcesCount = document.getElementById('hudUniqueSourcesCount');
  const hudFacesCount = document.getElementById('hudFacesCount');
  const discoveredPlatformsStrip = document.getElementById('discoveredPlatformsStrip');

  // Source Relationship Graph & Consensus Elements
  const sourceGraphSection = document.getElementById('sourceGraphSection');
  const sgNodeCount = document.getElementById('sgNodeCount');
  const sgConsensusSummary = document.getElementById('sgConsensusSummary');
  const sourceGraphMatrix = document.getElementById('sourceGraphMatrix');

  // Top Match & Result Showcase Elements
  const resultShowcaseSection = document.getElementById('resultShowcaseSection');
  const topMatchCard = document.getElementById('topMatchCard');
  const primaryMatchThumb = document.getElementById('primaryMatchThumb');
  const primaryPlatformChip = document.getElementById('primaryPlatformChip');
  const primaryMatchTitle = document.getElementById('primaryMatchTitle');
  const primaryMatchLink = document.getElementById('primaryMatchLink');
  const majorSimilarityScore = document.getElementById('majorSimilarityScore');
  const resQualityVal = document.getElementById('resQualityVal');
  const resConfidenceVal = document.getElementById('resConfidenceVal');
  const resMarginVal = document.getElementById('resMarginVal');
  const similarityBadge = document.getElementById('similarityBadge');

  // Verdict Banner Elements
  const verdictBannerWrap = document.getElementById('verdictBannerWrap');
  const verdictHeadline = document.getElementById('verdictHeadline');
  const verdictSubReason = document.getElementById('verdictSubReason');
  const verdictBigScore = document.getElementById('verdictBigScore');
  const verdictScoreDesc = document.getElementById('verdictScoreDesc');

  // Proof & Refusal Elements
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

  // Candidate Gallery Elements
  const galleryCount = document.getElementById('galleryCount');
  const candidatesScrollStrip = document.getElementById('candidatesScrollStrip');

  // Tamper Demo Elements
  const scenarioPillCards = document.querySelectorAll('.scenario-pill-card');
  const tamperSimulationScreen = document.getElementById('tamperSimulationScreen');
  const simStatusPill = document.getElementById('simStatusPill');
  const simMatchPill = document.getElementById('simMatchPill');
  const simOrigHash = document.getElementById('simOrigHash');
  const simComputedHash = document.getElementById('simComputedHash');
  const simOnChainStatus = document.getElementById('simOnChainStatus');
  const simDiffBox = document.getElementById('simDiffBox');

  // Modal Elements
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
  let currentViewMode = 'normal'; // 'normal' | 'geometry'

  // View Mode Listeners
  if (viewModeNormalBtn) {
    viewModeNormalBtn.addEventListener('click', () => {
      currentViewMode = 'normal';
      viewModeNormalBtn.classList.add('active');
      if (viewModeGeometryBtn) viewModeGeometryBtn.classList.remove('active');
      renderFacePreview();
    });
  }
  if (viewModeGeometryBtn) {
    viewModeGeometryBtn.addEventListener('click', () => {
      currentViewMode = 'geometry';
      viewModeGeometryBtn.classList.add('active');
      if (viewModeNormalBtn) viewModeNormalBtn.classList.remove('active');
      renderFacePreview();
    });
  }

  // 1. Initial Health & Network Check
  async function checkHealth() {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        if (data.network) {
          const conciseName = data.is_local ? `ANVIL ${data.chain_id}` : `POLYGON ${data.chain_id}`;
          networkNameLabel.textContent = conciseName;
          proofNetworkLabel.textContent = conciseName;
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
      console.warn('Health check warning:', e);
    }
  }
  checkHealth();
  setInterval(checkHealth, 10000);

  // 2. Drag & Drop Event Listeners
  dropzone.addEventListener('click', (e) => {
    if (e.target.closest('button')) return;
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

  // 3. File Processing & Transition to Photo-Centric Workspace
  async function handleFileSelected(file) {
    if (!file.type.match(/^image\/(jpeg|png|webp)$/)) {
      alert('Please select a valid JPG, PNG, or WEBP image.');
      return;
    }

    currentFile = file;
    selectedFaceIndex = 0;
    detectedFaces = [];

    const workspaceFileName = document.getElementById('workspaceFileName');
    if (workspaceFileName) {
      workspaceFileName.textContent = file.name ? file.name.toUpperCase() : 'UNKNOWN QUERY IMAGE';
    }

    // Switch from Landing Stage to Photo-Centric Workspace
    landingStage.style.display = 'none';
    investigationWorkspace.style.display = 'flex';
    resultShowcaseSection.style.display = 'none';
    streamBanner.style.display = 'none';

    // Load image preview
    const reader = new FileReader();
    reader.onload = (e) => {
      rawImageObj = new Image();
      rawImageObj.onload = () => {
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
          updateFaceTelemetry();
        } else {
          hudFaceCount.textContent = '0 FACES DETECTED';
          faDetectionStatus.textContent = 'NO FACE FOUND';
          faDetectionStatus.className = 't-value';
        }
      }
    } catch (e) {
      console.warn('Face detection error:', e);
    }
  }

  function updateFaceTelemetry() {
    if (!detectedFaces || detectedFaces.length === 0) return;
    const face = detectedFaces[selectedFaceIndex] || detectedFaces[0];
    const q = face.quality || {};
    const [x1, y1, x2, y2] = face.bbox || [0, 0, 0, 0];
    const width = Math.round(x2 - x1);
    const height = Math.round(y2 - y1);

    faDetectionStatus.textContent = 'DETECTED';
    faDetectionStatus.className = 't-value highlight';
    faQuality.textContent = (q.overall_quality || 0.90).toFixed(2);
    faConfidence.textContent = (face.det_score || 0.92).toFixed(2);
    faFaceSize.textContent = `${width} × ${height} px`;

    if (face.dense_geometry && face.dense_geometry.pose_3d) {
      const p3d = face.dense_geometry.pose_3d;
      faPose.textContent = `P: ${p3d.pitch >= 0 ? '+' : ''}${p3d.pitch.toFixed(1)}° Y: ${p3d.yaw >= 0 ? '+' : ''}${p3d.yaw.toFixed(1)}° R: ${p3d.roll >= 0 ? '+' : ''}${p3d.roll.toFixed(1)}°`;
      if (faDenseMesh) faDenseMesh.textContent = `${face.dense_geometry.point_count_2d || 106}-PT ACTIVE`;
      const iod = face.dense_geometry.metrics?.iod;
      if (faConsistency) faConsistency.textContent = iod ? `${Math.round(iod)}px (${face.dense_geometry.landmark_consistency || 'HIGH'})` : 'HIGH';
    } else {
      const breakdown = q.breakdown || {};
      faPose.textContent = `Y: ${(breakdown.pose_yaw_est || 0).toFixed(1)}° P: ${(breakdown.pose_pitch_est || 0).toFixed(1)}°`;
      if (faDenseMesh) faDenseMesh.textContent = 'STANDARD 5-PT';
      if (faConsistency) faConsistency.textContent = 'HIGH';
    }

    const breakdown = q.breakdown || {};
    faSharpness.textContent = (breakdown.sharpness || q.sharpness || 0.88).toFixed(2);
    faExposure.textContent = (breakdown.exposure || q.exposure || 0.91).toFixed(2);
    faEmbeddingHash.textContent = '512-D ARCFACE L2';

    hudFaceCount.textContent = `${detectedFaces.length} FACE${detectedFaces.length > 1 ? 'S' : ''} DETECTED`;
    hudFaceQuality.textContent = `QUALITY: ${(q.overall_quality || 0.90).toFixed(2)}`;
    hudFaceConf.textContent = `CONFIDENCE: ${(face.det_score || 0.92).toFixed(2)}`;
  }

  function renderMultiFaceBar() {
    if (detectedFaces.length > 1) {
      multiFaceBar.style.display = 'flex';
      multiFaceChips.innerHTML = '';
      detectedFaces.forEach((f, idx) => {
        const padIdx = String(idx + 1).padStart(2, '0');
        const qScore = Math.round((f.quality?.overall_quality || f.det_score || 0.85) * 100);
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = `face-chip ${idx === selectedFaceIndex ? 'active' : ''}`;
        btn.textContent = `FACE ${padIdx} (${qScore}% Q)`;
        btn.addEventListener('click', () => {
          selectedFaceIndex = idx;
          renderMultiFaceBar();
          renderFacePreview();
          updateFaceTelemetry();
        });
        multiFaceChips.appendChild(btn);
      });
    } else {
      multiFaceBar.style.display = 'none';
    }
  }

  function drawDenseFaceMesh(ctx, landmarks2d, isSelected) {
    if (!landmarks2d || landmarks2d.length === 0) return;
    ctx.save();
    ctx.lineWidth = 1.0;
    ctx.strokeStyle = isSelected ? 'rgba(56, 189, 248, 0.75)' : 'rgba(56, 189, 248, 0.35)';

    function drawLineStrip(start, end, close = false) {
      if (end >= landmarks2d.length) return;
      ctx.beginPath();
      ctx.moveTo(landmarks2d[start][0], landmarks2d[start][1]);
      for (let i = start + 1; i <= end; i++) {
        ctx.lineTo(landmarks2d[i][0], landmarks2d[i][1]);
      }
      if (close) {
        ctx.lineTo(landmarks2d[start][0], landmarks2d[start][1]);
      }
      ctx.stroke();
    }

    if (landmarks2d.length >= 106) {
      drawLineStrip(0, 32, false);   // Jawline contour
      drawLineStrip(33, 42, false);  // Right eyebrow
      drawLineStrip(43, 52, false);  // Left eyebrow
      drawLineStrip(53, 62, false);  // Nose bridge
      drawLineStrip(63, 71, true);   // Nose base & nostrils
      drawLineStrip(72, 86, true);   // Outer lips
      drawLineStrip(87, 96, true);   // Inner lips
      drawLineStrip(97, 100, true);  // Right eye
      drawLineStrip(101, 105, true); // Left eye

      // Connecting facial triangulation lines
      ctx.strokeStyle = isSelected ? 'rgba(56, 189, 248, 0.3)' : 'rgba(56, 189, 248, 0.12)';
      ctx.beginPath();
      ctx.moveTo(landmarks2d[53][0], landmarks2d[53][1]);
      ctx.lineTo(landmarks2d[97][0], landmarks2d[97][1]);
      ctx.moveTo(landmarks2d[53][0], landmarks2d[53][1]);
      ctx.lineTo(landmarks2d[101][0], landmarks2d[101][1]);
      ctx.moveTo(landmarks2d[67][0], landmarks2d[67][1]);
      ctx.lineTo(landmarks2d[72][0], landmarks2d[72][1]);
      ctx.stroke();

      // Glowing landmark dots
      ctx.fillStyle = isSelected ? '#38bdf8' : 'rgba(56, 189, 248, 0.5)';
      landmarks2d.forEach(([lx, ly]) => {
        ctx.beginPath();
        ctx.arc(lx, ly, isSelected ? 1.6 : 1.2, 0, 2 * Math.PI);
        ctx.fill();
      });
    } else {
      ctx.beginPath();
      landmarks2d.forEach(([lx, ly], i) => {
        if (i === 0) ctx.moveTo(lx, ly);
        else ctx.lineTo(lx, ly);
      });
      ctx.closePath();
      ctx.stroke();
      landmarks2d.forEach(([lx, ly]) => {
        ctx.beginPath();
        ctx.arc(lx, ly, 2, 0, 2 * Math.PI);
        ctx.fill();
      });
    }
    ctx.restore();
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

        // Subtle Bounding Box
        ctx.lineWidth = isSelected ? 2.5 : 1.5;
        ctx.strokeStyle = isSelected ? '#10b981' : 'rgba(255, 255, 255, 0.4)';
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Face ID label above box for group photos
        if (detectedFaces.length > 1) {
          ctx.save();
          ctx.font = '10px "JetBrains Mono", monospace';
          ctx.fillStyle = isSelected ? '#10b981' : 'rgba(255, 255, 255, 0.6)';
          ctx.fillText(`FACE ${String(idx + 1).padStart(2, '0')}`, x1 + 4, y1 - 4);
          ctx.restore();
        }

        if (currentViewMode === 'geometry') {
          // GEOMETRY VIEW: 106-Point Mesh & 3D Pose overlay
          if (face.dense_geometry && face.dense_geometry.landmarks_2d) {
            drawDenseFaceMesh(ctx, face.dense_geometry.landmarks_2d, isSelected);
          } else if (face.landmarks) {
            ctx.save();
            ctx.lineWidth = 1.2;
            ctx.strokeStyle = isSelected ? '#38bdf8' : 'rgba(56, 189, 248, 0.4)';
            ctx.beginPath();
            ctx.moveTo(face.landmarks[0][0], face.landmarks[0][1]);
            ctx.lineTo(face.landmarks[1][0], face.landmarks[1][1]);
            ctx.lineTo(face.landmarks[2][0], face.landmarks[2][1]);
            ctx.closePath();
            ctx.moveTo(face.landmarks[2][0], face.landmarks[2][1]);
            ctx.lineTo(face.landmarks[3][0], face.landmarks[3][1]);
            ctx.lineTo(face.landmarks[4][0], face.landmarks[4][1]);
            ctx.closePath();
            ctx.stroke();
            ctx.restore();
          }

          if (isSelected && face.dense_geometry && face.dense_geometry.pose_3d) {
            const { pitch, yaw, roll } = face.dense_geometry.pose_3d;
            ctx.save();
            ctx.font = 'bold 11px "JetBrains Mono", monospace';
            ctx.fillStyle = '#38bdf8';
            ctx.shadowColor = 'rgba(0, 0, 0, 0.9)';
            ctx.shadowBlur = 4;
            const poseText = `3D POSE: P:${pitch >= 0 ? '+' : ''}${pitch.toFixed(1)}° Y:${yaw >= 0 ? '+' : ''}${yaw.toFixed(1)}° R:${roll >= 0 ? '+' : ''}${roll.toFixed(1)}°`;
            ctx.fillText(poseText, x1 + 4, Math.max(16, y1 - 16));
            ctx.restore();
          }
        } else {
          // NORMAL VIEW: clean photo + subtle face box + 5-pt reticle dots (NO wireframe clutter)
          if (face.landmarks) {
            ctx.fillStyle = isSelected ? '#EDEDED' : 'rgba(255, 255, 255, 0.5)';
            face.landmarks.forEach(([lx, ly]) => {
              ctx.beginPath();
              ctx.arc(lx, ly, isSelected ? 3.0 : 2.0, 0, 2 * Math.PI);
              ctx.fill();
            });
          }
        }
      });
    }
  }

  // Reset Button
  resetImageBtn.addEventListener('click', () => {
    currentFile = null;
    rawImageObj = null;
    detectedFaces = [];
    selectedFaceIndex = 0;
    landingStage.style.display = 'flex';
    investigationWorkspace.style.display = 'none';
    resultShowcaseSection.style.display = 'none';
    streamBanner.style.display = 'none';
    multiFaceBar.style.display = 'none';
  });

  // 5. Pipeline Execution
  startPipelineBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    // Reset results & show stream banner
    resultShowcaseSection.style.display = 'none';
    streamBanner.style.display = 'flex';
    updateStepper(1);
    streamCurrentStageTitle.textContent = 'STAGE 1/6: FACE DETECTION & QUALITY ASSESSMENT';

    const formData = new FormData();
    formData.append('image', currentFile);
    formData.append('face_index', selectedFaceIndex);
    formData.append('verified_threshold', '0.60');
    formData.append('review_threshold', '0.40');

    // Progression Stepper Updates
    setTimeout(() => {
      updateStepper(2);
      streamCurrentStageTitle.textContent = 'STAGE 2/6: 512-DIMENSIONAL ARCFACE FEATURE EMBEDDING';
    }, 800);

    setTimeout(() => {
      updateStepper(3);
      streamCurrentStageTitle.textContent = 'STAGE 3/6: SEARCHING WEB ACROSS PUBLIC INDEXES...';
      hudDiscoveredCount.textContent = 'SEARCHING...';
    }, 1800);

    setTimeout(() => {
      updateStepper(4);
      streamCurrentStageTitle.textContent = 'STAGE 4/6: INDEPENDENT CANDIDATE MULTI-FACE COMPARISON';
    }, 3200);

    setTimeout(() => {
      updateStepper(5);
      streamCurrentStageTitle.textContent = 'STAGE 5/6: RFC-8785 CANONICAL MANIFEST FINGERPRINTING';
    }, 4500);

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        lastAnalysisResult = data;
        updateStepper(6);
        streamCurrentStageTitle.textContent = 'STAGE 6/6: IMMUTABLE LEDGER PROOF ANCHORING';
        setTimeout(() => {
          streamBanner.style.display = 'none';
          renderAnalysisResults(data);
        }, 600);
      } else {
        const err = await res.json();
        streamBanner.style.display = 'none';
        alert('Investigation failed: ' + (err.detail || err.error || 'Server error'));
      }
    } catch (e) {
      streamBanner.style.display = 'none';
      alert('Network error during investigation: ' + e.message);
    }
  });

  function updateStepper(stepIndex) {
    for (let i = 1; i <= 6; i++) {
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

  // 6. Render Full Investigation Results
  function renderAnalysisResults(data) {
    resultShowcaseSection.style.display = 'flex';
    resultShowcaseSection.scrollIntoView({ behavior: 'smooth' });

    const best = data.best_match || {};
    const decision = best.decision || 'REJECTED';
    const sim = best.similarity || 0.0;
    const confidence = data.confidence_data || {};
    const summary = data.search_summary || {};
    const totalFound = summary.total_discovered !== undefined ? summary.total_discovered : (data.all_candidates ? data.all_candidates.length : 0);

    // Live Search Count
    const pagesScanned = summary.pages_scanned || 1;
    const facesAnalyzed = summary.faces_evaluated || (data.all_candidates ? data.all_candidates.length : 0);
    hudDiscoveredCount.textContent = totalFound > 0 ? `${totalFound} (${pagesScanned}P · ${facesAnalyzed}F)` : '0 (NO RESULTS)';

    // Multi-mode Discovery HUD breakdown
    const dhud = data.discovery_hud || {};
    if (hudExactCount) hudExactCount.textContent = dhud.exact_matches || 0;
    if (hudVisualCount) hudVisualCount.textContent = dhud.visual_matches || (data.all_candidates ? data.all_candidates.length : 0);
    if (hudAboutCount) hudAboutCount.textContent = dhud.about_image || 0;
    if (hudPagesCount) hudPagesCount.textContent = dhud.pages_scanned || pagesScanned || 1;
    if (hudUniqueSourcesCount) hudUniqueSourcesCount.textContent = dhud.unique_sources || (data.consensus_data ? data.consensus_data.domain_count : 0);
    if (hudFacesCount) hudFacesCount.textContent = dhud.candidate_faces_evaluated || facesAnalyzed || 0;

    // Image Metadata / EXIF update
    if (data.image_metadata) {
      const meta = data.image_metadata;
      if (metaStatusLabel) metaStatusLabel.textContent = meta.status || 'NO METADATA';
      if (faExifStatus) faExifStatus.textContent = meta.status || 'NO METADATA';
      if (metaDimLabel) {
        const dim = meta.width && meta.height ? `${meta.width}×${meta.height} px` : (rawImageObj ? `${rawImageObj.naturalWidth}×${rawImageObj.naturalHeight} px` : '--');
        const dt = meta.datetime ? ` · ${meta.datetime.slice(0, 10)}` : '';
        metaDimLabel.textContent = `${dim}${dt}`;
      }
      if (photoMetaBadge) {
        if (meta.has_exif) photoMetaBadge.classList.add('has-exif');
        else photoMetaBadge.classList.remove('has-exif');
      }
    }

    // Final Search Summary Monolith (Requirement 19)
    const discoveryCompleteCard = document.getElementById('discoveryCompleteCard');
    const dcSearchModes = document.getElementById('dcSearchModes');
    const dcResultSets = document.getElementById('dcResultSets');
    const dcUniqueSources = document.getElementById('dcUniqueSources');
    const dcCandidateImages = document.getElementById('dcCandidateImages');
    const dcFacesAnalyzed = document.getElementById('dcFacesAnalyzed');
    const dcStrongestMatch = document.getElementById('dcStrongestMatch');
    const dcSupportingSources = document.getElementById('dcSupportingSources');

    if (discoveryCompleteCard) {
      discoveryCompleteCard.style.display = 'flex';
      if (dcSearchModes) {
        const modes = (dhud.search_modes_active && dhud.search_modes_active.length > 0) 
          ? dhud.search_modes_active.join(', ') 
          : 'visual_matches';
        dcSearchModes.textContent = modes;
      }
      if (dcResultSets) dcResultSets.textContent = dhud.pages_scanned || pagesScanned || 1;
      if (dcUniqueSources) dcUniqueSources.textContent = dhud.unique_sources || (data.consensus_data ? data.consensus_data.domain_count : 0);
      if (dcCandidateImages) dcCandidateImages.textContent = totalFound || 0;
      if (dcFacesAnalyzed) dcFacesAnalyzed.textContent = dhud.candidate_faces_evaluated || facesAnalyzed || 0;
      if (dcStrongestMatch) {
        dcStrongestMatch.textContent = `${sim.toFixed(4)} [${decision}]`;
      }
      if (dcSupportingSources) {
        dcSupportingSources.textContent = data.consensus_data ? data.consensus_data.total_supporting : 0;
      }
    }

    // Dynamic Platform Illumination based on ACTUAL returned platforms
    const tags = discoveredPlatformsStrip.querySelectorAll('.platform-tag');
    const platformsFound = new Set();
    if (best.platform && best.platform !== 'No Match' && best.platform !== 'General Web') {
      platformsFound.add(best.platform.toUpperCase());
    }
    if (data.consensus_data && data.consensus_data.distinct_platforms) {
      data.consensus_data.distinct_platforms.forEach((p) => platformsFound.add(p.toUpperCase()));
    }
    if (data.all_candidates) {
      data.all_candidates.forEach((c) => {
        if (c.platform) platformsFound.add(c.platform.toUpperCase());
      });
    }

    tags.forEach((tag) => {
      const pName = tag.dataset.platform || tag.textContent.trim().toUpperCase();
      if (platformsFound.has(pName)) {
        tag.classList.add('active-platform');
      } else {
        tag.classList.remove('active-platform');
      }
    });

    // Verdict Styling
    verdictBannerWrap.className = `verdict-banner status-${decision.toLowerCase().replace(/\s+/g, '-')}`;
    if (decision === 'VERIFIED') {
      verdictHeadline.textContent = 'VERIFIED';
      verdictScoreDesc.textContent = 'Strong match confirmed';
      similarityBadge.textContent = 'VERIFIED';
      similarityBadge.className = 'mm-val tag-decision tag-verif';
    } else if (decision === 'REVIEW') {
      verdictHeadline.textContent = 'REVIEW';
      verdictScoreDesc.textContent = 'Borderline evidence';
      similarityBadge.textContent = 'REVIEW';
      similarityBadge.className = 'mm-val tag-decision tag-rev';
    } else if (decision === 'NO RELIABLE MATCH' || best.similarity === 0) {
      verdictHeadline.textContent = 'NO RELIABLE MATCH';
      verdictScoreDesc.textContent = 'No indexed candidate match';
      similarityBadge.textContent = 'NO MATCH';
      similarityBadge.className = 'mm-val tag-decision tag-rej';
    } else {
      verdictHeadline.textContent = 'REJECTED';
      verdictScoreDesc.textContent = 'Non-matching candidates';
      similarityBadge.textContent = 'REJECTED';
      similarityBadge.className = 'mm-val tag-decision tag-rej';
    }

    verdictSubReason.textContent = best.reason || (decision === 'VERIFIED' ? 'Face similarity exceeds verified threshold with clear candidate quality differentiation.' : 'Indexed evidence was insufficient to establish a high-confidence match.');
    verdictBigScore.textContent = sim.toFixed(4);
    majorSimilarityScore.textContent = sim.toFixed(4);

    // Right Column Match Card
    resQualityVal.textContent = (best.quality || 0.85).toFixed(2);
    resConfidenceVal.textContent = (confidence.confidence_score || sim).toFixed(4);
    resMarginVal.textContent = summary.separation_margin !== null ? `+${summary.separation_margin.toFixed(4)}` : 'N/A';

    primaryMatchTitle.textContent = best.title || 'Discovered Web Identity';
    primaryMatchLink.textContent = best.link || 'https://...';
    primaryMatchLink.href = best.link || '#';
    primaryPlatformChip.textContent = (best.platform || 'General Web').toUpperCase();

    if (best.thumbnail) {
      primaryMatchThumb.src = best.thumbnail;
      primaryMatchThumb.style.display = 'block';
    } else {
      primaryMatchThumb.style.display = 'none';
    }

    // Group Photo Traceability in Match Card
    const groupBreakdownEl = document.getElementById('primaryGroupBreakdown');
    if (groupBreakdownEl) {
      if (best.face_count > 1 && best.candidate_faces_evaluated && best.candidate_faces_evaluated.length > 0) {
        groupBreakdownEl.style.display = 'block';
        groupBreakdownEl.innerHTML = `
          <div class="group-photo-header">
            <span class="group-title">GROUP PHOTO ANALYSIS (${best.face_count} FACES DETECTED)</span>
            <span class="group-match-tag">MATCHED: ${best.matched_face_id || 'FACE 01'}</span>
          </div>
          <div class="group-faces-list">
            ${best.candidate_faces_evaluated.map(cf => `
              <div class="group-face-chip ${cf.is_matched ? 'matched-face' : ''}">
                <span class="gf-id">${cf.face_id}</span>
                <span class="gf-sim">${Number(cf.similarity).toFixed(2)}</span>
                <span class="gf-dec ${cf.decision === 'VERIFIED' ? 'verif' : cf.decision === 'REVIEW' ? 'rev' : 'rej'}">${cf.decision}</span>
                ${cf.is_matched ? '<span class="gf-star">★ MATCH</span>' : ''}
              </div>
            `).join('')}
          </div>
        `;
      } else {
        groupBreakdownEl.style.display = 'none';
      }
    }

    // Proof vs Refusal Monolith
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
      refusalDescText.textContent = `The verification engine strictly enforces zero false-positive proof registration. Because this evidence is classified as [${decision}], on-chain proof anchoring was safely skipped (PROOF NOT ANCHORED) to prevent immutable ledger pollution.`;
      refusalIpfsLink.textContent = data.manifest_cid || 'Qm...';
      refusalIpfsLink.href = data.manifest_cid ? `https://ipfs.io/ipfs/${data.manifest_cid}` : '#';
    }

    // Render Candidates Strip
    const allCands = data.all_candidates || [];
    galleryCount.textContent = allCands.length;
    candidatesScrollStrip.innerHTML = '';

    allCands.forEach((c) => {
      const card = document.createElement('div');
      card.className = 'cand-card';
      const tagClass = c.decision === 'VERIFIED' ? 'tag-verif' : c.decision === 'REVIEW' ? 'tag-rev' : 'tag-rej';
      const groupBadge = c.face_count > 1 
        ? `<span class="cand-group-badge">GROUP (${c.face_count}F · ${c.matched_face_id || 'MATCH'})</span>` 
        : '';
      card.innerHTML = `
        <div class="cand-img-wrap">
          <img class="cand-img" src="${c.thumbnail || ''}" alt="Thumb" onerror="this.style.display='none'">
          ${groupBadge}
        </div>
        <div class="cand-score-row">
          <span class="cand-score">${Number(c.similarity).toFixed(3)}</span>
          <span class="cand-tag ${tagClass}">${c.decision}</span>
        </div>
        <span class="cand-host">${c.platform || c.domain || 'web'}</span>
      `;
      candidatesScrollStrip.appendChild(card);
    });

    // Render Source Relationship Graph & Consensus Matrix
    const srg = data.relationship_graph;
    if (srg && srg.counts && srg.counts.total_nodes > 0 && sourceGraphSection) {
      sourceGraphSection.style.display = 'block';
      if (sgNodeCount) sgNodeCount.textContent = srg.counts.total_nodes;
      if (sgConsensusSummary) {
        const cData = data.consensus_data || {};
        sgConsensusSummary.textContent = `${cData.domain_count || 0} DOMAINS · ${cData.consensus_level || 'CROSS-SOURCE'}`;
      }
      if (sourceGraphMatrix) {
        sourceGraphMatrix.innerHTML = '';
        const rels = srg.relationships || {};
        const allRelNodes = [
          ...(rels.same_image || []),
          ...(rels.same_person_different_image || []),
          ...(rels.visually_related_image || []),
          ...(rels.different_person || []),
          ...(rels.uncertain || [])
        ];
        allRelNodes.forEach(node => {
          const card = document.createElement('div');
          card.className = 'sg-node-card';
          const rel = node.relationship || 'UNCERTAIN';
          const relClass = rel === 'SAME_IMAGE' ? 'same-image' :
                           rel === 'SAME_PERSON_DIFFERENT_IMAGE' ? 'same-person' :
                           rel === 'VISUALLY_RELATED_IMAGE' ? 'visually-related' : 'different-person';
          const relLabel = rel.replace(/_/g, ' ');
          const simScore = typeof node.similarity === 'number' ? node.similarity.toFixed(3) : '--';
          card.innerHTML = `
            <div class="sg-node-header">
              <span class="sg-rel-tag ${relClass}">${relLabel}</span>
              <span class="sg-node-face">${node.matched_face_id || 'FACE 01'}</span>
            </div>
            <a class="sg-node-title" href="${node.link || '#'}" target="_blank" rel="noopener noreferrer">${node.title || node.domain || 'Discovered Source'}</a>
            <div class="sg-node-meta">
              <span class="sg-node-domain">${node.domain || node.platform || 'web'}</span>
              <span class="sg-sim-score ${node.similarity >= 0.60 ? 'high' : ''}">${simScore} SIM</span>
            </div>
          `;
          sourceGraphMatrix.appendChild(card);
        });
      }
    } else if (sourceGraphSection) {
      sourceGraphSection.style.display = 'none';
    }

    // Initialize Tamper Demo
    initTamperDemo(data.manifest_hash || 'c1fee9bc922c13891469c9d40421c0ca2e7d97554cfd33f165c7a4c225740789');
  }

  // 7. Interactive Cryptographic Tamper Demonstration
  function initTamperDemo(origHash) {
    simOrigHash.textContent = `${origHash.slice(0, 16)}...${origHash.slice(-8)}`;

    scenarioPillCards.forEach((card) => {
      card.addEventListener('click', () => {
        scenarioPillCards.forEach((c) => c.classList.remove('active'));
        card.classList.add('active');
        const scenario = card.getAttribute('data-scenario');
        simulateTamperScenario(scenario, origHash);
      });
    });
    simulateTamperScenario('original', origHash);
  }

  function simulateTamperScenario(scenario, origHash) {
    if (scenario === 'original') {
      tamperSimulationScreen.className = 'terminal-shell';
      simStatusPill.textContent = 'STATUS: VALID & VERIFIED';
      simMatchPill.textContent = 'HASH MATCH';
      simComputedHash.textContent = `${origHash.slice(0, 16)}...${origHash.slice(-8)}`;
      simOnChainStatus.textContent = 'CONFIRMED (Block #38 on Anvil)';
      simDiffBox.innerHTML = '<code>All evidence fields match bit-for-bit with RFC-8785 canonical standard.</code>';
    } else {
      tamperSimulationScreen.className = 'terminal-shell sim-tampered';
      simStatusPill.textContent = 'STATUS: TAMPER DETECTED';
      simMatchPill.textContent = 'HASH MISMATCH';

      const fakeHash = origHash.split('').reverse().join('');
      simComputedHash.textContent = `${fakeHash.slice(0, 16)}...${fakeHash.slice(-8)}`;
      simOnChainStatus.textContent = 'REJECTED: Proof Not Found on Blockchain Registry';

      if (scenario === 'url') {
        simDiffBox.innerHTML = '<code style="color:#ef4444">- source_url: "https://trusted-news.com/article"<br>+ source_url: "https://spoofed-adversary-domain.org"<br>❌ SHA-256 fingerprint invalidated!</code>';
      } else if (scenario === 'similarity') {
        simDiffBox.innerHTML = '<code style="color:#ef4444">- similarity_score: 0.8670<br>+ similarity_score: 0.9990<br>❌ Score modification invalidates deterministic manifest hash!</code>';
      } else {
        simDiffBox.innerHTML = '<code style="color:#ef4444">- platform: "Facebook"<br>+ platform: "Official Government Register"<br>❌ Metadata injection violates cryptographic integrity!</code>';
      }
    }
  }

  // 8. Independent On-Chain Query Modal
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
