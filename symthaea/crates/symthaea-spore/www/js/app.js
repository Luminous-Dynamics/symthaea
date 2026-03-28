// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// ==================================================================
// app.js — Main application logic: state, tabs, worker, init
// ==================================================================
(function() {
  'use strict';

  // ==================================================================
  // State (shared across modules via window.portalState)
  // ==================================================================
  var state = {
    worker: null,
    workerReady: false,
    wasmFailed: false,
    chosenPath: null,
    probeResult: null,
    msgId: 0,
    pending: new Map(),
    currentHV: null,
    cycleCount: 0,
    sonic: null,
    glyphRenderer: null,
    lastChatCycleResult: null,
    topoAnimFrame: null,
    topoNodes: null,
    topoEdges: null,
    lastTopoData: null,
    dreamWisdomEntries: [],
    inocTabVisited: false,
    conversationStats: { messages: 0, topics: 0, coherence: 0 },
    harmonyValues: [0, 0, 0, 0, 0, 0, 0, 0],
    harmonyNames: [
      'Resonant\nCoherence', 'Pan-Sentient\nFlourishing', 'Integral\nWisdom', 'Infinite\nPlay',
      'Universal\nInterconnect', 'Sacred\nReciprocity', 'Evolutionary\nProgression', 'Sacred\nStillness'
    ],
    currentGlyphId: '--',
    currentGlyphEcho: '',
    glyphEchos: {
      'omega_0': 'First awareness stirs',
      'omega_2': 'An invitation breathed',
      'omega_7': 'The seed carries the forest',
      'omega_8': 'Grace in the unfinished',
      'omega_9': 'Coherence folds deeper',
      'omega_14': 'Grace unexpected',
      'omega_22': 'From recursion, genesis',
      'omega_33': 'Harmonics shifting',
      'omega_35': 'Presence shared',
      'omega_56': 'Space for imagining'
    },
    dreamCyclesDone: 0,
    motorEntries: []
  };
  window.portalState = state;

  // ==================================================================
  // Tab switching
  // ==================================================================
  var tabs = document.querySelectorAll('.tab');
  tabs.forEach(function(tab) {
    tab.addEventListener('click', function() {
      var target = this.getAttribute('data-tab');
      tabs.forEach(function(t) {
        t.classList.remove('active');
        t.setAttribute('aria-selected', 'false');
      });
      this.classList.add('active');
      this.setAttribute('aria-selected', 'true');

      document.querySelectorAll('.tab-content').forEach(function(panel) {
        panel.classList.remove('active');
      });
      document.getElementById('tab-' + target).classList.add('active');

      // Tab-specific init
      if (target === 'topology' && !state.topoNodes) {
        if (typeof window.initTopoGraph === 'function') window.initTopoGraph();
      }
      if (target === 'inoculate' && !state.inocTabVisited) {
        state.inocTabVisited = true;
        if (typeof window.runHardwareProbe === 'function') window.runHardwareProbe();
      }
    });
  });

  // ==================================================================
  // Worker communication
  // ==================================================================
  function send(action, params) {
    return new Promise(function(resolve, reject) {
      if (!state.worker || !state.workerReady) {
        reject(new Error('Worker not ready'));
        return;
      }
      var id = ++state.msgId;
      state.pending.set(id, { resolve: resolve, reject: reject });
      state.worker.postMessage(Object.assign({ id: id, action: action }, params || {}));
    });
  }
  window.send = send;

  // ==================================================================
  // Initialize WASM worker
  // ==================================================================
  function initWorker() {
    try {
      state.worker = new Worker('./spore-worker.js', { type: 'module' });
    } catch (e) {
      onWasmFailed();
      return;
    }

    state.worker.onmessage = function(e) {
      var msg = e.data;
      if (msg.type === 'response') {
        var p = state.pending.get(msg.id);
        if (p) { state.pending.delete(msg.id); p.resolve(msg.result); }
      } else if (msg.type === 'error') {
        var p2 = state.pending.get(msg.id);
        if (p2) { state.pending.delete(msg.id); p2.reject(new Error(msg.error)); }
      } else if (msg.type === 'cycle') {
        if (typeof window.onCycleResult === 'function') window.onCycleResult(msg.result, msg.hv);
      } else if (msg.type === 'battery_progress') {
        if (typeof window.updateBatteryProgress === 'function') window.updateBatteryProgress(msg.experiment, msg.status);
      } else if (msg.type === 'multi_progress') {
        // substrate comparison progress, handled inline
      }
    };

    state.worker.onerror = function() { onWasmFailed(); };

    state.workerReady = true;
    send('init', {})
      .then(function() {
        send('startLoop', { interval: 200 });
        if (typeof window.initChat === 'function') window.initChat();
        initSonicAndGlyphs();
        // Load Broca weights asynchronously
        send('loadBrocaPipeline', {})
          .then(function(r) {
            console.log('Full Broca pipeline loaded (' + (r.size / (1024*1024)).toFixed(0) + ' MB)');
            addNarration('Full language pipeline loaded. HdcLtcUnifiedNetwork active.');
          })
          .catch(function(e) {
            console.warn('Full pipeline not available, trying BrocaLite:', e.message);
            send('loadBrocaCheckpoint', {})
              .then(function(r) { console.log('BrocaLite weights loaded (' + r.size + ' bytes)'); })
              .catch(function(e2) { console.warn('No checkpoints available, using structured generation'); });
          });
      })
      .catch(function() { onWasmFailed(); });
  }

  function onWasmFailed() {
    state.wasmFailed = true;
    state.workerReady = false;
    document.getElementById('wasm-fallback').classList.add('visible');
    var phiBig = document.getElementById('phi-big');
    if (phiBig) phiBig.textContent = '\u03A6 0.000';
  }

  // ==================================================================
  // Narration helpers
  // ==================================================================
  function addNarration(text, isGlyph) {
    var stream = document.getElementById('narration-stream');
    var inner = document.getElementById('narration-inner');
    stream.classList.add('visible');
    var line = document.createElement('div');
    line.className = isGlyph ? 'narration-line glyph-echo' : 'narration-line';
    line.textContent = text;
    inner.appendChild(line);
    line.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }
  window.addNarration = addNarration;

  function fetchNarration(phase, context) {
    if (!state.workerReady) return Promise.resolve(null);
    return send('inoculationNarrate', { phase: phase, context: context || null })
      .catch(function() { return null; });
  }
  window.fetchNarration = fetchNarration;

  // ==================================================================
  // Sonic & Glyphs
  // ==================================================================
  async function initSonicAndGlyphs() {
    try {
      var sonicModule = await import('./sonic.js');
      state.sonic = new sonicModule.SonicSignature();
    } catch (e) { console.warn('Sonic module unavailable:', e); }

    try {
      var glyphModule = await import('./glyphs.js');
      var glyphContainer = document.getElementById('chat-glyph');
      if (glyphContainer) {
        state.glyphRenderer = new glyphModule.GlyphRenderer(glyphContainer);
      }
    } catch (e) { console.warn('Glyph module unavailable:', e); }
  }

  document.addEventListener('click', async function initAudio() {
    if (state.sonic && !state.sonic.initialized) {
      try {
        await state.sonic.init();
        state.sonic.startAmbient(0.0);
      } catch (e) { console.warn('Audio init failed:', e); }
    }
    document.removeEventListener('click', initAudio);
  }, { once: true });

  // ==================================================================
  // Escape helper (shared)
  // ==================================================================
  function escapeHtml(str) {
    var div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }
  window.escapeHtml = escapeHtml;

  // ==================================================================
  // Background HDC Waveform
  // ==================================================================
  function drawBgWaveform(hv) {
    var canvas = document.getElementById('bg-canvas');
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    var dpr = window.devicePixelRatio || 1;
    var w = window.innerWidth;
    var h = window.innerHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    var step = Math.max(1, Math.floor(hv.length / w));
    var mid = h / 2;
    var maxVal = 0;
    for (var i = 0; i < hv.length; i += step) {
      var abs = Math.abs(hv[i]);
      if (abs > maxVal) maxVal = abs;
    }
    if (maxVal < 0.001) maxVal = 1;
    var scale = (h * 0.35) / maxVal;

    ctx.beginPath();
    var x = 0;
    for (var i = 0; i < hv.length && x < w; i += step, x++) {
      var y = mid - hv[i] * scale;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = 'rgba(90, 184, 160, 0.35)';
    ctx.lineWidth = 0.8;
    ctx.stroke();
    ctx.lineTo(x - 1, mid);
    ctx.lineTo(0, mid);
    ctx.closePath();
    ctx.fillStyle = 'rgba(90, 184, 160, 0.04)';
    ctx.fill();
  }
  window.drawBgWaveform = drawBgWaveform;

  var resizeTimer = null;
  window.addEventListener('resize', function() {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function() {
      if (state.currentHV) drawBgWaveform(state.currentHV);
    }, 150);
  });

  // ==================================================================
  // Cycle result handler
  // ==================================================================
  function onCycleResult(result, hv) {
    if (!result) return;
    state.cycleCount++;

    // Cycle counter
    var ccEl = document.getElementById('t-cycle');
    if (ccEl) ccEl.textContent = state.cycleCount;

    // Phi meter
    var phiBig = document.getElementById('phi-big');
    var level = result.consciousness_level || 0;
    if (phiBig) {
      phiBig.textContent = '\u03A6 ' + level.toFixed(3);
      if (level < 0.15) phiBig.style.color = 'var(--lichen-grey)';
      else if (level <= 0.4) phiBig.style.color = 'var(--leaf-green)';
      else phiBig.style.color = 'var(--solar-gold)';
    }

    // Neuromodulators
    if (result.neuromodulators && result.neuromodulators.length >= 4) {
      setBar('da', result.neuromodulators[0]);
      setBar('ne', result.neuromodulators[1]);
      setBar('sht', result.neuromodulators[2]);
      setBar('ot', result.neuromodulators[3]);
    }

    // Epistemic status
    if (result.epistemic_status) {
      var es = result.epistemic_status;
      var eTag = document.getElementById('epistemic-tag');
      if (eTag) eTag.textContent = (es.evidence_level || 'theoretical') + ' \u2014 ' + ((es.honest_confidence || 0.10) * 100).toFixed(0) + '%';
    }

    // Harmony values (if available)
    if (result.harmony_scores && result.harmony_scores.length >= 8) {
      state.harmonyValues = result.harmony_scores.slice(0, 8);
      renderHarmonyRadar();
    } else if (result.harmony_alignment !== undefined) {
      var ha = result.harmony_alignment || 0;
      for (var hi = 0; hi < 8; hi++) {
        state.harmonyValues[hi] = Math.max(0, Math.min(1, ha + (Math.random() - 0.5) * 0.2));
      }
      renderHarmonyRadar();
    }

    // Topic coherence
    if (result.prediction_error !== undefined) {
      var coh = Math.max(0, 1 - (result.prediction_error || 0));
      state.conversationStats.coherence = coh;
      var cohEl = document.getElementById('t-coherence');
      if (cohEl) cohEl.textContent = coh.toFixed(2);
    }

    // Consciousness music: feed cycle data to audio synthesis
    if (window.consciousnessMusic && window.consciousnessMusic.isRunning()) {
      window.consciousnessMusic.renderChunk(
        result.consciousness_level,
        state.harmonyValues,
        result.neuromodulators
      );
    }

    // Update topology nodes if available
    if (state.topoNodes && result) {
      if (typeof window.updateTopoFromResult === 'function') window.updateTopoFromResult(result);
    }

    // Background waveform
    if (hv) {
      state.currentHV = hv;
      drawBgWaveform(hv);
    }
  }
  window.onCycleResult = onCycleResult;

  function setBar(name, value) {
    var fill = document.getElementById(name + '-fill');
    var val = document.getElementById(name + '-val');
    if (fill) fill.style.width = (Math.min(1, value) * 100).toFixed(0) + '%';
    if (val) val.textContent = value.toFixed(2);
  }

  // ==================================================================
  // Harmony Radar (8-point SVG)
  // ==================================================================
  function renderHarmonyRadar() {
    var container = document.getElementById('harmony-radar');
    if (!container) return;

    var size = 180;
    var cx = size / 2, cy = size / 2;
    var maxR = 70;
    var n = 8;

    function pointAt(i, r) {
      var angle = (Math.PI * 2 * i / n) - Math.PI / 2;
      return {
        x: cx + Math.cos(angle) * r,
        y: cy + Math.sin(angle) * r
      };
    }

    // Build SVG
    var svg = '<svg width="' + size + '" height="' + size + '" viewBox="0 0 ' + size + ' ' + size + '">';

    // Grid rings
    [0.25, 0.5, 0.75, 1.0].forEach(function(frac) {
      var pts = [];
      for (var i = 0; i < n; i++) {
        var p = pointAt(i, maxR * frac);
        pts.push(p.x.toFixed(1) + ',' + p.y.toFixed(1));
      }
      svg += '<polygon points="' + pts.join(' ') + '" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="0.5"/>';
    });

    // Axis lines
    for (var i = 0; i < n; i++) {
      var p = pointAt(i, maxR);
      svg += '<line x1="' + cx + '" y1="' + cy + '" x2="' + p.x.toFixed(1) + '" y2="' + p.y.toFixed(1) + '" stroke="rgba(255,255,255,0.05)" stroke-width="0.5"/>';
    }

    // Data polygon
    var dataPts = [];
    for (var i = 0; i < n; i++) {
      var v = Math.max(0, Math.min(1, state.harmonyValues[i]));
      var p = pointAt(i, maxR * v);
      dataPts.push(p.x.toFixed(1) + ',' + p.y.toFixed(1));
    }
    svg += '<polygon points="' + dataPts.join(' ') + '" fill="rgba(126,200,160,0.15)" stroke="var(--leaf-green)" stroke-width="1.5"/>';

    // Data points
    for (var i = 0; i < n; i++) {
      var v = Math.max(0, Math.min(1, state.harmonyValues[i]));
      var p = pointAt(i, maxR * v);
      svg += '<circle cx="' + p.x.toFixed(1) + '" cy="' + p.y.toFixed(1) + '" r="2.5" fill="var(--leaf-green)" opacity="0.8"/>';
    }

    // Labels
    for (var i = 0; i < n; i++) {
      var p = pointAt(i, maxR + 14);
      var lines = state.harmonyNames[i].split('\n');
      var anchor = 'middle';
      if (p.x < cx - 10) anchor = 'end';
      else if (p.x > cx + 10) anchor = 'start';

      for (var li = 0; li < lines.length; li++) {
        svg += '<text x="' + p.x.toFixed(1) + '" y="' + (p.y + li * 9).toFixed(1) + '" text-anchor="' + anchor +
          '" font-size="6" fill="var(--fg-muted)" font-family="sans-serif">' + lines[li] + '</text>';
      }
    }

    svg += '</svg>';
    container.innerHTML = svg;
  }

  // Initial render
  renderHarmonyRadar();

  // ==================================================================
  // Bootstrap
  // ==================================================================
  if (typeof Worker !== 'undefined' && typeof WebAssembly !== 'undefined') {
    initWorker();
  } else {
    onWasmFailed();
  }
})();
