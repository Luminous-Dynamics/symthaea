// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// ==================================================================
// tab-topology.js — Tab 2: Force-directed graph, Betti numbers
// ==================================================================
(function() {
  'use strict';
  var state = window.portalState;
  var TOPO_NODE_NAMES = ['Phi', 'Binding', 'Workspace', 'Attention', 'Recurrence', 'Embodiment', 'Knowledge'];

  function initTopoGraph() {
    state.topoNodes = TOPO_NODE_NAMES.map(function(name, i) {
      var angle = (Math.PI * 2 * i) / TOPO_NODE_NAMES.length;
      return {
        name: name,
        x: 300 + Math.cos(angle) * 120,
        y: 210 + Math.sin(angle) * 100,
        vx: 0, vy: 0,
        value: 0.3 + Math.random() * 0.3
      };
    });

    // All-pairs edges
    state.topoEdges = [];
    for (var i = 0; i < state.topoNodes.length; i++) {
      for (var j = i + 1; j < state.topoNodes.length; j++) {
        state.topoEdges.push({ a: i, b: j, strength: 0.1 + Math.random() * 0.3 });
      }
    }

    startTopoAnimation();

    // Auto-run analysis if engine is ready
    if (state.workerReady) {
      runTopoAnalysis();
    }
  }
  window.initTopoGraph = initTopoGraph;

  function updateTopoFromResult(result) {
    if (!state.topoNodes) return;
    var cl = result.consciousness_level || 0;
    var pe = result.prediction_error || 0;
    var ha = result.harmony_alignment || 0;

    state.topoNodes[0].value = cl; // Phi
    state.topoNodes[1].value = Math.max(0.1, cl * 0.8 + Math.random() * 0.1); // Binding
    state.topoNodes[2].value = Math.max(0.1, cl * 0.7 + pe * 0.2); // Workspace
    state.topoNodes[3].value = Math.max(0.1, (1 - pe) * 0.6 + 0.2); // Attention
    state.topoNodes[4].value = Math.max(0.1, ha * 0.5 + 0.3); // Recurrence
    state.topoNodes[5].value = Math.max(0.1, cl * 0.4 + 0.1); // Embodiment
    state.topoNodes[6].value = Math.max(0.1, ha * 0.6 + 0.2); // Knowledge

    // Update edge strengths
    for (var e = 0; e < state.topoEdges.length; e++) {
      var edge = state.topoEdges[e];
      edge.strength = (state.topoNodes[edge.a].value + state.topoNodes[edge.b].value) / 4;
    }
  }
  window.updateTopoFromResult = updateTopoFromResult;

  function startTopoAnimation() {
    var canvas = document.getElementById('topo-canvas');
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    var time = 0;

    function frame() {
      state.topoAnimFrame = requestAnimationFrame(frame);
      time += 0.016;

      var dpr = window.devicePixelRatio || 1;
      var rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
      var W = rect.width, H = rect.height;
      ctx.clearRect(0, 0, W, H);

      if (!state.topoNodes) return;

      // Force simulation
      var dt = 0.3;
      var repulsion = 5000;
      var springK = 0.02;
      var damping = 0.92;
      var centerX = W / 2, centerY = H / 2;

      // Coulomb repulsion
      for (var i = 0; i < state.topoNodes.length; i++) {
        for (var j = i + 1; j < state.topoNodes.length; j++) {
          var dx = state.topoNodes[j].x - state.topoNodes[i].x;
          var dy = state.topoNodes[j].y - state.topoNodes[i].y;
          var dist = Math.sqrt(dx * dx + dy * dy) || 1;
          var force = repulsion / (dist * dist);
          var fx = (dx / dist) * force;
          var fy = (dy / dist) * force;
          state.topoNodes[i].vx -= fx * dt;
          state.topoNodes[i].vy -= fy * dt;
          state.topoNodes[j].vx += fx * dt;
          state.topoNodes[j].vy += fy * dt;
        }
      }

      // Spring attraction (Hooke)
      for (var e = 0; e < state.topoEdges.length; e++) {
        var edge = state.topoEdges[e];
        var na = state.topoNodes[edge.a], nb = state.topoNodes[edge.b];
        var edx = nb.x - na.x;
        var edy = nb.y - na.y;
        var edist = Math.sqrt(edx * edx + edy * edy) || 1;
        var restLen = 100;
        var eforce = springK * (edist - restLen) * edge.strength;
        var efx = (edx / edist) * eforce;
        var efy = (edy / edist) * eforce;
        na.vx += efx * dt;
        na.vy += efy * dt;
        nb.vx -= efx * dt;
        nb.vy -= efy * dt;
      }

      // Center gravity + damping + bounds
      for (var k = 0; k < state.topoNodes.length; k++) {
        var n = state.topoNodes[k];
        n.vx += (centerX - n.x) * 0.001;
        n.vy += (centerY - n.y) * 0.001;
        n.vx *= damping;
        n.vy *= damping;
        n.x += n.vx * dt;
        n.y += n.vy * dt;
        n.x = Math.max(40, Math.min(W - 40, n.x));
        n.y = Math.max(40, Math.min(H - 40, n.y));
      }

      // Draw edges
      for (var e2 = 0; e2 < state.topoEdges.length; e2++) {
        var edge2 = state.topoEdges[e2];
        var na2 = state.topoNodes[edge2.a], nb2 = state.topoNodes[edge2.b];
        ctx.beginPath();
        ctx.moveTo(na2.x, na2.y);
        ctx.lineTo(nb2.x, nb2.y);
        ctx.strokeStyle = 'rgba(126, 200, 160, ' + (edge2.strength * 0.6).toFixed(2) + ')';
        ctx.lineWidth = Math.max(0.5, edge2.strength * 4);
        ctx.stroke();
      }

      // Draw nodes
      for (var m = 0; m < state.topoNodes.length; m++) {
        var nd = state.topoNodes[m];
        var r = 8 + nd.value * 20;
        var breathe = 1 + Math.sin(time * Math.PI / 2) * 0.08;
        r *= breathe;

        var green = Math.floor(160 + nd.value * 60);
        var alpha = 0.4 + nd.value * 0.5;

        // Glow
        ctx.beginPath();
        ctx.arc(nd.x, nd.y, r * 1.8, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(126, ' + green + ', 160, ' + (alpha * 0.12).toFixed(2) + ')';
        ctx.fill();

        // Node
        ctx.beginPath();
        ctx.arc(nd.x, nd.y, r, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(126, ' + green + ', 160, ' + alpha.toFixed(2) + ')';
        ctx.fill();
        ctx.strokeStyle = 'rgba(126, ' + green + ', 160, ' + (alpha * 0.8).toFixed(2) + ')';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Label
        ctx.fillStyle = 'rgba(208, 216, 208, 0.8)';
        ctx.font = '11px -apple-system, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(nd.name, nd.x, nd.y + r + 14);

        // Value
        ctx.fillStyle = 'rgba(232, 197, 71, 0.7)';
        ctx.font = '10px "SF Mono", monospace';
        ctx.fillText(nd.value.toFixed(3), nd.x, nd.y);
      }
    }

    frame();
  }

  // Run topology analysis
  async function runTopoAnalysis() {
    if (!state.workerReady) return;

    var btn = document.getElementById('btn-topo-analyze');
    if (btn) { btn.disabled = true; btn.textContent = 'Analyzing...'; }

    try {
      // First run a few cycles to build up observation data
      for (var i = 0; i < 5; i++) {
        await window.send('cycle', { input: 'topology observation cycle ' + i });
      }

      var topo = await window.send('topologyAnalysis', {});
      state.lastTopoData = topo;

      if (topo) {
        // TopologyAnalysis has nested structs:
        // topo.betti = { beta_0, beta_1, beta_2, euler_characteristic }
        // topo.persistence_pairs = [{ dimension, birth, death }]
        // topo.wave_packets = [{ center: [7], amplitude, frequency, width, created_cycle }]
        // topo.interference_events = [{ position: [7], ... }]
        // topo.interpretation = { unity, complexity, fragmentation }
        // topo.points_analyzed = usize

        // Update Betti numbers
        var betti = topo.betti || {};
        document.getElementById('betti-b0').textContent = betti.beta_0 !== undefined ? betti.beta_0 : '--';
        document.getElementById('betti-b1').textContent = betti.beta_1 !== undefined ? betti.beta_1 : '--';
        document.getElementById('betti-b2').textContent = betti.beta_2 !== undefined ? betti.beta_2 : '--';
        document.getElementById('betti-euler').textContent = betti.euler_characteristic !== undefined ? betti.euler_characteristic : '--';

        // Interpretation
        var interp = topo.interpretation || {};
        var interpEl = document.getElementById('topo-interpretation');
        if (interpEl) {
          interpEl.innerHTML =
            '<div style="display:flex;gap:1.5rem;flex-wrap:wrap">' +
            '<div><span style="color:var(--leaf-green)">Unity:</span> ' + (interp.unity || 0).toFixed(2) + '</div>' +
            '<div><span style="color:var(--solar-gold)">Complexity:</span> ' + (interp.complexity || 0).toFixed(3) + '</div>' +
            '<div><span style="color:var(--clay)">Fragmentation:</span> ' + (interp.fragmentation || 0).toFixed(3) + '</div>' +
            '<div><span style="color:var(--lichen-grey)">Points analyzed:</span> ' + (topo.points_analyzed || 0) + '</div>' +
            '</div>';
        }

        // Persistence pairs
        var ppEl = document.getElementById('topo-persistence');
        if (ppEl && topo.persistence_pairs && topo.persistence_pairs.length > 0) {
          ppEl.innerHTML = topo.persistence_pairs.map(function(pp) {
            var deathStr = pp.death !== null && pp.death !== undefined ? pp.death.toFixed(3) : '∞';
            var lifetime = pp.death !== null && pp.death !== undefined ? (pp.death - pp.birth).toFixed(3) : '∞';
            return '<div style="font-family:monospace;font-size:0.8rem;margin-bottom:0.3rem;color:var(--lichen-grey)">' +
              'dim=' + pp.dimension + '  birth=' + pp.birth.toFixed(3) + '  death=' + deathStr + '  lifetime=' + lifetime +
              '</div>';
          }).join('');
        } else if (ppEl) {
          ppEl.innerHTML = '<div style="color:var(--lichen-grey);font-size:0.8rem">No persistence pairs (need more observation cycles)</div>';
        }

        // Wave packets
        var wpEl = document.getElementById('wave-packets');
        if (wpEl && topo.wave_packets && topo.wave_packets.length > 0) {
          wpEl.innerHTML = topo.wave_packets.map(function(wp) {
            return '<div style="font-family:monospace;font-size:0.8rem;margin-bottom:0.3rem;color:var(--leaf-green)">' +
              'A=' + (wp.amplitude || 0).toFixed(3) +
              '  f=' + (wp.frequency || 0).toFixed(2) +
              '  σ=' + (wp.width || 0).toFixed(2) +
              '  E=' + ((wp.amplitude || 0) * (wp.amplitude || 0) * Math.pow(wp.width || 1, 3.5)).toFixed(4) +
              '</div>';
          }).join('');
        } else if (wpEl) {
          wpEl.innerHTML = '<div style="color:var(--lichen-grey);font-size:0.8rem">No active wave packets</div>';
        }

        // Interference
        var intfEl = document.getElementById('topo-interference');
        if (intfEl) {
          intfEl.textContent = (topo.interference_events || []).length;
        }

        // Update graph node values from the last cycle's consciousness
        // The topology analysis doesn't return per-node values directly,
        // but we can derive them from the interpretation
        if (state.topoNodes) {
          state.topoNodes[0].value = interp.unity || 0.5; // Phi ~ unity
          state.topoNodes[1].value = Math.max(0.1, (interp.complexity || 0) * 2 + 0.3); // Binding ~ complexity
          state.topoNodes[2].value = Math.max(0.1, (interp.unity || 0.5) * 0.8); // Workspace ~ unity
          state.topoNodes[3].value = Math.max(0.1, 1.0 - (interp.fragmentation || 0) * 5); // Attention ~ !fragmentation
          state.topoNodes[4].value = Math.max(0.1, (interp.complexity || 0) * 3 + 0.2); // Recurrence ~ complexity
          state.topoNodes[5].value = Math.max(0.1, (interp.unity || 0.5) * 0.4); // Embodiment
          state.topoNodes[6].value = Math.max(0.1, (interp.complexity || 0) * 2 + 0.3); // Knowledge ~ complexity

          // Update edge strengths from new node values
          for (var e = 0; e < state.topoEdges.length; e++) {
            var edge = state.topoEdges[e];
            edge.strength = (state.topoNodes[edge.a].value + state.topoNodes[edge.b].value) / 4;
          }
        }
      }
    } catch (e) {
      console.warn('Topology analysis failed:', e);
      var interpEl2 = document.getElementById('topo-interpretation');
      if (interpEl2) {
        interpEl2.innerHTML = '<div style="color:var(--clay)">Analysis failed: ' + e.message + '</div>';
      }
    }

    if (btn) { btn.disabled = false; btn.textContent = 'Run Analysis'; }
  }

  // Topology analysis button
  var topoBtn = document.getElementById('btn-topo-analyze');
  if (topoBtn) {
    topoBtn.addEventListener('click', function() {
      runTopoAnalysis();
    });
  }

  // Auto-update graph from cycle results
  window.updateTopoFromResult = updateTopoFromResult;
})();
