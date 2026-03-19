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
        var dx = nb.x - na.x;
        var dy = nb.y - na.y;
        var dist = Math.sqrt(dx * dx + dy * dy) || 1;
        var restLen = 100;
        var force = springK * (dist - restLen) * edge.strength;
        var fx = (dx / dist) * force;
        var fy = (dy / dist) * force;
        na.vx += fx * dt;
        na.vy += fy * dt;
        nb.vx -= fx * dt;
        nb.vy -= fy * dt;
      }

      // Center gravity
      for (var i = 0; i < state.topoNodes.length; i++) {
        var n = state.topoNodes[i];
        n.vx += (centerX - n.x) * 0.001;
        n.vy += (centerY - n.y) * 0.001;
        n.vx *= damping;
        n.vy *= damping;
        n.x += n.vx * dt;
        n.y += n.vy * dt;
        // Bounds
        n.x = Math.max(40, Math.min(W - 40, n.x));
        n.y = Math.max(40, Math.min(H - 40, n.y));
      }

      // Draw edges
      for (var e = 0; e < state.topoEdges.length; e++) {
        var edge = state.topoEdges[e];
        var na = state.topoNodes[edge.a], nb = state.topoNodes[edge.b];
        ctx.beginPath();
        ctx.moveTo(na.x, na.y);
        ctx.lineTo(nb.x, nb.y);
        ctx.strokeStyle = 'rgba(126, 200, 160, ' + (edge.strength * 0.6).toFixed(2) + ')';
        ctx.lineWidth = Math.max(0.5, edge.strength * 4);
        ctx.stroke();
      }

      // Draw nodes
      for (var i = 0; i < state.topoNodes.length; i++) {
        var n = state.topoNodes[i];
        var r = 8 + n.value * 20;
        // Breathing animation (4s period)
        var breathe = 1 + Math.sin(time * Math.PI / 2) * 0.08;
        r *= breathe;

        // Color: green for high, grey for low
        var green = Math.floor(160 + n.value * 60);
        var alpha = 0.4 + n.value * 0.5;

        // Glow
        ctx.beginPath();
        ctx.arc(n.x, n.y, r * 1.8, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(126, ' + green + ', 160, ' + (alpha * 0.12).toFixed(2) + ')';
        ctx.fill();

        // Node
        ctx.beginPath();
        ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(126, ' + green + ', 160, ' + alpha.toFixed(2) + ')';
        ctx.fill();
        ctx.strokeStyle = 'rgba(126, ' + green + ', 160, ' + (alpha * 0.8).toFixed(2) + ')';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Label
        ctx.fillStyle = 'rgba(208, 216, 208, 0.8)';
        ctx.font = '10px -apple-system, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(n.name, n.x, n.y + r + 14);

        // Value
        ctx.fillStyle = 'rgba(208, 216, 208, 0.5)';
        ctx.font = '9px "SF Mono", monospace';
        ctx.fillText(n.value.toFixed(2), n.x, n.y);
      }
    }

    frame();
  }

  // Topology analysis button
  document.getElementById('btn-topo-analyze').addEventListener('click', async function() {
    if (!state.workerReady) return;
    this.disabled = true;
    this.textContent = 'Analyzing...';
    try {
      var topo = await window.send('topologyAnalysis', {});
      state.lastTopoData = topo;

      // Update Betti numbers
      if (topo) {
        document.getElementById('betti-b0').textContent = topo.betti_0 !== undefined ? topo.betti_0 : '--';
        document.getElementById('betti-b1').textContent = topo.betti_1 !== undefined ? topo.betti_1 : '--';
        document.getElementById('betti-b2').textContent = topo.betti_2 !== undefined ? topo.betti_2 : '--';
        var euler = (topo.betti_0 || 0) - (topo.betti_1 || 0) + (topo.betti_2 || 0);
        document.getElementById('betti-euler').textContent = euler;

        // Wave packets
        var wpEl = document.getElementById('wave-packets');
        if (topo.wave_packets && topo.wave_packets.length > 0) {
          wpEl.innerHTML = topo.wave_packets.map(function(wp) {
            return '<div class="wave-packet-item">' +
              'A=' + (wp.amplitude || 0).toFixed(3) +
              '  f=' + (wp.frequency || 0).toFixed(2) +
              '  center=(' + (wp.center_x || 0).toFixed(1) + ', ' + (wp.center_y || 0).toFixed(1) + ')' +
              '</div>';
          }).join('');
        } else {
          wpEl.innerHTML = '<div class="wave-packet-item">No active wave packets detected</div>';
        }

        document.getElementById('topo-interference').textContent = topo.interference_count !== undefined ? topo.interference_count : '0';

        // Update node values from topology
        if (state.topoNodes && topo.node_values) {
          for (var i = 0; i < Math.min(state.topoNodes.length, topo.node_values.length); i++) {
            state.topoNodes[i].value = topo.node_values[i];
          }
        }
      }
    } catch (e) {
      console.warn('Topology analysis failed:', e);
    }
    this.disabled = false;
    this.textContent = 'Run Analysis';
  });
})();
