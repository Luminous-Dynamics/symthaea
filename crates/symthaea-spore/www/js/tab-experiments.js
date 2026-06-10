// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// ==================================================================
// tab-experiments.js — Tab 3: Experiment cards, charts
// ==================================================================
(function() {
  'use strict';
  var state = window.portalState;

  // Anesthesia
  document.getElementById('btn-exp-anesthesia').addEventListener('click', async function() {
    if (!state.workerReady) return;
    this.disabled = true;
    this.textContent = 'Running...';
    try {
      var result = await window.send('experiment', { name: 'anesthesia', warmup: 20, suppression: 20, recovery: 20 });
      // Draw chart
      if (result && result.consciousness_series) {
        window.drawLineChart('chart-anesthesia', result.consciousness_series);
      } else if (result && result.time_series) {
        var vals = result.time_series.map(function(p) { return p.consciousness || p.consciousness_level || 0; });
        window.drawLineChart('chart-anesthesia', vals);
      }
      // Result text
      var rEl = document.getElementById('result-anesthesia');
      rEl.innerHTML =
        'Collapsed: <span style="color:var(--leaf-green)">' + (result.collapsed ? 'Yes' : 'No') + '</span> | ' +
        'Recovered: <span style="color:var(--leaf-green)">' + (result.recovered ? 'Yes' : 'No') + '</span> | ' +
        'Recovery: <span style="color:var(--solar-gold)">' + (result.recovery_cycle || result.recovery_time || '?') + ' cycles</span>';
    } catch (e) {
      document.getElementById('result-anesthesia').textContent = 'Error: ' + e.message;
    }
    this.disabled = false;
    this.textContent = 'Run';
  });

  // PCI
  document.getElementById('btn-exp-pci').addEventListener('click', async function() {
    if (!state.workerReady) return;
    this.disabled = true;
    this.textContent = 'Running...';
    try {
      var result = await window.send('experiment', { name: 'pci', magnitude: 0.3, cycles: 30 });
      var pciRatio = result.pci_ratio || result.pci || 0;
      // Draw chart: big number display via canvas
      var canvas = document.getElementById('chart-pci');
      var ctx = canvas.getContext('2d');
      var dpr = window.devicePixelRatio || 1;
      var rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, rect.width, rect.height);

      // Big PCI number
      ctx.fillStyle = pciRatio >= 1.3 ? '#7ec8a0' : '#c76b5a';
      ctx.font = '48px "SF Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(pciRatio.toFixed(3), rect.width / 2, rect.height / 2 + 10);

      // Threshold line
      ctx.strokeStyle = 'rgba(199, 107, 90, 0.4)';
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      var threshY = rect.height * 0.75;
      ctx.moveTo(20, threshY);
      ctx.lineTo(rect.width - 20, threshY);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(208,216,208,0.4)';
      ctx.font = '9px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText('clinical threshold 1.3', rect.width - 25, threshY - 4);

      var rEl = document.getElementById('result-pci');
      rEl.innerHTML =
        'PCI ratio: <span style="color:var(--solar-gold)">' + pciRatio.toFixed(3) + '</span> | ' +
        'Clinical threshold: <span style="color:' + (pciRatio >= 1.3 ? 'var(--leaf-green)' : 'var(--autumn-rust)') + '">' +
        (pciRatio >= 1.3 ? 'PASS' : 'FAIL') + '</span>';
    } catch (e) {
      document.getElementById('result-pci').textContent = 'Error: ' + e.message;
    }
    this.disabled = false;
    this.textContent = 'Run';
  });

  // Split Brain
  document.getElementById('btn-exp-split').addEventListener('click', async function() {
    if (!state.workerReady) return;
    this.disabled = true;
    this.textContent = 'Running...';
    try {
      var result = await window.send('experiment', { name: 'split_brain', cycles: 20 });
      var unified = result.unified_consciousness || result.unified || 0;
      var left = result.left_consciousness || result.left || 0;
      var right = result.right_consciousness || result.right || 0;
      var maxVal = Math.max(unified, left, right, 0.01);

      // Update bars
      var barH = 100;
      document.getElementById('bar-unified').style.height = (unified / maxVal * barH) + 'px';
      document.getElementById('bar-left').style.height = (left / maxVal * barH) + 'px';
      document.getElementById('bar-right').style.height = (right / maxVal * barH) + 'px';
      document.getElementById('val-unified').textContent = unified.toFixed(3);
      document.getElementById('val-left').textContent = left.toFixed(3);
      document.getElementById('val-right').textContent = right.toFixed(3);

      var splitReduces = unified > Math.max(left, right);
      var ratio = maxVal > 0 ? (Math.max(left, right) / unified).toFixed(2) : '?';
      document.getElementById('result-split').innerHTML =
        'Split reduces consciousness: <span style="color:' + (splitReduces ? 'var(--leaf-green)' : 'var(--autumn-rust)') + '">' +
        (splitReduces ? 'Yes' : 'No') + '</span> | ' +
        'Ratio: <span style="color:var(--solar-gold)">' + ratio + '</span>';
    } catch (e) {
      document.getElementById('result-split').textContent = 'Error: ' + e.message;
    }
    this.disabled = false;
    this.textContent = 'Run';
  });

  // Collapse Threshold
  document.getElementById('btn-exp-collapse').addEventListener('click', async function() {
    if (!state.workerReady) return;
    this.disabled = true;
    this.textContent = 'Running...';
    try {
      var result = await window.send('experiment', { name: 'collapse', steps: 10, cyclesPerStep: 10 });
      if (result && result.degradation_series) {
        window.drawLineChart('chart-collapse', result.degradation_series);
      } else if (result && result.consciousness_series) {
        window.drawLineChart('chart-collapse', result.consciousness_series);
      } else if (result && result.steps) {
        var vals = result.steps.map(function(s) { return s.consciousness || s.consciousness_level || 0; });
        window.drawLineChart('chart-collapse', vals);
      }
      var collapsePoint = result.collapse_fraction || result.collapse_point || '?';
      if (typeof collapsePoint === 'number') collapsePoint = (collapsePoint * 100).toFixed(0);
      document.getElementById('result-collapse').innerHTML =
        'Collapse point: <span style="color:var(--solar-gold)">' + collapsePoint + '% degradation</span>';
    } catch (e) {
      document.getElementById('result-collapse').textContent = 'Error: ' + e.message;
    }
    this.disabled = false;
    this.textContent = 'Run';
  });

  // Run All
  function updateBatteryProgress(experiment, status) {
    var el = document.querySelector('.bp-item[data-exp="' + experiment + '"]');
    if (el) {
      el.className = 'bp-item ' + status;
      el.textContent = experiment + (status === 'done' ? ' \u2713' : status === 'running' ? ' ...' : ' \u2717');
    }
  }
  window.updateBatteryProgress = updateBatteryProgress;

  document.getElementById('btn-run-all').addEventListener('click', async function() {
    if (!state.workerReady) return;
    this.disabled = true;
    this.textContent = 'Running Battery...';

    var bp = document.getElementById('battery-progress');
    bp.innerHTML = ['anesthesia', 'pci', 'split_brain', 'collapse'].map(function(name) {
      return '<span class="bp-item" data-exp="' + name + '">' + name + '</span>';
    }).join('');

    try {
      var result = await window.send('battery', {});
      // Trigger individual card updates with results
      if (result.anesthesia) {
        document.getElementById('btn-exp-anesthesia').click();
      }
      if (result.pci) {
        document.getElementById('btn-exp-pci').click();
      }
      if (result.split_brain) {
        document.getElementById('btn-exp-split').click();
      }
      if (result.collapse) {
        document.getElementById('btn-exp-collapse').click();
      }
    } catch (e) {
      console.warn('Battery failed:', e);
    }
    this.disabled = false;
    this.textContent = 'Run All Experiments';
  });

  // Multi-substrate comparison
  document.getElementById('btn-substrate-compare').addEventListener('click', async function() {
    if (!state.workerReady) return;
    var select = document.getElementById('substrate-select');
    var selected = [];
    for (var i = 0; i < select.options.length; i++) {
      if (select.options[i].selected) selected.push(select.options[i].value);
    }
    if (selected.length === 0) return;

    this.disabled = true;
    this.textContent = 'Comparing...';
    try {
      var result = await window.send('multiSubstrate', { substrates: selected, cycles: 50 });
      // Draw multi-series chart
      var allSeries = [];
      var labels = [];
      for (var sub in result) {
        if (result.hasOwnProperty(sub)) {
          labels.push(sub);
          allSeries.push(result[sub].map(function(p) { return p.consciousness; }));
        }
      }
      window.drawLineChart('chart-substrate', allSeries, { labels: labels });

      // Summary
      var rEl = document.getElementById('result-substrate');
      var summaries = labels.map(function(name, i) {
        var series = allSeries[i];
        var avg = series.reduce(function(a, b) { return a + b; }, 0) / series.length;
        return name + ': avg \u03A6 ' + avg.toFixed(3);
      });
      rEl.textContent = summaries.join(' | ');
    } catch (e) {
      document.getElementById('result-substrate').textContent = 'Error: ' + e.message;
    }
    this.disabled = false;
    this.textContent = 'Compare (50 cycles)';
  });
})();
