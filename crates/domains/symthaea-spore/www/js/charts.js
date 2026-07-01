// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// ==================================================================
// charts.js — Shared canvas chart drawing functions
// ==================================================================

// Helper: draw a line chart on a canvas
function drawLineChart(canvasId, data, opts) {
  var canvas = document.getElementById(canvasId);
  if (!canvas) return;
  var ctx = canvas.getContext('2d');
  var dpr = window.devicePixelRatio || 1;
  var rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  var W = rect.width, H = rect.height;
  ctx.clearRect(0, 0, W, H);

  if (!data || data.length === 0) return;

  var pad = { top: 10, right: 10, bottom: 20, left: 35 };
  var plotW = W - pad.left - pad.right;
  var plotH = H - pad.top - pad.bottom;

  // Find min/max
  var minY = Infinity, maxY = -Infinity;
  if (Array.isArray(data[0])) {
    // Multiple series
    for (var s = 0; s < data.length; s++) {
      for (var i = 0; i < data[s].length; i++) {
        if (data[s][i] < minY) minY = data[s][i];
        if (data[s][i] > maxY) maxY = data[s][i];
      }
    }
  } else {
    for (var i = 0; i < data.length; i++) {
      if (data[i] < minY) minY = data[i];
      if (data[i] > maxY) maxY = data[i];
    }
  }
  if (minY === maxY) { minY -= 0.1; maxY += 0.1; }
  var rangeY = maxY - minY;

  // Axes
  ctx.strokeStyle = 'rgba(255,255,255,0.1)';
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, H - pad.bottom);
  ctx.lineTo(W - pad.right, H - pad.bottom);
  ctx.stroke();

  // Y labels
  ctx.fillStyle = 'rgba(208,216,208,0.4)';
  ctx.font = '8px "SF Mono", monospace';
  ctx.textAlign = 'right';
  ctx.fillText(maxY.toFixed(2), pad.left - 3, pad.top + 8);
  ctx.fillText(minY.toFixed(2), pad.left - 3, H - pad.bottom);

  // Threshold line
  if (opts && opts.threshold !== undefined) {
    var ty = pad.top + plotH * (1 - (opts.threshold - minY) / rangeY);
    ctx.strokeStyle = 'rgba(199, 107, 90, 0.4)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pad.left, ty);
    ctx.lineTo(W - pad.right, ty);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(199, 107, 90, 0.6)';
    ctx.textAlign = 'left';
    ctx.fillText(opts.thresholdLabel || '', W - pad.right - 40, ty - 3);
  }

  var colors = opts && opts.colors ? opts.colors :
    ['rgba(126, 200, 160, 1)', 'rgba(232, 197, 71, 1)', 'rgba(196, 149, 106, 1)', 'rgba(90, 184, 160, 1)', 'rgba(212, 168, 71, 1)', 'rgba(212, 160, 196, 1)'];

  function drawSeries(values, color, fillColor) {
    var len = values.length;
    var dx = plotW / Math.max(1, len - 1);
    ctx.beginPath();
    for (var i = 0; i < len; i++) {
      var x = pad.left + i * dx;
      var y = pad.top + plotH * (1 - (values[i] - minY) / rangeY);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Fill
    if (fillColor) {
      ctx.lineTo(pad.left + (len - 1) * dx, H - pad.bottom);
      ctx.lineTo(pad.left, H - pad.bottom);
      ctx.closePath();
      ctx.fillStyle = fillColor;
      ctx.fill();
    }
  }

  if (Array.isArray(data[0])) {
    for (var s = 0; s < data.length; s++) {
      drawSeries(data[s], colors[s % colors.length], null);
    }
    // Legend
    if (opts && opts.labels) {
      ctx.font = '8px sans-serif';
      for (var s = 0; s < opts.labels.length; s++) {
        ctx.fillStyle = colors[s % colors.length];
        ctx.textAlign = 'left';
        ctx.fillText(opts.labels[s], pad.left + 5 + s * 80, pad.top + 10);
      }
    }
  } else {
    drawSeries(data, colors[0], 'rgba(126, 200, 160, 0.08)');
  }
}
window.drawLineChart = drawLineChart;
