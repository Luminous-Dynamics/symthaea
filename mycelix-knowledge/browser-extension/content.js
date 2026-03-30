// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Knowledge Browser Extension - Content Script
 */

// ============================================================================
// UI Elements
// ============================================================================

let overlay = null;
let tooltip = null;

// ============================================================================
// Message Handler
// ============================================================================

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  switch (request.action) {
    case 'showLoading':
      showTooltip(request.text, 'loading');
      break;

    case 'showFactCheckResult':
      hideTooltip();
      showFactCheckOverlay(request.result, request.text);
      break;

    case 'showError':
      showTooltip(request.error, 'error');
      setTimeout(hideTooltip, 3000);
      break;

    case 'showSubmitForm':
      showSubmitForm(request.text);
      break;

    case 'showClassification':
      showClassificationResult(request.classification, request.text);
      break;
  }
});

// ============================================================================
// Tooltip
// ============================================================================

function showTooltip(text, type = 'info') {
  hideTooltip();

  tooltip = document.createElement('div');
  tooltip.className = `mycelix-tooltip mycelix-tooltip-${type}`;
  tooltip.innerHTML = `
    <div class="mycelix-tooltip-content">
      ${type === 'loading' ? '<div class="mycelix-spinner"></div>' : ''}
      <span>${escapeHtml(text)}</span>
    </div>
  `;

  document.body.appendChild(tooltip);

  // Position near selection
  const selection = window.getSelection();
  if (selection.rangeCount > 0) {
    const range = selection.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    tooltip.style.top = `${rect.bottom + window.scrollY + 10}px`;
    tooltip.style.left = `${rect.left + window.scrollX}px`;
  }
}

function hideTooltip() {
  if (tooltip) {
    tooltip.remove();
    tooltip = null;
  }
}

// ============================================================================
// Fact Check Overlay
// ============================================================================

function showFactCheckOverlay(result, originalText) {
  hideOverlay();

  const verdictColors = {
    TRUE: { bg: '#22c55e', text: 'white' },
    MOSTLY_TRUE: { bg: '#84cc16', text: 'white' },
    MIXED: { bg: '#eab308', text: 'black' },
    MOSTLY_FALSE: { bg: '#f97316', text: 'white' },
    FALSE: { bg: '#ef4444', text: 'white' },
    UNVERIFIABLE: { bg: '#6b7280', text: 'white' },
    INSUFFICIENT_EVIDENCE: { bg: '#8b5cf6', text: 'white' },
  };

  const color = verdictColors[result.verdict] || verdictColors.UNVERIFIABLE;

  overlay = document.createElement('div');
  overlay.className = 'mycelix-overlay';
  overlay.innerHTML = `
    <div class="mycelix-overlay-backdrop"></div>
    <div class="mycelix-overlay-content">
      <button class="mycelix-close-btn">&times;</button>

      <div class="mycelix-header">
        <img src="${chrome.runtime.getURL('icons/icon48.png')}" alt="Mycelix" class="mycelix-logo">
        <h2>Fact Check Result</h2>
      </div>

      <div class="mycelix-statement">
        <strong>Statement:</strong>
        <p>"${escapeHtml(originalText.substring(0, 200))}${originalText.length > 200 ? '...' : ''}"</p>
      </div>

      <div class="mycelix-verdict" style="background: ${color.bg}; color: ${color.text}">
        ${result.verdict.replace('_', ' ')}
      </div>

      <div class="mycelix-confidence">
        <strong>Confidence: ${Math.round(result.confidence * 100)}%</strong>
        <div class="mycelix-progress-bar">
          <div class="mycelix-progress-fill" style="width: ${result.confidence * 100}%"></div>
        </div>
      </div>

      <div class="mycelix-explanation">
        <h3>Explanation</h3>
        <p>${escapeHtml(result.explanation)}</p>
      </div>

      ${result.supportingClaims.length > 0 ? `
        <div class="mycelix-claims">
          <h3>Supporting Claims (${result.supportingClaims.length})</h3>
          <ul>
            ${result.supportingClaims
              .slice(0, 3)
              .map((c) => `<li class="mycelix-claim-support">✓ ${escapeHtml(c.content)}</li>`)
              .join('')}
          </ul>
        </div>
      ` : ''}

      ${result.contradictingClaims.length > 0 ? `
        <div class="mycelix-claims">
          <h3>Contradicting Claims (${result.contradictingClaims.length})</h3>
          <ul>
            ${result.contradictingClaims
              .slice(0, 3)
              .map((c) => `<li class="mycelix-claim-contradict">✗ ${escapeHtml(c.content)}</li>`)
              .join('')}
          </ul>
        </div>
      ` : ''}

      ${result.sources.length > 0 ? `
        <div class="mycelix-sources">
          <h3>Sources</h3>
          <ul>
            ${result.sources
              .slice(0, 5)
              .map((s) => `<li>📄 ${escapeHtml(s)}</li>`)
              .join('')}
          </ul>
        </div>
      ` : ''}

      <div class="mycelix-footer">
        <small>Checked at: ${new Date(result.checkedAt).toLocaleString()}</small>
        <a href="https://mycelix.net" target="_blank" rel="noopener">Learn more about Mycelix</a>
      </div>
    </div>
  `;

  document.body.appendChild(overlay);

  // Add close handlers
  overlay.querySelector('.mycelix-close-btn').addEventListener('click', hideOverlay);
  overlay.querySelector('.mycelix-overlay-backdrop').addEventListener('click', hideOverlay);

  // Close on Escape
  document.addEventListener('keydown', handleEscape);
}

function hideOverlay() {
  if (overlay) {
    overlay.remove();
    overlay = null;
    document.removeEventListener('keydown', handleEscape);
  }
}

function handleEscape(e) {
  if (e.key === 'Escape') {
    hideOverlay();
  }
}

// ============================================================================
// Classification Result
// ============================================================================

function showClassificationResult(classification, text) {
  hideOverlay();

  overlay = document.createElement('div');
  overlay.className = 'mycelix-overlay';
  overlay.innerHTML = `
    <div class="mycelix-overlay-backdrop"></div>
    <div class="mycelix-overlay-content mycelix-classification">
      <button class="mycelix-close-btn">&times;</button>

      <div class="mycelix-header">
        <h2>Epistemic Classification</h2>
      </div>

      <div class="mycelix-statement">
        <p>"${escapeHtml(text.substring(0, 150))}${text.length > 150 ? '...' : ''}"</p>
      </div>

      <div class="mycelix-dominant">
        Dominant Type: <strong>${classification.dominant.toUpperCase()}</strong>
      </div>

      <div class="mycelix-bars">
        <div class="mycelix-bar-item">
          <div class="mycelix-bar-label">
            <span style="color: #ef4444">Empirical</span>
            <span>${Math.round(classification.empirical * 100)}%</span>
          </div>
          <div class="mycelix-bar">
            <div class="mycelix-bar-fill" style="width: ${classification.empirical * 100}%; background: #ef4444"></div>
          </div>
          <small>Verifiable through observation</small>
        </div>

        <div class="mycelix-bar-item">
          <div class="mycelix-bar-label">
            <span style="color: #22c55e">Normative</span>
            <span>${Math.round(classification.normative * 100)}%</span>
          </div>
          <div class="mycelix-bar">
            <div class="mycelix-bar-fill" style="width: ${classification.normative * 100}%; background: #22c55e"></div>
          </div>
          <small>Value-based, ethical</small>
        </div>

        <div class="mycelix-bar-item">
          <div class="mycelix-bar-label">
            <span style="color: #3b82f6">Mythic</span>
            <span>${Math.round(classification.mythic * 100)}%</span>
          </div>
          <div class="mycelix-bar">
            <div class="mycelix-bar-fill" style="width: ${classification.mythic * 100}%; background: #3b82f6"></div>
          </div>
          <small>Meaning-making, narrative</small>
        </div>
      </div>

      <div class="mycelix-footer">
        <button class="mycelix-btn mycelix-btn-primary" id="mycelix-submit-classified">
          Submit as Claim
        </button>
      </div>
    </div>
  `;

  document.body.appendChild(overlay);

  overlay.querySelector('.mycelix-close-btn').addEventListener('click', hideOverlay);
  overlay.querySelector('.mycelix-overlay-backdrop').addEventListener('click', hideOverlay);
  overlay.querySelector('#mycelix-submit-classified').addEventListener('click', () => {
    hideOverlay();
    showSubmitForm(text, classification);
  });

  document.addEventListener('keydown', handleEscape);
}

// ============================================================================
// Submit Form
// ============================================================================

function showSubmitForm(text, preClassification = null) {
  hideOverlay();

  const classification = preClassification || {
    empirical: 0.5,
    normative: 0.3,
    mythic: 0.2,
  };

  overlay = document.createElement('div');
  overlay.className = 'mycelix-overlay';
  overlay.innerHTML = `
    <div class="mycelix-overlay-backdrop"></div>
    <div class="mycelix-overlay-content mycelix-submit-form">
      <button class="mycelix-close-btn">&times;</button>

      <div class="mycelix-header">
        <h2>Submit Claim</h2>
      </div>

      <form id="mycelix-claim-form">
        <div class="mycelix-form-group">
          <label>Claim Content</label>
          <textarea id="claim-content" rows="4">${escapeHtml(text)}</textarea>
        </div>

        <div class="mycelix-form-group">
          <label>Epistemic Classification</label>
          <div class="mycelix-slider-group">
            <label>Empirical: <span id="empirical-value">${Math.round(classification.empirical * 100)}%</span></label>
            <input type="range" id="empirical" min="0" max="100" value="${classification.empirical * 100}">
          </div>
          <div class="mycelix-slider-group">
            <label>Normative: <span id="normative-value">${Math.round(classification.normative * 100)}%</span></label>
            <input type="range" id="normative" min="0" max="100" value="${classification.normative * 100}">
          </div>
          <div class="mycelix-slider-group">
            <label>Mythic: <span id="mythic-value">${Math.round(classification.mythic * 100)}%</span></label>
            <input type="range" id="mythic" min="0" max="100" value="${classification.mythic * 100}">
          </div>
        </div>

        <div class="mycelix-form-group">
          <label>Tags (comma-separated)</label>
          <input type="text" id="claim-tags" placeholder="science, health, technology">
        </div>

        <div class="mycelix-form-actions">
          <button type="button" class="mycelix-btn" onclick="hideOverlay()">Cancel</button>
          <button type="submit" class="mycelix-btn mycelix-btn-primary">Submit Claim</button>
        </div>
      </form>
    </div>
  `;

  document.body.appendChild(overlay);

  // Update value displays
  ['empirical', 'normative', 'mythic'].forEach((field) => {
    const slider = overlay.querySelector(`#${field}`);
    const display = overlay.querySelector(`#${field}-value`);
    slider.addEventListener('input', () => {
      display.textContent = `${slider.value}%`;
    });
  });

  // Form submission
  overlay.querySelector('#mycelix-claim-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const claim = {
      content: overlay.querySelector('#claim-content').value,
      classification: {
        empirical: parseInt(overlay.querySelector('#empirical').value) / 100,
        normative: parseInt(overlay.querySelector('#normative').value) / 100,
        mythic: parseInt(overlay.querySelector('#mythic').value) / 100,
      },
      tags: overlay
        .querySelector('#claim-tags')
        .value.split(',')
        .map((t) => t.trim())
        .filter(Boolean),
    };

    const response = await chrome.runtime.sendMessage({
      action: 'submitClaim',
      claim,
    });

    if (response.success) {
      hideOverlay();
      showTooltip('Claim submitted successfully!', 'success');
      setTimeout(hideTooltip, 3000);
    } else {
      showTooltip(response.error, 'error');
    }
  });

  overlay.querySelector('.mycelix-close-btn').addEventListener('click', hideOverlay);
  overlay.querySelector('.mycelix-overlay-backdrop').addEventListener('click', hideOverlay);

  document.addEventListener('keydown', handleEscape);
}

// ============================================================================
// Utilities
// ============================================================================

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// Make hideOverlay available globally for inline onclick
window.hideOverlay = hideOverlay;
