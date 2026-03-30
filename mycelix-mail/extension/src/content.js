// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Mail Browser Extension - Content Script
 *
 * Injects trust indicators into Gmail and Outlook web interfaces.
 */

// ============================================================================
// Configuration
// ============================================================================

const TRUST_COLORS = {
  high: '#4CAF50',
  medium: '#FFC107',
  low: '#FF9800',
  untrusted: '#F44336',
  unknown: '#9E9E9E',
};

const SELECTORS = {
  gmail: {
    emailRow: 'tr.zA',
    senderSpan: '.yW span[email]',
    senderEmail: 'email',
    emailContainer: '.ae4',
  },
  outlook: {
    emailRow: '[data-convid]',
    senderSpan: '.OZZZK',
    emailContainer: '.jGG6V',
  },
};

let currentPlatform = null;
let observer = null;
let trustCache = new Map();

// ============================================================================
// Platform Detection
// ============================================================================

function detectPlatform() {
  if (window.location.hostname.includes('mail.google.com')) {
    return 'gmail';
  } else if (window.location.hostname.includes('outlook.live.com')) {
    return 'outlook';
  }
  return null;
}

// ============================================================================
// Trust Badge Creation
// ============================================================================

function createTrustBadge(trustData) {
  const badge = document.createElement('span');
  badge.className = 'mycelix-trust-badge';

  const score = trustData.score ?? 0;
  let level, color, icon;

  if (trustData.error) {
    level = 'unknown';
    color = TRUST_COLORS.unknown;
    icon = '?';
  } else if (score >= 0.8) {
    level = 'high';
    color = TRUST_COLORS.high;
    icon = '✓';
  } else if (score >= 0.5) {
    level = 'medium';
    color = TRUST_COLORS.medium;
    icon = '~';
  } else if (score >= 0.2) {
    level = 'low';
    color = TRUST_COLORS.low;
    icon = '!';
  } else {
    level = 'untrusted';
    color = TRUST_COLORS.untrusted;
    icon = '✗';
  }

  badge.innerHTML = `
    <span class="mycelix-trust-icon" style="background-color: ${color}">${icon}</span>
    <span class="mycelix-trust-tooltip">
      <strong>Mycelix Trust Score</strong>
      <div class="mycelix-trust-score-bar">
        <div class="mycelix-trust-score-fill" style="width: ${score * 100}%; background-color: ${color}"></div>
      </div>
      <div class="mycelix-trust-level">Level: ${level.charAt(0).toUpperCase() + level.slice(1)}</div>
      ${trustData.attestations ? `<div class="mycelix-trust-attestations">${trustData.attestations} attestations</div>` : ''}
      ${trustData.pathLength ? `<div class="mycelix-trust-path">Trust path length: ${trustData.pathLength}</div>` : ''}
      <a href="#" class="mycelix-trust-details" data-email="${trustData.email}">View Details</a>
    </span>
  `;

  badge.dataset.trustLevel = level;
  badge.dataset.email = trustData.email;

  // Add click handler for details
  badge.querySelector('.mycelix-trust-details').addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    showTrustDetails(trustData.email);
  });

  return badge;
}

// ============================================================================
// Trust Details Modal
// ============================================================================

function showTrustDetails(email) {
  let modal = document.getElementById('mycelix-trust-modal');

  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'mycelix-trust-modal';
    modal.className = 'mycelix-modal';
    modal.innerHTML = `
      <div class="mycelix-modal-content">
        <div class="mycelix-modal-header">
          <h3>Trust Details</h3>
          <button class="mycelix-modal-close">&times;</button>
        </div>
        <div class="mycelix-modal-body">
          <div class="mycelix-loading">Loading...</div>
        </div>
        <div class="mycelix-modal-footer">
          <a href="#" class="mycelix-btn mycelix-btn-primary" id="mycelix-view-full">
            View in Mycelix Mail
          </a>
          <button class="mycelix-btn mycelix-btn-secondary mycelix-modal-close">
            Close
          </button>
        </div>
      </div>
    `;
    document.body.appendChild(modal);

    // Close handlers
    modal.querySelectorAll('.mycelix-modal-close').forEach((btn) => {
      btn.addEventListener('click', () => {
        modal.classList.remove('open');
      });
    });

    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        modal.classList.remove('open');
      }
    });
  }

  // Update content
  const body = modal.querySelector('.mycelix-modal-body');
  body.innerHTML = '<div class="mycelix-loading">Loading trust information...</div>';

  const viewFullLink = modal.querySelector('#mycelix-view-full');
  viewFullLink.href = `https://mail.mycelix.com/trust/${encodeURIComponent(email)}`;

  modal.classList.add('open');

  // Fetch trust details
  chrome.runtime.sendMessage({ action: 'checkTrust', email }, (response) => {
    if (response.error) {
      body.innerHTML = `
        <div class="mycelix-error">
          <p>Unable to load trust information.</p>
          <p>${response.error}</p>
        </div>
      `;
    } else {
      const score = response.score ?? 0;
      body.innerHTML = `
        <div class="mycelix-trust-detail">
          <div class="mycelix-trust-email">${email}</div>
          <div class="mycelix-trust-score-large">
            <div class="mycelix-score-circle" style="--score: ${score * 100}%">
              <span>${Math.round(score * 100)}%</span>
            </div>
          </div>
          <div class="mycelix-trust-info">
            <div class="mycelix-info-row">
              <span class="label">Attestations:</span>
              <span class="value">${response.attestations || 0}</span>
            </div>
            <div class="mycelix-info-row">
              <span class="label">Trust Path:</span>
              <span class="value">${response.pathLength || 'N/A'} hops</span>
            </div>
            <div class="mycelix-info-row">
              <span class="label">First Contact:</span>
              <span class="value">${response.firstContact || 'Unknown'}</span>
            </div>
            <div class="mycelix-info-row">
              <span class="label">Last Interaction:</span>
              <span class="value">${response.lastInteraction || 'Unknown'}</span>
            </div>
          </div>
          ${
            response.attestors && response.attestors.length > 0
              ? `
            <div class="mycelix-attestors">
              <h4>Attested by:</h4>
              <ul>
                ${response.attestors.map((a) => `<li>${a.name} (${a.type})</li>`).join('')}
              </ul>
            </div>
          `
              : ''
          }
        </div>
      `;
    }
  });
}

// ============================================================================
// Email Row Processing
// ============================================================================

async function processEmailRow(row) {
  if (row.dataset.mycelixProcessed) return;
  row.dataset.mycelixProcessed = 'true';

  const selectors = SELECTORS[currentPlatform];
  const senderElement = row.querySelector(selectors.senderSpan);

  if (!senderElement) return;

  let email;
  if (currentPlatform === 'gmail') {
    email = senderElement.getAttribute(selectors.senderEmail);
  } else {
    // Try to extract email from element or title
    email = senderElement.getAttribute('title') || senderElement.textContent;
    const emailMatch = email.match(/[\w.-]+@[\w.-]+\.\w+/);
    email = emailMatch ? emailMatch[0] : null;
  }

  if (!email) return;

  // Check cache first
  let trustData = trustCache.get(email);

  if (!trustData) {
    // Request trust check from background
    trustData = await new Promise((resolve) => {
      chrome.runtime.sendMessage({ action: 'checkTrust', email }, resolve);
    });
    trustData.email = email;
    trustCache.set(email, trustData);
  }

  // Create and insert badge
  const badge = createTrustBadge(trustData);

  // Insert after sender name
  const parent = senderElement.parentElement;
  if (parent && !parent.querySelector('.mycelix-trust-badge')) {
    parent.style.display = 'flex';
    parent.style.alignItems = 'center';
    parent.style.gap = '4px';
    parent.appendChild(badge);
  }
}

async function processAllEmailRows() {
  const selectors = SELECTORS[currentPlatform];
  const rows = document.querySelectorAll(selectors.emailRow);

  // Collect emails for batch processing
  const emails = [];
  const rowsToProcess = [];

  rows.forEach((row) => {
    if (row.dataset.mycelixProcessed) return;

    const senderElement = row.querySelector(selectors.senderSpan);
    if (!senderElement) return;

    let email;
    if (currentPlatform === 'gmail') {
      email = senderElement.getAttribute(selectors.senderEmail);
    } else {
      const text = senderElement.getAttribute('title') || senderElement.textContent;
      const match = text.match(/[\w.-]+@[\w.-]+\.\w+/);
      email = match ? match[0] : null;
    }

    if (email) {
      emails.push(email);
      rowsToProcess.push({ row, email, senderElement });
    }
  });

  if (emails.length === 0) return;

  // Batch check trust
  const results = await new Promise((resolve) => {
    chrome.runtime.sendMessage({ action: 'batchCheckTrust', emails }, resolve);
  });

  // Apply badges
  results.forEach((trustData, index) => {
    const { row, email, senderElement } = rowsToProcess[index];

    row.dataset.mycelixProcessed = 'true';
    trustData.email = email;
    trustCache.set(email, trustData);

    const badge = createTrustBadge(trustData);
    const parent = senderElement.parentElement;

    if (parent && !parent.querySelector('.mycelix-trust-badge')) {
      parent.style.display = 'flex';
      parent.style.alignItems = 'center';
      parent.style.gap = '4px';
      parent.appendChild(badge);
    }
  });
}

// ============================================================================
// DOM Observer
// ============================================================================

function startObserver() {
  if (observer) return;

  const selectors = SELECTORS[currentPlatform];
  const container = document.querySelector(selectors.emailContainer);

  if (!container) {
    // Retry later
    setTimeout(startObserver, 1000);
    return;
  }

  observer = new MutationObserver((mutations) => {
    let shouldProcess = false;

    for (const mutation of mutations) {
      if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
        shouldProcess = true;
        break;
      }
    }

    if (shouldProcess) {
      requestAnimationFrame(processAllEmailRows);
    }
  });

  observer.observe(container, {
    childList: true,
    subtree: true,
  });

  // Initial processing
  processAllEmailRows();
}

// ============================================================================
// Message Handler
// ============================================================================

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'showTrustOverlay') {
    // Get currently selected/hovered email
    const selection = window.getSelection().toString();
    const emailMatch = selection.match(/[\w.-]+@[\w.-]+\.\w+/);

    if (emailMatch) {
      showTrustDetails(emailMatch[0]);
    } else {
      // Try to get from focused email
      const focusedRow = document.querySelector('.zA.btb, [data-convid].selected');
      if (focusedRow) {
        const badge = focusedRow.querySelector('.mycelix-trust-badge');
        if (badge && badge.dataset.email) {
          showTrustDetails(badge.dataset.email);
        }
      }
    }
  }

  sendResponse({ success: true });
});

// ============================================================================
// Initialization
// ============================================================================

function init() {
  currentPlatform = detectPlatform();

  if (!currentPlatform) {
    console.log('Mycelix Mail: Unsupported platform');
    return;
  }

  console.log(`Mycelix Mail: Initializing for ${currentPlatform}`);

  // Wait for page to be ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', startObserver);
  } else {
    startObserver();
  }
}

init();
