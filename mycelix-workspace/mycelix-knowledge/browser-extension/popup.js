// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Knowledge Browser Extension - Popup Script
 */

// ============================================================================
// State
// ============================================================================

let isConnected = false;
const recentChecks = [];

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
  // Check connection status
  await checkConnectionStatus();

  // Setup tab navigation
  setupTabs();

  // Setup search
  setupSearch();

  // Setup actions
  setupActions();

  // Load stats
  loadStats();

  // Load history
  loadHistory();
});

// ============================================================================
// Connection Status
// ============================================================================

async function checkConnectionStatus() {
  const statusDot = document.getElementById('status-dot');
  const statusText = document.getElementById('status-text');

  try {
    const settings = await chrome.runtime.sendMessage({ action: 'getSettings' });
    const apiUrl = settings.apiUrl || 'http://localhost:4000/graphql';

    // Try a simple query
    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: '{ __typename }',
      }),
    });

    if (response.ok) {
      isConnected = true;
      statusDot.classList.remove('disconnected');
      statusText.textContent = 'Connected to Mycelix Network';
    } else {
      throw new Error('Connection failed');
    }
  } catch (error) {
    isConnected = false;
    statusDot.classList.add('disconnected');
    statusText.textContent = 'Not connected - Check settings';
  }
}

// ============================================================================
// Tab Navigation
// ============================================================================

function setupTabs() {
  const tabs = document.querySelectorAll('.tab');
  const panels = document.querySelectorAll('.panel');

  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      const targetId = `panel-${tab.dataset.tab}`;

      tabs.forEach((t) => t.classList.remove('active'));
      panels.forEach((p) => p.classList.remove('active'));

      tab.classList.add('active');
      document.getElementById(targetId).classList.add('active');
    });
  });
}

// ============================================================================
// Search
// ============================================================================

function setupSearch() {
  const searchInput = document.getElementById('search-input');
  const searchBtn = document.getElementById('search-btn');
  const resultsContainer = document.getElementById('search-results');

  const performSearch = async () => {
    const query = searchInput.value.trim();
    if (!query) return;

    resultsContainer.innerHTML = '<div class="empty-state"><div class="icon">⏳</div><p>Searching...</p></div>';

    const response = await chrome.runtime.sendMessage({
      action: 'searchClaims',
      query,
    });

    if (response.success && response.results.claims.length > 0) {
      resultsContainer.innerHTML = response.results.claims
        .map(
          (claim) => `
          <div class="result-item" data-id="${claim.id}">
            <div class="result-content">${escapeHtml(claim.content)}</div>
            <div class="result-meta">
              ${claim.tags.map((t) => `<span class="result-tag">${escapeHtml(t)}</span>`).join('')}
              ${claim.credibility ? `<span class="result-score">${Math.round(claim.credibility * 100)}% credible</span>` : ''}
            </div>
          </div>
        `
        )
        .join('');
    } else if (response.success) {
      resultsContainer.innerHTML = '<div class="empty-state"><div class="icon">🔍</div><p>No results found</p></div>';
    } else {
      resultsContainer.innerHTML = `<div class="empty-state"><div class="icon">❌</div><p>${response.error}</p></div>`;
    }
  };

  searchBtn.addEventListener('click', performSearch);
  searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') performSearch();
  });
}

// ============================================================================
// Actions
// ============================================================================

function setupActions() {
  document.getElementById('action-factcheck').addEventListener('click', async () => {
    // Get selected text from active tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    chrome.tabs.sendMessage(tab.id, { action: 'getSelection' }, async (selection) => {
      if (selection) {
        const response = await chrome.runtime.sendMessage({
          action: 'factCheck',
          text: selection,
        });

        if (response.success) {
          addToHistory(selection, response.result);
        }
      } else {
        alert('Please select some text on the page first');
      }
    });
  });

  document.getElementById('action-submit').addEventListener('click', async () => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    chrome.tabs.sendMessage(tab.id, { action: 'triggerSubmit' });
    window.close();
  });

  document.getElementById('action-classify').addEventListener('click', async () => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    chrome.tabs.sendMessage(tab.id, { action: 'triggerClassify' });
    window.close();
  });

  document.getElementById('action-settings').addEventListener('click', () => {
    chrome.runtime.openOptionsPage();
  });
}

// ============================================================================
// Stats
// ============================================================================

async function loadStats() {
  const stats = await chrome.storage.local.get(['totalClaims', 'totalChecks', 'totalSubmitted']);

  document.getElementById('stat-claims').textContent = formatNumber(stats.totalClaims || 0);
  document.getElementById('stat-checks').textContent = formatNumber(stats.totalChecks || 0);
  document.getElementById('stat-submitted').textContent = formatNumber(stats.totalSubmitted || 0);
}

function formatNumber(num) {
  if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
  if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
  return num.toString();
}

// ============================================================================
// History
// ============================================================================

async function loadHistory() {
  const { factCheckHistory = [] } = await chrome.storage.local.get(['factCheckHistory']);
  renderHistory(factCheckHistory);
}

function addToHistory(text, result) {
  chrome.storage.local.get(['factCheckHistory'], ({ factCheckHistory = [] }) => {
    factCheckHistory.unshift({
      text: text.substring(0, 100),
      verdict: result.verdict,
      timestamp: Date.now(),
    });

    // Keep only last 20
    if (factCheckHistory.length > 20) {
      factCheckHistory = factCheckHistory.slice(0, 20);
    }

    chrome.storage.local.set({ factCheckHistory });
    renderHistory(factCheckHistory);
  });
}

function renderHistory(history) {
  const container = document.getElementById('recent-list');

  if (history.length === 0) {
    container.innerHTML = '<div class="empty-state"><div class="icon">📋</div><p>No recent fact checks</p></div>';
    return;
  }

  const verdictIcons = {
    TRUE: '✓',
    MOSTLY_TRUE: '✓',
    MIXED: '?',
    MOSTLY_FALSE: '✗',
    FALSE: '✗',
    UNVERIFIABLE: '?',
    INSUFFICIENT_EVIDENCE: '?',
  };

  const verdictClasses = {
    TRUE: 'true',
    MOSTLY_TRUE: 'true',
    MIXED: 'mixed',
    MOSTLY_FALSE: 'false',
    FALSE: 'false',
    UNVERIFIABLE: 'unknown',
    INSUFFICIENT_EVIDENCE: 'unknown',
  };

  container.innerHTML = history
    .map(
      (item) => `
      <div class="recent-item">
        <div class="recent-verdict ${verdictClasses[item.verdict] || 'unknown'}">
          ${verdictIcons[item.verdict] || '?'}
        </div>
        <div class="recent-text">${escapeHtml(item.text)}</div>
      </div>
    `
    )
    .join('');
}

// ============================================================================
// Utilities
// ============================================================================

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
