// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Mail Browser Extension - Popup Script
 */

const API_BASE = 'https://mail.mycelix.com';

// ============================================================================
// DOM Elements
// ============================================================================

const elements = {
  loading: document.getElementById('loading'),
  authSection: document.getElementById('auth-section'),
  mainSection: document.getElementById('main-section'),
  loginBtn: document.getElementById('login-btn'),
  logoutLink: document.getElementById('logout-link'),
  settingsLink: document.getElementById('settings-link'),
  composeBtn: document.getElementById('compose-btn'),
  inboxBtn: document.getElementById('inbox-btn'),
  unreadCount: document.getElementById('unread-count'),
  trustAvg: document.getElementById('trust-avg'),
  trustEmail: document.getElementById('trust-email'),
  trustCheckBtn: document.getElementById('trust-check-btn'),
  trustResult: document.getElementById('trust-result'),
  trustScoreFill: document.getElementById('trust-score-fill'),
  trustScoreValue: document.getElementById('trust-score-value'),
  recentEmails: document.getElementById('recent-emails'),
};

// ============================================================================
// State Management
// ============================================================================

let state = {
  isAuthenticated: false,
  unreadCount: 0,
};

async function loadState() {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({ action: 'getState' }, (response) => {
      state = { ...state, ...response };
      resolve(state);
    });
  });
}

// ============================================================================
// UI Updates
// ============================================================================

function showLoading() {
  elements.loading.classList.remove('hidden');
  elements.authSection.classList.add('hidden');
  elements.mainSection.classList.add('hidden');
}

function showAuth() {
  elements.loading.classList.add('hidden');
  elements.authSection.classList.remove('hidden');
  elements.mainSection.classList.add('hidden');
  elements.logoutLink.classList.add('hidden');
}

function showMain() {
  elements.loading.classList.add('hidden');
  elements.authSection.classList.add('hidden');
  elements.mainSection.classList.remove('hidden');
  elements.logoutLink.classList.remove('hidden');
}

function updateStats() {
  elements.unreadCount.textContent = state.unreadCount || 0;
}

function renderRecentEmails(emails) {
  if (!emails || emails.length === 0) {
    elements.recentEmails.innerHTML = `
      <div class="recent-item">
        <div class="recent-info">
          <div class="recent-from">No recent emails</div>
        </div>
      </div>
    `;
    return;
  }

  elements.recentEmails.innerHTML = emails
    .slice(0, 5)
    .map((email) => {
      const initial = (email.from || 'U')[0].toUpperCase();
      const time = formatTime(email.receivedAt);

      return `
      <div class="recent-item" data-id="${email.id}">
        <div class="recent-avatar">${initial}</div>
        <div class="recent-info">
          <div class="recent-from">${escapeHtml(email.from)}</div>
          <div class="recent-subject">${escapeHtml(email.subject)}</div>
        </div>
        <div class="recent-time">${time}</div>
      </div>
    `;
    })
    .join('');

  // Add click handlers
  elements.recentEmails.querySelectorAll('.recent-item').forEach((item) => {
    item.addEventListener('click', () => {
      const id = item.dataset.id;
      chrome.tabs.create({ url: `${API_BASE}/inbox/${id}` });
    });
  });
}

// ============================================================================
// Trust Check
// ============================================================================

async function checkTrust() {
  const email = elements.trustEmail.value.trim();
  if (!email) return;

  elements.trustCheckBtn.disabled = true;
  elements.trustCheckBtn.textContent = '...';

  try {
    const result = await new Promise((resolve) => {
      chrome.runtime.sendMessage({ action: 'checkTrust', email }, resolve);
    });

    const score = result.score ?? 0;
    const percentage = Math.round(score * 100);

    elements.trustResult.classList.add('visible');
    elements.trustScoreValue.textContent = `${percentage}%`;
    elements.trustScoreFill.style.width = `${percentage}%`;

    // Color based on score
    let color;
    if (score >= 0.8) color = '#4CAF50';
    else if (score >= 0.5) color = '#FFC107';
    else if (score >= 0.2) color = '#FF9800';
    else color = '#F44336';

    elements.trustScoreFill.style.backgroundColor = color;
  } catch (error) {
    console.error('Trust check failed:', error);
  } finally {
    elements.trustCheckBtn.disabled = false;
    elements.trustCheckBtn.textContent = 'Check';
  }
}

// ============================================================================
// Authentication
// ============================================================================

async function login() {
  elements.loginBtn.disabled = true;
  elements.loginBtn.textContent = 'Signing in...';

  try {
    const result = await new Promise((resolve) => {
      chrome.runtime.sendMessage({ action: 'authenticate' }, resolve);
    });

    if (result.success) {
      state.isAuthenticated = true;
      await init();
    } else {
      alert('Sign in failed: ' + (result.error || 'Unknown error'));
    }
  } catch (error) {
    console.error('Login failed:', error);
    alert('Sign in failed');
  } finally {
    elements.loginBtn.disabled = false;
    elements.loginBtn.textContent = 'Sign In';
  }
}

async function logout() {
  await new Promise((resolve) => {
    chrome.runtime.sendMessage({ action: 'logout' }, resolve);
  });
  state.isAuthenticated = false;
  showAuth();
}

// ============================================================================
// Data Loading
// ============================================================================

async function loadRecentEmails() {
  try {
    const response = await fetch(`${API_BASE}/api/emails/recent?limit=5`, {
      headers: await getAuthHeaders(),
    });

    if (response.ok) {
      const data = await response.json();
      renderRecentEmails(data.emails);
    }
  } catch (error) {
    console.error('Failed to load recent emails:', error);
    elements.recentEmails.innerHTML = `
      <div class="recent-item">
        <div class="recent-info">
          <div class="recent-subject">Unable to load emails</div>
        </div>
      </div>
    `;
  }
}

async function loadTrustAverage() {
  try {
    const response = await fetch(`${API_BASE}/api/trust/average`, {
      headers: await getAuthHeaders(),
    });

    if (response.ok) {
      const data = await response.json();
      elements.trustAvg.textContent = `${Math.round(data.average * 100)}%`;
    }
  } catch (error) {
    console.error('Failed to load trust average:', error);
  }
}

async function getAuthHeaders() {
  const token = await new Promise((resolve) => {
    chrome.storage.local.get(['authToken'], (result) => {
      resolve(result.authToken);
    });
  });

  return {
    Authorization: `Bearer ${token}`,
    'Content-Type': 'application/json',
  };
}

// ============================================================================
// Utilities
// ============================================================================

function formatTime(dateString) {
  if (!dateString) return '';

  const date = new Date(dateString);
  const now = new Date();
  const diff = now - date;

  if (diff < 60000) return 'now';
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h`;
  if (diff < 604800000) return `${Math.floor(diff / 86400000)}d`;

  return date.toLocaleDateString();
}

function escapeHtml(text) {
  if (!text) return '';
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// ============================================================================
// Event Handlers
// ============================================================================

elements.loginBtn.addEventListener('click', login);
elements.logoutLink.addEventListener('click', (e) => {
  e.preventDefault();
  logout();
});

elements.settingsLink.addEventListener('click', (e) => {
  e.preventDefault();
  chrome.runtime.openOptionsPage();
});

elements.composeBtn.addEventListener('click', () => {
  chrome.tabs.create({ url: `${API_BASE}/compose` });
});

elements.inboxBtn.addEventListener('click', () => {
  chrome.tabs.create({ url: `${API_BASE}/inbox` });
});

elements.trustCheckBtn.addEventListener('click', checkTrust);
elements.trustEmail.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') checkTrust();
});

// ============================================================================
// Initialization
// ============================================================================

async function init() {
  showLoading();

  await loadState();

  if (state.isAuthenticated) {
    showMain();
    updateStats();
    loadRecentEmails();
    loadTrustAverage();
  } else {
    showAuth();
  }
}

init();
