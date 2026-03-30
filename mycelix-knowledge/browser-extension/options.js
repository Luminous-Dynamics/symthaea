// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Options page script for Mycelix Knowledge browser extension

const DEFAULT_SETTINGS = {
  holochainUrl: 'ws://localhost:8888',
  appId: 'mycelix-knowledge',
  autoHighlight: true,
  minCredibility: 0.5,
  theme: 'system',
  showBadge: true,
};

// Load settings from storage
async function loadSettings() {
  const settings = await chrome.storage.local.get(DEFAULT_SETTINGS);

  document.getElementById('holochainUrl').value = settings.holochainUrl;
  document.getElementById('appId').value = settings.appId;
  document.getElementById('autoHighlight').checked = settings.autoHighlight;
  document.getElementById('minCredibility').value = settings.minCredibility;
  document.getElementById('theme').value = settings.theme;
  document.getElementById('showBadge').checked = settings.showBadge;
}

// Save settings to storage
async function saveSettings() {
  const settings = {
    holochainUrl: document.getElementById('holochainUrl').value,
    appId: document.getElementById('appId').value,
    autoHighlight: document.getElementById('autoHighlight').checked,
    minCredibility: parseFloat(document.getElementById('minCredibility').value),
    theme: document.getElementById('theme').value,
    showBadge: document.getElementById('showBadge').checked,
  };

  await chrome.storage.local.set(settings);
  showStatus('Settings saved successfully!', 'success');

  // Notify background script of settings change
  chrome.runtime.sendMessage({ type: 'SETTINGS_UPDATED', settings });
}

// Reset to defaults
async function resetSettings() {
  await chrome.storage.local.set(DEFAULT_SETTINGS);
  await loadSettings();
  showStatus('Settings reset to defaults', 'success');
}

// Test connection to Holochain
async function testConnection() {
  const url = document.getElementById('holochainUrl').value;
  const statusDot = document.getElementById('statusDot');
  const statusText = document.getElementById('statusText');

  statusDot.className = 'status-dot connecting';
  statusText.textContent = 'Connecting...';

  try {
    const ws = new WebSocket(url);

    await new Promise((resolve, reject) => {
      ws.onopen = () => {
        ws.close();
        resolve();
      };
      ws.onerror = reject;
      setTimeout(() => reject(new Error('Timeout')), 5000);
    });

    statusDot.className = 'status-dot connected';
    statusText.textContent = 'Connected successfully!';
    showStatus('Connection successful!', 'success');
  } catch (error) {
    statusDot.className = 'status-dot disconnected';
    statusText.textContent = 'Connection failed';
    showStatus(`Connection failed: ${error.message}`, 'error');
  }
}

// Show status message
function showStatus(message, type) {
  const status = document.getElementById('status');
  status.textContent = message;
  status.className = `status ${type}`;

  setTimeout(() => {
    status.className = 'status';
  }, 3000);
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  loadSettings();

  document.getElementById('saveBtn').addEventListener('click', saveSettings);
  document.getElementById('resetBtn').addEventListener('click', resetSettings);
  document.getElementById('testBtn').addEventListener('click', testConnection);
});
