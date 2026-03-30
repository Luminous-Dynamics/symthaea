// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Mail Browser Extension - Background Service Worker
 *
 * Handles:
 * - Trust verification API calls
 * - Badge updates
 * - Notifications
 * - Context menu integration
 */

const API_BASE = 'https://api.mycelix.com';

// ============================================================================
// State Management
// ============================================================================

let state = {
  isAuthenticated: false,
  unreadCount: 0,
  trustCache: new Map(),
  settings: {
    showNotifications: true,
    autoCheckTrust: true,
    trustThreshold: 0.5,
  },
};

// Load state from storage
chrome.storage.local.get(['state', 'settings'], (result) => {
  if (result.state) {
    state = { ...state, ...result.state };
  }
  if (result.settings) {
    state.settings = { ...state.settings, ...result.settings };
  }
  updateBadge();
});

// ============================================================================
// Badge Management
// ============================================================================

function updateBadge() {
  const { unreadCount, isAuthenticated } = state;

  if (!isAuthenticated) {
    chrome.action.setBadgeText({ text: '' });
    chrome.action.setBadgeBackgroundColor({ color: '#9E9E9E' });
    return;
  }

  if (unreadCount > 0) {
    const text = unreadCount > 99 ? '99+' : unreadCount.toString();
    chrome.action.setBadgeText({ text });
    chrome.action.setBadgeBackgroundColor({ color: '#2196F3' });
  } else {
    chrome.action.setBadgeText({ text: '' });
  }
}

// ============================================================================
// Trust Verification
// ============================================================================

async function checkTrust(email) {
  // Check cache first
  const cached = state.trustCache.get(email);
  if (cached && Date.now() - cached.timestamp < 3600000) { // 1 hour cache
    return cached.data;
  }

  try {
    const token = await getAuthToken();
    if (!token) {
      return { error: 'Not authenticated' };
    }

    const response = await fetch(`${API_BASE}/trust/check`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify({ email }),
    });

    if (!response.ok) {
      throw new Error('Trust check failed');
    }

    const data = await response.json();

    // Cache result
    state.trustCache.set(email, {
      data,
      timestamp: Date.now(),
    });

    return data;
  } catch (error) {
    console.error('Trust check error:', error);
    return { error: error.message };
  }
}

async function batchCheckTrust(emails) {
  const uncached = emails.filter((email) => {
    const cached = state.trustCache.get(email);
    return !cached || Date.now() - cached.timestamp >= 3600000;
  });

  if (uncached.length === 0) {
    return emails.map((email) => state.trustCache.get(email).data);
  }

  try {
    const token = await getAuthToken();
    if (!token) return emails.map(() => ({ error: 'Not authenticated' }));

    const response = await fetch(`${API_BASE}/trust/batch-check`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify({ emails: uncached }),
    });

    if (!response.ok) throw new Error('Batch trust check failed');

    const results = await response.json();

    // Cache results
    results.forEach((result, index) => {
      state.trustCache.set(uncached[index], {
        data: result,
        timestamp: Date.now(),
      });
    });

    // Return all results including cached
    return emails.map((email) => state.trustCache.get(email)?.data || { error: 'Not found' });
  } catch (error) {
    console.error('Batch trust check error:', error);
    return emails.map(() => ({ error: error.message }));
  }
}

// ============================================================================
// Authentication
// ============================================================================

async function getAuthToken() {
  return new Promise((resolve) => {
    chrome.storage.local.get(['authToken'], (result) => {
      resolve(result.authToken);
    });
  });
}

async function authenticate() {
  return new Promise((resolve) => {
    chrome.identity.launchWebAuthFlow(
      {
        url: `${API_BASE}/auth/extension?redirect=${chrome.identity.getRedirectURL()}`,
        interactive: true,
      },
      (responseUrl) => {
        if (chrome.runtime.lastError) {
          resolve({ success: false, error: chrome.runtime.lastError.message });
          return;
        }

        try {
          const url = new URL(responseUrl);
          const token = url.searchParams.get('token');
          if (token) {
            chrome.storage.local.set({ authToken: token });
            state.isAuthenticated = true;
            updateBadge();
            resolve({ success: true });
          } else {
            resolve({ success: false, error: 'No token received' });
          }
        } catch (error) {
          resolve({ success: false, error: error.message });
        }
      }
    );
  });
}

async function logout() {
  chrome.storage.local.remove(['authToken']);
  state.isAuthenticated = false;
  state.trustCache.clear();
  updateBadge();
}

// ============================================================================
// Notifications
// ============================================================================

function showNotification(title, message, id) {
  if (!state.settings.showNotifications) return;

  chrome.notifications.create(id || `mycelix-${Date.now()}`, {
    type: 'basic',
    iconUrl: 'icons/icon128.png',
    title,
    message,
    priority: 1,
  });
}

chrome.notifications.onClicked.addListener((notificationId) => {
  if (notificationId.startsWith('mycelix-email-')) {
    const emailId = notificationId.replace('mycelix-email-', '');
    chrome.tabs.create({ url: `https://mail.mycelix.com/inbox/${emailId}` });
  } else {
    chrome.tabs.create({ url: 'https://mail.mycelix.com/inbox' });
  }
  chrome.notifications.clear(notificationId);
});

// ============================================================================
// Context Menu
// ============================================================================

chrome.runtime.onInstalled.addListener(() => {
  // Create context menu items
  chrome.contextMenus.create({
    id: 'check-trust',
    title: 'Check Trust Score',
    contexts: ['selection', 'link'],
  });

  chrome.contextMenus.create({
    id: 'compose-to',
    title: 'Compose Email with Mycelix',
    contexts: ['selection', 'link'],
  });

  chrome.contextMenus.create({
    id: 'save-contact',
    title: 'Save to Mycelix Contacts',
    contexts: ['selection'],
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  const text = info.selectionText || info.linkUrl;

  switch (info.menuItemId) {
    case 'check-trust': {
      // Extract email from selection or link
      const emailMatch = text.match(/[\w.-]+@[\w.-]+\.\w+/);
      if (emailMatch) {
        const result = await checkTrust(emailMatch[0]);
        showNotification(
          'Trust Check Result',
          result.error
            ? `Error: ${result.error}`
            : `Trust score for ${emailMatch[0]}: ${Math.round(result.score * 100)}%`
        );
      } else {
        showNotification('Trust Check', 'No email address found in selection');
      }
      break;
    }

    case 'compose-to': {
      const emailMatch = text.match(/[\w.-]+@[\w.-]+\.\w+/);
      if (emailMatch) {
        chrome.tabs.create({
          url: `https://mail.mycelix.com/compose?to=${encodeURIComponent(emailMatch[0])}`,
        });
      }
      break;
    }

    case 'save-contact': {
      const emailMatch = text.match(/[\w.-]+@[\w.-]+\.\w+/);
      if (emailMatch) {
        const token = await getAuthToken();
        if (token) {
          try {
            await fetch(`${API_BASE}/contacts`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`,
              },
              body: JSON.stringify({ email: emailMatch[0] }),
            });
            showNotification('Contact Saved', `Added ${emailMatch[0]} to your contacts`);
          } catch (error) {
            showNotification('Error', 'Failed to save contact');
          }
        }
      }
      break;
    }
  }
});

// ============================================================================
// Message Handling
// ============================================================================

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  (async () => {
    switch (message.action) {
      case 'checkTrust':
        sendResponse(await checkTrust(message.email));
        break;

      case 'batchCheckTrust':
        sendResponse(await batchCheckTrust(message.emails));
        break;

      case 'authenticate':
        sendResponse(await authenticate());
        break;

      case 'logout':
        await logout();
        sendResponse({ success: true });
        break;

      case 'getState':
        sendResponse({
          isAuthenticated: state.isAuthenticated,
          unreadCount: state.unreadCount,
          settings: state.settings,
        });
        break;

      case 'updateSettings':
        state.settings = { ...state.settings, ...message.settings };
        chrome.storage.local.set({ settings: state.settings });
        sendResponse({ success: true });
        break;

      case 'updateUnreadCount':
        state.unreadCount = message.count;
        updateBadge();
        sendResponse({ success: true });
        break;

      case 'showNotification':
        showNotification(message.title, message.message, message.id);
        sendResponse({ success: true });
        break;

      default:
        sendResponse({ error: 'Unknown action' });
    }
  })();

  return true; // Keep channel open for async response
});

// ============================================================================
// Keyboard Commands
// ============================================================================

chrome.commands.onCommand.addListener((command) => {
  switch (command) {
    case 'compose':
      chrome.tabs.create({ url: 'https://mail.mycelix.com/compose' });
      break;

    case 'quick-trust-check':
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        chrome.tabs.sendMessage(tabs[0].id, { action: 'showTrustOverlay' });
      });
      break;
  }
});

// ============================================================================
// Alarm for Periodic Sync
// ============================================================================

chrome.alarms.create('sync', { periodInMinutes: 5 });

chrome.alarms.onAlarm.addListener(async (alarm) => {
  if (alarm.name === 'sync' && state.isAuthenticated) {
    try {
      const token = await getAuthToken();
      const response = await fetch(`${API_BASE}/inbox/unread-count`, {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (response.ok) {
        const { count } = await response.json();
        const previousCount = state.unreadCount;
        state.unreadCount = count;
        updateBadge();

        // Show notification for new emails
        if (count > previousCount && state.settings.showNotifications) {
          showNotification(
            'New Email',
            `You have ${count - previousCount} new email(s)`
          );
        }
      }
    } catch (error) {
      console.error('Sync error:', error);
    }
  }
});

// ============================================================================
// Initialization
// ============================================================================

(async () => {
  const token = await getAuthToken();
  state.isAuthenticated = !!token;
  updateBadge();

  console.log('Mycelix Mail extension initialized');
})();
