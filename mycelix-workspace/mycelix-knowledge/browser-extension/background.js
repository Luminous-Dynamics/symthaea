// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Knowledge Browser Extension - Background Service Worker
 */

// ============================================================================
// Configuration
// ============================================================================

const DEFAULT_API_URL = 'http://localhost:4000/graphql';

// ============================================================================
// Context Menu Setup
// ============================================================================

chrome.runtime.onInstalled.addListener(() => {
  // Create context menu item
  chrome.contextMenus.create({
    id: 'mycelix-factcheck',
    title: 'Fact-check with Mycelix',
    contexts: ['selection'],
  });

  chrome.contextMenus.create({
    id: 'mycelix-submit',
    title: 'Submit as Claim',
    contexts: ['selection'],
  });

  chrome.contextMenus.create({
    id: 'mycelix-classify',
    title: 'Classify Epistemic Type',
    contexts: ['selection'],
  });

  console.log('Mycelix Knowledge extension installed');
});

// ============================================================================
// Context Menu Handler
// ============================================================================

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  const selectedText = info.selectionText;
  if (!selectedText) return;

  switch (info.menuItemId) {
    case 'mycelix-factcheck':
      await handleFactCheck(selectedText, tab);
      break;
    case 'mycelix-submit':
      await handleSubmitClaim(selectedText, tab);
      break;
    case 'mycelix-classify':
      await handleClassify(selectedText, tab);
      break;
  }
});

// ============================================================================
// Message Handler
// ============================================================================

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  switch (request.action) {
    case 'factCheck':
      handleFactCheckMessage(request.text).then(sendResponse);
      return true;

    case 'submitClaim':
      handleSubmitClaimMessage(request.claim).then(sendResponse);
      return true;

    case 'searchClaims':
      handleSearchMessage(request.query).then(sendResponse);
      return true;

    case 'getSettings':
      chrome.storage.sync.get(['apiUrl', 'autoHighlight', 'minConfidence'], sendResponse);
      return true;

    case 'saveSettings':
      chrome.storage.sync.set(request.settings, () => sendResponse({ success: true }));
      return true;
  }
});

// ============================================================================
// Fact Check Handler
// ============================================================================

async function handleFactCheck(text, tab) {
  // Show loading state
  chrome.tabs.sendMessage(tab.id, {
    action: 'showLoading',
    text: 'Fact-checking...',
  });

  try {
    const result = await factCheck(text);

    // Send result to content script
    chrome.tabs.sendMessage(tab.id, {
      action: 'showFactCheckResult',
      result,
      text,
    });
  } catch (error) {
    chrome.tabs.sendMessage(tab.id, {
      action: 'showError',
      error: error.message,
    });
  }
}

async function handleFactCheckMessage(text) {
  try {
    const result = await factCheck(text);
    return { success: true, result };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

async function factCheck(statement) {
  const settings = await chrome.storage.sync.get(['apiUrl']);
  const apiUrl = settings.apiUrl || DEFAULT_API_URL;

  const query = `
    query FactCheck($input: FactCheckInput!) {
      factCheck(input: $input) {
        verdict
        confidence
        explanation
        supportingClaims {
          id
          content
          credibility
        }
        contradictingClaims {
          id
          content
          credibility
        }
        sources
        checkedAt
      }
    }
  `;

  const response = await fetch(apiUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      variables: {
        input: {
          statement,
          minEpistemicLevel: 2,
          minConfidence: 0.6,
        },
      },
    }),
  });

  if (!response.ok) {
    throw new Error('Failed to connect to Mycelix Knowledge network');
  }

  const data = await response.json();

  if (data.errors) {
    throw new Error(data.errors[0].message);
  }

  return data.data.factCheck;
}

// ============================================================================
// Submit Claim Handler
// ============================================================================

async function handleSubmitClaim(text, tab) {
  // Show submission form in popup
  chrome.tabs.sendMessage(tab.id, {
    action: 'showSubmitForm',
    text,
  });
}

async function handleSubmitClaimMessage(claim) {
  try {
    const settings = await chrome.storage.sync.get(['apiUrl']);
    const apiUrl = settings.apiUrl || DEFAULT_API_URL;

    const mutation = `
      mutation CreateClaim($input: CreateClaimInput!) {
        createClaim(input: $input) {
          id
          content
          classification {
            empirical
            normative
            mythic
          }
        }
      }
    `;

    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: mutation,
        variables: {
          input: claim,
        },
      }),
    });

    const data = await response.json();

    if (data.errors) {
      throw new Error(data.errors[0].message);
    }

    return { success: true, claim: data.data.createClaim };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

// ============================================================================
// Search Handler
// ============================================================================

async function handleSearchMessage(query) {
  try {
    const settings = await chrome.storage.sync.get(['apiUrl']);
    const apiUrl = settings.apiUrl || DEFAULT_API_URL;

    const gqlQuery = `
      query Search($input: SearchInput!) {
        search(input: $input) {
          claims {
            id
            content
            classification {
              empirical
              normative
              mythic
            }
            credibility
            author
            tags
          }
          total
          took
        }
      }
    `;

    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: gqlQuery,
        variables: {
          input: {
            query,
            limit: 10,
          },
        },
      }),
    });

    const data = await response.json();

    if (data.errors) {
      throw new Error(data.errors[0].message);
    }

    return { success: true, results: data.data.search };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

// ============================================================================
// Classification Handler
// ============================================================================

async function handleClassify(text, tab) {
  // Simple client-side classification
  const classification = classifyText(text);

  chrome.tabs.sendMessage(tab.id, {
    action: 'showClassification',
    classification,
    text,
  });
}

function classifyText(text) {
  const lower = text.toLowerCase();

  const empiricalKeywords = [
    'data', 'study', 'research', 'evidence', 'percent', '%',
    'measured', 'observed', 'experiment', 'statistics', 'survey',
    'found', 'discovered', 'shows', 'indicates', 'demonstrates',
  ];

  const normativeKeywords = [
    'should', 'ought', 'must', 'right', 'wrong', 'good', 'bad',
    'better', 'worse', 'fair', 'unfair', 'just', 'unjust',
    'moral', 'ethical', 'deserve', 'obligation', 'duty',
  ];

  const mythicKeywords = [
    'believe', 'meaning', 'purpose', 'destiny', 'spirit', 'soul',
    'tradition', 'sacred', 'divine', 'story', 'narrative', 'myth',
    'transcend', 'eternal', 'infinite', 'wisdom', 'truth',
  ];

  const empiricalScore = empiricalKeywords.filter((k) => lower.includes(k)).length;
  const normativeScore = normativeKeywords.filter((k) => lower.includes(k)).length;
  const mythicScore = mythicKeywords.filter((k) => lower.includes(k)).length;

  const total = empiricalScore + normativeScore + mythicScore || 1;

  return {
    empirical: Math.min(1, empiricalScore / total + 0.1),
    normative: Math.min(1, normativeScore / total + 0.1),
    mythic: Math.min(1, mythicScore / total + 0.1),
    dominant:
      empiricalScore >= normativeScore && empiricalScore >= mythicScore
        ? 'empirical'
        : normativeScore >= mythicScore
          ? 'normative'
          : 'mythic',
  };
}
