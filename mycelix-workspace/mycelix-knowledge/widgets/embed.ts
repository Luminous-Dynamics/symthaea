// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Knowledge - Embeddable Widgets
 *
 * Standalone widgets that can be embedded in any website
 *
 * Usage:
 * <script src="https://cdn.mycelix.net/widgets/embed.js"></script>
 * <div data-mycelix-widget="fact-check" data-claim="Your claim here"></div>
 */

// ============================================================================
// Types
// ============================================================================

interface WidgetConfig {
  apiUrl?: string;
  theme?: 'light' | 'dark' | 'auto';
  locale?: string;
  onReady?: () => void;
  onError?: (error: Error) => void;
}

interface FactCheckWidgetConfig extends WidgetConfig {
  claim: string;
  showSources?: boolean;
  showRelated?: boolean;
  compact?: boolean;
}

interface ClaimCardWidgetConfig extends WidgetConfig {
  claimId: string;
  showCredibility?: boolean;
  showAuthor?: boolean;
}

interface SearchWidgetConfig extends WidgetConfig {
  placeholder?: string;
  maxResults?: number;
  tags?: string[];
}

interface CredibilityBadgeConfig extends WidgetConfig {
  claimId: string;
  size?: 'small' | 'medium' | 'large';
}

// ============================================================================
// Global Configuration
// ============================================================================

const DEFAULT_API_URL = 'https://api.mycelix.net/graphql';

let globalConfig: WidgetConfig = {
  apiUrl: DEFAULT_API_URL,
  theme: 'auto',
  locale: 'en',
};

// ============================================================================
// Widget Registry
// ============================================================================

const widgets: Map<string, HTMLElement> = new Map();

// ============================================================================
// Initialization
// ============================================================================

function init(config?: WidgetConfig): void {
  globalConfig = { ...globalConfig, ...config };

  // Inject styles
  injectStyles();

  // Auto-discover widgets
  discoverWidgets();

  // Setup mutation observer for dynamic widgets
  observeDOM();

  // Call ready callback
  globalConfig.onReady?.();
}

function discoverWidgets(): void {
  document.querySelectorAll('[data-mycelix-widget]').forEach((el) => {
    const element = el as HTMLElement;
    const widgetType = element.dataset.mycelixWidget;

    if (widgets.has(element.id)) return;

    renderWidget(element, widgetType || '');
  });
}

function observeDOM(): void {
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      mutation.addedNodes.forEach((node) => {
        if (node instanceof HTMLElement) {
          if (node.dataset.mycelixWidget) {
            renderWidget(node, node.dataset.mycelixWidget);
          }
          node.querySelectorAll('[data-mycelix-widget]').forEach((el) => {
            const element = el as HTMLElement;
            renderWidget(element, element.dataset.mycelixWidget || '');
          });
        }
      });
    });
  });

  observer.observe(document.body, { childList: true, subtree: true });
}

// ============================================================================
// Widget Renderers
// ============================================================================

function renderWidget(container: HTMLElement, type: string): void {
  const id = container.id || `mycelix-${Math.random().toString(36).substring(7)}`;
  container.id = id;
  widgets.set(id, container);

  switch (type) {
    case 'fact-check':
      renderFactCheckWidget(container);
      break;
    case 'claim-card':
      renderClaimCardWidget(container);
      break;
    case 'search':
      renderSearchWidget(container);
      break;
    case 'credibility-badge':
      renderCredibilityBadge(container);
      break;
    case 'epistemic-meter':
      renderEpistemicMeter(container);
      break;
    default:
      console.warn(`Unknown Mycelix widget type: ${type}`);
  }
}

// ============================================================================
// Fact Check Widget
// ============================================================================

async function renderFactCheckWidget(container: HTMLElement): Promise<void> {
  const claim = container.dataset.claim || '';
  const showSources = container.dataset.showSources !== 'false';
  const showRelated = container.dataset.showRelated !== 'false';
  const compact = container.dataset.compact === 'true';

  container.innerHTML = `
    <div class="mycelix-widget mycelix-fact-check ${compact ? 'mycelix-compact' : ''}">
      <div class="mycelix-loading">
        <div class="mycelix-spinner"></div>
        <span>Checking...</span>
      </div>
    </div>
  `;

  try {
    const result = await factCheck(claim);

    const verdictColors: Record<string, string> = {
      TRUE: '#22c55e',
      MOSTLY_TRUE: '#84cc16',
      MIXED: '#eab308',
      MOSTLY_FALSE: '#f97316',
      FALSE: '#ef4444',
      UNVERIFIABLE: '#6b7280',
      INSUFFICIENT_EVIDENCE: '#8b5cf6',
    };

    container.innerHTML = `
      <div class="mycelix-widget mycelix-fact-check ${compact ? 'mycelix-compact' : ''}">
        <div class="mycelix-header">
          <span class="mycelix-logo">⚡</span>
          <span class="mycelix-title">Fact Check</span>
        </div>

        <div class="mycelix-claim">"${escapeHtml(claim)}"</div>

        <div class="mycelix-verdict" style="background: ${verdictColors[result.verdict]}">
          ${result.verdict.replace('_', ' ')}
        </div>

        <div class="mycelix-confidence">
          <div class="mycelix-confidence-label">
            <span>Confidence</span>
            <span>${Math.round(result.confidence * 100)}%</span>
          </div>
          <div class="mycelix-progress">
            <div class="mycelix-progress-fill" style="width: ${result.confidence * 100}%"></div>
          </div>
        </div>

        ${!compact ? `
          <div class="mycelix-explanation">${escapeHtml(result.explanation)}</div>

          ${showSources && result.sources.length > 0 ? `
            <div class="mycelix-sources">
              <strong>Sources:</strong>
              ${result.sources.slice(0, 3).map((s) => `<span class="mycelix-source">${escapeHtml(s)}</span>`).join('')}
            </div>
          ` : ''}

          ${showRelated && (result.supportingClaims.length > 0 || result.contradictingClaims.length > 0) ? `
            <div class="mycelix-related">
              ${result.supportingClaims.length > 0 ? `<span class="mycelix-supporting">+${result.supportingClaims.length} supporting</span>` : ''}
              ${result.contradictingClaims.length > 0 ? `<span class="mycelix-contradicting">-${result.contradictingClaims.length} contradicting</span>` : ''}
            </div>
          ` : ''}
        ` : ''}

        <a href="https://mycelix.net" target="_blank" class="mycelix-powered">
          Powered by Mycelix
        </a>
      </div>
    `;
  } catch (error) {
    container.innerHTML = `
      <div class="mycelix-widget mycelix-fact-check mycelix-error">
        <div class="mycelix-error-message">
          Unable to fact-check. <a href="https://mycelix.net">Learn more</a>
        </div>
      </div>
    `;
    globalConfig.onError?.(error as Error);
  }
}

// ============================================================================
// Claim Card Widget
// ============================================================================

async function renderClaimCardWidget(container: HTMLElement): Promise<void> {
  const claimId = container.dataset.claimId || '';
  const showCredibility = container.dataset.showCredibility !== 'false';
  const showAuthor = container.dataset.showAuthor !== 'false';

  container.innerHTML = `
    <div class="mycelix-widget mycelix-claim-card">
      <div class="mycelix-loading">
        <div class="mycelix-spinner"></div>
      </div>
    </div>
  `;

  try {
    const claim = await getClaim(claimId);

    container.innerHTML = `
      <div class="mycelix-widget mycelix-claim-card">
        <div class="mycelix-claim-content">${escapeHtml(claim.content)}</div>

        <div class="mycelix-classification">
          <div class="mycelix-class-bar">
            <div class="mycelix-class-empirical" style="width: ${claim.classification.empirical * 100}%"></div>
            <div class="mycelix-class-normative" style="width: ${claim.classification.normative * 100}%"></div>
            <div class="mycelix-class-mythic" style="width: ${claim.classification.mythic * 100}%"></div>
          </div>
          <div class="mycelix-class-labels">
            <span style="color: #ef4444">E ${Math.round(claim.classification.empirical * 100)}%</span>
            <span style="color: #22c55e">N ${Math.round(claim.classification.normative * 100)}%</span>
            <span style="color: #3b82f6">M ${Math.round(claim.classification.mythic * 100)}%</span>
          </div>
        </div>

        ${showCredibility && claim.credibility ? `
          <div class="mycelix-credibility">
            Credibility: <strong>${Math.round(claim.credibility.score * 100)}%</strong>
          </div>
        ` : ''}

        ${showAuthor ? `
          <div class="mycelix-author">By: ${claim.author.did.substring(0, 20)}...</div>
        ` : ''}

        <div class="mycelix-tags">
          ${claim.tags.map((t) => `<span class="mycelix-tag">${escapeHtml(t)}</span>`).join('')}
        </div>
      </div>
    `;
  } catch (error) {
    container.innerHTML = `
      <div class="mycelix-widget mycelix-claim-card mycelix-error">
        Claim not found
      </div>
    `;
  }
}

// ============================================================================
// Search Widget
// ============================================================================

function renderSearchWidget(container: HTMLElement): void {
  const placeholder = container.dataset.placeholder || 'Search the knowledge graph...';
  const maxResults = parseInt(container.dataset.maxResults || '5', 10);

  container.innerHTML = `
    <div class="mycelix-widget mycelix-search">
      <div class="mycelix-search-box">
        <input type="text" class="mycelix-search-input" placeholder="${escapeHtml(placeholder)}">
        <button class="mycelix-search-btn">🔍</button>
      </div>
      <div class="mycelix-search-results"></div>
    </div>
  `;

  const input = container.querySelector('.mycelix-search-input') as HTMLInputElement;
  const btn = container.querySelector('.mycelix-search-btn') as HTMLButtonElement;
  const results = container.querySelector('.mycelix-search-results') as HTMLElement;

  const doSearch = async () => {
    const query = input.value.trim();
    if (!query) return;

    results.innerHTML = '<div class="mycelix-loading"><div class="mycelix-spinner"></div></div>';

    try {
      const searchResults = await search(query, maxResults);

      if (searchResults.length === 0) {
        results.innerHTML = '<div class="mycelix-empty">No results found</div>';
        return;
      }

      results.innerHTML = searchResults
        .map(
          (claim) => `
          <div class="mycelix-search-result">
            <div class="mycelix-result-content">${escapeHtml(claim.content.substring(0, 100))}...</div>
            <div class="mycelix-result-meta">
              ${claim.credibility ? `<span class="mycelix-result-score">${Math.round(claim.credibility * 100)}%</span>` : ''}
              ${claim.tags.slice(0, 2).map((t) => `<span class="mycelix-result-tag">${escapeHtml(t)}</span>`).join('')}
            </div>
          </div>
        `
        )
        .join('');
    } catch (error) {
      results.innerHTML = '<div class="mycelix-error">Search failed</div>';
    }
  };

  btn.addEventListener('click', doSearch);
  input.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') doSearch();
  });
}

// ============================================================================
// Credibility Badge Widget
// ============================================================================

async function renderCredibilityBadge(container: HTMLElement): Promise<void> {
  const claimId = container.dataset.claimId || '';
  const size = (container.dataset.size || 'medium') as 'small' | 'medium' | 'large';

  const sizes = {
    small: { width: 60, fontSize: 12 },
    medium: { width: 80, fontSize: 14 },
    large: { width: 100, fontSize: 16 },
  };

  const { width, fontSize } = sizes[size];

  try {
    const credibility = await getCredibility(claimId);
    const score = Math.round(credibility.score * 100);

    const color =
      score >= 70 ? '#22c55e' :
      score >= 50 ? '#eab308' :
      '#ef4444';

    container.innerHTML = `
      <div class="mycelix-badge" style="width: ${width}px">
        <svg viewBox="0 0 36 36" class="mycelix-badge-ring">
          <path
            d="M18 2.0845
              a 15.9155 15.9155 0 0 1 0 31.831
              a 15.9155 15.9155 0 0 1 0 -31.831"
            fill="none"
            stroke="#e2e8f0"
            stroke-width="3"
          />
          <path
            d="M18 2.0845
              a 15.9155 15.9155 0 0 1 0 31.831
              a 15.9155 15.9155 0 0 1 0 -31.831"
            fill="none"
            stroke="${color}"
            stroke-width="3"
            stroke-dasharray="${score}, 100"
          />
        </svg>
        <div class="mycelix-badge-score" style="font-size: ${fontSize}px">${score}%</div>
        <div class="mycelix-badge-label">Credibility</div>
      </div>
    `;
  } catch {
    container.innerHTML = `<div class="mycelix-badge mycelix-badge-error">-</div>`;
  }
}

// ============================================================================
// Epistemic Meter Widget
// ============================================================================

async function renderEpistemicMeter(container: HTMLElement): Promise<void> {
  const claimId = container.dataset.claimId;
  const empirical = parseFloat(container.dataset.empirical || '0.33');
  const normative = parseFloat(container.dataset.normative || '0.33');
  const mythic = parseFloat(container.dataset.mythic || '0.34');

  let classification = { empirical, normative, mythic };

  if (claimId) {
    try {
      const claim = await getClaim(claimId);
      classification = claim.classification;
    } catch {
      // Use defaults
    }
  }

  container.innerHTML = `
    <div class="mycelix-meter">
      <div class="mycelix-meter-bar">
        <div class="mycelix-meter-segment mycelix-meter-empirical" style="width: ${classification.empirical * 100}%">
          <span>E</span>
        </div>
        <div class="mycelix-meter-segment mycelix-meter-normative" style="width: ${classification.normative * 100}%">
          <span>N</span>
        </div>
        <div class="mycelix-meter-segment mycelix-meter-mythic" style="width: ${classification.mythic * 100}%">
          <span>M</span>
        </div>
      </div>
      <div class="mycelix-meter-labels">
        <span>Empirical ${Math.round(classification.empirical * 100)}%</span>
        <span>Normative ${Math.round(classification.normative * 100)}%</span>
        <span>Mythic ${Math.round(classification.mythic * 100)}%</span>
      </div>
    </div>
  `;
}

// ============================================================================
// API Functions
// ============================================================================

async function factCheck(statement: string): Promise<FactCheckResult> {
  const response = await fetch(globalConfig.apiUrl!, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: `
        query FactCheck($input: FactCheckInput!) {
          factCheck(input: $input) {
            verdict
            confidence
            explanation
            supportingClaims { id content }
            contradictingClaims { id content }
            sources
          }
        }
      `,
      variables: { input: { statement, minConfidence: 0.6 } },
    }),
  });

  const data = await response.json();
  if (data.errors) throw new Error(data.errors[0].message);
  return data.data.factCheck;
}

async function getClaim(id: string): Promise<Claim> {
  const response = await fetch(globalConfig.apiUrl!, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: `
        query GetClaim($id: ID!) {
          claim(id: $id) {
            id
            content
            classification { empirical normative mythic }
            author { did }
            tags
            credibility { score confidence }
          }
        }
      `,
      variables: { id },
    }),
  });

  const data = await response.json();
  if (data.errors) throw new Error(data.errors[0].message);
  return data.data.claim;
}

async function search(query: string, limit: number): Promise<ClaimSummary[]> {
  const response = await fetch(globalConfig.apiUrl!, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: `
        query Search($input: SearchInput!) {
          search(input: $input) {
            claims { id content credibility tags }
          }
        }
      `,
      variables: { input: { query, limit } },
    }),
  });

  const data = await response.json();
  if (data.errors) throw new Error(data.errors[0].message);
  return data.data.search.claims;
}

async function getCredibility(claimId: string): Promise<{ score: number; confidence: number }> {
  const response = await fetch(globalConfig.apiUrl!, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: `
        query GetCredibility($id: ID!) {
          claim(id: $id) {
            credibility { score confidence }
          }
        }
      `,
      variables: { id: claimId },
    }),
  });

  const data = await response.json();
  if (data.errors) throw new Error(data.errors[0].message);
  return data.data.claim.credibility;
}

// ============================================================================
// Styles
// ============================================================================

function injectStyles(): void {
  if (document.getElementById('mycelix-widget-styles')) return;

  const style = document.createElement('style');
  style.id = 'mycelix-widget-styles';
  style.textContent = `
    .mycelix-widget {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: white;
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      padding: 16px;
      max-width: 400px;
    }

    .mycelix-loading {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 20px;
      color: #64748b;
    }

    .mycelix-spinner {
      width: 20px;
      height: 20px;
      border: 2px solid #e2e8f0;
      border-top-color: #3b82f6;
      border-radius: 50%;
      animation: mycelix-spin 0.8s linear infinite;
    }

    @keyframes mycelix-spin {
      to { transform: rotate(360deg); }
    }

    .mycelix-header {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 12px;
    }

    .mycelix-logo {
      font-size: 20px;
    }

    .mycelix-title {
      font-weight: 600;
      color: #1e293b;
    }

    .mycelix-claim {
      background: #f8fafc;
      padding: 12px;
      border-radius: 8px;
      border-left: 3px solid #3b82f6;
      margin-bottom: 16px;
      color: #334155;
      line-height: 1.5;
    }

    .mycelix-verdict {
      display: inline-block;
      padding: 8px 16px;
      border-radius: 6px;
      color: white;
      font-weight: 600;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 16px;
    }

    .mycelix-confidence {
      margin-bottom: 16px;
    }

    .mycelix-confidence-label {
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      color: #64748b;
      margin-bottom: 4px;
    }

    .mycelix-progress {
      height: 6px;
      background: #e2e8f0;
      border-radius: 3px;
      overflow: hidden;
    }

    .mycelix-progress-fill {
      height: 100%;
      background: linear-gradient(90deg, #3b82f6, #8b5cf6);
      border-radius: 3px;
    }

    .mycelix-explanation {
      color: #475569;
      font-size: 14px;
      line-height: 1.6;
      margin-bottom: 12px;
    }

    .mycelix-sources {
      font-size: 12px;
      color: #64748b;
      margin-bottom: 12px;
    }

    .mycelix-source {
      display: inline-block;
      background: #f1f5f9;
      padding: 2px 8px;
      border-radius: 4px;
      margin: 2px;
    }

    .mycelix-related {
      display: flex;
      gap: 12px;
      font-size: 12px;
      margin-bottom: 12px;
    }

    .mycelix-supporting { color: #22c55e; }
    .mycelix-contradicting { color: #ef4444; }

    .mycelix-powered {
      display: block;
      text-align: center;
      font-size: 11px;
      color: #94a3b8;
      text-decoration: none;
      padding-top: 12px;
      border-top: 1px solid #e2e8f0;
    }

    .mycelix-powered:hover {
      color: #3b82f6;
    }

    .mycelix-compact {
      padding: 12px;
    }

    .mycelix-compact .mycelix-claim {
      font-size: 13px;
      padding: 8px;
      margin-bottom: 12px;
    }

    .mycelix-compact .mycelix-verdict {
      padding: 4px 12px;
      font-size: 12px;
    }

    /* Search Widget */
    .mycelix-search-box {
      display: flex;
      gap: 8px;
      margin-bottom: 12px;
    }

    .mycelix-search-input {
      flex: 1;
      padding: 8px 12px;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      font-size: 14px;
    }

    .mycelix-search-btn {
      padding: 8px 12px;
      background: #3b82f6;
      color: white;
      border: none;
      border-radius: 6px;
      cursor: pointer;
    }

    .mycelix-search-result {
      padding: 10px;
      background: #f8fafc;
      border-radius: 6px;
      margin-bottom: 8px;
    }

    .mycelix-result-content {
      font-size: 13px;
      color: #1e293b;
      margin-bottom: 4px;
    }

    .mycelix-result-meta {
      display: flex;
      gap: 8px;
      font-size: 11px;
    }

    .mycelix-result-score {
      color: #22c55e;
      font-weight: 500;
    }

    .mycelix-result-tag {
      background: #e2e8f0;
      padding: 1px 6px;
      border-radius: 3px;
      color: #64748b;
    }

    /* Badge Widget */
    .mycelix-badge {
      position: relative;
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    .mycelix-badge-ring {
      transform: rotate(-90deg);
    }

    .mycelix-badge-score {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -70%);
      font-weight: 700;
      color: #1e293b;
    }

    .mycelix-badge-label {
      font-size: 10px;
      color: #64748b;
      margin-top: -8px;
    }

    /* Meter Widget */
    .mycelix-meter-bar {
      display: flex;
      height: 24px;
      border-radius: 4px;
      overflow: hidden;
      margin-bottom: 8px;
    }

    .mycelix-meter-segment {
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-size: 12px;
      font-weight: 600;
      min-width: 20px;
    }

    .mycelix-meter-empirical { background: #ef4444; }
    .mycelix-meter-normative { background: #22c55e; }
    .mycelix-meter-mythic { background: #3b82f6; }

    .mycelix-meter-labels {
      display: flex;
      justify-content: space-between;
      font-size: 11px;
      color: #64748b;
    }

    /* Error State */
    .mycelix-error {
      color: #ef4444;
      text-align: center;
      padding: 20px;
    }

    .mycelix-empty {
      color: #94a3b8;
      text-align: center;
      padding: 20px;
    }

    /* Claim Card */
    .mycelix-claim-content {
      font-size: 14px;
      color: #1e293b;
      line-height: 1.5;
      margin-bottom: 12px;
    }

    .mycelix-classification {
      margin-bottom: 12px;
    }

    .mycelix-class-bar {
      display: flex;
      height: 8px;
      border-radius: 4px;
      overflow: hidden;
      margin-bottom: 4px;
    }

    .mycelix-class-empirical { background: #ef4444; }
    .mycelix-class-normative { background: #22c55e; }
    .mycelix-class-mythic { background: #3b82f6; }

    .mycelix-class-labels {
      display: flex;
      justify-content: space-between;
      font-size: 11px;
    }

    .mycelix-credibility {
      font-size: 13px;
      color: #64748b;
      margin-bottom: 8px;
    }

    .mycelix-author {
      font-size: 12px;
      color: #94a3b8;
      margin-bottom: 8px;
    }

    .mycelix-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
    }

    .mycelix-tag {
      font-size: 11px;
      background: #f1f5f9;
      color: #64748b;
      padding: 2px 8px;
      border-radius: 4px;
    }
  `;

  document.head.appendChild(style);
}

// ============================================================================
// Utilities
// ============================================================================

function escapeHtml(text: string): string {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// ============================================================================
// Types (for API responses)
// ============================================================================

interface FactCheckResult {
  verdict: string;
  confidence: number;
  explanation: string;
  supportingClaims: Array<{ id: string; content: string }>;
  contradictingClaims: Array<{ id: string; content: string }>;
  sources: string[];
}

interface Claim {
  id: string;
  content: string;
  classification: {
    empirical: number;
    normative: number;
    mythic: number;
  };
  author: { did: string };
  tags: string[];
  credibility?: { score: number; confidence: number };
}

interface ClaimSummary {
  id: string;
  content: string;
  credibility?: number;
  tags: string[];
}

// ============================================================================
// Export
// ============================================================================

const Mycelix = {
  init,
  factCheck,
  getClaim,
  search,
  getCredibility,
};

// Auto-init when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => init());
} else {
  init();
}

// Expose to window
(window as unknown as { Mycelix: typeof Mycelix }).Mycelix = Mycelix;

export default Mycelix;
