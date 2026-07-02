// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Knowledge - VS Code Extension
 *
 * Fact-check text, explore claims, and contribute to the knowledge graph
 */

import * as vscode from 'vscode';
import { KnowledgeClient } from '@mycelix/knowledge-client';

// ============================================================================
// Global State
// ============================================================================

let client: KnowledgeClient | null = null;
let statusBarItem: vscode.StatusBarItem;
let outputChannel: vscode.OutputChannel;

// ============================================================================
// Extension Activation
// ============================================================================

export async function activate(context: vscode.ExtensionContext) {
  outputChannel = vscode.window.createOutputChannel('Mycelix Knowledge');
  outputChannel.appendLine('Mycelix Knowledge extension activated');

  // Create status bar item
  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusBarItem.text = '$(search) Mycelix';
  statusBarItem.tooltip = 'Mycelix Knowledge - Click to connect';
  statusBarItem.command = 'mycelix.connectHolochain';
  statusBarItem.show();
  context.subscriptions.push(statusBarItem);

  // Register commands
  context.subscriptions.push(
    vscode.commands.registerCommand('mycelix.factCheck', factCheckCommand),
    vscode.commands.registerCommand('mycelix.submitClaim', submitClaimCommand),
    vscode.commands.registerCommand('mycelix.searchClaims', searchClaimsCommand),
    vscode.commands.registerCommand('mycelix.classifyText', classifyTextCommand),
    vscode.commands.registerCommand('mycelix.showCredibility', showCredibilityCommand),
    vscode.commands.registerCommand('mycelix.connectHolochain', connectHolochainCommand)
  );

  // Register tree data providers
  const claimsProvider = new ClaimsTreeProvider();
  const factChecksProvider = new FactChecksTreeProvider();
  const graphProvider = new GraphTreeProvider();

  context.subscriptions.push(
    vscode.window.registerTreeDataProvider('mycelix-claims', claimsProvider),
    vscode.window.registerTreeDataProvider('mycelix-factchecks', factChecksProvider),
    vscode.window.registerTreeDataProvider('mycelix-graph', graphProvider)
  );

  // Register hover provider for inline fact-checks
  const config = vscode.workspace.getConfiguration('mycelix');
  if (config.get('autoFactCheck')) {
    context.subscriptions.push(
      vscode.languages.registerHoverProvider('*', new FactCheckHoverProvider())
    );
  }

  // Register code lens provider for credibility annotations
  if (config.get('showInlineDecorations')) {
    context.subscriptions.push(
      vscode.languages.registerCodeLensProvider('*', new CredibilityCodeLensProvider())
    );
  }

  // Auto-connect if previously connected
  try {
    await connectToHolochain();
  } catch {
    outputChannel.appendLine('Auto-connect failed, will connect on first use');
  }
}

export function deactivate() {
  client?.disconnect();
}

// ============================================================================
// Connection
// ============================================================================

async function connectToHolochain(): Promise<void> {
  const config = vscode.workspace.getConfiguration('mycelix');
  const url = config.get<string>('holochainUrl') || 'ws://localhost:8888';
  const appId = config.get<string>('appId') || 'mycelix-knowledge';

  try {
    client = new KnowledgeClient({ url, appId });
    await client.connect();
    statusBarItem.text = '$(check) Mycelix';
    statusBarItem.backgroundColor = undefined;
    outputChannel.appendLine(`Connected to Holochain at ${url}`);
    vscode.window.showInformationMessage('Connected to Mycelix Knowledge network');
  } catch (error) {
    statusBarItem.text = '$(error) Mycelix';
    statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.errorBackground');
    throw error;
  }
}

async function ensureConnected(): Promise<KnowledgeClient> {
  if (!client) {
    await connectToHolochain();
  }
  return client!;
}

// ============================================================================
// Commands
// ============================================================================

async function connectHolochainCommand(): Promise<void> {
  try {
    await connectToHolochain();
  } catch (error) {
    vscode.window.showErrorMessage(`Failed to connect: ${error}`);
  }
}

async function factCheckCommand(): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage('No text selected');
    return;
  }

  const selection = editor.selection;
  const text = editor.document.getText(selection);

  if (!text.trim()) {
    vscode.window.showWarningMessage('Please select some text to fact-check');
    return;
  }

  try {
    const knowledgeClient = await ensureConnected();

    await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: 'Fact-checking...',
        cancellable: false,
      },
      async () => {
        const result = await knowledgeClient.factcheck.factCheck({
          statement: text,
          minEpistemicLevel: 2,
          minConfidence: 0.6,
        });

        showFactCheckResult(text, result);
      }
    );
  } catch (error) {
    vscode.window.showErrorMessage(`Fact-check failed: ${error}`);
  }
}

function showFactCheckResult(text: string, result: FactCheckResult): void {
  const panel = vscode.window.createWebviewPanel(
    'factCheckResult',
    'Fact Check Result',
    vscode.ViewColumn.Beside,
    { enableScripts: true }
  );

  const verdictColors: Record<string, string> = {
    TRUE: '#22c55e',
    MOSTLY_TRUE: '#84cc16',
    MIXED: '#eab308',
    MOSTLY_FALSE: '#f97316',
    FALSE: '#ef4444',
    UNVERIFIABLE: '#6b7280',
    INSUFFICIENT_EVIDENCE: '#8b5cf6',
  };

  panel.webview.html = `
    <!DOCTYPE html>
    <html>
    <head>
      <style>
        body { font-family: var(--vscode-font-family); padding: 20px; color: var(--vscode-foreground); }
        .statement { background: var(--vscode-textBlockQuote-background); padding: 16px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid var(--vscode-textLink-foreground); }
        .verdict { font-size: 24px; font-weight: bold; padding: 12px 24px; border-radius: 8px; display: inline-block; margin: 16px 0; }
        .confidence { margin: 16px 0; }
        .confidence-bar { height: 8px; background: var(--vscode-progressBar-background); border-radius: 4px; overflow: hidden; }
        .confidence-fill { height: 100%; background: var(--vscode-progressBar-foreground); }
        .section { margin: 24px 0; }
        .section h3 { margin-bottom: 12px; }
        .claim-list { list-style: none; padding: 0; }
        .claim-list li { padding: 8px; background: var(--vscode-editor-background); margin: 4px 0; border-radius: 4px; }
        .explanation { background: var(--vscode-textBlockQuote-background); padding: 16px; border-radius: 8px; line-height: 1.6; }
      </style>
    </head>
    <body>
      <h2>Fact Check Result</h2>

      <div class="statement">
        <strong>Statement:</strong><br>
        "${escapeHtml(text)}"
      </div>

      <div class="verdict" style="background: ${verdictColors[result.verdict]}; color: white;">
        ${result.verdict.replace('_', ' ')}
      </div>

      <div class="confidence">
        <strong>Confidence: ${Math.round(result.confidence * 100)}%</strong>
        <div class="confidence-bar">
          <div class="confidence-fill" style="width: ${result.confidence * 100}%"></div>
        </div>
      </div>

      <div class="section">
        <h3>Explanation</h3>
        <div class="explanation">${escapeHtml(result.explanation)}</div>
      </div>

      ${result.supportingClaims.length > 0 ? `
        <div class="section">
          <h3>Supporting Claims (${result.supportingClaims.length})</h3>
          <ul class="claim-list">
            ${result.supportingClaims.map((c) => `<li>✓ ${escapeHtml(c.content)}</li>`).join('')}
          </ul>
        </div>
      ` : ''}

      ${result.contradictingClaims.length > 0 ? `
        <div class="section">
          <h3>Contradicting Claims (${result.contradictingClaims.length})</h3>
          <ul class="claim-list">
            ${result.contradictingClaims.map((c) => `<li>✗ ${escapeHtml(c.content)}</li>`).join('')}
          </ul>
        </div>
      ` : ''}

      ${result.sources.length > 0 ? `
        <div class="section">
          <h3>Sources</h3>
          <ul class="claim-list">
            ${result.sources.map((s) => `<li>📄 ${escapeHtml(s)}</li>`).join('')}
          </ul>
        </div>
      ` : ''}

      <p style="color: var(--vscode-descriptionForeground); font-size: 12px;">
        Checked at: ${result.checkedAt}
      </p>
    </body>
    </html>
  `;
}

async function submitClaimCommand(): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor) return;

  const text = editor.document.getText(editor.selection);
  if (!text.trim()) {
    vscode.window.showWarningMessage('Please select text to submit as a claim');
    return;
  }

  // Get classification from user
  const empirical = await vscode.window.showInputBox({
    prompt: 'Empirical score (0-1): How verifiable through observation?',
    value: '0.5',
    validateInput: (v) => {
      const n = parseFloat(v);
      return n >= 0 && n <= 1 ? null : 'Enter a number between 0 and 1';
    },
  });
  if (!empirical) return;

  const normative = await vscode.window.showInputBox({
    prompt: 'Normative score (0-1): How value-based/ethical?',
    value: '0.3',
    validateInput: (v) => {
      const n = parseFloat(v);
      return n >= 0 && n <= 1 ? null : 'Enter a number between 0 and 1';
    },
  });
  if (!normative) return;

  const mythic = await vscode.window.showInputBox({
    prompt: 'Mythic score (0-1): How meaning-making/narrative?',
    value: '0.2',
    validateInput: (v) => {
      const n = parseFloat(v);
      return n >= 0 && n <= 1 ? null : 'Enter a number between 0 and 1';
    },
  });
  if (!mythic) return;

  const tags = await vscode.window.showInputBox({
    prompt: 'Tags (comma-separated)',
    placeHolder: 'science, technology, health',
  });

  try {
    const knowledgeClient = await ensureConnected();

    const claimId = await knowledgeClient.claims.createClaim({
      content: text,
      classification: {
        empirical: parseFloat(empirical),
        normative: parseFloat(normative),
        mythic: parseFloat(mythic),
      },
      tags: tags ? tags.split(',').map((t) => t.trim()) : [],
    });

    vscode.window.showInformationMessage(`Claim submitted! ID: ${claimId}`);
    outputChannel.appendLine(`Created claim: ${claimId}`);
  } catch (error) {
    vscode.window.showErrorMessage(`Failed to submit claim: ${error}`);
  }
}

async function searchClaimsCommand(): Promise<void> {
  const query = await vscode.window.showInputBox({
    prompt: 'Search the knowledge graph',
    placeHolder: 'Enter search query...',
  });

  if (!query) return;

  try {
    const knowledgeClient = await ensureConnected();

    const results = await knowledgeClient.query.search(query, { limit: 20 });

    const items = results.map((claim) => ({
      label: claim.content.substring(0, 80) + (claim.content.length > 80 ? '...' : ''),
      description: `E:${claim.classification.empirical.toFixed(2)} N:${claim.classification.normative.toFixed(2)} M:${claim.classification.mythic.toFixed(2)}`,
      detail: claim.tags.join(', '),
      claim,
    }));

    const selected = await vscode.window.showQuickPick(items, {
      placeHolder: `Found ${results.length} claims`,
      matchOnDescription: true,
      matchOnDetail: true,
    });

    if (selected) {
      // Show full claim details
      const panel = vscode.window.createWebviewPanel(
        'claimDetail',
        'Claim Detail',
        vscode.ViewColumn.Beside,
        {}
      );
      panel.webview.html = generateClaimDetailHtml(selected.claim);
    }
  } catch (error) {
    vscode.window.showErrorMessage(`Search failed: ${error}`);
  }
}

async function classifyTextCommand(): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor) return;

  const text = editor.document.getText(editor.selection);
  if (!text.trim()) {
    vscode.window.showWarningMessage('Please select text to classify');
    return;
  }

  // Simple heuristic classification (in production, use ML model)
  const classification = classifyText(text);

  const dominant =
    classification.empirical >= classification.normative &&
    classification.empirical >= classification.mythic
      ? 'Empirical'
      : classification.normative >= classification.mythic
        ? 'Normative'
        : 'Mythic';

  vscode.window.showInformationMessage(
    `Classification: ${dominant} (E:${classification.empirical.toFixed(2)} N:${classification.normative.toFixed(2)} M:${classification.mythic.toFixed(2)})`
  );
}

async function showCredibilityCommand(): Promise<void> {
  const claimId = await vscode.window.showInputBox({
    prompt: 'Enter claim ID to check credibility',
  });

  if (!claimId) return;

  try {
    const knowledgeClient = await ensureConnected();

    const credibility = await knowledgeClient.inference.calculateEnhancedCredibility(claimId, 'Claim');

    vscode.window.showInformationMessage(
      `Credibility: ${(credibility.score * 100).toFixed(1)}% (confidence: ${(credibility.confidence * 100).toFixed(1)}%)`
    );
  } catch (error) {
    vscode.window.showErrorMessage(`Failed to get credibility: ${error}`);
  }
}

// ============================================================================
// Tree Data Providers
// ============================================================================

class ClaimsTreeProvider implements vscode.TreeDataProvider<ClaimItem> {
  private _onDidChangeTreeData = new vscode.EventEmitter<ClaimItem | undefined>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  refresh(): void {
    this._onDidChangeTreeData.fire(undefined);
  }

  getTreeItem(element: ClaimItem): vscode.TreeItem {
    return element;
  }

  async getChildren(): Promise<ClaimItem[]> {
    if (!client) return [];

    try {
      const claims = await client.query.listClaims({ limit: 20 });
      return claims.items.map(
        (claim) =>
          new ClaimItem(claim.content.substring(0, 50), claim.id, vscode.TreeItemCollapsibleState.None)
      );
    } catch {
      return [];
    }
  }
}

class ClaimItem extends vscode.TreeItem {
  constructor(
    label: string,
    public readonly claimId: string,
    collapsibleState: vscode.TreeItemCollapsibleState
  ) {
    super(label, collapsibleState);
    this.tooltip = `Claim ID: ${claimId}`;
    this.contextValue = 'claim';
  }
}

class FactChecksTreeProvider implements vscode.TreeDataProvider<FactCheckItem> {
  private results: FactCheckResult[] = [];

  addResult(result: FactCheckResult): void {
    this.results.unshift(result);
    if (this.results.length > 10) this.results.pop();
  }

  getTreeItem(element: FactCheckItem): vscode.TreeItem {
    return element;
  }

  getChildren(): FactCheckItem[] {
    return this.results.map(
      (r) => new FactCheckItem(r.verdict, r.confidence, vscode.TreeItemCollapsibleState.None)
    );
  }
}

class FactCheckItem extends vscode.TreeItem {
  constructor(verdict: string, confidence: number, collapsibleState: vscode.TreeItemCollapsibleState) {
    super(`${verdict} (${Math.round(confidence * 100)}%)`, collapsibleState);
    this.contextValue = 'factcheck';
  }
}

class GraphTreeProvider implements vscode.TreeDataProvider<GraphItem> {
  getTreeItem(element: GraphItem): vscode.TreeItem {
    return element;
  }

  getChildren(): GraphItem[] {
    return [
      new GraphItem('Total Claims', '1,234', vscode.TreeItemCollapsibleState.None),
      new GraphItem('Relationships', '5,678', vscode.TreeItemCollapsibleState.None),
      new GraphItem('Active Markets', '42', vscode.TreeItemCollapsibleState.None),
    ];
  }
}

class GraphItem extends vscode.TreeItem {
  constructor(label: string, value: string, collapsibleState: vscode.TreeItemCollapsibleState) {
    super(`${label}: ${value}`, collapsibleState);
  }
}

// ============================================================================
// Providers
// ============================================================================

class FactCheckHoverProvider implements vscode.HoverProvider {
  async provideHover(
    document: vscode.TextDocument,
    position: vscode.Position
  ): Promise<vscode.Hover | null> {
    const wordRange = document.getWordRangeAtPosition(position);
    if (!wordRange) return null;

    // Only provide hover for substantial text
    const text = document.getText(wordRange);
    if (text.length < 10) return null;

    return new vscode.Hover('Right-click to fact-check this text');
  }
}

class CredibilityCodeLensProvider implements vscode.CodeLensProvider {
  provideCodeLenses(document: vscode.TextDocument): vscode.CodeLens[] {
    // Find claim-like statements in the document
    const codeLenses: vscode.CodeLens[] = [];
    const text = document.getText();

    // Simple pattern to find assertion-like sentences
    const assertionPattern = /^[A-Z][^.!?]*(?:is|are|was|were|will|can|should)[^.!?]*[.!?]/gm;
    let match;

    while ((match = assertionPattern.exec(text)) !== null) {
      const startPos = document.positionAt(match.index);
      const range = new vscode.Range(startPos, startPos);
      codeLenses.push(
        new vscode.CodeLens(range, {
          title: '🔍 Fact-check',
          command: 'mycelix.factCheck',
        })
      );
    }

    return codeLenses.slice(0, 5); // Limit to 5 per document
  }
}

// ============================================================================
// Utilities
// ============================================================================

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function classifyText(text: string): { empirical: number; normative: number; mythic: number } {
  const lower = text.toLowerCase();

  // Simple keyword-based classification
  const empiricalKeywords = ['data', 'study', 'research', 'evidence', 'percent', 'measured', 'observed'];
  const normativeKeywords = ['should', 'ought', 'must', 'right', 'wrong', 'good', 'bad', 'better'];
  const mythicKeywords = ['believe', 'meaning', 'purpose', 'destiny', 'spirit', 'soul', 'tradition'];

  const empiricalScore = empiricalKeywords.filter((k) => lower.includes(k)).length;
  const normativeScore = normativeKeywords.filter((k) => lower.includes(k)).length;
  const mythicScore = mythicKeywords.filter((k) => lower.includes(k)).length;

  const total = empiricalScore + normativeScore + mythicScore || 1;

  return {
    empirical: empiricalScore / total || 0.33,
    normative: normativeScore / total || 0.33,
    mythic: mythicScore / total || 0.34,
  };
}

function generateClaimDetailHtml(claim: ClaimSummary): string {
  return `
    <!DOCTYPE html>
    <html>
    <head>
      <style>
        body { font-family: var(--vscode-font-family); padding: 20px; }
        .content { font-size: 16px; line-height: 1.6; margin-bottom: 20px; }
        .classification { display: flex; gap: 16px; margin: 16px 0; }
        .class-item { text-align: center; }
        .class-value { font-size: 24px; font-weight: bold; }
        .tags { display: flex; gap: 8px; flex-wrap: wrap; }
        .tag { background: var(--vscode-badge-background); color: var(--vscode-badge-foreground); padding: 4px 8px; border-radius: 4px; font-size: 12px; }
      </style>
    </head>
    <body>
      <h2>Claim Details</h2>
      <div class="content">${escapeHtml(claim.content)}</div>

      <h3>Epistemic Classification</h3>
      <div class="classification">
        <div class="class-item">
          <div class="class-value" style="color: #ef4444">${(claim.classification.empirical * 100).toFixed(0)}%</div>
          <div>Empirical</div>
        </div>
        <div class="class-item">
          <div class="class-value" style="color: #22c55e">${(claim.classification.normative * 100).toFixed(0)}%</div>
          <div>Normative</div>
        </div>
        <div class="class-item">
          <div class="class-value" style="color: #3b82f6">${(claim.classification.mythic * 100).toFixed(0)}%</div>
          <div>Mythic</div>
        </div>
      </div>

      ${claim.tags.length > 0 ? `
        <h3>Tags</h3>
        <div class="tags">
          ${claim.tags.map((t) => `<span class="tag">${escapeHtml(t)}</span>`).join('')}
        </div>
      ` : ''}

      <p><small>Author: ${claim.author}</small></p>
    </body>
    </html>
  `;
}

// ============================================================================
// Types (from SDK)
// ============================================================================

interface FactCheckResult {
  verdict: string;
  confidence: number;
  explanation: string;
  supportingClaims: ClaimSummary[];
  contradictingClaims: ClaimSummary[];
  sources: string[];
  checkedAt: string;
}

interface ClaimSummary {
  id: string;
  content: string;
  author: string;
  classification: {
    empirical: number;
    normative: number;
    mythic: number;
  };
  tags: string[];
}
