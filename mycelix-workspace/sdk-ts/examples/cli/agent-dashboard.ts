#!/usr/bin/env npx ts-node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Agent Dashboard CLI
 *
 * Interactive command-line tool for monitoring epistemic-aware AI agents.
 * Displays real-time K-Vector evolution, trust scores, and KREDIT balances.
 *
 * Usage:
 *   npx ts-node examples/cli/agent-dashboard.ts [--demo]
 *
 * Options:
 *   --demo    Run with simulated agents (no API required)
 *   --api     API endpoint (default: http://localhost:8080)
 */

import {
  AgentClass,
  AgentStatus,
  KVectorValues,
  computeTrustScore,
  calculateKreditFromTrust,
  createDefaultCalibration,
  CalibrationSummary,
} from '../../src/index.js';

// =============================================================================
// ANSI Color Codes
// =============================================================================

const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  dim: '\x1b[2m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
  white: '\x1b[37m',
  bgBlue: '\x1b[44m',
  bgGreen: '\x1b[42m',
  bgYellow: '\x1b[43m',
  bgRed: '\x1b[41m',
};

// =============================================================================
// Types
// =============================================================================

interface DashboardAgent {
  id: string;
  sponsorDid: string;
  agentClass: AgentClass;
  status: AgentStatus;
  kVector: KVectorValues;
  kreditBalance: number;
  kreditCap: number;
  totalActions: number;
  successfulActions: number;
  pendingEscalations: string[];
  calibration: CalibrationSummary;
  lastActivity: Date;
}

interface KVectorHistory {
  timestamp: Date;
  trustScore: number;
  kVector: KVectorValues;
}

// =============================================================================
// Rendering Functions
// =============================================================================

function clearScreen(): void {
  process.stdout.write('\x1b[2J\x1b[H');
}

function moveCursor(row: number, col: number): void {
  process.stdout.write(`\x1b[${row};${col}H`);
}

function renderBar(value: number, maxWidth: number = 20, filled: string = '█', empty: string = '░'): string {
  const filledCount = Math.round(value * maxWidth);
  const emptyCount = maxWidth - filledCount;
  return filled.repeat(filledCount) + empty.repeat(emptyCount);
}

function colorValue(value: number): string {
  if (value >= 0.8) return colors.green;
  if (value >= 0.5) return colors.yellow;
  return colors.red;
}

function formatKVectorDimension(name: string, abbrev: string, value: number, description: string): string {
  const color = colorValue(value);
  const bar = renderBar(value, 15);
  return `  ${colors.cyan}${abbrev.padEnd(6)}${colors.reset} ${bar} ${color}${value.toFixed(3)}${colors.reset} ${colors.dim}${description}${colors.reset}`;
}

function renderHeader(): void {
  console.log(`${colors.bgBlue}${colors.bright}                                                                        ${colors.reset}`);
  console.log(`${colors.bgBlue}${colors.bright}   MYCELIX AGENT DASHBOARD - Epistemic-Aware AI Agency Monitor         ${colors.reset}`);
  console.log(`${colors.bgBlue}${colors.bright}                                                                        ${colors.reset}`);
  console.log();
}

function renderAgentSummary(agent: DashboardAgent): void {
  const trustScore = computeTrustScore(agent.kVector);
  const trustColor = colorValue(trustScore);
  const successRate = agent.totalActions > 0
    ? agent.successfulActions / agent.totalActions
    : 0;
  const successColor = colorValue(successRate);

  const statusColors: Record<AgentStatus, string> = {
    [AgentStatus.Active]: colors.green,
    [AgentStatus.Throttled]: colors.yellow,
    [AgentStatus.Suspended]: colors.red,
    [AgentStatus.Revoked]: colors.red,
  };

  console.log(`${colors.bright}Agent: ${colors.cyan}${agent.id}${colors.reset}`);
  console.log(`  Sponsor: ${colors.dim}${agent.sponsorDid}${colors.reset}`);
  console.log(`  Class: ${colors.magenta}${agent.agentClass}${colors.reset}  Status: ${statusColors[agent.status]}${agent.status}${colors.reset}`);
  console.log();
  console.log(`  ${colors.bright}Trust Score:${colors.reset} ${trustColor}${trustScore.toFixed(4)}${colors.reset}  ${renderBar(trustScore, 25)}`);
  console.log(`  ${colors.bright}Success Rate:${colors.reset} ${successColor}${(successRate * 100).toFixed(1)}%${colors.reset}  (${agent.successfulActions}/${agent.totalActions} actions)`);
  console.log();
}

function renderKreditStatus(agent: DashboardAgent): void {
  const utilizationRate = agent.kreditBalance / agent.kreditCap;
  const utilizationColor = utilizationRate > 0.8 ? colors.green : utilizationRate > 0.3 ? colors.yellow : colors.red;

  console.log(`${colors.bright}KREDIT Status${colors.reset}`);
  console.log(`  Balance: ${colors.green}${agent.kreditBalance.toLocaleString()}${colors.reset} / ${agent.kreditCap.toLocaleString()}`);
  console.log(`  Utilization: ${utilizationColor}${renderBar(utilizationRate, 30)}${colors.reset} ${(utilizationRate * 100).toFixed(1)}%`);
  console.log();
}

function renderKVector(kVector: KVectorValues): void {
  console.log(`${colors.bright}K-Vector Trust Profile${colors.reset}`);
  console.log(formatKVectorDimension('Reputation', 'k_r', kVector.k_r, 'Peer feedback & outcomes'));
  console.log(formatKVectorDimension('Activity', 'k_a', kVector.k_a, 'Engagement level'));
  console.log(formatKVectorDimension('Integrity', 'k_i', kVector.k_i, 'Constraint compliance'));
  console.log(formatKVectorDimension('Performance', 'k_p', kVector.k_p, 'Output quality'));
  console.log(formatKVectorDimension('Membership', 'k_m', kVector.k_m, 'Time in network'));
  console.log(formatKVectorDimension('Stake', 'k_s', kVector.k_s, 'KREDIT efficiency'));
  console.log(formatKVectorDimension('Historical', 'k_h', kVector.k_h, 'Behavioral consistency'));
  console.log(formatKVectorDimension('Topology', 'k_topo', kVector.k_topo, 'Network connections'));
  console.log(formatKVectorDimension('Verification', 'k_v', kVector.k_v ?? 0, 'Identity verification'));
  console.log(formatKVectorDimension('Coherence', 'k_coherence', kVector.k_coherence ?? 0, 'Output consistency'));
  console.log();
}

function renderCalibration(calibration: CalibrationSummary): void {
  const calibrationColor = colorValue(calibration.calibration_score);

  console.log(`${colors.bright}Uncertainty Calibration${colors.reset}`);
  console.log(`  Score: ${calibrationColor}${calibration.calibration_score.toFixed(3)}${colors.reset}`);
  console.log(`  ${colors.green}Appropriate:${colors.reset} ${calibration.appropriate_uncertainty + calibration.appropriate_confidence}`);
  console.log(`  ${colors.red}Overconfident:${colors.reset} ${calibration.overconfident}  ${colors.yellow}Overcautious:${colors.reset} ${calibration.overcautious}`);
  console.log();
}

function renderEscalations(escalations: string[]): void {
  if (escalations.length === 0) {
    console.log(`${colors.bright}Pending Escalations${colors.reset}: ${colors.green}None${colors.reset}`);
  } else {
    console.log(`${colors.bright}Pending Escalations${colors.reset}: ${colors.yellow}${escalations.length}${colors.reset}`);
    for (const esc of escalations) {
      console.log(`  ${colors.yellow}*${colors.reset} ${esc}`);
    }
  }
  console.log();
}

function renderTrustHistory(history: KVectorHistory[]): void {
  if (history.length === 0) return;

  console.log(`${colors.bright}Trust Score History${colors.reset} (last ${history.length} snapshots)`);

  // Find min/max for scaling
  const scores = history.map(h => h.trustScore);
  const minScore = Math.min(...scores);
  const maxScore = Math.max(...scores);
  const range = maxScore - minScore || 0.1;

  // Render sparkline
  const width = 50;
  const height = 5;
  const canvas: string[][] = Array(height).fill(null).map(() => Array(width).fill(' '));

  for (let i = 0; i < Math.min(history.length, width); i++) {
    const normalized = (history[i].trustScore - minScore) / range;
    const row = height - 1 - Math.floor(normalized * (height - 1));
    canvas[row][i] = colors.cyan + '*' + colors.reset;
  }

  for (const row of canvas) {
    console.log('  ' + row.join(''));
  }

  console.log(`  ${colors.dim}${minScore.toFixed(3)}${' '.repeat(width - 12)}${maxScore.toFixed(3)}${colors.reset}`);
  console.log();
}

function renderDashboard(agent: DashboardAgent, history: KVectorHistory[]): void {
  clearScreen();
  renderHeader();
  renderAgentSummary(agent);
  renderKreditStatus(agent);
  renderKVector(agent.kVector);
  renderCalibration(agent.calibration);
  renderEscalations(agent.pendingEscalations);
  renderTrustHistory(history);
  console.log(`${colors.dim}Last updated: ${new Date().toISOString()}  |  Press Ctrl+C to exit${colors.reset}`);
}

// =============================================================================
// Demo Mode - Simulated Agent Evolution
// =============================================================================

function createDemoAgent(): DashboardAgent {
  return {
    id: 'trading-bot-alpha',
    sponsorDid: 'did:mycelix:hedge-fund-123',
    agentClass: AgentClass.Supervised,
    status: AgentStatus.Active,
    kVector: {
      k_r: 0.5,
      k_a: 0.3,
      k_i: 1.0,
      k_p: 0.5,
      k_m: 0.1,
      k_s: 0.5,
      k_h: 0.5,
      k_topo: 0.2,
      k_v: 0.8,
      k_coherence: 0.7,
    },
    kreditBalance: 5000,
    kreditCap: 10000,
    totalActions: 0,
    successfulActions: 0,
    pendingEscalations: [],
    calibration: createDefaultCalibration('trading-bot-alpha'),
    lastActivity: new Date(),
  };
}

function evolveAgent(agent: DashboardAgent): DashboardAgent {
  const updated = { ...agent, kVector: { ...agent.kVector } };

  // Simulate random action
  const isSuccess = Math.random() > 0.15;
  updated.totalActions++;
  if (isSuccess) {
    updated.successfulActions++;
    // Positive evolution
    updated.kVector.k_r = Math.min(1.0, updated.kVector.k_r + 0.02 * Math.random());
    updated.kVector.k_p = Math.min(1.0, updated.kVector.k_p + 0.015 * Math.random());
    updated.kVector.k_h = Math.min(1.0, updated.kVector.k_h + 0.01 * Math.random());
    updated.kreditBalance = Math.min(updated.kreditCap, updated.kreditBalance + 10);
  } else {
    // Negative evolution
    updated.kVector.k_r = Math.max(0.0, updated.kVector.k_r - 0.03 * Math.random());
    updated.kVector.k_p = Math.max(0.0, updated.kVector.k_p - 0.02 * Math.random());
    updated.kreditBalance = Math.max(0, updated.kreditBalance - 50);
  }

  // Activity increases
  updated.kVector.k_a = Math.min(1.0, updated.kVector.k_a + 0.01);

  // Membership increases slowly
  updated.kVector.k_m = Math.min(1.0, updated.kVector.k_m + 0.005);

  // Coherence fluctuates slightly
  updated.kVector.k_coherence = Math.max(0.3, Math.min(1.0,
    (updated.kVector.k_coherence ?? 0.7) + (Math.random() - 0.5) * 0.05
  ));

  // Random escalation
  if (Math.random() > 0.95 && updated.pendingEscalations.length < 3) {
    const actions = ['high_value_trade', 'portfolio_rebalance', 'margin_increase'];
    updated.pendingEscalations.push(actions[Math.floor(Math.random() * actions.length)]);
  }

  // Random escalation resolution
  if (Math.random() > 0.7 && updated.pendingEscalations.length > 0) {
    updated.pendingEscalations.pop();
    updated.calibration.appropriate_uncertainty++;
    updated.calibration.total_events++;
    updated.calibration.calibration_score =
      (updated.calibration.appropriate_uncertainty + updated.calibration.appropriate_confidence) /
      Math.max(1, updated.calibration.total_events);
  }

  // Update KREDIT cap based on trust
  const trustScore = computeTrustScore(updated.kVector);
  updated.kreditCap = calculateKreditFromTrust(trustScore, 5000, 10);

  updated.lastActivity = new Date();

  return updated;
}

async function runDemoMode(): Promise<void> {
  let agent = createDemoAgent();
  const history: KVectorHistory[] = [];

  console.log('Starting Agent Dashboard in demo mode...');
  console.log('Press Ctrl+C to exit\n');

  // Initial render
  history.push({
    timestamp: new Date(),
    trustScore: computeTrustScore(agent.kVector),
    kVector: { ...agent.kVector },
  });
  renderDashboard(agent, history);

  // Update loop
  const interval = setInterval(() => {
    agent = evolveAgent(agent);

    // Add to history
    history.push({
      timestamp: new Date(),
      trustScore: computeTrustScore(agent.kVector),
      kVector: { ...agent.kVector },
    });

    // Keep last 50 entries
    if (history.length > 50) {
      history.shift();
    }

    renderDashboard(agent, history);
  }, 1000);

  // Handle exit
  process.on('SIGINT', () => {
    clearInterval(interval);
    console.log('\n\nDashboard stopped.');
    process.exit(0);
  });
}

// =============================================================================
// Main
// =============================================================================

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const isDemo = args.includes('--demo') || args.length === 0;

  if (isDemo) {
    await runDemoMode();
  } else {
    console.log('API mode not yet implemented. Use --demo for simulation.');
    process.exit(1);
  }
}

main().catch(console.error);
