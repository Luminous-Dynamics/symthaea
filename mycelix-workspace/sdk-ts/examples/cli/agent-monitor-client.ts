#!/usr/bin/env npx ts-node
/**
 * Agent Monitor WebSocket Client
 *
 * Connects to the Agent Monitor Server and displays real-time updates.
 *
 * Usage:
 *   npx ts-node examples/cli/agent-monitor-client.ts [--url ws://localhost:8080] [--agent agent-1]
 */

import WebSocket from 'ws';
import { KVectorValues, computeTrustScore } from '../../src/index.js';

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
};

// =============================================================================
// Types
// =============================================================================

interface AgentUpdate {
  type: 'update';
  timestamp: string;
  agent: {
    id: string;
    agentClass: string;
    status: string;
    kVector: KVectorValues;
    kreditBalance: number;
    kreditCap: number;
    trustScore: number;
    totalActions: number;
    successfulActions: number;
  };
  changes: string[];
}

interface AgentList {
  type: 'agents';
  agents: string[];
}

// =============================================================================
// Rendering
// =============================================================================

function clearLine(): void {
  process.stdout.write('\x1b[2K\r');
}

function renderBar(value: number, width: number = 20): string {
  const filled = Math.round(value * width);
  const empty = width - filled;
  const bar = '█'.repeat(filled) + '░'.repeat(empty);
  const color = value >= 0.7 ? colors.green : value >= 0.4 ? colors.yellow : colors.red;
  return `${color}${bar}${colors.reset}`;
}

function formatChange(change: string): string {
  if (change.endsWith('+')) {
    return `${colors.green}▲${change.slice(0, -1)}${colors.reset}`;
  } else if (change.endsWith('-')) {
    return `${colors.red}▼${change.slice(0, -1)}${colors.reset}`;
  }
  return change;
}

function renderUpdate(update: AgentUpdate): void {
  const { agent, changes, timestamp } = update;

  console.log();
  console.log(`${colors.bright}${colors.cyan}═══ ${agent.id} ═══${colors.reset} ${colors.dim}${timestamp}${colors.reset}`);
  console.log(`  Class: ${colors.magenta}${agent.agentClass}${colors.reset}  Status: ${colors.green}${agent.status}${colors.reset}`);
  console.log();

  // Trust score bar
  console.log(`  Trust:  ${renderBar(agent.trustScore, 25)} ${agent.trustScore.toFixed(4)}`);

  // Success rate
  const successRate = agent.totalActions > 0 ? agent.successfulActions / agent.totalActions : 0;
  console.log(`  Success: ${renderBar(successRate, 25)} ${(successRate * 100).toFixed(1)}% (${agent.successfulActions}/${agent.totalActions})`);

  // KREDIT
  const utilization = agent.kreditBalance / agent.kreditCap;
  console.log(`  KREDIT: ${renderBar(utilization, 25)} ${agent.kreditBalance}/${agent.kreditCap}`);

  // K-Vector summary
  console.log();
  console.log(`  ${colors.dim}K-Vector:${colors.reset}`);
  const kv = agent.kVector;
  console.log(`    r=${kv.k_r.toFixed(2)} a=${kv.k_a.toFixed(2)} i=${kv.k_i.toFixed(2)} p=${kv.k_p.toFixed(2)} m=${kv.k_m.toFixed(2)}`);
  console.log(`    s=${kv.k_s.toFixed(2)} h=${kv.k_h.toFixed(2)} t=${kv.k_topo.toFixed(2)} v=${(kv.k_v ?? 0).toFixed(2)} φ=${(kv.k_coherence ?? 0).toFixed(2)}`);

  // Changes
  if (changes.length > 0 && changes[0] !== 'subscribed') {
    console.log();
    console.log(`  Changes: ${changes.map(formatChange).join(' ')}`);
  }
}

// =============================================================================
// WebSocket Client
// =============================================================================

class AgentMonitorClient {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private subscribedAgent: string | null = null;

  constructor(
    private url: string,
    private agentId?: string,
  ) {}

  async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      console.log(`Connecting to ${this.url}...`);

      this.ws = new WebSocket(this.url);

      this.ws.on('open', () => {
        console.log(`${colors.green}Connected!${colors.reset}`);
        this.reconnectAttempts = 0;

        if (this.agentId) {
          this.subscribe(this.agentId);
        } else {
          this.subscribeAll();
        }

        resolve();
      });

      this.ws.on('message', (data) => {
        try {
          const message = JSON.parse(data.toString());
          this.handleMessage(message);
        } catch (e) {
          console.error('Failed to parse message:', e);
        }
      });

      this.ws.on('close', () => {
        console.log(`${colors.yellow}Disconnected${colors.reset}`);
        this.attemptReconnect();
      });

      this.ws.on('error', (error) => {
        console.error(`${colors.red}Error:${colors.reset}`, error.message);
        if (this.reconnectAttempts === 0) {
          reject(error);
        }
      });
    });
  }

  private handleMessage(message: any): void {
    switch (message.type) {
      case 'agents':
        console.log(`\nAvailable agents: ${message.agents.join(', ')}`);
        break;

      case 'update':
        renderUpdate(message as AgentUpdate);
        break;

      case 'error':
        console.error(`${colors.red}Server error:${colors.reset} ${message.message}`);
        break;

      case 'pong':
        console.log(`${colors.dim}Pong: ${message.timestamp}${colors.reset}`);
        break;

      default:
        console.log(`Unknown message type: ${message.type}`);
    }
  }

  private subscribe(agentId: string): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.subscribedAgent = agentId;
      this.ws.send(JSON.stringify({ type: 'subscribe', agentId }));
      console.log(`Subscribed to ${agentId}`);
    }
  }

  private subscribeAll(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'subscribe', all: true }));
      console.log('Subscribed to all agents');
    }
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Reconnecting (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
      setTimeout(() => this.connect().catch(() => {}), 2000);
    } else {
      console.log('Max reconnect attempts reached. Exiting.');
      process.exit(1);
    }
  }

  public close(): void {
    if (this.ws) {
      this.ws.close();
    }
  }
}

// =============================================================================
// Main
// =============================================================================

async function main(): Promise<void> {
  const args = process.argv.slice(2);

  let url = 'ws://localhost:8080';
  let agentId: string | undefined;

  const urlIndex = args.indexOf('--url');
  if (urlIndex !== -1 && args[urlIndex + 1]) {
    url = args[urlIndex + 1];
  }

  const agentIndex = args.indexOf('--agent');
  if (agentIndex !== -1 && args[agentIndex + 1]) {
    agentId = args[agentIndex + 1];
  }

  console.log(`${colors.bright}Agent Monitor Client${colors.reset}`);
  console.log(`Press Ctrl+C to exit\n`);

  const client = new AgentMonitorClient(url, agentId);

  process.on('SIGINT', () => {
    console.log('\nDisconnecting...');
    client.close();
    process.exit(0);
  });

  try {
    await client.connect();
  } catch (e) {
    console.error('Failed to connect:', (e as Error).message);
    process.exit(1);
  }
}

main().catch(console.error);
