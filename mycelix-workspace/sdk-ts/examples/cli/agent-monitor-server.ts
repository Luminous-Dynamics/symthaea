#!/usr/bin/env npx ts-node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Agent Monitor WebSocket Server
 *
 * Real-time streaming of agent trust profiles via WebSocket.
 * Broadcasts K-Vector updates, trust scores, and KREDIT changes.
 *
 * Usage:
 *   npx ts-node examples/cli/agent-monitor-server.ts [--port 8080]
 *
 * WebSocket API:
 *   - Connect to ws://localhost:8080
 *   - Send: { "type": "subscribe", "agentId": "agent-1" }
 *   - Receive: { "type": "update", "agent": {...} }
 */

import { WebSocketServer, WebSocket } from 'ws';
import {
  AgentClass,
  AgentStatus,
  KVectorValues,
  computeTrustScore,
  calculateKreditFromTrust,
  createDefaultCalibration,
} from '../../src/index.js';

// =============================================================================
// Types
// =============================================================================

interface MonitoredAgent {
  id: string;
  sponsorDid: string;
  agentClass: AgentClass;
  status: AgentStatus;
  kVector: KVectorValues;
  kreditBalance: number;
  kreditCap: number;
  totalActions: number;
  successfulActions: number;
  trustScore: number;
  lastActivity: Date;
}

interface ClientSubscription {
  ws: WebSocket;
  subscribedAgents: Set<string>;
  allAgents: boolean;
}

interface WebSocketMessage {
  type: 'subscribe' | 'unsubscribe' | 'list' | 'ping';
  agentId?: string;
  all?: boolean;
}

interface AgentUpdate {
  type: 'update';
  timestamp: string;
  agent: MonitoredAgent;
  changes: string[];
}

interface AgentList {
  type: 'agents';
  agents: string[];
}

interface ErrorMessage {
  type: 'error';
  message: string;
}

// =============================================================================
// Agent Simulation
// =============================================================================

function createInitialKVector(): KVectorValues {
  return {
    k_r: 0.5 + Math.random() * 0.2,
    k_a: 0.3 + Math.random() * 0.2,
    k_i: 0.9 + Math.random() * 0.1,
    k_p: 0.5 + Math.random() * 0.2,
    k_m: 0.1 + Math.random() * 0.1,
    k_s: 0.5 + Math.random() * 0.2,
    k_h: 0.5 + Math.random() * 0.2,
    k_topo: 0.2 + Math.random() * 0.2,
    k_v: 0.8,
    k_coherence: 0.6 + Math.random() * 0.2,
  };
}

function createAgent(id: string): MonitoredAgent {
  const kVector = createInitialKVector();
  const trustScore = computeTrustScore(kVector);
  const kreditCap = calculateKreditFromTrust(trustScore, 5000, 10);

  return {
    id,
    sponsorDid: `did:mycelix:sponsor-${id}`,
    agentClass: [AgentClass.Autonomous, AgentClass.Supervised, AgentClass.Assistive][
      Math.floor(Math.random() * 3)
    ],
    status: AgentStatus.Active,
    kVector,
    kreditBalance: Math.floor(kreditCap * 0.5),
    kreditCap,
    totalActions: 0,
    successfulActions: 0,
    trustScore,
    lastActivity: new Date(),
  };
}

function evolveAgent(agent: MonitoredAgent): { agent: MonitoredAgent; changes: string[] } {
  const changes: string[] = [];
  const updated = {
    ...agent,
    kVector: { ...agent.kVector },
  };

  // Simulate action
  const isSuccess = Math.random() > 0.15;
  updated.totalActions++;

  if (isSuccess) {
    updated.successfulActions++;

    // Positive evolution
    const oldKr = updated.kVector.k_r;
    updated.kVector.k_r = Math.min(1.0, updated.kVector.k_r + 0.01 + Math.random() * 0.02);
    if (updated.kVector.k_r !== oldKr) changes.push('k_r+');

    const oldKp = updated.kVector.k_p;
    updated.kVector.k_p = Math.min(1.0, updated.kVector.k_p + 0.005 + Math.random() * 0.01);
    if (updated.kVector.k_p !== oldKp) changes.push('k_p+');

    updated.kreditBalance = Math.min(updated.kreditCap, updated.kreditBalance + 5);
    changes.push('kredit+');
  } else {
    // Negative evolution
    const oldKr = updated.kVector.k_r;
    updated.kVector.k_r = Math.max(0.0, updated.kVector.k_r - 0.02 - Math.random() * 0.03);
    if (updated.kVector.k_r !== oldKr) changes.push('k_r-');

    const oldKp = updated.kVector.k_p;
    updated.kVector.k_p = Math.max(0.0, updated.kVector.k_p - 0.01 - Math.random() * 0.02);
    if (updated.kVector.k_p !== oldKp) changes.push('k_p-');

    updated.kreditBalance = Math.max(0, updated.kreditBalance - 20);
    changes.push('kredit-');
  }

  // Activity increases
  const oldKa = updated.kVector.k_a;
  updated.kVector.k_a = Math.min(1.0, updated.kVector.k_a + 0.005);
  if (updated.kVector.k_a !== oldKa) changes.push('k_a+');

  // Membership increases
  updated.kVector.k_m = Math.min(1.0, updated.kVector.k_m + 0.002);

  // Coherence fluctuates
  updated.kVector.k_coherence = Math.max(0.3, Math.min(1.0,
    (updated.kVector.k_coherence ?? 0.7) + (Math.random() - 0.5) * 0.03
  ));

  // Recalculate trust
  const oldTrust = updated.trustScore;
  updated.trustScore = computeTrustScore(updated.kVector);
  if (Math.abs(updated.trustScore - oldTrust) > 0.01) {
    changes.push(updated.trustScore > oldTrust ? 'trust+' : 'trust-');
  }

  // Update KREDIT cap
  const oldCap = updated.kreditCap;
  updated.kreditCap = calculateKreditFromTrust(updated.trustScore, 5000, 10);
  if (updated.kreditCap !== oldCap) {
    changes.push(updated.kreditCap > oldCap ? 'cap+' : 'cap-');
  }

  updated.lastActivity = new Date();

  return { agent: updated, changes };
}

// =============================================================================
// WebSocket Server
// =============================================================================

class AgentMonitorServer {
  private wss: WebSocketServer;
  private agents: Map<string, MonitoredAgent> = new Map();
  private clients: Map<WebSocket, ClientSubscription> = new Map();
  private updateInterval: NodeJS.Timeout | null = null;

  constructor(port: number) {
    // Create demo agents
    for (let i = 1; i <= 5; i++) {
      const agent = createAgent(`agent-${i}`);
      this.agents.set(agent.id, agent);
    }

    // Create WebSocket server
    this.wss = new WebSocketServer({ port });

    this.wss.on('connection', (ws) => {
      console.log('Client connected');

      // Initialize client subscription
      this.clients.set(ws, {
        ws,
        subscribedAgents: new Set(),
        allAgents: false,
      });

      // Handle messages
      ws.on('message', (data) => {
        try {
          const message = JSON.parse(data.toString()) as WebSocketMessage;
          this.handleMessage(ws, message);
        } catch (e) {
          this.sendError(ws, 'Invalid JSON message');
        }
      });

      // Handle disconnect
      ws.on('close', () => {
        console.log('Client disconnected');
        this.clients.delete(ws);
      });

      // Send welcome message with agent list
      this.sendAgentList(ws);
    });

    // Start agent evolution loop
    this.startEvolutionLoop();

    console.log(`Agent Monitor WebSocket Server running on ws://localhost:${port}`);
    console.log('Monitoring agents:', Array.from(this.agents.keys()).join(', '));
  }

  private handleMessage(ws: WebSocket, message: WebSocketMessage): void {
    const subscription = this.clients.get(ws);
    if (!subscription) return;

    switch (message.type) {
      case 'subscribe':
        if (message.all) {
          subscription.allAgents = true;
          console.log('Client subscribed to all agents');
        } else if (message.agentId) {
          if (this.agents.has(message.agentId)) {
            subscription.subscribedAgents.add(message.agentId);
            console.log(`Client subscribed to ${message.agentId}`);

            // Send current state
            const agent = this.agents.get(message.agentId)!;
            this.sendUpdate(ws, agent, ['subscribed']);
          } else {
            this.sendError(ws, `Agent not found: ${message.agentId}`);
          }
        }
        break;

      case 'unsubscribe':
        if (message.all) {
          subscription.allAgents = false;
          subscription.subscribedAgents.clear();
          console.log('Client unsubscribed from all');
        } else if (message.agentId) {
          subscription.subscribedAgents.delete(message.agentId);
          console.log(`Client unsubscribed from ${message.agentId}`);
        }
        break;

      case 'list':
        this.sendAgentList(ws);
        break;

      case 'ping':
        ws.send(JSON.stringify({ type: 'pong', timestamp: new Date().toISOString() }));
        break;

      default:
        this.sendError(ws, `Unknown message type: ${(message as any).type}`);
    }
  }

  private sendUpdate(ws: WebSocket, agent: MonitoredAgent, changes: string[]): void {
    const message: AgentUpdate = {
      type: 'update',
      timestamp: new Date().toISOString(),
      agent,
      changes,
    };
    ws.send(JSON.stringify(message));
  }

  private sendAgentList(ws: WebSocket): void {
    const message: AgentList = {
      type: 'agents',
      agents: Array.from(this.agents.keys()),
    };
    ws.send(JSON.stringify(message));
  }

  private sendError(ws: WebSocket, message: string): void {
    const errorMsg: ErrorMessage = {
      type: 'error',
      message,
    };
    ws.send(JSON.stringify(errorMsg));
  }

  private startEvolutionLoop(): void {
    this.updateInterval = setInterval(() => {
      // Evolve each agent
      for (const [id, agent] of this.agents) {
        const { agent: evolved, changes } = evolveAgent(agent);
        this.agents.set(id, evolved);

        // Broadcast to subscribed clients
        for (const [_, subscription] of this.clients) {
          if (subscription.allAgents || subscription.subscribedAgents.has(id)) {
            if (subscription.ws.readyState === WebSocket.OPEN) {
              this.sendUpdate(subscription.ws, evolved, changes);
            }
          }
        }
      }
    }, 1000);
  }

  public stop(): void {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
    }
    this.wss.close();
  }
}

// =============================================================================
// Main
// =============================================================================

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  let port = 8080;

  const portIndex = args.indexOf('--port');
  if (portIndex !== -1 && args[portIndex + 1]) {
    port = parseInt(args[portIndex + 1], 10);
  }

  const server = new AgentMonitorServer(port);

  // Handle shutdown
  process.on('SIGINT', () => {
    console.log('\nShutting down...');
    server.stop();
    process.exit(0);
  });

  // Keep running
  await new Promise(() => {});
}

main().catch(console.error);
