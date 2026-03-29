// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * WebRTC Signaling Client
 *
 * Handles peer discovery and connection negotiation for Mycelix real-time collaboration.
 * Uses WebSocket for signaling, integrates with DID-based identity.
 */

import type { AgentId } from '../utils/index.js';

// =============================================================================
// Types
// =============================================================================

export interface SignalingConfig {
  serverUrl: string;
  roomId: string;
  agentId: AgentId;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

export type SignalType =
  | 'join'
  | 'leave'
  | 'offer'
  | 'answer'
  | 'ice-candidate'
  | 'peer-list'
  | 'error';

export interface SignalingMessage {
  type: SignalType;
  from: AgentId;
  to?: AgentId;
  roomId: string;
  payload: unknown;
  timestamp: number;
}

export interface PeerInfo {
  agentId: AgentId;
  displayName?: string;
  joinedAt: number;
  metadata?: Record<string, unknown>;
}

export type SignalingEventHandler = (message: SignalingMessage) => void;

export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting';

// =============================================================================
// Signaling Client
// =============================================================================

export class SignalingClient {
  private config: Required<SignalingConfig>;
  private ws: WebSocket | null = null;
  private state: ConnectionState = 'disconnected';
  private reconnectAttempts = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private handlers: Map<SignalType, Set<SignalingEventHandler>> = new Map();
  private stateListeners: Set<(state: ConnectionState) => void> = new Set();
  private peers: Map<string, PeerInfo> = new Map();

  constructor(config: SignalingConfig) {
    this.config = {
      reconnectInterval: 3000,
      maxReconnectAttempts: 10,
      ...config,
    };
  }

  /**
   * Connect to signaling server
   */
  async connect(): Promise<void> {
    if (this.state === 'connected' || this.state === 'connecting') {
      return;
    }

    this.setState('connecting');

    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.config.serverUrl);

        this.ws.onopen = () => {
          this.setState('connected');
          this.reconnectAttempts = 0;
          this.joinRoom();
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data) as SignalingMessage;
            this.handleMessage(message);
          } catch (error) {
            console.error('[Signaling] Failed to parse message:', error);
          }
        };

        this.ws.onclose = () => {
          this.handleDisconnect();
        };

        this.ws.onerror = (error) => {
          console.error('[Signaling] WebSocket error:', error);
          if (this.state === 'connecting') {
            reject(new Error('Failed to connect to signaling server'));
          }
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Disconnect from signaling server
   */
  disconnect(): void {
    this.clearReconnectTimer();
    this.reconnectAttempts = this.config.maxReconnectAttempts; // Prevent auto-reconnect

    if (this.ws) {
      this.send({
        type: 'leave',
        from: this.config.agentId,
        roomId: this.config.roomId,
        payload: {},
        timestamp: Date.now(),
      });

      this.ws.close();
      this.ws = null;
    }

    this.peers.clear();
    this.setState('disconnected');
  }

  /**
   * Send an offer to a specific peer
   */
  sendOffer(to: AgentId, offer: RTCSessionDescriptionInit): void {
    this.send({
      type: 'offer',
      from: this.config.agentId,
      to,
      roomId: this.config.roomId,
      payload: offer,
      timestamp: Date.now(),
    });
  }

  /**
   * Send an answer to a specific peer
   */
  sendAnswer(to: AgentId, answer: RTCSessionDescriptionInit): void {
    this.send({
      type: 'answer',
      from: this.config.agentId,
      to,
      roomId: this.config.roomId,
      payload: answer,
      timestamp: Date.now(),
    });
  }

  /**
   * Send an ICE candidate to a specific peer
   */
  sendIceCandidate(to: AgentId, candidate: RTCIceCandidate): void {
    this.send({
      type: 'ice-candidate',
      from: this.config.agentId,
      to,
      roomId: this.config.roomId,
      payload: candidate.toJSON(),
      timestamp: Date.now(),
    });
  }

  /**
   * Subscribe to a specific signal type
   */
  on(type: SignalType, handler: SignalingEventHandler): () => void {
    if (!this.handlers.has(type)) {
      this.handlers.set(type, new Set());
    }
    this.handlers.get(type)!.add(handler);

    // Return unsubscribe function
    return () => {
      this.handlers.get(type)?.delete(handler);
    };
  }

  /**
   * Subscribe to connection state changes
   */
  onStateChange(listener: (state: ConnectionState) => void): () => void {
    this.stateListeners.add(listener);
    return () => this.stateListeners.delete(listener);
  }

  /**
   * Get current peers in the room
   */
  getPeers(): PeerInfo[] {
    return Array.from(this.peers.values());
  }

  /**
   * Get current connection state
   */
  getState(): ConnectionState {
    return this.state;
  }

  /**
   * Get the agent ID
   */
  getAgentId(): AgentId {
    return this.config.agentId;
  }

  // =============================================================================
  // Private Methods
  // =============================================================================

  private joinRoom(): void {
    this.send({
      type: 'join',
      from: this.config.agentId,
      roomId: this.config.roomId,
      payload: {
        displayName: this.config.agentId.slice(0, 8),
      },
      timestamp: Date.now(),
    });
  }

  private send(message: SignalingMessage): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('[Signaling] Cannot send message - not connected');
    }
  }

  private handleMessage(message: SignalingMessage): void {
    // Update peer list on join/leave
    if (message.type === 'join' && message.from !== this.config.agentId) {
      const payload = message.payload as { displayName?: string; metadata?: Record<string, unknown> };
      this.peers.set(message.from, {
        agentId: message.from,
        displayName: payload.displayName,
        joinedAt: message.timestamp,
        metadata: payload.metadata,
      });
    } else if (message.type === 'leave') {
      this.peers.delete(message.from);
    } else if (message.type === 'peer-list') {
      const peerList = message.payload as PeerInfo[];
      this.peers.clear();
      for (const peer of peerList) {
        if (peer.agentId !== this.config.agentId) {
          this.peers.set(peer.agentId, peer);
        }
      }
    }

    // Dispatch to handlers
    const handlers = this.handlers.get(message.type);
    if (handlers) {
      for (const handler of handlers) {
        try {
          handler(message);
        } catch (error) {
          console.error('[Signaling] Handler error:', error);
        }
      }
    }
  }

  private handleDisconnect(): void {
    this.ws = null;

    if (this.reconnectAttempts < this.config.maxReconnectAttempts) {
      this.setState('reconnecting');
      this.scheduleReconnect();
    } else {
      this.setState('disconnected');
      this.peers.clear();
    }
  }

  private scheduleReconnect(): void {
    this.clearReconnectTimer();
    this.reconnectTimer = setTimeout(() => {
      this.reconnectAttempts++;
      this.connect().catch((error) => {
        console.error('[Signaling] Reconnect failed:', error);
      });
    }, this.config.reconnectInterval);
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private setState(state: ConnectionState): void {
    this.state = state;
    for (const listener of this.stateListeners) {
      try {
        listener(state);
      } catch (error) {
        console.error('[Signaling] State listener error:', error);
      }
    }
  }
}

// =============================================================================
// Factory
// =============================================================================

export function createSignalingClient(config: SignalingConfig): SignalingClient {
  return new SignalingClient(config);
}
