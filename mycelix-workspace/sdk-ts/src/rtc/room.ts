// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Collaboration Room
 *
 * High-level abstraction for real-time collaboration rooms.
 * Combines signaling, peer connections, and presence into a simple API.
 */

import {
  type PeerManager,
  createPeerManager,
  type PeerConnectionConfig,
  type PeerConnectionState,
} from './peer.js';
import {
  type SignalingClient,
  createSignalingClient,
  type PeerInfo,
} from './signaling.js';

import type { AgentId } from '../utils/index.js';

// =============================================================================
// Types
// =============================================================================

export interface RoomConfig {
  signalingUrl: string;
  roomId: string;
  agentId: AgentId;
  displayName?: string;
  autoConnect?: boolean;
  iceServers?: RTCIceServer[];
}

export interface RoomState {
  connected: boolean;
  connecting: boolean;
  roomId: string;
  localAgentId: AgentId;
  peers: RoomPeer[];
}

export interface RoomPeer {
  agentId: AgentId;
  displayName?: string;
  connectionState: PeerConnectionState;
  joinedAt: number;
}

export interface PresenceData {
  agentId: AgentId;
  displayName?: string;
  status: 'online' | 'away' | 'busy';
  cursor?: { x: number; y: number };
  selection?: unknown;
  custom?: Record<string, unknown>;
  lastUpdate: number;
}

export type RoomEventType =
  | 'connected'
  | 'disconnected'
  | 'peer-joined'
  | 'peer-left'
  | 'peer-connected'
  | 'peer-disconnected'
  | 'message'
  | 'presence';

export interface RoomEvent {
  type: RoomEventType;
  peerId?: AgentId;
  data?: unknown;
  timestamp: number;
}

export type RoomEventHandler = (event: RoomEvent) => void;

// =============================================================================
// Collaboration Room
// =============================================================================

export class CollaborationRoom {
  private config: Required<RoomConfig>;
  private signalingClient: SignalingClient;
  private peerManager: PeerManager;
  private eventHandlers: Map<RoomEventType, Set<RoomEventHandler>> = new Map();
  private presenceMap: Map<string, PresenceData> = new Map();
  private presenceInterval: ReturnType<typeof setInterval> | null = null;
  private localPresence: Partial<PresenceData> = { status: 'online' };

  constructor(config: RoomConfig) {
    this.config = {
      displayName: config.agentId.slice(0, 8),
      autoConnect: true,
      iceServers: [],
      ...config,
    };

    // Initialize signaling client
    this.signalingClient = createSignalingClient({
      serverUrl: this.config.signalingUrl,
      roomId: this.config.roomId,
      agentId: this.config.agentId,
    });

    // Initialize peer manager
    const peerConfig: PeerConnectionConfig = {
      iceServers: this.config.iceServers.length > 0 ? this.config.iceServers : undefined,
    };
    this.peerManager = createPeerManager(this.config.agentId, this.signalingClient, peerConfig);

    this.setupEventHandlers();
  }

  /**
   * Join the collaboration room
   */
  async join(): Promise<void> {
    await this.signalingClient.connect();
    this.startPresenceBroadcast();
  }

  /**
   * Leave the collaboration room
   */
  leave(): void {
    this.stopPresenceBroadcast();
    this.peerManager.disconnectAll();
    this.signalingClient.disconnect();
    this.presenceMap.clear();
    this.emit({ type: 'disconnected', timestamp: Date.now() });
  }

  /**
   * Send a message to all peers
   */
  broadcast(type: string, data: unknown): number {
    return this.peerManager.broadcast('mycelix-data', type, data);
  }

  /**
   * Send a message to a specific peer
   */
  sendTo(peerId: AgentId, type: string, data: unknown): boolean {
    return this.peerManager.sendToPeer(peerId, 'mycelix-data', type, data);
  }

  /**
   * Update local presence information
   */
  updatePresence(data: Partial<PresenceData>): void {
    this.localPresence = { ...this.localPresence, ...data };
  }

  /**
   * Get presence data for a peer
   */
  getPresence(peerId: AgentId): PresenceData | null {
    return this.presenceMap.get(peerId) ?? null;
  }

  /**
   * Get all presence data
   */
  getAllPresence(): PresenceData[] {
    return Array.from(this.presenceMap.values());
  }

  /**
   * Get current room state
   */
  getState(): RoomState {
    const signalState = this.signalingClient.getState();
    const signalingPeers = this.signalingClient.getPeers();

    const peers: RoomPeer[] = signalingPeers.map((peer) => ({
      agentId: peer.agentId,
      displayName: peer.displayName,
      connectionState: this.peerManager.getPeerState(peer.agentId) ?? 'new',
      joinedAt: peer.joinedAt,
    }));

    return {
      connected: signalState === 'connected',
      connecting: signalState === 'connecting' || signalState === 'reconnecting',
      roomId: this.config.roomId,
      localAgentId: this.config.agentId,
      peers,
    };
  }

  /**
   * Subscribe to room events
   */
  on(type: RoomEventType, handler: RoomEventHandler): () => void {
    if (!this.eventHandlers.has(type)) {
      this.eventHandlers.set(type, new Set());
    }
    this.eventHandlers.get(type)!.add(handler);

    return () => {
      this.eventHandlers.get(type)?.delete(handler);
    };
  }

  /**
   * Subscribe to all events
   */
  onAny(handler: RoomEventHandler): () => void {
    const unsubscribers: (() => void)[] = [];
    const eventTypes: RoomEventType[] = [
      'connected',
      'disconnected',
      'peer-joined',
      'peer-left',
      'peer-connected',
      'peer-disconnected',
      'message',
      'presence',
    ];

    for (const type of eventTypes) {
      unsubscribers.push(this.on(type, handler));
    }

    return () => {
      for (const unsub of unsubscribers) {
        unsub();
      }
    };
  }

  // =============================================================================
  // Private Methods
  // =============================================================================

  private setupEventHandlers(): void {
    // Signaling state changes
    this.signalingClient.onStateChange((state) => {
      if (state === 'connected') {
        this.emit({ type: 'connected', timestamp: Date.now() });
      } else if (state === 'disconnected') {
        this.emit({ type: 'disconnected', timestamp: Date.now() });
      }
    });

    // Peer joins
    this.signalingClient.on('join', (message) => {
      if (message.from !== this.config.agentId) {
        this.emit({
          type: 'peer-joined',
          peerId: message.from,
          data: message.payload,
          timestamp: message.timestamp,
        });

        // Auto-connect to new peer
        if (this.config.autoConnect) {
          this.peerManager.connectToPeer(message.from).catch((error) => {
            console.error(`[Room] Failed to connect to peer ${message.from}:`, error);
          });
        }
      }
    });

    // Peer list (on initial join)
    this.signalingClient.on('peer-list', (message) => {
      const peers = message.payload as PeerInfo[];
      for (const peer of peers) {
        if (peer.agentId !== this.config.agentId && this.config.autoConnect) {
          this.peerManager.connectToPeer(peer.agentId).catch((error) => {
            console.error(`[Room] Failed to connect to peer ${peer.agentId}:`, error);
          });
        }
      }
    });

    // Peer leaves
    this.signalingClient.on('leave', (message) => {
      this.presenceMap.delete(message.from);
      this.emit({
        type: 'peer-left',
        peerId: message.from,
        timestamp: message.timestamp,
      });
    });

    // Peer connection state changes
    this.peerManager.onConnectionChange((peerId, state) => {
      if (state === 'connected') {
        this.emit({
          type: 'peer-connected',
          peerId,
          timestamp: Date.now(),
        });
      } else if (state === 'disconnected' || state === 'failed' || state === 'closed') {
        this.emit({
          type: 'peer-disconnected',
          peerId,
          timestamp: Date.now(),
        });
      }
    });

    // Data channel messages
    this.peerManager.onChannelMessage('mycelix-data', (message) => {
      this.emit({
        type: 'message',
        peerId: message.from,
        data: { messageType: message.type, payload: message.payload },
        timestamp: message.timestamp,
      });
    });

    // Presence updates
    this.peerManager.onChannelMessage('mycelix-presence', (message) => {
      if (message.type === 'presence') {
        const presence = message.payload as PresenceData;
        this.presenceMap.set(message.from, presence);
        this.emit({
          type: 'presence',
          peerId: message.from,
          data: presence,
          timestamp: message.timestamp,
        });
      }
    });
  }

  private startPresenceBroadcast(): void {
    this.stopPresenceBroadcast();

    // Broadcast presence every 5 seconds
    this.presenceInterval = setInterval(() => {
      const presence: PresenceData = {
        agentId: this.config.agentId,
        displayName: this.config.displayName,
        status: (this.localPresence.status as 'online' | 'away' | 'busy') ?? 'online',
        cursor: this.localPresence.cursor,
        selection: this.localPresence.selection,
        custom: this.localPresence.custom,
        lastUpdate: Date.now(),
      };

      this.peerManager.broadcast('mycelix-presence', 'presence', presence);
    }, 5000);
  }

  private stopPresenceBroadcast(): void {
    if (this.presenceInterval) {
      clearInterval(this.presenceInterval);
      this.presenceInterval = null;
    }
  }

  private emit(event: RoomEvent): void {
    const handlers = this.eventHandlers.get(event.type);
    if (handlers) {
      for (const handler of handlers) {
        try {
          handler(event);
        } catch (error) {
          console.error('[Room] Event handler error:', error);
        }
      }
    }
  }
}

// =============================================================================
// Factory
// =============================================================================

export function createRoom(config: RoomConfig): CollaborationRoom {
  return new CollaborationRoom(config);
}

/**
 * Create and join a room in one call
 */
export async function joinRoom(config: RoomConfig): Promise<CollaborationRoom> {
  const room = createRoom(config);
  await room.join();
  return room;
}
