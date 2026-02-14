/**
 * WebRTC Peer Connection Manager
 *
 * Manages peer-to-peer connections for real-time collaboration.
 * Handles connection lifecycle, data channels, and media streams.
 */

import type { SignalingClient } from './signaling.js';
import type { AgentId } from '../utils/index.js';

// =============================================================================
// Types
// =============================================================================

export interface PeerConnectionConfig {
  iceServers?: RTCIceServer[];
  dataChannels?: DataChannelConfig[];
}

export interface DataChannelConfig {
  label: string;
  options?: RTCDataChannelInit;
}

export type PeerConnectionState =
  | 'new'
  | 'connecting'
  | 'connected'
  | 'disconnected'
  | 'failed'
  | 'closed';

export interface PeerConnection {
  peerId: AgentId;
  state: PeerConnectionState;
  connection: RTCPeerConnection;
  dataChannels: Map<string, RTCDataChannel>;
  createdAt: number;
}

export type DataChannelMessage = {
  type: string;
  payload: unknown;
  timestamp: number;
  from: AgentId;
};

export type DataChannelHandler = (message: DataChannelMessage, channel: RTCDataChannel) => void;

// =============================================================================
// Default Configuration
// =============================================================================

const DEFAULT_ICE_SERVERS: RTCIceServer[] = [
  { urls: 'stun:stun.l.google.com:19302' },
  { urls: 'stun:stun1.l.google.com:19302' },
  { urls: 'stun:stun2.l.google.com:19302' },
];

const DEFAULT_DATA_CHANNELS: DataChannelConfig[] = [
  { label: 'mycelix-data', options: { ordered: true } },
  { label: 'mycelix-presence', options: { ordered: false, maxRetransmits: 0 } },
];

// =============================================================================
// Peer Manager
// =============================================================================

export class PeerManager {
  private localAgentId: AgentId;
  private signalingClient: SignalingClient;
  private config: Required<PeerConnectionConfig>;
  private peers: Map<string, PeerConnection> = new Map();
  private channelHandlers: Map<string, Set<DataChannelHandler>> = new Map();
  private connectionListeners: Set<(peerId: AgentId, state: PeerConnectionState) => void> =
    new Set();

  constructor(
    localAgentId: AgentId,
    signalingClient: SignalingClient,
    config?: PeerConnectionConfig
  ) {
    this.localAgentId = localAgentId;
    this.signalingClient = signalingClient;
    this.config = {
      iceServers: config?.iceServers ?? DEFAULT_ICE_SERVERS,
      dataChannels: config?.dataChannels ?? DEFAULT_DATA_CHANNELS,
    };

    this.setupSignalingHandlers();
  }

  /**
   * Initiate connection to a peer
   */
  async connectToPeer(peerId: AgentId): Promise<PeerConnection> {
    // Don't connect to self
    if (peerId === this.localAgentId) {
      throw new Error('Cannot connect to self');
    }

    // Check if already connected
    const existing = this.peers.get(peerId);
    if (existing && existing.state === 'connected') {
      return existing;
    }

    // Create new connection
    const pc = this.createPeerConnection(peerId);

    // Create data channels (as initiator)
    for (const channelConfig of this.config.dataChannels) {
      const channel = pc.connection.createDataChannel(channelConfig.label, channelConfig.options);
      this.setupDataChannel(pc, channel);
    }

    // Create and send offer
    const offer = await pc.connection.createOffer();
    await pc.connection.setLocalDescription(offer);
    this.signalingClient.sendOffer(peerId, offer);

    return pc;
  }

  /**
   * Disconnect from a specific peer
   */
  disconnectFromPeer(peerId: AgentId): void {
    const pc = this.peers.get(peerId);
    if (pc) {
      for (const channel of pc.dataChannels.values()) {
        channel.close();
      }
      pc.connection.close();
      this.peers.delete(peerId);
      this.notifyConnectionChange(peerId, 'closed');
    }
  }

  /**
   * Disconnect from all peers
   */
  disconnectAll(): void {
    for (const peerId of this.peers.keys()) {
      this.disconnectFromPeer(peerId);
    }
  }

  /**
   * Send data to a specific peer
   */
  sendToPeer(
    peerId: AgentId,
    channelLabel: string,
    type: string,
    payload: unknown
  ): boolean {
    const pc = this.peers.get(peerId);
    if (!pc) {
      console.warn(`[PeerManager] No connection to peer ${peerId}`);
      return false;
    }

    const channel = pc.dataChannels.get(channelLabel);
    if (!channel || channel.readyState !== 'open') {
      console.warn(`[PeerManager] Channel ${channelLabel} not open for peer ${peerId}`);
      return false;
    }

    const message: DataChannelMessage = {
      type,
      payload,
      timestamp: Date.now(),
      from: this.localAgentId,
    };

    channel.send(JSON.stringify(message));
    return true;
  }

  /**
   * Broadcast data to all connected peers
   */
  broadcast(channelLabel: string, type: string, payload: unknown): number {
    let sent = 0;
    for (const [peerId] of this.peers) {
      if (this.sendToPeer(peerId, channelLabel, type, payload)) {
        sent++;
      }
    }
    return sent;
  }

  /**
   * Subscribe to data channel messages
   */
  onChannelMessage(channelLabel: string, handler: DataChannelHandler): () => void {
    if (!this.channelHandlers.has(channelLabel)) {
      this.channelHandlers.set(channelLabel, new Set());
    }
    this.channelHandlers.get(channelLabel)!.add(handler);

    return () => {
      this.channelHandlers.get(channelLabel)?.delete(handler);
    };
  }

  /**
   * Subscribe to peer connection state changes
   */
  onConnectionChange(
    listener: (peerId: AgentId, state: PeerConnectionState) => void
  ): () => void {
    this.connectionListeners.add(listener);
    return () => this.connectionListeners.delete(listener);
  }

  /**
   * Get all connected peer IDs
   */
  getConnectedPeers(): AgentId[] {
    return Array.from(this.peers.entries())
      .filter(([, pc]) => pc.state === 'connected')
      .map(([peerId]) => peerId);
  }

  /**
   * Get connection state for a peer
   */
  getPeerState(peerId: AgentId): PeerConnectionState | null {
    return this.peers.get(peerId)?.state ?? null;
  }

  // =============================================================================
  // Private Methods
  // =============================================================================

  private setupSignalingHandlers(): void {
    // Handle incoming offers
    this.signalingClient.on('offer', (message) => {
      void (async () => {
        if (message.to !== this.localAgentId) return;

        const pc = this.createPeerConnection(message.from);
        const offer = message.payload as RTCSessionDescriptionInit;

        await pc.connection.setRemoteDescription(new RTCSessionDescription(offer));
        const answer = await pc.connection.createAnswer();
        await pc.connection.setLocalDescription(answer);
        this.signalingClient.sendAnswer(message.from, answer);
      })();
    });

    // Handle incoming answers
    this.signalingClient.on('answer', (message) => {
      void (async () => {
        if (message.to !== this.localAgentId) return;

        const pc = this.peers.get(message.from);
        if (pc) {
          const answer = message.payload as RTCSessionDescriptionInit;
          await pc.connection.setRemoteDescription(new RTCSessionDescription(answer));
        }
      })();
    });

    // Handle ICE candidates
    this.signalingClient.on('ice-candidate', (message) => {
      void (async () => {
        if (message.to !== this.localAgentId) return;

        const pc = this.peers.get(message.from);
        if (pc) {
          const candidate = message.payload as RTCIceCandidateInit;
          await pc.connection.addIceCandidate(new RTCIceCandidate(candidate));
        }
      })();
    });

    // Handle peer leaving
    this.signalingClient.on('leave', (message) => {
      this.disconnectFromPeer(message.from);
    });
  }

  private createPeerConnection(peerId: AgentId): PeerConnection {
    // Close existing connection if any
    this.disconnectFromPeer(peerId);

    const connection = new RTCPeerConnection({
      iceServers: this.config.iceServers,
    });

    const pc: PeerConnection = {
      peerId,
      state: 'new',
      connection,
      dataChannels: new Map(),
      createdAt: Date.now(),
    };

    this.peers.set(peerId, pc);

    // ICE candidate handling
    connection.onicecandidate = (event) => {
      if (event.candidate) {
        this.signalingClient.sendIceCandidate(peerId, event.candidate);
      }
    };

    // Connection state tracking
    connection.onconnectionstatechange = () => {
      const state = this.mapConnectionState(connection.connectionState);
      pc.state = state;
      this.notifyConnectionChange(peerId, state);

      if (state === 'failed' || state === 'closed') {
        this.peers.delete(peerId);
      }
    };

    // Data channel handling (for non-initiator)
    connection.ondatachannel = (event) => {
      this.setupDataChannel(pc, event.channel);
    };

    return pc;
  }

  private setupDataChannel(pc: PeerConnection, channel: RTCDataChannel): void {
    pc.dataChannels.set(channel.label, channel);

    channel.onopen = () => {
      console.log(`[PeerManager] Channel ${channel.label} opened with ${pc.peerId}`);
    };

    channel.onclose = () => {
      console.log(`[PeerManager] Channel ${channel.label} closed with ${pc.peerId}`);
      pc.dataChannels.delete(channel.label);
    };

    channel.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data) as DataChannelMessage;
        this.handleChannelMessage(channel.label, message, channel);
      } catch (error) {
        console.error('[PeerManager] Failed to parse channel message:', error);
      }
    };

    channel.onerror = (error) => {
      console.error(`[PeerManager] Channel ${channel.label} error:`, error);
    };
  }

  private handleChannelMessage(
    channelLabel: string,
    message: DataChannelMessage,
    channel: RTCDataChannel
  ): void {
    const handlers = this.channelHandlers.get(channelLabel);
    if (handlers) {
      for (const handler of handlers) {
        try {
          handler(message, channel);
        } catch (error) {
          console.error('[PeerManager] Channel handler error:', error);
        }
      }
    }
  }

  private mapConnectionState(rtcState: RTCPeerConnectionState): PeerConnectionState {
    switch (rtcState) {
      case 'new':
        return 'new';
      case 'connecting':
        return 'connecting';
      case 'connected':
        return 'connected';
      case 'disconnected':
        return 'disconnected';
      case 'failed':
        return 'failed';
      case 'closed':
        return 'closed';
      default:
        return 'new';
    }
  }

  private notifyConnectionChange(peerId: AgentId, state: PeerConnectionState): void {
    for (const listener of this.connectionListeners) {
      try {
        listener(peerId, state);
      } catch (error) {
        console.error('[PeerManager] Connection listener error:', error);
      }
    }
  }
}

// =============================================================================
// Factory
// =============================================================================

export function createPeerManager(
  localAgentId: AgentId,
  signalingClient: SignalingClient,
  config?: PeerConnectionConfig
): PeerManager {
  return new PeerManager(localAgentId, signalingClient, config);
}
