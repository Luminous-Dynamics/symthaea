// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * WebRTC Signaling Service for P2P Real-Time Messaging
 *
 * Features:
 * - Direct P2P connections for instant delivery
 * - Holochain-based signaling (no central server)
 * - Automatic fallback to DHT delivery
 * - ICE candidate handling
 * - Connection state management
 */

import { EventEmitter } from 'events';

export interface PeerConnection {
  id: string;
  agentPubKey: string;
  connection: RTCPeerConnection;
  dataChannel: RTCDataChannel | null;
  state: 'connecting' | 'connected' | 'disconnected' | 'failed';
  lastSeen: number;
}

export interface SignalingMessage {
  type: 'offer' | 'answer' | 'ice-candidate' | 'ping' | 'pong';
  from: string;
  to: string;
  payload: any;
  timestamp: number;
}

export interface WebRTCConfig {
  iceServers?: RTCIceServer[];
  maxPeers?: number;
  connectionTimeout?: number;
  keepAliveInterval?: number;
}

const DEFAULT_ICE_SERVERS: RTCIceServer[] = [
  { urls: 'stun:stun.l.google.com:19302' },
  { urls: 'stun:stun1.l.google.com:19302' },
];

export class SignalingService extends EventEmitter {
  private myAgentPubKey: string;
  private peers: Map<string, PeerConnection> = new Map();
  private config: Required<WebRTCConfig>;
  private holochain: any; // Holochain client for signaling
  private keepAliveTimer: ReturnType<typeof setInterval> | null = null;

  constructor(agentPubKey: string, holochain: any, config: WebRTCConfig = {}) {
    super();
    this.myAgentPubKey = agentPubKey;
    this.holochain = holochain;
    this.config = {
      iceServers: config.iceServers || DEFAULT_ICE_SERVERS,
      maxPeers: config.maxPeers || 50,
      connectionTimeout: config.connectionTimeout || 30000,
      keepAliveInterval: config.keepAliveInterval || 15000,
    };
  }

  /**
   * Start the signaling service
   */
  async start(): Promise<void> {
    // Subscribe to signaling messages from Holochain
    this.holochain.on('signal', (signal: any) => {
      if (signal.type === 'webrtc') {
        this.handleSignalingMessage(signal.payload);
      }
    });

    // Start keep-alive pings
    this.keepAliveTimer = setInterval(() => {
      this.sendKeepAlive();
    }, this.config.keepAliveInterval);

    this.emit('started');
  }

  /**
   * Stop the signaling service
   */
  stop(): void {
    if (this.keepAliveTimer) {
      clearInterval(this.keepAliveTimer);
    }

    // Close all peer connections
    for (const peer of this.peers.values()) {
      this.disconnectPeer(peer.agentPubKey);
    }

    this.emit('stopped');
  }

  /**
   * Connect to a peer
   */
  async connectToPeer(agentPubKey: string): Promise<PeerConnection> {
    if (this.peers.has(agentPubKey)) {
      return this.peers.get(agentPubKey)!;
    }

    if (this.peers.size >= this.config.maxPeers) {
      throw new Error('Maximum peer connections reached');
    }

    const connection = new RTCPeerConnection({
      iceServers: this.config.iceServers,
    });

    const peer: PeerConnection = {
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      agentPubKey,
      connection,
      dataChannel: null,
      state: 'connecting',
      lastSeen: Date.now(),
    };

    this.setupConnectionHandlers(peer);

    // Create data channel
    const dataChannel = connection.createDataChannel('mycelix-mail', {
      ordered: true,
    });
    this.setupDataChannelHandlers(peer, dataChannel);
    peer.dataChannel = dataChannel;

    // Create and send offer
    const offer = await connection.createOffer();
    await connection.setLocalDescription(offer);

    await this.sendSignalingMessage({
      type: 'offer',
      from: this.myAgentPubKey,
      to: agentPubKey,
      payload: {
        sdp: offer.sdp,
        type: offer.type,
      },
      timestamp: Date.now(),
    });

    this.peers.set(agentPubKey, peer);

    // Set connection timeout
    setTimeout(() => {
      if (peer.state === 'connecting') {
        this.emit('connection-timeout', agentPubKey);
        this.disconnectPeer(agentPubKey);
      }
    }, this.config.connectionTimeout);

    return peer;
  }

  /**
   * Disconnect from a peer
   */
  disconnectPeer(agentPubKey: string): void {
    const peer = this.peers.get(agentPubKey);
    if (!peer) return;

    peer.dataChannel?.close();
    peer.connection.close();
    this.peers.delete(agentPubKey);

    this.emit('peer-disconnected', agentPubKey);
  }

  /**
   * Send data to a peer
   */
  async sendToPeer(agentPubKey: string, data: any): Promise<boolean> {
    let peer = this.peers.get(agentPubKey);

    // Try to connect if not connected
    if (!peer || peer.state !== 'connected') {
      try {
        peer = await this.connectToPeer(agentPubKey);
        // Wait for connection
        await this.waitForConnection(peer);
      } catch {
        return false;
      }
    }

    if (peer.dataChannel?.readyState === 'open') {
      peer.dataChannel.send(JSON.stringify(data));
      return true;
    }

    return false;
  }

  /**
   * Broadcast data to all connected peers
   */
  broadcast(data: any): void {
    const message = JSON.stringify(data);
    for (const peer of this.peers.values()) {
      if (peer.dataChannel?.readyState === 'open') {
        peer.dataChannel.send(message);
      }
    }
  }

  /**
   * Get connected peers
   */
  getConnectedPeers(): string[] {
    return Array.from(this.peers.values())
      .filter((p) => p.state === 'connected')
      .map((p) => p.agentPubKey);
  }

  /**
   * Check if peer is connected
   */
  isPeerConnected(agentPubKey: string): boolean {
    const peer = this.peers.get(agentPubKey);
    return peer?.state === 'connected';
  }

  /**
   * Get peer connection stats
   */
  async getPeerStats(agentPubKey: string): Promise<RTCStatsReport | null> {
    const peer = this.peers.get(agentPubKey);
    if (!peer) return null;
    return peer.connection.getStats();
  }

  // Private methods

  private setupConnectionHandlers(peer: PeerConnection): void {
    peer.connection.onicecandidate = (event) => {
      if (event.candidate) {
        this.sendSignalingMessage({
          type: 'ice-candidate',
          from: this.myAgentPubKey,
          to: peer.agentPubKey,
          payload: event.candidate.toJSON(),
          timestamp: Date.now(),
        });
      }
    };

    peer.connection.onconnectionstatechange = () => {
      const state = peer.connection.connectionState;

      switch (state) {
        case 'connected':
          peer.state = 'connected';
          this.emit('peer-connected', peer.agentPubKey);
          break;
        case 'disconnected':
        case 'failed':
          peer.state = state as 'disconnected' | 'failed';
          this.emit('peer-disconnected', peer.agentPubKey);
          this.peers.delete(peer.agentPubKey);
          break;
      }
    };

    peer.connection.ondatachannel = (event) => {
      this.setupDataChannelHandlers(peer, event.channel);
      peer.dataChannel = event.channel;
    };
  }

  private setupDataChannelHandlers(
    peer: PeerConnection,
    channel: RTCDataChannel
  ): void {
    channel.onopen = () => {
      this.emit('channel-open', peer.agentPubKey);
    };

    channel.onclose = () => {
      this.emit('channel-close', peer.agentPubKey);
    };

    channel.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        peer.lastSeen = Date.now();
        this.emit('message', { from: peer.agentPubKey, data });
      } catch (e) {
        console.error('[WebRTC] Failed to parse message:', e);
      }
    };

    channel.onerror = (error) => {
      console.error('[WebRTC] Data channel error:', error);
      this.emit('error', { peer: peer.agentPubKey, error });
    };
  }

  private async handleSignalingMessage(message: SignalingMessage): Promise<void> {
    if (message.to !== this.myAgentPubKey) return;

    switch (message.type) {
      case 'offer':
        await this.handleOffer(message);
        break;
      case 'answer':
        await this.handleAnswer(message);
        break;
      case 'ice-candidate':
        await this.handleIceCandidate(message);
        break;
      case 'ping':
        await this.handlePing(message);
        break;
      case 'pong':
        this.handlePong(message);
        break;
    }
  }

  private async handleOffer(message: SignalingMessage): Promise<void> {
    let peer = this.peers.get(message.from);

    if (!peer) {
      const connection = new RTCPeerConnection({
        iceServers: this.config.iceServers,
      });

      peer = {
        id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        agentPubKey: message.from,
        connection,
        dataChannel: null,
        state: 'connecting',
        lastSeen: Date.now(),
      };

      this.setupConnectionHandlers(peer);
      this.peers.set(message.from, peer);
    }

    await peer.connection.setRemoteDescription(
      new RTCSessionDescription(message.payload)
    );

    const answer = await peer.connection.createAnswer();
    await peer.connection.setLocalDescription(answer);

    await this.sendSignalingMessage({
      type: 'answer',
      from: this.myAgentPubKey,
      to: message.from,
      payload: {
        sdp: answer.sdp,
        type: answer.type,
      },
      timestamp: Date.now(),
    });
  }

  private async handleAnswer(message: SignalingMessage): Promise<void> {
    const peer = this.peers.get(message.from);
    if (!peer) return;

    await peer.connection.setRemoteDescription(
      new RTCSessionDescription(message.payload)
    );
  }

  private async handleIceCandidate(message: SignalingMessage): Promise<void> {
    const peer = this.peers.get(message.from);
    if (!peer) return;

    await peer.connection.addIceCandidate(new RTCIceCandidate(message.payload));
  }

  private async handlePing(message: SignalingMessage): Promise<void> {
    await this.sendSignalingMessage({
      type: 'pong',
      from: this.myAgentPubKey,
      to: message.from,
      payload: { timestamp: message.payload.timestamp },
      timestamp: Date.now(),
    });
  }

  private handlePong(message: SignalingMessage): void {
    const peer = this.peers.get(message.from);
    if (peer) {
      peer.lastSeen = Date.now();
      const latency = Date.now() - message.payload.timestamp;
      this.emit('latency', { peer: message.from, latency });
    }
  }

  private async sendSignalingMessage(message: SignalingMessage): Promise<void> {
    // Send via Holochain signals
    await this.holochain.callZome({
      cap_secret: null,
      zome_name: 'federation',
      fn_name: 'send_signal',
      payload: {
        to_agent: message.to,
        signal: {
          type: 'webrtc',
          payload: message,
        },
      },
    });
  }

  private sendKeepAlive(): void {
    for (const peer of this.peers.values()) {
      if (peer.state === 'connected') {
        this.sendSignalingMessage({
          type: 'ping',
          from: this.myAgentPubKey,
          to: peer.agentPubKey,
          payload: { timestamp: Date.now() },
          timestamp: Date.now(),
        });
      }
    }
  }

  private waitForConnection(peer: PeerConnection): Promise<void> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Connection timeout'));
      }, this.config.connectionTimeout);

      const checkState = () => {
        if (peer.state === 'connected') {
          clearTimeout(timeout);
          resolve();
        } else if (peer.state === 'failed') {
          clearTimeout(timeout);
          reject(new Error('Connection failed'));
        } else {
          setTimeout(checkState, 100);
        }
      };

      checkState();
    });
  }
}

export default SignalingService;
