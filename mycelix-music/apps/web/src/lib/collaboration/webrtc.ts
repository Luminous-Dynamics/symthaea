// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * WebRTC Manager
 *
 * Peer-to-peer connections for:
 * - Real-time audio streaming
 * - Data channel for CRDT sync
 * - Screen sharing
 * - Video chat
 */

// ==================== Types ====================

export type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'failed';

export interface PeerInfo {
  id: string;
  name: string;
  avatar?: string;
  role: 'host' | 'collaborator' | 'viewer';
}

export interface RTCConfig {
  iceServers: RTCIceServer[];
  enableAudio: boolean;
  enableVideo: boolean;
  enableDataChannel: boolean;
}

export interface DataMessage {
  type: string;
  payload: unknown;
  senderId: string;
  timestamp: number;
}

export interface AudioStreamConfig {
  sampleRate: number;
  channelCount: number;
  echoCancellation: boolean;
  noiseSuppression: boolean;
  autoGainControl: boolean;
}

// ==================== Signaling ====================

export interface SignalingMessage {
  type: 'offer' | 'answer' | 'ice-candidate' | 'join' | 'leave' | 'error';
  from: string;
  to?: string;
  payload: unknown;
  roomId: string;
}

export class SignalingClient {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private messageHandlers: Map<string, Set<(msg: SignalingMessage) => void>> = new Map();
  private pendingMessages: SignalingMessage[] = [];

  constructor(private url: string, private peerId: string) {}

  async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        console.log('Signaling connected');
        this.reconnectAttempts = 0;

        // Send pending messages
        for (const msg of this.pendingMessages) {
          this.send(msg);
        }
        this.pendingMessages = [];

        resolve();
      };

      this.ws.onerror = (error) => {
        console.error('Signaling error:', error);
        reject(error);
      };

      this.ws.onclose = () => {
        console.log('Signaling disconnected');
        this.attemptReconnect();
      };

      this.ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as SignalingMessage;
          this.handleMessage(message);
        } catch (error) {
          console.error('Failed to parse signaling message:', error);
        }
      };
    });
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnect attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

    setTimeout(() => {
      console.log(`Reconnecting... attempt ${this.reconnectAttempts}`);
      this.connect().catch(console.error);
    }, delay);
  }

  private handleMessage(message: SignalingMessage): void {
    const handlers = this.messageHandlers.get(message.type);
    if (handlers) {
      for (const handler of handlers) {
        handler(message);
      }
    }

    // Also call 'all' handlers
    const allHandlers = this.messageHandlers.get('*');
    if (allHandlers) {
      for (const handler of allHandlers) {
        handler(message);
      }
    }
  }

  on(type: string, handler: (msg: SignalingMessage) => void): () => void {
    if (!this.messageHandlers.has(type)) {
      this.messageHandlers.set(type, new Set());
    }
    this.messageHandlers.get(type)!.add(handler);

    return () => {
      this.messageHandlers.get(type)?.delete(handler);
    };
  }

  send(message: SignalingMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      this.pendingMessages.push(message);
    }
  }

  disconnect(): void {
    this.ws?.close();
    this.ws = null;
    this.messageHandlers.clear();
    this.pendingMessages = [];
  }
}

// ==================== Peer Connection ====================

export class PeerConnection {
  private pc: RTCPeerConnection;
  private dataChannel: RTCDataChannel | null = null;
  private localStream: MediaStream | null = null;
  private remoteStream: MediaStream | null = null;
  private audioContext: AudioContext | null = null;
  private audioWorklet: AudioWorkletNode | null = null;

  public state: ConnectionState = 'disconnected';
  public readonly peerId: string;
  public peerInfo: PeerInfo | null = null;

  private onStateChange?: (state: ConnectionState) => void;
  private onDataMessage?: (message: DataMessage) => void;
  private onRemoteStream?: (stream: MediaStream) => void;
  private onAudioData?: (data: Float32Array) => void;

  constructor(
    peerId: string,
    private config: RTCConfig,
    callbacks: {
      onStateChange?: (state: ConnectionState) => void;
      onDataMessage?: (message: DataMessage) => void;
      onRemoteStream?: (stream: MediaStream) => void;
      onAudioData?: (data: Float32Array) => void;
    }
  ) {
    this.peerId = peerId;
    this.onStateChange = callbacks.onStateChange;
    this.onDataMessage = callbacks.onDataMessage;
    this.onRemoteStream = callbacks.onRemoteStream;
    this.onAudioData = callbacks.onAudioData;

    this.pc = new RTCPeerConnection({
      iceServers: config.iceServers,
    });

    this.setupPeerConnection();
  }

  private setupPeerConnection(): void {
    this.pc.onicecandidate = (event) => {
      if (event.candidate) {
        this.onIceCandidate?.(event.candidate);
      }
    };

    this.pc.oniceconnectionstatechange = () => {
      switch (this.pc.iceConnectionState) {
        case 'connected':
        case 'completed':
          this.setState('connected');
          break;
        case 'disconnected':
          this.setState('disconnected');
          break;
        case 'failed':
          this.setState('failed');
          break;
        case 'checking':
          this.setState('connecting');
          break;
      }
    };

    this.pc.ontrack = (event) => {
      this.remoteStream = event.streams[0];
      this.onRemoteStream?.(this.remoteStream);

      // Setup audio processing if needed
      if (this.onAudioData && event.track.kind === 'audio') {
        this.setupAudioProcessing(event.streams[0]);
      }
    };

    this.pc.ondatachannel = (event) => {
      this.setupDataChannel(event.channel);
    };
  }

  private setState(state: ConnectionState): void {
    this.state = state;
    this.onStateChange?.(state);
  }

  private setupDataChannel(channel: RTCDataChannel): void {
    this.dataChannel = channel;

    channel.onopen = () => {
      console.log(`Data channel open with ${this.peerId}`);
    };

    channel.onclose = () => {
      console.log(`Data channel closed with ${this.peerId}`);
    };

    channel.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data) as DataMessage;
        this.onDataMessage?.(message);
      } catch (error) {
        console.error('Failed to parse data message:', error);
      }
    };
  }

  private async setupAudioProcessing(stream: MediaStream): Promise<void> {
    this.audioContext = new AudioContext({ sampleRate: 48000 });

    // Load audio worklet for processing
    await this.audioContext.audioWorklet.addModule('/audio-worklets/stream-processor.js');

    const source = this.audioContext.createMediaStreamSource(stream);
    this.audioWorklet = new AudioWorkletNode(this.audioContext, 'stream-processor');

    this.audioWorklet.port.onmessage = (event) => {
      if (event.data.type === 'audio') {
        this.onAudioData?.(event.data.samples);
      }
    };

    source.connect(this.audioWorklet);
    this.audioWorklet.connect(this.audioContext.destination);
  }

  // ICE candidate handler (set by WebRTCManager)
  public onIceCandidate?: (candidate: RTCIceCandidate) => void;

  async createOffer(): Promise<RTCSessionDescriptionInit> {
    // Create data channel if enabled
    if (this.config.enableDataChannel) {
      const channel = this.pc.createDataChannel('data', {
        ordered: true,
      });
      this.setupDataChannel(channel);
    }

    const offer = await this.pc.createOffer({
      offerToReceiveAudio: this.config.enableAudio,
      offerToReceiveVideo: this.config.enableVideo,
    });

    await this.pc.setLocalDescription(offer);
    return offer;
  }

  async handleOffer(offer: RTCSessionDescriptionInit): Promise<RTCSessionDescriptionInit> {
    await this.pc.setRemoteDescription(new RTCSessionDescription(offer));
    const answer = await this.pc.createAnswer();
    await this.pc.setLocalDescription(answer);
    return answer;
  }

  async handleAnswer(answer: RTCSessionDescriptionInit): Promise<void> {
    await this.pc.setRemoteDescription(new RTCSessionDescription(answer));
  }

  async addIceCandidate(candidate: RTCIceCandidateInit): Promise<void> {
    await this.pc.addIceCandidate(new RTCIceCandidate(candidate));
  }

  async addLocalStream(stream: MediaStream): Promise<void> {
    this.localStream = stream;
    for (const track of stream.getTracks()) {
      this.pc.addTrack(track, stream);
    }
  }

  sendData(message: Omit<DataMessage, 'timestamp'>): void {
    if (this.dataChannel?.readyState === 'open') {
      this.dataChannel.send(JSON.stringify({
        ...message,
        timestamp: Date.now(),
      }));
    }
  }

  close(): void {
    this.dataChannel?.close();
    this.pc.close();
    this.localStream?.getTracks().forEach(t => t.stop());
    this.audioContext?.close();
    this.setState('disconnected');
  }

  getStats(): Promise<RTCStatsReport> {
    return this.pc.getStats();
  }
}

// ==================== WebRTC Manager ====================

export class WebRTCManager {
  private signaling: SignalingClient;
  private peers: Map<string, PeerConnection> = new Map();
  private localStream: MediaStream | null = null;
  private config: RTCConfig;
  private roomId: string = '';
  private localPeerId: string;

  public onPeerJoined?: (peer: PeerInfo) => void;
  public onPeerLeft?: (peerId: string) => void;
  public onDataMessage?: (peerId: string, message: DataMessage) => void;
  public onRemoteStream?: (peerId: string, stream: MediaStream) => void;
  public onAudioData?: (peerId: string, data: Float32Array) => void;
  public onConnectionStateChange?: (peerId: string, state: ConnectionState) => void;

  constructor(
    signalingUrl: string,
    localPeerId: string,
    config: Partial<RTCConfig> = {}
  ) {
    this.localPeerId = localPeerId;
    this.signaling = new SignalingClient(signalingUrl, localPeerId);
    this.config = {
      iceServers: config.iceServers || [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' },
      ],
      enableAudio: config.enableAudio ?? true,
      enableVideo: config.enableVideo ?? false,
      enableDataChannel: config.enableDataChannel ?? true,
    };

    this.setupSignalingHandlers();
  }

  private setupSignalingHandlers(): void {
    this.signaling.on('offer', async (msg) => {
      const peer = this.getOrCreatePeer(msg.from);
      const answer = await peer.handleOffer(msg.payload as RTCSessionDescriptionInit);

      if (this.localStream) {
        await peer.addLocalStream(this.localStream);
      }

      this.signaling.send({
        type: 'answer',
        from: this.localPeerId,
        to: msg.from,
        payload: answer,
        roomId: this.roomId,
      });
    });

    this.signaling.on('answer', async (msg) => {
      const peer = this.peers.get(msg.from);
      if (peer) {
        await peer.handleAnswer(msg.payload as RTCSessionDescriptionInit);
      }
    });

    this.signaling.on('ice-candidate', async (msg) => {
      const peer = this.peers.get(msg.from);
      if (peer) {
        await peer.addIceCandidate(msg.payload as RTCIceCandidateInit);
      }
    });

    this.signaling.on('join', (msg) => {
      const peerInfo = msg.payload as PeerInfo;
      this.onPeerJoined?.(peerInfo);

      // Initiate connection to new peer
      this.connectToPeer(peerInfo.id);
    });

    this.signaling.on('leave', (msg) => {
      const peerId = msg.from;
      this.peers.get(peerId)?.close();
      this.peers.delete(peerId);
      this.onPeerLeft?.(peerId);
    });
  }

  private getOrCreatePeer(peerId: string): PeerConnection {
    let peer = this.peers.get(peerId);
    if (!peer) {
      peer = new PeerConnection(peerId, this.config, {
        onStateChange: (state) => this.onConnectionStateChange?.(peerId, state),
        onDataMessage: (msg) => this.onDataMessage?.(peerId, msg),
        onRemoteStream: (stream) => this.onRemoteStream?.(peerId, stream),
        onAudioData: (data) => this.onAudioData?.(peerId, data),
      });

      peer.onIceCandidate = (candidate) => {
        this.signaling.send({
          type: 'ice-candidate',
          from: this.localPeerId,
          to: peerId,
          payload: candidate.toJSON(),
          roomId: this.roomId,
        });
      };

      this.peers.set(peerId, peer);
    }
    return peer;
  }

  async connect(): Promise<void> {
    await this.signaling.connect();
  }

  async joinRoom(roomId: string, peerInfo: PeerInfo): Promise<void> {
    this.roomId = roomId;

    // Get local media if needed
    if (this.config.enableAudio || this.config.enableVideo) {
      this.localStream = await navigator.mediaDevices.getUserMedia({
        audio: this.config.enableAudio ? {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 48000,
        } : false,
        video: this.config.enableVideo,
      });
    }

    // Announce join
    this.signaling.send({
      type: 'join',
      from: this.localPeerId,
      payload: peerInfo,
      roomId,
    });
  }

  private async connectToPeer(peerId: string): Promise<void> {
    const peer = this.getOrCreatePeer(peerId);

    if (this.localStream) {
      await peer.addLocalStream(this.localStream);
    }

    const offer = await peer.createOffer();

    this.signaling.send({
      type: 'offer',
      from: this.localPeerId,
      to: peerId,
      payload: offer,
      roomId: this.roomId,
    });
  }

  sendToAll(message: Omit<DataMessage, 'senderId' | 'timestamp'>): void {
    for (const peer of this.peers.values()) {
      peer.sendData({
        ...message,
        senderId: this.localPeerId,
      });
    }
  }

  sendToPeer(peerId: string, message: Omit<DataMessage, 'senderId' | 'timestamp'>): void {
    this.peers.get(peerId)?.sendData({
      ...message,
      senderId: this.localPeerId,
    });
  }

  leaveRoom(): void {
    this.signaling.send({
      type: 'leave',
      from: this.localPeerId,
      roomId: this.roomId,
      payload: null,
    });

    for (const peer of this.peers.values()) {
      peer.close();
    }
    this.peers.clear();

    this.localStream?.getTracks().forEach(t => t.stop());
    this.localStream = null;
  }

  disconnect(): void {
    this.leaveRoom();
    this.signaling.disconnect();
  }

  getPeers(): PeerConnection[] {
    return Array.from(this.peers.values());
  }

  getLocalStream(): MediaStream | null {
    return this.localStream;
  }

  async getConnectionStats(): Promise<Map<string, RTCStatsReport>> {
    const stats = new Map<string, RTCStatsReport>();
    for (const [peerId, peer] of this.peers) {
      stats.set(peerId, await peer.getStats());
    }
    return stats;
  }
}

export default WebRTCManager;
