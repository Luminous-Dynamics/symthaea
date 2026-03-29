// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Live Streaming
 *
 * Low-latency audio/video broadcasting:
 * - WebRTC and RTMP ingest
 * - Virtual concerts
 * - Live chat and reactions
 * - Multi-source mixing
 * - Recording and VOD
 * - Monetization (tickets, tips)
 */

// ==================== Types ====================

export interface LiveStream {
  id: string;
  title: string;
  description: string;
  host: StreamHost;
  status: StreamStatus;
  visibility: 'public' | 'private' | 'subscribers' | 'ticket';
  startTime?: Date;
  scheduledFor?: Date;
  viewers: number;
  peakViewers: number;
  likes: number;
  thumbnail: string;
  tags: string[];
  category: string;
  settings: StreamSettings;
  monetization: StreamMonetization;
}

export type StreamStatus = 'scheduled' | 'live' | 'ended' | 'cancelled';

export interface StreamHost {
  id: string;
  name: string;
  avatar: string;
  verified: boolean;
  followers: number;
}

export interface StreamSettings {
  audioQuality: 'low' | 'medium' | 'high' | 'lossless';
  videoEnabled: boolean;
  videoQuality: '480p' | '720p' | '1080p' | '4k';
  latencyMode: 'ultra-low' | 'low' | 'normal';
  chatEnabled: boolean;
  reactionsEnabled: boolean;
  recordStream: boolean;
  dvr: boolean;
  maxViewers?: number;
}

export interface StreamMonetization {
  ticketed: boolean;
  ticketPrice?: number;
  tipsEnabled: boolean;
  minTip?: number;
  subscriptionRequired: boolean;
}

export interface StreamKey {
  rtmpUrl: string;
  streamKey: string;
  webrtcUrl: string;
  expiresAt: Date;
}

export interface StreamMetrics {
  viewers: number;
  peakViewers: number;
  chatMessages: number;
  reactions: number;
  tips: number;
  tipsAmount: number;
  averageWatchTime: number;
  bufferRatio: number;
  bitrate: number;
  frameRate: number;
  latency: number;
}

export interface ChatMessage {
  id: string;
  userId: string;
  userName: string;
  userAvatar: string;
  userBadges: string[];
  content: string;
  timestamp: Date;
  type: 'message' | 'tip' | 'subscription' | 'system';
  tipAmount?: number;
  highlighted?: boolean;
}

export interface Reaction {
  type: 'heart' | 'fire' | 'clap' | 'wow' | 'sad' | 'laugh';
  count: number;
  userId: string;
}

export interface VirtualConcert {
  id: string;
  stream: LiveStream;
  venue: ConcertVenue;
  setlist: SetlistItem[];
  merchandise: MerchandiseItem[];
  vipFeatures: VIPFeature[];
}

export interface ConcertVenue {
  id: string;
  name: string;
  theme: string;
  capacity: number;
  features: string[];
  backgroundUrl: string;
  lightingPresets: string[];
}

export interface SetlistItem {
  trackId: string;
  title: string;
  duration: number;
  scheduled?: Date;
  played: boolean;
}

export interface MerchandiseItem {
  id: string;
  name: string;
  price: number;
  image: string;
  type: 'physical' | 'digital';
}

export interface VIPFeature {
  id: string;
  name: string;
  description: string;
  price: number;
  features: string[];
}

export interface StreamSource {
  id: string;
  type: 'camera' | 'screen' | 'audio' | 'media' | 'rtmp';
  name: string;
  stream?: MediaStream;
  url?: string;
  active: boolean;
  volume: number;
  position?: { x: number; y: number; width: number; height: number };
}

// ==================== Stream Manager ====================

export class StreamManager {
  private baseUrl: string;

  constructor(baseUrl = '/api/streaming') {
    this.baseUrl = baseUrl;
  }

  async createStream(params: {
    title: string;
    description: string;
    scheduledFor?: Date;
    visibility: LiveStream['visibility'];
    settings: Partial<StreamSettings>;
    monetization?: Partial<StreamMonetization>;
  }): Promise<LiveStream> {
    const response = await fetch(`${this.baseUrl}/streams`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return response.json();
  }

  async getStream(streamId: string): Promise<LiveStream> {
    const response = await fetch(`${this.baseUrl}/streams/${streamId}`);
    return response.json();
  }

  async updateStream(streamId: string, updates: Partial<LiveStream>): Promise<LiveStream> {
    const response = await fetch(`${this.baseUrl}/streams/${streamId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
    return response.json();
  }

  async deleteStream(streamId: string): Promise<void> {
    await fetch(`${this.baseUrl}/streams/${streamId}`, { method: 'DELETE' });
  }

  async getStreamKey(streamId: string): Promise<StreamKey> {
    const response = await fetch(`${this.baseUrl}/streams/${streamId}/key`);
    return response.json();
  }

  async regenerateStreamKey(streamId: string): Promise<StreamKey> {
    const response = await fetch(`${this.baseUrl}/streams/${streamId}/key`, {
      method: 'POST',
    });
    return response.json();
  }

  async startStream(streamId: string): Promise<void> {
    await fetch(`${this.baseUrl}/streams/${streamId}/start`, { method: 'POST' });
  }

  async endStream(streamId: string): Promise<void> {
    await fetch(`${this.baseUrl}/streams/${streamId}/end`, { method: 'POST' });
  }

  async getMyStreams(): Promise<LiveStream[]> {
    const response = await fetch(`${this.baseUrl}/streams/mine`);
    return response.json();
  }

  async getScheduledStreams(): Promise<LiveStream[]> {
    const response = await fetch(`${this.baseUrl}/streams/scheduled`);
    return response.json();
  }

  async getLiveStreams(category?: string): Promise<LiveStream[]> {
    const url = category
      ? `${this.baseUrl}/streams/live?category=${category}`
      : `${this.baseUrl}/streams/live`;
    const response = await fetch(url);
    return response.json();
  }

  async getStreamMetrics(streamId: string): Promise<StreamMetrics> {
    const response = await fetch(`${this.baseUrl}/streams/${streamId}/metrics`);
    return response.json();
  }
}

// ==================== WebRTC Broadcaster ====================

export class WebRTCBroadcaster {
  private peerConnection: RTCPeerConnection | null = null;
  private localStream: MediaStream | null = null;
  private dataChannel: RTCDataChannel | null = null;
  private signalingUrl: string;
  private streamId: string;
  private isConnected = false;

  public onConnectionStateChange?: (state: RTCPeerConnectionState) => void;
  public onError?: (error: Error) => void;
  public onStats?: (stats: RTCStatsReport) => void;

  constructor(signalingUrl: string, streamId: string) {
    this.signalingUrl = signalingUrl;
    this.streamId = streamId;
  }

  async start(stream: MediaStream): Promise<void> {
    this.localStream = stream;

    // Create peer connection with TURN/STUN servers
    this.peerConnection = new RTCPeerConnection({
      iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' },
      ],
      bundlePolicy: 'max-bundle',
      rtcpMuxPolicy: 'require',
    });

    // Add tracks
    for (const track of stream.getTracks()) {
      this.peerConnection.addTrack(track, stream);
    }

    // Create data channel for metadata
    this.dataChannel = this.peerConnection.createDataChannel('metadata', {
      ordered: true,
    });

    this.setupEventHandlers();

    // Create and send offer
    const offer = await this.peerConnection.createOffer({
      offerToReceiveAudio: false,
      offerToReceiveVideo: false,
    });

    await this.peerConnection.setLocalDescription(offer);

    // Send offer to signaling server
    const response = await fetch(`${this.signalingUrl}/offer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        streamId: this.streamId,
        sdp: offer.sdp,
        type: offer.type,
      }),
    });

    const answer = await response.json();
    await this.peerConnection.setRemoteDescription(new RTCSessionDescription(answer));

    this.isConnected = true;

    // Start stats monitoring
    this.startStatsMonitoring();
  }

  private setupEventHandlers(): void {
    if (!this.peerConnection) return;

    this.peerConnection.onconnectionstatechange = () => {
      this.onConnectionStateChange?.(this.peerConnection!.connectionState);

      if (this.peerConnection!.connectionState === 'failed') {
        this.onError?.(new Error('Connection failed'));
      }
    };

    this.peerConnection.onicecandidate = async (event) => {
      if (event.candidate) {
        await fetch(`${this.signalingUrl}/ice`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            streamId: this.streamId,
            candidate: event.candidate,
          }),
        });
      }
    };
  }

  private startStatsMonitoring(): void {
    setInterval(async () => {
      if (this.peerConnection && this.isConnected) {
        const stats = await this.peerConnection.getStats();
        this.onStats?.(stats);
      }
    }, 1000);
  }

  sendMetadata(data: any): void {
    if (this.dataChannel?.readyState === 'open') {
      this.dataChannel.send(JSON.stringify(data));
    }
  }

  async replaceTrack(newTrack: MediaStreamTrack): Promise<void> {
    const sender = this.peerConnection?.getSenders().find(
      s => s.track?.kind === newTrack.kind
    );

    if (sender) {
      await sender.replaceTrack(newTrack);
    }
  }

  stop(): void {
    if (this.localStream) {
      this.localStream.getTracks().forEach(track => track.stop());
    }

    if (this.dataChannel) {
      this.dataChannel.close();
    }

    if (this.peerConnection) {
      this.peerConnection.close();
    }

    this.isConnected = false;
  }
}

// ==================== Stream Viewer ====================

export class StreamViewer {
  private peerConnection: RTCPeerConnection | null = null;
  private remoteStream: MediaStream | null = null;
  private signalingUrl: string;
  private streamId: string;
  private videoElement: HTMLVideoElement | null = null;
  private audioElement: HTMLAudioElement | null = null;

  public onStream?: (stream: MediaStream) => void;
  public onMetadata?: (data: any) => void;
  public onBuffering?: (isBuffering: boolean) => void;
  public onLatency?: (latencyMs: number) => void;

  constructor(signalingUrl: string, streamId: string) {
    this.signalingUrl = signalingUrl;
    this.streamId = streamId;
  }

  async connect(options: {
    videoElement?: HTMLVideoElement;
    audioElement?: HTMLAudioElement;
    lowLatency?: boolean;
  } = {}): Promise<void> {
    this.videoElement = options.videoElement || null;
    this.audioElement = options.audioElement || null;

    this.peerConnection = new RTCPeerConnection({
      iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
      ],
    });

    this.peerConnection.ontrack = (event) => {
      if (!this.remoteStream) {
        this.remoteStream = new MediaStream();
      }

      this.remoteStream.addTrack(event.track);

      if (this.videoElement && event.track.kind === 'video') {
        this.videoElement.srcObject = this.remoteStream;
      }

      if (this.audioElement && event.track.kind === 'audio') {
        this.audioElement.srcObject = this.remoteStream;
      }

      this.onStream?.(this.remoteStream);
    };

    // Request stream
    const response = await fetch(`${this.signalingUrl}/watch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        streamId: this.streamId,
        lowLatency: options.lowLatency,
      }),
    });

    const offer = await response.json();
    await this.peerConnection.setRemoteDescription(new RTCSessionDescription(offer));

    const answer = await this.peerConnection.createAnswer();
    await this.peerConnection.setLocalDescription(answer);

    await fetch(`${this.signalingUrl}/answer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        streamId: this.streamId,
        sdp: answer.sdp,
        type: answer.type,
      }),
    });

    // Handle ICE candidates
    this.peerConnection.onicecandidate = async (event) => {
      if (event.candidate) {
        await fetch(`${this.signalingUrl}/ice`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            streamId: this.streamId,
            candidate: event.candidate,
          }),
        });
      }
    };

    // Monitor latency
    this.startLatencyMonitoring();
  }

  private startLatencyMonitoring(): void {
    setInterval(async () => {
      if (this.peerConnection) {
        const stats = await this.peerConnection.getStats();
        stats.forEach(stat => {
          if (stat.type === 'inbound-rtp' && stat.kind === 'video') {
            const latency = stat.jitterBufferDelay / stat.jitterBufferEmittedCount * 1000;
            this.onLatency?.(latency || 0);
          }
        });
      }
    }, 1000);
  }

  setVolume(volume: number): void {
    if (this.audioElement) {
      this.audioElement.volume = Math.max(0, Math.min(1, volume));
    }
    if (this.videoElement) {
      this.videoElement.volume = Math.max(0, Math.min(1, volume));
    }
  }

  setQuality(quality: 'auto' | 'low' | 'medium' | 'high'): void {
    // Request quality change from server
    fetch(`${this.signalingUrl}/quality`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        streamId: this.streamId,
        quality,
      }),
    });
  }

  disconnect(): void {
    if (this.peerConnection) {
      this.peerConnection.close();
    }

    if (this.remoteStream) {
      this.remoteStream.getTracks().forEach(track => track.stop());
    }

    if (this.videoElement) {
      this.videoElement.srcObject = null;
    }

    if (this.audioElement) {
      this.audioElement.srcObject = null;
    }
  }
}

// ==================== Live Chat ====================

export class LiveChat {
  private websocket: WebSocket | null = null;
  private streamId: string;
  private userId: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;

  public onMessage?: (message: ChatMessage) => void;
  public onReaction?: (reaction: Reaction) => void;
  public onViewerCount?: (count: number) => void;
  public onConnectionChange?: (connected: boolean) => void;

  constructor(streamId: string, userId: string) {
    this.streamId = streamId;
    this.userId = userId;
  }

  connect(wsUrl: string): void {
    this.websocket = new WebSocket(`${wsUrl}?streamId=${this.streamId}&userId=${this.userId}`);

    this.websocket.onopen = () => {
      this.reconnectAttempts = 0;
      this.onConnectionChange?.(true);
    };

    this.websocket.onclose = () => {
      this.onConnectionChange?.(false);
      this.attemptReconnect(wsUrl);
    };

    this.websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case 'message':
          this.onMessage?.(data.message);
          break;
        case 'reaction':
          this.onReaction?.(data.reaction);
          break;
        case 'viewers':
          this.onViewerCount?.(data.count);
          break;
      }
    };
  }

  private attemptReconnect(wsUrl: string): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        this.connect(wsUrl);
      }, Math.pow(2, this.reconnectAttempts) * 1000);
    }
  }

  sendMessage(content: string): void {
    if (this.websocket?.readyState === WebSocket.OPEN) {
      this.websocket.send(JSON.stringify({
        type: 'message',
        content,
      }));
    }
  }

  sendReaction(reactionType: Reaction['type']): void {
    if (this.websocket?.readyState === WebSocket.OPEN) {
      this.websocket.send(JSON.stringify({
        type: 'reaction',
        reactionType,
      }));
    }
  }

  sendTip(amount: number, message?: string): void {
    if (this.websocket?.readyState === WebSocket.OPEN) {
      this.websocket.send(JSON.stringify({
        type: 'tip',
        amount,
        message,
      }));
    }
  }

  disconnect(): void {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
  }
}

// ==================== Stream Mixer ====================

export class StreamMixer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private audioContext: AudioContext;
  private audioDestination: MediaStreamAudioDestinationNode;
  private sources: Map<string, StreamSource> = new Map();
  private outputStream: MediaStream | null = null;
  private animationFrame: number | null = null;

  constructor(width = 1920, height = 1080) {
    this.canvas = document.createElement('canvas');
    this.canvas.width = width;
    this.canvas.height = height;
    this.ctx = this.canvas.getContext('2d')!;

    this.audioContext = new AudioContext();
    this.audioDestination = this.audioContext.createMediaStreamDestination();
  }

  async addSource(source: StreamSource): Promise<void> {
    this.sources.set(source.id, source);

    if (source.stream) {
      // Connect audio
      const audioTracks = source.stream.getAudioTracks();
      if (audioTracks.length > 0) {
        const audioSource = this.audioContext.createMediaStreamSource(source.stream);
        const gainNode = this.audioContext.createGain();
        gainNode.gain.value = source.volume;
        audioSource.connect(gainNode);
        gainNode.connect(this.audioDestination);
      }
    }
  }

  removeSource(sourceId: string): void {
    this.sources.delete(sourceId);
  }

  updateSource(sourceId: string, updates: Partial<StreamSource>): void {
    const source = this.sources.get(sourceId);
    if (source) {
      Object.assign(source, updates);
    }
  }

  setSourcePosition(sourceId: string, position: StreamSource['position']): void {
    const source = this.sources.get(sourceId);
    if (source) {
      source.position = position;
    }
  }

  setSourceVolume(sourceId: string, volume: number): void {
    const source = this.sources.get(sourceId);
    if (source) {
      source.volume = volume;
    }
  }

  startMixing(): MediaStream {
    this.renderFrame();

    // Create output stream
    const videoTrack = this.canvas.captureStream(30).getVideoTracks()[0];
    const audioTrack = this.audioDestination.stream.getAudioTracks()[0];

    this.outputStream = new MediaStream([videoTrack, audioTrack]);
    return this.outputStream;
  }

  private renderFrame(): void {
    // Clear canvas
    this.ctx.fillStyle = '#000000';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    // Render each active source
    for (const source of this.sources.values()) {
      if (!source.active || !source.stream) continue;

      const videoTracks = source.stream.getVideoTracks();
      if (videoTracks.length === 0) continue;

      // Create video element to capture frames
      const video = document.createElement('video');
      video.srcObject = source.stream;
      video.muted = true;

      if (source.position) {
        const { x, y, width, height } = source.position;
        this.ctx.drawImage(
          video,
          x * this.canvas.width,
          y * this.canvas.height,
          width * this.canvas.width,
          height * this.canvas.height
        );
      } else {
        this.ctx.drawImage(video, 0, 0, this.canvas.width, this.canvas.height);
      }
    }

    this.animationFrame = requestAnimationFrame(() => this.renderFrame());
  }

  stopMixing(): void {
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
    }

    if (this.outputStream) {
      this.outputStream.getTracks().forEach(track => track.stop());
    }
  }

  getCanvas(): HTMLCanvasElement {
    return this.canvas;
  }

  setResolution(width: number, height: number): void {
    this.canvas.width = width;
    this.canvas.height = height;
  }
}

// ==================== Stream Recorder ====================

export class StreamRecorder {
  private mediaRecorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private stream: MediaStream | null = null;

  public onDataAvailable?: (blob: Blob) => void;
  public onStop?: (blob: Blob) => void;

  start(stream: MediaStream, options?: MediaRecorderOptions): void {
    this.stream = stream;
    this.chunks = [];

    const mimeType = this.getSupportedMimeType();

    this.mediaRecorder = new MediaRecorder(stream, {
      mimeType,
      videoBitsPerSecond: 8000000,
      audioBitsPerSecond: 320000,
      ...options,
    });

    this.mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        this.chunks.push(event.data);
        this.onDataAvailable?.(event.data);
      }
    };

    this.mediaRecorder.onstop = () => {
      const blob = new Blob(this.chunks, { type: mimeType });
      this.onStop?.(blob);
    };

    this.mediaRecorder.start(1000); // Chunk every second
  }

  private getSupportedMimeType(): string {
    const types = [
      'video/webm;codecs=vp9,opus',
      'video/webm;codecs=vp8,opus',
      'video/webm',
      'video/mp4',
    ];

    for (const type of types) {
      if (MediaRecorder.isTypeSupported(type)) {
        return type;
      }
    }

    return 'video/webm';
  }

  pause(): void {
    if (this.mediaRecorder?.state === 'recording') {
      this.mediaRecorder.pause();
    }
  }

  resume(): void {
    if (this.mediaRecorder?.state === 'paused') {
      this.mediaRecorder.resume();
    }
  }

  stop(): Blob {
    if (this.mediaRecorder) {
      this.mediaRecorder.stop();
    }

    return new Blob(this.chunks, { type: this.mediaRecorder?.mimeType });
  }

  getRecording(): Blob {
    return new Blob(this.chunks, { type: this.mediaRecorder?.mimeType });
  }
}

// ==================== Virtual Concert Manager ====================

export class VirtualConcertManager {
  private baseUrl: string;

  constructor(baseUrl = '/api/concerts') {
    this.baseUrl = baseUrl;
  }

  async createConcert(params: {
    title: string;
    description: string;
    scheduledFor: Date;
    venueId: string;
    ticketPrice?: number;
    setlist: Array<{ trackId: string }>;
    vipOptions?: VIPFeature[];
  }): Promise<VirtualConcert> {
    const response = await fetch(`${this.baseUrl}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return response.json();
  }

  async getConcert(concertId: string): Promise<VirtualConcert> {
    const response = await fetch(`${this.baseUrl}/${concertId}`);
    return response.json();
  }

  async getUpcomingConcerts(): Promise<VirtualConcert[]> {
    const response = await fetch(`${this.baseUrl}/upcoming`);
    return response.json();
  }

  async purchaseTicket(concertId: string, tier: string): Promise<{
    ticketId: string;
    accessCode: string;
  }> {
    const response = await fetch(`${this.baseUrl}/${concertId}/tickets`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tier }),
    });
    return response.json();
  }

  async getVenues(): Promise<ConcertVenue[]> {
    const response = await fetch(`${this.baseUrl}/venues`);
    return response.json();
  }

  async updateSetlist(concertId: string, setlist: SetlistItem[]): Promise<void> {
    await fetch(`${this.baseUrl}/${concertId}/setlist`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ setlist }),
    });
  }

  async markSongPlayed(concertId: string, trackId: string): Promise<void> {
    await fetch(`${this.baseUrl}/${concertId}/setlist/${trackId}/played`, {
      method: 'POST',
    });
  }

  async getConcertAnalytics(concertId: string): Promise<{
    attendance: number;
    peakViewers: number;
    revenue: number;
    tipTotal: number;
    merchandiseSales: number;
    averageWatchTime: number;
    reactions: Record<string, number>;
    chatActivity: Array<{ time: string; count: number }>;
  }> {
    const response = await fetch(`${this.baseUrl}/${concertId}/analytics`);
    return response.json();
  }
}

// ==================== Live Streaming Manager ====================

export class LiveStreamingManager {
  public readonly streams: StreamManager;
  public readonly concerts: VirtualConcertManager;
  private broadcaster: WebRTCBroadcaster | null = null;
  private viewer: StreamViewer | null = null;
  private chat: LiveChat | null = null;
  private mixer: StreamMixer | null = null;
  private recorder: StreamRecorder | null = null;

  constructor(baseUrl = '/api') {
    this.streams = new StreamManager(`${baseUrl}/streaming`);
    this.concerts = new VirtualConcertManager(`${baseUrl}/concerts`);
  }

  async startBroadcast(
    streamId: string,
    stream: MediaStream,
    signalingUrl: string
  ): Promise<WebRTCBroadcaster> {
    this.broadcaster = new WebRTCBroadcaster(signalingUrl, streamId);
    await this.broadcaster.start(stream);

    // Start stream
    await this.streams.startStream(streamId);

    return this.broadcaster;
  }

  async watchStream(
    streamId: string,
    signalingUrl: string,
    options?: Parameters<StreamViewer['connect']>[0]
  ): Promise<StreamViewer> {
    this.viewer = new StreamViewer(signalingUrl, streamId);
    await this.viewer.connect(options);
    return this.viewer;
  }

  connectChat(streamId: string, userId: string, wsUrl: string): LiveChat {
    this.chat = new LiveChat(streamId, userId);
    this.chat.connect(wsUrl);
    return this.chat;
  }

  createMixer(width?: number, height?: number): StreamMixer {
    this.mixer = new StreamMixer(width, height);
    return this.mixer;
  }

  startRecording(stream: MediaStream): StreamRecorder {
    this.recorder = new StreamRecorder();
    this.recorder.start(stream);
    return this.recorder;
  }

  async stopBroadcast(streamId: string): Promise<void> {
    this.broadcaster?.stop();
    await this.streams.endStream(streamId);
  }

  stopWatching(): void {
    this.viewer?.disconnect();
  }

  disconnectChat(): void {
    this.chat?.disconnect();
  }

  stopRecording(): Blob | undefined {
    return this.recorder?.stop();
  }

  dispose(): void {
    this.broadcaster?.stop();
    this.viewer?.disconnect();
    this.chat?.disconnect();
    this.mixer?.stopMixing();
  }
}

// ==================== Singleton ====================

let liveStreamingManager: LiveStreamingManager | null = null;

export function getLiveStreamingManager(): LiveStreamingManager {
  if (!liveStreamingManager) {
    liveStreamingManager = new LiveStreamingManager();
  }
  return liveStreamingManager;
}

export default {
  LiveStreamingManager,
  getLiveStreamingManager,
  StreamManager,
  WebRTCBroadcaster,
  StreamViewer,
  LiveChat,
  StreamMixer,
  StreamRecorder,
  VirtualConcertManager,
};
