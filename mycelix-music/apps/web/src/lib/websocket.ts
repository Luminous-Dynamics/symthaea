// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * WebSocket Client for Mycelix Rust Backend
 *
 * Provides real-time communication for:
 * - Session synchronization
 * - Playback state updates
 * - Peer presence (collaborative listening)
 * - Live audio streaming
 */

const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:3000/ws';

// === Types ===

export interface SessionState {
  current_track?: string;
  position: number;
  playback: string;
  volume: number;
}

// Incoming messages (from server)
export type WsMessage =
  | { type: 'connected'; connection_id: string }
  | { type: 'session_created'; session_id: string }
  | { type: 'session_joined'; session_id: string; state: SessionState }
  | { type: 'playback_state_changed'; session_id: string; state: string }
  | { type: 'position_changed'; session_id: string; position: number }
  | { type: 'volume_changed'; session_id: string; volume: number }
  | { type: 'track_changed'; session_id: string; track_id: string }
  | { type: 'peer_joined'; session_id: string; peer_id: string }
  | { type: 'peer_left'; session_id: string; peer_id: string }
  | { type: 'subscribed'; session_id: string; events: string[] }
  | { type: 'audio_data'; session_id: string; timestamp: number; data: string }
  | { type: 'error'; message: string }
  | { type: 'pong'; data: number[] };

// Outgoing commands (to server)
export type WsCommand =
  | { type: 'create_session' }
  | { type: 'join_session'; session_id: string }
  | { type: 'play'; session_id: string }
  | { type: 'pause'; session_id: string }
  | { type: 'seek'; session_id: string; position: number }
  | { type: 'set_volume'; session_id: string; volume: number }
  | { type: 'subscribe_to_session'; session_id: string }
  | { type: 'ping' };

// Event handlers
export interface WsEventHandlers {
  onConnected?: (connectionId: string) => void;
  onSessionCreated?: (sessionId: string) => void;
  onSessionJoined?: (sessionId: string, state: SessionState) => void;
  onPlaybackStateChanged?: (sessionId: string, state: string) => void;
  onPositionChanged?: (sessionId: string, position: number) => void;
  onVolumeChanged?: (sessionId: string, volume: number) => void;
  onTrackChanged?: (sessionId: string, trackId: string) => void;
  onPeerJoined?: (sessionId: string, peerId: string) => void;
  onPeerLeft?: (sessionId: string, peerId: string) => void;
  onSubscribed?: (sessionId: string, events: string[]) => void;
  onAudioData?: (sessionId: string, timestamp: number, data: ArrayBuffer) => void;
  onError?: (message: string) => void;
  onDisconnected?: () => void;
  onReconnecting?: (attempt: number) => void;
}

// Connection options
export interface WsOptions {
  url?: string;
  reconnect?: boolean;
  reconnectAttempts?: number;
  reconnectDelay?: number;
  pingInterval?: number;
}

// === WebSocket Client Class ===

export class MycelixWebSocket {
  private ws: WebSocket | null = null;
  private url: string;
  private handlers: WsEventHandlers = {};
  private reconnect: boolean;
  private reconnectAttempts: number;
  private reconnectDelay: number;
  private reconnectCount = 0;
  private pingInterval: number;
  private pingTimer: ReturnType<typeof setInterval> | null = null;
  private connectionId: string | null = null;

  constructor(options: WsOptions = {}) {
    this.url = options.url || WS_BASE;
    this.reconnect = options.reconnect ?? true;
    this.reconnectAttempts = options.reconnectAttempts ?? 5;
    this.reconnectDelay = options.reconnectDelay ?? 1000;
    this.pingInterval = options.pingInterval ?? 30000;
  }

  /**
   * Set event handlers
   */
  on(handlers: WsEventHandlers): this {
    this.handlers = { ...this.handlers, ...handlers };
    return this;
  }

  /**
   * Connect to WebSocket server
   */
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
          this.reconnectCount = 0;
          this.startPing();
          resolve();
        };

        this.ws.onmessage = (event) => {
          this.handleMessage(event.data);
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          reject(error);
        };

        this.ws.onclose = () => {
          this.stopPing();
          this.connectionId = null;
          this.handlers.onDisconnected?.();

          if (this.reconnect && this.reconnectCount < this.reconnectAttempts) {
            this.reconnectCount++;
            this.handlers.onReconnecting?.(this.reconnectCount);
            setTimeout(() => this.connect(), this.reconnectDelay * this.reconnectCount);
          }
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Disconnect from server
   */
  disconnect(): void {
    this.reconnect = false;
    this.stopPing();
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  /**
   * Check if connected
   */
  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  /**
   * Get connection ID
   */
  get id(): string | null {
    return this.connectionId;
  }

  // --- Commands ---

  /**
   * Send a command to the server
   */
  send(command: WsCommand): void {
    if (!this.isConnected) {
      console.error('WebSocket not connected');
      return;
    }

    this.ws!.send(JSON.stringify(command));
  }

  /**
   * Create a new session
   */
  createSession(): void {
    this.send({ type: 'create_session' });
  }

  /**
   * Join an existing session
   */
  joinSession(sessionId: string): void {
    this.send({ type: 'join_session', session_id: sessionId });
  }

  /**
   * Play in session
   */
  play(sessionId: string): void {
    this.send({ type: 'play', session_id: sessionId });
  }

  /**
   * Pause in session
   */
  pause(sessionId: string): void {
    this.send({ type: 'pause', session_id: sessionId });
  }

  /**
   * Seek in session
   */
  seek(sessionId: string, position: number): void {
    this.send({ type: 'seek', session_id: sessionId, position });
  }

  /**
   * Set volume in session
   */
  setVolume(sessionId: string, volume: number): void {
    this.send({ type: 'set_volume', session_id: sessionId, volume });
  }

  /**
   * Subscribe to session events
   */
  subscribeToSession(sessionId: string): void {
    this.send({ type: 'subscribe_to_session', session_id: sessionId });
  }

  /**
   * Send ping (heartbeat)
   */
  ping(): void {
    this.send({ type: 'ping' });
  }

  // --- Private Methods ---

  private handleMessage(data: string): void {
    try {
      const message: WsMessage = JSON.parse(data);

      switch (message.type) {
        case 'connected':
          this.connectionId = message.connection_id;
          this.handlers.onConnected?.(message.connection_id);
          break;

        case 'session_created':
          this.handlers.onSessionCreated?.(message.session_id);
          break;

        case 'session_joined':
          this.handlers.onSessionJoined?.(message.session_id, message.state);
          break;

        case 'playback_state_changed':
          this.handlers.onPlaybackStateChanged?.(message.session_id, message.state);
          break;

        case 'position_changed':
          this.handlers.onPositionChanged?.(message.session_id, message.position);
          break;

        case 'volume_changed':
          this.handlers.onVolumeChanged?.(message.session_id, message.volume);
          break;

        case 'track_changed':
          this.handlers.onTrackChanged?.(message.session_id, message.track_id);
          break;

        case 'peer_joined':
          this.handlers.onPeerJoined?.(message.session_id, message.peer_id);
          break;

        case 'peer_left':
          this.handlers.onPeerLeft?.(message.session_id, message.peer_id);
          break;

        case 'subscribed':
          this.handlers.onSubscribed?.(message.session_id, message.events);
          break;

        case 'audio_data':
          // Decode base64 audio data
          const binary = atob(message.data);
          const bytes = new Uint8Array(binary.length);
          for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
          }
          this.handlers.onAudioData?.(message.session_id, message.timestamp, bytes.buffer);
          break;

        case 'error':
          this.handlers.onError?.(message.message);
          break;

        case 'pong':
          // Heartbeat response, ignore
          break;
      }
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
    }
  }

  private startPing(): void {
    this.stopPing();
    this.pingTimer = setInterval(() => {
      if (this.isConnected) {
        this.ping();
      }
    }, this.pingInterval);
  }

  private stopPing(): void {
    if (this.pingTimer) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
  }
}

// === Singleton Instance ===

let wsInstance: MycelixWebSocket | null = null;

/**
 * Get or create the WebSocket instance
 */
export function getWebSocket(options?: WsOptions): MycelixWebSocket {
  if (!wsInstance) {
    wsInstance = new MycelixWebSocket(options);
  }
  return wsInstance;
}

/**
 * Connect the singleton WebSocket
 */
export async function connectWebSocket(options?: WsOptions): Promise<MycelixWebSocket> {
  const ws = getWebSocket(options);
  if (!ws.isConnected) {
    await ws.connect();
  }
  return ws;
}

/**
 * Disconnect the singleton WebSocket
 */
export function disconnectWebSocket(): void {
  if (wsInstance) {
    wsInstance.disconnect();
    wsInstance = null;
  }
}

export default MycelixWebSocket;
