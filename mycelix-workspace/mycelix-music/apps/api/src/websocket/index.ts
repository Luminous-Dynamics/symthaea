// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * WebSocket Server with Event System Integration
 *
 * Real-time communication layer that bridges the event system
 * with connected clients. Supports channels, authentication,
 * and message broadcasting.
 */

import { Server as HttpServer } from 'http';
import { WebSocket, WebSocketServer } from 'ws';
import { IncomingMessage } from 'http';
import { parse as parseUrl } from 'url';
import { TypedEventEmitter, getEventEmitter } from '../events';
import { verifySignature, AuthPayload } from '../middleware/wallet-auth';

/**
 * WebSocket message types
 */
export enum MessageType {
  // Client -> Server
  SUBSCRIBE = 'subscribe',
  UNSUBSCRIBE = 'unsubscribe',
  AUTHENTICATE = 'authenticate',
  PING = 'ping',

  // Server -> Client
  EVENT = 'event',
  SUBSCRIBED = 'subscribed',
  UNSUBSCRIBED = 'unsubscribed',
  AUTHENTICATED = 'authenticated',
  ERROR = 'error',
  PONG = 'pong',
}

/**
 * WebSocket message structure
 */
export interface WSMessage<T = unknown> {
  type: MessageType;
  channel?: string;
  data?: T;
  timestamp: number;
}

/**
 * Client connection state
 */
interface ClientState {
  id: string;
  socket: WebSocket;
  authenticated: boolean;
  walletAddress?: string;
  channels: Set<string>;
  lastPing: number;
  metadata: Record<string, unknown>;
}

/**
 * Channel types for subscriptions
 */
export type Channel =
  | 'plays' // All play events
  | 'songs' // Song updates
  | `song:${string}` // Specific song events
  | `artist:${string}` // Artist-specific events
  | `wallet:${string}` // Wallet-specific events
  | 'trending' // Trending updates
  | 'system'; // System events

/**
 * WebSocket server options
 */
export interface WebSocketOptions {
  /** Heartbeat interval in milliseconds */
  heartbeatInterval?: number;
  /** Client timeout in milliseconds */
  clientTimeout?: number;
  /** Maximum connections per IP */
  maxConnectionsPerIP?: number;
  /** Require authentication for subscriptions */
  requireAuth?: boolean;
  /** Path for WebSocket endpoint */
  path?: string;
}

/**
 * WebSocket server with event system integration
 */
export class WebSocketManager {
  private wss: WebSocketServer;
  private clients: Map<string, ClientState> = new Map();
  private channels: Map<string, Set<string>> = new Map();
  private connectionCounts: Map<string, number> = new Map();
  private heartbeatTimer?: NodeJS.Timeout;
  private events: TypedEventEmitter;
  private options: Required<WebSocketOptions>;

  constructor(server: HttpServer, options: WebSocketOptions = {}) {
    this.options = {
      heartbeatInterval: options.heartbeatInterval ?? 30000,
      clientTimeout: options.clientTimeout ?? 60000,
      maxConnectionsPerIP: options.maxConnectionsPerIP ?? 10,
      requireAuth: options.requireAuth ?? false,
      path: options.path ?? '/ws',
    };

    this.wss = new WebSocketServer({
      server,
      path: this.options.path,
      verifyClient: this.verifyClient.bind(this),
    });

    this.events = getEventEmitter();
    this.setupEventListeners();
    this.setupWebSocketHandlers();
    this.startHeartbeat();

    console.log(`🔌 WebSocket server listening on ${this.options.path}`);
  }

  /**
   * Verify client connection
   */
  private verifyClient(
    info: { origin: string; secure: boolean; req: IncomingMessage },
    callback: (result: boolean, code?: number, message?: string) => void
  ): void {
    const ip = this.getClientIP(info.req);

    // Check connection limit per IP
    const currentCount = this.connectionCounts.get(ip) || 0;
    if (currentCount >= this.options.maxConnectionsPerIP) {
      callback(false, 429, 'Too many connections');
      return;
    }

    callback(true);
  }

  /**
   * Setup WebSocket event handlers
   */
  private setupWebSocketHandlers(): void {
    this.wss.on('connection', (socket: WebSocket, req: IncomingMessage) => {
      const id = this.generateClientId();
      const ip = this.getClientIP(req);

      // Track connection count
      this.connectionCounts.set(ip, (this.connectionCounts.get(ip) || 0) + 1);

      // Initialize client state
      const client: ClientState = {
        id,
        socket,
        authenticated: false,
        channels: new Set(),
        lastPing: Date.now(),
        metadata: { ip, connectedAt: new Date().toISOString() },
      };

      this.clients.set(id, client);
      console.log(`WebSocket client connected: ${id}`);

      // Handle messages
      socket.on('message', (data: Buffer) => {
        this.handleMessage(client, data);
      });

      // Handle close
      socket.on('close', () => {
        this.handleDisconnect(client);
        this.connectionCounts.set(ip, Math.max(0, (this.connectionCounts.get(ip) || 1) - 1));
      });

      // Handle errors
      socket.on('error', (error: Error) => {
        console.error(`WebSocket error for ${id}:`, error.message);
      });

      // Send welcome message
      this.send(client, {
        type: MessageType.EVENT,
        channel: 'system',
        data: { event: 'connected', clientId: id },
        timestamp: Date.now(),
      });
    });
  }

  /**
   * Setup event system listeners
   */
  private setupEventListeners(): void {
    // Play events -> broadcast to 'plays' channel and song-specific channels
    this.events.on('play:recorded', (data) => {
      this.broadcast('plays', 'play:recorded', data);
      this.broadcast(`song:${data.songId}`, 'play:recorded', data);
      this.broadcast(`artist:${data.artistAddress}`, 'play:recorded', data);
    });

    // Song events
    this.events.on('song:created', (data) => {
      this.broadcast('songs', 'song:created', data);
      this.broadcast(`artist:${data.artistAddress}`, 'song:created', data);
    });

    this.events.on('song:updated', (data) => {
      this.broadcast('songs', 'song:updated', data);
      this.broadcast(`song:${data.id}`, 'song:updated', data);
    });

    // Analytics events
    this.events.on('analytics:updated', (data) => {
      this.broadcast('trending', 'analytics:updated', data);
    });

    // System events
    this.events.on('system:shutdown', () => {
      this.broadcast('system', 'system:shutdown', {
        message: 'Server shutting down',
      });
      this.close();
    });
  }

  /**
   * Handle incoming message
   */
  private handleMessage(client: ClientState, data: Buffer): void {
    let message: WSMessage;

    try {
      message = JSON.parse(data.toString());
    } catch {
      this.sendError(client, 'Invalid JSON');
      return;
    }

    client.lastPing = Date.now();

    switch (message.type) {
      case MessageType.SUBSCRIBE:
        this.handleSubscribe(client, message.channel as Channel);
        break;

      case MessageType.UNSUBSCRIBE:
        this.handleUnsubscribe(client, message.channel as Channel);
        break;

      case MessageType.AUTHENTICATE:
        this.handleAuthenticate(client, message.data as {
          payload: AuthPayload;
          signature: string;
        });
        break;

      case MessageType.PING:
        this.send(client, { type: MessageType.PONG, timestamp: Date.now() });
        break;

      default:
        this.sendError(client, 'Unknown message type');
    }
  }

  /**
   * Handle channel subscription
   */
  private handleSubscribe(client: ClientState, channel: Channel): void {
    if (!channel) {
      this.sendError(client, 'Channel required');
      return;
    }

    // Check if auth required for private channels
    if (this.isPrivateChannel(channel) && !client.authenticated) {
      this.sendError(client, 'Authentication required');
      return;
    }

    // Check if wallet channel matches authenticated wallet
    if (channel.startsWith('wallet:')) {
      const walletAddress = channel.replace('wallet:', '');
      if (client.walletAddress?.toLowerCase() !== walletAddress.toLowerCase()) {
        this.sendError(client, 'Cannot subscribe to other wallet channels');
        return;
      }
    }

    // Add to channel
    client.channels.add(channel);

    if (!this.channels.has(channel)) {
      this.channels.set(channel, new Set());
    }
    this.channels.get(channel)!.add(client.id);

    this.send(client, {
      type: MessageType.SUBSCRIBED,
      channel,
      timestamp: Date.now(),
    });

    console.log(`Client ${client.id} subscribed to ${channel}`);
  }

  /**
   * Handle channel unsubscription
   */
  private handleUnsubscribe(client: ClientState, channel: Channel): void {
    if (!channel) {
      this.sendError(client, 'Channel required');
      return;
    }

    client.channels.delete(channel);
    this.channels.get(channel)?.delete(client.id);

    this.send(client, {
      type: MessageType.UNSUBSCRIBED,
      channel,
      timestamp: Date.now(),
    });
  }

  /**
   * Handle authentication
   */
  private async handleAuthenticate(
    client: ClientState,
    data: { payload: AuthPayload; signature: string }
  ): Promise<void> {
    if (!data?.payload || !data?.signature) {
      this.sendError(client, 'Payload and signature required');
      return;
    }

    const { payload, signature } = data;

    // Create message from payload
    const message = [
      'Mycelix Music WebSocket Authentication',
      '',
      `Address: ${payload.address}`,
      `Timestamp: ${payload.timestamp}`,
      `Nonce: ${payload.nonce}`,
    ].join('\n');

    // Verify signature
    const result = await verifySignature(message, signature, payload.address);

    if (!result.valid) {
      this.sendError(client, result.error || 'Invalid signature');
      return;
    }

    // Check timestamp
    const age = Date.now() - payload.timestamp;
    if (age > 5 * 60 * 1000) {
      this.sendError(client, 'Signature expired');
      return;
    }

    client.authenticated = true;
    client.walletAddress = result.address;

    this.send(client, {
      type: MessageType.AUTHENTICATED,
      data: { address: result.address },
      timestamp: Date.now(),
    });

    console.log(`Client ${client.id} authenticated as ${result.address}`);
  }

  /**
   * Handle client disconnect
   */
  private handleDisconnect(client: ClientState): void {
    // Remove from all channels
    for (const channel of client.channels) {
      this.channels.get(channel)?.delete(client.id);
    }

    this.clients.delete(client.id);
    console.log(`WebSocket client disconnected: ${client.id}`);
  }

  /**
   * Broadcast event to channel
   */
  broadcast(channel: string, event: string, data: unknown): void {
    const subscribers = this.channels.get(channel);
    if (!subscribers || subscribers.size === 0) return;

    const message: WSMessage = {
      type: MessageType.EVENT,
      channel,
      data: { event, ...data as object },
      timestamp: Date.now(),
    };

    for (const clientId of subscribers) {
      const client = this.clients.get(clientId);
      if (client) {
        this.send(client, message);
      }
    }
  }

  /**
   * Send message to specific client
   */
  send(client: ClientState, message: WSMessage): void {
    if (client.socket.readyState === WebSocket.OPEN) {
      client.socket.send(JSON.stringify(message));
    }
  }

  /**
   * Send error to client
   */
  private sendError(client: ClientState, error: string): void {
    this.send(client, {
      type: MessageType.ERROR,
      data: { error },
      timestamp: Date.now(),
    });
  }

  /**
   * Start heartbeat monitoring
   */
  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      const now = Date.now();
      const timeout = this.options.clientTimeout;

      for (const [id, client] of this.clients) {
        if (now - client.lastPing > timeout) {
          console.log(`Client ${id} timed out`);
          client.socket.terminate();
          this.handleDisconnect(client);
        } else if (client.socket.readyState === WebSocket.OPEN) {
          this.send(client, {
            type: MessageType.PING,
            timestamp: now,
          });
        }
      }
    }, this.options.heartbeatInterval);
  }

  /**
   * Check if channel requires authentication
   */
  private isPrivateChannel(channel: string): boolean {
    return (
      channel.startsWith('wallet:') ||
      channel.startsWith('artist:') ||
      (this.options.requireAuth && channel !== 'system')
    );
  }

  /**
   * Generate unique client ID
   */
  private generateClientId(): string {
    return `ws_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`;
  }

  /**
   * Get client IP address
   */
  private getClientIP(req: IncomingMessage): string {
    const forwarded = req.headers['x-forwarded-for'];
    if (typeof forwarded === 'string') {
      return forwarded.split(',')[0].trim();
    }
    return req.socket.remoteAddress || 'unknown';
  }

  /**
   * Get connection statistics
   */
  getStats(): {
    totalConnections: number;
    authenticatedConnections: number;
    channels: Record<string, number>;
  } {
    const channels: Record<string, number> = {};
    for (const [name, subscribers] of this.channels) {
      channels[name] = subscribers.size;
    }

    return {
      totalConnections: this.clients.size,
      authenticatedConnections: Array.from(this.clients.values()).filter(c => c.authenticated).length,
      channels,
    };
  }

  /**
   * Send to specific wallet
   */
  sendToWallet(walletAddress: string, event: string, data: unknown): void {
    const normalizedAddress = walletAddress.toLowerCase();

    for (const client of this.clients.values()) {
      if (client.walletAddress?.toLowerCase() === normalizedAddress) {
        this.send(client, {
          type: MessageType.EVENT,
          channel: `wallet:${normalizedAddress}`,
          data: { event, ...data as object },
          timestamp: Date.now(),
        });
      }
    }
  }

  /**
   * Close all connections and shutdown
   */
  close(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
    }

    for (const client of this.clients.values()) {
      this.send(client, {
        type: MessageType.EVENT,
        channel: 'system',
        data: { event: 'disconnecting', reason: 'Server shutting down' },
        timestamp: Date.now(),
      });
      client.socket.close(1001, 'Server shutting down');
    }

    this.wss.close();
    console.log('🔌 WebSocket server closed');
  }
}

/**
 * Singleton instance
 */
let wsManager: WebSocketManager | null = null;

export function initWebSocket(server: HttpServer, options?: WebSocketOptions): WebSocketManager {
  if (!wsManager) {
    wsManager = new WebSocketManager(server, options);
  }
  return wsManager;
}

export function getWebSocketManager(): WebSocketManager | null {
  return wsManager;
}

export function closeWebSocket(): void {
  wsManager?.close();
  wsManager = null;
}

export default WebSocketManager;
