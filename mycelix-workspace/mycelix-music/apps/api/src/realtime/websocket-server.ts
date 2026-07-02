// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * WebSocket Server for Real-time Features
 *
 * Handles:
 * - Live listening parties
 * - Collaborative playlists
 * - Real-time presence
 * - Live chat
 * - Notifications
 */

import { Server as HTTPServer } from 'http';
import { WebSocketServer, WebSocket, RawData } from 'ws';
import { v4 as uuidv4 } from 'uuid';
import { Redis } from 'ioredis';
import { verifyToken, JWTPayload } from '../middleware/auth';
import { logger } from '../lib/observability/logger';
import { activeConnections } from '../lib/observability/metrics';

// ============================================================
// Types
// ============================================================

interface AuthenticatedWebSocket extends WebSocket {
  id: string;
  userId?: string;
  user?: JWTPayload;
  isAlive: boolean;
  rooms: Set<string>;
  lastActivity: number;
}

type MessageType =
  | 'auth'
  | 'subscribe'
  | 'unsubscribe'
  | 'message'
  | 'presence'
  | 'sync'
  | 'ping'
  | 'pong';

interface WebSocketMessage {
  type: MessageType;
  room?: string;
  payload?: unknown;
  timestamp?: number;
}

interface Room {
  id: string;
  type: 'listening-party' | 'collaborative-playlist' | 'chat' | 'studio';
  members: Set<string>;
  state: Record<string, unknown>;
  createdAt: number;
  ownerId?: string;
}

// ============================================================
// WebSocket Server
// ============================================================

export class RealtimeServer {
  private wss: WebSocketServer;
  private clients: Map<string, AuthenticatedWebSocket> = new Map();
  private rooms: Map<string, Room> = new Map();
  private redis: Redis;
  private pubsub: Redis;
  private heartbeatInterval: NodeJS.Timeout | null = null;

  constructor(server: HTTPServer, redis: Redis) {
    this.redis = redis;
    this.pubsub = redis.duplicate();

    this.wss = new WebSocketServer({
      server,
      path: '/ws',
      verifyClient: this.verifyClient.bind(this),
    });

    this.setupEventHandlers();
    this.setupPubSub();
    this.startHeartbeat();

    logger.info('WebSocket server initialized');
  }

  // ============================================================
  // Client Connection
  // ============================================================

  private verifyClient(
    info: { origin: string; secure: boolean; req: any },
    callback: (verified: boolean, code?: number, message?: string) => void
  ) {
    // Allow all origins in development, restrict in production
    const allowedOrigins = process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'];

    if (process.env.NODE_ENV === 'production' && !allowedOrigins.includes(info.origin)) {
      callback(false, 403, 'Origin not allowed');
      return;
    }

    callback(true);
  }

  private setupEventHandlers() {
    this.wss.on('connection', (ws: WebSocket, req) => {
      const client = ws as AuthenticatedWebSocket;
      client.id = uuidv4();
      client.isAlive = true;
      client.rooms = new Set();
      client.lastActivity = Date.now();

      this.clients.set(client.id, client);
      activeConnections.inc({ type: 'websocket' });

      logger.debug('WebSocket client connected', { clientId: client.id });

      client.on('message', (data) => this.handleMessage(client, data));
      client.on('close', () => this.handleDisconnect(client));
      client.on('error', (error) => this.handleError(client, error));
      client.on('pong', () => {
        client.isAlive = true;
      });

      // Send welcome message with connection ID
      this.send(client, {
        type: 'message',
        payload: {
          event: 'connected',
          clientId: client.id,
          timestamp: Date.now(),
        },
      });
    });
  }

  // ============================================================
  // Message Handling
  // ============================================================

  private async handleMessage(client: AuthenticatedWebSocket, data: RawData) {
    try {
      const message: WebSocketMessage = JSON.parse(data.toString());
      client.lastActivity = Date.now();

      switch (message.type) {
        case 'auth':
          await this.handleAuth(client, message);
          break;
        case 'subscribe':
          await this.handleSubscribe(client, message);
          break;
        case 'unsubscribe':
          await this.handleUnsubscribe(client, message);
          break;
        case 'message':
          await this.handleRoomMessage(client, message);
          break;
        case 'presence':
          await this.handlePresence(client, message);
          break;
        case 'sync':
          await this.handleSync(client, message);
          break;
        case 'ping':
          this.send(client, { type: 'pong', timestamp: Date.now() });
          break;
      }
    } catch (error) {
      logger.error('Error handling WebSocket message', { error, clientId: client.id });
      this.send(client, {
        type: 'message',
        payload: { event: 'error', message: 'Invalid message format' },
      });
    }
  }

  private async handleAuth(client: AuthenticatedWebSocket, message: WebSocketMessage) {
    const { token } = message.payload as { token: string };

    try {
      const decoded = await verifyToken(token);
      client.userId = decoded.userId;
      client.user = decoded;

      // Store user connection in Redis for cross-instance presence
      await this.redis.hset(`user:${client.userId}:connections`, client.id, JSON.stringify({
        connectedAt: Date.now(),
        lastActivity: client.lastActivity,
      }));

      this.send(client, {
        type: 'message',
        payload: {
          event: 'authenticated',
          userId: client.userId,
          user: { id: decoded.userId, role: decoded.role },
        },
      });

      logger.debug('WebSocket client authenticated', {
        clientId: client.id,
        userId: client.userId,
      });
    } catch (error) {
      this.send(client, {
        type: 'message',
        payload: { event: 'auth_error', message: 'Invalid token' },
      });
    }
  }

  // ============================================================
  // Room Management
  // ============================================================

  private async handleSubscribe(client: AuthenticatedWebSocket, message: WebSocketMessage) {
    const { room: roomId, type = 'chat' } = message.payload as { room: string; type?: Room['type'] };

    if (!roomId) {
      this.send(client, {
        type: 'message',
        payload: { event: 'error', message: 'Room ID required' },
      });
      return;
    }

    let room = this.rooms.get(roomId);

    if (!room) {
      // Create room if it doesn't exist
      room = {
        id: roomId,
        type,
        members: new Set(),
        state: {},
        createdAt: Date.now(),
        ownerId: client.userId,
      };
      this.rooms.set(roomId, room);

      // Also subscribe to Redis pub/sub for cross-instance
      await this.pubsub.subscribe(`room:${roomId}`);
    }

    room.members.add(client.id);
    client.rooms.add(roomId);

    // Notify room of new member
    this.broadcastToRoom(roomId, {
      type: 'presence',
      room: roomId,
      payload: {
        event: 'member_joined',
        userId: client.userId,
        clientId: client.id,
        members: Array.from(room.members).length,
      },
    }, client.id);

    // Send current room state to new member
    this.send(client, {
      type: 'message',
      payload: {
        event: 'subscribed',
        room: roomId,
        state: room.state,
        members: Array.from(room.members).length,
      },
    });

    logger.debug('Client subscribed to room', {
      clientId: client.id,
      roomId,
      memberCount: room.members.size,
    });
  }

  private async handleUnsubscribe(client: AuthenticatedWebSocket, message: WebSocketMessage) {
    const { room: roomId } = message.payload as { room: string };

    if (!roomId || !client.rooms.has(roomId)) return;

    const room = this.rooms.get(roomId);
    if (room) {
      room.members.delete(client.id);

      // Notify room of member leaving
      this.broadcastToRoom(roomId, {
        type: 'presence',
        room: roomId,
        payload: {
          event: 'member_left',
          userId: client.userId,
          clientId: client.id,
          members: room.members.size,
        },
      });

      // Clean up empty rooms
      if (room.members.size === 0) {
        this.rooms.delete(roomId);
        await this.pubsub.unsubscribe(`room:${roomId}`);
      }
    }

    client.rooms.delete(roomId);

    this.send(client, {
      type: 'message',
      payload: { event: 'unsubscribed', room: roomId },
    });
  }

  private async handleRoomMessage(client: AuthenticatedWebSocket, message: WebSocketMessage) {
    const { room: roomId, payload } = message;

    if (!roomId || !client.rooms.has(roomId)) {
      this.send(client, {
        type: 'message',
        payload: { event: 'error', message: 'Not subscribed to room' },
      });
      return;
    }

    // Broadcast to local clients
    this.broadcastToRoom(roomId, {
      type: 'message',
      room: roomId,
      payload: {
        ...(payload as object),
        from: client.userId,
        timestamp: Date.now(),
      },
    }, client.id);

    // Publish to Redis for other instances
    await this.redis.publish(`room:${roomId}`, JSON.stringify({
      type: 'message',
      room: roomId,
      payload,
      from: client.userId,
      originInstance: process.env.INSTANCE_ID,
    }));
  }

  // ============================================================
  // Presence & Sync
  // ============================================================

  private async handlePresence(client: AuthenticatedWebSocket, message: WebSocketMessage) {
    const { room: roomId, payload } = message;

    if (!roomId || !client.rooms.has(roomId)) return;

    const room = this.rooms.get(roomId);
    if (!room) return;

    // Update room state with presence info
    const presenceKey = `presence:${client.userId}`;
    room.state[presenceKey] = {
      ...(payload as object),
      lastSeen: Date.now(),
    };

    // Broadcast presence update
    this.broadcastToRoom(roomId, {
      type: 'presence',
      room: roomId,
      payload: {
        event: 'presence_update',
        userId: client.userId,
        data: payload,
      },
    }, client.id);
  }

  private async handleSync(client: AuthenticatedWebSocket, message: WebSocketMessage) {
    const { room: roomId, payload } = message;

    if (!roomId || !client.rooms.has(roomId)) return;

    const room = this.rooms.get(roomId);
    if (!room) return;

    // Only room owner can sync state in some room types
    if (room.type === 'listening-party' && room.ownerId !== client.userId) {
      this.send(client, {
        type: 'message',
        payload: { event: 'error', message: 'Only host can sync playback' },
      });
      return;
    }

    // Update room state
    const { key, value } = payload as { key: string; value: unknown };
    room.state[key] = value;

    // Broadcast sync to all members
    this.broadcastToRoom(roomId, {
      type: 'sync',
      room: roomId,
      payload: {
        key,
        value,
        timestamp: Date.now(),
      },
    });
  }

  // ============================================================
  // Broadcasting
  // ============================================================

  private broadcastToRoom(
    roomId: string,
    message: WebSocketMessage,
    excludeClientId?: string
  ) {
    const room = this.rooms.get(roomId);
    if (!room) return;

    for (const clientId of room.members) {
      if (clientId === excludeClientId) continue;

      const client = this.clients.get(clientId);
      if (client && client.readyState === WebSocket.OPEN) {
        this.send(client, message);
      }
    }
  }

  public broadcastToUser(userId: string, message: WebSocketMessage) {
    for (const client of this.clients.values()) {
      if (client.userId === userId && client.readyState === WebSocket.OPEN) {
        this.send(client, message);
      }
    }
  }

  public broadcastToAll(message: WebSocketMessage) {
    for (const client of this.clients.values()) {
      if (client.readyState === WebSocket.OPEN) {
        this.send(client, message);
      }
    }
  }

  // ============================================================
  // Redis Pub/Sub
  // ============================================================

  private setupPubSub() {
    this.pubsub.on('message', (channel, data) => {
      try {
        const message = JSON.parse(data);

        // Don't rebroadcast messages from this instance
        if (message.originInstance === process.env.INSTANCE_ID) return;

        const roomId = channel.replace('room:', '');
        const room = this.rooms.get(roomId);

        if (room) {
          this.broadcastToRoom(roomId, {
            type: message.type,
            room: roomId,
            payload: message.payload,
          });
        }
      } catch (error) {
        logger.error('Error processing pub/sub message', { error, channel });
      }
    });
  }

  // ============================================================
  // Connection Management
  // ============================================================

  private handleDisconnect(client: AuthenticatedWebSocket) {
    // Leave all rooms
    for (const roomId of client.rooms) {
      const room = this.rooms.get(roomId);
      if (room) {
        room.members.delete(client.id);

        this.broadcastToRoom(roomId, {
          type: 'presence',
          room: roomId,
          payload: {
            event: 'member_left',
            userId: client.userId,
            clientId: client.id,
          },
        });

        if (room.members.size === 0) {
          this.rooms.delete(roomId);
        }
      }
    }

    // Remove from Redis
    if (client.userId) {
      this.redis.hdel(`user:${client.userId}:connections`, client.id);
    }

    this.clients.delete(client.id);
    activeConnections.dec({ type: 'websocket' });

    logger.debug('WebSocket client disconnected', {
      clientId: client.id,
      userId: client.userId,
    });
  }

  private handleError(client: AuthenticatedWebSocket, error: Error) {
    logger.error('WebSocket error', {
      clientId: client.id,
      error: error.message,
    });
  }

  // ============================================================
  // Heartbeat
  // ============================================================

  private startHeartbeat() {
    this.heartbeatInterval = setInterval(() => {
      for (const [id, client] of this.clients) {
        if (!client.isAlive) {
          client.terminate();
          this.handleDisconnect(client);
          continue;
        }

        client.isAlive = false;
        client.ping();
      }
    }, 30000);
  }

  // ============================================================
  // Utilities
  // ============================================================

  private send(client: AuthenticatedWebSocket, message: WebSocketMessage) {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify({
        ...message,
        timestamp: message.timestamp || Date.now(),
      }));
    }
  }

  public getClientCount(): number {
    return this.clients.size;
  }

  public getRoomCount(): number {
    return this.rooms.size;
  }

  public getRoomInfo(roomId: string): Room | undefined {
    return this.rooms.get(roomId);
  }

  public shutdown() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
    }

    for (const client of this.clients.values()) {
      client.close(1001, 'Server shutting down');
    }

    this.wss.close();
    this.pubsub.quit();

    logger.info('WebSocket server shut down');
  }
}

// Export factory function
export function createRealtimeServer(server: HTTPServer, redis: Redis): RealtimeServer {
  return new RealtimeServer(server, redis);
}
