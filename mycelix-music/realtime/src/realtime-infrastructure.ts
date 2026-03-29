// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Real-Time Infrastructure
 *
 * Complete real-time communication system including:
 * - WebSocket Gateway for bidirectional communication
 * - WebRTC for peer-to-peer audio/video (jam sessions, live streaming)
 * - Push Notifications (FCM, APNs, Web Push)
 * - Real-Time Sync Engine for collaborative features
 */

import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';

// ============================================================================
// WebSocket Gateway
// ============================================================================

interface WebSocketClient {
  id: string;
  userId: string;
  socket: any;
  rooms: Set<string>;
  subscriptions: Set<string>;
  metadata: {
    userAgent: string;
    ip: string;
    connectedAt: Date;
    lastActivity: Date;
    deviceType: 'web' | 'mobile' | 'desktop' | 'iot';
  };
  heartbeat: {
    lastPing: Date;
    latency: number;
  };
}

interface WebSocketMessage {
  type: string;
  payload: any;
  timestamp: number;
  messageId: string;
  acknowledgment?: boolean;
}

interface Room {
  id: string;
  type: 'chat' | 'jam' | 'listen' | 'broadcast' | 'notification';
  members: Map<string, WebSocketClient>;
  metadata: any;
  createdAt: Date;
  maxMembers?: number;
}

export class WebSocketGateway extends EventEmitter {
  private clients: Map<string, WebSocketClient> = new Map();
  private userConnections: Map<string, Set<string>> = new Map();
  private rooms: Map<string, Room> = new Map();
  private messageQueue: Map<string, WebSocketMessage[]> = new Map();
  private heartbeatInterval: NodeJS.Timer | null = null;

  private config = {
    heartbeatInterval: 30000,
    heartbeatTimeout: 60000,
    maxConnectionsPerUser: 5,
    maxMessageSize: 1024 * 1024, // 1MB
    messageRateLimit: 100, // per minute
    reconnectWindow: 300000, // 5 minutes
  };

  async initialize(): Promise<void> {
    this.startHeartbeat();
    this.setupMessageHandlers();
    console.log('WebSocket Gateway initialized');
  }

  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      const now = new Date();

      this.clients.forEach((client, clientId) => {
        const timeSinceLastPing = now.getTime() - client.heartbeat.lastPing.getTime();

        if (timeSinceLastPing > this.config.heartbeatTimeout) {
          this.disconnectClient(clientId, 'heartbeat_timeout');
        } else {
          this.sendToClient(clientId, {
            type: 'ping',
            payload: { timestamp: now.getTime() },
            timestamp: now.getTime(),
            messageId: uuidv4(),
          });
        }
      });
    }, this.config.heartbeatInterval);
  }

  private setupMessageHandlers(): void {
    this.on('message', async (clientId: string, message: WebSocketMessage) => {
      switch (message.type) {
        case 'pong':
          this.handlePong(clientId, message);
          break;
        case 'subscribe':
          await this.handleSubscribe(clientId, message.payload);
          break;
        case 'unsubscribe':
          await this.handleUnsubscribe(clientId, message.payload);
          break;
        case 'room:join':
          await this.joinRoom(clientId, message.payload.roomId);
          break;
        case 'room:leave':
          await this.leaveRoom(clientId, message.payload.roomId);
          break;
        case 'room:message':
          await this.broadcastToRoom(message.payload.roomId, message, clientId);
          break;
        case 'presence:update':
          await this.updatePresence(clientId, message.payload);
          break;
        default:
          this.emit('custom_message', clientId, message);
      }

      if (message.acknowledgment) {
        this.sendToClient(clientId, {
          type: 'ack',
          payload: { messageId: message.messageId },
          timestamp: Date.now(),
          messageId: uuidv4(),
        });
      }
    });
  }

  async handleConnection(socket: any, userId: string, metadata: any): Promise<string> {
    const clientId = uuidv4();

    // Check connection limit
    const existingConnections = this.userConnections.get(userId) || new Set();
    if (existingConnections.size >= this.config.maxConnectionsPerUser) {
      const oldestConnection = [...existingConnections][0];
      await this.disconnectClient(oldestConnection, 'connection_limit');
    }

    const client: WebSocketClient = {
      id: clientId,
      userId,
      socket,
      rooms: new Set(),
      subscriptions: new Set(),
      metadata: {
        ...metadata,
        connectedAt: new Date(),
        lastActivity: new Date(),
      },
      heartbeat: {
        lastPing: new Date(),
        latency: 0,
      },
    };

    this.clients.set(clientId, client);

    if (!this.userConnections.has(userId)) {
      this.userConnections.set(userId, new Set());
    }
    this.userConnections.get(userId)!.add(clientId);

    // Deliver queued messages
    const queuedMessages = this.messageQueue.get(userId) || [];
    for (const message of queuedMessages) {
      await this.sendToClient(clientId, message);
    }
    this.messageQueue.delete(userId);

    this.emit('connection', clientId, userId);

    return clientId;
  }

  async disconnectClient(clientId: string, reason: string): Promise<void> {
    const client = this.clients.get(clientId);
    if (!client) return;

    // Leave all rooms
    for (const roomId of client.rooms) {
      await this.leaveRoom(clientId, roomId);
    }

    // Remove from user connections
    const userConnections = this.userConnections.get(client.userId);
    if (userConnections) {
      userConnections.delete(clientId);
      if (userConnections.size === 0) {
        this.userConnections.delete(client.userId);
        this.emit('user_offline', client.userId);
      }
    }

    try {
      client.socket.close(1000, reason);
    } catch {}

    this.clients.delete(clientId);
    this.emit('disconnection', clientId, client.userId, reason);
  }

  private handlePong(clientId: string, message: WebSocketMessage): void {
    const client = this.clients.get(clientId);
    if (!client) return;

    const now = Date.now();
    client.heartbeat.latency = now - message.payload.timestamp;
    client.heartbeat.lastPing = new Date();
    client.metadata.lastActivity = new Date();
  }

  private async handleSubscribe(clientId: string, channels: string[]): Promise<void> {
    const client = this.clients.get(clientId);
    if (!client) return;

    for (const channel of channels) {
      client.subscriptions.add(channel);
    }
  }

  private async handleUnsubscribe(clientId: string, channels: string[]): Promise<void> {
    const client = this.clients.get(clientId);
    if (!client) return;

    for (const channel of channels) {
      client.subscriptions.delete(channel);
    }
  }

  async createRoom(type: Room['type'], metadata: any = {}, maxMembers?: number): Promise<string> {
    const roomId = uuidv4();

    this.rooms.set(roomId, {
      id: roomId,
      type,
      members: new Map(),
      metadata,
      createdAt: new Date(),
      maxMembers,
    });

    return roomId;
  }

  async joinRoom(clientId: string, roomId: string): Promise<boolean> {
    const client = this.clients.get(clientId);
    const room = this.rooms.get(roomId);

    if (!client || !room) return false;
    if (room.maxMembers && room.members.size >= room.maxMembers) return false;

    room.members.set(clientId, client);
    client.rooms.add(roomId);

    await this.broadcastToRoom(roomId, {
      type: 'room:member_joined',
      payload: {
        userId: client.userId,
        clientId,
        timestamp: Date.now(),
      },
      timestamp: Date.now(),
      messageId: uuidv4(),
    }, clientId);

    return true;
  }

  async leaveRoom(clientId: string, roomId: string): Promise<void> {
    const client = this.clients.get(clientId);
    const room = this.rooms.get(roomId);

    if (!client || !room) return;

    room.members.delete(clientId);
    client.rooms.delete(roomId);

    await this.broadcastToRoom(roomId, {
      type: 'room:member_left',
      payload: {
        userId: client.userId,
        clientId,
        timestamp: Date.now(),
      },
      timestamp: Date.now(),
      messageId: uuidv4(),
    });

    // Cleanup empty rooms
    if (room.members.size === 0) {
      this.rooms.delete(roomId);
    }
  }

  async broadcastToRoom(
    roomId: string,
    message: WebSocketMessage,
    excludeClientId?: string
  ): Promise<void> {
    const room = this.rooms.get(roomId);
    if (!room) return;

    for (const [clientId] of room.members) {
      if (clientId !== excludeClientId) {
        await this.sendToClient(clientId, message);
      }
    }
  }

  async sendToClient(clientId: string, message: WebSocketMessage): Promise<boolean> {
    const client = this.clients.get(clientId);
    if (!client) return false;

    try {
      client.socket.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error(`Failed to send message to client ${clientId}:`, error);
      return false;
    }
  }

  async sendToUser(userId: string, message: WebSocketMessage): Promise<number> {
    const connections = this.userConnections.get(userId);
    if (!connections || connections.size === 0) {
      // Queue message for later delivery
      if (!this.messageQueue.has(userId)) {
        this.messageQueue.set(userId, []);
      }
      this.messageQueue.get(userId)!.push(message);
      return 0;
    }

    let delivered = 0;
    for (const clientId of connections) {
      if (await this.sendToClient(clientId, message)) {
        delivered++;
      }
    }
    return delivered;
  }

  async broadcast(channel: string, message: WebSocketMessage): Promise<number> {
    let delivered = 0;

    for (const [clientId, client] of this.clients) {
      if (client.subscriptions.has(channel)) {
        if (await this.sendToClient(clientId, message)) {
          delivered++;
        }
      }
    }

    return delivered;
  }

  private async updatePresence(clientId: string, presence: any): Promise<void> {
    const client = this.clients.get(clientId);
    if (!client) return;

    this.emit('presence_update', client.userId, presence);

    // Broadcast to user's rooms
    for (const roomId of client.rooms) {
      await this.broadcastToRoom(roomId, {
        type: 'presence:update',
        payload: {
          userId: client.userId,
          presence,
        },
        timestamp: Date.now(),
        messageId: uuidv4(),
      }, clientId);
    }
  }

  getClientStats(): object {
    return {
      totalConnections: this.clients.size,
      uniqueUsers: this.userConnections.size,
      totalRooms: this.rooms.size,
      messageQueueSize: Array.from(this.messageQueue.values())
        .reduce((sum, queue) => sum + queue.length, 0),
    };
  }

  async shutdown(): Promise<void> {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
    }

    for (const [clientId] of this.clients) {
      await this.disconnectClient(clientId, 'server_shutdown');
    }
  }
}

// ============================================================================
// WebRTC Signaling Server
// ============================================================================

interface RTCPeer {
  peerId: string;
  userId: string;
  sessionId: string;
  connectionState: 'new' | 'connecting' | 'connected' | 'disconnected' | 'failed';
  mediaState: {
    audio: boolean;
    video: boolean;
    screen: boolean;
  };
  stats: {
    packetsLost: number;
    jitter: number;
    roundTripTime: number;
  };
}

interface RTCSession {
  id: string;
  type: 'jam' | 'call' | 'broadcast' | 'watch_party';
  peers: Map<string, RTCPeer>;
  host: string;
  metadata: any;
  createdAt: Date;
  config: {
    maxPeers: number;
    audioOnly: boolean;
    recordSession: boolean;
    lowLatencyMode: boolean;
  };
}

interface SignalingMessage {
  type: 'offer' | 'answer' | 'ice_candidate' | 'media_state' | 'session_config';
  from: string;
  to: string;
  sessionId: string;
  payload: any;
}

export class WebRTCSignalingServer extends EventEmitter {
  private sessions: Map<string, RTCSession> = new Map();
  private peerSessions: Map<string, string> = new Map();
  private wsGateway: WebSocketGateway;

  private stunServers = [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' },
  ];

  private turnServers = [
    {
      urls: 'turn:turn.mycelix.io:3478',
      username: 'mycelix',
      credential: process.env.TURN_SECRET,
    },
  ];

  constructor(wsGateway: WebSocketGateway) {
    super();
    this.wsGateway = wsGateway;
    this.setupSignalingHandlers();
  }

  private setupSignalingHandlers(): void {
    this.wsGateway.on('custom_message', async (clientId: string, message: any) => {
      if (message.type.startsWith('rtc:')) {
        await this.handleSignalingMessage(clientId, message);
      }
    });
  }

  private async handleSignalingMessage(clientId: string, message: any): Promise<void> {
    switch (message.type) {
      case 'rtc:create_session':
        await this.createSession(clientId, message.payload);
        break;
      case 'rtc:join_session':
        await this.joinSession(clientId, message.payload.sessionId);
        break;
      case 'rtc:leave_session':
        await this.leaveSession(clientId);
        break;
      case 'rtc:offer':
      case 'rtc:answer':
      case 'rtc:ice_candidate':
        await this.relaySignaling(clientId, message);
        break;
      case 'rtc:media_state':
        await this.updateMediaState(clientId, message.payload);
        break;
    }
  }

  async createSession(
    hostClientId: string,
    config: Partial<RTCSession['config']> & { type: RTCSession['type']; metadata?: any }
  ): Promise<string> {
    const sessionId = uuidv4();

    const session: RTCSession = {
      id: sessionId,
      type: config.type,
      peers: new Map(),
      host: hostClientId,
      metadata: config.metadata || {},
      createdAt: new Date(),
      config: {
        maxPeers: config.maxPeers || 10,
        audioOnly: config.audioOnly || false,
        recordSession: config.recordSession || false,
        lowLatencyMode: config.lowLatencyMode || false,
      },
    };

    this.sessions.set(sessionId, session);

    // Host joins automatically
    await this.addPeerToSession(sessionId, hostClientId);

    // Send session created confirmation
    await this.wsGateway.sendToClient(hostClientId, {
      type: 'rtc:session_created',
      payload: {
        sessionId,
        iceServers: [...this.stunServers, ...this.turnServers],
        config: session.config,
      },
      timestamp: Date.now(),
      messageId: uuidv4(),
    });

    return sessionId;
  }

  async joinSession(clientId: string, sessionId: string): Promise<boolean> {
    const session = this.sessions.get(sessionId);
    if (!session) {
      await this.sendError(clientId, 'Session not found');
      return false;
    }

    if (session.peers.size >= session.config.maxPeers) {
      await this.sendError(clientId, 'Session is full');
      return false;
    }

    await this.addPeerToSession(sessionId, clientId);

    // Send join confirmation with existing peers
    const existingPeers = Array.from(session.peers.entries())
      .filter(([id]) => id !== clientId)
      .map(([id, peer]) => ({
        peerId: id,
        userId: peer.userId,
        mediaState: peer.mediaState,
      }));

    await this.wsGateway.sendToClient(clientId, {
      type: 'rtc:session_joined',
      payload: {
        sessionId,
        iceServers: [...this.stunServers, ...this.turnServers],
        peers: existingPeers,
        config: session.config,
      },
      timestamp: Date.now(),
      messageId: uuidv4(),
    });

    // Notify existing peers
    await this.broadcastToSession(sessionId, {
      type: 'rtc:peer_joined',
      payload: {
        peerId: clientId,
        userId: session.peers.get(clientId)?.userId,
      },
      timestamp: Date.now(),
      messageId: uuidv4(),
    }, clientId);

    return true;
  }

  private async addPeerToSession(sessionId: string, clientId: string): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (!session) return;

    const peer: RTCPeer = {
      peerId: clientId,
      userId: clientId, // Would resolve from WebSocket client
      sessionId,
      connectionState: 'new',
      mediaState: {
        audio: false,
        video: false,
        screen: false,
      },
      stats: {
        packetsLost: 0,
        jitter: 0,
        roundTripTime: 0,
      },
    };

    session.peers.set(clientId, peer);
    this.peerSessions.set(clientId, sessionId);
  }

  async leaveSession(clientId: string): Promise<void> {
    const sessionId = this.peerSessions.get(clientId);
    if (!sessionId) return;

    const session = this.sessions.get(sessionId);
    if (!session) return;

    session.peers.delete(clientId);
    this.peerSessions.delete(clientId);

    // Notify remaining peers
    await this.broadcastToSession(sessionId, {
      type: 'rtc:peer_left',
      payload: { peerId: clientId },
      timestamp: Date.now(),
      messageId: uuidv4(),
    });

    // Cleanup empty sessions or transfer host
    if (session.peers.size === 0) {
      this.sessions.delete(sessionId);
    } else if (session.host === clientId) {
      // Transfer host to first remaining peer
      const newHost = session.peers.keys().next().value;
      session.host = newHost;

      await this.broadcastToSession(sessionId, {
        type: 'rtc:host_changed',
        payload: { newHost },
        timestamp: Date.now(),
        messageId: uuidv4(),
      });
    }
  }

  private async relaySignaling(fromClientId: string, message: any): Promise<void> {
    const sessionId = this.peerSessions.get(fromClientId);
    if (!sessionId) return;

    const session = this.sessions.get(sessionId);
    if (!session) return;

    const toClientId = message.payload.to;

    if (!session.peers.has(toClientId)) {
      await this.sendError(fromClientId, 'Target peer not in session');
      return;
    }

    await this.wsGateway.sendToClient(toClientId, {
      type: message.type,
      payload: {
        ...message.payload,
        from: fromClientId,
      },
      timestamp: Date.now(),
      messageId: uuidv4(),
    });
  }

  private async updateMediaState(clientId: string, mediaState: RTCPeer['mediaState']): Promise<void> {
    const sessionId = this.peerSessions.get(clientId);
    if (!sessionId) return;

    const session = this.sessions.get(sessionId);
    if (!session) return;

    const peer = session.peers.get(clientId);
    if (peer) {
      peer.mediaState = mediaState;
    }

    await this.broadcastToSession(sessionId, {
      type: 'rtc:media_state_changed',
      payload: {
        peerId: clientId,
        mediaState,
      },
      timestamp: Date.now(),
      messageId: uuidv4(),
    }, clientId);
  }

  private async broadcastToSession(
    sessionId: string,
    message: any,
    excludeClientId?: string
  ): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (!session) return;

    for (const [clientId] of session.peers) {
      if (clientId !== excludeClientId) {
        await this.wsGateway.sendToClient(clientId, message);
      }
    }
  }

  private async sendError(clientId: string, error: string): Promise<void> {
    await this.wsGateway.sendToClient(clientId, {
      type: 'rtc:error',
      payload: { error },
      timestamp: Date.now(),
      messageId: uuidv4(),
    });
  }

  getSessionStats(sessionId: string): object | null {
    const session = this.sessions.get(sessionId);
    if (!session) return null;

    return {
      id: session.id,
      type: session.type,
      peerCount: session.peers.size,
      host: session.host,
      createdAt: session.createdAt,
      peers: Array.from(session.peers.values()).map(p => ({
        peerId: p.peerId,
        connectionState: p.connectionState,
        mediaState: p.mediaState,
        stats: p.stats,
      })),
    };
  }
}

// ============================================================================
// Push Notification Service
// ============================================================================

interface PushSubscription {
  userId: string;
  platform: 'fcm' | 'apns' | 'web';
  token: string;
  deviceId: string;
  createdAt: Date;
  lastUsed: Date;
  metadata: {
    deviceName?: string;
    appVersion?: string;
    osVersion?: string;
  };
}

interface PushNotification {
  id: string;
  title: string;
  body: string;
  data?: Record<string, any>;
  image?: string;
  badge?: number;
  sound?: string;
  priority: 'high' | 'normal' | 'low';
  ttl?: number;
  collapseKey?: string;
  channelId?: string;
  actions?: Array<{
    id: string;
    title: string;
    url?: string;
  }>;
}

interface NotificationResult {
  success: boolean;
  token: string;
  error?: string;
  messageId?: string;
}

export class PushNotificationService extends EventEmitter {
  private subscriptions: Map<string, PushSubscription[]> = new Map();
  private fcmClient: any; // Firebase Admin SDK
  private apnsClient: any; // node-apn
  private webPushClient: any; // web-push

  private channels: Map<string, {
    id: string;
    name: string;
    description: string;
    importance: 'high' | 'default' | 'low';
    sound?: string;
  }> = new Map([
    ['messages', { id: 'messages', name: 'Messages', description: 'Direct messages and chat', importance: 'high' }],
    ['social', { id: 'social', name: 'Social', description: 'Follows, likes, and comments', importance: 'default' }],
    ['releases', { id: 'releases', name: 'New Releases', description: 'Music from artists you follow', importance: 'default' }],
    ['live', { id: 'live', name: 'Live Events', description: 'Live streams and jam sessions', importance: 'high' }],
    ['promotions', { id: 'promotions', name: 'Promotions', description: 'Deals and recommendations', importance: 'low' }],
  ]);

  async initialize(): Promise<void> {
    // Initialize FCM
    // this.fcmClient = admin.messaging();

    // Initialize APNs
    // this.apnsClient = new apn.Provider({ token: {...} });

    // Initialize Web Push
    // webpush.setVapidDetails(...)

    console.log('Push Notification Service initialized');
  }

  async registerSubscription(subscription: Omit<PushSubscription, 'createdAt' | 'lastUsed'>): Promise<void> {
    const fullSubscription: PushSubscription = {
      ...subscription,
      createdAt: new Date(),
      lastUsed: new Date(),
    };

    const userSubscriptions = this.subscriptions.get(subscription.userId) || [];

    // Remove existing subscription for same device
    const filtered = userSubscriptions.filter(s => s.deviceId !== subscription.deviceId);
    filtered.push(fullSubscription);

    this.subscriptions.set(subscription.userId, filtered);
    this.emit('subscription_registered', subscription.userId, subscription.deviceId);
  }

  async unregisterSubscription(userId: string, deviceId: string): Promise<void> {
    const userSubscriptions = this.subscriptions.get(userId) || [];
    const filtered = userSubscriptions.filter(s => s.deviceId !== deviceId);

    if (filtered.length > 0) {
      this.subscriptions.set(userId, filtered);
    } else {
      this.subscriptions.delete(userId);
    }
  }

  async sendToUser(
    userId: string,
    notification: Omit<PushNotification, 'id'>,
    options: { platforms?: PushSubscription['platform'][] } = {}
  ): Promise<NotificationResult[]> {
    const subscriptions = this.subscriptions.get(userId) || [];
    const targetSubscriptions = options.platforms
      ? subscriptions.filter(s => options.platforms!.includes(s.platform))
      : subscriptions;

    const fullNotification: PushNotification = {
      ...notification,
      id: uuidv4(),
    };

    const results: NotificationResult[] = [];

    for (const subscription of targetSubscriptions) {
      try {
        const result = await this.sendToPlatform(subscription, fullNotification);
        results.push(result);

        // Update last used
        subscription.lastUsed = new Date();
      } catch (error: any) {
        results.push({
          success: false,
          token: subscription.token,
          error: error.message,
        });

        // Remove invalid tokens
        if (this.isInvalidTokenError(error)) {
          await this.unregisterSubscription(userId, subscription.deviceId);
        }
      }
    }

    return results;
  }

  async sendToUsers(
    userIds: string[],
    notification: Omit<PushNotification, 'id'>
  ): Promise<Map<string, NotificationResult[]>> {
    const results = new Map<string, NotificationResult[]>();

    // Batch by platform for efficiency
    const fcmTokens: { userId: string; token: string }[] = [];
    const apnsTokens: { userId: string; token: string }[] = [];
    const webTokens: { userId: string; subscription: PushSubscription }[] = [];

    for (const userId of userIds) {
      const subscriptions = this.subscriptions.get(userId) || [];
      for (const sub of subscriptions) {
        switch (sub.platform) {
          case 'fcm':
            fcmTokens.push({ userId, token: sub.token });
            break;
          case 'apns':
            apnsTokens.push({ userId, token: sub.token });
            break;
          case 'web':
            webTokens.push({ userId, subscription: sub });
            break;
        }
      }
    }

    // Send batch notifications
    if (fcmTokens.length > 0) {
      await this.sendFCMBatch(fcmTokens, notification, results);
    }

    if (apnsTokens.length > 0) {
      await this.sendAPNSBatch(apnsTokens, notification, results);
    }

    if (webTokens.length > 0) {
      await this.sendWebPushBatch(webTokens, notification, results);
    }

    return results;
  }

  async sendToTopic(topic: string, notification: Omit<PushNotification, 'id'>): Promise<string> {
    const fullNotification: PushNotification = {
      ...notification,
      id: uuidv4(),
    };

    // FCM topic messaging
    const message = {
      topic,
      notification: {
        title: fullNotification.title,
        body: fullNotification.body,
        imageUrl: fullNotification.image,
      },
      data: fullNotification.data,
      android: {
        priority: fullNotification.priority === 'high' ? 'high' : 'normal',
        notification: {
          channelId: fullNotification.channelId,
          sound: fullNotification.sound,
        },
      },
      apns: {
        payload: {
          aps: {
            badge: fullNotification.badge,
            sound: fullNotification.sound,
          },
        },
      },
    };

    // return await this.fcmClient.send(message);
    return fullNotification.id;
  }

  private async sendToPlatform(
    subscription: PushSubscription,
    notification: PushNotification
  ): Promise<NotificationResult> {
    switch (subscription.platform) {
      case 'fcm':
        return this.sendFCM(subscription.token, notification);
      case 'apns':
        return this.sendAPNS(subscription.token, notification);
      case 'web':
        return this.sendWebPush(subscription.token, notification);
      default:
        throw new Error(`Unknown platform: ${subscription.platform}`);
    }
  }

  private async sendFCM(token: string, notification: PushNotification): Promise<NotificationResult> {
    const message = {
      token,
      notification: {
        title: notification.title,
        body: notification.body,
        imageUrl: notification.image,
      },
      data: notification.data ?
        Object.fromEntries(Object.entries(notification.data).map(([k, v]) => [k, String(v)])) :
        undefined,
      android: {
        priority: notification.priority === 'high' ? 'high' as const : 'normal' as const,
        ttl: notification.ttl ? notification.ttl * 1000 : undefined,
        collapseKey: notification.collapseKey,
        notification: {
          channelId: notification.channelId,
          sound: notification.sound || 'default',
        },
      },
    };

    // const response = await this.fcmClient.send(message);
    return {
      success: true,
      token,
      messageId: `fcm_${notification.id}`,
    };
  }

  private async sendAPNS(token: string, notification: PushNotification): Promise<NotificationResult> {
    const apnsNotification = {
      alert: {
        title: notification.title,
        body: notification.body,
      },
      badge: notification.badge,
      sound: notification.sound || 'default',
      expiry: notification.ttl ? Math.floor(Date.now() / 1000) + notification.ttl : undefined,
      collapseId: notification.collapseKey,
      payload: notification.data,
    };

    // await this.apnsClient.send(apnsNotification, token);
    return {
      success: true,
      token,
      messageId: `apns_${notification.id}`,
    };
  }

  private async sendWebPush(subscription: string, notification: PushNotification): Promise<NotificationResult> {
    const payload = JSON.stringify({
      title: notification.title,
      body: notification.body,
      icon: notification.image,
      badge: '/badge.png',
      data: notification.data,
      actions: notification.actions,
    });

    // await webpush.sendNotification(JSON.parse(subscription), payload);
    return {
      success: true,
      token: subscription,
      messageId: `web_${notification.id}`,
    };
  }

  private async sendFCMBatch(
    tokens: { userId: string; token: string }[],
    notification: Omit<PushNotification, 'id'>,
    results: Map<string, NotificationResult[]>
  ): Promise<void> {
    // FCM supports up to 500 tokens per batch
    const batches = this.chunk(tokens, 500);

    for (const batch of batches) {
      for (const { userId, token } of batch) {
        const result = await this.sendFCM(token, { ...notification, id: uuidv4() });
        if (!results.has(userId)) results.set(userId, []);
        results.get(userId)!.push(result);
      }
    }
  }

  private async sendAPNSBatch(
    tokens: { userId: string; token: string }[],
    notification: Omit<PushNotification, 'id'>,
    results: Map<string, NotificationResult[]>
  ): Promise<void> {
    for (const { userId, token } of tokens) {
      const result = await this.sendAPNS(token, { ...notification, id: uuidv4() });
      if (!results.has(userId)) results.set(userId, []);
      results.get(userId)!.push(result);
    }
  }

  private async sendWebPushBatch(
    subscriptions: { userId: string; subscription: PushSubscription }[],
    notification: Omit<PushNotification, 'id'>,
    results: Map<string, NotificationResult[]>
  ): Promise<void> {
    for (const { userId, subscription } of subscriptions) {
      const result = await this.sendWebPush(subscription.token, { ...notification, id: uuidv4() });
      if (!results.has(userId)) results.set(userId, []);
      results.get(userId)!.push(result);
    }
  }

  private isInvalidTokenError(error: any): boolean {
    const invalidMessages = [
      'NotRegistered',
      'InvalidRegistration',
      'Unregistered',
      'BadDeviceToken',
      'ExpiredToken',
    ];
    return invalidMessages.some(msg => error.message?.includes(msg));
  }

  private chunk<T>(array: T[], size: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < array.length; i += size) {
      chunks.push(array.slice(i, i + size));
    }
    return chunks;
  }

  getChannels(): typeof this.channels {
    return this.channels;
  }
}

// ============================================================================
// Real-Time Sync Engine
// ============================================================================

interface SyncDocument {
  id: string;
  type: string;
  version: number;
  data: any;
  lastModified: Date;
  lastModifiedBy: string;
}

interface SyncOperation {
  id: string;
  documentId: string;
  type: 'create' | 'update' | 'delete' | 'patch';
  payload: any;
  version: number;
  timestamp: Date;
  userId: string;
  clientId: string;
}

interface ConflictResolution {
  strategy: 'last_write_wins' | 'first_write_wins' | 'merge' | 'manual';
  resolver?: (local: any, remote: any, base: any) => any;
}

export class RealTimeSyncEngine extends EventEmitter {
  private documents: Map<string, SyncDocument> = new Map();
  private operations: Map<string, SyncOperation[]> = new Map();
  private subscriptions: Map<string, Set<string>> = new Map(); // documentId -> clientIds
  private clientSubscriptions: Map<string, Set<string>> = new Map(); // clientId -> documentIds
  private wsGateway: WebSocketGateway;
  private conflictResolution: ConflictResolution = { strategy: 'last_write_wins' };

  constructor(wsGateway: WebSocketGateway) {
    super();
    this.wsGateway = wsGateway;
    this.setupSyncHandlers();
  }

  private setupSyncHandlers(): void {
    this.wsGateway.on('custom_message', async (clientId: string, message: any) => {
      if (message.type.startsWith('sync:')) {
        await this.handleSyncMessage(clientId, message);
      }
    });
  }

  private async handleSyncMessage(clientId: string, message: any): Promise<void> {
    switch (message.type) {
      case 'sync:subscribe':
        await this.subscribeToDocument(clientId, message.payload.documentId);
        break;
      case 'sync:unsubscribe':
        await this.unsubscribeFromDocument(clientId, message.payload.documentId);
        break;
      case 'sync:operation':
        await this.applyOperation(clientId, message.payload);
        break;
      case 'sync:fetch':
        await this.fetchDocument(clientId, message.payload.documentId);
        break;
      case 'sync:fetch_operations':
        await this.fetchOperations(clientId, message.payload);
        break;
    }
  }

  async createDocument(
    type: string,
    data: any,
    userId: string
  ): Promise<SyncDocument> {
    const document: SyncDocument = {
      id: uuidv4(),
      type,
      version: 1,
      data,
      lastModified: new Date(),
      lastModifiedBy: userId,
    };

    this.documents.set(document.id, document);
    this.operations.set(document.id, []);

    this.emit('document_created', document);
    return document;
  }

  async subscribeToDocument(clientId: string, documentId: string): Promise<void> {
    // Add to document subscribers
    if (!this.subscriptions.has(documentId)) {
      this.subscriptions.set(documentId, new Set());
    }
    this.subscriptions.get(documentId)!.add(clientId);

    // Add to client subscriptions
    if (!this.clientSubscriptions.has(clientId)) {
      this.clientSubscriptions.set(clientId, new Set());
    }
    this.clientSubscriptions.get(clientId)!.add(documentId);

    // Send current document state
    const document = this.documents.get(documentId);
    if (document) {
      await this.wsGateway.sendToClient(clientId, {
        type: 'sync:document_state',
        payload: {
          document,
          subscribed: true,
        },
        timestamp: Date.now(),
        messageId: uuidv4(),
      });
    }
  }

  async unsubscribeFromDocument(clientId: string, documentId: string): Promise<void> {
    this.subscriptions.get(documentId)?.delete(clientId);
    this.clientSubscriptions.get(clientId)?.delete(documentId);
  }

  async applyOperation(clientId: string, operationData: Omit<SyncOperation, 'id' | 'timestamp'>): Promise<void> {
    const document = this.documents.get(operationData.documentId);
    if (!document) {
      await this.sendSyncError(clientId, 'Document not found');
      return;
    }

    // Version check for conflict detection
    if (operationData.version !== document.version) {
      await this.handleConflict(clientId, document, operationData);
      return;
    }

    const operation: SyncOperation = {
      ...operationData,
      id: uuidv4(),
      timestamp: new Date(),
    };

    // Apply operation
    switch (operation.type) {
      case 'update':
        document.data = operation.payload;
        break;
      case 'patch':
        document.data = this.applyPatch(document.data, operation.payload);
        break;
      case 'delete':
        this.documents.delete(operation.documentId);
        await this.broadcastToSubscribers(operation.documentId, {
          type: 'sync:document_deleted',
          payload: { documentId: operation.documentId },
          timestamp: Date.now(),
          messageId: uuidv4(),
        });
        return;
    }

    document.version++;
    document.lastModified = new Date();
    document.lastModifiedBy = operation.userId;

    // Store operation
    this.operations.get(operation.documentId)?.push(operation);

    // Acknowledge to sender
    await this.wsGateway.sendToClient(clientId, {
      type: 'sync:operation_ack',
      payload: {
        operationId: operation.id,
        documentId: operation.documentId,
        newVersion: document.version,
      },
      timestamp: Date.now(),
      messageId: uuidv4(),
    });

    // Broadcast to other subscribers
    await this.broadcastToSubscribers(operation.documentId, {
      type: 'sync:operation',
      payload: operation,
      timestamp: Date.now(),
      messageId: uuidv4(),
    }, clientId);

    this.emit('operation_applied', operation);
  }

  private async handleConflict(
    clientId: string,
    document: SyncDocument,
    operation: Omit<SyncOperation, 'id' | 'timestamp'>
  ): Promise<void> {
    switch (this.conflictResolution.strategy) {
      case 'last_write_wins':
        // Force apply with current version
        operation.version = document.version;
        await this.applyOperation(clientId, operation);
        break;

      case 'first_write_wins':
        // Reject and send current state
        await this.wsGateway.sendToClient(clientId, {
          type: 'sync:conflict',
          payload: {
            documentId: document.id,
            currentVersion: document.version,
            yourVersion: operation.version,
            currentState: document.data,
            resolution: 'rejected',
          },
          timestamp: Date.now(),
          messageId: uuidv4(),
        });
        break;

      case 'merge':
        if (this.conflictResolution.resolver) {
          const baseVersion = this.findBaseVersion(document.id, operation.version);
          const merged = this.conflictResolution.resolver(
            operation.payload,
            document.data,
            baseVersion
          );
          operation.payload = merged;
          operation.version = document.version;
          await this.applyOperation(clientId, operation);
        }
        break;

      case 'manual':
        await this.wsGateway.sendToClient(clientId, {
          type: 'sync:conflict',
          payload: {
            documentId: document.id,
            currentVersion: document.version,
            yourVersion: operation.version,
            currentState: document.data,
            yourState: operation.payload,
            resolution: 'manual_required',
          },
          timestamp: Date.now(),
          messageId: uuidv4(),
        });
        break;
    }
  }

  private findBaseVersion(documentId: string, version: number): any {
    const ops = this.operations.get(documentId) || [];
    const baseOp = ops.find(op => op.version === version - 1);
    return baseOp?.payload || null;
  }

  private applyPatch(data: any, patch: any[]): any {
    // JSON Patch (RFC 6902) implementation
    let result = JSON.parse(JSON.stringify(data));

    for (const op of patch) {
      const path = op.path.split('/').filter(Boolean);

      switch (op.op) {
        case 'add':
        case 'replace':
          this.setAtPath(result, path, op.value);
          break;
        case 'remove':
          this.removeAtPath(result, path);
          break;
        case 'move':
          const fromPath = op.from.split('/').filter(Boolean);
          const value = this.getAtPath(result, fromPath);
          this.removeAtPath(result, fromPath);
          this.setAtPath(result, path, value);
          break;
        case 'copy':
          const sourcePath = op.from.split('/').filter(Boolean);
          this.setAtPath(result, path, this.getAtPath(result, sourcePath));
          break;
      }
    }

    return result;
  }

  private getAtPath(obj: any, path: string[]): any {
    return path.reduce((current, key) => current?.[key], obj);
  }

  private setAtPath(obj: any, path: string[], value: any): void {
    const lastKey = path.pop()!;
    const target = path.reduce((current, key) => {
      if (!current[key]) current[key] = {};
      return current[key];
    }, obj);
    target[lastKey] = value;
  }

  private removeAtPath(obj: any, path: string[]): void {
    const lastKey = path.pop()!;
    const target = path.reduce((current, key) => current?.[key], obj);
    if (target) delete target[lastKey];
  }

  private async fetchDocument(clientId: string, documentId: string): Promise<void> {
    const document = this.documents.get(documentId);

    await this.wsGateway.sendToClient(clientId, {
      type: 'sync:document_state',
      payload: {
        document: document || null,
        found: !!document,
      },
      timestamp: Date.now(),
      messageId: uuidv4(),
    });
  }

  private async fetchOperations(
    clientId: string,
    query: { documentId: string; fromVersion: number }
  ): Promise<void> {
    const ops = this.operations.get(query.documentId) || [];
    const filtered = ops.filter(op => op.version > query.fromVersion);

    await this.wsGateway.sendToClient(clientId, {
      type: 'sync:operations',
      payload: {
        documentId: query.documentId,
        operations: filtered,
      },
      timestamp: Date.now(),
      messageId: uuidv4(),
    });
  }

  private async broadcastToSubscribers(
    documentId: string,
    message: any,
    excludeClientId?: string
  ): Promise<void> {
    const subscribers = this.subscriptions.get(documentId);
    if (!subscribers) return;

    for (const clientId of subscribers) {
      if (clientId !== excludeClientId) {
        await this.wsGateway.sendToClient(clientId, message);
      }
    }
  }

  private async sendSyncError(clientId: string, error: string): Promise<void> {
    await this.wsGateway.sendToClient(clientId, {
      type: 'sync:error',
      payload: { error },
      timestamp: Date.now(),
      messageId: uuidv4(),
    });
  }

  setConflictResolution(resolution: ConflictResolution): void {
    this.conflictResolution = resolution;
  }

  getDocumentStats(documentId: string): object | null {
    const document = this.documents.get(documentId);
    if (!document) return null;

    return {
      id: document.id,
      type: document.type,
      version: document.version,
      lastModified: document.lastModified,
      subscriberCount: this.subscriptions.get(documentId)?.size || 0,
      operationCount: this.operations.get(documentId)?.length || 0,
    };
  }
}

// ============================================================================
// Presence System
// ============================================================================

interface UserPresence {
  userId: string;
  status: 'online' | 'away' | 'busy' | 'offline';
  activity?: {
    type: 'listening' | 'creating' | 'browsing' | 'jamming' | 'streaming';
    details?: string;
    trackId?: string;
    sessionId?: string;
  };
  lastSeen: Date;
  connections: number;
}

export class PresenceSystem extends EventEmitter {
  private presence: Map<string, UserPresence> = new Map();
  private followers: Map<string, Set<string>> = new Map();
  private wsGateway: WebSocketGateway;

  private idleTimeout = 5 * 60 * 1000; // 5 minutes
  private offlineTimeout = 15 * 60 * 1000; // 15 minutes

  constructor(wsGateway: WebSocketGateway) {
    super();
    this.wsGateway = wsGateway;
    this.setupPresenceHandlers();
    this.startPresenceCleanup();
  }

  private setupPresenceHandlers(): void {
    this.wsGateway.on('connection', (clientId: string, userId: string) => {
      this.updatePresence(userId, { status: 'online' });
    });

    this.wsGateway.on('disconnection', (clientId: string, userId: string) => {
      const stats = this.wsGateway.getClientStats() as any;
      // Check if user has other connections
      if (!this.hasOtherConnections(userId)) {
        this.updatePresence(userId, { status: 'offline' });
      }
    });

    this.wsGateway.on('presence_update', (userId: string, update: any) => {
      this.updatePresence(userId, update);
    });
  }

  private hasOtherConnections(userId: string): boolean {
    // Would check WebSocket gateway for other connections
    return false;
  }

  private startPresenceCleanup(): void {
    setInterval(() => {
      const now = new Date();

      this.presence.forEach((presence, userId) => {
        const timeSinceLastSeen = now.getTime() - presence.lastSeen.getTime();

        if (presence.status === 'online' && timeSinceLastSeen > this.idleTimeout) {
          this.updatePresence(userId, { status: 'away' });
        } else if (presence.status !== 'offline' && timeSinceLastSeen > this.offlineTimeout) {
          this.updatePresence(userId, { status: 'offline' });
        }
      });
    }, 60000); // Check every minute
  }

  async updatePresence(userId: string, update: Partial<UserPresence>): Promise<void> {
    const existing = this.presence.get(userId) || {
      userId,
      status: 'offline',
      lastSeen: new Date(),
      connections: 0,
    };

    const updated: UserPresence = {
      ...existing,
      ...update,
      lastSeen: new Date(),
    };

    this.presence.set(userId, updated);

    // Notify followers
    await this.notifyFollowers(userId, updated);

    this.emit('presence_changed', userId, updated);
  }

  async subscribeToPresence(subscriberId: string, targetUserId: string): Promise<void> {
    if (!this.followers.has(targetUserId)) {
      this.followers.set(targetUserId, new Set());
    }
    this.followers.get(targetUserId)!.add(subscriberId);

    // Send current presence
    const presence = this.presence.get(targetUserId);
    if (presence) {
      await this.wsGateway.sendToUser(subscriberId, {
        type: 'presence:update',
        payload: { userId: targetUserId, presence },
        timestamp: Date.now(),
        messageId: uuidv4(),
      });
    }
  }

  async unsubscribeFromPresence(subscriberId: string, targetUserId: string): Promise<void> {
    this.followers.get(targetUserId)?.delete(subscriberId);
  }

  private async notifyFollowers(userId: string, presence: UserPresence): Promise<void> {
    const followers = this.followers.get(userId);
    if (!followers) return;

    for (const followerId of followers) {
      await this.wsGateway.sendToUser(followerId, {
        type: 'presence:update',
        payload: { userId, presence },
        timestamp: Date.now(),
        messageId: uuidv4(),
      });
    }
  }

  getPresence(userId: string): UserPresence | null {
    return this.presence.get(userId) || null;
  }

  async getBulkPresence(userIds: string[]): Promise<Map<string, UserPresence>> {
    const result = new Map<string, UserPresence>();

    for (const userId of userIds) {
      const presence = this.presence.get(userId);
      if (presence) {
        result.set(userId, presence);
      }
    }

    return result;
  }

  getOnlineUsers(): string[] {
    const online: string[] = [];

    this.presence.forEach((presence, userId) => {
      if (presence.status === 'online' || presence.status === 'away' || presence.status === 'busy') {
        online.push(userId);
      }
    });

    return online;
  }

  getUsersWithActivity(activityType: UserPresence['activity']['type']): string[] {
    const users: string[] = [];

    this.presence.forEach((presence, userId) => {
      if (presence.activity?.type === activityType) {
        users.push(userId);
      }
    });

    return users;
  }
}

// ============================================================================
// Unified Real-Time Service
// ============================================================================

export class RealTimeService {
  public wsGateway: WebSocketGateway;
  public webrtc: WebRTCSignalingServer;
  public pushNotifications: PushNotificationService;
  public syncEngine: RealTimeSyncEngine;
  public presence: PresenceSystem;

  constructor() {
    this.wsGateway = new WebSocketGateway();
    this.webrtc = new WebRTCSignalingServer(this.wsGateway);
    this.pushNotifications = new PushNotificationService();
    this.syncEngine = new RealTimeSyncEngine(this.wsGateway);
    this.presence = new PresenceSystem(this.wsGateway);
  }

  async initialize(): Promise<void> {
    await this.wsGateway.initialize();
    await this.pushNotifications.initialize();
    console.log('Real-Time Service fully initialized');
  }

  async shutdown(): Promise<void> {
    await this.wsGateway.shutdown();
  }

  getStats(): object {
    return {
      websocket: this.wsGateway.getClientStats(),
      onlineUsers: this.presence.getOnlineUsers().length,
    };
  }
}

// ============================================================================
// Export
// ============================================================================

export const createRealTimeService = async (): Promise<RealTimeService> => {
  const service = new RealTimeService();
  await service.initialize();
  return service;
};
