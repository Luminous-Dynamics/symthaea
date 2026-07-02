// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Collaboration Session Manager
 *
 * Orchestrates real-time collaboration:
 * - Session lifecycle management
 * - Document synchronization
 * - Presence awareness
 * - Audio/video streaming
 * - Undo/redo with CRDT
 * - Conflict resolution
 */

import { DocumentCRDT, DocumentOperation, VectorClockImpl } from './crdt';
import { WebRTCManager, PeerInfo, DataMessage, ConnectionState } from './webrtc';
import { PresenceManager, UserPresence, PresenceUpdate, CursorPosition, Selection } from './presence';

// ==================== Types ====================

export interface SessionConfig {
  signalingUrl: string;
  roomId: string;
  userId: string;
  userName: string;
  userAvatar?: string;
  enableAudio?: boolean;
  enableVideo?: boolean;
  autoReconnect?: boolean;
}

export interface SessionState {
  status: 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'error';
  roomId: string;
  participants: UserPresence[];
  localUser: UserPresence;
  document: Record<string, unknown>;
  error?: Error;
}

export interface SessionCallbacks {
  onStateChange?: (state: SessionState) => void;
  onDocumentChange?: (document: Record<string, unknown>, operation?: DocumentOperation) => void;
  onParticipantJoin?: (participant: UserPresence) => void;
  onParticipantLeave?: (participantId: string) => void;
  onPresenceUpdate?: (participants: UserPresence[]) => void;
  onRemoteAudio?: (participantId: string, data: Float32Array) => void;
  onRemoteStream?: (participantId: string, stream: MediaStream) => void;
  onChatMessage?: (message: ChatMessage) => void;
  onError?: (error: Error) => void;
}

export interface ChatMessage {
  id: string;
  senderId: string;
  senderName: string;
  content: string;
  timestamp: number;
  type: 'text' | 'system' | 'reaction';
}

// ==================== Collaboration Session ====================

export class CollaborationSession {
  private config: SessionConfig;
  private callbacks: SessionCallbacks;
  private document: DocumentCRDT;
  private webrtc: WebRTCManager;
  private presence: PresenceManager;
  private undoStack: DocumentOperation[][] = [];
  private redoStack: DocumentOperation[][] = [];
  private pendingOps: DocumentOperation[] = [];
  private syncInterval: NodeJS.Timeout | null = null;
  private state: SessionState;
  private chatHistory: ChatMessage[] = [];

  constructor(config: SessionConfig, callbacks: SessionCallbacks = {}) {
    this.config = config;
    this.callbacks = callbacks;

    // Initialize CRDT document
    this.document = new DocumentCRDT(config.userId);

    // Initialize WebRTC
    this.webrtc = new WebRTCManager(
      config.signalingUrl,
      config.userId,
      {
        enableAudio: config.enableAudio ?? true,
        enableVideo: config.enableVideo ?? false,
        enableDataChannel: true,
      }
    );

    // Initialize presence
    this.presence = new PresenceManager(config.userId, config.userName, {
      onUpdate: (users) => this.handlePresenceUpdate(users),
      onBroadcast: (update) => this.broadcastPresence(update),
    });

    // Set up WebRTC handlers
    this.setupWebRTCHandlers();

    // Initialize state
    this.state = {
      status: 'disconnected',
      roomId: config.roomId,
      participants: [],
      localUser: this.presence.getLocalUser(),
      document: {},
    };
  }

  private setupWebRTCHandlers(): void {
    this.webrtc.onPeerJoined = (peer) => {
      this.handlePeerJoined(peer);
    };

    this.webrtc.onPeerLeft = (peerId) => {
      this.handlePeerLeft(peerId);
    };

    this.webrtc.onDataMessage = (peerId, message) => {
      this.handleDataMessage(peerId, message);
    };

    this.webrtc.onRemoteStream = (peerId, stream) => {
      this.callbacks.onRemoteStream?.(peerId, stream);
    };

    this.webrtc.onAudioData = (peerId, data) => {
      this.callbacks.onRemoteAudio?.(peerId, data);
    };

    this.webrtc.onConnectionStateChange = (peerId, state) => {
      this.handleConnectionStateChange(peerId, state);
    };
  }

  // ==================== Session Lifecycle ====================

  async connect(): Promise<void> {
    try {
      this.updateState({ status: 'connecting' });

      // Connect to signaling
      await this.webrtc.connect();

      // Join room
      await this.webrtc.joinRoom(this.config.roomId, {
        id: this.config.userId,
        name: this.config.userName,
        avatar: this.config.userAvatar,
        role: 'collaborator',
      });

      // Start sync interval
      this.syncInterval = setInterval(() => {
        this.syncDocument();
      }, 1000);

      this.updateState({ status: 'connected' });
    } catch (error) {
      this.updateState({
        status: 'error',
        error: error instanceof Error ? error : new Error('Connection failed'),
      });
      throw error;
    }
  }

  disconnect(): void {
    if (this.syncInterval) {
      clearInterval(this.syncInterval);
      this.syncInterval = null;
    }

    this.webrtc.disconnect();
    this.presence.dispose();
    this.updateState({ status: 'disconnected' });
  }

  // ==================== Document Operations ====================

  setDocument(path: string[], value: unknown): void {
    const op = this.document.set(path, value);
    this.pendingOps.push(op);
    this.broadcastOperation(op);
    this.notifyDocumentChange(op);
  }

  deleteFromDocument(path: string[]): void {
    const op = this.document.delete(path);
    this.pendingOps.push(op);
    this.broadcastOperation(op);
    this.notifyDocumentChange(op);
  }

  insertIntoArray(path: string[], index: number, value: unknown): void {
    const op = this.document.insertAt(path, index, value);
    this.pendingOps.push(op);
    this.broadcastOperation(op);
    this.notifyDocumentChange(op);
  }

  getDocument(): Record<string, unknown> {
    return this.document.getDocument();
  }

  // ==================== Undo/Redo ====================

  beginUndoGroup(): void {
    this.pendingOps = [];
  }

  endUndoGroup(): void {
    if (this.pendingOps.length > 0) {
      this.undoStack.push([...this.pendingOps]);
      this.redoStack = []; // Clear redo on new action
      this.pendingOps = [];
    }
  }

  undo(): void {
    const ops = this.undoStack.pop();
    if (!ops) return;

    // Create inverse operations
    const inverseOps: DocumentOperation[] = [];
    for (const op of ops.reverse()) {
      // For simplicity, we'll just track the inverse
      // Real implementation would store the previous value
      const inverseOp: DocumentOperation = {
        ...op,
        id: `${op.id}-inverse`,
        timestamp: Date.now(),
      };

      if (op.type === 'update') {
        inverseOp.type = 'delete';
      } else if (op.type === 'delete') {
        // Would need previous value
        continue;
      }

      inverseOps.push(inverseOp);
      this.document.applyOperation(inverseOp);
      this.broadcastOperation(inverseOp);
    }

    this.redoStack.push(ops);
    this.notifyDocumentChange();
  }

  redo(): void {
    const ops = this.redoStack.pop();
    if (!ops) return;

    for (const op of ops) {
      this.document.applyOperation(op);
      this.broadcastOperation(op);
    }

    this.undoStack.push(ops);
    this.notifyDocumentChange();
  }

  // ==================== Presence ====================

  setCursor(position: CursorPosition): void {
    this.presence.setCursor(position);
  }

  setSelection(selection: Selection | undefined): void {
    this.presence.setSelection(selection);
  }

  setTyping(isTyping: boolean): void {
    this.presence.setTyping(isTyping);
  }

  setView(view: string): void {
    this.presence.setCurrentView(view);
  }

  getParticipants(): UserPresence[] {
    return this.presence.getAllUsers();
  }

  // ==================== Chat ====================

  sendChatMessage(content: string): void {
    const message: ChatMessage = {
      id: `${this.config.userId}-${Date.now()}`,
      senderId: this.config.userId,
      senderName: this.config.userName,
      content,
      timestamp: Date.now(),
      type: 'text',
    };

    this.chatHistory.push(message);
    this.webrtc.sendToAll({
      type: 'chat',
      payload: message,
    });

    this.callbacks.onChatMessage?.(message);
  }

  getChatHistory(): ChatMessage[] {
    return [...this.chatHistory];
  }

  // ==================== Audio Control ====================

  async muteLocalAudio(): Promise<void> {
    const stream = this.webrtc.getLocalStream();
    if (stream) {
      stream.getAudioTracks().forEach(track => {
        track.enabled = false;
      });
    }
  }

  async unmuteLocalAudio(): Promise<void> {
    const stream = this.webrtc.getLocalStream();
    if (stream) {
      stream.getAudioTracks().forEach(track => {
        track.enabled = true;
      });
    }
  }

  isLocalAudioMuted(): boolean {
    const stream = this.webrtc.getLocalStream();
    if (!stream) return true;
    const tracks = stream.getAudioTracks();
    return tracks.length === 0 || !tracks[0].enabled;
  }

  // ==================== Internal Handlers ====================

  private handlePeerJoined(peer: PeerInfo): void {
    // Request document state from peer
    this.webrtc.sendToPeer(peer.id, {
      type: 'sync-request',
      payload: {
        vectorClock: Object.fromEntries(this.document.getVectorClock()),
      },
    });

    this.callbacks.onParticipantJoin?.(this.presence.getUser(peer.id) || {
      id: peer.id,
      name: peer.name,
      color: '#8B5CF6',
      status: 'online',
      lastActive: Date.now(),
      permissions: ['view'],
    });
  }

  private handlePeerLeft(peerId: string): void {
    this.presence.removeUser(peerId);
    this.callbacks.onParticipantLeave?.(peerId);
    this.updateState({ participants: this.presence.getAllUsers() });
  }

  private handleDataMessage(peerId: string, message: DataMessage): void {
    switch (message.type) {
      case 'operation':
        this.handleRemoteOperation(message.payload as DocumentOperation);
        break;

      case 'sync-request':
        this.handleSyncRequest(peerId, message.payload as { vectorClock: Record<string, number> });
        break;

      case 'sync-response':
        this.handleSyncResponse(message.payload as {
          operations: DocumentOperation[];
          document: Record<string, unknown>;
        });
        break;

      case 'presence':
        this.handleRemotePresence(message.payload as PresenceUpdate);
        break;

      case 'chat':
        this.handleChatMessage(message.payload as ChatMessage);
        break;
    }
  }

  private handleRemoteOperation(op: DocumentOperation): void {
    if (this.document.applyOperation(op)) {
      this.notifyDocumentChange(op);
    }
  }

  private handleSyncRequest(peerId: string, request: { vectorClock: Record<string, number> }): void {
    // Send full document state to new peer
    this.webrtc.sendToPeer(peerId, {
      type: 'sync-response',
      payload: {
        document: this.document.getDocument(),
        operations: [], // Could send delta based on vector clock
      },
    });
  }

  private handleSyncResponse(response: { operations: DocumentOperation[]; document: Record<string, unknown> }): void {
    // Apply operations
    this.document.applyOperations(response.operations);

    // Merge document state if needed
    for (const [key, value] of Object.entries(response.document)) {
      this.document.set([key], value);
    }

    this.notifyDocumentChange();
  }

  private handleRemotePresence(update: PresenceUpdate): void {
    this.presence.handleRemoteUpdate(update);
  }

  private handleChatMessage(message: ChatMessage): void {
    this.chatHistory.push(message);
    this.callbacks.onChatMessage?.(message);
  }

  private handleConnectionStateChange(peerId: string, state: ConnectionState): void {
    // Update presence based on connection state
    if (state === 'disconnected' || state === 'failed') {
      const user = this.presence.getUser(peerId);
      if (user) {
        user.status = 'offline';
        this.presence.handleRemoteUpdate({
          type: 'status',
          userId: peerId,
          data: { status: 'offline' },
          timestamp: Date.now(),
        });
      }
    }
  }

  private handlePresenceUpdate(users: UserPresence[]): void {
    this.updateState({ participants: users });
    this.callbacks.onPresenceUpdate?.(users);
  }

  // ==================== Broadcasting ====================

  private broadcastOperation(op: DocumentOperation): void {
    this.webrtc.sendToAll({
      type: 'operation',
      payload: op,
    });
  }

  private broadcastPresence(update: PresenceUpdate): void {
    this.webrtc.sendToAll({
      type: 'presence',
      payload: update,
    });
  }

  private syncDocument(): void {
    // Periodic sync to ensure consistency
    // Could implement more sophisticated sync here
  }

  // ==================== State Management ====================

  private updateState(partial: Partial<SessionState>): void {
    this.state = { ...this.state, ...partial };
    this.callbacks.onStateChange?.(this.state);
  }

  private notifyDocumentChange(operation?: DocumentOperation): void {
    const doc = this.document.getDocument();
    this.updateState({ document: doc });
    this.callbacks.onDocumentChange?.(doc, operation);
  }

  getState(): SessionState {
    return { ...this.state };
  }
}

// ==================== React Hook ====================

export function useCollaborationSession(
  config: SessionConfig | null,
  callbacks: SessionCallbacks = {}
): {
  session: CollaborationSession | null;
  state: SessionState | null;
  connect: () => Promise<void>;
  disconnect: () => void;
} {
  // This would be implemented as a React hook
  // Returning placeholder for now
  return {
    session: null,
    state: null,
    connect: async () => {},
    disconnect: () => {},
  };
}

export default CollaborationSession;
