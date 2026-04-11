// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Real-Time Sync Service
 *
 * Provides:
 * - WebSocket connection management
 * - Real-time email updates
 * - Presence detection (who's online, typing indicators)
 * - Optimistic updates with rollback
 * - Connection state management
 * - Auto-reconnection with exponential backoff
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

// ============================================================================
// Types
// ============================================================================

export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting';

export interface SyncMessage {
  id: string;
  type: SyncMessageType;
  payload: unknown;
  timestamp: number;
  senderId?: string;
}

export type SyncMessageType =
  | 'email:new'
  | 'email:update'
  | 'email:delete'
  | 'email:read'
  | 'email:archive'
  | 'email:move'
  | 'folder:update'
  | 'trust:attestation'
  | 'trust:update'
  | 'presence:join'
  | 'presence:leave'
  | 'presence:typing'
  | 'presence:viewing'
  | 'sync:request'
  | 'sync:response'
  | 'conflict:detected'
  | 'conflict:resolved';

export interface PresenceUser {
  id: string;
  name: string;
  email: string;
  avatar?: string;
  status: 'online' | 'away' | 'busy';
  currentView?: string;
  lastSeen: Date;
  isTyping?: boolean;
  typingIn?: string;
}

export interface ConflictResolution {
  id: string;
  resourceType: 'email' | 'folder' | 'trust';
  resourceId: string;
  localVersion: unknown;
  remoteVersion: unknown;
  baseVersion?: unknown;
  resolvedVersion?: unknown;
  strategy: 'local_wins' | 'remote_wins' | 'merge' | 'manual';
  resolvedAt?: Date;
}

export interface OptimisticUpdate {
  id: string;
  type: SyncMessageType;
  payload: unknown;
  timestamp: number;
  status: 'pending' | 'confirmed' | 'failed';
  retryCount: number;
  rollbackData?: unknown;
}

// ============================================================================
// WebSocket Manager
// ============================================================================

class WebSocketManager {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelay = 1000;
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private messageQueue: SyncMessage[] = [];
  private listeners: Map<string, Set<(message: SyncMessage) => void>> = new Map();

  constructor(url: string) {
    this.url = url;
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }

      useRealtimeSyncStore.getState().setConnectionState('connecting');

      try {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
          this.reconnectAttempts = 0;
          this.reconnectDelay = 1000;
          useRealtimeSyncStore.getState().setConnectionState('connected');
          this.startHeartbeat();
          this.flushMessageQueue();
          this.sendPresence('join');
          resolve();
        };

        this.ws.onclose = (event) => {
          this.stopHeartbeat();
          useRealtimeSyncStore.getState().setConnectionState('disconnected');

          if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          reject(error);
        };

        this.ws.onmessage = (event) => {
          try {
            const message: SyncMessage = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (err) {
            console.error('Failed to parse WebSocket message:', err);
          }
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  disconnect(): void {
    this.sendPresence('leave');
    this.stopHeartbeat();
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    useRealtimeSyncStore.getState().setConnectionState('disconnected');
  }

  send(message: Omit<SyncMessage, 'id' | 'timestamp'>): string {
    const fullMessage: SyncMessage = {
      ...message,
      id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: Date.now(),
    };

    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(fullMessage));
    } else {
      this.messageQueue.push(fullMessage);
    }

    // Track optimistic update
    useRealtimeSyncStore.getState().addOptimisticUpdate({
      id: fullMessage.id,
      type: fullMessage.type,
      payload: fullMessage.payload,
      timestamp: fullMessage.timestamp,
      status: 'pending',
      retryCount: 0,
    });

    return fullMessage.id;
  }

  subscribe(type: SyncMessageType | '*', callback: (message: SyncMessage) => void): () => void {
    if (!this.listeners.has(type)) {
      this.listeners.set(type, new Set());
    }
    this.listeners.get(type)!.add(callback);

    return () => {
      this.listeners.get(type)?.delete(callback);
    };
  }

  private handleMessage(message: SyncMessage): void {
    // Notify specific listeners
    this.listeners.get(message.type)?.forEach((cb) => cb(message));
    // Notify wildcard listeners
    this.listeners.get('*')?.forEach((cb) => cb(message));

    // Update store based on message type
    const store = useRealtimeSyncStore.getState();

    switch (message.type) {
      case 'presence:join':
      case 'presence:leave':
        store.updatePresence(message.payload as PresenceUser, message.type === 'presence:leave');
        break;

      case 'presence:typing':
        store.setTypingUser(
          (message.payload as { userId: string; emailId: string }).userId,
          (message.payload as { userId: string; emailId: string }).emailId
        );
        break;

      case 'conflict:detected':
        store.addConflict(message.payload as ConflictResolution);
        break;

      case 'sync:response':
        // Mark optimistic update as confirmed
        store.confirmOptimisticUpdate(message.id);
        break;
    }
  }

  private sendPresence(action: 'join' | 'leave'): void {
    const user = useRealtimeSyncStore.getState().currentUser;
    if (user) {
      this.send({
        type: action === 'join' ? 'presence:join' : 'presence:leave',
        payload: user,
      });
    }
  }

  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  private scheduleReconnect(): void {
    useRealtimeSyncStore.getState().setConnectionState('reconnecting');
    this.reconnectAttempts++;

    const delay = Math.min(
      this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1),
      30000
    );

    setTimeout(() => {
      this.connect().catch(console.error);
    }, delay);
  }

  private flushMessageQueue(): void {
    while (this.messageQueue.length > 0 && this.ws?.readyState === WebSocket.OPEN) {
      const message = this.messageQueue.shift()!;
      this.ws.send(JSON.stringify(message));
    }
  }
}

// ============================================================================
// Store
// ============================================================================

interface RealtimeSyncState {
  connectionState: ConnectionState;
  currentUser: PresenceUser | null;
  onlineUsers: Map<string, PresenceUser>;
  typingUsers: Map<string, string>; // emailId -> userId
  optimisticUpdates: Map<string, OptimisticUpdate>;
  conflicts: ConflictResolution[];
  lastSyncTimestamp: number;

  // Actions
  setConnectionState: (state: ConnectionState) => void;
  setCurrentUser: (user: PresenceUser | null) => void;
  updatePresence: (user: PresenceUser, isLeaving?: boolean) => void;
  setTypingUser: (userId: string, emailId: string | null) => void;
  addOptimisticUpdate: (update: OptimisticUpdate) => void;
  confirmOptimisticUpdate: (id: string) => void;
  failOptimisticUpdate: (id: string) => void;
  addConflict: (conflict: ConflictResolution) => void;
  resolveConflict: (id: string, resolution: ConflictResolution) => void;
  setLastSyncTimestamp: (timestamp: number) => void;
}

export const useRealtimeSyncStore = create<RealtimeSyncState>()(
  subscribeWithSelector((set, get) => ({
    connectionState: 'disconnected',
    currentUser: null,
    onlineUsers: new Map(),
    typingUsers: new Map(),
    optimisticUpdates: new Map(),
    conflicts: [],
    lastSyncTimestamp: 0,

    setConnectionState: (state) => set({ connectionState: state }),

    setCurrentUser: (user) => set({ currentUser: user }),

    updatePresence: (user, isLeaving = false) => {
      set((state) => {
        const onlineUsers = new Map(state.onlineUsers);
        if (isLeaving) {
          onlineUsers.delete(user.id);
        } else {
          onlineUsers.set(user.id, { ...user, lastSeen: new Date() });
        }
        return { onlineUsers };
      });
    },

    setTypingUser: (userId, emailId) => {
      set((state) => {
        const typingUsers = new Map(state.typingUsers);
        if (emailId) {
          typingUsers.set(emailId, userId);
          // Auto-clear after 3 seconds
          setTimeout(() => {
            const current = get().typingUsers.get(emailId);
            if (current === userId) {
              set((s) => {
                const updated = new Map(s.typingUsers);
                updated.delete(emailId);
                return { typingUsers: updated };
              });
            }
          }, 3000);
        } else {
          // Remove all typing indicators for this user
          typingUsers.forEach((uid, eid) => {
            if (uid === userId) typingUsers.delete(eid);
          });
        }
        return { typingUsers };
      });
    },

    addOptimisticUpdate: (update) => {
      set((state) => {
        const optimisticUpdates = new Map(state.optimisticUpdates);
        optimisticUpdates.set(update.id, update);
        return { optimisticUpdates };
      });
    },

    confirmOptimisticUpdate: (id) => {
      set((state) => {
        const optimisticUpdates = new Map(state.optimisticUpdates);
        const update = optimisticUpdates.get(id);
        if (update) {
          optimisticUpdates.set(id, { ...update, status: 'confirmed' });
          // Remove after short delay
          setTimeout(() => {
            set((s) => {
              const updates = new Map(s.optimisticUpdates);
              updates.delete(id);
              return { optimisticUpdates: updates };
            });
          }, 1000);
        }
        return { optimisticUpdates };
      });
    },

    failOptimisticUpdate: (id) => {
      set((state) => {
        const optimisticUpdates = new Map(state.optimisticUpdates);
        const update = optimisticUpdates.get(id);
        if (update) {
          optimisticUpdates.set(id, { ...update, status: 'failed' });
        }
        return { optimisticUpdates };
      });
    },

    addConflict: (conflict) => {
      set((state) => ({
        conflicts: [...state.conflicts, conflict],
      }));
    },

    resolveConflict: (id, resolution) => {
      set((state) => ({
        conflicts: state.conflicts.map((c) =>
          c.id === id ? { ...c, ...resolution, resolvedAt: new Date() } : c
        ),
      }));
    },

    setLastSyncTimestamp: (timestamp) => set({ lastSyncTimestamp: timestamp }),
  }))
);

// ============================================================================
// Singleton Instance
// ============================================================================

let wsManager: WebSocketManager | null = null;

export function getWebSocketManager(url?: string): WebSocketManager {
  if (!wsManager && url) {
    wsManager = new WebSocketManager(url);
  }
  if (!wsManager) {
    throw new Error('WebSocket manager not initialized. Provide URL on first call.');
  }
  return wsManager;
}

// ============================================================================
// React Hooks
// ============================================================================

import { useEffect, useCallback, useRef } from 'react';

export function useRealtimeSync(url: string) {
  const wsRef = useRef<WebSocketManager | null>(null);

  useEffect(() => {
    wsRef.current = getWebSocketManager(url);
    wsRef.current.connect().catch(console.error);

    return () => {
      wsRef.current?.disconnect();
    };
  }, [url]);

  const send = useCallback((type: SyncMessageType, payload: unknown) => {
    return wsRef.current?.send({ type, payload });
  }, []);

  const subscribe = useCallback(
    (type: SyncMessageType | '*', callback: (message: SyncMessage) => void) => {
      return wsRef.current?.subscribe(type, callback) || (() => {});
    },
    []
  );

  const connectionState = useRealtimeSyncStore((s) => s.connectionState);
  const onlineUsers = useRealtimeSyncStore((s) => s.onlineUsers);

  return {
    connectionState,
    onlineUsers,
    send,
    subscribe,
    isConnected: connectionState === 'connected',
  };
}

export function usePresence() {
  const onlineUsers = useRealtimeSyncStore((s) => s.onlineUsers);
  const typingUsers = useRealtimeSyncStore((s) => s.typingUsers);
  const currentUser = useRealtimeSyncStore((s) => s.currentUser);

  const setTyping = useCallback((emailId: string) => {
    const ws = getWebSocketManager();
    ws.send({
      type: 'presence:typing',
      payload: { userId: currentUser?.id, emailId },
    });
  }, [currentUser]);

  const setViewing = useCallback((view: string) => {
    const ws = getWebSocketManager();
    ws.send({
      type: 'presence:viewing',
      payload: { userId: currentUser?.id, view },
    });
  }, [currentUser]);

  return {
    onlineUsers: Array.from(onlineUsers.values()),
    typingUsers,
    setTyping,
    setViewing,
    onlineCount: onlineUsers.size,
  };
}

export function useOptimisticUpdates() {
  const optimisticUpdates = useRealtimeSyncStore((s) => s.optimisticUpdates);
  const confirmOptimisticUpdate = useRealtimeSyncStore((s) => s.confirmOptimisticUpdate);
  const failOptimisticUpdate = useRealtimeSyncStore((s) => s.failOptimisticUpdate);

  const pendingUpdates = Array.from(optimisticUpdates.values()).filter(
    (u) => u.status === 'pending'
  );

  const failedUpdates = Array.from(optimisticUpdates.values()).filter(
    (u) => u.status === 'failed'
  );

  return {
    pendingUpdates,
    failedUpdates,
    hasPending: pendingUpdates.length > 0,
    confirmOptimisticUpdate,
    failOptimisticUpdate,
  };
}

export function useConflictResolution() {
  const conflicts = useRealtimeSyncStore((s) => s.conflicts);
  const resolveConflict = useRealtimeSyncStore((s) => s.resolveConflict);

  const unresolvedConflicts = conflicts.filter((c) => !c.resolvedAt);

  const autoResolve = useCallback(
    (id: string, strategy: ConflictResolution['strategy']) => {
      const conflict = conflicts.find((c) => c.id === id);
      if (!conflict) return;

      let resolvedVersion: unknown;
      switch (strategy) {
        case 'local_wins':
          resolvedVersion = conflict.localVersion;
          break;
        case 'remote_wins':
          resolvedVersion = conflict.remoteVersion;
          break;
        case 'merge':
          // Simple merge strategy - in production would be more sophisticated
          resolvedVersion = {
            ...(conflict.remoteVersion as object),
            ...(conflict.localVersion as object),
          };
          break;
      }

      resolveConflict(id, { ...conflict, resolvedVersion, strategy });
    },
    [conflicts, resolveConflict]
  );

  return {
    conflicts: unresolvedConflicts,
    hasConflicts: unresolvedConflicts.length > 0,
    resolveConflict,
    autoResolve,
  };
}

// ============================================================================
// Sync Utilities
// ============================================================================

export function createSyncedAction<T>(
  type: SyncMessageType,
  action: (payload: T) => void,
  rollback?: (payload: T) => void
) {
  return (payload: T) => {
    // Apply optimistically
    action(payload);

    // Send to server
    const ws = getWebSocketManager();
    const messageId = ws.send({ type, payload });

    // Store rollback data
    if (rollback && messageId) {
      const store = useRealtimeSyncStore.getState();
      const update = store.optimisticUpdates.get(messageId);
      if (update) {
        store.addOptimisticUpdate({ ...update, rollbackData: payload });
      }
    }

    return messageId;
  };
}

export function withOptimisticUpdate<T, R>(
  asyncFn: (payload: T) => Promise<R>,
  optimisticUpdate: (payload: T) => void,
  rollback: (payload: T) => void
) {
  return async (payload: T): Promise<R> => {
    // Apply optimistic update
    optimisticUpdate(payload);

    try {
      const result = await asyncFn(payload);
      return result;
    } catch (error) {
      // Rollback on failure
      rollback(payload);
      throw error;
    }
  };
}
