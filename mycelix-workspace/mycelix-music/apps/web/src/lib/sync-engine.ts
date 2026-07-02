// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Offline-First Sync Engine
 *
 * Handles data synchronization with:
 * - Operation queue with retry
 * - Conflict resolution
 * - Background sync
 * - Real-time updates via WebSocket
 * - Delta sync for efficiency
 */

import { useAppStore, PendingOperation } from '../store';

// ==================== Types ====================

export interface SyncConfig {
  apiBaseUrl: string;
  wsUrl: string;
  syncInterval: number;      // ms between sync attempts
  maxRetries: number;
  retryDelay: number;        // base delay in ms
  batchSize: number;         // max operations per batch
  enableRealtime: boolean;
}

export interface SyncResult {
  success: boolean;
  synced: number;
  failed: number;
  conflicts: ConflictResolution[];
  errors: SyncError[];
}

export interface SyncError {
  operationId: string;
  error: string;
  retryable: boolean;
}

export interface ConflictResolution {
  operationId: string;
  entity: string;
  entityId: string;
  resolution: 'local' | 'remote' | 'merged';
  mergedData?: unknown;
}

export interface ServerChange {
  entity: string;
  entityId: string;
  action: 'create' | 'update' | 'delete';
  data?: unknown;
  timestamp: Date;
  version: number;
}

export type ConflictStrategy = 'local-wins' | 'remote-wins' | 'last-write-wins' | 'manual';

// ==================== Default Config ====================

const defaultConfig: SyncConfig = {
  apiBaseUrl: process.env.NEXT_PUBLIC_API_URL || '/api',
  wsUrl: process.env.NEXT_PUBLIC_WS_URL || 'wss://api.mycelix.music/ws',
  syncInterval: 30000,       // 30 seconds
  maxRetries: 5,
  retryDelay: 1000,
  batchSize: 50,
  enableRealtime: true,
};

// ==================== Sync Engine Class ====================

class SyncEngine {
  private config: SyncConfig;
  private ws: WebSocket | null = null;
  private syncInterval: ReturnType<typeof setInterval> | null = null;
  private isSyncing = false;
  private retryCount: Map<string, number> = new Map();
  private localVersions: Map<string, number> = new Map();
  private conflictStrategy: ConflictStrategy = 'last-write-wins';
  private listeners: Set<(event: SyncEvent) => void> = new Set();

  constructor(config: Partial<SyncConfig> = {}) {
    this.config = { ...defaultConfig, ...config };
  }

  // ==================== Lifecycle ====================

  async start(): Promise<void> {
    // Start periodic sync
    this.syncInterval = setInterval(() => {
      this.sync();
    }, this.config.syncInterval);

    // Connect WebSocket for real-time updates
    if (this.config.enableRealtime) {
      this.connectWebSocket();
    }

    // Listen for online/offline events
    if (typeof window !== 'undefined') {
      window.addEventListener('online', this.handleOnline);
      window.addEventListener('offline', this.handleOffline);

      // Sync when coming back online
      if (navigator.onLine) {
        this.sync();
      }
    }

    this.emit({ type: 'started' });
  }

  stop(): void {
    if (this.syncInterval) {
      clearInterval(this.syncInterval);
      this.syncInterval = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    if (typeof window !== 'undefined') {
      window.removeEventListener('online', this.handleOnline);
      window.removeEventListener('offline', this.handleOffline);
    }

    this.emit({ type: 'stopped' });
  }

  // ==================== WebSocket ====================

  private connectWebSocket(): void {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    try {
      this.ws = new WebSocket(this.config.wsUrl);

      this.ws.onopen = () => {
        console.log('Sync WebSocket connected');
        this.emit({ type: 'connected' });

        // Authenticate
        const token = localStorage.getItem('auth_token');
        if (token) {
          this.ws?.send(JSON.stringify({ type: 'auth', token }));
        }
      };

      this.ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          this.handleServerMessage(message);
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      this.ws.onclose = () => {
        console.log('Sync WebSocket disconnected');
        this.emit({ type: 'disconnected' });

        // Reconnect after delay
        if (this.config.enableRealtime) {
          setTimeout(() => this.connectWebSocket(), 5000);
        }
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
    }
  }

  private handleServerMessage(message: {
    type: string;
    changes?: ServerChange[];
    data?: unknown;
  }): void {
    switch (message.type) {
      case 'changes':
        if (message.changes) {
          this.applyServerChanges(message.changes);
        }
        break;

      case 'sync_required':
        this.sync();
        break;

      case 'conflict':
        this.emit({ type: 'conflict', data: message.data });
        break;

      default:
        console.log('Unknown message type:', message.type);
    }
  }

  // ==================== Sync Logic ====================

  async sync(): Promise<SyncResult> {
    if (this.isSyncing || !navigator.onLine) {
      return { success: false, synced: 0, failed: 0, conflicts: [], errors: [] };
    }

    this.isSyncing = true;
    this.emit({ type: 'sync_started' });

    const result: SyncResult = {
      success: true,
      synced: 0,
      failed: 0,
      conflicts: [],
      errors: [],
    };

    try {
      const store = useAppStore.getState();
      const operations = [...store.pendingOperations];

      if (operations.length === 0) {
        this.isSyncing = false;
        this.emit({ type: 'sync_completed', data: result });
        return result;
      }

      // Process in batches
      for (let i = 0; i < operations.length; i += this.config.batchSize) {
        const batch = operations.slice(i, i + this.config.batchSize);
        const batchResult = await this.processBatch(batch);

        result.synced += batchResult.synced;
        result.failed += batchResult.failed;
        result.conflicts.push(...batchResult.conflicts);
        result.errors.push(...batchResult.errors);
      }

      // Clear synced operations
      const syncedIds = operations
        .slice(0, result.synced)
        .map(op => op.id);

      useAppStore.setState((state) => ({
        pendingOperations: state.pendingOperations.filter(
          op => !syncedIds.includes(op.id)
        ),
        syncStatus: 'idle',
        lastSyncTime: new Date(),
      }));

      result.success = result.failed === 0;
    } catch (error) {
      console.error('Sync failed:', error);
      result.success = false;

      useAppStore.setState({ syncStatus: 'error' });
    } finally {
      this.isSyncing = false;
      this.emit({ type: 'sync_completed', data: result });
    }

    return result;
  }

  private async processBatch(operations: PendingOperation[]): Promise<SyncResult> {
    const result: SyncResult = {
      success: true,
      synced: 0,
      failed: 0,
      conflicts: [],
      errors: [],
    };

    for (const operation of operations) {
      try {
        const response = await this.sendOperation(operation);

        if (response.conflict) {
          const resolution = await this.resolveConflict(operation, response);
          result.conflicts.push(resolution);

          if (resolution.resolution !== 'local') {
            // Apply remote changes
            this.applyServerChanges([{
              entity: operation.entity,
              entityId: operation.entityId,
              action: operation.type,
              data: resolution.mergedData || response.serverData,
              timestamp: new Date(),
              version: response.serverVersion,
            }]);
          }
        }

        result.synced++;
        this.retryCount.delete(operation.id);

        // Update local version
        const key = `${operation.entity}:${operation.entityId}`;
        this.localVersions.set(key, response.version || 1);
      } catch (error) {
        const retries = this.retryCount.get(operation.id) || 0;

        if (retries < this.config.maxRetries) {
          this.retryCount.set(operation.id, retries + 1);
          result.errors.push({
            operationId: operation.id,
            error: (error as Error).message,
            retryable: true,
          });
        } else {
          result.failed++;
          result.errors.push({
            operationId: operation.id,
            error: (error as Error).message,
            retryable: false,
          });
        }
      }
    }

    return result;
  }

  private async sendOperation(operation: PendingOperation): Promise<{
    success: boolean;
    version?: number;
    conflict?: boolean;
    serverData?: unknown;
    serverVersion?: number;
  }> {
    const endpoint = `${this.config.apiBaseUrl}/sync`;

    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
      },
      body: JSON.stringify({
        operation,
        clientVersion: this.localVersions.get(
          `${operation.entity}:${operation.entityId}`
        ) || 0,
      }),
    });

    if (!response.ok) {
      throw new Error(`Sync request failed: ${response.status}`);
    }

    return response.json();
  }

  // ==================== Conflict Resolution ====================

  private async resolveConflict(
    operation: PendingOperation,
    serverResponse: { serverData?: unknown; serverVersion?: number }
  ): Promise<ConflictResolution> {
    const resolution: ConflictResolution = {
      operationId: operation.id,
      entity: operation.entity,
      entityId: operation.entityId,
      resolution: 'remote',
    };

    switch (this.conflictStrategy) {
      case 'local-wins':
        resolution.resolution = 'local';
        break;

      case 'remote-wins':
        resolution.resolution = 'remote';
        break;

      case 'last-write-wins':
        // Compare timestamps
        const serverTime = new Date(serverResponse.serverData as { updatedAt: string }).getTime?.() || 0;
        const localTime = operation.timestamp.getTime();
        resolution.resolution = localTime > serverTime ? 'local' : 'remote';
        break;

      case 'manual':
        // Emit event for manual resolution
        this.emit({
          type: 'conflict_needs_resolution',
          data: { operation, serverData: serverResponse.serverData },
        });
        // Default to remote while waiting for resolution
        resolution.resolution = 'remote';
        break;
    }

    // For merged resolution, combine the data
    if (resolution.resolution === 'merged') {
      resolution.mergedData = this.mergeData(
        operation.data,
        serverResponse.serverData
      );
    }

    return resolution;
  }

  private mergeData(local: unknown, remote: unknown): unknown {
    // Simple merge: prefer local for non-null values
    if (typeof local !== 'object' || typeof remote !== 'object') {
      return local ?? remote;
    }

    return {
      ...(remote as object),
      ...(local as object),
    };
  }

  // ==================== Apply Server Changes ====================

  private applyServerChanges(changes: ServerChange[]): void {
    const store = useAppStore.getState();

    changes.forEach(change => {
      const key = `${change.entity}:${change.entityId}`;

      // Check if we have a newer local version
      const localVersion = this.localVersions.get(key) || 0;
      if (change.version <= localVersion) {
        return; // Skip older changes
      }

      this.localVersions.set(key, change.version);

      // Apply change based on entity type
      switch (change.entity) {
        case 'track':
          this.applyTrackChange(change);
          break;
        case 'playlist':
          this.applyPlaylistChange(change);
          break;
        case 'project':
          this.applyProjectChange(change);
          break;
        default:
          console.log('Unknown entity type:', change.entity);
      }
    });

    this.emit({ type: 'changes_applied', data: changes });
  }

  private applyTrackChange(change: ServerChange): void {
    const { addTrack, removeTrack } = useAppStore.getState();

    switch (change.action) {
      case 'create':
      case 'update':
        if (change.data) {
          addTrack(change.data as Parameters<typeof addTrack>[0]);
        }
        break;
      case 'delete':
        removeTrack(change.entityId);
        break;
    }
  }

  private applyPlaylistChange(change: ServerChange): void {
    const { addPlaylist, updatePlaylist, deletePlaylist } = useAppStore.getState();

    switch (change.action) {
      case 'create':
        if (change.data) {
          addPlaylist(change.data as Parameters<typeof addPlaylist>[0]);
        }
        break;
      case 'update':
        if (change.data) {
          updatePlaylist(change.entityId, change.data as Record<string, unknown>);
        }
        break;
      case 'delete':
        deletePlaylist(change.entityId);
        break;
    }
  }

  private applyProjectChange(change: ServerChange): void {
    const { updateProject } = useAppStore.getState();

    switch (change.action) {
      case 'update':
        if (change.data) {
          updateProject(change.data as Parameters<typeof updateProject>[0]);
        }
        break;
    }
  }

  // ==================== Event Handlers ====================

  private handleOnline = (): void => {
    console.log('Network: online');
    this.emit({ type: 'online' });

    // Reconnect WebSocket
    if (this.config.enableRealtime) {
      this.connectWebSocket();
    }

    // Trigger sync
    this.sync();
  };

  private handleOffline = (): void => {
    console.log('Network: offline');
    this.emit({ type: 'offline' });

    useAppStore.setState({ syncStatus: 'idle' });
  };

  // ==================== Event System ====================

  subscribe(listener: (event: SyncEvent) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private emit(event: SyncEvent): void {
    this.listeners.forEach(listener => listener(event));
  }

  // ==================== Public API ====================

  setConflictStrategy(strategy: ConflictStrategy): void {
    this.conflictStrategy = strategy;
  }

  getStatus(): {
    isConnected: boolean;
    isSyncing: boolean;
    pendingCount: number;
    lastSyncTime: Date | null;
  } {
    const store = useAppStore.getState();
    return {
      isConnected: this.ws?.readyState === WebSocket.OPEN,
      isSyncing: this.isSyncing,
      pendingCount: store.pendingOperations.length,
      lastSyncTime: store.lastSyncTime,
    };
  }

  async forcSync(): Promise<SyncResult> {
    return this.sync();
  }

  async resolveManualConflict(
    operationId: string,
    resolution: 'local' | 'remote',
    mergedData?: unknown
  ): Promise<void> {
    // Implementation for manual conflict resolution
    console.log('Manual conflict resolved:', operationId, resolution, mergedData);
  }
}

// ==================== Types ====================

export interface SyncEvent {
  type:
    | 'started'
    | 'stopped'
    | 'connected'
    | 'disconnected'
    | 'online'
    | 'offline'
    | 'sync_started'
    | 'sync_completed'
    | 'conflict'
    | 'conflict_needs_resolution'
    | 'changes_applied';
  data?: unknown;
}

// ==================== Singleton Instance ====================

let syncEngine: SyncEngine | null = null;

export function getSyncEngine(config?: Partial<SyncConfig>): SyncEngine {
  if (!syncEngine) {
    syncEngine = new SyncEngine(config);
  }
  return syncEngine;
}

// ==================== React Hook ====================

import { useState, useEffect, useCallback } from 'react';

export function useSync() {
  const [status, setStatus] = useState<ReturnType<SyncEngine['getStatus']>>({
    isConnected: false,
    isSyncing: false,
    pendingCount: 0,
    lastSyncTime: null,
  });

  useEffect(() => {
    const engine = getSyncEngine();

    // Update status periodically
    const updateStatus = () => setStatus(engine.getStatus());
    updateStatus();
    const interval = setInterval(updateStatus, 1000);

    // Listen for sync events
    const unsubscribe = engine.subscribe((event) => {
      updateStatus();

      // Handle specific events
      if (event.type === 'sync_completed') {
        console.log('Sync completed:', event.data);
      }
    });

    return () => {
      clearInterval(interval);
      unsubscribe();
    };
  }, []);

  const forceSync = useCallback(async () => {
    const engine = getSyncEngine();
    return engine.forcSync();
  }, []);

  return {
    ...status,
    forceSync,
  };
}

export default SyncEngine;
