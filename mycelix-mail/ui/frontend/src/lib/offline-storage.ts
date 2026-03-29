// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Offline Storage Service
 *
 * Provides:
 * - IndexedDB wrapper for offline data persistence
 * - Sync queue for pending operations
 * - Conflict detection and resolution
 * - Background sync via Service Worker
 * - Storage quota management
 */

// ============================================================================
// Types
// ============================================================================

export interface StoredEmail {
  id: string;
  threadId: string;
  folderId: string;
  from: { name: string; email: string };
  to: { name: string; email: string }[];
  cc?: { name: string; email: string }[];
  subject: string;
  body: string;
  bodyHtml?: string;
  snippet: string;
  date: Date;
  isRead: boolean;
  isStarred: boolean;
  isArchived: boolean;
  labels: string[];
  attachments: StoredAttachment[];
  trustScore?: number;
  aiAnalysis?: object;
  syncStatus: 'synced' | 'pending' | 'conflict';
  localVersion: number;
  serverVersion: number;
  lastModified: Date;
}

export interface StoredAttachment {
  id: string;
  emailId: string;
  filename: string;
  mimeType: string;
  size: number;
  data?: ArrayBuffer;
  isDownloaded: boolean;
}

export interface StoredFolder {
  id: string;
  name: string;
  parentId?: string;
  type: 'system' | 'user' | 'smart';
  icon?: string;
  color?: string;
  unreadCount: number;
  totalCount: number;
  syncStatus: 'synced' | 'pending';
}

export interface StoredContact {
  id: string;
  email: string;
  name?: string;
  avatar?: string;
  trustScore: number;
  lastContact?: Date;
  frequency: number;
  tags: string[];
  publicKey?: string;
  syncStatus: 'synced' | 'pending';
}

export interface SyncQueueItem {
  id: string;
  operation: 'create' | 'update' | 'delete';
  resourceType: 'email' | 'folder' | 'contact' | 'trust';
  resourceId: string;
  payload: unknown;
  timestamp: Date;
  retryCount: number;
  maxRetries: number;
  status: 'pending' | 'in_progress' | 'failed';
  error?: string;
}

export interface StorageStats {
  usedBytes: number;
  quotaBytes: number;
  emailCount: number;
  attachmentBytes: number;
  lastSync: Date | null;
  pendingOperations: number;
}

// ============================================================================
// IndexedDB Schema
// ============================================================================

const DB_NAME = 'mycelix-mail-offline';
const DB_VERSION = 1;

const STORES = {
  emails: 'emails',
  attachments: 'attachments',
  folders: 'folders',
  contacts: 'contacts',
  syncQueue: 'syncQueue',
  metadata: 'metadata',
} as const;

// ============================================================================
// Database Manager
// ============================================================================

class OfflineDatabase {
  private db: IDBDatabase | null = null;
  private dbPromise: Promise<IDBDatabase> | null = null;

  async open(): Promise<IDBDatabase> {
    if (this.db) return this.db;
    if (this.dbPromise) return this.dbPromise;

    this.dbPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onerror = () => reject(request.error);

      request.onsuccess = () => {
        this.db = request.result;
        resolve(this.db);
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;

        // Emails store
        if (!db.objectStoreNames.contains(STORES.emails)) {
          const emailStore = db.createObjectStore(STORES.emails, { keyPath: 'id' });
          emailStore.createIndex('threadId', 'threadId', { unique: false });
          emailStore.createIndex('folderId', 'folderId', { unique: false });
          emailStore.createIndex('date', 'date', { unique: false });
          emailStore.createIndex('from.email', 'from.email', { unique: false });
          emailStore.createIndex('syncStatus', 'syncStatus', { unique: false });
          emailStore.createIndex('isRead', 'isRead', { unique: false });
        }

        // Attachments store
        if (!db.objectStoreNames.contains(STORES.attachments)) {
          const attachmentStore = db.createObjectStore(STORES.attachments, { keyPath: 'id' });
          attachmentStore.createIndex('emailId', 'emailId', { unique: false });
        }

        // Folders store
        if (!db.objectStoreNames.contains(STORES.folders)) {
          const folderStore = db.createObjectStore(STORES.folders, { keyPath: 'id' });
          folderStore.createIndex('parentId', 'parentId', { unique: false });
          folderStore.createIndex('type', 'type', { unique: false });
        }

        // Contacts store
        if (!db.objectStoreNames.contains(STORES.contacts)) {
          const contactStore = db.createObjectStore(STORES.contacts, { keyPath: 'id' });
          contactStore.createIndex('email', 'email', { unique: true });
          contactStore.createIndex('trustScore', 'trustScore', { unique: false });
        }

        // Sync queue store
        if (!db.objectStoreNames.contains(STORES.syncQueue)) {
          const syncStore = db.createObjectStore(STORES.syncQueue, { keyPath: 'id' });
          syncStore.createIndex('status', 'status', { unique: false });
          syncStore.createIndex('timestamp', 'timestamp', { unique: false });
          syncStore.createIndex('resourceType', 'resourceType', { unique: false });
        }

        // Metadata store
        if (!db.objectStoreNames.contains(STORES.metadata)) {
          db.createObjectStore(STORES.metadata, { keyPath: 'key' });
        }
      };
    });

    return this.dbPromise;
  }

  async close(): Promise<void> {
    if (this.db) {
      this.db.close();
      this.db = null;
      this.dbPromise = null;
    }
  }

  private async getStore(
    storeName: string,
    mode: IDBTransactionMode = 'readonly'
  ): Promise<IDBObjectStore> {
    const db = await this.open();
    const tx = db.transaction(storeName, mode);
    return tx.objectStore(storeName);
  }

  // Generic CRUD operations
  async get<T>(storeName: string, key: string): Promise<T | undefined> {
    const store = await this.getStore(storeName);
    return new Promise((resolve, reject) => {
      const request = store.get(key);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async getAll<T>(storeName: string, query?: IDBKeyRange): Promise<T[]> {
    const store = await this.getStore(storeName);
    return new Promise((resolve, reject) => {
      const request = store.getAll(query);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async getAllByIndex<T>(
    storeName: string,
    indexName: string,
    value: IDBValidKey
  ): Promise<T[]> {
    const store = await this.getStore(storeName);
    const index = store.index(indexName);
    return new Promise((resolve, reject) => {
      const request = index.getAll(value);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async put<T>(storeName: string, value: T): Promise<void> {
    const store = await this.getStore(storeName, 'readwrite');
    return new Promise((resolve, reject) => {
      const request = store.put(value);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  async delete(storeName: string, key: string): Promise<void> {
    const store = await this.getStore(storeName, 'readwrite');
    return new Promise((resolve, reject) => {
      const request = store.delete(key);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  async clear(storeName: string): Promise<void> {
    const store = await this.getStore(storeName, 'readwrite');
    return new Promise((resolve, reject) => {
      const request = store.clear();
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  async count(storeName: string): Promise<number> {
    const store = await this.getStore(storeName);
    return new Promise((resolve, reject) => {
      const request = store.count();
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }
}

// ============================================================================
// Offline Storage Service
// ============================================================================

class OfflineStorageService {
  private db = new OfflineDatabase();
  private syncInProgress = false;

  // Email operations
  async getEmail(id: string): Promise<StoredEmail | undefined> {
    return this.db.get<StoredEmail>(STORES.emails, id);
  }

  async getEmails(folderId?: string): Promise<StoredEmail[]> {
    if (folderId) {
      return this.db.getAllByIndex<StoredEmail>(STORES.emails, 'folderId', folderId);
    }
    return this.db.getAll<StoredEmail>(STORES.emails);
  }

  async getEmailsByThread(threadId: string): Promise<StoredEmail[]> {
    return this.db.getAllByIndex<StoredEmail>(STORES.emails, 'threadId', threadId);
  }

  async saveEmail(email: StoredEmail): Promise<void> {
    await this.db.put(STORES.emails, {
      ...email,
      lastModified: new Date(),
    });
  }

  async saveEmails(emails: StoredEmail[]): Promise<void> {
    for (const email of emails) {
      await this.saveEmail(email);
    }
  }

  async deleteEmail(id: string): Promise<void> {
    await this.db.delete(STORES.emails, id);
    // Also delete attachments
    const attachments = await this.db.getAllByIndex<StoredAttachment>(
      STORES.attachments,
      'emailId',
      id
    );
    for (const att of attachments) {
      await this.db.delete(STORES.attachments, att.id);
    }
  }

  async updateEmailStatus(
    id: string,
    updates: Partial<Pick<StoredEmail, 'isRead' | 'isStarred' | 'isArchived' | 'folderId'>>
  ): Promise<void> {
    const email = await this.getEmail(id);
    if (email) {
      await this.saveEmail({
        ...email,
        ...updates,
        syncStatus: 'pending',
        localVersion: email.localVersion + 1,
      });
      await this.queueSync('update', 'email', id, updates);
    }
  }

  // Folder operations
  async getFolders(): Promise<StoredFolder[]> {
    return this.db.getAll<StoredFolder>(STORES.folders);
  }

  async saveFolder(folder: StoredFolder): Promise<void> {
    await this.db.put(STORES.folders, folder);
  }

  // Contact operations
  async getContact(id: string): Promise<StoredContact | undefined> {
    return this.db.get<StoredContact>(STORES.contacts, id);
  }

  async getContactByEmail(email: string): Promise<StoredContact | undefined> {
    const contacts = await this.db.getAllByIndex<StoredContact>(
      STORES.contacts,
      'email',
      email
    );
    return contacts[0];
  }

  async getContacts(): Promise<StoredContact[]> {
    return this.db.getAll<StoredContact>(STORES.contacts);
  }

  async saveContact(contact: StoredContact): Promise<void> {
    await this.db.put(STORES.contacts, contact);
  }

  // Attachment operations
  async getAttachment(id: string): Promise<StoredAttachment | undefined> {
    return this.db.get<StoredAttachment>(STORES.attachments, id);
  }

  async saveAttachment(attachment: StoredAttachment): Promise<void> {
    await this.db.put(STORES.attachments, attachment);
  }

  async downloadAttachment(id: string, data: ArrayBuffer): Promise<void> {
    const attachment = await this.getAttachment(id);
    if (attachment) {
      await this.saveAttachment({
        ...attachment,
        data,
        isDownloaded: true,
      });
    }
  }

  // Sync queue operations
  async queueSync(
    operation: SyncQueueItem['operation'],
    resourceType: SyncQueueItem['resourceType'],
    resourceId: string,
    payload: unknown
  ): Promise<void> {
    const item: SyncQueueItem = {
      id: `sync_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      operation,
      resourceType,
      resourceId,
      payload,
      timestamp: new Date(),
      retryCount: 0,
      maxRetries: 3,
      status: 'pending',
    };
    await this.db.put(STORES.syncQueue, item);

    // Trigger background sync if available
    if ('serviceWorker' in navigator && 'sync' in (ServiceWorkerRegistration.prototype as object)) {
      const registration = await navigator.serviceWorker.ready;
      await (registration as ServiceWorkerRegistration & { sync: { register: (tag: string) => Promise<void> } }).sync.register('sync-emails');
    }
  }

  async getPendingSyncItems(): Promise<SyncQueueItem[]> {
    return this.db.getAllByIndex<SyncQueueItem>(STORES.syncQueue, 'status', 'pending');
  }

  async updateSyncItem(id: string, updates: Partial<SyncQueueItem>): Promise<void> {
    const item = await this.db.get<SyncQueueItem>(STORES.syncQueue, id);
    if (item) {
      await this.db.put(STORES.syncQueue, { ...item, ...updates });
    }
  }

  async removeSyncItem(id: string): Promise<void> {
    await this.db.delete(STORES.syncQueue, id);
  }

  // Sync execution
  async executePendingSync(
    syncFn: (item: SyncQueueItem) => Promise<boolean>
  ): Promise<{ success: number; failed: number }> {
    if (this.syncInProgress) {
      return { success: 0, failed: 0 };
    }

    this.syncInProgress = true;
    let success = 0;
    let failed = 0;

    try {
      const pendingItems = await this.getPendingSyncItems();

      for (const item of pendingItems) {
        await this.updateSyncItem(item.id, { status: 'in_progress' });

        try {
          const result = await syncFn(item);
          if (result) {
            await this.removeSyncItem(item.id);
            success++;
          } else {
            throw new Error('Sync returned false');
          }
        } catch (error) {
          const newRetryCount = item.retryCount + 1;
          if (newRetryCount >= item.maxRetries) {
            await this.updateSyncItem(item.id, {
              status: 'failed',
              error: (error as Error).message,
              retryCount: newRetryCount,
            });
            failed++;
          } else {
            await this.updateSyncItem(item.id, {
              status: 'pending',
              retryCount: newRetryCount,
            });
          }
        }
      }

      // Update last sync timestamp
      await this.setMetadata('lastSync', new Date().toISOString());
    } finally {
      this.syncInProgress = false;
    }

    return { success, failed };
  }

  // Conflict detection
  async detectConflicts(
    serverEmails: { id: string; version: number; data: Partial<StoredEmail> }[]
  ): Promise<{ id: string; local: StoredEmail; server: Partial<StoredEmail> }[]> {
    const conflicts: { id: string; local: StoredEmail; server: Partial<StoredEmail> }[] = [];

    for (const serverEmail of serverEmails) {
      const localEmail = await this.getEmail(serverEmail.id);
      if (
        localEmail &&
        localEmail.syncStatus === 'pending' &&
        localEmail.serverVersion < serverEmail.version
      ) {
        conflicts.push({
          id: serverEmail.id,
          local: localEmail,
          server: serverEmail.data,
        });

        // Mark as conflict
        await this.saveEmail({
          ...localEmail,
          syncStatus: 'conflict',
        });
      }
    }

    return conflicts;
  }

  // Metadata operations
  async getMetadata<T>(key: string): Promise<T | undefined> {
    const result = await this.db.get<{ key: string; value: T }>(STORES.metadata, key);
    return result?.value;
  }

  async setMetadata<T>(key: string, value: T): Promise<void> {
    await this.db.put(STORES.metadata, { key, value });
  }

  // Storage stats
  async getStorageStats(): Promise<StorageStats> {
    const emailCount = await this.db.count(STORES.emails);
    const pendingOperations = await this.db.count(STORES.syncQueue);
    const lastSync = await this.getMetadata<string>('lastSync');

    // Get attachment sizes
    const attachments = await this.db.getAll<StoredAttachment>(STORES.attachments);
    const attachmentBytes = attachments.reduce(
      (sum, att) => sum + (att.data?.byteLength || 0),
      0
    );

    // Get storage quota if available
    let usedBytes = 0;
    let quotaBytes = 0;
    if ('storage' in navigator && 'estimate' in navigator.storage) {
      const estimate = await navigator.storage.estimate();
      usedBytes = estimate.usage || 0;
      quotaBytes = estimate.quota || 0;
    }

    return {
      usedBytes,
      quotaBytes,
      emailCount,
      attachmentBytes,
      lastSync: lastSync ? new Date(lastSync) : null,
      pendingOperations,
    };
  }

  // Clear all data
  async clearAll(): Promise<void> {
    await this.db.clear(STORES.emails);
    await this.db.clear(STORES.attachments);
    await this.db.clear(STORES.folders);
    await this.db.clear(STORES.contacts);
    await this.db.clear(STORES.syncQueue);
    await this.db.clear(STORES.metadata);
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

export const offlineStorage = new OfflineStorageService();

// ============================================================================
// React Hooks
// ============================================================================

import { useState, useEffect, useCallback } from 'react';
import { create } from 'zustand';

interface OfflineState {
  isOnline: boolean;
  isSyncing: boolean;
  pendingCount: number;
  lastSync: Date | null;
  storageUsed: number;
  storageQuota: number;

  setOnline: (online: boolean) => void;
  setSyncing: (syncing: boolean) => void;
  updateStats: (stats: Partial<OfflineState>) => void;
}

export const useOfflineStore = create<OfflineState>((set) => ({
  isOnline: typeof navigator !== 'undefined' ? navigator.onLine : true,
  isSyncing: false,
  pendingCount: 0,
  lastSync: null,
  storageUsed: 0,
  storageQuota: 0,

  setOnline: (online) => set({ isOnline: online }),
  setSyncing: (syncing) => set({ isSyncing: syncing }),
  updateStats: (stats) => set(stats),
}));

export function useOfflineSync() {
  const { isOnline, isSyncing, pendingCount, lastSync } = useOfflineStore();
  const { setOnline, setSyncing, updateStats } = useOfflineStore();

  useEffect(() => {
    const handleOnline = () => setOnline(true);
    const handleOffline = () => setOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [setOnline]);

  useEffect(() => {
    const loadStats = async () => {
      const stats = await offlineStorage.getStorageStats();
      updateStats({
        pendingCount: stats.pendingOperations,
        lastSync: stats.lastSync,
        storageUsed: stats.usedBytes,
        storageQuota: stats.quotaBytes,
      });
    };

    loadStats();
    const interval = setInterval(loadStats, 30000);
    return () => clearInterval(interval);
  }, [updateStats]);

  const syncNow = useCallback(async () => {
    if (!isOnline || isSyncing) return;

    setSyncing(true);
    try {
      await offlineStorage.executePendingSync(async (item) => {
        // In production, this would call actual API
        console.log('Syncing item:', item);
        return true;
      });

      const stats = await offlineStorage.getStorageStats();
      updateStats({
        pendingCount: stats.pendingOperations,
        lastSync: stats.lastSync,
      });
    } finally {
      setSyncing(false);
    }
  }, [isOnline, isSyncing, setSyncing, updateStats]);

  return {
    isOnline,
    isSyncing,
    pendingCount,
    lastSync,
    syncNow,
    hasPendingChanges: pendingCount > 0,
  };
}

export function useOfflineEmail(emailId: string) {
  const [email, setEmail] = useState<StoredEmail | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      const stored = await offlineStorage.getEmail(emailId);
      setEmail(stored || null);
      setLoading(false);
    };
    load();
  }, [emailId]);

  const update = useCallback(
    async (updates: Partial<StoredEmail>) => {
      if (!email) return;
      const updated = { ...email, ...updates };
      await offlineStorage.saveEmail(updated);
      setEmail(updated);
    },
    [email]
  );

  return { email, loading, update };
}

export function useOfflineEmails(folderId?: string) {
  const [emails, setEmails] = useState<StoredEmail[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      const stored = await offlineStorage.getEmails(folderId);
      setEmails(stored.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()));
      setLoading(false);
    };
    load();
  }, [folderId]);

  return { emails, loading };
}

export function useStorageQuota() {
  const { storageUsed, storageQuota } = useOfflineStore();

  const percentUsed = storageQuota > 0 ? (storageUsed / storageQuota) * 100 : 0;
  const isLow = percentUsed > 90;

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
  };

  return {
    used: formatBytes(storageUsed),
    quota: formatBytes(storageQuota),
    percentUsed,
    isLow,
  };
}
