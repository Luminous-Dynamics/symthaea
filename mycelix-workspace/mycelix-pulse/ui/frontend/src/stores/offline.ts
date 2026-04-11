// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Offline Store
 *
 * IndexedDB-backed storage for offline email access
 */

const DB_NAME = 'mycelix-mail-offline';
const DB_VERSION = 1;

interface Email {
  id: string;
  subject: string;
  fromAddress: string;
  fromName?: string;
  bodyText: string;
  bodyHtml?: string;
  receivedAt: string;
  folder: string;
  isRead: boolean;
  hasAttachments: boolean;
  syncedAt: number;
}

interface Draft {
  id: string;
  to: string[];
  cc: string[];
  bcc: string[];
  subject: string;
  body: string;
  attachmentIds: string[];
  createdAt: number;
  updatedAt: number;
  synced: boolean;
}

interface OfflineAction {
  id: string;
  type: 'send' | 'archive' | 'delete' | 'read' | 'star' | 'move';
  emailId: string;
  payload?: any;
  createdAt: number;
}

class OfflineStore {
  private db: IDBDatabase | null = null;

  async init(): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        resolve();
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;

        // Emails store
        if (!db.objectStoreNames.contains('emails')) {
          const emailStore = db.createObjectStore('emails', { keyPath: 'id' });
          emailStore.createIndex('folder', 'folder', { unique: false });
          emailStore.createIndex('fromAddress', 'fromAddress', { unique: false });
          emailStore.createIndex('receivedAt', 'receivedAt', { unique: false });
          emailStore.createIndex('syncedAt', 'syncedAt', { unique: false });
        }

        // Drafts store
        if (!db.objectStoreNames.contains('drafts')) {
          const draftStore = db.createObjectStore('drafts', { keyPath: 'id' });
          draftStore.createIndex('updatedAt', 'updatedAt', { unique: false });
          draftStore.createIndex('synced', 'synced', { unique: false });
        }

        // Offline actions queue
        if (!db.objectStoreNames.contains('actions')) {
          const actionStore = db.createObjectStore('actions', { keyPath: 'id' });
          actionStore.createIndex('createdAt', 'createdAt', { unique: false });
        }

        // Contacts cache
        if (!db.objectStoreNames.contains('contacts')) {
          const contactStore = db.createObjectStore('contacts', { keyPath: 'id' });
          contactStore.createIndex('email', 'email', { unique: true });
          contactStore.createIndex('name', 'name', { unique: false });
        }

        // Search index
        if (!db.objectStoreNames.contains('searchIndex')) {
          db.createObjectStore('searchIndex', { keyPath: 'emailId' });
        }
      };
    });
  }

  // Email operations
  async saveEmail(email: Email): Promise<void> {
    const tx = this.db!.transaction('emails', 'readwrite');
    const store = tx.objectStore('emails');
    await this.promisifyRequest(store.put({ ...email, syncedAt: Date.now() }));
  }

  async saveEmails(emails: Email[]): Promise<void> {
    const tx = this.db!.transaction('emails', 'readwrite');
    const store = tx.objectStore('emails');
    const now = Date.now();

    for (const email of emails) {
      store.put({ ...email, syncedAt: now });
    }

    await this.promisifyTransaction(tx);
  }

  async getEmail(id: string): Promise<Email | undefined> {
    const tx = this.db!.transaction('emails', 'readonly');
    const store = tx.objectStore('emails');
    return this.promisifyRequest(store.get(id));
  }

  async getEmailsByFolder(folder: string): Promise<Email[]> {
    const tx = this.db!.transaction('emails', 'readonly');
    const store = tx.objectStore('emails');
    const index = store.index('folder');
    return this.promisifyRequest(index.getAll(folder));
  }

  async searchEmails(query: string): Promise<Email[]> {
    const tx = this.db!.transaction('emails', 'readonly');
    const store = tx.objectStore('emails');
    const allEmails: Email[] = await this.promisifyRequest(store.getAll());

    const lowerQuery = query.toLowerCase();
    return allEmails.filter(
      (email) =>
        email.subject.toLowerCase().includes(lowerQuery) ||
        email.fromAddress.toLowerCase().includes(lowerQuery) ||
        email.fromName?.toLowerCase().includes(lowerQuery) ||
        email.bodyText.toLowerCase().includes(lowerQuery)
    );
  }

  async deleteEmail(id: string): Promise<void> {
    const tx = this.db!.transaction('emails', 'readwrite');
    const store = tx.objectStore('emails');
    await this.promisifyRequest(store.delete(id));
  }

  async pruneOldEmails(maxAge: number = 30 * 24 * 60 * 60 * 1000): Promise<void> {
    const tx = this.db!.transaction('emails', 'readwrite');
    const store = tx.objectStore('emails');
    const index = store.index('syncedAt');
    const cutoff = Date.now() - maxAge;

    const range = IDBKeyRange.upperBound(cutoff);
    const cursor = index.openCursor(range);

    return new Promise((resolve, reject) => {
      cursor.onsuccess = (event) => {
        const cur = (event.target as IDBRequest).result;
        if (cur) {
          cur.delete();
          cur.continue();
        } else {
          resolve();
        }
      };
      cursor.onerror = () => reject(cursor.error);
    });
  }

  // Draft operations
  async saveDraft(draft: Draft): Promise<void> {
    const tx = this.db!.transaction('drafts', 'readwrite');
    const store = tx.objectStore('drafts');
    await this.promisifyRequest(
      store.put({ ...draft, updatedAt: Date.now(), synced: false })
    );
  }

  async getDraft(id: string): Promise<Draft | undefined> {
    const tx = this.db!.transaction('drafts', 'readonly');
    const store = tx.objectStore('drafts');
    return this.promisifyRequest(store.get(id));
  }

  async getAllDrafts(): Promise<Draft[]> {
    const tx = this.db!.transaction('drafts', 'readonly');
    const store = tx.objectStore('drafts');
    return this.promisifyRequest(store.getAll());
  }

  async getUnsyncedDrafts(): Promise<Draft[]> {
    const tx = this.db!.transaction('drafts', 'readonly');
    const store = tx.objectStore('drafts');
    const index = store.index('synced');
    return this.promisifyRequest(index.getAll(false));
  }

  async deleteDraft(id: string): Promise<void> {
    const tx = this.db!.transaction('drafts', 'readwrite');
    const store = tx.objectStore('drafts');
    await this.promisifyRequest(store.delete(id));
  }

  async markDraftSynced(id: string): Promise<void> {
    const tx = this.db!.transaction('drafts', 'readwrite');
    const store = tx.objectStore('drafts');
    const draft = await this.promisifyRequest(store.get(id));
    if (draft) {
      draft.synced = true;
      await this.promisifyRequest(store.put(draft));
    }
  }

  // Offline action queue
  async queueAction(action: Omit<OfflineAction, 'id' | 'createdAt'>): Promise<string> {
    const tx = this.db!.transaction('actions', 'readwrite');
    const store = tx.objectStore('actions');

    const fullAction: OfflineAction = {
      ...action,
      id: `action-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      createdAt: Date.now(),
    };

    await this.promisifyRequest(store.add(fullAction));
    return fullAction.id;
  }

  async getPendingActions(): Promise<OfflineAction[]> {
    const tx = this.db!.transaction('actions', 'readonly');
    const store = tx.objectStore('actions');
    const index = store.index('createdAt');
    return this.promisifyRequest(index.getAll());
  }

  async removeAction(id: string): Promise<void> {
    const tx = this.db!.transaction('actions', 'readwrite');
    const store = tx.objectStore('actions');
    await this.promisifyRequest(store.delete(id));
  }

  async clearActions(): Promise<void> {
    const tx = this.db!.transaction('actions', 'readwrite');
    const store = tx.objectStore('actions');
    await this.promisifyRequest(store.clear());
  }

  // Sync status
  async getLastSyncTime(): Promise<number> {
    const tx = this.db!.transaction('emails', 'readonly');
    const store = tx.objectStore('emails');
    const index = store.index('syncedAt');

    return new Promise((resolve) => {
      const cursor = index.openCursor(null, 'prev');
      cursor.onsuccess = (event) => {
        const cur = (event.target as IDBRequest).result;
        resolve(cur ? cur.value.syncedAt : 0);
      };
      cursor.onerror = () => resolve(0);
    });
  }

  async getStorageUsage(): Promise<{ emails: number; drafts: number; actions: number }> {
    const emailsTx = this.db!.transaction('emails', 'readonly');
    const draftsTx = this.db!.transaction('drafts', 'readonly');
    const actionsTx = this.db!.transaction('actions', 'readonly');

    const [emails, drafts, actions] = await Promise.all([
      this.promisifyRequest(emailsTx.objectStore('emails').count()),
      this.promisifyRequest(draftsTx.objectStore('drafts').count()),
      this.promisifyRequest(actionsTx.objectStore('actions').count()),
    ]);

    return { emails, drafts, actions };
  }

  async clearAll(): Promise<void> {
    const tx = this.db!.transaction(
      ['emails', 'drafts', 'actions', 'contacts', 'searchIndex'],
      'readwrite'
    );

    tx.objectStore('emails').clear();
    tx.objectStore('drafts').clear();
    tx.objectStore('actions').clear();
    tx.objectStore('contacts').clear();
    tx.objectStore('searchIndex').clear();

    await this.promisifyTransaction(tx);
  }

  // Helper methods
  private promisifyRequest<T>(request: IDBRequest<T>): Promise<T> {
    return new Promise((resolve, reject) => {
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  private promisifyTransaction(tx: IDBTransaction): Promise<void> {
    return new Promise((resolve, reject) => {
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error);
    });
  }
}

// Singleton instance
export const offlineStore = new OfflineStore();

// Sync manager
export class SyncManager {
  private isOnline = navigator.onLine;
  private syncInProgress = false;
  private listeners: ((online: boolean) => void)[] = [];

  constructor() {
    window.addEventListener('online', () => this.handleOnline());
    window.addEventListener('offline', () => this.handleOffline());
  }

  private handleOnline(): void {
    this.isOnline = true;
    this.notifyListeners();
    this.sync();
  }

  private handleOffline(): void {
    this.isOnline = false;
    this.notifyListeners();
  }

  private notifyListeners(): void {
    this.listeners.forEach((fn) => fn(this.isOnline));
  }

  onStatusChange(callback: (online: boolean) => void): () => void {
    this.listeners.push(callback);
    return () => {
      this.listeners = this.listeners.filter((fn) => fn !== callback);
    };
  }

  getStatus(): boolean {
    return this.isOnline;
  }

  async sync(): Promise<void> {
    if (!this.isOnline || this.syncInProgress) return;

    this.syncInProgress = true;

    try {
      // Sync pending actions
      const actions = await offlineStore.getPendingActions();

      for (const action of actions) {
        try {
          await this.processAction(action);
          await offlineStore.removeAction(action.id);
        } catch (error) {
          console.error('Failed to sync action:', action.id, error);
        }
      }

      // Sync drafts
      const unsyncedDrafts = await offlineStore.getUnsyncedDrafts();

      for (const draft of unsyncedDrafts) {
        try {
          await fetch('/api/drafts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(draft),
          });
          await offlineStore.markDraftSynced(draft.id);
        } catch (error) {
          console.error('Failed to sync draft:', draft.id, error);
        }
      }

      // Fetch new emails
      const lastSync = await offlineStore.getLastSyncTime();
      const response = await fetch(`/api/emails?since=${lastSync}`);

      if (response.ok) {
        const newEmails = await response.json();
        await offlineStore.saveEmails(newEmails);
      }
    } finally {
      this.syncInProgress = false;
    }
  }

  private async processAction(action: OfflineAction): Promise<void> {
    switch (action.type) {
      case 'send':
        await fetch('/api/emails/send', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(action.payload),
        });
        break;
      case 'archive':
        await fetch(`/api/emails/${action.emailId}/archive`, { method: 'POST' });
        break;
      case 'delete':
        await fetch(`/api/emails/${action.emailId}`, { method: 'DELETE' });
        break;
      case 'read':
        await fetch(`/api/emails/${action.emailId}/read`, { method: 'POST' });
        break;
      case 'star':
        await fetch(`/api/emails/${action.emailId}/star`, { method: 'POST' });
        break;
      case 'move':
        await fetch(`/api/emails/${action.emailId}/move`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ folder: action.payload.folder }),
        });
        break;
    }
  }
}

export const syncManager = new SyncManager();

// React hook for offline status
export function useOfflineStatus(): { isOnline: boolean; pendingActions: number } {
  const [isOnline, setIsOnline] = useState(syncManager.getStatus());
  const [pendingActions, setPendingActions] = useState(0);

  useEffect(() => {
    const unsubscribe = syncManager.onStatusChange(setIsOnline);

    const checkPending = async () => {
      const actions = await offlineStore.getPendingActions();
      setPendingActions(actions.length);
    };

    checkPending();
    const interval = setInterval(checkPending, 5000);

    return () => {
      unsubscribe();
      clearInterval(interval);
    };
  }, []);

  return { isOnline, pendingActions };
}

import { useState, useEffect } from 'react';
