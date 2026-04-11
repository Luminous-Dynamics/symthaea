// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * IndexedDB Cache Layer
 *
 * Offline-first data persistence for emails, contacts, and drafts
 */

import { openDB, DBSchema, IDBPDatabase } from 'idb';

// Database schema
interface MycelixMailDB extends DBSchema {
  emails: {
    key: string;
    value: CachedEmail;
    indexes: {
      'by-folder': string;
      'by-thread': string;
      'by-date': string;
      'by-sender': string;
    };
  };
  contacts: {
    key: string;
    value: CachedContact;
    indexes: {
      'by-email': string;
      'by-name': string;
      'by-trust': number;
    };
  };
  drafts: {
    key: string;
    value: CachedDraft;
    indexes: {
      'by-updated': number;
    };
  };
  attachments: {
    key: string;
    value: CachedAttachment;
    indexes: {
      'by-email': string;
    };
  };
  syncState: {
    key: string;
    value: SyncState;
  };
  outbox: {
    key: string;
    value: OutboxItem;
    indexes: {
      'by-status': string;
      'by-created': number;
    };
  };
}

// Data types
interface CachedEmail {
  id: string;
  folderId: string;
  threadId?: string;
  subject: string;
  bodyText?: string;
  bodyHtml?: string;
  from: { email: string; name?: string };
  to: { email: string; name?: string }[];
  cc: { email: string; name?: string }[];
  isRead: boolean;
  isStarred: boolean;
  isEncrypted: boolean;
  trustScore?: number;
  labels: string[];
  receivedAt: string;
  cachedAt: number;
  attachmentIds: string[];
}

interface CachedContact {
  id: string;
  email: string;
  displayName?: string;
  trustScore: number;
  interactionCount: number;
  lastInteraction?: string;
  notes?: string;
  cachedAt: number;
}

interface CachedDraft {
  id: string;
  to: string[];
  cc: string[];
  subject: string;
  bodyText: string;
  bodyHtml?: string;
  isEncrypted: boolean;
  attachmentIds: string[];
  replyToId?: string;
  updatedAt: number;
}

interface CachedAttachment {
  id: string;
  emailId: string;
  filename: string;
  contentType: string;
  size: number;
  data?: ArrayBuffer; // Only for offline-cached attachments
  isEncrypted: boolean;
  cachedAt: number;
}

interface SyncState {
  key: string;
  lastSync: number;
  cursor?: string;
  version: number;
}

interface OutboxItem {
  id: string;
  email: {
    to: string[];
    cc: string[];
    bcc: string[];
    subject: string;
    bodyText?: string;
    bodyHtml?: string;
    isEncrypted: boolean;
    attachmentIds: string[];
  };
  status: 'pending' | 'sending' | 'failed';
  retries: number;
  createdAt: number;
  error?: string;
}

// Database version
const DB_NAME = 'mycelix-mail';
const DB_VERSION = 1;

// Cache TTL (time to live)
const EMAIL_TTL = 7 * 24 * 60 * 60 * 1000; // 7 days
const CONTACT_TTL = 30 * 24 * 60 * 60 * 1000; // 30 days
const ATTACHMENT_TTL = 3 * 24 * 60 * 60 * 1000; // 3 days

class IndexedDBCache {
  private db: IDBPDatabase<MycelixMailDB> | null = null;
  private initPromise: Promise<IDBPDatabase<MycelixMailDB>> | null = null;

  async init(): Promise<IDBPDatabase<MycelixMailDB>> {
    if (this.db) return this.db;
    if (this.initPromise) return this.initPromise;

    this.initPromise = openDB<MycelixMailDB>(DB_NAME, DB_VERSION, {
      upgrade(db, oldVersion, newVersion, transaction) {
        console.log(`[IDB] Upgrading from v${oldVersion} to v${newVersion}`);

        // Emails store
        if (!db.objectStoreNames.contains('emails')) {
          const emailStore = db.createObjectStore('emails', { keyPath: 'id' });
          emailStore.createIndex('by-folder', 'folderId');
          emailStore.createIndex('by-thread', 'threadId');
          emailStore.createIndex('by-date', 'receivedAt');
          emailStore.createIndex('by-sender', 'from.email');
        }

        // Contacts store
        if (!db.objectStoreNames.contains('contacts')) {
          const contactStore = db.createObjectStore('contacts', { keyPath: 'id' });
          contactStore.createIndex('by-email', 'email', { unique: true });
          contactStore.createIndex('by-name', 'displayName');
          contactStore.createIndex('by-trust', 'trustScore');
        }

        // Drafts store
        if (!db.objectStoreNames.contains('drafts')) {
          const draftStore = db.createObjectStore('drafts', { keyPath: 'id' });
          draftStore.createIndex('by-updated', 'updatedAt');
        }

        // Attachments store
        if (!db.objectStoreNames.contains('attachments')) {
          const attachmentStore = db.createObjectStore('attachments', { keyPath: 'id' });
          attachmentStore.createIndex('by-email', 'emailId');
        }

        // Sync state store
        if (!db.objectStoreNames.contains('syncState')) {
          db.createObjectStore('syncState', { keyPath: 'key' });
        }

        // Outbox store
        if (!db.objectStoreNames.contains('outbox')) {
          const outboxStore = db.createObjectStore('outbox', { keyPath: 'id' });
          outboxStore.createIndex('by-status', 'status');
          outboxStore.createIndex('by-created', 'createdAt');
        }
      },
      blocked() {
        console.warn('[IDB] Database blocked by another connection');
      },
      blocking() {
        console.warn('[IDB] This connection is blocking a version upgrade');
      },
      terminated() {
        console.error('[IDB] Database connection terminated unexpectedly');
      },
    });

    this.db = await this.initPromise;
    return this.db;
  }

  // Emails
  async getEmail(id: string): Promise<CachedEmail | undefined> {
    const db = await this.init();
    const email = await db.get('emails', id);

    if (email && this.isExpired(email.cachedAt, EMAIL_TTL)) {
      await db.delete('emails', id);
      return undefined;
    }

    return email;
  }

  async getEmailsByFolder(folderId: string, limit = 50): Promise<CachedEmail[]> {
    const db = await this.init();
    const tx = db.transaction('emails', 'readonly');
    const index = tx.store.index('by-folder');

    const emails: CachedEmail[] = [];
    let cursor = await index.openCursor(IDBKeyRange.only(folderId));

    while (cursor && emails.length < limit) {
      if (!this.isExpired(cursor.value.cachedAt, EMAIL_TTL)) {
        emails.push(cursor.value);
      }
      cursor = await cursor.continue();
    }

    // Sort by date descending
    return emails.sort((a, b) =>
      new Date(b.receivedAt).getTime() - new Date(a.receivedAt).getTime()
    );
  }

  async cacheEmail(email: CachedEmail): Promise<void> {
    const db = await this.init();
    await db.put('emails', { ...email, cachedAt: Date.now() });
  }

  async cacheEmails(emails: CachedEmail[]): Promise<void> {
    const db = await this.init();
    const tx = db.transaction('emails', 'readwrite');
    const now = Date.now();

    await Promise.all([
      ...emails.map((email) => tx.store.put({ ...email, cachedAt: now })),
      tx.done,
    ]);
  }

  async deleteEmail(id: string): Promise<void> {
    const db = await this.init();
    await db.delete('emails', id);
  }

  async searchEmails(query: string, limit = 20): Promise<CachedEmail[]> {
    const db = await this.init();
    const allEmails = await db.getAll('emails');
    const queryLower = query.toLowerCase();

    return allEmails
      .filter((email) => {
        if (this.isExpired(email.cachedAt, EMAIL_TTL)) return false;
        return (
          email.subject.toLowerCase().includes(queryLower) ||
          email.bodyText?.toLowerCase().includes(queryLower) ||
          email.from.email.toLowerCase().includes(queryLower) ||
          email.from.name?.toLowerCase().includes(queryLower)
        );
      })
      .slice(0, limit);
  }

  // Contacts
  async getContact(id: string): Promise<CachedContact | undefined> {
    const db = await this.init();
    return db.get('contacts', id);
  }

  async getContactByEmail(email: string): Promise<CachedContact | undefined> {
    const db = await this.init();
    return db.getFromIndex('contacts', 'by-email', email);
  }

  async getAllContacts(): Promise<CachedContact[]> {
    const db = await this.init();
    return db.getAll('contacts');
  }

  async cacheContact(contact: CachedContact): Promise<void> {
    const db = await this.init();
    await db.put('contacts', { ...contact, cachedAt: Date.now() });
  }

  async cacheContacts(contacts: CachedContact[]): Promise<void> {
    const db = await this.init();
    const tx = db.transaction('contacts', 'readwrite');
    const now = Date.now();

    await Promise.all([
      ...contacts.map((contact) => tx.store.put({ ...contact, cachedAt: now })),
      tx.done,
    ]);
  }

  // Drafts
  async getDraft(id: string): Promise<CachedDraft | undefined> {
    const db = await this.init();
    return db.get('drafts', id);
  }

  async getAllDrafts(): Promise<CachedDraft[]> {
    const db = await this.init();
    const drafts = await db.getAll('drafts');
    return drafts.sort((a, b) => b.updatedAt - a.updatedAt);
  }

  async saveDraft(draft: CachedDraft): Promise<void> {
    const db = await this.init();
    await db.put('drafts', { ...draft, updatedAt: Date.now() });
  }

  async deleteDraft(id: string): Promise<void> {
    const db = await this.init();
    await db.delete('drafts', id);
  }

  // Attachments
  async getAttachment(id: string): Promise<CachedAttachment | undefined> {
    const db = await this.init();
    return db.get('attachments', id);
  }

  async cacheAttachment(attachment: CachedAttachment): Promise<void> {
    const db = await this.init();
    await db.put('attachments', { ...attachment, cachedAt: Date.now() });
  }

  async getAttachmentsByEmail(emailId: string): Promise<CachedAttachment[]> {
    const db = await this.init();
    return db.getAllFromIndex('attachments', 'by-email', emailId);
  }

  // Outbox
  async addToOutbox(item: Omit<OutboxItem, 'id' | 'createdAt' | 'status' | 'retries'>): Promise<string> {
    const db = await this.init();
    const id = crypto.randomUUID();
    await db.put('outbox', {
      ...item,
      id,
      status: 'pending',
      retries: 0,
      createdAt: Date.now(),
    });
    return id;
  }

  async getOutboxItems(): Promise<OutboxItem[]> {
    const db = await this.init();
    return db.getAll('outbox');
  }

  async getPendingOutboxItems(): Promise<OutboxItem[]> {
    const db = await this.init();
    return db.getAllFromIndex('outbox', 'by-status', 'pending');
  }

  async updateOutboxItem(id: string, updates: Partial<OutboxItem>): Promise<void> {
    const db = await this.init();
    const item = await db.get('outbox', id);
    if (item) {
      await db.put('outbox', { ...item, ...updates });
    }
  }

  async removeFromOutbox(id: string): Promise<void> {
    const db = await this.init();
    await db.delete('outbox', id);
  }

  // Sync state
  async getSyncState(key: string): Promise<SyncState | undefined> {
    const db = await this.init();
    return db.get('syncState', key);
  }

  async setSyncState(state: SyncState): Promise<void> {
    const db = await this.init();
    await db.put('syncState', state);
  }

  // Cleanup
  async cleanup(): Promise<{ deleted: number }> {
    const db = await this.init();
    let deleted = 0;
    const now = Date.now();

    // Clean expired emails
    const emails = await db.getAll('emails');
    for (const email of emails) {
      if (this.isExpired(email.cachedAt, EMAIL_TTL)) {
        await db.delete('emails', email.id);
        deleted++;
      }
    }

    // Clean expired attachments
    const attachments = await db.getAll('attachments');
    for (const att of attachments) {
      if (this.isExpired(att.cachedAt, ATTACHMENT_TTL)) {
        await db.delete('attachments', att.id);
        deleted++;
      }
    }

    console.log(`[IDB] Cleanup: removed ${deleted} expired items`);
    return { deleted };
  }

  async clear(): Promise<void> {
    const db = await this.init();
    const tx = db.transaction(
      ['emails', 'contacts', 'drafts', 'attachments', 'syncState', 'outbox'],
      'readwrite'
    );

    await Promise.all([
      tx.objectStore('emails').clear(),
      tx.objectStore('contacts').clear(),
      tx.objectStore('drafts').clear(),
      tx.objectStore('attachments').clear(),
      tx.objectStore('syncState').clear(),
      tx.objectStore('outbox').clear(),
      tx.done,
    ]);
  }

  async getStats(): Promise<{
    emails: number;
    contacts: number;
    drafts: number;
    attachments: number;
    outbox: number;
  }> {
    const db = await this.init();
    return {
      emails: await db.count('emails'),
      contacts: await db.count('contacts'),
      drafts: await db.count('drafts'),
      attachments: await db.count('attachments'),
      outbox: await db.count('outbox'),
    };
  }

  private isExpired(cachedAt: number, ttl: number): boolean {
    return Date.now() - cachedAt > ttl;
  }
}

// Singleton instance
export const emailCache = new IndexedDBCache();

// React hook for cache access
export function useOfflineCache() {
  return {
    cache: emailCache,

    async cacheEmailForOffline(email: CachedEmail) {
      await emailCache.cacheEmail(email);
      // Also notify service worker
      if ('serviceWorker' in navigator && navigator.serviceWorker.controller) {
        navigator.serviceWorker.controller.postMessage({
          type: 'CACHE_EMAIL',
          emailId: email.id,
          email,
        });
      }
    },

    async saveDraftOffline(draft: CachedDraft) {
      await emailCache.saveDraft(draft);
    },

    async queueEmailForSending(email: OutboxItem['email']) {
      return emailCache.addToOutbox({ email });
    },

    async getOfflineEmails(folderId: string) {
      return emailCache.getEmailsByFolder(folderId);
    },

    async searchOffline(query: string) {
      return emailCache.searchEmails(query);
    },
  };
}

export default emailCache;
