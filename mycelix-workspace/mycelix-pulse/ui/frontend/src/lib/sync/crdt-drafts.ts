// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * CRDT-based Draft Sync
 *
 * Conflict-free synchronization of email drafts across devices
 * Uses Yjs for CRDT implementation
 */

import * as Y from 'yjs';
import { WebsocketProvider } from 'y-websocket';
import { IndexeddbPersistence } from 'y-indexeddb';

// Types
export interface DraftContent {
  to: string[];
  cc: string[];
  bcc: string[];
  subject: string;
  bodyText: string;
  bodyHtml: string;
  isEncrypted: boolean;
  attachmentIds: string[];
  replyToId?: string;
}

export interface DraftMetadata {
  id: string;
  createdAt: number;
  updatedAt: number;
  syncedAt?: number;
  version: number;
}

export interface SyncStatus {
  isOnline: boolean;
  isSyncing: boolean;
  lastSyncedAt?: Date;
  pendingChanges: number;
}

// Draft document structure
interface DraftYDoc {
  content: Y.Map<any>;
  metadata: Y.Map<any>;
  history: Y.Array<any>;
}

/**
 * CRDT Draft Manager
 * Handles conflict-free synchronization of email drafts
 */
export class CRDTDraftManager {
  private docs: Map<string, Y.Doc> = new Map();
  private providers: Map<string, WebsocketProvider> = new Map();
  private persistence: Map<string, IndexeddbPersistence> = new Map();
  private wsUrl: string;
  private userId: string;
  private onStatusChange?: (status: SyncStatus) => void;

  constructor(wsUrl: string, userId: string) {
    this.wsUrl = wsUrl;
    this.userId = userId;
  }

  /**
   * Open or create a draft
   */
  async openDraft(draftId: string): Promise<DraftDocument> {
    if (this.docs.has(draftId)) {
      return new DraftDocument(this.docs.get(draftId)!, draftId);
    }

    const doc = new Y.Doc();
    const roomName = `draft:${this.userId}:${draftId}`;

    // Set up IndexedDB persistence for offline support
    const indexeddbProvider = new IndexeddbPersistence(roomName, doc);
    await indexeddbProvider.whenSynced;

    // Set up WebSocket sync for real-time collaboration
    const wsProvider = new WebsocketProvider(this.wsUrl, roomName, doc, {
      connect: true,
      params: { userId: this.userId },
    });

    // Track sync status
    wsProvider.on('status', (event: { status: string }) => {
      this.updateSyncStatus();
    });

    wsProvider.on('sync', (isSynced: boolean) => {
      if (isSynced) {
        this.updateSyncStatus();
      }
    });

    this.docs.set(draftId, doc);
    this.providers.set(draftId, wsProvider);
    this.persistence.set(draftId, indexeddbProvider);

    return new DraftDocument(doc, draftId);
  }

  /**
   * Create a new draft
   */
  async createDraft(initialContent?: Partial<DraftContent>): Promise<DraftDocument> {
    const draftId = crypto.randomUUID();
    const draft = await this.openDraft(draftId);

    // Set initial content
    if (initialContent) {
      draft.setContent(initialContent);
    }

    // Set metadata
    draft.setMetadata({
      id: draftId,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      version: 1,
    });

    return draft;
  }

  /**
   * Delete a draft
   */
  async deleteDraft(draftId: string): Promise<void> {
    const provider = this.providers.get(draftId);
    if (provider) {
      provider.disconnect();
      this.providers.delete(draftId);
    }

    const persistence = this.persistence.get(draftId);
    if (persistence) {
      await persistence.destroy();
      this.persistence.delete(draftId);
    }

    const doc = this.docs.get(draftId);
    if (doc) {
      doc.destroy();
      this.docs.delete(draftId);
    }

    // Remove from IndexedDB
    const dbName = `draft:${this.userId}:${draftId}`;
    indexedDB.deleteDatabase(dbName);
  }

  /**
   * List all drafts
   */
  async listDrafts(): Promise<DraftMetadata[]> {
    const drafts: DraftMetadata[] = [];

    // Get all draft databases from IndexedDB
    const databases = await indexedDB.databases();
    const draftDbs = databases.filter(
      (db) => db.name?.startsWith(`draft:${this.userId}:`)
    );

    for (const db of draftDbs) {
      const draftId = db.name!.split(':')[2];
      const draft = await this.openDraft(draftId);
      const metadata = draft.getMetadata();
      if (metadata) {
        drafts.push(metadata);
      }
    }

    // Sort by updatedAt descending
    drafts.sort((a, b) => b.updatedAt - a.updatedAt);

    return drafts;
  }

  /**
   * Get sync status
   */
  getSyncStatus(): SyncStatus {
    let isOnline = false;
    let isSyncing = false;
    let lastSyncedAt: Date | undefined;
    let pendingChanges = 0;

    for (const [draftId, provider] of this.providers) {
      if (provider.wsconnected) {
        isOnline = true;
      }
      if (provider.synced) {
        const doc = this.docs.get(draftId);
        if (doc) {
          // Check for unsynced changes
          const state = Y.encodeStateAsUpdate(doc);
          pendingChanges += state.length;
        }
      }
    }

    return {
      isOnline,
      isSyncing,
      lastSyncedAt,
      pendingChanges,
    };
  }

  /**
   * Register status change callback
   */
  onSyncStatusChange(callback: (status: SyncStatus) => void): void {
    this.onStatusChange = callback;
  }

  private updateSyncStatus(): void {
    if (this.onStatusChange) {
      this.onStatusChange(this.getSyncStatus());
    }
  }

  /**
   * Disconnect all providers
   */
  disconnect(): void {
    for (const provider of this.providers.values()) {
      provider.disconnect();
    }
  }

  /**
   * Reconnect all providers
   */
  reconnect(): void {
    for (const provider of this.providers.values()) {
      provider.connect();
    }
  }
}

/**
 * Individual draft document
 */
export class DraftDocument {
  private doc: Y.Doc;
  private content: Y.Map<any>;
  private metadata: Y.Map<any>;
  private history: Y.Array<any>;
  public readonly id: string;

  constructor(doc: Y.Doc, id: string) {
    this.doc = doc;
    this.id = id;

    // Get or create shared types
    this.content = doc.getMap('content');
    this.metadata = doc.getMap('metadata');
    this.history = doc.getArray('history');
  }

  /**
   * Get current content
   */
  getContent(): DraftContent {
    return {
      to: this.content.get('to') || [],
      cc: this.content.get('cc') || [],
      bcc: this.content.get('bcc') || [],
      subject: this.content.get('subject') || '',
      bodyText: this.content.get('bodyText') || '',
      bodyHtml: this.content.get('bodyHtml') || '',
      isEncrypted: this.content.get('isEncrypted') || false,
      attachmentIds: this.content.get('attachmentIds') || [],
      replyToId: this.content.get('replyToId'),
    };
  }

  /**
   * Set content (partial update)
   */
  setContent(content: Partial<DraftContent>): void {
    this.doc.transact(() => {
      for (const [key, value] of Object.entries(content)) {
        if (value !== undefined) {
          this.content.set(key, value);
        }
      }
      this.metadata.set('updatedAt', Date.now());
      this.metadata.set('version', (this.metadata.get('version') || 0) + 1);
    });

    // Record in history
    this.recordChange('update', content);
  }

  /**
   * Get metadata
   */
  getMetadata(): DraftMetadata | null {
    const id = this.metadata.get('id');
    if (!id) return null;

    return {
      id,
      createdAt: this.metadata.get('createdAt') || Date.now(),
      updatedAt: this.metadata.get('updatedAt') || Date.now(),
      syncedAt: this.metadata.get('syncedAt'),
      version: this.metadata.get('version') || 1,
    };
  }

  /**
   * Set metadata
   */
  setMetadata(metadata: DraftMetadata): void {
    this.doc.transact(() => {
      for (const [key, value] of Object.entries(metadata)) {
        this.metadata.set(key, value);
      }
    });
  }

  /**
   * Update subject with collaborative text
   */
  updateSubject(subject: string): void {
    this.content.set('subject', subject);
    this.metadata.set('updatedAt', Date.now());
  }

  /**
   * Update body with collaborative text
   */
  updateBody(bodyText: string, bodyHtml?: string): void {
    this.doc.transact(() => {
      this.content.set('bodyText', bodyText);
      if (bodyHtml !== undefined) {
        this.content.set('bodyHtml', bodyHtml);
      }
      this.metadata.set('updatedAt', Date.now());
    });
  }

  /**
   * Add recipient
   */
  addRecipient(field: 'to' | 'cc' | 'bcc', email: string): void {
    const recipients: string[] = this.content.get(field) || [];
    if (!recipients.includes(email)) {
      this.content.set(field, [...recipients, email]);
      this.metadata.set('updatedAt', Date.now());
    }
  }

  /**
   * Remove recipient
   */
  removeRecipient(field: 'to' | 'cc' | 'bcc', email: string): void {
    const recipients: string[] = this.content.get(field) || [];
    this.content.set(
      field,
      recipients.filter((r) => r !== email)
    );
    this.metadata.set('updatedAt', Date.now());
  }

  /**
   * Toggle encryption
   */
  setEncrypted(isEncrypted: boolean): void {
    this.content.set('isEncrypted', isEncrypted);
    this.metadata.set('updatedAt', Date.now());
  }

  /**
   * Add attachment
   */
  addAttachment(attachmentId: string): void {
    const attachments: string[] = this.content.get('attachmentIds') || [];
    if (!attachments.includes(attachmentId)) {
      this.content.set('attachmentIds', [...attachments, attachmentId]);
      this.metadata.set('updatedAt', Date.now());
    }
  }

  /**
   * Remove attachment
   */
  removeAttachment(attachmentId: string): void {
    const attachments: string[] = this.content.get('attachmentIds') || [];
    this.content.set(
      'attachmentIds',
      attachments.filter((a) => a !== attachmentId)
    );
    this.metadata.set('updatedAt', Date.now());
  }

  /**
   * Subscribe to content changes
   */
  onContentChange(callback: (content: DraftContent) => void): () => void {
    const observer = () => {
      callback(this.getContent());
    };

    this.content.observe(observer);
    return () => this.content.unobserve(observer);
  }

  /**
   * Get change history
   */
  getHistory(): Array<{ timestamp: number; type: string; changes: any }> {
    return this.history.toArray();
  }

  /**
   * Record change in history
   */
  private recordChange(type: string, changes: any): void {
    this.history.push([
      {
        timestamp: Date.now(),
        type,
        changes,
      },
    ]);

    // Keep only last 100 changes
    while (this.history.length > 100) {
      this.history.delete(0, 1);
    }
  }

  /**
   * Get underlying Y.Doc for advanced operations
   */
  getYDoc(): Y.Doc {
    return this.doc;
  }
}

// React hook for CRDT drafts
export function useCRDTDraft(draftId: string | null, manager: CRDTDraftManager) {
  const [draft, setDraft] = React.useState<DraftDocument | null>(null);
  const [content, setContent] = React.useState<DraftContent | null>(null);
  const [status, setStatus] = React.useState<SyncStatus>({
    isOnline: false,
    isSyncing: false,
    pendingChanges: 0,
  });

  React.useEffect(() => {
    if (!draftId) {
      setDraft(null);
      setContent(null);
      return;
    }

    let unsubscribe: (() => void) | undefined;

    manager.openDraft(draftId).then((d) => {
      setDraft(d);
      setContent(d.getContent());

      unsubscribe = d.onContentChange((newContent) => {
        setContent(newContent);
      });
    });

    manager.onSyncStatusChange(setStatus);

    return () => {
      if (unsubscribe) {
        unsubscribe();
      }
    };
  }, [draftId, manager]);

  return { draft, content, status };
}

// Need to import React for the hook
import React from 'react';
