// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Svelte Stores for Mycelix Mail Holochain
 *
 * Reactive stores with automatic updates from Holochain signals:
 * - inbox, sent, drafts
 * - trustScores
 * - syncState
 * - notifications
 */

import { writable, derived, readable, get } from 'svelte/store';
import type { Writable, Readable, Derived } from 'svelte/store';
import type { AgentPubKey, ActionHash } from '@holochain/client';
import type { MycelixMailClient } from '../index';
import type {
  DecryptedEmail,
  TrustScore,
  TrustCategory,
  SyncState,
  MycelixSignal,
  EmailFolder,
  EmailThread,
} from '../types';

// ==================== TYPES ====================

export interface StoreConfig {
  autoRefresh?: boolean;
  refreshInterval?: number;
  cacheResults?: boolean;
}

export interface EmailStore extends Readable<DecryptedEmail[]> {
  refresh: () => Promise<void>;
  loadMore: () => Promise<void>;
  markAsRead: (hash: ActionHash) => Promise<void>;
  archive: (hash: ActionHash) => Promise<void>;
  delete: (hash: ActionHash) => Promise<void>;
  isLoading: Readable<boolean>;
  error: Readable<Error | null>;
  hasMore: Readable<boolean>;
}

export interface TrustStore extends Readable<Map<string, TrustScore>> {
  getScore: (agent: AgentPubKey, category?: TrustCategory) => Promise<TrustScore | null>;
  refresh: (agent: AgentPubKey) => Promise<void>;
  isLoading: Readable<boolean>;
}

// ==================== CLIENT STORE ====================

let clientInstance: MycelixMailClient | null = null;
const clientStore = writable<MycelixMailClient | null>(null);

/**
 * Initialize the Mycelix client store
 */
export function initMycelixStores(client: MycelixMailClient): void {
  clientInstance = client;
  clientStore.set(client);

  // Setup signal listeners
  setupSignalListeners(client);
}

/**
 * Get the client store
 */
export const mycelixClient: Readable<MycelixMailClient | null> = {
  subscribe: clientStore.subscribe,
};

// ==================== CONNECTION STATE ====================

export const isConnected = writable(false);
export const isInitialized = writable(false);
export const myAgentPubKey = writable<AgentPubKey | null>(null);
export const connectionError = writable<Error | null>(null);

// ==================== EMAIL STORES ====================

function createEmailStore(
  fetchFn: (client: MycelixMailClient, limit: number) => Promise<DecryptedEmail[]>,
  signalTypes: string[] = ['EmailReceived', 'EmailStateChanged']
): EmailStore {
  const emails = writable<DecryptedEmail[]>([]);
  const loading = writable(true);
  const error = writable<Error | null>(null);
  const hasMore = writable(true);
  const limit = 50;

  async function refresh(): Promise<void> {
    const client = clientInstance;
    if (!client) return;

    loading.set(true);
    error.set(null);

    try {
      const result = await fetchFn(client, limit);
      emails.set(result);
      hasMore.set(result.length >= limit);
    } catch (e) {
      error.set(e as Error);
    } finally {
      loading.set(false);
    }
  }

  async function loadMore(): Promise<void> {
    const client = clientInstance;
    if (!client || !get(hasMore) || get(loading)) return;

    try {
      const current = get(emails);
      const more = await fetchFn(client, limit);
      emails.set([...current, ...more]);
      hasMore.set(more.length >= limit);
    } catch (e) {
      error.set(e as Error);
    }
  }

  async function markAsRead(hash: ActionHash): Promise<void> {
    const client = clientInstance;
    if (!client) return;

    await client.messages.markAsRead(hash);
    emails.update((list) =>
      list.map((e) => (e.hash === hash ? { ...e, state: 'Read' as const } : e))
    );
  }

  async function archive(hash: ActionHash): Promise<void> {
    const client = clientInstance;
    if (!client) return;

    await client.messages.archiveEmail(hash);
    emails.update((list) => list.filter((e) => e.hash !== hash));
  }

  async function deleteEmail(hash: ActionHash): Promise<void> {
    const client = clientInstance;
    if (!client) return;

    await client.messages.deleteEmail(hash);
    emails.update((list) => list.filter((e) => e.hash !== hash));
  }

  // Initial fetch
  if (clientInstance) {
    refresh();
  }

  return {
    subscribe: emails.subscribe,
    refresh,
    loadMore,
    markAsRead,
    archive,
    delete: deleteEmail,
    isLoading: { subscribe: loading.subscribe },
    error: { subscribe: error.subscribe },
    hasMore: { subscribe: hasMore.subscribe },
  };
}

export const inbox = createEmailStore(
  (client, limit) => client.messages.getInbox(limit),
  ['EmailReceived', 'EmailStateChanged']
);

export const sent = createEmailStore(
  (client, limit) => client.messages.getSent(limit),
  ['DeliveryConfirmed']
);

export const drafts = createEmailStore(
  (client, _limit) => client.messages.getDrafts(),
  ['DraftCreated', 'DraftUpdated']
);

// ==================== SINGLE EMAIL STORE ====================

export function createEmailDetailStore(hash: ActionHash): Readable<{
  email: DecryptedEmail | null;
  isLoading: boolean;
  error: Error | null;
}> {
  return readable({ email: null, isLoading: true, error: null }, (set) => {
    const client = clientInstance;
    if (!client) {
      set({ email: null, isLoading: false, error: new Error('Client not initialized') });
      return;
    }

    client.messages.getEmail(hash)
      .then((email) => set({ email, isLoading: false, error: null }))
      .catch((error) => set({ email: null, isLoading: false, error }));
  });
}

// ==================== TRUST STORES ====================

const trustScores = writable<Map<string, TrustScore>>(new Map());
const trustLoading = writable(false);

export const trust: TrustStore = {
  subscribe: trustScores.subscribe,

  async getScore(agent: AgentPubKey, category?: TrustCategory): Promise<TrustScore | null> {
    const client = clientInstance;
    if (!client) return null;

    const key = `${agent}:${category ?? 'all'}`;
    const cached = get(trustScores).get(key);
    if (cached) return cached;

    trustLoading.set(true);
    try {
      const score = await client.trust.getTrustScore({
        subject: agent,
        category,
        include_transitive: true,
      });

      trustScores.update((map) => {
        map.set(key, score);
        return new Map(map);
      });

      return score;
    } finally {
      trustLoading.set(false);
    }
  },

  async refresh(agent: AgentPubKey): Promise<void> {
    const client = clientInstance;
    if (!client) return;

    // Clear cached scores for this agent
    trustScores.update((map) => {
      for (const key of map.keys()) {
        if (key.startsWith(agent.toString())) {
          map.delete(key);
        }
      }
      return new Map(map);
    });
  },

  isLoading: { subscribe: trustLoading.subscribe },
};

/**
 * Create a derived store for a specific agent's trust score
 */
export function createTrustScoreStore(
  agent: AgentPubKey,
  category?: TrustCategory
): Readable<{ score: TrustScore | null; isLoading: boolean }> {
  const loading = writable(true);
  const score = writable<TrustScore | null>(null);

  // Fetch on creation
  trust.getScore(agent, category).then((s) => {
    score.set(s);
    loading.set(false);
  });

  return derived([score, loading], ([$score, $loading]) => ({
    score: $score,
    isLoading: $loading,
  }));
}

// ==================== SYNC STORES ====================

export const syncState = writable<SyncState | null>(null);
export const isOnline = writable(navigator.onLine);
export const pendingOperations = writable(0);
export const lastSyncTime = writable<Date | null>(null);

export const syncStatus = derived(
  [syncState, isOnline, pendingOperations],
  ([$syncState, $isOnline, $pendingOps]) => ({
    isOnline: $isOnline && ($syncState?.is_online ?? false),
    pendingOps: $pendingOps,
    lastSync: $syncState?.last_sync ? new Date(Number($syncState.last_sync) / 1000) : null,
  })
);

/**
 * Trigger a manual sync
 */
export async function triggerSync(): Promise<void> {
  const client = clientInstance;
  if (!client) return;

  try {
    await client.syncWithAllPeers();
    lastSyncTime.set(new Date());
  } catch (e) {
    console.error('Sync failed:', e);
  }
}

// ==================== FOLDER STORES ====================

export const folders = writable<EmailFolder[]>([]);
export const foldersLoading = writable(true);

export async function refreshFolders(): Promise<void> {
  const client = clientInstance;
  if (!client) return;

  foldersLoading.set(true);
  try {
    const result = await client.messages.getFolders();
    folders.set(result);
  } finally {
    foldersLoading.set(false);
  }
}

// ==================== THREAD STORES ====================

export const threads = writable<EmailThread[]>([]);
export const threadsLoading = writable(true);

export async function refreshThreads(limit?: number): Promise<void> {
  const client = clientInstance;
  if (!client) return;

  threadsLoading.set(true);
  try {
    const result = await client.messages.getThreads(limit);
    threads.set(result);
  } finally {
    threadsLoading.set(false);
  }
}

// ==================== SEARCH STORE ====================

export const searchQuery = writable('');
export const searchResults = writable<DecryptedEmail[]>([]);
export const searchLoading = writable(false);

let searchDebounceTimer: NodeJS.Timeout | null = null;

searchQuery.subscribe((query) => {
  if (searchDebounceTimer) {
    clearTimeout(searchDebounceTimer);
  }

  if (!query.trim()) {
    searchResults.set([]);
    return;
  }

  searchDebounceTimer = setTimeout(async () => {
    const client = clientInstance;
    if (!client) return;

    searchLoading.set(true);
    try {
      const results = await client.messages.searchEmails(query);
      searchResults.set(results);
    } finally {
      searchLoading.set(false);
    }
  }, 300);
});

// ==================== NOTIFICATION STORE ====================

export interface Notification {
  id: string;
  type: 'email' | 'trust' | 'sync' | 'system';
  title: string;
  message: string;
  timestamp: Date;
  read: boolean;
  data?: unknown;
}

export const notifications = writable<Notification[]>([]);
export const unreadCount = derived(notifications, ($notifications) =>
  $notifications.filter((n) => !n.read).length
);

export function addNotification(
  notification: Omit<Notification, 'id' | 'timestamp' | 'read'>
): void {
  notifications.update((list) => [
    {
      ...notification,
      id: crypto.randomUUID(),
      timestamp: new Date(),
      read: false,
    },
    ...list.slice(0, 99), // Keep last 100
  ]);
}

export function markNotificationRead(id: string): void {
  notifications.update((list) =>
    list.map((n) => (n.id === id ? { ...n, read: true } : n))
  );
}

export function clearNotifications(): void {
  notifications.set([]);
}

// ==================== SIGNAL LISTENERS ====================

function setupSignalListeners(client: MycelixMailClient): void {
  // Message signals
  client.signals.onMessage((signal) => {
    switch (signal.type) {
      case 'EmailReceived':
        inbox.refresh();
        addNotification({
          type: 'email',
          title: 'New Email',
          message: signal.preview,
          data: { hash: signal.email_hash, sender: signal.sender },
        });
        break;

      case 'EmailStateChanged':
        inbox.refresh();
        break;

      case 'DeliveryConfirmed':
        sent.refresh();
        break;

      case 'ReadReceiptReceived':
        addNotification({
          type: 'email',
          title: 'Read Receipt',
          message: 'Your email was read',
          data: { hash: signal.email_hash },
        });
        break;
    }
  });

  // Trust signals
  client.signals.onTrust((signal) => {
    switch (signal.type) {
      case 'TrustScoreChanged':
        trust.refresh(signal.subject);
        break;

      case 'ByzantineFlagRaised':
        addNotification({
          type: 'trust',
          title: 'Trust Warning',
          message: `Suspicious activity detected: ${signal.flag_type}`,
          data: signal,
        });
        break;
    }
  });

  // Sync signals
  client.signals.onSync((signal) => {
    switch (signal.type) {
      case 'Online':
        isOnline.set(true);
        break;

      case 'Offline':
        isOnline.set(false);
        break;

      case 'SyncStateChanged':
        pendingOperations.set(signal.pending_ops);
        break;

      case 'ConflictDetected':
        addNotification({
          type: 'sync',
          title: 'Sync Conflict',
          message: `Conflict detected: ${signal.conflict_type}`,
          data: signal,
        });
        break;

      case 'SyncComplete':
        lastSyncTime.set(new Date());
        // Refresh all data after sync
        inbox.refresh();
        sent.refresh();
        drafts.refresh();
        break;
    }
  });

  // Initialize connection state
  client.initialize().then(() => {
    isConnected.set(true);
    isInitialized.set(true);
    myAgentPubKey.set(client.myAgentPubKey);

    // Initial data fetch
    inbox.refresh();
    refreshFolders();
  }).catch((error) => {
    connectionError.set(error);
  });
}

// ==================== MAILBOX STATUS ====================

export const mailboxStatus = derived(
  [inbox, sent, drafts, syncStatus],
  ([$inbox, $sent, $drafts, $syncStatus]) => ({
    inboxCount: $inbox.length,
    unreadCount: $inbox.filter((e) => e.state === 'Unread').length,
    sentCount: $sent.length,
    draftCount: $drafts.length,
    ...$syncStatus,
  })
);

// ==================== EXPORTS ====================

export {
  initMycelixStores,
  mycelixClient,
  isConnected,
  isInitialized,
  myAgentPubKey,
  connectionError,
  inbox,
  sent,
  drafts,
  createEmailDetailStore,
  trust,
  createTrustScoreStore,
  syncState,
  isOnline,
  pendingOperations,
  lastSyncTime,
  syncStatus,
  triggerSync,
  folders,
  foldersLoading,
  refreshFolders,
  threads,
  threadsLoading,
  refreshThreads,
  searchQuery,
  searchResults,
  searchLoading,
  notifications,
  unreadCount,
  addNotification,
  markNotificationRead,
  clearNotifications,
  mailboxStatus,
};
