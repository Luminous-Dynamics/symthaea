// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Vue Composable - useMessages
 *
 * Reactive email management with inbox, sent, drafts, and folders.
 */

import { ref, computed, watch, onMounted, onUnmounted } from 'vue';
import type { Ref, ComputedRef } from 'vue';
import { getMycelixClient } from '../../bootstrap';

export interface EmailMessage {
  hash: string;
  from: { address: string; name?: string };
  to: Array<{ address: string; name?: string }>;
  cc?: Array<{ address: string; name?: string }>;
  subject: string;
  body: string;
  bodyHtml?: string;
  receivedAt: number;
  sentAt?: number;
  read: boolean;
  starred: boolean;
  labels: string[];
  hasAttachments: boolean;
  encrypted?: boolean;
  threadId?: string;
  trustLevel?: number;
}

export interface UseMessagesOptions {
  folder?: 'inbox' | 'sent' | 'drafts' | 'trash' | 'spam' | 'archive';
  pageSize?: number;
  autoRefresh?: boolean;
  refreshIntervalMs?: number;
}

export interface UseMessagesReturn {
  messages: Ref<EmailMessage[]>;
  loading: Ref<boolean>;
  error: Ref<string | null>;
  hasMore: Ref<boolean>;
  unreadCount: ComputedRef<number>;
  refresh: () => Promise<void>;
  loadMore: () => Promise<void>;
  markAsRead: (hash: string) => Promise<void>;
  markAsUnread: (hash: string) => Promise<void>;
  star: (hash: string) => Promise<void>;
  unstar: (hash: string) => Promise<void>;
  archive: (hash: string) => Promise<void>;
  trash: (hash: string) => Promise<void>;
  permanentDelete: (hash: string) => Promise<void>;
  moveToFolder: (hash: string, folder: string) => Promise<void>;
  addLabel: (hash: string, label: string) => Promise<void>;
  removeLabel: (hash: string, label: string) => Promise<void>;
}

export function useMessages(options: UseMessagesOptions = {}): UseMessagesReturn {
  const {
    folder = 'inbox',
    pageSize = 50,
    autoRefresh = true,
    refreshIntervalMs = 30000,
  } = options;

  const messages = ref<EmailMessage[]>([]);
  const loading = ref(false);
  const error = ref<string | null>(null);
  const hasMore = ref(true);
  const cursor = ref<string | null>(null);

  let refreshInterval: ReturnType<typeof setInterval> | null = null;
  let unsubscribe: (() => void) | null = null;

  const unreadCount = computed(() =>
    messages.value.filter((m) => !m.read).length
  );

  async function fetchMessages(append = false) {
    loading.value = true;
    error.value = null;

    try {
      const client = getMycelixClient();
      const services = client.getServices();

      const result = await services.messages.getMessages({
        folder,
        limit: pageSize,
        cursor: append ? cursor.value : null,
      });

      if (append) {
        messages.value = [...messages.value, ...result.messages];
      } else {
        messages.value = result.messages;
      }

      cursor.value = result.cursor;
      hasMore.value = result.hasMore;
    } catch (e) {
      error.value = String(e);
    } finally {
      loading.value = false;
    }
  }

  async function refresh() {
    cursor.value = null;
    await fetchMessages(false);
  }

  async function loadMore() {
    if (!hasMore.value || loading.value) return;
    await fetchMessages(true);
  }

  async function markAsRead(hash: string) {
    try {
      const client = getMycelixClient();
      await client.getServices().messages.markAsRead(hash);

      const msg = messages.value.find((m) => m.hash === hash);
      if (msg) msg.read = true;
    } catch (e) {
      error.value = String(e);
    }
  }

  async function markAsUnread(hash: string) {
    try {
      const client = getMycelixClient();
      await client.getServices().messages.markAsUnread(hash);

      const msg = messages.value.find((m) => m.hash === hash);
      if (msg) msg.read = false;
    } catch (e) {
      error.value = String(e);
    }
  }

  async function star(hash: string) {
    try {
      const client = getMycelixClient();
      await client.getServices().messages.star(hash);

      const msg = messages.value.find((m) => m.hash === hash);
      if (msg) msg.starred = true;
    } catch (e) {
      error.value = String(e);
    }
  }

  async function unstar(hash: string) {
    try {
      const client = getMycelixClient();
      await client.getServices().messages.unstar(hash);

      const msg = messages.value.find((m) => m.hash === hash);
      if (msg) msg.starred = false;
    } catch (e) {
      error.value = String(e);
    }
  }

  async function archive(hash: string) {
    try {
      const client = getMycelixClient();
      await client.getServices().messages.archive(hash);

      messages.value = messages.value.filter((m) => m.hash !== hash);
    } catch (e) {
      error.value = String(e);
    }
  }

  async function trash(hash: string) {
    try {
      const client = getMycelixClient();
      await client.getServices().messages.trash(hash);

      messages.value = messages.value.filter((m) => m.hash !== hash);
    } catch (e) {
      error.value = String(e);
    }
  }

  async function permanentDelete(hash: string) {
    try {
      const client = getMycelixClient();
      await client.getServices().messages.permanentDelete(hash);

      messages.value = messages.value.filter((m) => m.hash !== hash);
    } catch (e) {
      error.value = String(e);
    }
  }

  async function moveToFolder(hash: string, targetFolder: string) {
    try {
      const client = getMycelixClient();
      await client.getServices().messages.moveToFolder(hash, targetFolder);

      if (targetFolder !== folder) {
        messages.value = messages.value.filter((m) => m.hash !== hash);
      }
    } catch (e) {
      error.value = String(e);
    }
  }

  async function addLabel(hash: string, label: string) {
    try {
      const client = getMycelixClient();
      await client.getServices().messages.addLabel(hash, label);

      const msg = messages.value.find((m) => m.hash === hash);
      if (msg && !msg.labels.includes(label)) {
        msg.labels = [...msg.labels, label];
      }
    } catch (e) {
      error.value = String(e);
    }
  }

  async function removeLabel(hash: string, label: string) {
    try {
      const client = getMycelixClient();
      await client.getServices().messages.removeLabel(hash, label);

      const msg = messages.value.find((m) => m.hash === hash);
      if (msg) {
        msg.labels = msg.labels.filter((l) => l !== label);
      }
    } catch (e) {
      error.value = String(e);
    }
  }

  onMounted(async () => {
    await fetchMessages();

    // Subscribe to real-time updates
    const client = getMycelixClient();
    unsubscribe = client.getServices().signalHub.on('email_received', (signal) => {
      if (folder === 'inbox') {
        messages.value = [signal.email, ...messages.value];
      }
    });

    // Auto-refresh
    if (autoRefresh) {
      refreshInterval = setInterval(refresh, refreshIntervalMs);
    }
  });

  onUnmounted(() => {
    if (refreshInterval) clearInterval(refreshInterval);
    if (unsubscribe) unsubscribe();
  });

  // Re-fetch when folder changes
  watch(() => folder, () => {
    cursor.value = null;
    fetchMessages();
  });

  return {
    messages,
    loading,
    error,
    hasMore,
    unreadCount,
    refresh,
    loadMore,
    markAsRead,
    markAsUnread,
    star,
    unstar,
    archive,
    trash,
    permanentDelete,
    moveToFolder,
    addLabel,
    removeLabel,
  };
}

export default useMessages;
