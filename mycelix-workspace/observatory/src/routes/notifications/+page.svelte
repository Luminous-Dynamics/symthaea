<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
--><script lang="ts">
  import { onMount } from 'svelte';
  import { callZome, conductorStatus$ } from '$lib/conductor';

  // ============================================================================
  // Types
  // ============================================================================

  type Priority = 'Low' | 'Normal' | 'High' | 'Emergency';

  interface Notification {
    id: string;
    source_cluster: string;
    event_type: string;
    payload_preview: string;
    priority: Priority;
    timestamp: number;
    read: boolean;
  }

  // ============================================================================
  // State
  // ============================================================================

  let notifications: Notification[] = [];
  let unreadCount = 0;
  let loading = true;
  let error = '';

  // ============================================================================
  // Data fetching
  // ============================================================================

  onMount(async () => {
    if ($conductorStatus$ === 'disconnected') {
      loading = false;
      return;
    }
    try {
      const [notifs, count] = await Promise.all([
        callZome<Notification[]>({
          role_name: 'commons_care',
          zome_name: 'commons_bridge',
          fn_name: 'get_my_notifications',
          payload: null,
        }),
        callZome<number>({
          role_name: 'commons_care',
          zome_name: 'commons_bridge',
          fn_name: 'get_unread_count',
          payload: null,
        }),
      ]);
      notifications = notifs ?? [];
      unreadCount = count ?? 0;
    } catch (e) {
      error = String(e);
    } finally {
      loading = false;
    }
  });

  // ============================================================================
  // Helpers
  // ============================================================================

  function getPriorityBadge(priority: Priority): string {
    switch (priority) {
      case 'Low': return 'bg-gray-500/20 text-gray-400 border-gray-500/50';
      case 'Normal': return 'bg-blue-500/20 text-blue-400 border-blue-500/50';
      case 'High': return 'bg-orange-500/20 text-orange-400 border-orange-500/50';
      case 'Emergency': return 'bg-red-500/20 text-red-400 border-red-500/50';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/50';
    }
  }

  function getClusterColor(cluster: string): string {
    switch (cluster.toLowerCase()) {
      case 'commons': return 'text-green-400';
      case 'civic': return 'text-blue-400';
      case 'governance': return 'text-purple-400';
      case 'identity': return 'text-cyan-400';
      case 'hearth': return 'text-orange-400';
      case 'finance': return 'text-yellow-400';
      default: return 'text-gray-400';
    }
  }

  function formatTimestamp(ts: number): string {
    const now = Date.now();
    const diff = now - ts;
    if (diff < 60_000) return 'just now';
    if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
    if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
    return new Date(ts).toLocaleDateString();
  }
</script>

<svelte:head>
  <title>Notifications | Mycelix Observatory</title>
</svelte:head>

<div class="text-white">
  <!-- Page Header -->
  <header class="bg-gray-800/50 border-b border-gray-700 px-4 py-2">
    <div class="container mx-auto flex justify-between items-center">
      <div class="flex items-center gap-2">
        <span class="text-xl">&#x1F514;</span>
        <div>
          <h1 class="text-lg font-bold">Notifications</h1>
          <p class="text-xs text-gray-400">Cross-Cluster Event Feed</p>
        </div>
      </div>
      <div class="flex items-center gap-3">
        {#if unreadCount > 0}
          <span class="bg-red-500 text-white text-xs font-bold px-2 py-0.5 rounded-full">
            {unreadCount} unread
          </span>
        {/if}
        <div class="text-right">
          <p class="text-xs text-gray-400">Total</p>
          <p class="text-lg font-bold">{notifications.length}</p>
        </div>
      </div>
    </div>
  </header>

  <main class="container mx-auto p-6">
    <!-- Disconnected banner -->
    {#if $conductorStatus$ === 'disconnected' || $conductorStatus$ === 'demo'}
      <div class="bg-yellow-500/10 border border-yellow-500/50 rounded-lg p-4 mb-6 text-yellow-400 text-sm">
        Conductor is not connected. Notifications are unavailable until the Holochain conductor is running.
      </div>
    {/if}

    <!-- Loading -->
    {#if loading}
      <div class="flex items-center justify-center py-16">
        <div class="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-400"></div>
        <span class="ml-3 text-gray-400">Loading notifications...</span>
      </div>

    <!-- Error -->
    {:else if error}
      <div class="bg-red-500/10 border border-red-500/50 rounded-lg p-4 text-red-400 text-sm">
        Failed to load notifications: {error}
      </div>

    <!-- Empty -->
    {:else if notifications.length === 0}
      <div class="text-center py-16 text-gray-500">
        <p class="text-lg">No notifications</p>
        <p class="text-sm mt-1">Events from connected clusters will appear here.</p>
      </div>

    <!-- Notification Feed -->
    {:else}
      <div class="space-y-3">
        {#each notifications as notif}
          <div class="bg-gray-800 rounded-lg border border-gray-700 p-4 hover:bg-gray-700/50 transition-colors {notif.read ? 'opacity-70' : ''}">
            <div class="flex justify-between items-start">
              <div class="flex-1 min-w-0">
                <div class="flex items-center gap-2 flex-wrap">
                  <span class={`text-xs font-semibold ${getClusterColor(notif.source_cluster)}`}>
                    {notif.source_cluster.toUpperCase()}
                  </span>
                  <span class="text-gray-600">&middot;</span>
                  <span class="text-sm font-medium">{notif.event_type}</span>
                  <span class={`text-xs px-2 py-0.5 rounded border ${getPriorityBadge(notif.priority)}`}>
                    {notif.priority}
                  </span>
                  {#if !notif.read}
                    <span class="w-2 h-2 rounded-full bg-blue-400"></span>
                  {/if}
                </div>
                <p class="text-sm text-gray-400 mt-1 truncate">{notif.payload_preview}</p>
              </div>
              <span class="text-xs text-gray-500 whitespace-nowrap ml-4">{formatTimestamp(notif.timestamp)}</span>
            </div>
          </div>
        {/each}
      </div>
    {/if}

    <!-- Footer -->
    <footer class="mt-8 text-center text-gray-500 text-sm">
      <p>Mycelix Notifications v0.1.0 &bull; Commons Bridge &bull; HDK 0.6.0</p>
    </footer>
  </main>
</div>
