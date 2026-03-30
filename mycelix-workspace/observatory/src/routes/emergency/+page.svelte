<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';
  import {
    getChannels,
    getMessages,
    sendMessage,
    createChannel,
    type EmergencyChannel,
    type EmergencyMessage,
    type EmergencyPriority,
  } from '$lib/resilience-client';
  import { toasts } from '$lib/toast';
  import { exportEmergencyMessagesCsv } from '$lib/data-export';
  import { createFreshness } from '$lib/freshness';
  import FreshnessBar from '$lib/components/FreshnessBar.svelte';
  import {
    requestPermission,
    isSupported,
    isEnabled,
    notifyEmergency,
    pushEnabled,
  } from '$lib/push-notifications';

  // ============================================================================
  // Stores
  // ============================================================================

  const channels = writable<EmergencyChannel[]>([]);
  const messages = writable<EmergencyMessage[]>([]);
  const selectedChannel = writable<EmergencyChannel | null>(null);

  // Message form
  let msgContent = '';
  let msgPriority: EmergencyPriority = 'Routine';
  let submitting = false;

  // Channel form
  let newChannelName = '';
  let newChannelDesc = '';
  let showChannelForm = false;
  let loading = true;

  // Push notification state
  let notifSupported = false;
  let notifBannerDismissed = false;
  const seenMessageIds = new Set<string>();

  const priorities: EmergencyPriority[] = ['Flash', 'Immediate', 'Priority', 'Routine'];

  // ============================================================================
  // Freshness — 30s polling (emergency is safety-critical)
  // ============================================================================

  async function fetchData() {
    const ch = await getChannels();
    channels.set(ch);
    const sel = $selectedChannel;
    if (sel) {
      const msgs = await getMessages(sel.id);
      // Detect new Flash/Immediate messages and send push notifications
      for (const msg of msgs) {
        if (!seenMessageIds.has(msg.id)) {
          seenMessageIds.add(msg.id);
          if (msg.priority === 'Flash' || msg.priority === 'Immediate') {
            notifyEmergency(msg);
          }
        }
      }
      messages.set(msgs);
    } else if (ch.length > 0) {
      selectChannel(ch[0]);
    }
  }

  const freshness = createFreshness(fetchData, 30_000);
  const { lastUpdated, loadError, refreshing, startPolling, stopPolling, refresh } = freshness;

  // ============================================================================
  // Lifecycle
  // ============================================================================

  onMount(async () => {
    notifSupported = isSupported();
    await refresh();
    loading = false;
    startPolling();
  });

  onDestroy(() => stopPolling());

  async function selectChannel(ch: EmergencyChannel) {
    selectedChannel.set(ch);
    const msgs = await getMessages(ch.id);
    // Seed seen IDs so existing messages don't trigger notifications
    for (const msg of msgs) {
      seenMessageIds.add(msg.id);
    }
    messages.set(msgs);
  }

  async function handleSend() {
    const ch = $selectedChannel;
    if (!ch || !msgContent.trim()) return;
    submitting = true;
    try {
      const msg = await sendMessage(ch.id, msgContent.trim(), msgPriority);
      messages.update(list => [...list, msg]);
      msgContent = '';
      msgPriority = 'Routine';
      toasts.success('Message sent');
    } catch (e) {
      toasts.error(e instanceof Error ? e.message : 'Failed to send message');
    } finally {
      submitting = false;
    }
  }

  async function handleCreateChannel() {
    if (!newChannelName.trim()) return;
    submitting = true;
    try {
      const ch = await createChannel(newChannelName.trim(), newChannelDesc.trim());
      channels.update(list => [...list, ch]);
      newChannelName = '';
      newChannelDesc = '';
      showChannelForm = false;
      selectChannel(ch);
      toasts.success('Channel created');
    } catch (e) {
      toasts.error(e instanceof Error ? e.message : 'Failed to create channel');
    } finally {
      submitting = false;
    }
  }

  function priorityColor(p: string): string {
    switch (p) {
      case 'Flash': return 'bg-red-600 text-white';
      case 'Immediate': return 'bg-orange-500/20 text-orange-400 border border-orange-500/50';
      case 'Priority': return 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/50';
      case 'Routine': return 'bg-gray-600 text-gray-300';
      default: return 'bg-gray-600 text-gray-300';
    }
  }

  function priorityBorder(p: string): string {
    switch (p) {
      case 'Flash': return 'border-l-red-500';
      case 'Immediate': return 'border-l-orange-400';
      case 'Priority': return 'border-l-yellow-400';
      default: return 'border-l-gray-600';
    }
  }

  function formatTime(ts: number): string {
    return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  // Reactivity: selectedChannel is writable, bind to $selectedChannel
  $: activeChannel = $selectedChannel;
</script>

<svelte:head>
  <title>Emergency Comms | Mycelix Observatory</title>
</svelte:head>

{#if loading}
  <div class="text-white p-8 text-center text-gray-400">Loading emergency channels...</div>
{:else}
<div class="text-white">
  <header class="bg-gray-800/50 border-b border-gray-700 px-4 py-2">
    <div class="container mx-auto flex justify-between items-center">
      <div class="flex items-center gap-2">
        <span class="text-xl" aria-hidden="true">&#x1F6A8;</span>
        <div>
          <h1 class="text-lg font-bold">Emergency Communications</h1>
          <p class="text-xs text-gray-400">Priority messaging for community resilience</p>
        </div>
      </div>
      <div class="flex items-center gap-3">
        {#if notifSupported}
          <button
            on:click={async () => {
              if (isEnabled()) {
                pushEnabled.set(false);
              } else {
                await requestPermission();
              }
            }}
            class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors
              {$pushEnabled && isEnabled()
                ? 'bg-green-600/20 text-green-400 border border-green-500/50'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}"
            title={$pushEnabled && isEnabled() ? 'Notifications enabled' : 'Enable push notifications'}
          >
            <span aria-hidden="true">{$pushEnabled && isEnabled() ? '\u{1F514}' : '\u{1F515}'}</span>
            {$pushEnabled && isEnabled() ? 'Notifications on' : 'Enable notifications'}
          </button>
        {/if}
        <div class="text-right">
          <p class="text-xs text-gray-400">Channels</p>
          <p class="text-lg font-bold">{$channels.length}</p>
        </div>
      </div>
    </div>
  </header>

  <main class="container mx-auto p-6">
    <FreshnessBar {lastUpdated} {loadError} {refreshing} {refresh} />

    {#if notifSupported && !isEnabled() && !notifBannerDismissed}
      <div class="mb-4 flex items-center justify-between bg-orange-900/30 border border-orange-700 rounded-lg px-4 py-3">
        <p class="text-sm text-orange-300">
          Enable notifications to receive Flash and Immediate alerts even when this tab is in the background.
        </p>
        <div class="flex items-center gap-2 ml-4 shrink-0">
          <button
            on:click={async () => { await requestPermission(); notifBannerDismissed = true; }}
            class="px-3 py-1 bg-orange-600 hover:bg-orange-700 rounded text-xs font-medium text-white transition-colors">
            Enable
          </button>
          <button
            on:click={() => notifBannerDismissed = true}
            class="px-2 py-1 text-gray-400 hover:text-gray-200 text-xs transition-colors">
            Dismiss
          </button>
        </div>
      </div>
    {/if}

    <div class="grid grid-cols-1 lg:grid-cols-4 gap-6" style="min-height: 60vh;">
      <!-- Channel List -->
      <div class="bg-gray-800 rounded-lg border border-gray-700">
        <div class="p-4 border-b border-gray-700 flex justify-between items-center">
          <h2 class="text-sm font-semibold">Channels</h2>
          <button on:click={() => showChannelForm = !showChannelForm}
            class="text-xs px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded transition-colors"
            aria-label="New channel">
            + New
          </button>
        </div>

        {#if showChannelForm}
          <div class="p-3 border-b border-gray-700 bg-gray-700/50">
            <form on:submit|preventDefault={handleCreateChannel} class="space-y-2">
              <input bind:value={newChannelName} placeholder="Channel name"
                aria-label="Channel name"
                class="w-full bg-gray-600 border border-gray-500 rounded px-2 py-1 text-sm focus:outline-none focus:border-blue-500" />
              <input bind:value={newChannelDesc} placeholder="Description"
                aria-label="Channel description"
                class="w-full bg-gray-600 border border-gray-500 rounded px-2 py-1 text-sm focus:outline-none focus:border-blue-500" />
              <button type="submit" disabled={submitting || !newChannelName.trim()}
                class="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded px-2 py-1 text-xs font-medium">
                Create
              </button>
            </form>
          </div>
        {/if}

        <div class="p-2 space-y-1 max-h-96 overflow-y-auto">
          {#each $channels as ch}
            <button on:click={() => selectChannel(ch)}
              class="w-full text-left p-3 rounded-lg transition-colors {activeChannel?.id === ch.id ? 'bg-blue-600/30 border border-blue-500/50' : 'bg-gray-700/50 hover:bg-gray-700'}">
              <p class="font-medium text-sm">{ch.name}</p>
              <p class="text-xs text-gray-400 mt-0.5">{ch.description}</p>
              <p class="text-xs text-gray-500 mt-1">{ch.member_count} members</p>
            </button>
          {:else}
            <p class="text-gray-500 text-center py-4 text-sm">No channels</p>
          {/each}
        </div>
      </div>

      <!-- Messages -->
      <div class="bg-gray-800 rounded-lg border border-gray-700 lg:col-span-3 flex flex-col">
        {#if activeChannel}
          <div class="p-4 border-b border-gray-700">
            <h2 class="text-lg font-semibold">{activeChannel.name}</h2>
            <p class="text-xs text-gray-400">{activeChannel.description}</p>
          </div>

          <!-- Message List -->
          <div class="flex-1 p-4 space-y-3 overflow-y-auto max-h-96">
            {#each $messages as msg}
              <div class="p-3 bg-gray-700/50 rounded-lg border-l-4 {priorityBorder(msg.priority)}">
                <div class="flex justify-between items-start">
                  <div class="flex items-center gap-2">
                    <span class="font-medium text-sm">{msg.sender_did}</span>
                    <span class={`text-xs px-2 py-0.5 rounded ${priorityColor(msg.priority)}`}>
                      {msg.priority}
                    </span>
                  </div>
                  <div class="flex items-center gap-2 text-xs text-gray-400">
                    <span>{formatTime(msg.sent_at)}</span>
                    {#if msg.synced}
                      <span class="text-green-400" title="Synced" aria-hidden="true">&#x2713;</span>
                      <span class="sr-only">Synced</span>
                    {:else}
                      <span class="text-yellow-400 animate-pulse" title="Pending sync" aria-hidden="true">&#x25CF;</span>
                      <span class="sr-only">Pending sync</span>
                    {/if}
                  </div>
                </div>
                <p class="text-sm mt-2">{msg.content}</p>
              </div>
            {:else}
              <p class="text-gray-500 text-center py-8">No messages in this channel</p>
            {/each}
          </div>

          <!-- Send Form -->
          <div class="p-4 border-t border-gray-700">
            <form on:submit|preventDefault={handleSend} class="flex flex-col sm:flex-row gap-2">
              <select bind:value={msgPriority}
                aria-label="Message priority"
                class="bg-gray-700 border border-gray-600 rounded px-2 py-2 text-sm focus:outline-none focus:border-blue-500 w-full sm:w-32">
                {#each priorities as p}
                  <option value={p}>{p}</option>
                {/each}
              </select>
              <input bind:value={msgContent} placeholder="Type message..."
                aria-label="Message content"
                class="flex-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500" />
              <button type="submit" disabled={submitting || !msgContent.trim()}
                class="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded px-4 py-2 text-sm font-medium transition-colors">
                Send
              </button>
            </form>
          </div>
        {:else}
          <div class="flex-1 flex items-center justify-center text-gray-500">
            <div class="text-center">
              <p class="text-4xl mb-2" aria-hidden="true">&#x1F4E1;</p>
              <p>Select a channel to view messages</p>
            </div>
          </div>
        {/if}
      </div>
    </div>

    <!-- Priority Guide -->
    <div class="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
      <div class="bg-red-900/30 border border-red-700 rounded-lg p-3">
        <h3 class="text-sm font-bold text-red-400">FLASH</h3>
        <p class="text-xs text-gray-400 mt-1">Life-threatening. Immediate action required by all members.</p>
      </div>
      <div class="bg-orange-900/30 border border-orange-700 rounded-lg p-3">
        <h3 class="text-sm font-bold text-orange-400">IMMEDIATE</h3>
        <p class="text-xs text-gray-400 mt-1">Urgent situation. Response needed within minutes.</p>
      </div>
      <div class="bg-yellow-900/30 border border-yellow-700 rounded-lg p-3">
        <h3 class="text-sm font-bold text-yellow-400">PRIORITY</h3>
        <p class="text-xs text-gray-400 mt-1">Important update. Read when able, respond within hours.</p>
      </div>
      <div class="bg-gray-800 border border-gray-700 rounded-lg p-3">
        <h3 class="text-sm font-bold text-gray-300">ROUTINE</h3>
        <p class="text-xs text-gray-400 mt-1">General information. No time pressure.</p>
      </div>
    </div>

    <div class="mt-6 flex justify-end">
      <button on:click={() => exportEmergencyMessagesCsv($messages)}
        class="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs text-gray-300 transition-colors">
        Export Messages CSV
      </button>
    </div>

    <footer class="mt-4 text-center text-gray-500 text-sm">
      <p>Emergency Communications &middot; Mycelix Civic</p>
    </footer>
  </main>
</div>
{/if}
