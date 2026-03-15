<script lang="ts">
  import { onMount } from 'svelte';
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

  const priorities: EmergencyPriority[] = ['Flash', 'Immediate', 'Priority', 'Routine'];

  // ============================================================================
  // Lifecycle
  // ============================================================================

  onMount(async () => {
    try {
      const ch = await getChannels();
      channels.set(ch);
      if (ch.length > 0) {
        selectChannel(ch[0]);
      }
    } catch (e) {
      console.warn('[Emergency] Failed to load channels, using defaults:', e);
    }
  });

  async function selectChannel(ch: EmergencyChannel) {
    selectedChannel.set(ch);
    const msgs = await getMessages(ch.id);
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

<div class="text-white">
  <header class="bg-gray-800/50 border-b border-gray-700 px-4 py-2">
    <div class="container mx-auto flex justify-between items-center">
      <div class="flex items-center gap-2">
        <span class="text-xl">&#x1F6A8;</span>
        <div>
          <h1 class="text-lg font-bold">Emergency Communications</h1>
          <p class="text-xs text-gray-400">Priority messaging for community resilience</p>
        </div>
      </div>
      <div class="flex items-center gap-3">
        <div class="text-right">
          <p class="text-xs text-gray-400">Channels</p>
          <p class="text-lg font-bold">{$channels.length}</p>
        </div>
      </div>
    </div>
  </header>

  <main class="container mx-auto p-6">
    <div class="grid grid-cols-1 lg:grid-cols-4 gap-6" style="min-height: 60vh;">
      <!-- Channel List -->
      <div class="bg-gray-800 rounded-lg border border-gray-700">
        <div class="p-4 border-b border-gray-700 flex justify-between items-center">
          <h2 class="text-sm font-semibold">Channels</h2>
          <button on:click={() => showChannelForm = !showChannelForm}
            class="text-xs px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded transition-colors">
            + New
          </button>
        </div>

        {#if showChannelForm}
          <div class="p-3 border-b border-gray-700 bg-gray-700/50">
            <form on:submit|preventDefault={handleCreateChannel} class="space-y-2">
              <input bind:value={newChannelName} placeholder="Channel name"
                class="w-full bg-gray-600 border border-gray-500 rounded px-2 py-1 text-sm focus:outline-none focus:border-blue-500" />
              <input bind:value={newChannelDesc} placeholder="Description"
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
                      <span class="text-green-400" title="Synced">&#x2713;</span>
                    {:else}
                      <span class="text-yellow-400 animate-pulse" title="Pending sync">&#x25CF;</span>
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
            <form on:submit|preventDefault={handleSend} class="flex gap-2">
              <select bind:value={msgPriority}
                class="bg-gray-700 border border-gray-600 rounded px-2 py-2 text-sm focus:outline-none focus:border-blue-500 w-32">
                {#each priorities as p}
                  <option value={p}>{p}</option>
                {/each}
              </select>
              <input bind:value={msgContent} placeholder="Type message..."
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
              <p class="text-4xl mb-2">&#x1F4E1;</p>
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

    <footer class="mt-8 text-center text-gray-500 text-sm">
      <p>Emergency Communications &middot; Mycelix Civic</p>
    </footer>
  </main>
</div>
