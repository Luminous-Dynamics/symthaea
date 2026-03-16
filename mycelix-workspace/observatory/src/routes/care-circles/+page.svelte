<script lang="ts">
  import { onMount } from 'svelte';
  import {
    getAllCareCircles,
    getCircleMembers,
    joinCircle,
    type CareCircle,
    type CircleMembership,
    type MemberRole,
  } from '$lib/resilience-client';
  import { toasts } from '$lib/toast';

  let circles: CareCircle[] = [];
  let selectedCircle: CareCircle | null = null;
  let members: CircleMembership[] = [];
  let loading = true;
  let error = '';
  let filterType = '';
  let submitting = false;

  const CIRCLE_TYPES = ['Neighborhood', 'Workplace', 'Faith', 'Family', 'School'];

  onMount(async () => {
    try {
      circles = await getAllCareCircles();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load care circles';
    } finally {
      loading = false;
    }
  });

  async function selectCircle(circle: CareCircle) {
    selectedCircle = circle;
    members = await getCircleMembers(circle.id);
  }

  async function handleJoin(circle: CareCircle) {
    submitting = true;
    try {
      const membership = await joinCircle(circle.id);
      circle.member_count = (circle.member_count ?? 0) + 1;
      circles = circles;
      if (selectedCircle?.id === circle.id) {
        members = [...members, membership];
      }
      toasts.success('Joined circle');
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to join circle';
      toasts.error(error);
    } finally {
      submitting = false;
    }
  }

  $: filteredCircles = filterType
    ? circles.filter(c => c.circle_type === filterType)
    : circles;

  function roleColor(role: string): string {
    switch (role) {
      case 'Organizer': return 'bg-purple-900/50 text-purple-300';
      case 'Member': return 'bg-blue-900/50 text-blue-300';
      default: return 'bg-gray-800 text-gray-400';
    }
  }

  function totalMembers(): number {
    return circles.reduce((sum, c) => sum + (c.member_count ?? 0), 0);
  }
</script>

<div class="min-h-screen bg-gray-950 text-gray-100 p-6">
  <div class="container mx-auto max-w-6xl">
    <h1 class="text-2xl font-bold mb-1">Care Circles</h1>
    <p class="text-gray-400 mb-6">Community care networks — elder care, childwatch, first responders, and mutual support groups.</p>

    {#if loading}
      <div class="text-gray-400">Loading care circles...</div>
    {:else if error}
      <div class="bg-red-900/30 border border-red-500 rounded p-4 text-red-200">{error}</div>
    {:else}

      <!-- Summary -->
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
        <div class="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div class="text-3xl font-bold text-pink-400">{circles.length}</div>
          <div class="text-sm text-gray-400">Active Circles</div>
        </div>
        <div class="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div class="text-3xl font-bold text-pink-400">{totalMembers()}</div>
          <div class="text-sm text-gray-400">Total Members</div>
        </div>
        <div class="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div class="text-3xl font-bold text-pink-400">{new Set(circles.map(c => c.circle_type)).size}</div>
          <div class="text-sm text-gray-400">Circle Types</div>
        </div>
      </div>

      <!-- Filter -->
      <div class="flex flex-wrap gap-2 mb-6">
        <button
          class="px-3 py-1 rounded-full text-sm transition-colors {!filterType ? 'bg-pink-700 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}"
          on:click={() => filterType = ''}
          aria-label="Show all circle types"
        >All</button>
        {#each CIRCLE_TYPES as ct}
          <button
            class="px-3 py-1 rounded-full text-sm transition-colors {filterType === ct ? 'bg-pink-700 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}"
            on:click={() => filterType = ct}
            aria-label="Filter by {ct}"
          >{ct}</button>
        {/each}
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Circle List -->
        <div class="space-y-3">
          {#each filteredCircles as circle}
            <button
              class="w-full text-left bg-gray-900 rounded-lg border p-4 transition-colors {selectedCircle?.id === circle.id ? 'border-pink-500' : 'border-gray-800 hover:border-gray-700'}"
              on:click={() => selectCircle(circle)}
              aria-label="View members of {circle.name}"
            >
              <div class="flex items-start justify-between">
                <div>
                  <h3 class="font-semibold text-white">{circle.name}</h3>
                  <p class="text-sm text-gray-400 mt-1">{circle.description}</p>
                </div>
                <span class="text-xs px-2 py-0.5 rounded-full bg-gray-800 text-gray-400 flex-shrink-0 ml-2">{circle.circle_type}</span>
              </div>
              <div class="flex items-center gap-4 mt-2 text-xs text-gray-500">
                <span>{circle.member_count ?? '?'} / {circle.max_members} members</span>
                <span>{circle.location}</span>
                {#if (circle.member_count ?? 0) < circle.max_members}
                  <button
                    disabled={submitting}
                    class="ml-auto px-3 py-1 rounded-full bg-pink-600 hover:bg-pink-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white text-xs font-medium transition-colors"
                    on:click|stopPropagation|preventDefault={() => handleJoin(circle)}
                    aria-label="Join {circle.name}"
                  >
                    {submitting ? 'Joining...' : 'Join Circle'}
                  </button>
                {/if}
              </div>
            </button>
          {/each}

          {#if filteredCircles.length === 0}
            <div class="text-center text-gray-500 py-8">No circles found.</div>
          {/if}
        </div>

        <!-- Members Panel -->
        <div class="bg-gray-900 rounded-lg border border-gray-800 p-5">
          {#if selectedCircle}
            <h2 class="text-lg font-semibold mb-1">{selectedCircle.name}</h2>
            <p class="text-sm text-gray-400 mb-4">{members.length} active members</p>

            <div class="space-y-2">
              {#each members as member}
                <div class="flex items-center justify-between bg-gray-800/50 rounded px-3 py-2">
                  <span class="text-white">{member.member_did}</span>
                  <span class="text-xs px-2 py-0.5 rounded-full {roleColor(member.role)}">{member.role}</span>
                </div>
              {/each}
            </div>
          {:else}
            <div class="text-center text-gray-500 py-12">
              <p>Select a circle to view its members.</p>
            </div>
          {/if}
        </div>
      </div>
    {/if}
  </div>
</div>
