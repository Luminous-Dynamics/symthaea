<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';

  const STORAGE_KEY = 'mycelix-onboarding-complete';

  let step = 1;
  let dontShowAgain = true;

  onMount(() => {
    if (typeof localStorage !== 'undefined') {
      const done = localStorage.getItem(STORAGE_KEY);
      if (done === 'true') {
        goto('/resilience');
      }
    }
  });

  function next() {
    if (step < 3) step += 1;
  }

  function prev() {
    if (step > 1) step -= 1;
  }

  function finish() {
    if (dontShowAgain && typeof localStorage !== 'undefined') {
      localStorage.setItem(STORAGE_KEY, 'true');
    }
    goto('/resilience');
  }

  interface FeatureCard {
    icon: string;
    title: string;
    description: string;
    color: string;
  }

  const features: FeatureCard[] = [
    {
      icon: '&#9201;',
      title: 'TEND',
      description: 'Time-based mutual credit — 1 hour = 1 TEND, all labor equal',
      color: 'border-green-500 text-green-400',
    },
    {
      icon: '&#9752;',
      title: 'Food',
      description: 'Track community gardens, harvests, and soil health',
      color: 'border-emerald-500 text-emerald-400',
    },
    {
      icon: '&#9888;',
      title: 'Emergency',
      description: 'Priority messaging for crisis coordination',
      color: 'border-red-500 text-red-400',
    },
    {
      icon: '&#9829;',
      title: 'Mutual Aid',
      description: 'Post and find help — neighbor to neighbor',
      color: 'border-blue-500 text-blue-400',
    },
  ];

  interface QuickLink {
    label: string;
    href: string;
    description: string;
    color: string;
  }

  const quickLinks: QuickLink[] = [
    {
      label: 'Check your TEND balance',
      href: '/tend',
      description: 'See your time credits and recent exchanges',
      color: 'text-green-400 hover:text-green-300',
    },
    {
      label: 'Browse mutual aid',
      href: '/mutual-aid',
      description: 'Find or offer help in your community',
      color: 'text-blue-400 hover:text-blue-300',
    },
    {
      label: 'View emergency channels',
      href: '/emergency',
      description: 'Stay informed during crisis events',
      color: 'text-red-400 hover:text-red-300',
    },
    {
      label: 'See the dashboard',
      href: '/resilience',
      description: 'Overview of all community systems',
      color: 'text-amber-400 hover:text-amber-300',
    },
  ];
</script>

<svelte:head>
  <title>Welcome — Mycelix Resilience Kit</title>
</svelte:head>

<div class="min-h-screen bg-gray-950 text-gray-100 flex items-center justify-center p-4 sm:p-6">
  <div class="max-w-2xl w-full">

    <!-- Progress indicator -->
    <div
      class="flex items-center justify-center gap-2 mb-8"
      role="progressbar"
      aria-label="Onboarding progress"
      aria-valuenow={step}
      aria-valuemin={1}
      aria-valuemax={3}
    >
      {#each [1, 2, 3] as s}
        <div
          class="h-2 rounded-full transition-all duration-300 {s === step
            ? 'w-10 bg-indigo-500'
            : s < step
              ? 'w-6 bg-indigo-700'
              : 'w-6 bg-gray-700'}"
        ></div>
      {/each}
    </div>

    <!-- Step 1: What is this? -->
    {#if step === 1}
      <div class="text-center mb-10">
        <h1 class="text-3xl sm:text-4xl font-bold text-white mb-3">
          Welcome to Mycelix Resilience Kit
        </h1>
        <p class="text-lg text-gray-400">
          Community-powered tools for mutual aid, food security, and emergency coordination
        </p>
      </div>

      <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-10">
        {#each features as card}
          <div class="border {card.color} bg-gray-900 rounded-lg p-5 transition-colors">
            <div class="text-2xl mb-2">{@html card.icon}</div>
            <h3 class="text-lg font-semibold text-white mb-1">{card.title}</h3>
            <p class="text-sm text-gray-400">{card.description}</p>
          </div>
        {/each}
      </div>

      <div class="flex justify-center">
        <button
          on:click={next}
          aria-label="Go to next step"
          class="px-8 py-3 bg-indigo-600 hover:bg-indigo-500 text-white font-medium rounded-lg transition-colors"
        >
          Next &rarr;
        </button>
      </div>

    <!-- Step 2: Your Community -->
    {:else if step === 2}
      <div class="text-center mb-8">
        <h1 class="text-3xl sm:text-4xl font-bold text-white mb-3">
          Your Community
        </h1>
        <p class="text-lg text-gray-400">
          This app connects you with your local community. All data stays on your devices
          and your community's network.
        </p>
      </div>

      <div class="space-y-5 mb-10 max-w-lg mx-auto">
        <div class="flex items-start gap-4">
          <div class="flex-shrink-0 w-10 h-10 rounded-full bg-gray-800 border border-gray-700 flex items-center justify-center text-lg text-indigo-400">
            &#9919;
          </div>
          <div>
            <h3 class="text-white font-medium">No central server — your data, your control</h3>
            <p class="text-sm text-gray-500 mt-1">
              Data is distributed across community members, not held by any company.
            </p>
          </div>
        </div>

        <div class="flex items-start gap-4">
          <div class="flex-shrink-0 w-10 h-10 rounded-full bg-gray-800 border border-gray-700 flex items-center justify-center text-lg text-emerald-400">
            &#9678;
          </div>
          <div>
            <h3 class="text-white font-medium">Works offline — mesh network keeps you connected</h3>
            <p class="text-sm text-gray-500 mt-1">
              Local-first design means the app works even when the internet does not.
            </p>
          </div>
        </div>

        <div class="flex items-start gap-4">
          <div class="flex-shrink-0 w-10 h-10 rounded-full bg-gray-800 border border-gray-700 flex items-center justify-center text-lg text-amber-400">
            &#9878;
          </div>
          <div>
            <h3 class="text-white font-medium">Democratic governance — community decides the rules</h3>
            <p class="text-sm text-gray-500 mt-1">
              Proposals and voting let your community shape how the system evolves.
            </p>
          </div>
        </div>
      </div>

      <div class="flex items-center justify-center gap-4">
        <button
          on:click={prev}
          aria-label="Go to previous step"
          class="px-6 py-3 text-gray-400 hover:text-white transition-colors"
        >
          &larr; Back
        </button>
        <button
          on:click={next}
          aria-label="Go to next step"
          class="px-8 py-3 bg-indigo-600 hover:bg-indigo-500 text-white font-medium rounded-lg transition-colors"
        >
          Next &rarr;
        </button>
      </div>

    <!-- Step 3: Get Started -->
    {:else if step === 3}
      <div class="text-center mb-8">
        <h1 class="text-3xl sm:text-4xl font-bold text-white mb-3">
          Get Started
        </h1>
        <p class="text-lg text-gray-400">
          Jump into any area, or head to the dashboard for the full picture.
        </p>
      </div>

      <div class="space-y-3 mb-8 max-w-lg mx-auto">
        {#each quickLinks as link}
          <a
            href={link.href}
            class="block bg-gray-900 border border-gray-800 rounded-lg p-4 hover:border-gray-600 transition-colors group"
          >
            <span class="font-medium {link.color} group-hover:underline">{link.label}</span>
            <span class="block text-sm text-gray-500 mt-1">{link.description}</span>
          </a>
        {/each}
      </div>

      <div class="flex flex-col items-center gap-4">
        <label class="flex items-center gap-2 text-sm text-gray-500 cursor-pointer select-none">
          <input
            type="checkbox"
            bind:checked={dontShowAgain}
            aria-label="Don't show this welcome screen again"
            class="rounded bg-gray-800 border-gray-600 text-indigo-600 focus:ring-indigo-500 focus:ring-offset-0"
          />
          Don't show this again
        </label>

        <div class="flex items-center gap-4">
          <button
            on:click={prev}
            aria-label="Go to previous step"
            class="px-6 py-3 text-gray-400 hover:text-white transition-colors"
          >
            &larr; Back
          </button>
          <button
            on:click={finish}
            aria-label="Enter the resilience kit"
            class="px-8 py-3 bg-indigo-600 hover:bg-indigo-500 text-white font-medium rounded-lg transition-colors"
          >
            Enter the Kit
          </button>
        </div>
      </div>
    {/if}

    <!-- Step label -->
    <p class="text-center text-xs text-gray-600 mt-8">
      Step {step} of 3
    </p>
  </div>
</div>
