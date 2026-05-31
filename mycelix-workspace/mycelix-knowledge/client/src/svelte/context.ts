// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Svelte Context for Knowledge Client
 *
 * @example
 * ```svelte
 * <!-- App.svelte -->
 * <script>
 *   import { setKnowledgeClient } from '@mycelix/knowledge-sdk/svelte';
 *   import { KnowledgeClient } from '@mycelix/knowledge-sdk';
 *   import { AppWebsocket } from '@holochain/client';
 *   import { onMount } from 'svelte';
 *
 *   let ready = false;
 *
 *   onMount(async () => {
 *     const appClient = await AppWebsocket.connect('ws://localhost:8888');
 *     const client = new KnowledgeClient(appClient);
 *     setKnowledgeClient(client);
 *     ready = true;
 *   });
 * </script>
 *
 * {#if ready}
 *   <slot />
 * {:else}
 *   <Loading />
 * {/if}
 *
 * <!-- ChildComponent.svelte -->
 * <script>
 *   import { getKnowledgeClient } from '@mycelix/knowledge-sdk/svelte';
 *   const client = getKnowledgeClient();
 * </script>
 * ```
 */

import { getContext, setContext } from 'svelte';
import type { KnowledgeClient } from '../index';

const KNOWLEDGE_CLIENT_KEY = Symbol('knowledge-client');

/**
 * Set the Knowledge client in Svelte context
 * Call this in your root component after connecting
 */
export function setKnowledgeClient(client: KnowledgeClient): void {
  setContext(KNOWLEDGE_CLIENT_KEY, client);
}

/**
 * Get the Knowledge client from Svelte context
 * @throws Error if client not set in context
 */
export function getKnowledgeClient(): KnowledgeClient {
  const client = getContext<KnowledgeClient | undefined>(KNOWLEDGE_CLIENT_KEY);
  if (!client) {
    throw new Error(
      'Knowledge client not found in context. ' +
      'Make sure to call setKnowledgeClient(client) in a parent component.'
    );
  }
  return client;
}

/**
 * Get the Knowledge client from context, or null if not available
 */
export function getOptionalKnowledgeClient(): KnowledgeClient | null {
  return getContext<KnowledgeClient | undefined>(KNOWLEDGE_CLIENT_KEY) ?? null;
}

/**
 * Check if Knowledge client is available in context
 */
export function hasKnowledgeClient(): boolean {
  return !!getContext<KnowledgeClient | undefined>(KNOWLEDGE_CLIENT_KEY);
}
