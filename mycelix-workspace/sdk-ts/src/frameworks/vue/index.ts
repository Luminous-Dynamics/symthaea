/**
 * Vue 3 Framework Integration for Mycelix SDK
 *
 * Provides Vue 3 Composition API composables and a Vue plugin for building
 * Mycelix-powered Vue applications.
 *
 * @module frameworks/vue
 * @packageDocumentation
 *
 * @example Quick Start with Plugin
 * ```ts
 * // main.ts
 * import { createApp } from 'vue';
 * import { MycelixPlugin } from '@mycelix/sdk/vue';
 * import App from './App.vue';
 *
 * const app = createApp(App);
 *
 * app.use(MycelixPlugin, {
 *   conductorUrl: 'ws://localhost:8888',
 *   appId: 'my-app',
 *   autoConnect: true,
 *   onConnect: (mycelix) => console.log('Connected!'),
 * });
 *
 * app.mount('#app');
 * ```
 *
 * @example Using Composables
 * ```vue
 * <script setup lang="ts">
 * import { useMycelix, useIdentity, useGovernance, useWallet } from '@mycelix/sdk/vue';
 *
 * // Connection management
 * const { mycelix, status, connect } = useMycelix();
 *
 * // Identity
 * const { did, createDid } = useIdentity();
 *
 * // Governance
 * const { proposals, vote, refresh } = useGovernance();
 *
 * // Wallet
 * const { balance, send } = useWallet();
 * </script>
 *
 * <template>
 *   <div v-if="status === 'connected'">
 *     <header>
 *       <p>DID: {{ did }}</p>
 *       <p>Balance: {{ balance }} MYC</p>
 *     </header>
 *
 *     <section>
 *       <h2>Proposals</h2>
 *       <ProposalCard
 *         v-for="p in proposals"
 *         :key="p.id"
 *         :proposal="p"
 *         @vote="(choice) => vote(p.id, choice)"
 *       />
 *     </section>
 *   </div>
 * </template>
 * ```
 *
 * @example Advanced: Custom Queries
 * ```vue
 * <script setup lang="ts">
 * import { useMycelix, useQuery, useMutation } from '@mycelix/sdk/vue';
 *
 * const { mycelix } = useMycelix();
 *
 * // Query with auto-refresh
 * const { data: dao, loading, refetch } = useQuery(
 *   async () => mycelix.value?.governance.dao.getDao('my-dao-id'),
 *   { refetchInterval: 30000 }
 * );
 *
 * // Mutation with callbacks
 * const { mutate: createProposal, loading: creating } = useMutation(
 *   async (input) => mycelix.value?.governance.proposals.createProposal(input),
 *   {
 *     onSuccess: (proposal) => {
 *       toast.success(`Created: ${proposal.title}`);
 *       refetch();
 *     },
 *     onError: (error) => {
 *       toast.error(error.message);
 *     },
 *   }
 * );
 * </script>
 * ```
 *
 * @example Nuxt 3 Integration
 * ```ts
 * // plugins/mycelix.ts
 * import { createMycelixNuxtPlugin } from '@mycelix/sdk/vue';
 *
 * export default createMycelixNuxtPlugin({
 *   conductorUrl: process.env.CONDUCTOR_URL || 'ws://localhost:8888',
 *   appId: 'my-nuxt-app',
 *   autoConnect: true,
 * });
 * ```
 */

// =============================================================================
// Composables
// =============================================================================

export {
  // Main composable
  useMycelix,

  // Domain composables
  useIdentity,
  useGovernance,
  useWallet,

  // Generic composables
  useQuery,
  useMutation,

  // Utility functions
  requireMycelix,
  isConnected,

  // Types
  type UseMycelixReturn,
  type UseIdentityReturn,
  type UseGovernanceReturn,
  type UseWalletReturn,
  type UseQueryReturn,
  type UseQueryOptions,
  type UseMutationReturn,
  type UseMutationOptions,
  type Ref,
  type ReadonlyRef,
  type DID,
  type Proposal,
  type ProposalStatus,
  type VoteChoice,
  type ConnectionStatus,
} from './composables.js';

// =============================================================================
// Plugin
// =============================================================================

export {
  // Vue Plugin
  MycelixPlugin,

  // Injection key
  MYCELIX_KEY,

  // Helpers
  createMycelixMixin,
  createMycelixNuxtPlugin,

  // Types
  type Plugin,
  type VueApp,
  type MycelixPluginOptions,
  type MycelixGlobalProperties,
} from './plugin.js';

// =============================================================================
// Re-exports from Core
// =============================================================================

// Re-export core types for convenience
export type { Mycelix, MycelixConfig } from '../../core/index.js';
