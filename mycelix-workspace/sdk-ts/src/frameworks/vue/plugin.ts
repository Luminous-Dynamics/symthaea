/**
 * Vue 3 Plugin for Mycelix SDK
 *
 * Provides a Vue plugin for easy integration of Mycelix into Vue 3 applications.
 *
 * @module frameworks/vue/plugin
 *
 * @example Basic Installation
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
 *   appId: 'my-civilizational-app',
 *   autoConnect: true,
 * });
 *
 * app.mount('#app');
 * ```
 *
 * @example With Options
 * ```ts
 * app.use(MycelixPlugin, {
 *   conductorUrl: import.meta.env.VITE_CONDUCTOR_URL,
 *   appId: import.meta.env.VITE_APP_ID,
 *   autoConnect: true,
 *   debug: import.meta.env.DEV,
 *   onConnect: (mycelix) => {
 *     console.log('Connected to Mycelix!');
 *   },
 *   onError: (error) => {
 *     console.error('Connection failed:', error);
 *   },
 * });
 * ```
 */

import { useMycelix, useIdentity } from './composables.js';

import type { Mycelix, MycelixConfig } from '../../core/index.js';

// =============================================================================
// Types
// =============================================================================

/**
 * Vue Plugin interface (minimal for framework independence)
 */
export interface Plugin {
  install: (app: VueApp, options?: MycelixPluginOptions) => void;
}

/**
 * Vue App interface (minimal for framework independence)
 */
export interface VueApp {
  provide: (key: string | symbol, value: unknown) => VueApp;
  config: {
    globalProperties: Record<string, unknown>;
  };
}

/**
 * Options for MycelixPlugin
 */
export interface MycelixPluginOptions extends MycelixConfig {
  /** Whether to auto-connect on install (default: true) */
  autoConnect?: boolean;

  /** Callback when connection succeeds */
  onConnect?: (mycelix: Mycelix) => void;

  /** Callback when connection fails */
  onError?: (error: Error) => void;

  /** Callback when disconnected */
  onDisconnect?: () => void;
}

/**
 * Injection key for Mycelix context
 */
export const MYCELIX_KEY = Symbol('mycelix');

/**
 * Global properties added by the plugin
 */
export interface MycelixGlobalProperties {
  /** Mycelix client instance */
  $mycelix: Mycelix | null;
  /** Check if connected */
  $isMycelixConnected: () => boolean;
}

// Note: Module augmentation would go in a separate .d.ts file for @vue/runtime-core
// declare module '@vue/runtime-core' {
//   interface ComponentCustomProperties extends MycelixGlobalProperties {}
// }

// =============================================================================
// Plugin Implementation
// =============================================================================

/**
 * MycelixPlugin - Vue 3 plugin for Mycelix SDK
 *
 * Installs Mycelix into your Vue application, providing:
 * - Global $mycelix property on all components
 * - Injection key for provide/inject pattern
 * - Auto-connection option
 * - Connection lifecycle callbacks
 *
 * @example Installation
 * ```ts
 * import { createApp } from 'vue';
 * import { MycelixPlugin } from '@mycelix/sdk/vue';
 *
 * const app = createApp(App);
 *
 * app.use(MycelixPlugin, {
 *   conductorUrl: 'ws://localhost:8888',
 *   appId: 'my-app',
 * });
 *
 * app.mount('#app');
 * ```
 *
 * @example Using in Components
 * ```vue
 * <script setup lang="ts">
 * import { inject } from 'vue';
 * import { MYCELIX_KEY } from '@mycelix/sdk/vue';
 *
 * const mycelix = inject(MYCELIX_KEY);
 * </script>
 * ```
 *
 * @example Using Global Property (Options API)
 * ```vue
 * <script>
 * export default {
 *   methods: {
 *     async createIdentity() {
 *       const did = await this.$mycelix.identity.did.createDid();
 *       console.log('Created DID:', did);
 *     }
 *   }
 * }
 * </script>
 * ```
 */
export const MycelixPlugin: Plugin = {
  install(app: VueApp, options?: MycelixPluginOptions) {
    if (!options) {
      console.warn(
        '[Mycelix] No configuration provided to MycelixPlugin. ' +
          'You will need to connect manually using useMycelix().'
      );
      return;
    }

    const { autoConnect = true, onConnect, onError, onDisconnect: _onDisconnect, ...config } = options;

    // Get the composable
    const { mycelix, connect, disconnect, status } = useMycelix(config);

    // Provide injection key
    app.provide(MYCELIX_KEY, {
      mycelix,
      connect,
      disconnect,
      status,
    });

    // Add global properties
    app.config.globalProperties.$mycelix = mycelix.value;
    app.config.globalProperties.$isMycelixConnected = () => status.value === 'connected';

    // Keep global property in sync
    (mycelix as { _subscribe?: (fn: (client: Mycelix | null) => void) => () => void })._subscribe?.((client: Mycelix | null) => {
      app.config.globalProperties.$mycelix = client;
    });

    // Auto-connect if enabled
    if (autoConnect) {
      connect()
        .then(() => {
          if (mycelix.value) {
            onConnect?.(mycelix.value);
          }
        })
        .catch((error) => {
          onError?.(error);
        });
    }
  },
};

// =============================================================================
// Composable Helpers for Plugin Users
// =============================================================================

/**
 * Create a Mycelix-aware component options mixin
 *
 * @example
 * ```ts
 * import { createMycelixMixin } from '@mycelix/sdk/vue';
 *
 * export default {
 *   mixins: [createMycelixMixin()],
 *   computed: {
 *     userDid() {
 *       return this.mycelixIdentity;
 *     }
 *   }
 * }
 * ```
 */
export function createMycelixMixin() {
  return {
    computed: {
      mycelixConnected(): boolean {
        return (this as any).$isMycelixConnected?.() ?? false;
      },
      mycelixIdentity(): string | null {
        const { did } = useIdentity();
        return did.value;
      },
    },
    methods: {
      async mycelixConnect(): Promise<void> {
        const { connect } = useMycelix();
        await connect();
      },
      async mycelixDisconnect(): Promise<void> {
        const { disconnect } = useMycelix();
        await disconnect();
      },
    },
  };
}

// =============================================================================
// Nuxt 3 Compatibility
// =============================================================================

/**
 * Create a Nuxt 3 compatible plugin
 *
 * @example nuxt.config.ts
 * ```ts
 * export default defineNuxtConfig({
 *   plugins: ['~/plugins/mycelix.ts']
 * })
 * ```
 *
 * @example plugins/mycelix.ts
 * ```ts
 * import { createMycelixNuxtPlugin } from '@mycelix/sdk/vue';
 *
 * export default createMycelixNuxtPlugin({
 *   conductorUrl: process.env.CONDUCTOR_URL || 'ws://localhost:8888',
 *   appId: 'my-nuxt-app',
 * });
 * ```
 */
export function createMycelixNuxtPlugin(options: MycelixPluginOptions) {
  return (nuxtApp: { vueApp: VueApp }) => {
    MycelixPlugin.install(nuxtApp.vueApp, options);
  };
}

// =============================================================================
// Type Exports
// =============================================================================

export type { Mycelix, MycelixConfig } from '../../core/index.js';
export type {
  UseMycelixReturn,
  UseIdentityReturn,
  UseGovernanceReturn,
  UseWalletReturn,
  UseQueryReturn,
  UseQueryOptions,
  UseMutationReturn,
  UseMutationOptions,
  Ref,
  DID,
  Proposal,
  ProposalStatus,
  VoteChoice,
  ConnectionStatus,
} from './composables.js';
