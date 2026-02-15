/**
 * Vue 3 Composables for Mycelix SDK
 *
 * Provides Vue 3 Composition API composables for all 8 Civilizational OS hApps.
 * Works with Vue 3.3+ and Nuxt 3.
 *
 * @packageDocumentation
 * @module vue
 */

// Re-export specialized composables for all 8 Civilizational OS hApp modules
export * from './identity.js';
export * from './knowledge.js';
export * from './governance.js';
export * from './justice.js';
export * from './finance.js';
export * from './property.js';
export * from './energy.js';
export * from './media.js';

// WalletBridge imports (used throughout)
import type { ReputationQueryResponse } from '../bridge/cross-happ.js';
import type {
  WalletBridge,
  TransactionRecommendation,
  VerificationStatus,
} from '../wallet/bridge-integration.js';

// WalletBridge instance storage (singleton for composables)
let walletBridgeInstance: WalletBridge | null = null;

// ============================================================================
// Types
// ============================================================================

/** Vue ref type (minimal interface for framework independence) */
export interface Ref<T> {
  value: T;
}

/** Readonly ref type */
export interface ReadonlyRef<T> {
  readonly value: T;
}

/** Query composable return type */
export interface UseQueryReturn<T> {
  data: Ref<T | undefined>;
  loading: Ref<boolean>;
  error: Ref<Error | undefined>;
  refetch: () => Promise<void>;
  cleanup: () => void;
}

/** Mutation composable return type */
export interface UseMutationReturn<T, V> {
  data: Ref<T | undefined>;
  loading: Ref<boolean>;
  error: Ref<Error | undefined>;
  mutate: (variables: V) => Promise<T>;
  reset: () => void;
}

/** Mycelix context for Vue provide/inject */
export interface MycelixContext {
  // Services
  identityService: any;
  financeService: any;
  propertyService: any;
  energyService: any;
  mediaService: any;
  governanceService: any;
  justiceService: any;
  knowledgeService: any;

  // Connection state
  connected: Ref<boolean>;
  connecting: Ref<boolean>;
  error: Ref<Error | undefined>;

  // Methods
  connect: () => Promise<void>;
  disconnect: () => Promise<void>;
}

/** Vue injection key for Mycelix context */
export const MYCELIX_INJECTION_KEY = Symbol('mycelix');

// ============================================================================
// Reactive Utilities (Framework-agnostic implementation)
// ============================================================================

type RefSubscriber<T> = (value: T) => void;

/**
 * Create a reactive ref (Vue-compatible)
 */
export function createRef<T>(initial: T): Ref<T> {
  let _value = initial;
  const subscribers = new Set<RefSubscriber<T>>();

  return {
    get value(): T {
      return _value;
    },
    set value(newValue: T) {
      _value = newValue;
      subscribers.forEach((sub) => sub(_value));
    },
  };
}

/**
 * Create a readonly ref
 */
export function createReadonlyRef<T>(source: Ref<T>): ReadonlyRef<T> {
  return {
    get value(): T {
      return source.value;
    },
  };
}

// ============================================================================
// Context Management
// ============================================================================

let globalContext: MycelixContext | null = null;

/**
 * Provide Mycelix context (use in app initialization)
 *
 * @example
 * ```ts
 * // In App.vue or main.ts
 * import { provideMycelix } from '@mycelix/sdk/vue';
 *
 * const ctx = provideMycelix({
 *   identityService: getIdentityService(),
 *   financeService: getFinanceService(),
 *   // ... other services
 * });
 * ```
 */
export function provideMycelix(ctx: MycelixContext): MycelixContext {
  globalContext = ctx;
  // Note: In real Vue app, you would also call provide(MYCELIX_INJECTION_KEY, ctx)
  return ctx;
}

/**
 * Use Mycelix context (requires prior provideMycelix call)
 *
 * @example
 * ```ts
 * // In any component
 * import { useMycelix } from '@mycelix/sdk/vue';
 *
 * const { identityService, connected } = useMycelix();
 * ```
 */
export function useMycelix(): MycelixContext {
  if (!globalContext) {
    throw new Error(
      'Mycelix context not provided. ' +
        'Call provideMycelix() in your app initialization. ' +
        'See: https://mycelix.net/docs/vue-setup'
    );
  }
  return globalContext;
}

// ============================================================================
// Query Composable Factory
// ============================================================================

export interface UseQueryOptions {
  /** Auto-fetch on mount (default: true) */
  enabled?: boolean;
  /** Refetch interval in ms (0 = disabled) */
  refetchInterval?: number;
  /** Cache time in ms */
  cacheTime?: number;
}

/**
 * Create a query composable
 *
 * @example
 * ```ts
 * const useProfile = createQueryComposable(
 *   async (did: string, ctx) => {
 *     return ctx.identityService.getProfile(did);
 *   }
 * );
 *
 * // In component
 * const { data, loading, error, refetch } = useProfile('did:mycelix:...');
 * ```
 */
export function createQueryComposable<T, V = void>(
  fetcher: (variables: V, ctx: MycelixContext) => Promise<T>
) {
  return function useQuery(variables: V, options?: UseQueryOptions): UseQueryReturn<T> {
    const data = createRef<T | undefined>(undefined);
    const loading = createRef(options?.enabled !== false);
    const error = createRef<Error | undefined>(undefined);
    let intervalId: NodeJS.Timeout | null = null;

    async function refetch(): Promise<void> {
      const ctx = useMycelix();
      loading.value = true;
      error.value = undefined;

      try {
        const result = await fetcher(variables, ctx);
        data.value = result;
      } catch (e) {
        error.value = e instanceof Error ? e : new Error(String(e));
      } finally {
        loading.value = false;
      }
    }

    // Auto-fetch on creation if enabled
    if (options?.enabled !== false) {
      refetch().catch(() => {
        // Error already stored in error ref
      });
    }

    // Setup refetch interval if specified
    if (options?.refetchInterval && options.refetchInterval > 0) {
      intervalId = setInterval(() => {
        refetch().catch(() => {});
      }, options.refetchInterval);
    }

    // Cleanup function to clear interval
    function cleanup(): void {
      if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
      }
    }

    return {
      data,
      loading,
      error,
      refetch,
      cleanup,
    };
  };
}

// ============================================================================
// Mutation Composable Factory
// ============================================================================

/**
 * Create a mutation composable
 *
 * @example
 * ```ts
 * const useCreateProfile = createMutationComposable(
 *   async (input: CreateProfileInput, ctx) => {
 *     return ctx.identityService.createProfile(input);
 *   }
 * );
 *
 * // In component
 * const { mutate, loading, error, data } = useCreateProfile();
 * await mutate({ displayName: 'Alice' });
 * ```
 */
export function createMutationComposable<T, V>(
  mutator: (variables: V, ctx: MycelixContext) => Promise<T>
) {
  return function useMutation(): UseMutationReturn<T, V> {
    const data = createRef<T | undefined>(undefined);
    const loading = createRef(false);
    const error = createRef<Error | undefined>(undefined);

    async function mutate(variables: V): Promise<T> {
      const ctx = useMycelix();
      loading.value = true;
      error.value = undefined;

      try {
        const result = await mutator(variables, ctx);
        data.value = result;
        return result;
      } catch (e) {
        const err = e instanceof Error ? e : new Error(String(e));
        error.value = err;
        throw err;
      } finally {
        loading.value = false;
      }
    }

    function reset(): void {
      data.value = undefined;
      loading.value = false;
      error.value = undefined;
    }

    return {
      data,
      loading,
      error,
      mutate,
      reset,
    };
  };
}

// ============================================================================
// Connection Composable
// ============================================================================

export interface UseConnectionReturn {
  connected: ReadonlyRef<boolean>;
  connecting: ReadonlyRef<boolean>;
  error: ReadonlyRef<Error | undefined>;
  connect: () => Promise<void>;
  disconnect: () => Promise<void>;
}

/**
 * Composable for managing Mycelix connection
 *
 * @example
 * ```ts
 * const { connected, connecting, connect, disconnect } = useConnection();
 *
 * if (!connected.value) {
 *   await connect();
 * }
 * ```
 */
export function useConnection(): UseConnectionReturn {
  const ctx = useMycelix();

  return {
    connected: createReadonlyRef(ctx.connected),
    connecting: createReadonlyRef(ctx.connecting),
    error: createReadonlyRef(ctx.error),
    connect: ctx.connect,
    disconnect: ctx.disconnect,
  };
}

// ============================================================================
// NOTE: Domain-specific composables are exported from submodules above.
// This file only contains cross-domain composables and shared utilities.
// ============================================================================

// ============================================================================
// Cross-Domain Composables
// ============================================================================

export interface UseRelatedEntitiesReturn extends UseQueryReturn<{
  did: string;
  assets: any[];
  wallets: any[];
  content: any[];
  proposals: any[];
  cases: any[];
  claims: any[];
}> {}

/**
 * Composable to fetch all entities related to a DID
 */
export function useRelatedEntities(_did: string): UseRelatedEntitiesReturn {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to fetch entity reputation
 *
 * @deprecated Use useReputationQuery for WalletBridge-based reputation queries
 * @see useReputationQuery
 */
export function useReputation(did: string, options?: UseQueryOptions): UseQueryReturn<any> {
  // If WalletBridge is available, delegate to the new implementation
  if (walletBridgeInstance) {
    const result = useReputationQuery(did, options);
    return {
      data: result.data as Ref<any>,
      loading: result.loading,
      error: result.error,
      refetch: result.refetch,
      cleanup: result.cleanup,
    };
  }

  // Fallback for non-WalletBridge usage (legacy)
  throw new Error(
    'useReputation requires WalletBridge. Call setWalletBridge() first, or use useReputationQuery directly.'
  );
}

// ============================================================================
// Workflow Composables
// ============================================================================

export interface WorkflowStep {
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  result?: any;
  error?: Error;
}

export interface UseWorkflowReturn<T> {
  result: Ref<T | undefined>;
  steps: Ref<WorkflowStep[]>;
  loading: Ref<boolean>;
  error: Ref<Error | undefined>;
  execute: () => Promise<T>;
  reset: () => void;
}

/**
 * Composable for executing cross-hApp workflows
 *
 * @example
 * ```ts
 * const { execute, steps, loading, result } = useWorkflow('lending', {
 *   borrowerId: 'did:mycelix:...',
 *   amount: 1000,
 *   collateralAssetId: 'asset_123'
 * });
 *
 * // Execute the workflow
 * const loan = await execute();
 *
 * // Watch workflow progress
 * watchEffect(() => {
 *   console.log('Steps:', steps.value);
 * });
 * ```
 */
export function useWorkflow<T>(
  _workflowType: string,
  _params: Record<string, unknown>
): UseWorkflowReturn<T> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Real-time Subscription Composables
// ============================================================================

export interface UseSubscriptionReturn<T> {
  data: Ref<T | undefined>;
  connected: Ref<boolean>;
  error: Ref<Error | undefined>;
  unsubscribe: () => void;
}

/**
 * Composable for real-time subscriptions via Bridge
 *
 * @example
 * ```ts
 * const { data, connected } = useSubscription('proposal_votes', {
 *   proposalId: 'prop_123'
 * });
 *
 * watchEffect(() => {
 *   if (data.value) {
 *     console.log('New vote:', data.value);
 *   }
 * });
 * ```
 */
export function useSubscription<T>(
  _eventType: string,
  _filter?: Record<string, unknown>
): UseSubscriptionReturn<T> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// WalletBridge Composables
// ============================================================================

/**
 * Set the WalletBridge instance for Vue composables
 *
 * @example
 * ```ts
 * import { setWalletBridge, createWalletBridge } from '@mycelix/sdk/vue';
 *
 * const bridge = createWalletBridge(financeProvider);
 * setWalletBridge(bridge);
 * ```
 */
export function setWalletBridge(bridge: WalletBridge): void {
  walletBridgeInstance = bridge;
}

/**
 * Get the current WalletBridge instance
 */
export function getWalletBridge(): WalletBridge | null {
  return walletBridgeInstance;
}

/**
 * Get the WalletBridge, throwing if not set
 */
function requireWalletBridge(): WalletBridge {
  if (!walletBridgeInstance) {
    throw new Error(
      'WalletBridge not set. Call setWalletBridge() first. ' +
        'See: https://mycelix.net/docs/vue-wallet-bridge'
    );
  }
  return walletBridgeInstance;
}

/** Wallet bridge composable return type */
export interface UseWalletBridgeReturn {
  getReputation: (did: string, forceRefresh?: boolean) => Promise<ReputationQueryResponse>;
  isTrustworthy: (did: string, threshold?: number) => Promise<boolean>;
  getRecommendation: (did: string, amount: number) => Promise<TransactionRecommendation>;
  reportPositive: (did: string, txId: string) => void;
  reportNegative: (did: string, reason: 'payment_failed' | 'dispute' | 'fraud_suspected') => void;
  subscribe: (did: string, handler: (rep: ReputationQueryResponse) => void) => () => void;
  verifyIdentity: (did: string) => Promise<VerificationStatus>;
  verifyCredential: (did: string, credentialType: string) => Promise<VerificationStatus>;
}

/**
 * Composable for direct WalletBridge access
 *
 * @example
 * ```ts
 * const { getReputation, reportPositive, isTrustworthy } = useWalletBridge();
 *
 * const rep = await getReputation('did:mycelix:...');
 * if (await isTrustworthy('did:mycelix:...', 0.7)) {
 *   // Proceed with transaction
 *   reportPositive('did:mycelix:...', 'tx-123');
 * }
 * ```
 */
export function useWalletBridge(): UseWalletBridgeReturn {
  const bridge = requireWalletBridge();

  return {
    getReputation: (did: string, forceRefresh?: boolean) =>
      bridge.getAggregateReputation(did, forceRefresh),
    isTrustworthy: (did: string, threshold = 0.6) => bridge.isTrustworthy(did, threshold),
    getRecommendation: (did: string, amount: number) =>
      bridge.getTransactionRecommendation(did, amount),
    reportPositive: (did: string, txId: string) => bridge.reportPositiveInteraction(did, txId),
    reportNegative: (
      did: string,
      reason: 'payment_failed' | 'dispute' | 'fraud_suspected'
    ) => bridge.reportNegativeInteraction(did, reason),
    subscribe: (did: string, handler: (rep: ReputationQueryResponse) => void) =>
      bridge.onReputationChange(did, handler),
    verifyIdentity: (did: string) => bridge.requestIdentityVerification(did),
    verifyCredential: (did: string, credentialType: string) =>
      bridge.verifyCredential(did, credentialType),
  };
}

/** Reputation query composable return type */
export interface UseReputationQueryReturn extends UseQueryReturn<ReputationQueryResponse> {
  isTrustworthy: Ref<boolean>;
  reportPositive: (txId: string) => void;
  reportNegative: (reason: 'payment_failed' | 'dispute' | 'fraud_suspected') => void;
}

/**
 * Composable for querying and tracking reputation
 *
 * @example
 * ```ts
 * const {
 *   data: reputation,
 *   loading,
 *   error,
 *   isTrustworthy,
 *   reportPositive,
 *   refetch
 * } = useReputationQuery('did:mycelix:...', { threshold: 0.7 });
 *
 * watchEffect(() => {
 *   if (reputation.value) {
 *     console.log('Score:', reputation.value.aggregatedScore);
 *   }
 * });
 * ```
 */
export function useReputationQuery(
  did: string,
  options?: UseQueryOptions & { threshold?: number }
): UseReputationQueryReturn {
  const bridge = requireWalletBridge();
  const threshold = options?.threshold ?? 0.6;

  const data = createRef<ReputationQueryResponse | undefined>(undefined);
  const loading = createRef(options?.enabled !== false);
  const error = createRef<Error | undefined>(undefined);
  const isTrustworthy = createRef(false);
  let intervalId: NodeJS.Timeout | null = null;
  let unsubscribe: (() => void) | null = null;

  async function refetch(): Promise<void> {
    loading.value = true;
    error.value = undefined;

    try {
      const result = await bridge.getAggregateReputation(did, true);
      data.value = result;
      isTrustworthy.value = result.aggregatedScore >= threshold;
    } catch (e) {
      error.value = e instanceof Error ? e : new Error(String(e));
    } finally {
      loading.value = false;
    }
  }

  function reportPositive(txId: string): void {
    bridge.reportPositiveInteraction(did, txId);
    // Refresh to get updated reputation
    refetch().catch(() => {});
  }

  function reportNegative(reason: 'payment_failed' | 'dispute' | 'fraud_suspected'): void {
    bridge.reportNegativeInteraction(did, reason);
    refetch().catch(() => {});
  }

  function cleanup(): void {
    if (intervalId) {
      clearInterval(intervalId);
      intervalId = null;
    }
    if (unsubscribe) {
      unsubscribe();
      unsubscribe = null;
    }
  }

  // Auto-fetch on creation if enabled
  if (options?.enabled !== false) {
    refetch().catch(() => {});
  }

  // Setup refetch interval if specified
  if (options?.refetchInterval && options.refetchInterval > 0) {
    intervalId = setInterval(() => {
      refetch().catch(() => {});
    }, options.refetchInterval);
  }

  // Subscribe to reputation changes
  unsubscribe = bridge.onReputationChange(did, (newRep) => {
    data.value = newRep;
    isTrustworthy.value = newRep.aggregatedScore >= threshold;
  });

  return {
    data,
    loading,
    error,
    isTrustworthy,
    refetch,
    cleanup,
    reportPositive,
    reportNegative,
  };
}

/** Transaction recommendation composable return type */
export interface UseTransactionRecommendationReturn
  extends UseQueryReturn<TransactionRecommendation> {
  shouldProceed: Ref<boolean>;
  formattedRisk: Ref<{ label: string; color: string; description: string }>;
}

/**
 * Composable for getting transaction recommendations
 *
 * @example
 * ```ts
 * const {
 *   data: recommendation,
 *   loading,
 *   shouldProceed,
 *   formattedRisk
 * } = useTransactionRecommendation('did:mycelix:...', 100);
 *
 * watchEffect(() => {
 *   if (recommendation.value) {
 *     console.log('Risk:', formattedRisk.value.label);
 *     console.log('Should proceed:', shouldProceed.value);
 *   }
 * });
 * ```
 */
export function useTransactionRecommendation(
  did: string,
  amount: number,
  options?: UseQueryOptions
): UseTransactionRecommendationReturn {
  const bridge = requireWalletBridge();

  const data = createRef<TransactionRecommendation | undefined>(undefined);
  const loading = createRef(options?.enabled !== false);
  const error = createRef<Error | undefined>(undefined);
  const shouldProceed = createRef(false);
  const formattedRisk = createRef<{ label: string; color: string; description: string }>({
    label: 'Unknown',
    color: 'gray',
    description: 'Loading...',
  });

  let intervalId: NodeJS.Timeout | null = null;

  async function refetch(): Promise<void> {
    loading.value = true;
    error.value = undefined;

    try {
      const result = await bridge.getTransactionRecommendation(did, amount);
      data.value = result;
      shouldProceed.value = result.riskLevel === 'low' || result.riskLevel === 'medium';
      formattedRisk.value = formatRiskLevel(result.riskLevel);
    } catch (e) {
      error.value = e instanceof Error ? e : new Error(String(e));
    } finally {
      loading.value = false;
    }
  }

  function cleanup(): void {
    if (intervalId) {
      clearInterval(intervalId);
      intervalId = null;
    }
  }

  // Auto-fetch on creation if enabled
  if (options?.enabled !== false) {
    refetch().catch(() => {});
  }

  // Setup refetch interval if specified
  if (options?.refetchInterval && options.refetchInterval > 0) {
    intervalId = setInterval(() => {
      refetch().catch(() => {});
    }, options.refetchInterval);
  }

  return {
    data,
    loading,
    error,
    shouldProceed,
    formattedRisk,
    refetch,
    cleanup,
  };
}

/**
 * Composable for identity verification mutations
 *
 * @example
 * ```ts
 * const { mutate: verifyIdentity, loading, data } = useVerifyIdentity();
 *
 * const status = await verifyIdentity({ did: 'did:mycelix:...' });
 * console.log('Verified:', status.verified);
 * ```
 */
export function useVerifyIdentity(): UseMutationReturn<VerificationStatus, { did: string }> {
  const bridge = requireWalletBridge();

  const data = createRef<VerificationStatus | undefined>(undefined);
  const loading = createRef(false);
  const error = createRef<Error | undefined>(undefined);

  async function mutate(variables: { did: string }): Promise<VerificationStatus> {
    loading.value = true;
    error.value = undefined;

    try {
      const result = await bridge.requestIdentityVerification(variables.did);
      data.value = result;
      return result;
    } catch (e) {
      const err = e instanceof Error ? e : new Error(String(e));
      error.value = err;
      throw err;
    } finally {
      loading.value = false;
    }
  }

  function reset(): void {
    data.value = undefined;
    loading.value = false;
    error.value = undefined;
  }

  return {
    data,
    loading,
    error,
    mutate,
    reset,
  };
}

/**
 * Composable for credential verification mutations
 *
 * @example
 * ```ts
 * const { mutate: verifyCredential, loading } = useVerifyCredential();
 *
 * const status = await verifyCredential({
 *   did: 'did:mycelix:...',
 *   credentialType: 'diploma'
 * });
 * ```
 */
export function useVerifyCredential(): UseMutationReturn<
  VerificationStatus,
  { did: string; credentialType: string }
> {
  const bridge = requireWalletBridge();

  const data = createRef<VerificationStatus | undefined>(undefined);
  const loading = createRef(false);
  const error = createRef<Error | undefined>(undefined);

  async function mutate(variables: {
    did: string;
    credentialType: string;
  }): Promise<VerificationStatus> {
    loading.value = true;
    error.value = undefined;

    try {
      const result = await bridge.verifyCredential(variables.did, variables.credentialType);
      data.value = result;
      return result;
    } catch (e) {
      const err = e instanceof Error ? e : new Error(String(e));
      error.value = err;
      throw err;
    } finally {
      loading.value = false;
    }
  }

  function reset(): void {
    data.value = undefined;
    loading.value = false;
    error.value = undefined;
  }

  return {
    data,
    loading,
    error,
    mutate,
    reset,
  };
}

// ============================================================================
// WalletBridge Formatting Utilities
// ============================================================================

/**
 * Format reputation score for display
 */
export function formatReputationScore(score: number): string {
  return (score * 100).toFixed(1) + '%';
}

/**
 * Get reputation badge info based on score
 */
export function getReputationBadge(score: number): {
  label: string;
  color: string;
  icon: string;
} {
  if (score >= 0.9) {
    return { label: 'Excellent', color: 'emerald', icon: 'star' };
  } else if (score >= 0.75) {
    return { label: 'Good', color: 'green', icon: 'check-circle' };
  } else if (score >= 0.6) {
    return { label: 'Fair', color: 'yellow', icon: 'info' };
  } else if (score >= 0.4) {
    return { label: 'Low', color: 'orange', icon: 'alert-triangle' };
  } else {
    return { label: 'Poor', color: 'red', icon: 'x-circle' };
  }
}

/**
 * Format risk level for display
 */
export function formatRiskLevel(riskLevel: TransactionRecommendation['riskLevel']): {
  label: string;
  color: string;
  description: string;
} {
  switch (riskLevel) {
    case 'low':
      return {
        label: 'Low Risk',
        color: 'green',
        description: 'Transaction appears safe to proceed',
      };
    case 'medium':
      return {
        label: 'Medium Risk',
        color: 'yellow',
        description: 'Consider additional verification',
      };
    case 'high':
      return {
        label: 'High Risk',
        color: 'orange',
        description: 'Exercise caution, verify identity first',
      };
    case 'very_high':
      return {
        label: 'Very High Risk',
        color: 'red',
        description: 'Not recommended without thorough verification',
      };
    default:
      return {
        label: 'Unknown',
        color: 'gray',
        description: 'Unable to assess risk',
      };
  }
}

// ============================================================================
// Vue 3 Implementation Guide
// ============================================================================

/**
 * Vue 3 Implementation Example:
 *
 * ```ts
 * // main.ts
 * import { createApp, ref, provide } from 'vue';
 * import { MYCELIX_INJECTION_KEY, provideMycelix } from '@mycelix/sdk/vue';
 * import {
 *   getIdentityService,
 *   getFinanceService,
 *   getPropertyService,
 *   getEnergyService,
 *   getMediaService,
 *   getGovernanceService,
 *   getJusticeService,
 *   getKnowledgeService,
 * } from '@mycelix/sdk';
 *
 * const app = createApp(App);
 *
 * // Setup Mycelix context
 * const ctx = {
 *   identityService: getIdentityService(),
 *   financeService: getFinanceService(),
 *   propertyService: getPropertyService(),
 *   energyService: getEnergyService(),
 *   mediaService: getMediaService(),
 *   governanceService: getGovernanceService(),
 *   justiceService: getJusticeService(),
 *   knowledgeService: getKnowledgeService(),
 *   connected: ref(false),
 *   connecting: ref(false),
 *   error: ref(undefined),
 *   connect: async () => { ... },
 *   disconnect: async () => { ... },
 * };
 *
 * app.provide(MYCELIX_INJECTION_KEY, ctx);
 * provideMycelix(ctx);
 *
 * app.mount('#app');
 * ```
 *
 * ```vue
 * <!-- Component.vue -->
 * <script setup lang="ts">
 * import { useIdentityProfile, useCreateProposal } from '@mycelix/sdk/vue';
 * import { watchEffect } from 'vue';
 *
 * const props = defineProps<{ did: string }>();
 *
 * // Query example
 * const { data: profile, loading, error, refetch } = useIdentityProfile(props.did);
 *
 * // Mutation example
 * const { mutate: createProposal, loading: creating } = useCreateProposal();
 *
 * async function handleCreate() {
 *   await createProposal({
 *     daoId: 'dao_123',
 *     title: 'New Proposal',
 *     description: 'Description...',
 *     proposerId: props.did,
 *     votingPeriodHours: 168,
 *     quorumPercentage: 51,
 *   });
 * }
 *
 * watchEffect(() => {
 *   if (profile.value) {
 *     console.log('Profile:', profile.value);
 *   }
 * });
 * </script>
 *
 * <template>
 *   <div v-if="loading">Loading...</div>
 *   <div v-else-if="error">Error: {{ error.message }}</div>
 *   <div v-else-if="profile">
 *     <h1>{{ profile.displayName }}</h1>
 *     <button @click="handleCreate" :disabled="creating">
 *       {{ creating ? 'Creating...' : 'Create Proposal' }}
 *     </button>
 *   </div>
 * </template>
 * ```
 */
