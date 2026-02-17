/**
 * Vue 3 Composables for Mycelix SDK
 *
 * Provides Vue 3 Composition API composables for managing Mycelix state.
 *
 * @module frameworks/vue/composables
 *
 * @example Basic Usage
 * ```vue
 * <script setup lang="ts">
 * import { useMycelix, useIdentity, useGovernance } from '@mycelix/sdk/vue';
 *
 * const { mycelix, connect, status } = useMycelix();
 * const { did, loading: identityLoading } = useIdentity();
 * const { proposals, vote } = useGovernance();
 *
 * // Connect on mount
 * onMounted(async () => {
 *   await connect();
 * });
 * </script>
 *
 * <template>
 *   <div v-if="status === 'connected'">
 *     <p>Your DID: {{ did }}</p>
 *     <ProposalList :proposals="proposals" @vote="vote" />
 *   </div>
 *   <div v-else>Connecting...</div>
 * </template>
 * ```
 */

import type { Mycelix, MycelixConfig } from '../../core/index.js';

// =============================================================================
// Types
// =============================================================================

/**
 * Vue Ref interface (minimal for framework independence)
 */
export interface Ref<T> {
  value: T;
}

/**
 * Vue Readonly Ref interface
 */
export interface ReadonlyRef<T> {
  readonly value: T;
}

/**
 * DID (Decentralized Identifier) type
 */
export type DID = string;

/**
 * Connection status
 */
export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

/**
 * Proposal status
 */
export type ProposalStatus = 'draft' | 'active' | 'passed' | 'rejected' | 'executed' | 'expired';

/**
 * Proposal data structure
 */
export interface Proposal {
  id: string;
  title: string;
  description: string;
  proposerId: DID;
  daoId: string;
  status: ProposalStatus;
  votingEnds: number;
  quorumPercentage: number;
  approvesWeight: number;
  rejectsWeight: number;
  abstainWeight: number;
  category?: string;
  createdAt: number;
}

/**
 * Vote choice (matches governance zome types)
 */
export type VoteChoice = 'Yes' | 'No' | 'Abstain';

// =============================================================================
// Ref Implementation (for framework independence)
// =============================================================================

type Subscriber<T> = (value: T) => void;

/**
 * Create a reactive ref
 */
function createRef<T>(initial: T): Ref<T> & { _subscribe: (fn: Subscriber<T>) => () => void } {
  let _value = initial;
  const subscribers = new Set<Subscriber<T>>();

  return {
    get value(): T {
      return _value;
    },
    set value(newValue: T) {
      if (_value !== newValue) {
        _value = newValue;
        subscribers.forEach((sub) => sub(_value));
      }
    },
    _subscribe(fn: Subscriber<T>): () => void {
      subscribers.add(fn);
      return () => subscribers.delete(fn);
    },
  };
}

// =============================================================================
// Internal State
// =============================================================================

// Shared state across composables
let _mycelixInstance: Mycelix | null = null;
let _config: MycelixConfig | null = null;

const _status = createRef<ConnectionStatus>('disconnected');
const _error = createRef<Error | null>(null);
const _did = createRef<DID | null>(null);

// =============================================================================
// useMycelix Composable
// =============================================================================

/**
 * Return type for useMycelix composable
 */
export interface UseMycelixReturn {
  /** The Mycelix client instance (null if not connected) */
  mycelix: Ref<Mycelix | null>;
  /** Current connection status */
  status: Ref<ConnectionStatus>;
  /** Connection error (if any) */
  error: Ref<Error | null>;
  /** Connect to Holochain */
  connect: (config?: MycelixConfig) => Promise<void>;
  /** Disconnect from Holochain */
  disconnect: () => Promise<void>;
  /** Reconnect to Holochain */
  reconnect: () => Promise<void>;
}

/**
 * useMycelix - Main composable for Mycelix connection management
 *
 * @param config - Optional configuration (can also be passed to connect())
 * @returns Mycelix state and methods
 *
 * @example Basic usage
 * ```vue
 * <script setup lang="ts">
 * import { useMycelix } from '@mycelix/sdk/vue';
 * import { onMounted } from 'vue';
 *
 * const { mycelix, connect, status, error } = useMycelix();
 *
 * onMounted(async () => {
 *   await connect({
 *     conductorUrl: 'ws://localhost:8888',
 *     appId: 'my-app'
 *   });
 * });
 * </script>
 *
 * <template>
 *   <div v-if="status === 'connecting'">Connecting...</div>
 *   <div v-else-if="status === 'error'">Error: {{ error?.message }}</div>
 *   <div v-else-if="mycelix">Connected!</div>
 * </template>
 * ```
 *
 * @example With configuration
 * ```ts
 * const { connect } = useMycelix({
 *   conductorUrl: import.meta.env.VITE_CONDUCTOR_URL,
 *   appId: 'my-civilizational-app'
 * });
 *
 * // Connect will use the provided config
 * await connect();
 * ```
 */
export function useMycelix(config?: MycelixConfig): UseMycelixReturn {
  if (config) {
    _config = config;
  }

  const mycelixRef = createRef<Mycelix | null>(_mycelixInstance);

  // Keep ref in sync with instance
  const updateRef = () => {
    mycelixRef.value = _mycelixInstance;
  };

  const connect = async (connectConfig?: MycelixConfig): Promise<void> => {
    const cfg = connectConfig || _config;
    if (!cfg) {
      throw new Error('Configuration required. Pass config to useMycelix() or connect().');
    }

    _status.value = 'connecting';
    _error.value = null;

    try {
      const { Mycelix } = await import('../../core/index.js');
      const client = await Mycelix.connect(cfg);

      _mycelixInstance = client;
      _config = cfg;
      _status.value = 'connected';
      updateRef();

      // Fetch initial identity
      const didRecord = await client.getMyDid();
      _did.value = didRecord ? (typeof didRecord === 'string' ? didRecord : (didRecord as unknown as { did: string }).did) : null;

      // Set up signal handling
      client.onSignal((signal: unknown) => {
        handleSignal(signal);
      });
    } catch (err) {
      const connectError = err instanceof Error ? err : new Error(String(err));
      _error.value = connectError;
      _status.value = 'error';
      throw connectError;
    }
  };

  const disconnect = async (): Promise<void> => {
    if (_mycelixInstance) {
      await _mycelixInstance.disconnect();
      _mycelixInstance = null;
      _status.value = 'disconnected';
      _did.value = null;
      updateRef();
    }
  };

  const reconnect = async (): Promise<void> => {
    await disconnect();
    await connect();
  };

  return {
    mycelix: mycelixRef,
    status: _status,
    error: _error,
    connect,
    disconnect,
    reconnect,
  };
}

// =============================================================================
// Signal Handling
// =============================================================================

function handleSignal(signal: unknown): void {
  if (!signal || typeof signal !== 'object') return;
  // Handle signal types as needed
}

// =============================================================================
// useIdentity Composable
// =============================================================================

/**
 * Return type for useIdentity composable
 */
export interface UseIdentityReturn {
  /** User's DID (null if not created) */
  did: Ref<DID | null>;
  /** Whether identity is loading */
  loading: Ref<boolean>;
  /** Error if identity fetch failed */
  error: Ref<Error | null>;
  /** Create a new DID */
  createDid: () => Promise<DID>;
  /** Refresh identity data */
  refresh: () => Promise<void>;
}

/**
 * useIdentity - Composable for managing user identity
 *
 * @returns Identity state and methods
 *
 * @example
 * ```vue
 * <script setup lang="ts">
 * import { useIdentity } from '@mycelix/sdk/vue';
 *
 * const { did, loading, createDid } = useIdentity();
 * </script>
 *
 * <template>
 *   <div v-if="loading">Loading identity...</div>
 *   <div v-else-if="did">
 *     <p>Your DID: {{ did }}</p>
 *   </div>
 *   <button v-else @click="createDid">Create Identity</button>
 * </template>
 * ```
 */
export function useIdentity(): UseIdentityReturn {
  const loading = createRef(false);
  const error = createRef<Error | null>(null);

  const refresh = async (): Promise<void> => {
    if (!_mycelixInstance) return;

    loading.value = true;
    error.value = null;

    try {
      const didRecord = await _mycelixInstance.getMyDid();
      _did.value = didRecord ? (typeof didRecord === 'string' ? didRecord : (didRecord as unknown as { did: string }).did) : null;
    } catch (err) {
      error.value = err instanceof Error ? err : new Error(String(err));
    } finally {
      loading.value = false;
    }
  };

  const createDid = async (): Promise<DID> => {
    if (!_mycelixInstance) {
      throw new Error('Not connected to Mycelix');
    }

    loading.value = true;
    error.value = null;

    try {
      const didRecord = await _mycelixInstance.identity.did.createDid();
      const newDid = didRecord && typeof didRecord === 'object' ? (didRecord as unknown as { did: string }).did : String(didRecord);
      _did.value = newDid;
      return newDid;
    } catch (err) {
      const createError = err instanceof Error ? err : new Error(String(err));
      error.value = createError;
      throw createError;
    } finally {
      loading.value = false;
    }
  };

  return {
    did: _did,
    loading,
    error,
    createDid,
    refresh,
  };
}

// =============================================================================
// useGovernance Composable
// =============================================================================

/**
 * Return type for useGovernance composable
 */
export interface UseGovernanceReturn {
  /** List of proposals */
  proposals: Ref<Proposal[]>;
  /** Whether proposals are loading */
  loading: Ref<boolean>;
  /** Error if fetch failed */
  error: Ref<Error | null>;
  /** Cast a vote on a proposal */
  vote: (proposalId: string, choice: VoteChoice, weight?: number) => Promise<void>;
  /** Refresh proposals list */
  refresh: (filter?: { daoId?: string; status?: ProposalStatus }) => Promise<void>;
  /** Create a new proposal */
  createProposal: (input: {
    title: string;
    description: string;
    daoId: string;
    votingPeriodHours: number;
    quorumPercentage: number;
    category?: string;
  }) => Promise<string>;
}

/**
 * useGovernance - Composable for governance operations
 *
 * @param initialFilter - Optional initial filter for proposals
 * @returns Governance state and methods
 *
 * @example
 * ```vue
 * <script setup lang="ts">
 * import { useGovernance } from '@mycelix/sdk/vue';
 *
 * const { proposals, vote, loading, refresh } = useGovernance();
 *
 * // Refresh active proposals
 * onMounted(() => refresh({ status: 'active' }));
 *
 * const handleVote = async (proposalId: string, choice: 'approve' | 'reject') => {
 *   await vote(proposalId, choice);
 * };
 * </script>
 *
 * <template>
 *   <div v-if="loading">Loading proposals...</div>
 *   <ul v-else>
 *     <li v-for="proposal in proposals" :key="proposal.id">
 *       {{ proposal.title }}
 *       <button @click="handleVote(proposal.id, 'approve')">Approve</button>
 *       <button @click="handleVote(proposal.id, 'reject')">Reject</button>
 *     </li>
 *   </ul>
 * </template>
 * ```
 */
export function useGovernance(initialFilter?: {
  daoId?: string;
  status?: ProposalStatus;
}): UseGovernanceReturn {
  const proposals = createRef<Proposal[]>([]);
  const loading = createRef(false);
  const error = createRef<Error | null>(null);

  const refresh = async (filter?: { daoId?: string; status?: ProposalStatus }): Promise<void> => {
    if (!_mycelixInstance) return;

    loading.value = true;
    error.value = null;

    try {
      const f = filter || initialFilter || {};
      const result = await _mycelixInstance.governance.proposals.getProposals({
        dao_id: f.daoId || '',
        status_filter: f.status ? (f.status.charAt(0).toUpperCase() + f.status.slice(1)) as 'Draft' | 'Active' | 'Passed' | 'Failed' | 'Executed' | 'Vetoed' | 'Cancelled' : undefined,
      } as unknown as { dao_id: string });
      proposals.value = result as unknown as Proposal[];
    } catch (err) {
      error.value = err instanceof Error ? err : new Error(String(err));
    } finally {
      loading.value = false;
    }
  };

  const vote = async (
    proposalId: string,
    choice: VoteChoice,
    weight: number = 1
  ): Promise<void> => {
    if (!_mycelixInstance) {
      throw new Error('Not connected to Mycelix');
    }

    if (!_did.value) {
      throw new Error('No identity found. Create a DID first.');
    }

    await _mycelixInstance.governance.voting.castVote({
      id: `vote-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      proposal_id: proposalId,
      voter_did: _did.value,
      choice,
      weight,
    });

    // Refresh proposals to update vote counts
    await refresh();
  };

  const createProposal = async (input: {
    title: string;
    description: string;
    daoId: string;
    votingPeriodHours: number;
    quorumPercentage: number;
    category?: string;
  }): Promise<string> => {
    if (!_mycelixInstance) {
      throw new Error('Not connected to Mycelix');
    }

    if (!_did.value) {
      throw new Error('No identity found. Create a DID first.');
    }

    const proposal = await _mycelixInstance.governance.proposals.createProposal({
      id: `prop-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      dao_id: input.daoId,
      title: input.title,
      description: input.description,
      proposer_did: _did.value,
      proposal_type: 'Standard',
      quorum_percentage: input.quorumPercentage,
      approval_threshold: 50,
    });

    // Refresh proposals
    await refresh();

    return proposal.id;
  };

  // Auto-refresh if filter provided
  if (initialFilter && _mycelixInstance) {
    refresh();
  }

  return {
    proposals,
    loading,
    error,
    vote,
    refresh,
    createProposal,
  };
}

// =============================================================================
// useWallet Composable
// =============================================================================

/**
 * Return type for useWallet composable
 */
export interface UseWalletReturn {
  /** Current balance in primary currency */
  balance: Ref<bigint>;
  /** All balances by currency */
  balances: Ref<Record<string, bigint>>;
  /** Whether wallet is loading */
  loading: Ref<boolean>;
  /** Error if fetch failed */
  error: Ref<Error | null>;
  /** Send tokens to another user */
  send: (to: DID, amount: bigint, currency?: string) => Promise<string>;
  /** Refresh wallet balances */
  refresh: () => Promise<void>;
  /** Get balance for a specific currency */
  getBalance: (currency: string) => bigint;
}

/**
 * useWallet - Composable for wallet operations
 *
 * @param currency - Primary currency to track (default: 'MYC')
 * @returns Wallet state and methods
 *
 * @example
 * ```vue
 * <script setup lang="ts">
 * import { useWallet } from '@mycelix/sdk/vue';
 *
 * const { balance, send, loading, getBalance } = useWallet();
 *
 * const handleSend = async () => {
 *   await send('did:mycelix:alice', 100n);
 * };
 * </script>
 *
 * <template>
 *   <div v-if="loading">Loading wallet...</div>
 *   <div v-else>
 *     <p>Balance: {{ balance }} MYC</p>
 *     <p>Energy: {{ getBalance('ENERGY') }} credits</p>
 *     <button @click="handleSend">Send 100 MYC</button>
 *   </div>
 * </template>
 * ```
 */
export function useWallet(currency: string = 'MYC'): UseWalletReturn {
  const balance = createRef<bigint>(0n);
  const balances = createRef<Record<string, bigint>>({});
  const loading = createRef(false);
  const error = createRef<Error | null>(null);

  const refresh = async (): Promise<void> => {
    if (!_mycelixInstance || !_did.value) return;

    loading.value = true;
    error.value = null;

    try {
      const result = await _mycelixInstance.finance.wallets.getAllBalances(_did.value);
      const balanceMap: Record<string, bigint> = {};

      for (const [key, value] of Object.entries(result || {})) {
        balanceMap[key] = typeof value === 'bigint' ? value : BigInt(value);
      }

      balances.value = balanceMap;
      balance.value = balanceMap[currency] || 0n;
    } catch (err) {
      error.value = err instanceof Error ? err : new Error(String(err));
    } finally {
      loading.value = false;
    }
  };

  const send = async (
    to: DID,
    amount: bigint,
    curr: string = currency
  ): Promise<string> => {
    if (!_mycelixInstance) {
      throw new Error('Not connected to Mycelix');
    }

    const tx = await _mycelixInstance.finance.wallets.transfer({
      id: `tx-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      from_wallet: '', // Will be resolved
      to_wallet: to,
      amount: Number(amount),
      currency: curr,
    });
    const txId = (tx as { id: string })?.id || '';

    // Refresh balances
    await refresh();

    return txId;
  };

  const getBalance = (curr: string): bigint => {
    return balances.value[curr] || 0n;
  };

  // Auto-refresh if connected
  if (_mycelixInstance && _did.value) {
    refresh();
  }

  return {
    balance,
    balances,
    loading,
    error,
    send,
    refresh,
    getBalance,
  };
}

// =============================================================================
// useQuery Composable
// =============================================================================

/**
 * Options for useQuery composable
 */
export interface UseQueryOptions<T> {
  /** Whether to fetch immediately (default: true) */
  immediate?: boolean;
  /** Refetch interval in milliseconds */
  refetchInterval?: number;
  /** Initial data */
  initialData?: T;
  /** Callback on success */
  onSuccess?: (data: T) => void;
  /** Callback on error */
  onError?: (error: Error) => void;
}

/**
 * Return type for useQuery composable
 */
export interface UseQueryReturn<T> {
  data: Ref<T | undefined>;
  loading: Ref<boolean>;
  error: Ref<Error | null>;
  refetch: () => Promise<void>;
}

/**
 * useQuery - Generic query composable
 *
 * @param fetcher - Async function that fetches data
 * @param options - Query options
 * @returns Query state and methods
 *
 * @example
 * ```vue
 * <script setup lang="ts">
 * import { useMycelix, useQuery } from '@mycelix/sdk/vue';
 *
 * const { mycelix } = useMycelix();
 *
 * const { data: dao, loading, refetch } = useQuery(
 *   async () => mycelix.value?.governance.dao.getDao('my-dao-id'),
 *   { refetchInterval: 30000 }
 * );
 * </script>
 *
 * <template>
 *   <DAOCard v-if="dao" :dao="dao" />
 *   <LoadingSpinner v-else-if="loading" />
 * </template>
 * ```
 */
export function useQuery<T>(
  fetcher: () => Promise<T | undefined>,
  options: UseQueryOptions<T> = {}
): UseQueryReturn<T> {
  const { immediate = true, refetchInterval, initialData, onSuccess, onError } = options;

  const data = createRef<T | undefined>(initialData);
  const loading = createRef(immediate);
  const error = createRef<Error | null>(null);
  let intervalId: ReturnType<typeof setInterval> | null = null;

  const refetch = async (): Promise<void> => {
    loading.value = true;
    error.value = null;

    try {
      const result = await fetcher();
      data.value = result;
      if (result !== undefined) {
        onSuccess?.(result);
      }
    } catch (err) {
      const fetchError = err instanceof Error ? err : new Error(String(err));
      error.value = fetchError;
      onError?.(fetchError);
    } finally {
      loading.value = false;
    }
  };

  // Initial fetch
  if (immediate) {
    refetch();
  }

  // Set up interval
  if (refetchInterval && refetchInterval > 0) {
    intervalId = setInterval(refetch, refetchInterval);
  }

  // Cleanup function - would be called from onUnmounted in real Vue app
  // Store it for potential future use
  void function cleanup() {
    if (intervalId) {
      clearInterval(intervalId);
    }
  };

  return {
    data,
    loading,
    error,
    refetch,
  };
}

// =============================================================================
// useMutation Composable
// =============================================================================

/**
 * Options for useMutation composable
 */
export interface UseMutationOptions<T, V> {
  onSuccess?: (data: T, variables: V) => void;
  onError?: (error: Error, variables: V) => void;
}

/**
 * Return type for useMutation composable
 */
export interface UseMutationReturn<T, V> {
  data: Ref<T | undefined>;
  loading: Ref<boolean>;
  error: Ref<Error | null>;
  mutate: (variables: V) => Promise<T>;
  reset: () => void;
}

/**
 * useMutation - Generic mutation composable
 *
 * @param mutator - Async function that performs the mutation
 * @param options - Mutation options
 * @returns Mutation state and methods
 *
 * @example
 * ```vue
 * <script setup lang="ts">
 * import { useMutation, useMycelix } from '@mycelix/sdk/vue';
 *
 * const { mycelix } = useMycelix();
 *
 * const { mutate: createProposal, loading } = useMutation(
 *   async (input) => mycelix.value?.governance.proposals.createProposal(input),
 *   { onSuccess: () => toast.success('Proposal created!') }
 * );
 *
 * const handleSubmit = async (data) => {
 *   await createProposal(data);
 * };
 * </script>
 * ```
 */
export function useMutation<T, V>(
  mutator: (variables: V) => Promise<T>,
  options: UseMutationOptions<T, V> = {}
): UseMutationReturn<T, V> {
  const data = createRef<T | undefined>(undefined);
  const loading = createRef(false);
  const error = createRef<Error | null>(null);

  const mutate = async (variables: V): Promise<T> => {
    loading.value = true;
    error.value = null;

    try {
      const result = await mutator(variables);
      data.value = result;
      options.onSuccess?.(result, variables);
      return result;
    } catch (err) {
      const mutationError = err instanceof Error ? err : new Error(String(err));
      error.value = mutationError;
      options.onError?.(mutationError, variables);
      throw mutationError;
    } finally {
      loading.value = false;
    }
  };

  const reset = (): void => {
    data.value = undefined;
    error.value = null;
    loading.value = false;
  };

  return {
    data,
    loading,
    error,
    mutate,
    reset,
  };
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Get the current Mycelix instance (throws if not connected)
 */
export function requireMycelix(): Mycelix {
  if (!_mycelixInstance) {
    throw new Error('Not connected to Mycelix. Call connect() first.');
  }
  return _mycelixInstance;
}

/**
 * Check if Mycelix is connected
 */
export function isConnected(): boolean {
  return _status.value === 'connected' && _mycelixInstance !== null;
}
