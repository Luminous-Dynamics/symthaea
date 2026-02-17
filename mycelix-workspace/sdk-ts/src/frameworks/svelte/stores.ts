/**
 * Svelte Stores for Mycelix SDK
 *
 * Provides Svelte-compatible stores for managing Mycelix state.
 *
 * @module frameworks/svelte/stores
 *
 * @example Basic Usage
 * ```svelte
 * <script>
 *   import { mycelix, identity, proposals } from '@mycelix/sdk/svelte';
 *   import { initMycelix } from '@mycelix/sdk/svelte';
 *   import { onMount } from 'svelte';
 *
 *   onMount(async () => {
 *     await initMycelix({
 *       conductorUrl: 'ws://localhost:8888',
 *       appId: 'my-app'
 *     });
 *   });
 * </script>
 *
 * {#if $mycelix}
 *   <p>Connected! DID: {$identity}</p>
 *   <p>Active Proposals: {$proposals.length}</p>
 * {:else}
 *   <p>Connecting...</p>
 * {/if}
 * ```
 */

import type { Mycelix, MycelixConfig } from '../../core/index.js';

// =============================================================================
// Types
// =============================================================================

/**
 * DID (Decentralized Identifier) type
 */
export type DID = string;

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
  executionPayload?: string;
  category?: string;
  createdAt: number;
  executedAt?: number;
}

/**
 * Svelte-compatible Readable store interface
 */
export interface Readable<T> {
  subscribe(run: (value: T) => void, invalidate?: () => void): () => void;
}

/**
 * Svelte-compatible Writable store interface
 */
export interface Writable<T> extends Readable<T> {
  set(value: T): void;
  update(updater: (value: T) => T): void;
}

/**
 * Connection state
 */
export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'error';

/**
 * Store with async state
 */
export interface AsyncStore<T> extends Readable<AsyncState<T>> {
  refetch(): Promise<void>;
}

/**
 * Async state wrapper
 */
export interface AsyncState<T> {
  data: T;
  loading: boolean;
  error: Error | null;
}

// =============================================================================
// Store Implementation
// =============================================================================

type Subscriber<T> = (value: T) => void;
type Unsubscriber = () => void;
type Invalidator = () => void;
type SubscribeInvalidateTuple<T> = [Subscriber<T>, Invalidator];

/**
 * Create a basic writable store
 */
function createWritable<T>(initial: T): Writable<T> {
  let value = initial;
  const subscribers = new Set<SubscribeInvalidateTuple<T>>();

  function set(newValue: T): void {
    if (value !== newValue) {
      value = newValue;
      subscribers.forEach(([run]) => run(value));
    }
  }

  function update(fn: (value: T) => T): void {
    set(fn(value));
  }

  function subscribe(run: Subscriber<T>, invalidate: Invalidator = () => {}): Unsubscriber {
    const subscriber: SubscribeInvalidateTuple<T> = [run, invalidate];
    subscribers.add(subscriber);
    run(value);

    return () => {
      subscribers.delete(subscriber);
    };
  }

  return { subscribe, set, update };
}

/**
 * Create a derived readable store
 */
function createDerived<S, T>(
  store: Readable<S>,
  derive: (value: S) => T
): Readable<T> {
  let derivedValue: T;

  return {
    subscribe(run: Subscriber<T>, invalidate?: Invalidator): Unsubscriber {
      return store.subscribe(
        (value) => {
          derivedValue = derive(value);
          run(derivedValue);
        },
        invalidate
      );
    },
  };
}

// =============================================================================
// Core Stores
// =============================================================================

// Internal state
const _mycelix = createWritable<Mycelix | null>(null);
const _connectionState = createWritable<ConnectionState>('disconnected');
const _connectionError = createWritable<Error | null>(null);
const _identity = createWritable<DID | null>(null);
const _proposals = createWritable<Proposal[]>([]);

/**
 * The Mycelix client store
 *
 * Contains the connected Mycelix client instance or null if not connected.
 *
 * @example
 * ```svelte
 * <script>
 *   import { mycelix } from '@mycelix/sdk/svelte';
 * </script>
 *
 * {#if $mycelix}
 *   <p>Connected to Mycelix!</p>
 *   <button on:click={() => $mycelix.disconnect()}>Disconnect</button>
 * {:else}
 *   <p>Not connected</p>
 * {/if}
 * ```
 */
export const mycelix: Writable<Mycelix | null> = _mycelix;

/**
 * Connection state store
 *
 * @example
 * ```svelte
 * <script>
 *   import { connectionState } from '@mycelix/sdk/svelte';
 * </script>
 *
 * {#if $connectionState === 'connecting'}
 *   <LoadingSpinner />
 * {:else if $connectionState === 'connected'}
 *   <Dashboard />
 * {:else if $connectionState === 'error'}
 *   <ErrorMessage />
 * {/if}
 * ```
 */
export const connectionState: Readable<ConnectionState> = _connectionState;

/**
 * Connection error store (null if no error)
 */
export const connectionError: Readable<Error | null> = _connectionError;

/**
 * Current user's DID store
 *
 * Contains the DID of the connected user or null if not available.
 *
 * @example
 * ```svelte
 * <script>
 *   import { identity } from '@mycelix/sdk/svelte';
 * </script>
 *
 * {#if $identity}
 *   <p>Your DID: {$identity}</p>
 * {:else}
 *   <button on:click={createIdentity}>Create Identity</button>
 * {/if}
 * ```
 */
export const identity: Readable<DID | null> = _identity;

/**
 * Active proposals store
 *
 * Contains a list of currently active governance proposals.
 *
 * @example
 * ```svelte
 * <script>
 *   import { proposals } from '@mycelix/sdk/svelte';
 * </script>
 *
 * <h2>Active Proposals ({$proposals.length})</h2>
 * <ul>
 *   {#each $proposals as proposal}
 *     <li>{proposal.title} - {proposal.status}</li>
 *   {/each}
 * </ul>
 * ```
 */
export const proposals: Readable<Proposal[]> = _proposals;

/**
 * Whether the client is connected
 */
export const isConnected: Readable<boolean> = createDerived(
  connectionState,
  (state) => state === 'connected'
);

/**
 * Whether the client is currently connecting
 */
export const isConnecting: Readable<boolean> = createDerived(
  connectionState,
  (state) => state === 'connecting'
);

// =============================================================================
// Initialization & Connection
// =============================================================================

/**
 * Initialize Mycelix and connect to Holochain
 *
 * This function should be called once during app initialization.
 *
 * @param config - Mycelix configuration
 * @returns The connected Mycelix client
 *
 * @example
 * ```svelte
 * <script>
 *   import { initMycelix, mycelix } from '@mycelix/sdk/svelte';
 *   import { onMount } from 'svelte';
 *
 *   onMount(async () => {
 *     try {
 *       await initMycelix({
 *         conductorUrl: 'ws://localhost:8888',
 *         appId: 'my-civilizational-os'
 *       });
 *     } catch (error) {
 *       console.error('Failed to connect:', error);
 *     }
 *   });
 * </script>
 * ```
 */
export async function initMycelix(config: MycelixConfig): Promise<Mycelix> {
  _connectionState.set('connecting');
  _connectionError.set(null);

  try {
    // Dynamic import to avoid bundling Holochain client
    const { Mycelix } = await import('../../core/index.js');
    const client = await Mycelix.connect(config);

    _mycelix.set(client);
    _connectionState.set('connected');

    // Fetch initial identity
    const didRecord = await client.getMyDid();
    const did = didRecord ? (typeof didRecord === 'string' ? didRecord : (didRecord as { did: string }).did) : null;
    _identity.set(did);

    // Set up signal handling for real-time updates
    client.onSignal((signal: unknown) => {
      handleSignal(signal);
    });

    return client;
  } catch (error) {
    const err = error instanceof Error ? error : new Error(String(error));
    _connectionError.set(err);
    _connectionState.set('error');
    throw err;
  }
}

/**
 * Disconnect from Mycelix
 */
export async function disconnectMycelix(): Promise<void> {
  let client: Mycelix | null = null;
  const unsubscribe = _mycelix.subscribe((m) => (client = m));
  unsubscribe();

  if (client !== null) {
    await (client as Mycelix).disconnect();
    _mycelix.set(null);
    _connectionState.set('disconnected');
    _identity.set(null);
    _proposals.set([]);
  }
}

/**
 * Reconnect to Mycelix
 *
 * @param config - Optional new configuration
 */
export async function reconnectMycelix(config?: MycelixConfig): Promise<Mycelix> {
  await disconnectMycelix();

  if (!config) {
    throw new Error('Configuration required for reconnection');
  }

  return initMycelix(config);
}

// =============================================================================
// Signal Handling
// =============================================================================

function handleSignal(signal: unknown): void {
  // Type guard for signal structure
  if (!signal || typeof signal !== 'object') return;

  const signalData = signal as Record<string, unknown>;

  // Handle different signal types
  if (signalData.type === 'proposal_created' || signalData.type === 'proposal_updated') {
    refreshProposals();
  } else if (signalData.type === 'identity_updated') {
    refreshIdentity();
  }
}

// =============================================================================
// Refresh Functions
// =============================================================================

/**
 * Refresh the current user's identity
 */
export async function refreshIdentity(): Promise<void> {
  let client: Mycelix | null = null;
  const unsubscribe = _mycelix.subscribe((m) => (client = m));
  unsubscribe();

  if (client !== null) {
    const didRecord = await (client as Mycelix).getMyDid();
    const did = didRecord ? (typeof didRecord === 'string' ? didRecord : (didRecord as unknown as { did: string }).did) : null;
    _identity.set(did);
  }
}

/**
 * Refresh the proposals list
 *
 * @param filter - Optional filter for proposals
 */
export async function refreshProposals(filter?: {
  daoId?: string;
  status?: ProposalStatus;
}): Promise<void> {
  let client: Mycelix | null = null;
  const unsubscribe = _mycelix.subscribe((m) => (client = m));
  unsubscribe();

  if (client !== null) {
    try {
      const query = filter ? { dao_id: filter.daoId || '', status_filter: filter.status } : { dao_id: '' };
      const result = await (client as Mycelix).governance.proposals.getProposals(query as unknown as { dao_id: string });
      _proposals.set(result as unknown as Proposal[]);
    } catch (error) {
      console.error('Failed to refresh proposals:', error);
    }
  }
}

// =============================================================================
// Wallet Stores
// =============================================================================

const _walletBalance = createWritable<bigint>(0n);
const _walletLoading = createWritable(false);

/**
 * Current wallet balance in primary currency (MYC)
 */
export const walletBalance: Readable<bigint> = _walletBalance;

/**
 * Whether wallet data is loading
 */
export const walletLoading: Readable<boolean> = _walletLoading;

/**
 * Refresh wallet balance
 *
 * @param currency - Currency to check (default: 'MYC')
 */
export async function refreshWalletBalance(currency: string = 'MYC'): Promise<void> {
  let client: Mycelix | null = null;
  const unsubscribe = _mycelix.subscribe((m) => (client = m));
  unsubscribe();

  if (client === null) return;

  _walletLoading.set(true);

  try {
    const didRecord = await (client as Mycelix).getMyDid();
    const did = didRecord ? (typeof didRecord === 'string' ? didRecord : (didRecord as unknown as { did: string }).did) : null;
    if (!did) {
      _walletBalance.set(0n);
      return;
    }

    const balances = await (client as Mycelix).finance.wallets.getAllBalances(did);
    const balance = balances?.[currency] || 0;
    _walletBalance.set(BigInt(balance));
  } catch (error) {
    console.error('Failed to refresh wallet balance:', error);
  } finally {
    _walletLoading.set(false);
  }
}

/**
 * Send tokens to another user
 *
 * @param to - Recipient DID
 * @param amount - Amount to send
 * @param currency - Currency (default: 'MYC')
 * @returns Transaction ID
 */
export async function sendTokens(
  to: DID,
  amount: bigint,
  currency: string = 'MYC'
): Promise<string> {
  let client: Mycelix | null = null;
  const unsubscribe = _mycelix.subscribe((m) => (client = m));
  unsubscribe();

  if (client === null) {
    throw new Error('Not connected to Mycelix');
  }

  const tx = await (client as Mycelix).finance.wallets.transfer({
    id: `tx-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    from_wallet: '', // Will be resolved from DID
    to_wallet: to,
    amount: Number(amount),
    currency,
  });
  const txId = (tx as { id: string })?.id || '';

  // Refresh balance after sending
  await refreshWalletBalance(currency);

  return txId;
}

// =============================================================================
// Identity Actions
// =============================================================================

/**
 * Create a new DID for the current user
 *
 * @returns The created DID
 *
 * @example
 * ```svelte
 * <script>
 *   import { createDid, identity } from '@mycelix/sdk/svelte';
 *
 *   async function handleCreateIdentity() {
 *     const did = await createDid();
 *     console.log('Created DID:', did);
 *   }
 * </script>
 *
 * {#if !$identity}
 *   <button on:click={handleCreateIdentity}>Create Identity</button>
 * {/if}
 * ```
 */
export async function createDid(): Promise<DID> {
  let client: Mycelix | null = null;
  const unsubscribe = _mycelix.subscribe((m) => (client = m));
  unsubscribe();

  if (client === null) {
    throw new Error('Not connected to Mycelix');
  }

  const didRecord = await (client as Mycelix).identity.did.createDid();
  const did = didRecord && typeof didRecord === 'object' ? (didRecord as unknown as { did: string }).did : String(didRecord);
  _identity.set(did);
  return did;
}

// =============================================================================
// Governance Actions
// =============================================================================

/**
 * Cast a vote on a proposal
 *
 * @param proposalId - ID of the proposal
 * @param choice - Vote choice ('approve', 'reject', 'abstain')
 * @param weight - Optional vote weight
 *
 * @example
 * ```svelte
 * <script>
 *   import { castVote } from '@mycelix/sdk/svelte';
 *
 *   async function handleVote(choice) {
 *     await castVote(proposal.id, choice);
 *   }
 * </script>
 *
 * <button on:click={() => handleVote('approve')}>Approve</button>
 * <button on:click={() => handleVote('reject')}>Reject</button>
 * ```
 */
export async function castVote(
  proposalId: string,
  choice: 'approve' | 'reject' | 'abstain',
  weight?: number
): Promise<void> {
  let client: Mycelix | null = null;
  let did: DID | null = null;

  const unsubscribe1 = _mycelix.subscribe((m) => (client = m));
  const unsubscribe2 = _identity.subscribe((d) => (did = d));
  unsubscribe1();
  unsubscribe2();

  if (client === null) {
    throw new Error('Not connected to Mycelix');
  }

  if (!did) {
    throw new Error('No identity found. Create a DID first.');
  }

  // Convert lowercase choice to governance zome format
  const voteChoice = choice === 'approve' ? 'Yes' : choice === 'reject' ? 'No' : 'Abstain';
  await (client as Mycelix).governance.voting.castVote({
    id: `vote-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    proposal_id: proposalId,
    voter_did: did,
    choice: voteChoice,
    weight: weight || 1,
  });

  // Refresh proposals to update vote counts
  await refreshProposals();
}

// =============================================================================
// Export Store Creation Utilities
// =============================================================================

export { createWritable as writable, createDerived as derived };
