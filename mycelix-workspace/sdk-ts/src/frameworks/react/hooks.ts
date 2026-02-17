/**
 * React Hooks for Mycelix SDK
 *
 * Provides React hooks for common Mycelix operations including identity,
 * governance, wallet, and proposals.
 *
 * @module frameworks/react/hooks
 *
 * @example Identity Hook
 * ```tsx
 * import { useIdentity } from '@mycelix/sdk/react';
 *
 * function Profile() {
 *   const { did, loading, error } = useIdentity();
 *
 *   if (loading) return <Spinner />;
 *   if (error) return <Error message={error.message} />;
 *   if (!did) return <CreateIdentityButton />;
 *
 *   return <div>Your DID: {did}</div>;
 * }
 * ```
 *
 * @example Proposals Hook
 * ```tsx
 * import { useProposals } from '@mycelix/sdk/react';
 *
 * function ProposalList({ daoId }) {
 *   const { proposals, loading, refetch } = useProposals({ daoId, status: 'active' });
 *
 *   return (
 *     <ul>
 *       {proposals.map(p => <li key={p.id}>{p.title}</li>)}
 *     </ul>
 *   );
 * }
 * ```
 */

import * as React from 'react';

import { useMycelix, useMycelixContext } from './provider.js';

import type { Mycelix } from '../../core/index.js';

// =============================================================================
// Types
// =============================================================================

/**
 * DID (Decentralized Identifier) type
 */
export type DID = string;

/**
 * Proposal status filter options
 */
export type ProposalStatus = 'draft' | 'active' | 'passed' | 'rejected' | 'executed' | 'expired';

/**
 * Filter options for proposals
 */
export interface ProposalFilter {
  /** Filter by DAO ID */
  daoId?: string;
  /** Filter by proposal status */
  status?: ProposalStatus;
  /** Filter by proposer DID */
  proposerId?: DID;
  /** Filter by category */
  category?: string;
  /** Maximum number of proposals to return */
  limit?: number;
  /** Offset for pagination */
  offset?: number;
}

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
 * Wallet balance data
 */
export interface WalletBalance {
  currency: string;
  amount: bigint;
  formatted: string;
}

/**
 * Base async state for hooks
 */
export interface AsyncState<T> {
  data: T;
  loading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
}

// =============================================================================
// Identity Hook
// =============================================================================

/**
 * Return type for useIdentity hook
 */
export interface UseIdentityReturn {
  /** The user's DID (null if not created yet) */
  did: DID | null;
  /** Whether the identity is being loaded */
  loading: boolean;
  /** Error if identity fetch failed */
  error: Error | null;
  /** Refetch the identity */
  refetch: () => Promise<void>;
  /** Create a new DID for the user */
  createDid: () => Promise<DID>;
}

/**
 * useIdentity - Hook for managing user identity
 *
 * Fetches the current user's DID and provides methods to create one.
 *
 * @returns Identity state and methods
 *
 * @example
 * ```tsx
 * function IdentityCard() {
 *   const { did, loading, error, createDid } = useIdentity();
 *
 *   if (loading) return <Skeleton />;
 *
 *   if (!did) {
 *     return (
 *       <button onClick={createDid}>
 *         Create Identity
 *       </button>
 *     );
 *   }
 *
 *   return (
 *     <div>
 *       <h2>Your Identity</h2>
 *       <code>{did}</code>
 *     </div>
 *   );
 * }
 * ```
 */
export function useIdentity(): UseIdentityReturn {
  const { mycelix, status } = useMycelixContext();
  const [did, setDid] = React.useState<DID | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<Error | null>(null);

  const fetchIdentity = React.useCallback(async () => {
    if (!mycelix || status !== 'connected') {
      setLoading(status === 'connecting');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const didRecord = await mycelix.getMyDid();
      setDid(didRecord ? (typeof didRecord === 'string' ? didRecord : (didRecord as unknown as { did: string }).did) : null);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setLoading(false);
    }
  }, [mycelix, status]);

  React.useEffect(() => {
    fetchIdentity();
  }, [fetchIdentity]);

  const createDid = React.useCallback(async (): Promise<DID> => {
    if (!mycelix) {
      throw new Error('Mycelix not connected');
    }

    setLoading(true);
    setError(null);

    try {
      const didRecord = await mycelix.identity.did.createDid();
      const newDid = didRecord && typeof didRecord === 'object' ? (didRecord as unknown as { did: string }).did : String(didRecord);
      setDid(newDid);
      return newDid;
    } catch (err) {
      const createError = err instanceof Error ? err : new Error(String(err));
      setError(createError);
      throw createError;
    } finally {
      setLoading(false);
    }
  }, [mycelix]);

  return {
    did,
    loading,
    error,
    refetch: fetchIdentity,
    createDid,
  };
}

// =============================================================================
// Proposals Hook
// =============================================================================

/**
 * Return type for useProposals hook
 */
export interface UseProposalsReturn {
  /** List of proposals matching the filter */
  proposals: Proposal[];
  /** Whether proposals are being loaded */
  loading: boolean;
  /** Error if fetch failed */
  error: Error | null;
  /** Refetch proposals */
  refetch: () => Promise<void>;
  /** Total count of matching proposals */
  total: number;
}

/**
 * useProposals - Hook for fetching governance proposals
 *
 * Fetches proposals based on filter criteria with pagination support.
 *
 * @param filter - Optional filter criteria
 * @returns Proposals state
 *
 * @example Active proposals for a DAO
 * ```tsx
 * function ActiveProposals({ daoId }) {
 *   const { proposals, loading, refetch } = useProposals({
 *     daoId,
 *     status: 'active',
 *     limit: 10,
 *   });
 *
 *   if (loading) return <ProposalsSkeleton />;
 *
 *   return (
 *     <div>
 *       <button onClick={refetch}>Refresh</button>
 *       {proposals.map(p => (
 *         <ProposalCard key={p.id} proposal={p} />
 *       ))}
 *     </div>
 *   );
 * }
 * ```
 *
 * @example All proposals by a user
 * ```tsx
 * function MyProposals({ userDid }) {
 *   const { proposals, total } = useProposals({ proposerId: userDid });
 *
 *   return <div>You have created {total} proposals</div>;
 * }
 * ```
 */
export function useProposals(filter?: ProposalFilter): UseProposalsReturn {
  const { mycelix, status } = useMycelixContext();
  const [proposals, setProposals] = React.useState<Proposal[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<Error | null>(null);
  const [total, setTotal] = React.useState(0);

  // Memoize filter to prevent unnecessary refetches
  const filterKey = React.useMemo(() => JSON.stringify(filter || {}), [filter]);

  const fetchProposals = React.useCallback(async () => {
    if (!mycelix || status !== 'connected') {
      setLoading(status === 'connecting');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Build query based on filter
      const query: Record<string, unknown> = {};
      if (filter?.daoId) query.daoId = filter.daoId;
      if (filter?.status) query.status = filter.status;
      if (filter?.proposerId) query.proposerId = filter.proposerId;
      if (filter?.category) query.category = filter.category;

      const result = await mycelix.governance.proposals.getProposals(query as unknown as { dao_id: string });

      // Apply client-side pagination if needed
      let filtered = result as unknown as Proposal[];
      if (filter?.offset || filter?.limit) {
        const start = filter.offset || 0;
        const end = filter.limit ? start + filter.limit : undefined;
        filtered = filtered.slice(start, end);
      }

      setProposals(filtered);
      setTotal(result.length);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
      setProposals([]);
    } finally {
      setLoading(false);
    }
  }, [mycelix, status, filterKey]);

  React.useEffect(() => {
    fetchProposals();
  }, [fetchProposals]);

  return {
    proposals,
    loading,
    error,
    refetch: fetchProposals,
    total,
  };
}

// =============================================================================
// Wallet Hook
// =============================================================================

/**
 * Return type for useWallet hook
 */
export interface UseWalletReturn {
  /** The user's wallet (null if not loaded) */
  wallet: {
    id: string;
    balances: Map<string, bigint>;
    did: DID;
  } | null;
  /** Current balance in primary currency */
  balance: bigint;
  /** Whether wallet is loading */
  loading: boolean;
  /** Error if wallet fetch failed */
  error: Error | null;
  /** Refetch wallet data */
  refetch: () => Promise<void>;
  /** Send tokens to another user */
  send: (to: DID, amount: bigint, currency?: string) => Promise<string>;
  /** Get balance for a specific currency */
  getBalance: (currency: string) => bigint;
}

/**
 * useWallet - Hook for managing user's wallet
 *
 * Provides access to wallet balance and transaction methods.
 *
 * @param currency - Primary currency to track (default: 'MYC')
 * @returns Wallet state and methods
 *
 * @example
 * ```tsx
 * function WalletCard() {
 *   const { balance, loading, send, getBalance } = useWallet();
 *
 *   const handleSend = async () => {
 *     const txId = await send('did:mycelix:alice', 100n);
 *     console.log('Transaction:', txId);
 *   };
 *
 *   if (loading) return <WalletSkeleton />;
 *
 *   return (
 *     <div>
 *       <h2>Balance: {balance.toString()} MYC</h2>
 *       <p>Energy Credits: {getBalance('ENERGY').toString()}</p>
 *       <button onClick={handleSend}>Send 100 MYC</button>
 *     </div>
 *   );
 * }
 * ```
 */
export function useWallet(currency: string = 'MYC'): UseWalletReturn {
  const { mycelix, status } = useMycelixContext();
  const [wallet, setWallet] = React.useState<UseWalletReturn['wallet']>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<Error | null>(null);

  const fetchWallet = React.useCallback(async () => {
    if (!mycelix || status !== 'connected') {
      setLoading(status === 'connecting');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const did = await mycelix.getMyDid();
      if (!did) {
        setWallet(null);
        setLoading(false);
        return;
      }

      // Fetch wallet balances from finance module
      const balances = await mycelix.finance.wallets.getAllBalances(did);

      const balanceMap = new Map<string, bigint>();
      for (const [key, value] of Object.entries(balances || {})) {
        balanceMap.set(key, BigInt(value));
      }
      setWallet({
        id: did,
        balances: balanceMap,
        did,
      });
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setLoading(false);
    }
  }, [mycelix, status]);

  React.useEffect(() => {
    fetchWallet();
  }, [fetchWallet]);

  const balance = React.useMemo(() => {
    return wallet?.balances.get(currency) || 0n;
  }, [wallet, currency]);

  const getBalance = React.useCallback(
    (curr: string): bigint => {
      return wallet?.balances.get(curr) || 0n;
    },
    [wallet]
  );

  const send = React.useCallback(
    async (to: DID, amount: bigint, curr: string = currency): Promise<string> => {
      if (!mycelix) {
        throw new Error('Mycelix not connected');
      }

      const tx = await mycelix.finance.wallets.transfer({
        id: `tx-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        from_wallet: wallet?.id || '',
        to_wallet: to,
        amount: Number(amount),
        currency: curr,
      });
      const txId = (tx as { id: string })?.id || '';

      // Refetch wallet to update balance
      await fetchWallet();

      return txId;
    },
    [mycelix, currency, fetchWallet]
  );

  return {
    wallet,
    balance,
    loading,
    error,
    refetch: fetchWallet,
    send,
    getBalance,
  };
}

// =============================================================================
// Voting Hook
// =============================================================================

/**
 * Vote choice options (matches governance zome types)
 */
export type VoteChoice = 'Yes' | 'No' | 'Abstain';

/**
 * Return type for useVote hook
 */
export interface UseVoteReturn {
  /** Cast a vote on a proposal */
  vote: (choice: VoteChoice, weight?: number) => Promise<void>;
  /** Whether a vote is being cast */
  loading: boolean;
  /** Error if vote failed */
  error: Error | null;
  /** Whether the user has already voted */
  hasVoted: boolean;
  /** The user's current vote (if any) */
  currentVote: VoteChoice | null;
}

/**
 * useVote - Hook for voting on a proposal
 *
 * @param proposalId - The proposal to vote on
 * @returns Vote state and methods
 *
 * @example
 * ```tsx
 * function VotingControls({ proposalId }) {
 *   const { vote, loading, hasVoted, currentVote, error } = useVote(proposalId);
 *
 *   if (hasVoted) {
 *     return <div>You voted: {currentVote}</div>;
 *   }
 *
 *   return (
 *     <div>
 *       <button onClick={() => vote('approve')} disabled={loading}>
 *         Approve
 *       </button>
 *       <button onClick={() => vote('reject')} disabled={loading}>
 *         Reject
 *       </button>
 *       <button onClick={() => vote('abstain')} disabled={loading}>
 *         Abstain
 *       </button>
 *       {error && <p>Error: {error.message}</p>}
 *     </div>
 *   );
 * }
 * ```
 */
export function useVote(proposalId: string): UseVoteReturn {
  const mycelix = useMycelix();
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<Error | null>(null);
  const [hasVoted, setHasVoted] = React.useState(false);
  const [currentVote, setCurrentVote] = React.useState<VoteChoice | null>(null);

  // Check if user has already voted
  React.useEffect(() => {
    const checkVote = async () => {
      try {
        const did = await mycelix.getMyDid();
        if (!did) return;

        const existingVote = await (mycelix.governance.voting as { getVoteByVoter?: (proposalId: string, did: string) => Promise<unknown> }).getVoteByVoter?.(proposalId, did);
        if (existingVote) {
          setHasVoted(true);
          const rawChoice = (existingVote as { choice: string }).choice;
          setCurrentVote(rawChoice as VoteChoice);
        }
      } catch {
        // User hasn't voted yet
      }
    };

    checkVote();
  }, [mycelix, proposalId]);

  const vote = React.useCallback(
    async (choice: VoteChoice, weight?: number) => {
      setLoading(true);
      setError(null);

      try {
        const did = await mycelix.getMyDid();
        if (!did) {
          throw new Error('No identity found. Create a DID first.');
        }

        await mycelix.governance.voting.castVote({
          id: `vote-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          proposal_id: proposalId,
          voter_did: did,
          choice,
          weight: weight || 1,
        });

        setHasVoted(true);
        setCurrentVote(choice);
      } catch (err) {
        const voteError = err instanceof Error ? err : new Error(String(err));
        setError(voteError);
        throw voteError;
      } finally {
        setLoading(false);
      }
    },
    [mycelix, proposalId]
  );

  return {
    vote,
    loading,
    error,
    hasVoted,
    currentVote,
  };
}

// =============================================================================
// Query Hook Factory
// =============================================================================

/**
 * Options for useQuery hook
 */
export interface UseQueryOptions<T> {
  /** Initial data before first fetch */
  initialData?: T;
  /** Whether to fetch on mount (default: true) */
  enabled?: boolean;
  /** Refetch interval in milliseconds */
  refetchInterval?: number;
  /** Callback on successful fetch */
  onSuccess?: (data: T) => void;
  /** Callback on error */
  onError?: (error: Error) => void;
}

/**
 * useQuery - Generic query hook for Mycelix data fetching
 *
 * @param fetcher - Function that performs the data fetch
 * @param deps - Dependencies that trigger refetch
 * @param options - Query options
 * @returns Query state
 *
 * @example
 * ```tsx
 * function DAODetails({ daoId }) {
 *   const { data: dao, loading, error, refetch } = useQuery(
 *     async (mycelix) => mycelix.governance.dao.getDao(daoId),
 *     [daoId],
 *     { refetchInterval: 30000 }
 *   );
 *
 *   if (loading) return <Skeleton />;
 *   if (error) return <ErrorMessage error={error} />;
 *
 *   return <DAOCard dao={dao} />;
 * }
 * ```
 */
export function useQuery<T>(
  fetcher: (mycelix: Mycelix) => Promise<T>,
  deps: React.DependencyList = [],
  options: UseQueryOptions<T> = {}
): AsyncState<T | undefined> {
  const { mycelix, status } = useMycelixContext();
  const [data, setData] = React.useState<T | undefined>(options.initialData);
  const [loading, setLoading] = React.useState(options.enabled !== false);
  const [error, setError] = React.useState<Error | null>(null);

  const fetch = React.useCallback(async () => {
    if (!mycelix || status !== 'connected') {
      setLoading(status === 'connecting');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await fetcher(mycelix);
      setData(result);
      options.onSuccess?.(result);
    } catch (err) {
      const fetchError = err instanceof Error ? err : new Error(String(err));
      setError(fetchError);
      options.onError?.(fetchError);
    } finally {
      setLoading(false);
    }
  }, [mycelix, status, fetcher, ...deps]);

  // Initial fetch
  React.useEffect(() => {
    if (options.enabled !== false) {
      fetch();
    }
  }, [fetch, options.enabled]);

  // Refetch interval
  React.useEffect(() => {
    if (!options.refetchInterval || options.enabled === false) return;

    const interval = setInterval(fetch, options.refetchInterval);
    return () => clearInterval(interval);
  }, [fetch, options.refetchInterval, options.enabled]);

  return {
    data,
    loading,
    error,
    refetch: fetch,
  };
}

// =============================================================================
// Mutation Hook Factory
// =============================================================================

/**
 * Options for useMutation hook
 */
export interface UseMutationOptions<T, V> {
  /** Callback on successful mutation */
  onSuccess?: (data: T, variables: V) => void;
  /** Callback on error */
  onError?: (error: Error, variables: V) => void;
  /** Callback when mutation settles (success or error) */
  onSettled?: (data: T | undefined, error: Error | null, variables: V) => void;
}

/**
 * Return type for useMutation hook
 */
export interface UseMutationReturn<T, V> {
  /** Execute the mutation */
  mutate: (variables: V) => Promise<T>;
  /** Mutation result data */
  data: T | undefined;
  /** Whether mutation is in progress */
  loading: boolean;
  /** Error if mutation failed */
  error: Error | null;
  /** Reset mutation state */
  reset: () => void;
}

/**
 * useMutation - Generic mutation hook for Mycelix operations
 *
 * @param mutator - Function that performs the mutation
 * @param options - Mutation options
 * @returns Mutation state and methods
 *
 * @example
 * ```tsx
 * function CreateProposalForm({ daoId }) {
 *   const { mutate, loading, error, reset } = useMutation(
 *     async (mycelix, input) => {
 *       return mycelix.governance.proposals.createProposal(input);
 *     },
 *     {
 *       onSuccess: (proposal) => {
 *         toast.success(`Proposal "${proposal.title}" created!`);
 *       },
 *     }
 *   );
 *
 *   const handleSubmit = async (data) => {
 *     await mutate({ ...data, daoId });
 *   };
 *
 *   return (
 *     <form onSubmit={handleSubmit}>
 *       {error && <ErrorBanner error={error} onDismiss={reset} />}
 *       <input name="title" required />
 *       <textarea name="description" />
 *       <button type="submit" disabled={loading}>
 *         {loading ? 'Creating...' : 'Create Proposal'}
 *       </button>
 *     </form>
 *   );
 * }
 * ```
 */
export function useMutation<T, V>(
  mutator: (mycelix: Mycelix, variables: V) => Promise<T>,
  options: UseMutationOptions<T, V> = {}
): UseMutationReturn<T, V> {
  const mycelix = useMycelix();
  const [data, setData] = React.useState<T | undefined>(undefined);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<Error | null>(null);

  const mutate = React.useCallback(
    async (variables: V): Promise<T> => {
      setLoading(true);
      setError(null);

      try {
        const result = await mutator(mycelix, variables);
        setData(result);
        options.onSuccess?.(result, variables);
        options.onSettled?.(result, null, variables);
        return result;
      } catch (err) {
        const mutationError = err instanceof Error ? err : new Error(String(err));
        setError(mutationError);
        options.onError?.(mutationError, variables);
        options.onSettled?.(undefined, mutationError, variables);
        throw mutationError;
      } finally {
        setLoading(false);
      }
    },
    [mycelix, mutator, options]
  );

  const reset = React.useCallback(() => {
    setData(undefined);
    setError(null);
    setLoading(false);
  }, []);

  return {
    mutate,
    data,
    loading,
    error,
    reset,
  };
}

// =============================================================================
// MFA (Multi-Factor Authentication) Hooks
// =============================================================================

/**
 * Factor types from MFDI spec
 */
export type FactorType =
  | 'PrimaryKeyPair'
  | 'HardwareKey'
  | 'Biometric'
  | 'SocialRecovery'
  | 'ReputationAttestation'
  | 'GitcoinPassport'
  | 'VerifiableCredential'
  | 'RecoveryPhrase'
  | 'SecurityQuestions';

/**
 * Factor categories aligned with MFDI spec
 */
export type FactorCategory =
  | 'Cryptographic'
  | 'Biometric'
  | 'SocialProof'
  | 'ExternalVerification'
  | 'Knowledge';

/**
 * Assurance levels aligned with Epistemic E-Axis
 */
export type AssuranceLevel =
  | 'Anonymous'
  | 'Basic'
  | 'Verified'
  | 'HighlyAssured'
  | 'ConstitutionallyCritical';

/**
 * Enrolled factor data
 */
export interface EnrolledFactor {
  factorType: FactorType;
  factorId: string;
  enrolledAt: number;
  lastVerified: number;
  metadata: string;
  effectiveStrength: number;
  active: boolean;
}

/**
 * MFA state for a DID
 */
export interface MfaState {
  did: DID;
  owner: string;
  factors: EnrolledFactor[];
  assuranceLevel: AssuranceLevel;
  effectiveStrength: number;
  categoryCount: number;
  created: number;
  updated: number;
  version: number;
}

// =============================================================================
// API Type Adapters
// =============================================================================

/**
 * Adapts an EnrolledFactor from the MFA zome API to the hooks' view model.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function adaptEnrolledFactor(apiFactor: any): EnrolledFactor {
  return {
    factorType: apiFactor.factorType ?? apiFactor.factor_type,
    factorId: apiFactor.factorId ?? apiFactor.id ?? '',
    enrolledAt: apiFactor.enrolledAt ?? apiFactor.createdAt ?? 0,
    lastVerified: apiFactor.lastVerified ?? apiFactor.lastUsedAt ?? 0,
    metadata: typeof apiFactor.metadata === 'string'
      ? apiFactor.metadata
      : JSON.stringify(apiFactor.metadata ?? {}),
    effectiveStrength: apiFactor.effectiveStrength ?? (apiFactor.verified ? 1.0 : 0.0),
    active: apiFactor.active ?? apiFactor.verified ?? false,
  };
}

/**
 * Adapts an MfaState from the MFA zome API to the hooks' view model.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function adaptMfaState(apiState: any): MfaState {
  const factors = (apiState.factors ?? []).map(adaptEnrolledFactor);
  return {
    did: apiState.did ?? '',
    owner: apiState.owner ?? '',
    factors,
    assuranceLevel: apiState.assuranceLevel ?? 'Anonymous',
    effectiveStrength: apiState.effectiveStrength ?? factors.reduce(
      (sum: number, f: EnrolledFactor) => sum + f.effectiveStrength, 0
    ),
    categoryCount: apiState.categoryCount ?? new Set(factors.map(
      (f: EnrolledFactor) => f.factorType
    )).size,
    created: apiState.created ?? 0,
    updated: apiState.updated ?? 0,
    version: apiState.version ?? 1,
  };
}

/**
 * Assurance calculation output
 */
export interface AssuranceOutput {
  level: AssuranceLevel;
  score: number;
}

/**
 * FL eligibility result
 */
export interface FlEligibilityResult {
  eligible: boolean;
  requirements: FlRequirement[];
  currentScore: number;
  requiredScore: number;
}

/**
 * FL requirement
 */
export interface FlRequirement {
  name: string;
  met: boolean;
  description: string;
}

/**
 * Factor enrollment history entry
 */
export interface FactorEnrollment {
  factorId: string;
  factorType: FactorType;
  action: 'enrolled' | 'removed' | 'verified' | 'failed';
  timestamp: number;
  metadata?: Record<string, unknown>;
}

/**
 * Factor enrollment input for hooks
 */
export interface EnrollFactorInput {
  factorType: FactorType;
  metadata?: Record<string, unknown>;
}

/**
 * Return type for useMfaState hook
 */
export interface UseMfaStateReturn {
  /** Current MFA state (null if not set up) */
  mfaState: MfaState | null;
  /** Whether MFA state is loading */
  loading: boolean;
  /** Error if fetch failed */
  error: Error | null;
  /** Refetch MFA state */
  refetch: () => Promise<void>;
  /** Whether MFA is set up for this DID */
  hasMfa: boolean;
  /** Current assurance level */
  assuranceLevel: AssuranceLevel | null;
  /** Number of enrolled factors */
  factorCount: number;
}

/**
 * useMfaState - Hook for fetching MFA state for a DID
 *
 * @param did - The DID to fetch MFA state for (uses current user's DID if not provided)
 * @returns MFA state and metadata
 *
 * @example
 * ```tsx
 * function MfaStatusCard() {
 *   const { mfaState, loading, assuranceLevel, factorCount } = useMfaState();
 *
 *   if (loading) return <Skeleton />;
 *
 *   if (!mfaState) {
 *     return <SetupMfaPrompt />;
 *   }
 *
 *   return (
 *     <div>
 *       <h2>Identity Assurance: {assuranceLevel}</h2>
 *       <p>{factorCount} factors enrolled</p>
 *     </div>
 *   );
 * }
 * ```
 */
export function useMfaState(did?: DID): UseMfaStateReturn {
  const { mycelix, status } = useMycelixContext();
  const [mfaState, setMfaState] = React.useState<MfaState | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<Error | null>(null);
  const [targetDid, setTargetDid] = React.useState<DID | null>(did || null);

  // Fetch the user's DID if not provided
  React.useEffect(() => {
    if (did) {
      setTargetDid(did);
      return;
    }

    const fetchDid = async () => {
      if (!mycelix || status !== 'connected') return;
      try {
        const userDid = await mycelix.getMyDid();
        setTargetDid(userDid || null);
      } catch {
        setTargetDid(null);
      }
    };

    fetchDid();
  }, [mycelix, status, did]);

  const fetchMfaState = React.useCallback(async () => {
    if (!mycelix || status !== 'connected' || !targetDid) {
      setLoading(status === 'connecting');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const state = await mycelix.identity.mfa.getMfaState(targetDid);
      setMfaState(state ? adaptMfaState(state) : null);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
      setMfaState(null);
    } finally {
      setLoading(false);
    }
  }, [mycelix, status, targetDid]);

  React.useEffect(() => {
    fetchMfaState();
  }, [fetchMfaState]);

  const hasMfa = mfaState !== null;
  const assuranceLevel = mfaState?.assuranceLevel || null;
  const factorCount = mfaState?.factors.length || 0;

  return {
    mfaState,
    loading,
    error,
    refetch: fetchMfaState,
    hasMfa,
    assuranceLevel,
    factorCount,
  };
}

/**
 * Return type for useMfaEnroll hook
 */
export interface UseMfaEnrollReturn {
  /** Enroll a new factor */
  enrollFactor: (input: EnrollFactorInput) => Promise<EnrolledFactor>;
  /** Create initial MFA state (if none exists) */
  createMfaState: (primaryKeyHash: string) => Promise<MfaState>;
  /** Remove an existing factor */
  removeFactor: (factorId: string) => Promise<MfaState>;
  /** Whether an operation is in progress */
  loading: boolean;
  /** Error if operation failed */
  error: Error | null;
  /** Reset error state */
  reset: () => void;
}

/**
 * useMfaEnroll - Hook for enrolling and managing MFA factors
 *
 * @param did - The DID to manage MFA for (uses current user's DID if not provided)
 * @returns MFA management methods
 *
 * @example
 * ```tsx
 * function EnrollFactorForm() {
 *   const { enrollFactor, loading, error, reset } = useMfaEnroll();
 *
 *   const handleEnrollHardwareKey = async () => {
 *     const attestation = await getWebAuthnAttestation();
 *     await enrollFactor({
 *       factorType: 'HardwareKey',
 *       factorId: attestation.credentialId,
 *       metadata: JSON.stringify(attestation),
 *       reason: 'Added YubiKey',
 *     });
 *   };
 *
 *   return (
 *     <div>
 *       <button onClick={handleEnrollHardwareKey} disabled={loading}>
 *         {loading ? 'Enrolling...' : 'Add Hardware Key'}
 *       </button>
 *       {error && <ErrorBanner error={error} onDismiss={reset} />}
 *     </div>
 *   );
 * }
 * ```
 */
export function useMfaEnroll(did?: DID): UseMfaEnrollReturn {
  const { mycelix, status } = useMycelixContext();
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<Error | null>(null);
  const [targetDid, setTargetDid] = React.useState<DID | null>(did || null);

  // Fetch the user's DID if not provided
  React.useEffect(() => {
    if (did) {
      setTargetDid(did);
      return;
    }

    const fetchDid = async () => {
      if (!mycelix || status !== 'connected') return;
      try {
        const userDid = await mycelix.getMyDid();
        setTargetDid(userDid || null);
      } catch {
        setTargetDid(null);
      }
    };

    fetchDid();
  }, [mycelix, status, did]);

  const createMfaState = React.useCallback(
    async (primaryKeyHash: string): Promise<MfaState> => {
      if (!mycelix || !targetDid) {
        throw new Error('Mycelix not connected or DID not available');
      }

      setLoading(true);
      setError(null);

      try {
        const result = await mycelix.identity.mfa.createMfaState(targetDid, primaryKeyHash);
        return adaptMfaState(result);
      } catch (err) {
        const enrollError = err instanceof Error ? err : new Error(String(err));
        setError(enrollError);
        throw enrollError;
      } finally {
        setLoading(false);
      }
    },
    [mycelix, targetDid]
  );

  const enrollFactor = React.useCallback(
    async (input: EnrollFactorInput): Promise<EnrolledFactor> => {
      if (!mycelix || !targetDid) {
        throw new Error('Mycelix not connected or DID not available');
      }

      setLoading(true);
      setError(null);

      try {
        const result = await mycelix.identity.mfa.enrollFactor(targetDid, input);
        return adaptEnrolledFactor(result);
      } catch (err) {
        const enrollError = err instanceof Error ? err : new Error(String(err));
        setError(enrollError);
        throw enrollError;
      } finally {
        setLoading(false);
      }
    },
    [mycelix, targetDid]
  );

  const removeFactor = React.useCallback(
    async (factorId: string): Promise<MfaState> => {
      if (!mycelix || !targetDid) {
        throw new Error('Mycelix not connected or DID not available');
      }

      setLoading(true);
      setError(null);

      try {
        const result = await mycelix.identity.mfa.removeFactor(targetDid, factorId);
        return adaptMfaState(result);
      } catch (err) {
        const removeError = err instanceof Error ? err : new Error(String(err));
        setError(removeError);
        throw removeError;
      } finally {
        setLoading(false);
      }
    },
    [mycelix, targetDid]
  );

  const reset = React.useCallback(() => {
    setError(null);
    setLoading(false);
  }, []);

  return {
    enrollFactor,
    createMfaState,
    removeFactor,
    loading,
    error,
    reset,
  };
}

/**
 * Verification proof type for hooks (simplified)
 */
export type VerificationProof = { type: string; [key: string]: unknown };

/**
 * MFA verification result
 */
export interface MfaVerificationResult {
  success: boolean;
  factorId: string;
  verifiedAt: number;
  newAssuranceLevel?: AssuranceLevel;
}

/**
 * Return type for useMfaVerify hook
 */
export interface UseMfaVerifyReturn {
  /** Generate a challenge for a factor */
  generateChallenge: (factorId: string) => Promise<VerificationChallenge>;
  /** Verify a factor with proof */
  verifyFactor: (factorId: string, proof: VerificationProof) => Promise<MfaVerificationResult>;
  /** Whether verification is in progress */
  loading: boolean;
  /** Error if verification failed */
  error: Error | null;
  /** Reset error state */
  reset: () => void;
}

/**
 * Verification challenge from MFA zome
 */
export interface VerificationChallenge {
  challengeId: string;
  factorId: string;
  factorType: FactorType;
  challenge: string;
  expiresAt: number;
}

/**
 * useMfaVerify - Hook for verifying MFA factors
 *
 * Verification resets the decay timer on a factor, maintaining its full strength.
 *
 * @param did - The DID to verify factors for (uses current user's DID if not provided)
 * @returns Verification methods
 *
 * @example
 * ```tsx
 * function VerifyFactorButton({ factorId }) {
 *   const { verifyFactor, loading, error } = useMfaVerify();
 *
 *   const handleVerify = async () => {
 *     await verifyFactor(factorId);
 *     toast.success('Factor verified successfully');
 *   };
 *
 *   return (
 *     <button onClick={handleVerify} disabled={loading}>
 *       {loading ? 'Verifying...' : 'Verify Factor'}
 *     </button>
 *   );
 * }
 * ```
 */
export function useMfaVerify(did?: DID): UseMfaVerifyReturn {
  const { mycelix, status } = useMycelixContext();
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<Error | null>(null);
  const [targetDid, setTargetDid] = React.useState<DID | null>(did || null);

  // Fetch the user's DID if not provided
  React.useEffect(() => {
    if (did) {
      setTargetDid(did);
      return;
    }

    const fetchDid = async () => {
      if (!mycelix || status !== 'connected') return;
      try {
        const userDid = await mycelix.getMyDid();
        setTargetDid(userDid || null);
      } catch {
        setTargetDid(null);
      }
    };

    fetchDid();
  }, [mycelix, status, did]);

  const generateChallenge = React.useCallback(
    async (factorId: string): Promise<VerificationChallenge> => {
      if (!mycelix || !targetDid) {
        throw new Error('Mycelix not connected or DID not available');
      }

      setLoading(true);
      setError(null);

      try {
        const result = await mycelix.identity.mfa.generateChallenge(targetDid, factorId);
        return result;
      } catch (err) {
        const challengeError = err instanceof Error ? err : new Error(String(err));
        setError(challengeError);
        throw challengeError;
      } finally {
        setLoading(false);
      }
    },
    [mycelix, targetDid]
  );

  const verifyFactor = React.useCallback(
    async (factorId: string, proof: VerificationProof): Promise<MfaVerificationResult> => {
      if (!mycelix || !targetDid) {
        throw new Error('Mycelix not connected or DID not available');
      }

      setLoading(true);
      setError(null);

      try {
        const result = await mycelix.identity.mfa.verifyFactor(targetDid, factorId, proof as never);
        return result;
      } catch (err) {
        const verifyError = err instanceof Error ? err : new Error(String(err));
        setError(verifyError);
        throw verifyError;
      } finally {
        setLoading(false);
      }
    },
    [mycelix, targetDid]
  );

  const reset = React.useCallback(() => {
    setError(null);
    setLoading(false);
  }, []);

  return {
    generateChallenge,
    verifyFactor,
    loading,
    error,
    reset,
  };
}

/**
 * Return type for useFlEligibility hook
 */
export interface UseFlEligibilityReturn {
  /** FL eligibility result */
  eligibility: FlEligibilityResult | null;
  /** Whether eligibility is loading */
  loading: boolean;
  /** Error if check failed */
  error: Error | null;
  /** Refetch eligibility */
  refetch: () => Promise<void>;
  /** Whether the DID is eligible for FL participation */
  isEligible: boolean;
  /** Requirements status */
  requirements: FlRequirement[];
  /** Unmet requirements (those not yet satisfied) */
  unmetRequirements: FlRequirement[];
}

/**
 * useFlEligibility - Hook for checking Federated Learning eligibility
 *
 * FL participation requires:
 * - Verified assurance level or higher
 * - At least one Cryptographic factor
 * - At least one ExternalVerification factor (Gitcoin Passport or VC)
 *
 * @param did - The DID to check (uses current user's DID if not provided)
 * @returns FL eligibility state
 *
 * @example
 * ```tsx
 * function FlEligibilityBanner() {
 *   const { isEligible, unmetRequirements, loading } = useFlEligibility();
 *
 *   if (loading) return <Skeleton />;
 *
 *   if (isEligible) {
 *     return (
 *       <SuccessBanner>
 *         You are eligible to participate in Federated Learning!
 *       </SuccessBanner>
 *     );
 *   }
 *
 *   return (
 *     <WarningBanner>
 *       <h3>Not yet eligible for FL</h3>
 *       <ul>
 *         {unmetRequirements.map((req, i) => (
 *           <li key={i}>{req.description}</li>
 *         ))}
 *       </ul>
 *     </WarningBanner>
 *   );
 * }
 * ```
 */
export function useFlEligibility(did?: DID): UseFlEligibilityReturn {
  const { mycelix, status } = useMycelixContext();
  const [eligibility, setEligibility] = React.useState<FlEligibilityResult | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<Error | null>(null);
  const [targetDid, setTargetDid] = React.useState<DID | null>(did || null);

  // Fetch the user's DID if not provided
  React.useEffect(() => {
    if (did) {
      setTargetDid(did);
      return;
    }

    const fetchDid = async () => {
      if (!mycelix || status !== 'connected') return;
      try {
        const userDid = await mycelix.getMyDid();
        setTargetDid(userDid || null);
      } catch {
        setTargetDid(null);
      }
    };

    fetchDid();
  }, [mycelix, status, did]);

  const fetchEligibility = React.useCallback(async () => {
    if (!mycelix || status !== 'connected' || !targetDid) {
      setLoading(status === 'connecting');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await mycelix.identity.mfa.checkFlEligibility(targetDid);
      setEligibility(result);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
      setEligibility(null);
    } finally {
      setLoading(false);
    }
  }, [mycelix, status, targetDid]);

  React.useEffect(() => {
    fetchEligibility();
  }, [fetchEligibility]);

  const isEligible = eligibility?.eligible || false;
  const requirements = eligibility?.requirements || [];
  const unmetRequirements = requirements.filter(r => !r.met);

  return {
    eligibility,
    loading,
    error,
    refetch: fetchEligibility,
    isEligible,
    requirements,
    unmetRequirements,
  };
}

/**
 * Return type for useAssuranceLevel hook
 */
export interface UseAssuranceLevelReturn {
  /** Current assurance level */
  level: AssuranceLevel | null;
  /** Numeric score (0.0-1.0) */
  score: number;
  /** Whether loading */
  loading: boolean;
  /** Error if fetch failed */
  error: Error | null;
  /** Refetch assurance */
  refetch: () => Promise<void>;
}

/**
 * useAssuranceLevel - Hook for getting current assurance level
 *
 * A lightweight hook that only fetches the assurance calculation,
 * not the full MFA state.
 *
 * @param did - The DID to check (uses current user's DID if not provided)
 * @returns Assurance level state
 *
 * @example
 * ```tsx
 * function AssuranceBadge() {
 *   const { level, score } = useAssuranceLevel();
 *
 *   return (
 *     <div>
 *       <Badge variant={level}>{level}</Badge>
 *       <span>{(score * 100).toFixed(0)}%</span>
 *     </div>
 *   );
 * }
 * ```
 */
export function useAssuranceLevel(did?: DID): UseAssuranceLevelReturn {
  const { mycelix, status } = useMycelixContext();
  const [assurance, setAssurance] = React.useState<AssuranceOutput | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<Error | null>(null);
  const [targetDid, setTargetDid] = React.useState<DID | null>(did || null);

  // Fetch the user's DID if not provided
  React.useEffect(() => {
    if (did) {
      setTargetDid(did);
      return;
    }

    const fetchDid = async () => {
      if (!mycelix || status !== 'connected') return;
      try {
        const userDid = await mycelix.getMyDid();
        setTargetDid(userDid || null);
      } catch {
        setTargetDid(null);
      }
    };

    fetchDid();
  }, [mycelix, status, did]);

  const fetchAssurance = React.useCallback(async () => {
    if (!mycelix || status !== 'connected' || !targetDid) {
      setLoading(status === 'connecting');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await mycelix.identity.mfa.getAssuranceLevel(targetDid);
      setAssurance(result);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
      setAssurance(null);
    } finally {
      setLoading(false);
    }
  }, [mycelix, status, targetDid]);

  React.useEffect(() => {
    fetchAssurance();
  }, [fetchAssurance]);

  return {
    level: assurance?.level || null,
    score: assurance?.score || 0,
    loading,
    error,
    refetch: fetchAssurance,
  };
}

/**
 * Return type for useEnrollmentHistory hook
 */
export interface UseEnrollmentHistoryReturn {
  /** Enrollment history entries */
  history: FactorEnrollment[];
  /** Whether loading */
  loading: boolean;
  /** Error if fetch failed */
  error: Error | null;
  /** Refetch history */
  refetch: () => Promise<void>;
}

/**
 * useEnrollmentHistory - Hook for fetching MFA enrollment history
 *
 * @param did - The DID to fetch history for (uses current user's DID if not provided)
 * @returns Enrollment history state
 *
 * @example
 * ```tsx
 * function EnrollmentTimeline() {
 *   const { history, loading } = useEnrollmentHistory();
 *
 *   if (loading) return <Skeleton />;
 *
 *   return (
 *     <Timeline>
 *       {history.map((entry, i) => (
 *         <TimelineItem key={i}>
 *           <span>{entry.action}: {entry.factorType}</span>
 *           <span>{new Date(entry.timestamp / 1000).toLocaleString()}</span>
 *         </TimelineItem>
 *       ))}
 *     </Timeline>
 *   );
 * }
 * ```
 */
export function useEnrollmentHistory(did?: DID): UseEnrollmentHistoryReturn {
  const { mycelix, status } = useMycelixContext();
  const [history, setHistory] = React.useState<FactorEnrollment[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<Error | null>(null);
  const [targetDid, setTargetDid] = React.useState<DID | null>(did || null);

  // Fetch the user's DID if not provided
  React.useEffect(() => {
    if (did) {
      setTargetDid(did);
      return;
    }

    const fetchDid = async () => {
      if (!mycelix || status !== 'connected') return;
      try {
        const userDid = await mycelix.getMyDid();
        setTargetDid(userDid || null);
      } catch {
        setTargetDid(null);
      }
    };

    fetchDid();
  }, [mycelix, status, did]);

  const fetchHistory = React.useCallback(async () => {
    if (!mycelix || status !== 'connected' || !targetDid) {
      setLoading(status === 'connecting');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await mycelix.identity.mfa.getEnrollmentHistory(targetDid);
      setHistory(result);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
      setHistory([]);
    } finally {
      setLoading(false);
    }
  }, [mycelix, status, targetDid]);

  React.useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  return {
    history,
    loading,
    error,
    refetch: fetchHistory,
  };
}
