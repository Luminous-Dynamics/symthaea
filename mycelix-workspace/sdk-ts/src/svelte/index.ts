// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Svelte Stores for Mycelix SDK
 *
 * Provides reactive Svelte stores for all 8 Civilizational OS hApps.
 * Works with Svelte 4+ and SvelteKit.
 */

// ============================================================================
// Types
// ============================================================================

/** Svelte-compatible readable store interface */
export interface Readable<T> {
  subscribe(subscriber: (value: T) => void): () => void;
}

/** Svelte-compatible writable store interface */
export interface Writable<T> extends Readable<T> {
  set(value: T): void;
  update(updater: (value: T) => T): void;
}

/** Query store state */
export interface QueryStore<T> extends Readable<QueryState<T>> {
  refetch(): Promise<void>;
}

export interface QueryState<T> {
  data: T | undefined;
  loading: boolean;
  error: Error | undefined;
}

/** Mutation store state */
export interface MutationStore<T, V> extends Readable<MutationState<T>> {
  mutate(variables: V): Promise<T>;
  reset(): void;
}

export interface MutationState<T> {
  data: T | undefined;
  loading: boolean;
  error: Error | undefined;
}

/** Service context */
export interface MycelixContext {
  identityService: any;
  financeService: any;
  propertyService: any;
  energyService: any;
  mediaService: any;
  governanceService: any;
  justiceService: any;
  knowledgeService: any;
}

// ============================================================================
// Store Implementation (Framework-agnostic)
// ============================================================================

type Subscriber<T> = (value: T) => void;
type Unsubscriber = () => void;

/**
 * Create a simple writable store
 */
export function writable<T>(initial: T): Writable<T> {
  let value = initial;
  const subscribers = new Set<Subscriber<T>>();

  function set(newValue: T): void {
    value = newValue;
    subscribers.forEach((sub) => sub(value));
  }

  function update(updater: (value: T) => T): void {
    set(updater(value));
  }

  function subscribe(subscriber: Subscriber<T>): Unsubscriber {
    subscribers.add(subscriber);
    subscriber(value);
    return () => subscribers.delete(subscriber);
  }

  return { subscribe, set, update };
}

/**
 * Create a derived store
 */
export function derived<T, S>(store: Readable<S>, derive: (value: S) => T): Readable<T> {
  const subscribers = new Set<Subscriber<T>>();
  let value: T;

  store.subscribe((storeValue) => {
    value = derive(storeValue);
    subscribers.forEach((sub) => sub(value));
  });

  return {
    subscribe(subscriber: Subscriber<T>): Unsubscriber {
      subscribers.add(subscriber);
      subscriber(value);
      return () => subscribers.delete(subscriber);
    },
  };
}

// ============================================================================
// Context Store
// ============================================================================

const contextStore: Writable<MycelixContext | null> = writable(null);

/**
 * Set the Mycelix context (call this in your app initialization)
 */
export function setMycelixContext(ctx: MycelixContext): void {
  contextStore.set(ctx);
}

/**
 * Get the context store
 */
export function getContextStore(): Readable<MycelixContext | null> {
  return contextStore;
}

// ============================================================================
// Query Store Factory
// ============================================================================

/**
 * Create a query store
 */
export function createQueryStore<T, V = void>(
  fetcher: (variables: V, ctx: MycelixContext) => Promise<T>,
  variables: V
): QueryStore<T> {
  const store = writable<QueryState<T>>({
    data: undefined,
    loading: true,
    error: undefined,
  });

  let currentContext: MycelixContext | null = null;

  // Subscribe to context changes
  contextStore.subscribe((ctx) => {
    currentContext = ctx;
    if (ctx) {
      void refetch();
    }
  });

  async function refetch(): Promise<void> {
    if (!currentContext) return;

    store.update((s) => ({ ...s, loading: true, error: undefined }));

    try {
      const data = await fetcher(variables, currentContext);
      store.set({ data, loading: false, error: undefined });
    } catch (error) {
      store.set({
        data: undefined,
        loading: false,
        error: error instanceof Error ? error : new Error(String(error)),
      });
    }
  }

  return {
    subscribe: store.subscribe,
    refetch,
  };
}

/**
 * Create a mutation store
 */
export function createMutationStore<T, V>(
  mutator: (variables: V, ctx: MycelixContext) => Promise<T>
): MutationStore<T, V> {
  const store = writable<MutationState<T>>({
    data: undefined,
    loading: false,
    error: undefined,
  });

  let currentContext: MycelixContext | null = null;

  contextStore.subscribe((ctx) => {
    currentContext = ctx;
  });

  async function mutate(variables: V): Promise<T> {
    if (!currentContext) {
      throw new Error('Mycelix context not initialized');
    }

    store.update((s) => ({ ...s, loading: true, error: undefined }));

    try {
      const data = await mutator(variables, currentContext);
      store.set({ data, loading: false, error: undefined });
      return data;
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      store.set({ data: undefined, loading: false, error: err });
      throw err;
    }
  }

  function reset(): void {
    store.set({ data: undefined, loading: false, error: undefined });
  }

  return {
    subscribe: store.subscribe,
    mutate,
    reset,
  };
}

// ============================================================================
// Identity Stores
// ============================================================================

/**
 * Create a store for fetching an identity profile
 */
export function createIdentityProfileStore(did: string): QueryStore<any> {
  return createQueryStore((_, ctx) => ctx.identityService?.getProfile(did), undefined);
}

/**
 * Create a mutation store for creating identities
 */
export function createCreateIdentityStore(): MutationStore<
  any,
  { publicKey: string; displayName?: string }
> {
  return createMutationStore((vars, ctx) =>
    ctx.identityService?.createIdentity(vars.publicKey, vars.displayName)
  );
}

/**
 * Create a mutation store for trust attestation
 */
export function createAttestTrustStore(): MutationStore<
  any,
  { attesterId: string; targetId: string }
> {
  return createMutationStore((vars, ctx) =>
    ctx.identityService?.attestTrust(vars.attesterId, vars.targetId)
  );
}

// ============================================================================
// Finance Stores
// ============================================================================

/**
 * Create a store for fetching a wallet
 */
export function createWalletStore(walletId: string): QueryStore<any> {
  return createQueryStore((_, ctx) => ctx.financeService?.getWallet(walletId), undefined);
}

/**
 * Create a store for fetching wallets by owner
 */
export function createWalletsByOwnerStore(ownerId: string): QueryStore<any[]> {
  return createQueryStore(
    (_, ctx) => ctx.financeService?.getWalletsByOwner(ownerId) ?? [],
    undefined
  );
}

/**
 * Create a store for credit score
 */
export function createCreditScoreStore(did: string): QueryStore<any> {
  return createQueryStore((_, ctx) => ctx.financeService?.getCreditScore(did), undefined);
}

/**
 * Create a mutation store for transfers
 */
export function createTransferStore(): MutationStore<
  any,
  { fromWallet: string; toWallet: string; amount: number; currency: string; memo?: string }
> {
  return createMutationStore((vars, ctx) =>
    ctx.financeService?.transfer(
      vars.fromWallet,
      vars.toWallet,
      vars.amount,
      vars.currency,
      vars.memo
    )
  );
}

// ============================================================================
// Property Stores
// ============================================================================

/**
 * Create a store for fetching an asset
 */
export function createAssetStore(assetId: string): QueryStore<any> {
  return createQueryStore((_, ctx) => ctx.propertyService?.getAsset(assetId), undefined);
}

/**
 * Create a store for assets by owner
 */
export function createAssetsByOwnerStore(ownerId: string): QueryStore<any[]> {
  return createQueryStore(
    (_, ctx) => ctx.propertyService?.getAssetsByOwner(ownerId) ?? [],
    undefined
  );
}

/**
 * Create a mutation store for registering assets
 */
export function createRegisterAssetStore(): MutationStore<
  any,
  { type: string; name: string; description?: string; ownerId: string; ownershipType: string }
> {
  return createMutationStore((vars, ctx) =>
    ctx.propertyService?.registerAsset(
      vars.type,
      vars.name,
      vars.description,
      vars.ownerId,
      vars.ownershipType
    )
  );
}

// ============================================================================
// Energy Stores
// ============================================================================

/**
 * Create a store for energy participant
 */
export function createEnergyParticipantStore(participantId: string): QueryStore<any> {
  return createQueryStore((_, ctx) => ctx.energyService?.getParticipant(participantId), undefined);
}

/**
 * Create a store for energy credits
 */
export function createEnergyCreditsStore(participantId: string): QueryStore<any[]> {
  return createQueryStore(
    (_, ctx) => ctx.energyService?.getCredits(participantId) ?? [],
    undefined
  );
}

/**
 * Create a mutation store for energy trading
 */
export function createTradeEnergyStore(): MutationStore<
  any,
  { sellerId: string; buyerId: string; amountKwh: number; source: string; pricePerKwh: number }
> {
  return createMutationStore((vars, ctx) =>
    ctx.energyService?.tradeEnergy(
      vars.sellerId,
      vars.buyerId,
      vars.amountKwh,
      vars.source,
      vars.pricePerKwh
    )
  );
}

// ============================================================================
// Media Stores
// ============================================================================

/**
 * Create a store for content
 */
export function createContentStore(contentId: string): QueryStore<any> {
  return createQueryStore((_, ctx) => ctx.mediaService?.getContent(contentId), undefined);
}

/**
 * Create a mutation store for publishing content
 */
export function createPublishContentStore(): MutationStore<
  any,
  {
    authorDid: string;
    type: string;
    title: string;
    description?: string;
    contentHash: string;
    storageUri: string;
    license: string;
    tags?: string[];
  }
> {
  return createMutationStore((vars, ctx) =>
    ctx.mediaService?.publishContent(
      vars.authorDid,
      vars.type,
      vars.title,
      vars.description,
      vars.contentHash,
      vars.storageUri,
      vars.license,
      vars.tags
    )
  );
}

// ============================================================================
// Governance Stores
// ============================================================================

/**
 * Create a store for a DAO
 */
export function createDAOStore(daoId: string): QueryStore<any> {
  return createQueryStore((_, ctx) => ctx.governanceService?.getDAO(daoId), undefined);
}

/**
 * Create a store for proposals
 */
export function createProposalsStore(daoId: string, status?: string): QueryStore<any[]> {
  return createQueryStore(
    (_, ctx) => ctx.governanceService?.listProposals({ daoId, status }) ?? [],
    undefined
  );
}

/**
 * Create a mutation store for voting
 */
export function createCastVoteStore(): MutationStore<
  boolean,
  { proposalId: string; voterId: string; choice: string; weight: number }
> {
  return createMutationStore(async (vars, ctx) => {
    await ctx.governanceService?.castVote(vars);
    return true;
  });
}

// ============================================================================
// Justice Stores
// ============================================================================

/**
 * Create a store for a case
 */
export function createCaseStore(caseId: string): QueryStore<any> {
  return createQueryStore((_, ctx) => ctx.justiceService?.getCase(caseId), undefined);
}

/**
 * Create a mutation store for filing cases
 */
export function createFileCaseStore(): MutationStore<
  any,
  {
    complainantId: string;
    respondentId: string;
    title: string;
    description: string;
    category: string;
  }
> {
  return createMutationStore((vars, ctx) =>
    ctx.justiceService?.fileCase(
      vars.complainantId,
      vars.respondentId,
      vars.title,
      vars.description,
      vars.category
    )
  );
}

// ============================================================================
// Knowledge Stores
// ============================================================================

/**
 * Create a store for a claim
 */
export function createClaimStore(claimId: string): QueryStore<any> {
  return createQueryStore((_, ctx) => ctx.knowledgeService?.getClaim(claimId), undefined);
}

/**
 * Create a mutation store for submitting claims
 */
export function createSubmitClaimStore(): MutationStore<
  any,
  {
    authorId: string;
    title: string;
    content: string;
    empirical: number;
    normative: number;
    tags?: string[];
  }
> {
  return createMutationStore((vars, ctx) =>
    ctx.knowledgeService?.submitClaim(
      vars.authorId,
      vars.title,
      vars.content,
      vars.empirical,
      vars.normative,
      vars.tags
    )
  );
}

/**
 * Create a mutation store for endorsing claims
 */
export function createEndorseClaimStore(): MutationStore<
  boolean,
  { claimId: string; endorserId: string }
> {
  return createMutationStore(async (vars, ctx) => {
    await ctx.knowledgeService?.endorseClaim(vars.claimId, vars.endorserId);
    return true;
  });
}

// ============================================================================
// Cross-Domain Stores
// ============================================================================

/**
 * Create a store for related entities
 */
export function createRelatedEntitiesStore(did: string): QueryStore<{
  did: string;
  assets: any[];
  wallets: any[];
  content: any[];
  proposals: any[];
  cases: any[];
  claims: any[];
}> {
  return createQueryStore(
    async (_, ctx) => ({
      did,
      assets: (await ctx.propertyService?.getAssetsByOwner?.(did)) ?? [],
      wallets: (await ctx.financeService?.getWalletsByOwner?.(did)) ?? [],
      content: (await ctx.mediaService?.getContentByAuthor?.(did)) ?? [],
      proposals: (await ctx.governanceService?.getProposalsByMember?.(did)) ?? [],
      cases: (await ctx.justiceService?.getCasesByParty?.(did)) ?? [],
      claims: (await ctx.knowledgeService?.getClaimsByAuthor?.(did)) ?? [],
    }),
    undefined
  );
}

// ============================================================================
// Wallet Bridge Stores (Cross-hApp Reputation)
// ============================================================================

import type { ReputationQueryResponse } from '../bridge/cross-happ.js';
import type { WalletBridge, TransactionRecommendation, VerificationStatus } from '../wallet/bridge-integration.js';

/** Wallet bridge context store */
const walletBridgeStore: Writable<WalletBridge | null> = writable(null);

/**
 * Set the wallet bridge instance
 */
export function setWalletBridge(bridge: WalletBridge): void {
  walletBridgeStore.set(bridge);
}

/**
 * Get the wallet bridge store
 */
export function getWalletBridgeStore(): Readable<WalletBridge | null> {
  return walletBridgeStore;
}

/** Reputation store state */
export interface ReputationStoreState {
  reputation: ReputationQueryResponse | undefined;
  loading: boolean;
  error: Error | undefined;
  isTrustworthy: boolean;
}

/**
 * Create a store for cross-hApp reputation
 *
 * @example
 * ```svelte
 * <script>
 *   import { createReputationStore, setWalletBridge } from '@mycelix/svelte';
 *
 *   const reputation = createReputationStore('did:mycelix:alice');
 * </script>
 *
 * {#if $reputation.loading}
 *   <p>Loading reputation...</p>
 * {:else if $reputation.reputation}
 *   <p>Trust Score: {($reputation.reputation.aggregatedScore * 100).toFixed(0)}%</p>
 *   <p>Trustworthy: {$reputation.isTrustworthy ? 'Yes' : 'No'}</p>
 * {/if}
 * ```
 */
export function createReputationStore(did: string, threshold = 0.6): QueryStore<ReputationStoreState> & {
  reportPositive: (txId: string) => void;
  reportNegative: (reason: 'payment_failed' | 'dispute' | 'fraud_suspected') => void;
} {
  const store = writable<QueryState<ReputationStoreState>>({
    data: undefined,
    loading: true,
    error: undefined,
  });

  let currentBridge: WalletBridge | null = null;
  let unsubscribeReputation: (() => void) | null = null;

  async function refetch(): Promise<void> {
    if (!currentBridge) return;

    store.update((s) => ({ ...s, loading: true, error: undefined }));

    try {
      const reputation = await currentBridge.getAggregateReputation(did, true);
      const isTrustworthy = reputation.aggregatedScore >= threshold && reputation.confidence >= 0.5;

      store.set({
        data: { reputation, loading: false, error: undefined, isTrustworthy },
        loading: false,
        error: undefined,
      });
    } catch (error) {
      store.set({
        data: undefined,
        loading: false,
        error: error instanceof Error ? error : new Error(String(error)),
      });
    }
  }

  // Subscribe to wallet bridge changes
  walletBridgeStore.subscribe((bridge) => {
    // Clean up previous subscription
    if (unsubscribeReputation) {
      unsubscribeReputation();
      unsubscribeReputation = null;
    }

    currentBridge = bridge;
    if (bridge) {
      void refetch();

      // Subscribe to reputation updates
      unsubscribeReputation = bridge.onReputationChange(did, (rep) => {
        const isTrustworthy = rep.aggregatedScore >= threshold && rep.confidence >= 0.5;
        store.set({
          data: { reputation: rep, loading: false, error: undefined, isTrustworthy },
          loading: false,
          error: undefined,
        });
      });
    }
  });

  return {
    subscribe: store.subscribe,
    refetch,
    reportPositive: (txId: string) => {
      currentBridge?.reportPositiveInteraction(did, txId);
    },
    reportNegative: (reason: 'payment_failed' | 'dispute' | 'fraud_suspected') => {
      currentBridge?.reportNegativeInteraction(did, reason);
    },
  };
}

/** Transaction recommendation store state */
export interface RecommendationStoreState {
  recommendation: TransactionRecommendation | undefined;
  loading: boolean;
  error: Error | undefined;
}

/**
 * Create a store for transaction recommendations
 *
 * @example
 * ```svelte
 * <script>
 *   import { createTransactionRecommendationStore } from '@mycelix/svelte';
 *
 *   let amount = 100;
 *   $: recommendation = createTransactionRecommendationStore('did:mycelix:recipient', amount);
 * </script>
 *
 * {#if $recommendation.data?.recommendation}
 *   <p>Risk: {$recommendation.data.recommendation.riskLevel}</p>
 *   <p>{$recommendation.data.recommendation.recommendation}</p>
 * {/if}
 * ```
 */
export function createTransactionRecommendationStore(
  did: string,
  amount: number
): QueryStore<RecommendationStoreState> {
  const store = writable<QueryState<RecommendationStoreState>>({
    data: undefined,
    loading: true,
    error: undefined,
  });

  let currentBridge: WalletBridge | null = null;

  async function refetch(): Promise<void> {
    if (!currentBridge) return;

    store.update((s) => ({ ...s, loading: true, error: undefined }));

    try {
      const recommendation = await currentBridge.getTransactionRecommendation(did, amount);
      store.set({
        data: { recommendation, loading: false, error: undefined },
        loading: false,
        error: undefined,
      });
    } catch (error) {
      store.set({
        data: undefined,
        loading: false,
        error: error instanceof Error ? error : new Error(String(error)),
      });
    }
  }

  walletBridgeStore.subscribe((bridge) => {
    currentBridge = bridge;
    if (bridge) {
      void refetch();
    }
  });

  return {
    subscribe: store.subscribe,
    refetch,
  };
}

/**
 * Create a mutation store for identity verification
 */
export function createVerifyIdentityStore(): MutationStore<VerificationStatus, { did: string }> {
  const store = writable<MutationState<VerificationStatus>>({
    data: undefined,
    loading: false,
    error: undefined,
  });

  let currentBridge: WalletBridge | null = null;
  walletBridgeStore.subscribe((b) => (currentBridge = b));

  return {
    subscribe: store.subscribe,
    async mutate(vars: { did: string }): Promise<VerificationStatus> {
      if (!currentBridge) {
        throw new Error('Wallet bridge not initialized');
      }

      store.update((s) => ({ ...s, loading: true, error: undefined }));

      try {
        const result = await currentBridge.requestIdentityVerification(vars.did);
        store.set({ data: result, loading: false, error: undefined });
        return result;
      } catch (error) {
        const err = error instanceof Error ? error : new Error(String(error));
        store.update((s) => ({ ...s, loading: false, error: err }));
        throw err;
      }
    },
    reset(): void {
      store.set({ data: undefined, loading: false, error: undefined });
    },
  };
}

/**
 * Create a mutation store for credential verification
 */
export function createVerifyCredentialStore(): MutationStore<
  VerificationStatus,
  { did: string; credentialType: string }
> {
  const store = writable<MutationState<VerificationStatus>>({
    data: undefined,
    loading: false,
    error: undefined,
  });

  let currentBridge: WalletBridge | null = null;
  walletBridgeStore.subscribe((b) => (currentBridge = b));

  return {
    subscribe: store.subscribe,
    async mutate(vars: { did: string; credentialType: string }): Promise<VerificationStatus> {
      if (!currentBridge) {
        throw new Error('Wallet bridge not initialized');
      }

      store.update((s) => ({ ...s, loading: true, error: undefined }));

      try {
        const result = await currentBridge.verifyCredential(vars.did, vars.credentialType);
        store.set({ data: result, loading: false, error: undefined });
        return result;
      } catch (error) {
        const err = error instanceof Error ? error : new Error(String(error));
        store.set({ data: undefined, loading: false, error: err });
        throw err;
      }
    },
    reset(): void {
      store.set({ data: undefined, loading: false, error: undefined });
    },
  };
}

/**
 * Utility: Format reputation score for display
 */
export function formatReputationScore(score: number): string {
  return (score * 100).toFixed(0) + '%';
}

/**
 * Utility: Get reputation badge based on score
 */
export function getReputationBadge(score: number): {
  label: string;
  color: 'green' | 'yellow' | 'orange' | 'red' | 'gray';
} {
  if (score >= 0.9) return { label: 'Excellent', color: 'green' };
  if (score >= 0.7) return { label: 'Good', color: 'green' };
  if (score >= 0.5) return { label: 'Fair', color: 'yellow' };
  if (score >= 0.3) return { label: 'Low', color: 'orange' };
  if (score > 0) return { label: 'Poor', color: 'red' };
  return { label: 'Unknown', color: 'gray' };
}

/**
 * Utility: Format risk level for display
 */
export function formatRiskLevel(riskLevel: 'low' | 'medium' | 'high' | 'very_high'): {
  label: string;
  color: 'green' | 'yellow' | 'orange' | 'red';
} {
  switch (riskLevel) {
    case 'low':
      return { label: 'Low Risk', color: 'green' };
    case 'medium':
      return { label: 'Medium Risk', color: 'yellow' };
    case 'high':
      return { label: 'High Risk', color: 'orange' };
    case 'very_high':
      return { label: 'Very High Risk', color: 'red' };
  }
}

// ============================================================================
// Svelte Usage Example
// ============================================================================

/**
 * Example usage in a Svelte component:
 *
 * ```svelte
 * <script>
 *   import { setMycelixContext, createIdentityProfileStore } from '@mycelix/svelte';
 *   import { getIdentityService, getFinanceService } from '@mycelix/sdk';
 *   import { onMount } from 'svelte';
 *
 *   // Initialize context
 *   onMount(() => {
 *     setMycelixContext({
 *       identityService: getIdentityService(),
 *       financeService: getFinanceService(),
 *       // ... other services
 *     });
 *   });
 *
 *   // Create reactive store
 *   const did = 'did:mycelix:abc123';
 *   const profile = createIdentityProfileStore(did);
 * </script>
 *
 * {#if $profile.loading}
 *   <p>Loading...</p>
 * {:else if $profile.error}
 *   <p>Error: {$profile.error.message}</p>
 * {:else if $profile.data}
 *   <p>Name: {$profile.data.displayName}</p>
 *   <p>Level: {$profile.data.verificationLevel}</p>
 * {/if}
 * ```
 */
