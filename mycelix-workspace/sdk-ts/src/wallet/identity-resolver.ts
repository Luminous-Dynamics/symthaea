// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Identity Resolver - Human-Readable Everything
 *
 * Transforms raw agent IDs (uhC0k...) into human-readable identities (@alice).
 * Integrates with the Identity hApp to provide:
 * - Profile resolution (nickname, avatar, display name)
 * - DID resolution (did:mycelix:...)
 * - Trust status from MATL
 * - Caching for instant lookups
 *
 * @example
 * ```typescript
 * const resolver = new HolochainIdentityResolver(client);
 *
 * // Never show raw hashes to users
 * const identity = await resolver.resolve('uhC0k...');
 * console.log(`${identity.nickname}`); // @alice
 *
 * // Lookup by nickname
 * const alice = await resolver.resolveNickname('@alice');
 * ```
 */

import { BehaviorSubject } from '../reactive/index.js';

import type { IdentityResolver, Identity } from './index.js';

// =============================================================================
// Types
// =============================================================================

/** Profile data from Identity hApp */
export interface ProfileData {
  agentId: string;
  nickname: string;
  displayName?: string;
  bio?: string;
  avatar?: string;
  location?: string;
  links?: Record<string, string>;
  createdAt: number;
  updatedAt: number;
}

/** DID Document */
export interface DIDDocument {
  id: string; // did:mycelix:...
  controller: string;
  verificationMethod: Array<{
    id: string;
    type: string;
    controller: string;
    publicKeyMultibase?: string;
  }>;
  authentication: string[];
  created: string;
  updated: string;
}

/** Cache entry */
interface CacheEntry<T> {
  value: T;
  timestamp: number;
}

/** Zome callable interface */
export interface ZomeCallable {
  callZome(params: {
    role_name?: string;
    zome_name: string;
    fn_name: string;
    payload: unknown;
  }): Promise<unknown>;
}

// =============================================================================
// Caching Identity Resolver
// =============================================================================

/**
 * In-memory cache for identity resolution
 */
export class IdentityCache {
  private byAgentId: Map<string, CacheEntry<Identity>> = new Map();
  private byNickname: Map<string, CacheEntry<Identity>> = new Map();
  private ttlMs: number;

  constructor(ttlMs: number = 5 * 60 * 1000) {
    // 5 minute default TTL
    this.ttlMs = ttlMs;
  }

  get(agentId: string): Identity | null {
    const entry = this.byAgentId.get(agentId);
    if (!entry) return null;
    if (Date.now() - entry.timestamp > this.ttlMs) {
      this.byAgentId.delete(agentId);
      return null;
    }
    return entry.value;
  }

  getByNickname(nickname: string): Identity | null {
    const normalizedNickname = nickname.toLowerCase();
    const entry = this.byNickname.get(normalizedNickname);
    if (!entry) return null;
    if (Date.now() - entry.timestamp > this.ttlMs) {
      this.byNickname.delete(normalizedNickname);
      return null;
    }
    return entry.value;
  }

  set(identity: Identity): void {
    const now = Date.now();
    this.byAgentId.set(identity.agentId, { value: identity, timestamp: now });
    if (identity.nickname) {
      this.byNickname.set(identity.nickname.toLowerCase(), { value: identity, timestamp: now });
    }
  }

  invalidate(agentId: string): void {
    const entry = this.byAgentId.get(agentId);
    if (entry?.value.nickname) {
      this.byNickname.delete(entry.value.nickname.toLowerCase());
    }
    this.byAgentId.delete(agentId);
  }

  clear(): void {
    this.byAgentId.clear();
    this.byNickname.clear();
  }
}

// =============================================================================
// Holochain Identity Resolver
// =============================================================================

/**
 * Identity resolver that connects to the Identity hApp via Holochain
 */
export class HolochainIdentityResolver implements IdentityResolver {
  private client: ZomeCallable;
  private cache: IdentityCache;
  private roleName: string;

  // Observable for known identities
  private _identities$: BehaviorSubject<Map<string, Identity>> = new BehaviorSubject(new Map());

  constructor(client: ZomeCallable, options?: { roleName?: string; cacheTtlMs?: number }) {
    this.client = client;
    this.roleName = options?.roleName ?? 'identity';
    this.cache = new IdentityCache(options?.cacheTtlMs);
  }

  /** Observable of all known identities */
  get identities$() {
    return this._identities$.asObservable();
  }

  /**
   * Resolve agent ID to full identity
   */
  async resolve(agentId: string): Promise<Identity> {
    // Check cache first
    const cached = this.cache.get(agentId);
    if (cached) return cached;

    try {
      // Call Identity hApp
      const profile = (await this.client.callZome({
        role_name: this.roleName,
        zome_name: 'profiles',
        fn_name: 'get_profile',
        payload: agentId,
      })) as ProfileData | null;

      // Get DID if available
      let did: string | undefined;
      try {
        const didDoc = (await this.client.callZome({
          role_name: this.roleName,
          zome_name: 'identity',
          fn_name: 'get_did',
          payload: agentId,
        })) as DIDDocument | null;
        did = didDoc?.id;
      } catch {
        // DID not available, that's ok
      }

      // Get trust score from MATL if available
      let verified = false;
      try {
        const trustScore = (await this.client.callZome({
          role_name: 'matl',
          zome_name: 'reputation',
          fn_name: 'get_reputation',
          payload: agentId,
        })) as { score: number } | null;
        verified = (trustScore?.score ?? 0) > 0.7;
      } catch {
        // MATL not available, that's ok
      }

      const identity: Identity = {
        agentId,
        nickname: profile?.nickname ? `@${profile.nickname}` : undefined,
        displayName: profile?.displayName,
        avatar: profile?.avatar,
        did,
        verified,
      };

      // Cache it
      this.cache.set(identity);
      this.updateKnownIdentities(identity);

      return identity;
    } catch (error) {
      // If Identity hApp is unavailable, return minimal identity
      console.warn(`Failed to resolve identity for ${agentId}:`, error);
      return {
        agentId,
        verified: false,
      };
    }
  }

  /**
   * Resolve nickname to identity
   */
  async resolveNickname(nickname: string): Promise<Identity | null> {
    // Normalize nickname (remove @ if present)
    const normalizedNickname = nickname.startsWith('@') ? nickname.slice(1) : nickname;

    // Check cache
    const cached = this.cache.getByNickname(`@${normalizedNickname}`);
    if (cached) return cached;

    try {
      // Search in Identity hApp
      const result = (await this.client.callZome({
        role_name: this.roleName,
        zome_name: 'profiles',
        fn_name: 'search_profiles',
        payload: { nickname: normalizedNickname },
      })) as ProfileData[] | null;

      if (!result || result.length === 0) return null;

      // Find exact match
      const exactMatch = result.find(
        (p) => p.nickname.toLowerCase() === normalizedNickname.toLowerCase()
      );

      if (!exactMatch) return null;

      // Resolve full identity
      return this.resolve(exactMatch.agentId);
    } catch (error) {
      console.warn(`Failed to resolve nickname ${nickname}:`, error);
      return null;
    }
  }

  /**
   * Search for identities by query
   */
  async search(query: string): Promise<Identity[]> {
    try {
      const results = (await this.client.callZome({
        role_name: this.roleName,
        zome_name: 'profiles',
        fn_name: 'search_profiles',
        payload: { query },
      })) as ProfileData[] | null;

      if (!results) return [];

      // Resolve all identities
      return Promise.all(results.map((p) => this.resolve(p.agentId)));
    } catch (error) {
      console.warn('Failed to search identities:', error);
      return [];
    }
  }

  /**
   * Batch resolve multiple agent IDs (efficient)
   */
  async resolveMany(agentIds: string[]): Promise<Map<string, Identity>> {
    const result = new Map<string, Identity>();
    const toResolve: string[] = [];

    // Check cache first
    for (const agentId of agentIds) {
      const cached = this.cache.get(agentId);
      if (cached) {
        result.set(agentId, cached);
      } else {
        toResolve.push(agentId);
      }
    }

    // Batch resolve the rest
    if (toResolve.length > 0) {
      try {
        const profiles = (await this.client.callZome({
          role_name: this.roleName,
          zome_name: 'profiles',
          fn_name: 'get_profiles_batch',
          payload: toResolve,
        })) as ProfileData[] | null;

        if (profiles) {
          for (const profile of profiles) {
            const identity: Identity = {
              agentId: profile.agentId,
              nickname: profile.nickname ? `@${profile.nickname}` : undefined,
              displayName: profile.displayName,
              avatar: profile.avatar,
            };
            this.cache.set(identity);
            result.set(profile.agentId, identity);
            this.updateKnownIdentities(identity);
          }
        }
      } catch {
        // Fall back to individual resolution
        await Promise.all(
          toResolve.map(async (agentId) => {
            const identity = await this.resolve(agentId);
            result.set(agentId, identity);
          })
        );
      }
    }

    return result;
  }

  /**
   * Invalidate cache for an agent
   */
  invalidateCache(agentId: string): void {
    this.cache.invalidate(agentId);
  }

  /**
   * Clear all cached identities
   */
  clearCache(): void {
    this.cache.clear();
  }

  private updateKnownIdentities(identity: Identity): void {
    const current = this._identities$.value;
    const updated = new Map(current);
    updated.set(identity.agentId, identity);
    this._identities$.next(updated);
  }
}

// =============================================================================
// Formatting Utilities
// =============================================================================

/**
 * Format an identity for display
 */
export function formatIdentity(identity: Identity, style: 'short' | 'full' = 'short'): string {
  if (style === 'short') {
    return identity.nickname ?? identity.displayName ?? truncateAgentId(identity.agentId);
  }

  const parts: string[] = [];
  if (identity.displayName) parts.push(identity.displayName);
  if (identity.nickname) parts.push(`(${identity.nickname})`);
  if (parts.length === 0) parts.push(truncateAgentId(identity.agentId));
  if (identity.verified) parts.push('✓');

  return parts.join(' ');
}

/**
 * Truncate agent ID for display (when no identity available)
 */
export function truncateAgentId(agentId: string): string {
  if (agentId.length <= 12) return agentId;
  return `${agentId.slice(0, 6)}...${agentId.slice(-4)}`;
}

/**
 * Generate avatar URL from agent ID (deterministic)
 */
export function generateAvatarUrl(agentId: string): string {
  // Use a deterministic avatar service
  const hash = simpleHash(agentId);
  return `https://api.dicebear.com/7.x/identicon/svg?seed=${hash}`;
}

/**
 * Simple hash for avatar generation
 */
function simpleHash(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash;
  }
  return Math.abs(hash).toString(16);
}

// =============================================================================
// Hydration Utilities
// =============================================================================

/**
 * Hydrate a transaction with identity information
 */
export async function hydrateTransaction<T extends { from: string | Identity; to: string | Identity }>(
  tx: T,
  resolver: IdentityResolver
): Promise<T & { from: Identity; to: Identity }> {
  const [from, to] = await Promise.all([
    typeof tx.from === 'string' ? resolver.resolve(tx.from) : Promise.resolve(tx.from),
    typeof tx.to === 'string' ? resolver.resolve(tx.to) : Promise.resolve(tx.to),
  ]);

  return { ...tx, from, to };
}

/**
 * Hydrate multiple transactions efficiently
 */
export async function hydrateTransactions<T extends { from: string | Identity; to: string | Identity }>(
  transactions: T[],
  resolver: IdentityResolver
): Promise<Array<T & { from: Identity; to: Identity }>> {
  // Collect unique agent IDs
  const agentIds = new Set<string>();
  for (const tx of transactions) {
    if (typeof tx.from === 'string') agentIds.add(tx.from);
    if (typeof tx.to === 'string') agentIds.add(tx.to);
  }

  // Batch resolve
  let identities: Map<string, Identity>;
  if ('resolveMany' in resolver && typeof resolver.resolveMany === 'function') {
    identities = await (resolver as HolochainIdentityResolver).resolveMany(Array.from(agentIds));
  } else {
    identities = new Map();
    await Promise.all(
      Array.from(agentIds).map(async (id) => {
        identities.set(id, await resolver.resolve(id));
      })
    );
  }

  // Hydrate transactions
  return transactions.map((tx) => ({
    ...tx,
    from: typeof tx.from === 'string' ? identities.get(tx.from)! : tx.from,
    to: typeof tx.to === 'string' ? identities.get(tx.to)! : tx.to,
  }));
}

// IdentityCache is already exported via its class declaration
