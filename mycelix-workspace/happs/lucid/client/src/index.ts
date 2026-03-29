// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * LUCID Client SDK
 *
 * TypeScript client for the LUCID Personal Knowledge Graph hApp.
 *
 * @example
 * ```typescript
 * import { LucidClient } from '@mycelix/lucid-client';
 * import { AppWebsocket } from '@holochain/client';
 *
 * const client = await AppWebsocket.connect('ws://localhost:8888');
 * const lucid = new LucidClient(client);
 *
 * // Create a thought
 * const thought = await lucid.createThought({
 *   content: 'Knowledge is power',
 *   thought_type: 'Claim',
 *   confidence: 0.8,
 *   tags: ['philosophy', 'epistemology'],
 * });
 *
 * // Search thoughts
 * const results = await lucid.searchThoughts({
 *   tags: ['philosophy'],
 *   min_confidence: 0.7,
 * });
 * ```
 */

import type { AppClient } from '@holochain/client';
import { LucidZomeClient } from './zomes/lucid';
import { SourcesZomeClient } from './zomes/sources';
import { TemporalZomeClient } from './zomes/temporal';
import { TemporalConsciousnessZomeClient } from './zomes/temporal-consciousness';
import { PrivacyZomeClient } from './zomes/privacy';
import { ReasoningZomeClient } from './zomes/reasoning';
import { BridgeZomeClient } from './zomes/bridge';
import { CollectiveZomeClient } from './zomes/collective';

// Re-export types
export * from './types';
export * from './utils';

import { CitationRelationship } from './types';

// Re-export zome clients
export { LucidZomeClient } from './zomes/lucid';
export { SourcesZomeClient } from './zomes/sources';
export { TemporalZomeClient } from './zomes/temporal';
export { TemporalConsciousnessZomeClient } from './zomes/temporal-consciousness';
export { PrivacyZomeClient } from './zomes/privacy';
export { ReasoningZomeClient } from './zomes/reasoning';
export { BridgeZomeClient } from './zomes/bridge';
export { CollectiveZomeClient } from './zomes/collective';

/**
 * Main LUCID client providing access to all zome functionality.
 */
export class LucidClient {
  /** Core thought operations */
  public readonly lucid: LucidZomeClient;

  /** Source and citation operations */
  public readonly sources: SourcesZomeClient;

  /** Temporal versioning operations */
  public readonly temporal: TemporalZomeClient;

  /** Temporal consciousness: trajectory tracking and evolution analysis */
  public readonly temporalConsciousness: TemporalConsciousnessZomeClient;

  /** Privacy and sharing operations */
  public readonly privacy: PrivacyZomeClient;

  /** Reasoning and coherence operations */
  public readonly reasoning: ReasoningZomeClient;

  /** Cross-hApp bridge operations */
  public readonly bridge: BridgeZomeClient;

  /** Collective sensemaking operations */
  public readonly collective: CollectiveZomeClient;

  constructor(
    private client: AppClient,
    private roleName: string = 'lucid'
  ) {
    this.lucid = new LucidZomeClient(client, roleName, 'lucid');
    this.sources = new SourcesZomeClient(client, roleName, 'sources');
    this.temporal = new TemporalZomeClient(client, roleName, 'temporal');
    this.temporalConsciousness = new TemporalConsciousnessZomeClient(client, roleName, 'temporal_consciousness');
    this.privacy = new PrivacyZomeClient(client, roleName, 'privacy');
    this.reasoning = new ReasoningZomeClient(client, roleName, 'reasoning');
    this.bridge = new BridgeZomeClient(client, roleName, 'bridge');
    this.collective = new CollectiveZomeClient(client, roleName, 'collective');
  }

  // ============================================================================
  // CONVENIENCE METHODS (delegate to lucid zome)
  // ============================================================================

  /** Create a new thought */
  async createThought(
    ...args: Parameters<LucidZomeClient['createThought']>
  ): ReturnType<LucidZomeClient['createThought']> {
    return this.lucid.createThought(...args);
  }

  /** Get a thought by ID */
  async getThought(
    ...args: Parameters<LucidZomeClient['getThought']>
  ): ReturnType<LucidZomeClient['getThought']> {
    return this.lucid.getThought(...args);
  }

  /** Update an existing thought */
  async updateThought(
    ...args: Parameters<LucidZomeClient['updateThought']>
  ): ReturnType<LucidZomeClient['updateThought']> {
    return this.lucid.updateThought(...args);
  }

  /** Delete a thought */
  async deleteThought(
    ...args: Parameters<LucidZomeClient['deleteThought']>
  ): ReturnType<LucidZomeClient['deleteThought']> {
    return this.lucid.deleteThought(...args);
  }

  /** Get all thoughts for the current user */
  async getMyThoughts(): ReturnType<LucidZomeClient['getMyThoughts']> {
    return this.lucid.getMyThoughts();
  }

  /** Search thoughts with filters */
  async searchThoughts(
    ...args: Parameters<LucidZomeClient['searchThoughts']>
  ): ReturnType<LucidZomeClient['searchThoughts']> {
    return this.lucid.searchThoughts(...args);
  }

  /** Get statistics about the knowledge graph */
  async getStats(): ReturnType<LucidZomeClient['getStats']> {
    return this.lucid.getStats();
  }

  // ============================================================================
  // HIGH-LEVEL OPERATIONS
  // ============================================================================

  /**
   * Create a thought with a source citation in one call.
   */
  async createThoughtWithSource(
    thoughtInput: Parameters<LucidZomeClient['createThought']>[0],
    sourceInput: Parameters<SourcesZomeClient['createSource']>[0],
    citationOptions?: {
      location?: string;
      quote?: string;
      relationship?: CitationRelationship;
    }
  ) {
    // Create the thought
    const thought = await this.lucid.createThought(thoughtInput);

    // Create the source
    const source = await this.sources.createSource(sourceInput);

    // Create the citation
    const citation = await this.sources.createCitation({
      thought_id: thought.id,
      source_id: source.id,
      location: citationOptions?.location,
      quote: citationOptions?.quote,
      relationship: citationOptions?.relationship ?? CitationRelationship.Supports,
    });

    return { thought, source, citation };
  }

  /**
   * Get a thought with its full context (sources, relationships, history).
   */
  async getThoughtWithContext(thoughtId: string) {
    const [thought, sources, relationships, history, contradictions] = await Promise.all([
      this.lucid.getThought(thoughtId),
      this.sources.getThoughtSources(thoughtId),
      this.lucid.getThoughtRelationships(thoughtId),
      this.temporal.getThoughtHistory(thoughtId),
      this.reasoning.getThoughtContradictions(thoughtId),
    ]);

    return {
      thought,
      sources,
      relationships,
      history,
      contradictions,
    };
  }

  /**
   * Share a thought with specific agents.
   */
  async shareThought(
    thoughtId: string,
    agents: Parameters<PrivacyZomeClient['grantAccess']>[0]['grantee'][],
    permissions: string[] = ['read']
  ) {
    const grants = await Promise.all(
      agents.map((agent) =>
        this.privacy.grantAccess({
          thought_id: thoughtId,
          grantee: agent,
          permissions,
        })
      )
    );

    return grants;
  }

  /**
   * Make a thought public.
   */
  async makePublic(thoughtId: string) {
    return this.privacy.setSharingPolicy({
      thought_id: thoughtId,
      visibility: { type: 'Public' },
    });
  }

  /**
   * Make a thought private.
   */
  async makePrivate(thoughtId: string) {
    return this.privacy.setSharingPolicy({
      thought_id: thoughtId,
      visibility: { type: 'Private' },
    });
  }
}

// Default export
export default LucidClient;
