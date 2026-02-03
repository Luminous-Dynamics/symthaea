/**
 * @mycelix/civic-client
 *
 * TypeScript client for the Civic hApp - knowledge storage and agent reputation
 * for Symthaea AI agents.
 *
 * @example
 * ```typescript
 * import { AppWebsocket } from '@holochain/client';
 * import { CivicClient } from '@mycelix/civic-client';
 *
 * const client = await AppWebsocket.connect('ws://localhost:8888');
 * const civic = new CivicClient(client);
 *
 * // Search for knowledge
 * const results = await civic.knowledge.searchByDomain('benefits');
 *
 * // Record feedback
 * await civic.reputation.recordHelpful(agentPubkey, 'conv-123');
 *
 * // Check trust score
 * const score = await civic.reputation.getTrustScore(agentPubkey);
 * if (MATL.isTrustworthy(score)) {
 *   console.log('Agent is trustworthy');
 * }
 * ```
 */

import type { AppClient } from '@holochain/client';
import { CivicKnowledgeClient, createKnowledgeClient } from './knowledge.js';
import { AgentReputationClient, createReputationClient, MATL } from './reputation.js';

export * from './types.js';
export * from './knowledge.js';
export * from './reputation.js';

/**
 * Combined client for all Civic hApp functionality
 */
export class CivicClient {
  /** Civic knowledge client */
  public readonly knowledge: CivicKnowledgeClient;

  /** Agent reputation client */
  public readonly reputation: AgentReputationClient;

  constructor(client: AppClient, roleName = 'civic') {
    this.knowledge = createKnowledgeClient(client, roleName);
    this.reputation = createReputationClient(client, roleName);
  }

  /**
   * Search for knowledge relevant to a question and return with trust info
   */
  async findTrustedKnowledge(
    question: string,
    domain?: import('./types.js').CivicDomain,
    location?: string,
  ): Promise<Array<{
    knowledge: import('./types.js').CivicKnowledge;
    actionHash: import('@holochain/client').ActionHash;
    validationCount: number;
  }>> {
    const results = await this.knowledge.findRelevantKnowledge(question, domain, location);

    // Enrich with validation counts
    const enriched = await Promise.all(
      results.map(async (result) => {
        const validations = await this.knowledge.getKnowledgeValidations(result.action_hash);
        const positiveValidations = validations.filter(v => v.is_valid).length;

        return {
          knowledge: result.knowledge,
          actionHash: result.action_hash,
          validationCount: positiveValidations,
        };
      }),
    );

    // Sort by validation count
    return enriched.sort((a, b) => b.validationCount - a.validationCount);
  }
}

/**
 * Create a CivicClient from an AppClient
 */
export function createCivicClient(client: AppClient, roleName?: string): CivicClient {
  return new CivicClient(client, roleName);
}

// Re-export MATL utilities
export { MATL };
