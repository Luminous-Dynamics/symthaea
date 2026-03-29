// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Knowledge - GraphQL Resolvers
 *
 * Connects GraphQL schema to Holochain zome calls
 */

import type { KnowledgeClient } from '../client/src';

// ============================================================================
// Types
// ============================================================================

interface Context {
  client: KnowledgeClient;
  pubsub: PubSubEngine;
}

interface PubSubEngine {
  publish(topic: string, payload: any): void;
  subscribe(topic: string): AsyncIterator<any>;
}

// ============================================================================
// Resolvers
// ============================================================================

export const resolvers = {
  // ==========================================================================
  // Query Resolvers
  // ==========================================================================
  Query: {
    claim: async (_: any, { id }: { id: string }, { client }: Context) => {
      return client.claims.getClaim(id);
    },

    claims: async (
      _: any,
      args: {
        first?: number;
        after?: string;
        author?: string;
        minCredibility?: number;
        tags?: string[];
      },
      { client }: Context
    ) => {
      const claims = await client.query.listClaims({
        limit: args.first || 20,
        cursor: args.after,
        author: args.author,
        minCredibility: args.minCredibility,
        tags: args.tags,
      });

      return {
        edges: claims.items.map((claim, i) => ({
          node: claim,
          cursor: Buffer.from(`${i}`).toString('base64'),
        })),
        pageInfo: {
          hasNextPage: claims.items.length === (args.first || 20),
          hasPreviousPage: !!args.after,
          startCursor: claims.items[0]
            ? Buffer.from('0').toString('base64')
            : null,
          endCursor: claims.items.length
            ? Buffer.from(`${claims.items.length - 1}`).toString('base64')
            : null,
        },
        totalCount: claims.total,
      };
    },

    search: async (
      _: any,
      { input }: { input: any },
      { client }: Context
    ) => {
      const startTime = Date.now();
      const results = await client.query.search(input.query, {
        epistemicType: input.epistemicType,
        minCredibility: input.minCredibility,
        tags: input.tags,
        limit: input.limit,
        offset: input.offset,
        sortBy: input.sortBy,
      });

      return {
        claims: results,
        total: results.length,
        took: Date.now() - startTime,
      };
    },

    author: async (_: any, { did }: { did: string }, { client }: Context) => {
      return { did };
    },

    authorReputation: async (
      _: any,
      { did }: { did: string },
      { client }: Context
    ) => {
      return client.inference.getAuthorReputation(did);
    },

    factCheck: async (
      _: any,
      { input }: { input: any },
      { client }: Context
    ) => {
      return client.factcheck.factCheck(input);
    },

    informationValueRanking: async (
      _: any,
      { limit }: { limit: number },
      { client }: Context
    ) => {
      return client.graph.rankByInformationValue(limit);
    },

    claimMarkets: async (
      _: any,
      { claimId }: { claimId: string },
      { client }: Context
    ) => {
      return client.marketsIntegration.getClaimMarkets(claimId);
    },
  },

  // ==========================================================================
  // Mutation Resolvers
  // ==========================================================================
  Mutation: {
    createClaim: async (
      _: any,
      { input }: { input: any },
      { client, pubsub }: Context
    ) => {
      const claimId = await client.claims.createClaim(input);
      const claim = await client.claims.getClaim(claimId);

      pubsub.publish('CLAIM_UPDATED', { claimUpdated: claim });

      return claim;
    },

    updateClaim: async (
      _: any,
      { id, input }: { id: string; input: any },
      { client, pubsub }: Context
    ) => {
      await client.claims.updateClaim({ originalHash: id, ...input });
      const claim = await client.claims.getClaim(id);

      pubsub.publish('CLAIM_UPDATED', { claimUpdated: claim });

      return claim;
    },

    deleteClaim: async (
      _: any,
      { id }: { id: string },
      { client }: Context
    ) => {
      await client.claims.deleteClaim(id);
      return true;
    },

    createRelationship: async (
      _: any,
      { input }: { input: any },
      { client, pubsub }: Context
    ) => {
      const relationshipId = await client.graph.createRelationship(
        input.sourceId,
        input.targetId,
        input.type,
        input.weight
      );

      const relationship = {
        id: relationshipId,
        ...input,
        createdAt: new Date().toISOString(),
      };

      pubsub.publish('RELATIONSHIP_CREATED', {
        relationshipCreated: relationship,
      });

      return relationship;
    },

    deleteRelationship: async (
      _: any,
      { id }: { id: string },
      { client }: Context
    ) => {
      // Implementation depends on zome function
      return true;
    },

    propagateBelief: async (
      _: any,
      { claimId }: { claimId: string },
      { client, pubsub }: Context
    ) => {
      const result = await client.graph.propagateBelief(claimId);

      // Notify about credibility changes
      for (const affectedId of result.affectedClaims) {
        const credibility = await client.inference.calculateEnhancedCredibility(
          affectedId,
          'Claim'
        );
        pubsub.publish('CREDIBILITY_CHANGED', {
          credibilityChanged: credibility,
        });
      }

      return result;
    },

    spawnVerificationMarket: async (
      _: any,
      { input }: { input: any },
      { client, pubsub }: Context
    ) => {
      const marketId = await client.claims.spawnVerificationMarket({
        claimId: input.claimId,
        targetE: input.targetE,
        minConfidence: input.minConfidence || 0.7,
        closesAt: input.closesAt || Date.now() + 14 * 24 * 60 * 60 * 1000,
        initialSubsidy: input.initialSubsidy || 100,
        tags: [],
      });

      const markets = await client.marketsIntegration.getClaimMarkets(
        input.claimId
      );
      const market = markets.find((m) => m.marketId === marketId);

      pubsub.publish('MARKET_UPDATED', { marketUpdated: market });

      return market;
    },
  },

  // ==========================================================================
  // Subscription Resolvers
  // ==========================================================================
  Subscription: {
    claimUpdated: {
      subscribe: (_: any, { claimId }: { claimId?: string }, { pubsub }: Context) => {
        return pubsub.subscribe('CLAIM_UPDATED');
      },
    },

    credibilityChanged: {
      subscribe: (
        _: any,
        { claimId, threshold }: { claimId?: string; threshold?: number },
        { pubsub }: Context
      ) => {
        return pubsub.subscribe('CREDIBILITY_CHANGED');
      },
    },

    relationshipCreated: {
      subscribe: (_: any, { claimId }: { claimId?: string }, { pubsub }: Context) => {
        return pubsub.subscribe('RELATIONSHIP_CREATED');
      },
    },

    marketUpdated: {
      subscribe: (_: any, { claimId }: { claimId?: string }, { pubsub }: Context) => {
        return pubsub.subscribe('MARKET_UPDATED');
      },
    },

    factCheckCompleted: {
      subscribe: (_: any, __: any, { pubsub }: Context) => {
        return pubsub.subscribe('FACT_CHECK_COMPLETED');
      },
    },
  },

  // ==========================================================================
  // Type Resolvers
  // ==========================================================================
  Claim: {
    credibility: async (claim: any, _: any, { client }: Context) => {
      return client.inference.calculateEnhancedCredibility(claim.id, 'Claim');
    },

    relationships: async (
      claim: any,
      { type, limit }: { type?: string; limit?: number },
      { client }: Context
    ) => {
      const relationships = await client.graph.getRelationships(claim.id);
      let filtered = relationships;

      if (type) {
        filtered = filtered.filter((r) => r.type === type);
      }

      return filtered.slice(0, limit || 20);
    },

    dependencyTree: async (
      claim: any,
      { maxDepth }: { maxDepth?: number },
      { client }: Context
    ) => {
      return client.graph.getDependencyTree(claim.id, maxDepth || 3);
    },

    markets: async (claim: any, _: any, { client }: Context) => {
      return client.marketsIntegration.getClaimMarkets(claim.id);
    },

    author: async (claim: any) => {
      return { did: claim.author };
    },
  },

  Author: {
    reputation: async (author: any, _: any, { client }: Context) => {
      return client.inference.getAuthorReputation(author.did);
    },

    claims: async (
      author: any,
      { limit, offset }: { limit?: number; offset?: number },
      { client }: Context
    ) => {
      const claims = await client.query.getClaimsByAuthor(author.did, {
        limit: limit || 20,
        offset: offset || 0,
      });

      return {
        edges: claims.map((claim, i) => ({
          node: claim,
          cursor: Buffer.from(`${i}`).toString('base64'),
        })),
        pageInfo: {
          hasNextPage: claims.length === (limit || 20),
          hasPreviousPage: (offset || 0) > 0,
        },
        totalCount: claims.length,
      };
    },
  },

  Relationship: {
    source: async (rel: any, _: any, { client }: Context) => {
      return client.claims.getClaim(rel.sourceId);
    },

    target: async (rel: any, _: any, { client }: Context) => {
      return client.claims.getClaim(rel.targetId);
    },
  },
};

export default resolvers;
