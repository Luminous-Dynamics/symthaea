// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * UESS GraphQL API Layer
 *
 * Provides a GraphQL schema and resolvers for EpistemicStorage operations.
 *
 * ```typescript
 * import { createStorageSchema, createStorageResolvers } from '@mycelix/sdk/storage/graphql';
 *
 * const typeDefs = createStorageSchema();
 * const resolvers = createStorageResolvers(storage);
 * ```
 */

import type { EpistemicStorageImpl } from './epistemic-storage.js';
import type { EpistemicClassification } from '../epistemic/types.js';

// =============================================================================
// GraphQL Schema (SDL)
// =============================================================================

/**
 * Returns the GraphQL type definitions for UESS storage.
 */
export function createStorageSchema(): string {
  return `
    enum EmpiricalLevel {
      E0_Unverified
      E1_Testimonial
      E2_PrivateVerify
      E3_Cryptographic
      E4_Consensus
    }

    enum NormativeLevel {
      N0_Personal
      N1_Group
      N2_Network
      N3_Universal
    }

    enum MaterialityLevel {
      M0_Ephemeral
      M1_Temporal
      M2_Persistent
      M3_Immutable
    }

    type EpistemicClassification {
      empirical: Int!
      normative: Int!
      materiality: Int!
    }

    type SchemaIdentity {
      id: String!
      version: String!
      family: String
    }

    type StorageTier {
      backend: String!
      mutability: String!
      accessControl: String!
      replication: Int!
      encrypted: Boolean!
    }

    type StorageReceipt {
      key: String!
      cid: String!
      classification: EpistemicClassification!
      schema: SchemaIdentity!
      storedAt: Float!
      tier: StorageTier!
      version: Int!
      shreddable: Boolean!
    }

    type StorageMetadata {
      cid: String
      sizeBytes: Int
      storedAt: Float
      version: Int
      createdBy: String
      classification: EpistemicClassification
      schema: SchemaIdentity
    }

    type StoredData {
      data: String!
      metadata: StorageMetadata!
      verified: Boolean!
    }

    type StorageInfo {
      exists: Boolean!
      classification: EpistemicClassification
      schema: SchemaIdentity
      version: Int
      sizeBytes: Int
      locations: [String!]
    }

    type VerificationResult {
      key: String!
      verified: Boolean!
      cidValid: Boolean!
      actualReplication: Int!
      expectedReplication: Int!
      errors: [String!]!
    }

    type BackendItemCount {
      memory: Int!
      local: Int!
      dht: Int!
      ipfs: Int!
      filecoin: Int!
    }

    type StorageStats {
      totalItems: Int!
      totalSizeBytes: Int!
      itemsByBackend: BackendItemCount!
      cacheHitRate: Float!
      avgRetrievalTimeMs: Float!
    }

    type QueryResultItem {
      data: String!
      metadata: StorageMetadata!
      verified: Boolean!
    }

    type QueryResult {
      items: [QueryResultItem!]!
      totalCount: Int!
      hasMore: Boolean!
    }

    input ClassificationInput {
      empirical: Int!
      normative: Int!
      materiality: Int!
    }

    input SchemaInput {
      id: String!
      version: String!
      family: String
    }

    input StoreInput {
      key: String!
      data: String!
      classification: ClassificationInput!
      schema: SchemaInput!
    }

    input ClassificationFilterInput {
      minEmpirical: Int
      maxEmpirical: Int
      minNormative: Int
      maxNormative: Int
      minMateriality: Int
      maxMateriality: Int
    }

    input QueryInput {
      classification: ClassificationFilterInput
      schemaId: String
      createdBy: String
      limit: Int
      offset: Int
      orderBy: String
      orderDirection: String
    }

    type Query {
      retrieve(key: String!): StoredData
      retrieveByCID(cid: String!): StoredData
      storageInfo(key: String!): StorageInfo
      verify(key: String!): VerificationResult!
      exists(key: String!): Boolean!
      stats: StorageStats!
      query(input: QueryInput!): QueryResult!
    }

    type Mutation {
      store(input: StoreInput!): StorageReceipt!
    }
  `;
}

// =============================================================================
// Resolvers
// =============================================================================

/** GraphQL resolver map for UESS storage operations */
export interface StorageResolvers {
  Query: {
    retrieve: (_: unknown, args: { key: string }) => Promise<{ data: string; metadata: unknown; verified: boolean } | null>;
    retrieveByCID: (_: unknown, args: { cid: string }) => Promise<{ data: string; metadata: unknown; verified: boolean } | null>;
    storageInfo: (_: unknown, args: { key: string }) => Promise<unknown>;
    verify: (_: unknown, args: { key: string }) => Promise<unknown>;
    exists: (_: unknown, args: { key: string }) => Promise<boolean>;
    stats: () => Promise<unknown>;
    query: (_: unknown, args: { input: QueryInput }) => Promise<unknown>;
  };
  Mutation: {
    store: (_: unknown, args: { input: StoreInput }) => Promise<unknown>;
  };
}

interface StoreInput {
  key: string;
  data: string;
  classification: EpistemicClassification;
  schema: { id: string; version: string; family?: string };
}

interface QueryInput {
  classification?: {
    minEmpirical?: number;
    maxEmpirical?: number;
    minNormative?: number;
    maxNormative?: number;
    minMateriality?: number;
    maxMateriality?: number;
  };
  schemaId?: string;
  createdBy?: string;
  limit?: number;
  offset?: number;
  orderBy?: string;
  orderDirection?: string;
}

/**
 * Create GraphQL resolvers backed by an EpistemicStorage instance.
 *
 * ```typescript
 * const resolvers = createStorageResolvers(storage);
 *
 * // Use with any GraphQL server (Apollo, Yoga, etc.)
 * const server = createYoga({ schema: makeExecutableSchema({ typeDefs, resolvers }) });
 * ```
 */
export function createStorageResolvers(storage: EpistemicStorageImpl): StorageResolvers {
  return {
    Query: {
      retrieve: async (_, { key }) => {
        const result = await storage.retrieve(key);
        if (!result) return null;
        return {
          data: JSON.stringify(result.data),
          metadata: result.metadata,
          verified: result.verified,
        };
      },

      retrieveByCID: async (_, { cid }) => {
        const result = await storage.retrieveByCID(cid);
        if (!result) return null;
        return {
          data: JSON.stringify(result.data),
          metadata: result.metadata,
          verified: result.verified,
        };
      },

      storageInfo: async (_, { key }) => {
        return storage.getStorageInfo(key);
      },

      verify: async (_, { key }) => {
        return storage.verify(key);
      },

      exists: async (_, { key }) => {
        return storage.exists(key);
      },

      stats: async () => {
        return storage.getStats();
      },

      query: async (_, { input }) => {
        const result = await storage.query({
          classification: input.classification,
          schema: input.schemaId ? { id: input.schemaId } : undefined,
          createdBy: input.createdBy,
          limit: input.limit,
          offset: input.offset,
          orderBy: input.orderBy as 'storedAt' | 'modifiedAt' | 'classification' | undefined,
          orderDirection: input.orderDirection as 'asc' | 'desc' | undefined,
        });

        return {
          items: result.items.map((item) => ({
            data: JSON.stringify(item.data),
            metadata: item.metadata,
            verified: item.verified,
          })),
          totalCount: result.totalCount,
          hasMore: result.hasMore,
        };
      },
    },

    Mutation: {
      store: async (_, { input }) => {
        const data = JSON.parse(input.data);
        return storage.store(
          input.key,
          data,
          input.classification,
          { schema: input.schema }
        );
      },
    },
  };
}
