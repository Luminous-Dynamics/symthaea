/**
 * Mycelix Knowledge Client - Master SDK (Phase 4)
 *
 * Unified client for the mycelix-knowledge hApp.
 * Provides epistemic claims with E/N/M classification, knowledge graph,
 * query engine, ML-powered inference, and cross-hApp knowledge bridge.
 *
 * Re-exports from existing comprehensive implementation.
 *
 * @module @mycelix/sdk/clients/knowledge
 */

// Re-export all types and clients from the existing comprehensive implementation
export * from '../../knowledge/index.js';

// Re-export default
export { default } from '../../knowledge/index.js';

// Explicit re-exports for clarity
export {
  // Types
  type HolochainRecord,
  type ZomeCallable,
  type EpistemicPosition,
  type ClaimType,
  type SourceType,
  type Claim,
  type UpdateClaimInput,
  type EvidenceType,
  type Evidence,
  type ChallengeStatus,
  type ClaimChallenge,
  type ChallengeClaimInput,
  type EpistemicRangeSearch,
  type RelationshipType,
  type Relationship,
  type Ontology,
  type Concept,
  type FindPathInput,
  type GraphStats,
  type SavedQuery,
  type QueryStepType,
  type FilterOperator,
  type QueryFilter,
  type QueryStep,
  type QueryPlan,
  type ExecuteQueryInput,
  type QueryExecutionResult,
  type InferenceType,
  type Inference,
  type CredibilitySubjectType,
  type CredibilityComponents,
  type CredibilityScore,
  type AssessCredibilityInput,
  type PatternType,
  type Pattern,
  type DetectPatternsInput,
  type VerifyInferenceInput,
  type KnowledgeQueryType,
  type KnowledgeEventType,
  type QueryKnowledgeInput,
  type QueryKnowledgeResult,
  type FactCheckVerdict,
  type FactCheckInput,
  type FactCheckResult,
  type RegisterClaimInput,
  type BroadcastKnowledgeEventInput,
  type KnowledgeBridgeEvent,
  // Clients
  ClaimsClient,
  GraphClient,
  QueryClient,
  InferenceClient,
  KnowledgeBridgeClient,
  // Factory
  createKnowledgeClients,
} from '../../knowledge/index.js';

/**
 * Knowledge Client - Unified client for mycelix-knowledge hApp
 *
 * Total: ~30 functions covered across 5 zome clients
 * - claims: Submit, get, update, challenge claims (10 functions)
 * - graph: Relationships, ontologies, concepts, paths (9 functions)
 * - query: Execute queries, save queries (4 functions)
 * - inference: Create inferences, credibility, patterns (7 functions)
 * - bridge: Cross-hApp knowledge queries, fact-checking (5 functions)
 */
import {
  ClaimsClient,
  GraphClient,
  QueryClient,
  InferenceClient,
  KnowledgeBridgeClient,
  type ZomeCallable,
} from '../../knowledge/index.js';

import type { AppClient } from '@holochain/client';

export class KnowledgeClient {
  /** Epistemic claims management */
  readonly claims: ClaimsClient;

  /** Knowledge graph relationships and ontologies */
  readonly graph: GraphClient;

  /** Query engine for claim discovery */
  readonly query: QueryClient;

  /** ML-powered inference and credibility scoring */
  readonly inference: InferenceClient;

  /** Cross-hApp knowledge bridge */
  readonly bridge: KnowledgeBridgeClient;

  constructor(
    client: AppClient | ZomeCallable,
    _appId: string = 'knowledge'
  ) {
    const callable = client as ZomeCallable;
    this.claims = new ClaimsClient(callable);
    this.graph = new GraphClient(callable);
    this.query = new QueryClient(callable);
    this.inference = new InferenceClient(callable);
    this.bridge = new KnowledgeBridgeClient(callable);
  }
}
