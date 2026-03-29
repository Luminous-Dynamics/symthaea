// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Knowledge Integration
 *
 * hApp-specific adapter for Mycelix-Knowledge providing:
 * - Distributed knowledge base with epistemic classification
 * - Claim verification and evidence tracking
 * - Community knowledge curation with reputation
 * - Cross-hApp knowledge sharing via Bridge
 * - Federated Learning for knowledge synthesis
 *
 * @packageDocumentation
 * @module integrations/knowledge
 */



import { LocalBridge } from '../../bridge/index.js';
import { type MycelixClient } from '../../client/index.js';
import {
  EmpiricalLevel,
  type NormativeLevel,
  MaterialityLevel,
  type EpistemicClassification,
} from '../../epistemic/index.js';
import {
  createReputation,
  recordPositive,
  recordNegative,
  type ReputationScore,
} from '../../matl/index.js';
import { KnowledgeValidators } from '../../utils/validation.js';

// ============================================================================
// Bridge Zome Types (matching Rust knowledge_bridge zome)
// ============================================================================

/** Knowledge query types */
export type KnowledgeQueryType =
  | 'VerifyClaim'
  | 'ClaimsBySubject'
  | 'EpistemicScore'
  | 'GraphTraversal'
  | 'FactCheck';

/** Knowledge query from another hApp */
export interface KnowledgeQuery {
  id: string;
  query_type: KnowledgeQueryType;
  query_params: string;
  source_happ: string;
  queried_at: number;
}

/** Epistemic result with 3D classification */
export interface EpistemicResult {
  id: string;
  query_id: string;
  empirical: number;
  normative: number;
  mythic: number;
  confidence: number;
  source_claims: string[];
  calculated_at: number;
}

/** Claim reference from another hApp */
export interface ClaimReference {
  id: string;
  claim_hash: string;
  source_happ: string;
  title: string;
  empirical: number;
  normative: number;
  mythic: number;
  author_did: string;
  created_at: number;
}

/** Knowledge bridge event types */
export type KnowledgeBridgeEventType =
  | 'ClaimCreated'
  | 'ClaimVerified'
  | 'ClaimDisputed'
  | 'EvidenceAdded'
  | 'SynthesisCompleted'
  | 'FactCheckResult';

/** Knowledge bridge event */
export interface KnowledgeBridgeEvent {
  id: string;
  event_type: KnowledgeBridgeEventType;
  claim_hash?: string;
  subject?: string;
  payload: string;
  source_happ: string;
  timestamp: number;
}

/** Query knowledge input */
export interface QueryKnowledgeInput {
  query_type: KnowledgeQueryType;
  query_params: string;
  source_happ: string;
}

/** Fact check input */
export interface FactCheckInput {
  claim_text: string;
  source_happ: string;
}

/** Fact check result */
export interface FactCheckResult {
  claim_text: string;
  verified: boolean;
  confidence: number;
  supporting_claims: string[];
  contradicting_claims: string[];
  epistemic_score: EpistemicResult;
}

/** Register external claim input */
export interface RegisterExternalClaimInput {
  claim_hash: string;
  source_happ: string;
  title: string;
  empirical: number;
  normative: number;
  mythic: number;
  author_did: string;
}

/** Broadcast knowledge event input */
export interface BroadcastKnowledgeEventInput {
  event_type: KnowledgeBridgeEventType;
  claim_hash?: string;
  subject?: string;
  payload: string;
}

// ============================================================================
// Knowledge-Specific Types
// ============================================================================

/** Knowledge entry type */
export type KnowledgeType = 'claim' | 'evidence' | 'synthesis' | 'question' | 'answer';

/** Knowledge status */
export type KnowledgeStatus = 'proposed' | 'verified' | 'disputed' | 'deprecated';

/** Knowledge entry */
export interface KnowledgeEntry {
  id: string;
  type: KnowledgeType;
  title: string;
  content: string;
  authorId: string;
  classification: EpistemicClassification;
  status: KnowledgeStatus;
  citations: string[];
  endorsements: number;
  challenges: number;
  tags: string[];
  createdAt: number;
  updatedAt: number;
}

/** Evidence submission */
export interface EvidenceSubmission {
  id: string;
  claimId: string;
  submitterId: string;
  evidenceType: 'empirical' | 'testimonial' | 'analytical' | 'cryptographic';
  content: string;
  sourceUrl?: string;
  verificationMethod?: string;
  timestamp: number;
}

/** Knowledge contributor profile */
export interface ContributorProfile {
  did: string;
  reputation: ReputationScore;
  contributions: number;
  endorsementsReceived: number;
  accuracyScore: number;
  specializations: string[];
  joinedAt: number;
}

/** Synthesis request */
export interface SynthesisRequest {
  id: string;
  topic: string;
  sourceClaimIds: string[];
  requesterId: string;
  status: 'pending' | 'in_progress' | 'completed';
  synthesisId?: string;
  createdAt: number;
}

// ============================================================================
// Knowledge Service
// ============================================================================

/**
 * Knowledge service for distributed knowledge management
 */
export class KnowledgeService {
  private entries = new Map<string, KnowledgeEntry>();
  private evidence = new Map<string, EvidenceSubmission[]>();
  private contributors = new Map<string, ContributorProfile>();
  private syntheses = new Map<string, SynthesisRequest>();
  private bridge: LocalBridge;

  constructor() {
    this.bridge = new LocalBridge();
    this.bridge.registerHapp('knowledge');
  }

  /**
   * Submit a knowledge claim
   */
  submitClaim(
    authorId: string,
    title: string,
    content: string,
    empiricalLevel: EmpiricalLevel,
    normativeLevel: NormativeLevel,
    tags: string[] = []
  ): KnowledgeEntry {
    // [FIXED] Validate non-empty title and content
    KnowledgeValidators.claim(title, content);

    const id = `claim-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

    const classification: EpistemicClassification = {
      empirical: empiricalLevel,
      normative: normativeLevel,
      materiality: MaterialityLevel.M2_Persistent,
    };

    const entry: KnowledgeEntry = {
      id,
      type: 'claim',
      title,
      content,
      authorId,
      classification,
      status: 'proposed',
      citations: [],
      endorsements: 0,
      challenges: 0,
      tags,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };

    this.entries.set(id, entry);
    this.evidence.set(id, []);

    // Update contributor stats
    this.ensureContributor(authorId);
    const contributor = this.contributors.get(authorId)!;
    contributor.contributions++;

    return entry;
  }

  /**
   * Submit evidence for a claim
   */
  submitEvidence(
    claimId: string,
    submitterId: string,
    evidenceType: EvidenceSubmission['evidenceType'],
    content: string,
    sourceUrl?: string
  ): EvidenceSubmission {
    // [FIXED] Validate source URL when provided
    if (sourceUrl !== undefined && sourceUrl.trim() !== '') {
      KnowledgeValidators.evidence(sourceUrl);
    }

    const claim = this.entries.get(claimId);
    if (!claim) throw new Error('Claim not found');

    const submission: EvidenceSubmission = {
      id: `evidence-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      claimId,
      submitterId,
      evidenceType,
      content,
      sourceUrl,
      timestamp: Date.now(),
    };

    const existingEvidence = this.evidence.get(claimId) || [];
    existingEvidence.push(submission);
    this.evidence.set(claimId, existingEvidence);

    // Update claim empirical level based on evidence
    if (evidenceType === 'cryptographic') {
      claim.classification.empirical = EmpiricalLevel.E3_Cryptographic;
    } else if (evidenceType === 'empirical' && claim.classification.empirical < EmpiricalLevel.E2_PrivateVerify) {
      claim.classification.empirical = EmpiricalLevel.E2_PrivateVerify;
    }

    // Contributor reputation boost
    this.ensureContributor(submitterId);
    const contributor = this.contributors.get(submitterId)!;
    contributor.reputation = recordPositive(contributor.reputation);

    return submission;
  }

  /**
   * Endorse a knowledge claim
   */
  endorseClaim(claimId: string, endorserId: string): void {
    // [FIXED] Validate endorser DID
    KnowledgeValidators.endorsement(endorserId);

    const claim = this.entries.get(claimId);
    if (!claim) throw new Error('Claim not found');

    claim.endorsements++;
    claim.updatedAt = Date.now();

    // Update author reputation
    const author = this.contributors.get(claim.authorId);
    if (author) {
      author.reputation = recordPositive(author.reputation);
      author.endorsementsReceived++;
    }

    // Check for verification threshold
    if (claim.endorsements >= 5 && claim.status === 'proposed') {
      claim.status = 'verified';
    }
  }

  /**
   * Challenge a knowledge claim
   */
  challengeClaim(claimId: string, _challengerId: string, _reason: string): void {
    const claim = this.entries.get(claimId);
    if (!claim) throw new Error('Claim not found');

    claim.challenges++;
    claim.updatedAt = Date.now();

    // Mark as disputed if significant challenges
    if (claim.challenges >= 3 && claim.status === 'verified') {
      claim.status = 'disputed';
    }

    // Update author reputation (slight penalty)
    const author = this.contributors.get(claim.authorId);
    if (author && claim.challenges > claim.endorsements) {
      author.reputation = recordNegative(author.reputation);
    }
  }

  /**
   * Request knowledge synthesis
   */
  requestSynthesis(topic: string, sourceClaimIds: string[], requesterId: string): SynthesisRequest {
    const request: SynthesisRequest = {
      id: `synthesis-req-${Date.now()}`,
      topic,
      sourceClaimIds,
      requesterId,
      status: 'pending',
      createdAt: Date.now(),
    };
    this.syntheses.set(request.id, request);
    return request;
  }

  /**
   * Search knowledge base
   */
  search(query: string, tags?: string[]): KnowledgeEntry[] {
    const queryLower = query.toLowerCase();
    return Array.from(this.entries.values()).filter((entry) => {
      const matchesQuery =
        entry.title.toLowerCase().includes(queryLower) ||
        entry.content.toLowerCase().includes(queryLower);
      const matchesTags = !tags || tags.some((tag) => entry.tags.includes(tag));
      return matchesQuery && matchesTags;
    });
  }

  /**
   * Get claim by ID
   */
  getClaim(claimId: string): KnowledgeEntry | undefined {
    return this.entries.get(claimId);
  }

  /**
   * Get evidence for a claim
   */
  getEvidence(claimId: string): EvidenceSubmission[] {
    return this.evidence.get(claimId) || [];
  }

  /**
   * Get contributor profile
   */
  getContributor(did: string): ContributorProfile | undefined {
    return this.contributors.get(did);
  }

  private ensureContributor(did: string): void {
    if (!this.contributors.has(did)) {
      this.contributors.set(did, {
        did,
        reputation: createReputation(did),
        contributions: 0,
        endorsementsReceived: 0,
        accuracyScore: 1.0,
        specializations: [],
        joinedAt: Date.now(),
      });
    }
  }
}

// Singleton
let instance: KnowledgeService | null = null;

export function getKnowledgeService(): KnowledgeService {
  if (!instance) instance = new KnowledgeService();
  return instance;
}

export function resetKnowledgeService(): void {
  instance = null;
}

// ============================================================================
// Knowledge Bridge Client (Holochain Zome Calls)
// ============================================================================

const KNOWLEDGE_ROLE = 'knowledge';
const BRIDGE_ZOME = 'knowledge_bridge';

/**
 * Knowledge Bridge Client - Direct Holochain zome calls for cross-hApp knowledge
 */
export class KnowledgeBridgeClient {
  constructor(private client: MycelixClient) {}

  /**
   * Query knowledge from the knowledge graph
   */
  async queryKnowledge(input: QueryKnowledgeInput): Promise<EpistemicResult> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'query_knowledge',
      payload: input,
    });
  }

  /**
   * Fact-check a claim against the knowledge graph
   */
  async factCheck(input: FactCheckInput): Promise<FactCheckResult> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'fact_check',
      payload: input,
    });
  }

  /**
   * Register an external claim from another hApp
   */
  async registerExternalClaim(input: RegisterExternalClaimInput): Promise<ClaimReference> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'register_external_claim',
      payload: input,
    });
  }

  /**
   * Get claims by subject
   */
  async getClaimsBySubject(subject: string): Promise<ClaimReference[]> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_claims_by_subject',
      payload: subject,
    });
  }

  /**
   * Get claims from a specific hApp
   */
  async getClaimsByHapp(sourceHapp: string): Promise<ClaimReference[]> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_claims_by_happ',
      payload: sourceHapp,
    });
  }

  /**
   * Broadcast a knowledge event
   */
  async broadcastKnowledgeEvent(input: BroadcastKnowledgeEventInput): Promise<KnowledgeBridgeEvent> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'broadcast_knowledge_event',
      payload: input,
    });
  }

  /**
   * Get recent knowledge events
   */
  async getRecentEvents(limit?: number): Promise<KnowledgeBridgeEvent[]> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_recent_events',
      payload: limit ?? 50,
    });
  }

  /**
   * Get events by subject
   */
  async getEventsBySubject(subject: string): Promise<KnowledgeBridgeEvent[]> {
    return this.client.callZome({
      role_name: KNOWLEDGE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_events_by_subject',
      payload: subject,
    });
  }
}

// Bridge client singleton
let bridgeInstance: KnowledgeBridgeClient | null = null;

export function getKnowledgeBridgeClient(client: MycelixClient): KnowledgeBridgeClient {
  if (!bridgeInstance) bridgeInstance = new KnowledgeBridgeClient(client);
  return bridgeInstance;
}

export function resetKnowledgeBridgeClient(): void {
  bridgeInstance = null;
}

// ============================================================================
// Holochain Zome Clients (New SDK)
// ============================================================================

/**
 * New Holochain-based Knowledge SDK
 *
 * These clients provide direct zome calls to the knowledge Holochain hApp.
 * Use MycelixKnowledgeClient for the unified interface.
 *
 * @example
 * ```typescript
 * import { MycelixKnowledgeClient } from '@mycelix/sdk/integrations/knowledge';
 *
 * const knowledge = await MycelixKnowledgeClient.connect({
 *   url: 'ws://localhost:8888',
 * });
 *
 * // Submit an epistemic claim
 * const claim = await knowledge.claims.submitClaim({
 *   id: 'claim-001',
 *   author_did: 'did:mycelix:alice',
 *   title: 'Climate Evidence',
 *   content: 'Global temperatures have risen...',
 *   empirical: 0.95,
 *   normative: 0.05,
 *   metaphysical: 0.0,
 *   tags: ['climate', 'science'],
 * });
 *
 * // Traverse the knowledge graph
 * const result = await knowledge.graph.traverseGraph({
 *   start_claim_id: 'claim-001',
 *   direction: 'Both',
 *   max_depth: 3,
 *   limit: 100,
 * });
 * ```
 */
export { MycelixKnowledgeClient } from './client';
export type {
  KnowledgeClientConfig,
  KnowledgeConnectionOptions,
} from './client';

// Zome clients for advanced usage
export {
  ClaimsClient,
  GraphClient,
  FactCheckClient,
} from './zomes';
export type {
  ClaimsClientConfig,
  GraphClientConfig,
  FactCheckClientConfig,
} from './zomes';

// Types from the new SDK
export type {
  // Claim types
  Claim as ZomeClaim,
  SubmitClaimInput,
  ClaimUpdate,
  ClaimEndorsement as ZomeEndorsement,
  Ontology as ZomeOntology,
  // Relation types
  ClaimRelation,
  RelationType,
  // Source/Citation types
  Source,
  SourceType,
  Citation,
  CitationType,
  AddSourceInput,
  CiteSourceInput,
  // Inference types
  InferenceRequest,
  InferenceType,
  InferenceStatus,
  ConsistencyResult,
  ClaimConflict,
  ReasoningInput,
  InferenceRule,
  ReasoningResult,
  // Graph traversal types
  TraversalInput,
  TraversalDirection,
  TraversalResult,
  ClaimPath,
  PathInput,
  ClaimCluster,
  // Search types
  GraphQuery,
  SemanticSearchInput,
  SearchFilters,
  SearchResult,
  // Fact-check types
  FactCheckRequest as ZomeFactCheckRequest,
  FactCheckStatus,
  FactCheckResult as ZomeFactCheckResult,
  FactCheckVerdict,
  SubmitFactCheckInput,
  CompleteFactCheckInput,
  // Cross-hApp types
  CrossHappLink,
  CrossHappLinkType,
  LinkExternalInput,
  GetLinkedClaimsInput,
  // Authority types
  KnowledgeAuthority,
  DomainExpertise,
  ExpertiseLevel,
  ContradictionReport,
  PotentialContradiction,
  ContradictionType,
  // Error types
  KnowledgeSdkErrorCode,
} from './types';

export { KnowledgeSdkError } from './types';
