// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Knowledge Validated Clients
 *
 * Provides input-validated versions of all Knowledge clients.
 * All inputs are validated using Zod schemas before being sent to the conductor.
 *
 * @packageDocumentation
 * @module knowledge/validated
 */

import { z } from 'zod';

import { MycelixError, ErrorCode } from '../errors.js';

import {
  ClaimsClient,
  GraphClient,
  QueryClient,
  InferenceClient,
  KnowledgeBridgeClient,
  type ZomeCallable,
  type Claim,
  type UpdateClaimInput,
  type Evidence,
  type ChallengeClaimInput,
  type EpistemicRangeSearch,
  type Relationship,
  type Ontology,
  type Concept,
  type FindPathInput,
  type SavedQuery,
  type ExecuteQueryInput,
  type QueryExecutionResult,
  type Inference,
  type AssessCredibilityInput,
  type DetectPatternsInput,
  type VerifyInferenceInput,
  type QueryKnowledgeInput,
  type FactCheckInput,
  type RegisterClaimInput,
  type BroadcastKnowledgeEventInput,
  type HolochainRecord,
  type GraphStats,
  type CredibilityScore,
  type Pattern,
  type QueryKnowledgeResult,
  type FactCheckResult,
  type KnowledgeBridgeEvent,
  type ClaimChallenge,
} from './index.js';

// ============================================================================
// Validation Schemas
// ============================================================================

const didSchema = z
  .string()
  .refine((val) => val.startsWith('did:'), { message: 'Must be a valid DID (start with "did:")' });

const epistemicValueSchema = z.number().min(0).max(1);

const epistemicPositionSchema = z.object({
  empirical: epistemicValueSchema,
  normative: epistemicValueSchema,
  mythic: epistemicValueSchema,
});

const claimTypeSchema = z.enum([
  'Fact',
  'Opinion',
  'Prediction',
  'Hypothesis',
  'Definition',
  'Historical',
  'Normative',
  'Narrative',
]);

const claimSchema = z.object({
  id: z.string().min(1),
  content: z.string().min(1, 'Claim content is required'),
  classification: epistemicPositionSchema,
  author: didSchema,
  sources: z.array(z.string()),
  tags: z.array(z.string()),
  claim_type: claimTypeSchema,
  confidence: z.number().min(0).max(1),
  expires: z.number().positive().optional(),
  created: z.number().positive(),
  updated: z.number().positive(),
  version: z.number().min(1),
});

const updateClaimInputSchema = z.object({
  claim_id: z.string().min(1),
  content: z.string().min(1).optional(),
  classification: epistemicPositionSchema.optional(),
  sources: z.array(z.string()).optional(),
  tags: z.array(z.string()).optional(),
  confidence: z.number().min(0).max(1).optional(),
  expires: z.number().positive().optional(),
});

const evidenceTypeSchema = z.enum([
  'Observation',
  'Experiment',
  'Statistical',
  'Expert',
  'Document',
  'CrossReference',
  'Logical',
]);

const evidenceSchema = z.object({
  id: z.string().min(1),
  claim_id: z.string().min(1),
  evidence_type: evidenceTypeSchema,
  source_uri: z.string().min(1),
  content: z.string().min(1),
  strength: z.number().min(0).max(1),
  submitted_by: didSchema,
  submitted_at: z.number().positive(),
});

const challengeClaimInputSchema = z.object({
  claim_id: z.string().min(1),
  challenger_did: didSchema,
  reason: z.string().min(1, 'Challenge reason is required'),
  counter_evidence: z.array(z.string()),
});

const epistemicRangeSearchSchema = z
  .object({
    e_min: epistemicValueSchema,
    e_max: epistemicValueSchema,
    n_min: epistemicValueSchema,
    n_max: epistemicValueSchema,
    m_min: epistemicValueSchema,
    m_max: epistemicValueSchema,
  })
  .refine(
    (data) => data.e_min <= data.e_max && data.n_min <= data.n_max && data.m_min <= data.m_max,
    { message: 'Min values must be <= max values' }
  );

const relationshipTypeSchema = z.union([
  z.enum([
    'Supports',
    'Contradicts',
    'DerivedFrom',
    'ExampleOf',
    'Generalizes',
    'PartOf',
    'Causes',
    'RelatedTo',
    'Equivalent',
    'SpecializedFrom',
  ]),
  z.object({ Custom: z.string().min(1) }),
]);

const relationshipSchema = z.object({
  id: z.string().min(1),
  source: z.string().min(1),
  target: z.string().min(1),
  relationship_type: relationshipTypeSchema,
  weight: z.number().min(0).max(1),
  properties: z.string().optional(),
  creator: didSchema,
  created: z.number().positive(),
});

const ontologySchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1, 'Ontology name is required'),
  description: z.string(),
  namespace: z.string().min(1),
  schema: z.string().min(1),
  version: z.string().min(1),
  creator: didSchema,
  created: z.number().positive(),
  updated: z.number().positive(),
});

const conceptSchema = z.object({
  id: z.string().min(1),
  ontology_id: z.string().min(1),
  name: z.string().min(1),
  definition: z.string().min(1),
  parent: z.string().optional(),
  synonyms: z.array(z.string()),
  created: z.number().positive(),
});

const findPathInputSchema = z.object({
  source: z.string().min(1),
  target: z.string().min(1),
  max_depth: z.number().min(1).max(20),
});

const savedQuerySchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1, 'Query name is required'),
  description: z.string(),
  query: z.string().min(1, 'Query string is required'),
  parameters: z.string().optional(),
  creator: didSchema,
  public: z.boolean(),
  created: z.number().positive(),
  updated: z.number().positive(),
});

const executeQueryInputSchema = z.object({
  query: z.string().min(1, 'Query is required'),
  parameters: z.string().optional(),
  use_cache: z.boolean(),
  limit: z.number().min(1).max(10000).optional(),
  offset: z.number().min(0).optional(),
});

const inferenceTypeSchema = z.enum([
  'Implication',
  'Contradiction',
  'Causation',
  'Correlation',
  'Generalization',
  'Specialization',
]);

const inferenceSchema = z.object({
  id: z.string().min(1),
  inference_type: inferenceTypeSchema,
  source_claims: z.array(z.string()).min(1, 'At least one source claim required'),
  conclusion: z.string().min(1),
  confidence: z.number().min(0).max(1),
  reasoning: z.string().min(1),
  model: z.string().min(1),
  created: z.number().positive(),
  verified: z.boolean(),
});

const credibilitySubjectTypeSchema = z.enum(['Claim', 'Author', 'Source', 'Evidence']);

const assessCredibilityInputSchema = z.object({
  subject: z.string().min(1),
  subject_type: credibilitySubjectTypeSchema,
  expires_at: z.number().positive().optional(),
});

const patternTypeSchema = z.enum(['Cluster', 'Trend', 'CausalChain', 'Contradiction', 'Emergence']);

const detectPatternsInputSchema = z.object({
  claims: z.array(z.string()).min(2, 'At least 2 claims required for pattern detection'),
  pattern_types: z.array(patternTypeSchema).optional(),
});

const verifyInferenceInputSchema = z.object({
  inference_id: z.string().min(1),
  is_correct: z.boolean(),
  verifier_did: didSchema,
  comment: z.string().optional(),
});

const knowledgeQueryTypeSchema = z.enum([
  'VerifyClaim',
  'ClaimsBySubject',
  'EpistemicScore',
  'GraphTraversal',
  'FactCheck',
]);

const knowledgeEventTypeSchema = z.enum([
  'ClaimCreated',
  'ClaimUpdated',
  'ClaimDisputed',
  'ClaimVerified',
  'GraphUpdated',
  'InferenceCompleted',
]);

const queryKnowledgeInputSchema = z.object({
  source_happ: z.string().min(1),
  query_type: knowledgeQueryTypeSchema,
  parameters: z.record(z.string(), z.unknown()),
});

const factCheckInputSchema = z.object({
  source_happ: z.string().min(1),
  statement: z.string().min(1, 'Statement to fact-check is required'),
  context: z.string().optional(),
});

const registerClaimInputSchema = z.object({
  source_happ: z.string().min(1),
  subject: z.string().min(1),
  predicate: z.string().min(1),
  object: z.string().min(1),
  epistemic_e: epistemicValueSchema.optional(),
  epistemic_n: epistemicValueSchema.optional(),
  epistemic_m: epistemicValueSchema.optional(),
});

const broadcastKnowledgeEventInputSchema = z.object({
  event_type: knowledgeEventTypeSchema,
  claim_id: z.string().optional(),
  subject: z.string().min(1),
  payload: z.string(),
});

// ============================================================================
// Validation Utility
// ============================================================================

function validateOrThrow<T>(schema: z.ZodSchema<T>, data: unknown, context: string): T {
  const result = schema.safeParse(data);
  if (!result.success) {
    const errors = result.error.issues.map((e) => `${e.path.join('.')}: ${e.message}`).join('; ');
    throw new MycelixError(
      `Validation failed for ${context}: ${errors}`,
      ErrorCode.INVALID_ARGUMENT
    );
  }
  return result.data;
}

// ============================================================================
// Validated Claims Client
// ============================================================================

/**
 * Validated Claims Client - All inputs are validated before zome calls
 */
export class ValidatedClaimsClient {
  private client: ClaimsClient;

  constructor(zomeClient: ZomeCallable) {
    this.client = new ClaimsClient(zomeClient);
  }

  async submitClaim(claim: Claim): Promise<HolochainRecord<Claim>> {
    validateOrThrow(claimSchema, claim, 'claim');
    return this.client.submitClaim(claim);
  }

  async getClaim(claimId: string): Promise<HolochainRecord<Claim> | null> {
    validateOrThrow(z.string().min(1), claimId, 'claimId');
    return this.client.getClaim(claimId);
  }

  async getClaimsByAuthor(authorDid: string): Promise<HolochainRecord<Claim>[]> {
    validateOrThrow(didSchema, authorDid, 'authorDid');
    return this.client.getClaimsByAuthor(authorDid);
  }

  async getClaimsByTag(tag: string): Promise<HolochainRecord<Claim>[]> {
    validateOrThrow(z.string().min(1), tag, 'tag');
    return this.client.getClaimsByTag(tag);
  }

  async updateClaim(input: UpdateClaimInput): Promise<HolochainRecord<Claim>> {
    validateOrThrow(updateClaimInputSchema, input, 'update claim input');
    return this.client.updateClaim(input);
  }

  async addEvidence(evidence: Evidence): Promise<HolochainRecord<Evidence>> {
    validateOrThrow(evidenceSchema, evidence, 'evidence');
    return this.client.addEvidence(evidence);
  }

  async getClaimEvidence(claimId: string): Promise<HolochainRecord<Evidence>[]> {
    validateOrThrow(z.string().min(1), claimId, 'claimId');
    return this.client.getClaimEvidence(claimId);
  }

  async challengeClaim(input: ChallengeClaimInput): Promise<HolochainRecord<ClaimChallenge>> {
    validateOrThrow(challengeClaimInputSchema, input, 'challenge claim input');
    return this.client.challengeClaim(input);
  }

  async getClaimChallenges(claimId: string): Promise<HolochainRecord<ClaimChallenge>[]> {
    validateOrThrow(z.string().min(1), claimId, 'claimId');
    return this.client.getClaimChallenges(claimId);
  }

  async searchByEpistemicRange(input: EpistemicRangeSearch): Promise<HolochainRecord<Claim>[]> {
    validateOrThrow(epistemicRangeSearchSchema, input, 'epistemic range search');
    return this.client.searchByEpistemicRange(input);
  }
}

// ============================================================================
// Validated Graph Client
// ============================================================================

/**
 * Validated Graph Client
 */
export class ValidatedGraphClient {
  private client: GraphClient;

  constructor(zomeClient: ZomeCallable) {
    this.client = new GraphClient(zomeClient);
  }

  async createRelationship(relationship: Relationship): Promise<HolochainRecord<Relationship>> {
    validateOrThrow(relationshipSchema, relationship, 'relationship');
    return this.client.createRelationship(relationship);
  }

  async getOutgoingRelationships(claimId: string): Promise<HolochainRecord<Relationship>[]> {
    validateOrThrow(z.string().min(1), claimId, 'claimId');
    return this.client.getOutgoingRelationships(claimId);
  }

  async getIncomingRelationships(claimId: string): Promise<HolochainRecord<Relationship>[]> {
    validateOrThrow(z.string().min(1), claimId, 'claimId');
    return this.client.getIncomingRelationships(claimId);
  }

  async findPath(input: FindPathInput): Promise<string[]> {
    validateOrThrow(findPathInputSchema, input, 'find path input');
    return this.client.findPath(input);
  }

  async createOntology(ontology: Ontology): Promise<HolochainRecord<Ontology>> {
    validateOrThrow(ontologySchema, ontology, 'ontology');
    return this.client.createOntology(ontology);
  }

  async createConcept(concept: Concept): Promise<HolochainRecord<Concept>> {
    validateOrThrow(conceptSchema, concept, 'concept');
    return this.client.createConcept(concept);
  }

  async getOntologyConcepts(ontologyId: string): Promise<HolochainRecord<Concept>[]> {
    validateOrThrow(z.string().min(1), ontologyId, 'ontologyId');
    return this.client.getOntologyConcepts(ontologyId);
  }

  async getChildConcepts(conceptId: string): Promise<HolochainRecord<Concept>[]> {
    validateOrThrow(z.string().min(1), conceptId, 'conceptId');
    return this.client.getChildConcepts(conceptId);
  }

  async getGraphStats(): Promise<GraphStats> {
    return this.client.getGraphStats();
  }
}

// ============================================================================
// Validated Query Client
// ============================================================================

/**
 * Validated Query Client
 */
export class ValidatedQueryClient {
  private client: QueryClient;

  constructor(zomeClient: ZomeCallable) {
    this.client = new QueryClient(zomeClient);
  }

  async executeQuery(input: ExecuteQueryInput): Promise<QueryExecutionResult> {
    validateOrThrow(executeQueryInputSchema, input, 'execute query input');
    return this.client.executeQuery(input);
  }

  async saveQuery(query: SavedQuery): Promise<HolochainRecord<SavedQuery>> {
    validateOrThrow(savedQuerySchema, query, 'saved query');
    return this.client.saveQuery(query);
  }

  async getMyQueries(creatorDid: string): Promise<HolochainRecord<SavedQuery>[]> {
    validateOrThrow(didSchema, creatorDid, 'creatorDid');
    return this.client.getMyQueries(creatorDid);
  }

  async getPublicQueries(): Promise<HolochainRecord<SavedQuery>[]> {
    return this.client.getPublicQueries();
  }
}

// ============================================================================
// Validated Inference Client
// ============================================================================

/**
 * Validated Inference Client
 */
export class ValidatedInferenceClient {
  private client: InferenceClient;

  constructor(zomeClient: ZomeCallable) {
    this.client = new InferenceClient(zomeClient);
  }

  async createInference(inference: Inference): Promise<HolochainRecord<Inference>> {
    validateOrThrow(inferenceSchema, inference, 'inference');
    return this.client.createInference(inference);
  }

  async getClaimInferences(claimId: string): Promise<HolochainRecord<Inference>[]> {
    validateOrThrow(z.string().min(1), claimId, 'claimId');
    return this.client.getClaimInferences(claimId);
  }

  async assessCredibility(
    input: AssessCredibilityInput
  ): Promise<HolochainRecord<CredibilityScore>> {
    validateOrThrow(assessCredibilityInputSchema, input, 'assess credibility input');
    return this.client.assessCredibility(input);
  }

  async getCredibility(subject: string): Promise<HolochainRecord<CredibilityScore> | null> {
    validateOrThrow(z.string().min(1), subject, 'subject');
    return this.client.getCredibility(subject);
  }

  async detectPatterns(input: DetectPatternsInput): Promise<HolochainRecord<Pattern>[]> {
    validateOrThrow(detectPatternsInputSchema, input, 'detect patterns input');
    return this.client.detectPatterns(input);
  }

  async findContradictions(claimIds: string[]): Promise<HolochainRecord<Inference>[]> {
    validateOrThrow(z.array(z.string().min(1)).min(2), claimIds, 'claimIds');
    return this.client.findContradictions(claimIds);
  }

  async verifyInference(input: VerifyInferenceInput): Promise<HolochainRecord<Inference>> {
    validateOrThrow(verifyInferenceInputSchema, input, 'verify inference input');
    return this.client.verifyInference(input);
  }
}

// ============================================================================
// Validated Knowledge Bridge Client
// ============================================================================

/**
 * Validated Knowledge Bridge Client
 */
export class ValidatedKnowledgeBridgeClient {
  private client: KnowledgeBridgeClient;

  constructor(zomeClient: ZomeCallable) {
    this.client = new KnowledgeBridgeClient(zomeClient);
  }

  async queryKnowledge(input: QueryKnowledgeInput): Promise<QueryKnowledgeResult> {
    validateOrThrow(queryKnowledgeInputSchema, input, 'query knowledge input');
    return this.client.queryKnowledge(input);
  }

  async factCheck(input: FactCheckInput): Promise<FactCheckResult> {
    validateOrThrow(factCheckInputSchema, input, 'fact check input');
    return this.client.factCheck(input);
  }

  async registerExternalClaim(input: RegisterClaimInput): Promise<HolochainRecord<unknown>> {
    validateOrThrow(registerClaimInputSchema, input, 'register claim input');
    return this.client.registerExternalClaim(input);
  }

  async broadcastKnowledgeEvent(
    input: BroadcastKnowledgeEventInput
  ): Promise<HolochainRecord<KnowledgeBridgeEvent>> {
    validateOrThrow(broadcastKnowledgeEventInputSchema, input, 'broadcast knowledge event input');
    return this.client.broadcastKnowledgeEvent(input);
  }

  async getRecentKnowledgeEvents(since?: number): Promise<HolochainRecord<KnowledgeBridgeEvent>[]> {
    if (since !== undefined) {
      validateOrThrow(z.number().positive(), since, 'since');
    }
    return this.client.getRecentKnowledgeEvents(since);
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create all validated Knowledge hApp clients
 */
export function createValidatedKnowledgeClients(client: ZomeCallable) {
  return {
    claims: new ValidatedClaimsClient(client),
    graph: new ValidatedGraphClient(client),
    query: new ValidatedQueryClient(client),
    inference: new ValidatedInferenceClient(client),
    bridge: new ValidatedKnowledgeBridgeClient(client),
  };
}
