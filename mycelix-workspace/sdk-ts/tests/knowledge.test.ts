// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Knowledge Module Tests
 *
 * Tests for ClaimsClient, GraphClient, QueryClient, InferenceClient, and KnowledgeBridgeClient
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  ClaimsClient,
  GraphClient,
  QueryClient,
  InferenceClient,
  KnowledgeBridgeClient,
  createKnowledgeClients,
  type ZomeCallable,
  type Claim,
  type Evidence,
  type Relationship,
  type Ontology,
  type Concept,
  type SavedQuery,
  type Inference,
  type CredibilityScore,
  type Pattern,
  type EpistemicPosition,
} from '../src/knowledge/index.js';

// Mock ZomeCallable implementation
function createMockClient(): ZomeCallable {
  return {
    callZome: vi.fn().mockResolvedValue({}),
  };
}

describe('Knowledge Module', () => {
  describe('ClaimsClient', () => {
    let mockClient: ZomeCallable;
    let claimsClient: ClaimsClient;

    beforeEach(() => {
      mockClient = createMockClient();
      claimsClient = new ClaimsClient(mockClient);
    });

    const mockClaim: Claim = {
      id: 'claim-1',
      content: 'The Earth is approximately 4.5 billion years old',
      classification: { empirical: 0.95, normative: 0.1, mythic: 0.2 },
      author: 'did:mycelix:abc123',
      sources: ['https://science.nasa.gov'],
      tags: ['geology', 'earth-science'],
      claim_type: 'Fact',
      confidence: 0.99,
      created: Date.now() * 1000,
      updated: Date.now() * 1000,
      version: 1,
    };

    it('should submit a claim', async () => {
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue({
        entry: { Present: mockClaim },
      });

      const result = await claimsClient.submitClaim(mockClaim);

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'claims',
        fn_name: 'submit_claim',
        payload: mockClaim,
      });
      expect(result.entry.Present.id).toBe('claim-1');
    });

    it('should get a claim by ID', async () => {
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue({
        entry: { Present: mockClaim },
      });

      const result = await claimsClient.getClaim('claim-1');

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'claims',
        fn_name: 'get_claim',
        payload: 'claim-1',
      });
      expect(result?.entry.Present.content).toContain('Earth');
    });

    it('should get claims by author', async () => {
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue([
        { entry: { Present: mockClaim } },
      ]);

      const result = await claimsClient.getClaimsByAuthor('did:mycelix:abc123');

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'claims',
        fn_name: 'get_claims_by_author',
        payload: 'did:mycelix:abc123',
      });
      expect(result).toHaveLength(1);
    });

    it('should get claims by tag', async () => {
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue([
        { entry: { Present: mockClaim } },
      ]);

      const result = await claimsClient.getClaimsByTag('geology');

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'claims',
        fn_name: 'get_claims_by_tag',
        payload: 'geology',
      });
      expect(result).toHaveLength(1);
    });

    it('should update a claim', async () => {
      const updated = { ...mockClaim, version: 2 };
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue({
        entry: { Present: updated },
      });

      const result = await claimsClient.updateClaim({
        claim_id: 'claim-1',
        confidence: 0.95,
      });

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'claims',
        fn_name: 'update_claim',
        payload: { claim_id: 'claim-1', confidence: 0.95 },
      });
      expect(result.entry.Present.version).toBe(2);
    });

    it('should add evidence to a claim', async () => {
      const evidence: Evidence = {
        id: 'evidence-1',
        claim_id: 'claim-1',
        evidence_type: 'Document',
        source_uri: 'https://example.com/paper.pdf',
        content: 'Scientific paper supporting the claim',
        strength: 0.9,
        submitted_by: 'did:mycelix:def456',
        submitted_at: Date.now() * 1000,
      };

      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue({
        entry: { Present: evidence },
      });

      const result = await claimsClient.addEvidence(evidence);

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'claims',
        fn_name: 'add_evidence',
        payload: evidence,
      });
      expect(result.entry.Present.id).toBe('evidence-1');
    });

    it('should get claim evidence', async () => {
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue([]);

      const result = await claimsClient.getClaimEvidence('claim-1');

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'claims',
        fn_name: 'get_claim_evidence',
        payload: 'claim-1',
      });
      expect(result).toEqual([]);
    });

    it('should challenge a claim', async () => {
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue({
        entry: { Present: { id: 'challenge-1', status: 'Pending' } },
      });

      const result = await claimsClient.challengeClaim({
        claim_id: 'claim-1',
        challenger_did: 'did:mycelix:xyz789',
        reason: 'Outdated methodology',
        counter_evidence: ['https://newer-study.com'],
      });

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'claims',
        fn_name: 'challenge_claim',
        payload: expect.objectContaining({ claim_id: 'claim-1' }),
      });
      expect(result.entry.Present.status).toBe('Pending');
    });

    it('should get claim challenges', async () => {
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue([]);

      const result = await claimsClient.getClaimChallenges('claim-1');

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'claims',
        fn_name: 'get_claim_challenges',
        payload: 'claim-1',
      });
      expect(result).toEqual([]);
    });

    it('should search by epistemic range', async () => {
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue([
        { entry: { Present: mockClaim } },
      ]);

      const result = await claimsClient.searchByEpistemicRange({
        e_min: 0.8,
        e_max: 1.0,
        n_min: 0.0,
        n_max: 0.5,
        m_min: 0.0,
        m_max: 1.0,
      });

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'claims',
        fn_name: 'search_by_epistemic_range',
        payload: expect.objectContaining({ e_min: 0.8 }),
      });
      expect(result).toHaveLength(1);
    });
  });

  describe('GraphClient', () => {
    let mockClient: ZomeCallable;
    let graphClient: GraphClient;

    beforeEach(() => {
      mockClient = createMockClient();
      graphClient = new GraphClient(mockClient);
    });

    const mockRelationship: Relationship = {
      id: 'rel-1',
      source: 'claim-1',
      target: 'claim-2',
      relationship_type: 'Supports',
      weight: 0.8,
      creator: 'did:mycelix:abc123',
      created: Date.now() * 1000,
    };

    it('should create a relationship', async () => {
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue({
        entry: { Present: mockRelationship },
      });

      const result = await graphClient.createRelationship(mockRelationship);

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'graph',
        fn_name: 'create_relationship',
        payload: mockRelationship,
      });
      expect(result.entry.Present.relationship_type).toBe('Supports');
    });

    it('should get outgoing relationships', async () => {
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue([
        { entry: { Present: mockRelationship } },
      ]);

      const result = await graphClient.getOutgoingRelationships('claim-1');

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'graph',
        fn_name: 'get_outgoing_relationships',
        payload: 'claim-1',
      });
      expect(result).toHaveLength(1);
    });

    it('should get incoming relationships', async () => {
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue([]);

      const result = await graphClient.getIncomingRelationships('claim-2');

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'graph',
        fn_name: 'get_incoming_relationships',
        payload: 'claim-2',
      });
      expect(result).toEqual([]);
    });

    it('should find path between claims', async () => {
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue([
        'claim-1',
        'claim-3',
        'claim-5',
      ]);

      const result = await graphClient.findPath({
        source: 'claim-1',
        target: 'claim-5',
        max_depth: 5,
      });

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'graph',
        fn_name: 'find_path',
        payload: { source: 'claim-1', target: 'claim-5', max_depth: 5 },
      });
      expect(result).toHaveLength(3);
    });

    it('should create an ontology', async () => {
      const ontology: Ontology = {
        id: 'ontology-1',
        name: 'Science Ontology',
        description: 'Ontology for scientific domains',
        namespace: 'https://ontology.mycelix.net/science',
        schema: '{}',
        version: '1.0.0',
        creator: 'did:mycelix:abc123',
        created: Date.now() * 1000,
        updated: Date.now() * 1000,
      };

      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue({
        entry: { Present: ontology },
      });

      const result = await graphClient.createOntology(ontology);

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'graph',
        fn_name: 'create_ontology',
        payload: ontology,
      });
      expect(result.entry.Present.name).toBe('Science Ontology');
    });

    it('should create a concept', async () => {
      const concept: Concept = {
        id: 'concept-1',
        ontology_id: 'ontology-1',
        name: 'Physics',
        definition: 'Study of matter and energy',
        synonyms: ['natural philosophy'],
        created: Date.now() * 1000,
      };

      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue({
        entry: { Present: concept },
      });

      const result = await graphClient.createConcept(concept);

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'graph',
        fn_name: 'create_concept',
        payload: concept,
      });
      expect(result.entry.Present.name).toBe('Physics');
    });

    it('should get ontology concepts', async () => {
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue([]);

      const result = await graphClient.getOntologyConcepts('ontology-1');

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'graph',
        fn_name: 'get_ontology_concepts',
        payload: 'ontology-1',
      });
      expect(result).toEqual([]);
    });

    it('should get child concepts', async () => {
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue([]);

      const result = await graphClient.getChildConcepts('concept-1');

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'graph',
        fn_name: 'get_child_concepts',
        payload: 'concept-1',
      });
      expect(result).toEqual([]);
    });

    it('should get graph statistics', async () => {
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue({
        relationship_count: 1500,
      });

      const result = await graphClient.getGraphStats();

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'graph',
        fn_name: 'get_graph_stats',
        payload: null,
      });
      expect(result.relationship_count).toBe(1500);
    });
  });

  describe('QueryClient', () => {
    let mockClient: ZomeCallable;
    let queryClient: QueryClient;

    beforeEach(() => {
      mockClient = createMockClient();
      queryClient = new QueryClient(mockClient);
    });

    it('should execute a query', async () => {
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue({
        results: ['claim-1', 'claim-2'],
        count: 2,
        execution_time_ms: 15,
        plan: { steps: [], estimated_cost: 1.0, use_cache: true },
      });

      const result = await queryClient.executeQuery({
        query: "SELECT * FROM claims WHERE tag = 'science'",
        use_cache: true,
        limit: 100,
      });

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'query',
        fn_name: 'execute_query',
        payload: expect.objectContaining({ use_cache: true }),
      });
      expect(result.results).toHaveLength(2);
      expect(result.execution_time_ms).toBe(15);
    });

    it('should save a query', async () => {
      const savedQuery: SavedQuery = {
        id: 'query-1',
        name: 'High Empirical Claims',
        description: 'Find claims with high empirical validity',
        query: 'SELECT * FROM claims WHERE e > 0.8',
        creator: 'did:mycelix:abc123',
        public: true,
        created: Date.now() * 1000,
        updated: Date.now() * 1000,
      };

      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue({
        entry: { Present: savedQuery },
      });

      const result = await queryClient.saveQuery(savedQuery);

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'query',
        fn_name: 'save_query',
        payload: savedQuery,
      });
      expect(result.entry.Present.name).toBe('High Empirical Claims');
    });

    it('should get my queries', async () => {
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue([]);

      const result = await queryClient.getMyQueries('did:mycelix:abc123');

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'query',
        fn_name: 'get_my_queries',
        payload: 'did:mycelix:abc123',
      });
      expect(result).toEqual([]);
    });

    it('should get public queries', async () => {
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue([]);

      const result = await queryClient.getPublicQueries();

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'query',
        fn_name: 'get_public_queries',
        payload: null,
      });
      expect(result).toEqual([]);
    });
  });

  describe('InferenceClient', () => {
    let mockClient: ZomeCallable;
    let inferenceClient: InferenceClient;

    beforeEach(() => {
      mockClient = createMockClient();
      inferenceClient = new InferenceClient(mockClient);
    });

    it('should create an inference', async () => {
      const inference: Inference = {
        id: 'inference-1',
        inference_type: 'Implication',
        source_claims: ['claim-1', 'claim-2'],
        conclusion: 'Claim A implies Claim B',
        confidence: 0.85,
        reasoning: 'Logical analysis',
        model: 'simple_inference_v1',
        created: Date.now() * 1000,
        verified: false,
      };

      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue({
        entry: { Present: inference },
      });

      const result = await inferenceClient.createInference(inference);

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'inference',
        fn_name: 'create_inference',
        payload: inference,
      });
      expect(result.entry.Present.inference_type).toBe('Implication');
    });

    it('should get claim inferences', async () => {
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue([]);

      const result = await inferenceClient.getClaimInferences('claim-1');

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'inference',
        fn_name: 'get_claim_inferences',
        payload: 'claim-1',
      });
      expect(result).toEqual([]);
    });

    it('should assess credibility', async () => {
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue({
        entry: {
          Present: {
            id: 'credibility-1',
            subject: 'claim-1',
            subject_type: 'Claim',
            score: 0.75,
            components: {
              accuracy: 0.8,
              consistency: 0.7,
              transparency: 0.7,
              track_record: 0.8,
              corroboration: 0.75,
            },
          },
        },
      });

      const result = await inferenceClient.assessCredibility({
        subject: 'claim-1',
        subject_type: 'Claim',
      });

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'inference',
        fn_name: 'assess_credibility',
        payload: { subject: 'claim-1', subject_type: 'Claim' },
      });
      expect(result.entry.Present.score).toBe(0.75);
    });

    it('should get credibility score', async () => {
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue(null);

      const result = await inferenceClient.getCredibility('claim-1');

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'inference',
        fn_name: 'get_credibility',
        payload: 'claim-1',
      });
      expect(result).toBeNull();
    });

    it('should detect patterns', async () => {
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue([
        {
          entry: {
            Present: {
              id: 'pattern-1',
              pattern_type: 'Cluster',
              description: 'Cluster of 3 related claims',
              claims: ['claim-1', 'claim-2', 'claim-3'],
              strength: 0.3,
            },
          },
        },
      ]);

      const result = await inferenceClient.detectPatterns({
        claims: ['claim-1', 'claim-2', 'claim-3'],
        pattern_types: ['Cluster'],
      });

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'inference',
        fn_name: 'detect_patterns',
        payload: expect.objectContaining({ claims: ['claim-1', 'claim-2', 'claim-3'] }),
      });
      expect(result).toHaveLength(1);
    });

    it('should find contradictions', async () => {
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue([]);

      const result = await inferenceClient.findContradictions(['claim-1', 'claim-2']);

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'inference',
        fn_name: 'find_contradictions',
        payload: ['claim-1', 'claim-2'],
      });
      expect(result).toEqual([]);
    });

    it('should verify an inference', async () => {
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue({
        entry: { Present: { id: 'inference-1', verified: true } },
      });

      const result = await inferenceClient.verifyInference({
        inference_id: 'inference-1',
        is_correct: true,
        verifier_did: 'did:mycelix:abc123',
        comment: 'Verified correct',
      });

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'inference',
        fn_name: 'verify_inference',
        payload: expect.objectContaining({ inference_id: 'inference-1' }),
      });
      expect(result.entry.Present.verified).toBe(true);
    });
  });

  describe('KnowledgeBridgeClient', () => {
    let mockClient: ZomeCallable;
    let bridgeClient: KnowledgeBridgeClient;

    beforeEach(() => {
      mockClient = createMockClient();
      bridgeClient = new KnowledgeBridgeClient(mockClient);
    });

    it('should query knowledge', async () => {
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue({
        success: true,
        data: { claim_id: 'claim-1', exists: true },
      });

      const result = await bridgeClient.queryKnowledge({
        source_happ: 'my-marketplace',
        query_type: 'VerifyClaim',
        parameters: { claim_id: 'claim-1' },
      });

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'knowledge_bridge',
        fn_name: 'query_knowledge',
        payload: expect.objectContaining({ source_happ: 'my-marketplace' }),
      });
      expect(result.success).toBe(true);
    });

    it('should fact-check a statement', async () => {
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue({
        statement: 'The economy grew by 3%',
        verdict: 'True',
        confidence: 0.85,
        supporting_claims: ['claim-123'],
        contradicting_claims: [],
      });

      const result = await bridgeClient.factCheck({
        source_happ: 'news-app',
        statement: 'The economy grew by 3%',
      });

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'knowledge_bridge',
        fn_name: 'fact_check',
        payload: expect.objectContaining({ statement: 'The economy grew by 3%' }),
      });
      expect(result.verdict).toBe('True');
    });

    it('should register an external claim', async () => {
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue({
        entry: { Present: { claim_id: 'ext:marketplace:abc:123' } },
      });

      const result = await bridgeClient.registerExternalClaim({
        source_happ: 'marketplace',
        subject: 'product-123',
        predicate: 'has_rating',
        object: '4.5',
        epistemic_e: 0.9,
      });

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'knowledge_bridge',
        fn_name: 'register_external_claim',
        payload: expect.objectContaining({ source_happ: 'marketplace' }),
      });
      expect(result.entry.Present).toBeDefined();
    });

    it('should broadcast knowledge event', async () => {
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue({
        entry: {
          Present: {
            id: 'event-1',
            event_type: 'ClaimCreated',
            source_happ: 'mycelix-knowledge',
          },
        },
      });

      const result = await bridgeClient.broadcastKnowledgeEvent({
        event_type: 'ClaimCreated',
        claim_id: 'claim-123',
        subject: 'New claim created',
        payload: '{}',
      });

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'knowledge_bridge',
        fn_name: 'broadcast_knowledge_event',
        payload: expect.objectContaining({ event_type: 'ClaimCreated' }),
      });
      expect(result.entry.Present.event_type).toBe('ClaimCreated');
    });

    it('should get recent knowledge events', async () => {
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue([]);

      const result = await bridgeClient.getRecentKnowledgeEvents();

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'knowledge_bridge',
        fn_name: 'get_recent_knowledge_events',
        payload: null,
      });
      expect(result).toEqual([]);
    });

    it('should get recent knowledge events since timestamp', async () => {
      const since = Date.now() * 1000 - 3600000000;
      (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValue([]);

      const result = await bridgeClient.getRecentKnowledgeEvents(since);

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'knowledge',
        zome_name: 'knowledge_bridge',
        fn_name: 'get_recent_knowledge_events',
        payload: since,
      });
      expect(result).toEqual([]);
    });
  });

  describe('createKnowledgeClients', () => {
    it('should create all client instances', () => {
      const mockClient = createMockClient();
      const clients = createKnowledgeClients(mockClient);

      expect(clients.claims).toBeInstanceOf(ClaimsClient);
      expect(clients.graph).toBeInstanceOf(GraphClient);
      expect(clients.query).toBeInstanceOf(QueryClient);
      expect(clients.inference).toBeInstanceOf(InferenceClient);
      expect(clients.bridge).toBeInstanceOf(KnowledgeBridgeClient);
    });

    it('should use the same underlying client for all instances', async () => {
      const mockClient = createMockClient();
      const clients = createKnowledgeClients(mockClient);

      // Call methods on different clients
      await clients.claims.getClaim('test');
      await clients.graph.getGraphStats();
      await clients.query.getPublicQueries();

      // All should use the same mockClient
      expect(mockClient.callZome).toHaveBeenCalledTimes(3);
    });
  });

  describe('Type Validation', () => {
    it('should enforce EpistemicPosition structure', () => {
      const position: EpistemicPosition = {
        empirical: 0.8,
        normative: 0.5,
        mythic: 0.3,
      };

      expect(position.empirical).toBeGreaterThanOrEqual(0);
      expect(position.empirical).toBeLessThanOrEqual(1);
      expect(position.normative).toBeGreaterThanOrEqual(0);
      expect(position.normative).toBeLessThanOrEqual(1);
      expect(position.mythic).toBeGreaterThanOrEqual(0);
      expect(position.mythic).toBeLessThanOrEqual(1);
    });

    it('should support all ClaimType values', () => {
      const types: string[] = [
        'Fact',
        'Opinion',
        'Prediction',
        'Hypothesis',
        'Definition',
        'Historical',
        'Normative',
        'Narrative',
      ];

      types.forEach((type) => {
        expect(typeof type).toBe('string');
      });
    });

    it('should support all RelationshipType values', () => {
      const relationship: Relationship = {
        id: 'rel-1',
        source: 'claim-1',
        target: 'claim-2',
        relationship_type: 'Supports',
        weight: 0.8,
        creator: 'did:mycelix:abc123',
        created: Date.now() * 1000,
      };

      expect(relationship.relationship_type).toBe('Supports');
    });

    it('should support custom relationship types', () => {
      const relationship: Relationship = {
        id: 'rel-2',
        source: 'claim-1',
        target: 'claim-2',
        relationship_type: { Custom: 'MyCustomRelation' },
        weight: 0.5,
        creator: 'did:mycelix:abc123',
        created: Date.now() * 1000,
      };

      expect(
        typeof relationship.relationship_type === 'object' &&
          'Custom' in relationship.relationship_type
      ).toBe(true);
    });
  });
});
