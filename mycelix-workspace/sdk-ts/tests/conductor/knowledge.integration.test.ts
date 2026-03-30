// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Knowledge hApp Conductor Integration Tests
 *
 * These tests verify the Knowledge clients work correctly with a real
 * Holochain conductor. They require the conductor harness to be available.
 *
 * Run with: npm run test:conductor
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import {
  ClaimsClient,
  GraphClient,
  QueryClient,
  InferenceClient,
  KnowledgeBridgeClient,
  createKnowledgeClients,
  type ZomeCallable,
  type Claim,
  type Relationship,
} from '../../src/knowledge/index.js';
import {
  createValidatedKnowledgeClients,
  ValidatedClaimsClient,
} from '../../src/knowledge/validated.js';
import { MycelixError, ErrorCode } from '../../src/errors.js';

// Conductor harness (dynamically imported if available)
let conductorHarness: {
  start: () => Promise<void>;
  stop: () => Promise<void>;
  createClient: () => ZomeCallable;
  getAgentPubKey: () => string;
} | null = null;

let client: ZomeCallable;
let agentPubKey: string;

// Skip all tests if conductor is not available
const describeConductor = process.env.HOLOCHAIN_CONDUCTOR_AVAILABLE ? describe : describe.skip;

describeConductor('Knowledge Conductor Integration Tests', () => {
  beforeAll(async () => {
    try {
      // Try to import the conductor harness
      const harness = await import('./conductor-harness.js');
      conductorHarness = harness.default || harness;
      await conductorHarness.start();
      client = conductorHarness.createClient();
      agentPubKey = conductorHarness.getAgentPubKey();
    } catch (error) {
      console.log('Conductor not available, skipping integration tests');
    }
  }, 60000);

  afterAll(async () => {
    if (conductorHarness) {
      await conductorHarness.stop();
    }
  }, 30000);

  describe('ClaimsClient', () => {
    let claimsClient: ClaimsClient;

    beforeAll(() => {
      claimsClient = new ClaimsClient(client);
    });

    it('should submit a claim', async () => {
      const claim: Claim = {
        id: `claim-${Date.now()}`,
        content: 'The Earth orbits the Sun',
        classification: { empirical: 0.99, normative: 0.1, mythic: 0.1 },
        author: `did:mycelix:${agentPubKey}`,
        sources: ['https://nasa.gov'],
        tags: ['astronomy', 'science'],
        claim_type: 'Fact',
        confidence: 0.99,
        created: Date.now() * 1000,
        updated: Date.now() * 1000,
        version: 1,
      };

      const result = await claimsClient.submitClaim(claim);

      expect(result).toBeDefined();
      expect(result.signed_action).toBeDefined();
      expect((result.entry.Present as Claim).id).toBe(claim.id);
    });

    it('should get a claim by ID', async () => {
      // First submit a claim
      const claimId = `claim-get-${Date.now()}`;
      await claimsClient.submitClaim({
        id: claimId,
        content: 'Test claim',
        classification: { empirical: 0.5, normative: 0.5, mythic: 0.5 },
        author: `did:mycelix:${agentPubKey}`,
        sources: [],
        tags: ['test'],
        claim_type: 'Hypothesis',
        confidence: 0.5,
        created: Date.now() * 1000,
        updated: Date.now() * 1000,
        version: 1,
      });

      const result = await claimsClient.getClaim(claimId);

      expect(result).toBeDefined();
      if (result) {
        expect((result.entry.Present as Claim).id).toBe(claimId);
      }
    });

    it('should get claims by author', async () => {
      const authorDid = `did:mycelix:${agentPubKey}`;

      const result = await claimsClient.getClaimsByAuthor(authorDid);

      expect(Array.isArray(result)).toBe(true);
    });

    it('should get claims by tag', async () => {
      // Submit a claim with a specific tag
      const tag = `test-tag-${Date.now()}`;
      await claimsClient.submitClaim({
        id: `claim-tag-${Date.now()}`,
        content: 'Tagged claim',
        classification: { empirical: 0.5, normative: 0.5, mythic: 0.5 },
        author: `did:mycelix:${agentPubKey}`,
        sources: [],
        tags: [tag],
        claim_type: 'Opinion',
        confidence: 0.7,
        created: Date.now() * 1000,
        updated: Date.now() * 1000,
        version: 1,
      });

      const result = await claimsClient.getClaimsByTag(tag);

      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBeGreaterThan(0);
    });

    it('should add evidence to a claim', async () => {
      // First submit a claim
      const claimId = `claim-evidence-${Date.now()}`;
      await claimsClient.submitClaim({
        id: claimId,
        content: 'Claim needing evidence',
        classification: { empirical: 0.8, normative: 0.1, mythic: 0.1 },
        author: `did:mycelix:${agentPubKey}`,
        sources: [],
        tags: ['evidence-test'],
        claim_type: 'Hypothesis',
        confidence: 0.6,
        created: Date.now() * 1000,
        updated: Date.now() * 1000,
        version: 1,
      });

      const evidence = {
        id: `evidence-${Date.now()}`,
        claim_id: claimId,
        evidence_type: 'Document' as const,
        source_uri: 'https://example.com/paper.pdf',
        content: 'Supporting research paper',
        strength: 0.8,
        submitted_by: `did:mycelix:${agentPubKey}`,
        submitted_at: Date.now() * 1000,
      };

      const result = await claimsClient.addEvidence(evidence);

      expect(result).toBeDefined();
    });

    it('should challenge a claim', async () => {
      // First submit a claim
      const claimId = `claim-challenge-${Date.now()}`;
      await claimsClient.submitClaim({
        id: claimId,
        content: 'Challengeable claim',
        classification: { empirical: 0.3, normative: 0.5, mythic: 0.2 },
        author: `did:mycelix:${agentPubKey}`,
        sources: [],
        tags: ['challenge-test'],
        claim_type: 'Opinion',
        confidence: 0.4,
        created: Date.now() * 1000,
        updated: Date.now() * 1000,
        version: 1,
      });

      const challenge = {
        claim_id: claimId,
        challenger_did: `did:mycelix:${agentPubKey}`,
        reason: 'This claim lacks supporting evidence',
        counter_evidence: ['https://counter.example.com'],
      };

      const result = await claimsClient.challengeClaim(challenge);

      expect(result).toBeDefined();
    });

    it('should search by epistemic range', async () => {
      const result = await claimsClient.searchByEpistemicRange({
        e_min: 0.7,
        e_max: 1.0,
        n_min: 0.0,
        n_max: 0.3,
        m_min: 0.0,
        m_max: 0.3,
      });

      expect(Array.isArray(result)).toBe(true);
    });
  });

  describe('GraphClient', () => {
    let graphClient: GraphClient;
    let claimsClient: ClaimsClient;

    beforeAll(() => {
      graphClient = new GraphClient(client);
      claimsClient = new ClaimsClient(client);
    });

    it('should create a relationship between claims', async () => {
      // Create two claims first
      const claim1Id = `claim-rel1-${Date.now()}`;
      const claim2Id = `claim-rel2-${Date.now()}`;

      await claimsClient.submitClaim({
        id: claim1Id,
        content: 'First claim',
        classification: { empirical: 0.8, normative: 0.1, mythic: 0.1 },
        author: `did:mycelix:${agentPubKey}`,
        sources: [],
        tags: ['graph-test'],
        claim_type: 'Fact',
        confidence: 0.9,
        created: Date.now() * 1000,
        updated: Date.now() * 1000,
        version: 1,
      });

      await claimsClient.submitClaim({
        id: claim2Id,
        content: 'Second claim that supports the first',
        classification: { empirical: 0.8, normative: 0.1, mythic: 0.1 },
        author: `did:mycelix:${agentPubKey}`,
        sources: [],
        tags: ['graph-test'],
        claim_type: 'Fact',
        confidence: 0.85,
        created: Date.now() * 1000,
        updated: Date.now() * 1000,
        version: 1,
      });

      const relationship: Relationship = {
        id: `rel-${Date.now()}`,
        source: claim1Id,
        target: claim2Id,
        relationship_type: 'Supports',
        weight: 0.9,
        creator: `did:mycelix:${agentPubKey}`,
        created: Date.now() * 1000,
      };

      const result = await graphClient.createRelationship(relationship);

      expect(result).toBeDefined();
    });

    it('should get outgoing relationships', async () => {
      const claimId = `claim-outgoing-${Date.now()}`;

      const result = await graphClient.getOutgoingRelationships(claimId);

      expect(Array.isArray(result)).toBe(true);
    });

    it('should get incoming relationships', async () => {
      const claimId = `claim-incoming-${Date.now()}`;

      const result = await graphClient.getIncomingRelationships(claimId);

      expect(Array.isArray(result)).toBe(true);
    });

    it('should find path between claims', async () => {
      const result = await graphClient.findPath({
        source: 'claim-1',
        target: 'claim-5',
        max_depth: 5,
      });

      expect(Array.isArray(result)).toBe(true);
    });

    it('should get graph statistics', async () => {
      const result = await graphClient.getGraphStats();

      expect(result).toBeDefined();
      expect(typeof result.relationship_count).toBe('number');
    });

    it('should create an ontology', async () => {
      const ontology = {
        id: `ontology-${Date.now()}`,
        name: 'Climate Science Ontology',
        description: 'Ontology for climate-related claims',
        namespace: 'https://mycelix.net/ontologies/climate',
        schema: JSON.stringify({
          '@context': 'https://schema.org',
          '@type': 'Ontology',
        }),
        version: '1.0.0',
        creator: `did:mycelix:${agentPubKey}`,
        created: Date.now() * 1000,
        updated: Date.now() * 1000,
      };

      const result = await graphClient.createOntology(ontology);

      expect(result).toBeDefined();
    });

    it('should create a concept in an ontology', async () => {
      const ontologyId = `ontology-concept-${Date.now()}`;

      // First create ontology
      await graphClient.createOntology({
        id: ontologyId,
        name: 'Test Ontology',
        description: 'Test',
        namespace: 'https://test.example.com',
        schema: '{}',
        version: '1.0.0',
        creator: `did:mycelix:${agentPubKey}`,
        created: Date.now() * 1000,
        updated: Date.now() * 1000,
      });

      const concept = {
        id: `concept-${Date.now()}`,
        ontology_id: ontologyId,
        name: 'Temperature',
        definition: 'A measure of thermal energy',
        synonyms: ['heat', 'warmth'],
        created: Date.now() * 1000,
      };

      const result = await graphClient.createConcept(concept);

      expect(result).toBeDefined();
    });
  });

  describe('QueryClient', () => {
    let queryClient: QueryClient;

    beforeAll(() => {
      queryClient = new QueryClient(client);
    });

    it('should execute a query', async () => {
      const result = await queryClient.executeQuery({
        query: "SELECT * FROM claims WHERE tag = 'test'",
        use_cache: true,
        limit: 100,
      });

      expect(result).toBeDefined();
      expect(Array.isArray(result.results)).toBe(true);
      expect(typeof result.count).toBe('number');
      expect(typeof result.execution_time_ms).toBe('number');
    });

    it('should save a query', async () => {
      const savedQuery = {
        id: `query-${Date.now()}`,
        name: 'High Empirical Claims',
        description: 'Find claims with high empirical validity',
        query: 'SELECT * FROM claims WHERE e > 0.8',
        creator: `did:mycelix:${agentPubKey}`,
        public: true,
        created: Date.now() * 1000,
        updated: Date.now() * 1000,
      };

      const result = await queryClient.saveQuery(savedQuery);

      expect(result).toBeDefined();
    });

    it('should get my queries', async () => {
      const result = await queryClient.getMyQueries(`did:mycelix:${agentPubKey}`);

      expect(Array.isArray(result)).toBe(true);
    });

    it('should get public queries', async () => {
      const result = await queryClient.getPublicQueries();

      expect(Array.isArray(result)).toBe(true);
    });
  });

  describe('InferenceClient', () => {
    let inferenceClient: InferenceClient;

    beforeAll(() => {
      inferenceClient = new InferenceClient(client);
    });

    it('should assess credibility', async () => {
      const result = await inferenceClient.assessCredibility({
        subject: `did:mycelix:${agentPubKey}`,
        subject_type: 'Author',
      });

      expect(result).toBeDefined();
    });

    it('should get credibility score', async () => {
      const result = await inferenceClient.getCredibility(`did:mycelix:${agentPubKey}`);

      // May or may not exist
      if (result) {
        expect(result.entry.Present).toHaveProperty('score');
      }
    });

    it('should detect patterns', async () => {
      const result = await inferenceClient.detectPatterns({
        claims: ['claim-1', 'claim-2', 'claim-3'],
        pattern_types: ['Cluster', 'Trend'],
      });

      expect(Array.isArray(result)).toBe(true);
    });

    it('should find contradictions', async () => {
      const result = await inferenceClient.findContradictions(['claim-a', 'claim-b', 'claim-c']);

      expect(Array.isArray(result)).toBe(true);
    });
  });

  describe('KnowledgeBridgeClient', () => {
    let bridgeClient: KnowledgeBridgeClient;

    beforeAll(() => {
      bridgeClient = new KnowledgeBridgeClient(client);
    });

    it('should query knowledge', async () => {
      const result = await bridgeClient.queryKnowledge({
        source_happ: 'test-app',
        query_type: 'VerifyClaim',
        parameters: { claim_id: 'test-claim' },
      });

      expect(result).toBeDefined();
      expect(typeof result.success).toBe('boolean');
    });

    it('should fact-check a statement', async () => {
      const result = await bridgeClient.factCheck({
        source_happ: 'news-app',
        statement: 'Water boils at 100 degrees Celsius at sea level',
      });

      expect(result).toBeDefined();
      expect(result.statement).toBe('Water boils at 100 degrees Celsius at sea level');
      expect(['True', 'False', 'PartiallyTrue', 'Misleading', 'Unverified']).toContain(
        result.verdict
      );
    });

    it('should register an external claim', async () => {
      const result = await bridgeClient.registerExternalClaim({
        source_happ: 'marketplace',
        subject: 'did:mycelix:seller123',
        predicate: 'hasReputation',
        object: 'excellent',
        epistemic_e: 0.7,
        epistemic_n: 0.2,
        epistemic_m: 0.1,
      });

      expect(result).toBeDefined();
    });

    it('should broadcast a knowledge event', async () => {
      const result = await bridgeClient.broadcastKnowledgeEvent({
        event_type: 'ClaimCreated',
        claim_id: 'new-claim-123',
        subject: 'climate-science',
        payload: JSON.stringify({ importance: 'high' }),
      });

      expect(result).toBeDefined();
    });

    it('should get recent knowledge events', async () => {
      const result = await bridgeClient.getRecentKnowledgeEvents();

      expect(Array.isArray(result)).toBe(true);
    });
  });

  describe('Validated Clients', () => {
    it('should reject invalid DID format', async () => {
      const clients = createValidatedKnowledgeClients(client);

      await expect(clients.claims.getClaimsByAuthor('invalid-did')).rejects.toThrow(MycelixError);
    });

    it('should reject invalid epistemic values', async () => {
      const clients = createValidatedKnowledgeClients(client);

      await expect(
        clients.claims.searchByEpistemicRange({
          e_min: -0.1, // Invalid: must be >= 0
          e_max: 1.0,
          n_min: 0.0,
          n_max: 1.0,
          m_min: 0.0,
          m_max: 1.0,
        })
      ).rejects.toThrow(MycelixError);
    });

    it('should reject invalid claim content', async () => {
      const clients = createValidatedKnowledgeClients(client);

      await expect(
        clients.claims.submitClaim({
          id: 'test',
          content: '', // Invalid: empty content
          classification: { empirical: 0.5, normative: 0.5, mythic: 0.5 },
          author: 'did:mycelix:test',
          sources: [],
          tags: [],
          claim_type: 'Fact',
          confidence: 0.5,
          created: Date.now() * 1000,
          updated: Date.now() * 1000,
          version: 1,
        })
      ).rejects.toThrow(MycelixError);
    });

    it('should accept valid inputs', async () => {
      const clients = createValidatedKnowledgeClients(client);

      // This should not throw validation errors
      const result = await clients.claims.getClaimsByTag('valid-tag');

      // Result may be empty array, but should not throw
      expect(Array.isArray(result)).toBe(true);
    });
  });

  describe('createKnowledgeClients factory', () => {
    it('should create all clients', () => {
      const clients = createKnowledgeClients(client);

      expect(clients.claims).toBeInstanceOf(ClaimsClient);
      expect(clients.graph).toBeInstanceOf(GraphClient);
      expect(clients.query).toBeInstanceOf(QueryClient);
      expect(clients.inference).toBeInstanceOf(InferenceClient);
      expect(clients.bridge).toBeInstanceOf(KnowledgeBridgeClient);
    });
  });
});
