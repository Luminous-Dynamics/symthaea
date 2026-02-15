/**
 * Tests for knowledge/validated.ts
 *
 * Tests validation schemas and validated client wrappers for Knowledge hApp.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  ValidatedClaimsClient,
  ValidatedGraphClient,
  ValidatedQueryClient,
  ValidatedInferenceClient,
  ValidatedKnowledgeBridgeClient,
  createValidatedKnowledgeClients,
} from '../../src/knowledge/validated.js';
import { MycelixError, ErrorCode } from '../../src/errors.js';

// Helper to create Holochain record structure
const createHolochainRecord = <T>(entry: T) => ({
  entry: { Present: { entry } },
  signed_action: {
    hashed: {
      content: {
        entry_hash: 'test-entry-hash',
        entry_type: 'test-type',
      },
      hash: 'test-action-hash',
    },
  },
});

// Mock zome callable that returns proper Holochain record format
const createMockZomeCallable = () => ({
  callZome: vi.fn().mockImplementation(async ({ fn_name }) => {
    // Return array for list-type functions
    const isArrayReturn =
      fn_name.startsWith('get_') &&
      (fn_name.includes('_by_') ||
        fn_name.endsWith('relationships') ||
        fn_name.endsWith('evidence') ||
        fn_name.endsWith('challenges') ||
        fn_name.endsWith('inferences') ||
        fn_name.endsWith('concepts') ||
        fn_name.endsWith('queries') ||
        fn_name.endsWith('events') ||
        fn_name.endsWith('patterns'));

    if (isArrayReturn || fn_name.startsWith('search_') || fn_name.startsWith('find_') || fn_name.startsWith('detect_')) {
      return [createHolochainRecord({})];
    }

    // Return stats object for get_graph_stats
    if (fn_name === 'get_graph_stats') {
      return { total_claims: 100, total_relationships: 50 };
    }

    // Return result objects for query/fact-check
    if (fn_name === 'execute_query' || fn_name === 'query_knowledge' || fn_name === 'fact_check') {
      return { results: [], metadata: {} };
    }

    // Return single record for others
    return createHolochainRecord({});
  }),
});

describe('Knowledge Validated Clients', () => {
  let mockZomeClient: ReturnType<typeof createMockZomeCallable>;

  beforeEach(() => {
    mockZomeClient = createMockZomeCallable();
    vi.clearAllMocks();
  });

  describe('ValidatedClaimsClient', () => {
    let client: ValidatedClaimsClient;

    beforeEach(() => {
      client = new ValidatedClaimsClient(mockZomeClient);
    });

    describe('submitClaim', () => {
      const validClaim = {
        id: 'claim-1',
        content: 'Test claim content',
        classification: { empirical: 0.8, normative: 0.5, mythic: 0.2 },
        author: 'did:example:author',
        sources: ['source1'],
        tags: ['test'],
        claim_type: 'Fact' as const,
        confidence: 0.9,
        created: Date.now(),
        updated: Date.now(),
        version: 1,
      };

      it('should accept valid claim', async () => {
        await expect(client.submitClaim(validClaim)).resolves.toBeDefined();
      });

      it('should reject empty content', async () => {
        await expect(client.submitClaim({ ...validClaim, content: '' })).rejects.toThrow(
          MycelixError
        );
      });

      it('should reject invalid author DID', async () => {
        await expect(
          client.submitClaim({ ...validClaim, author: 'not-a-did' })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject confidence above 1', async () => {
        await expect(
          client.submitClaim({ ...validClaim, confidence: 1.5 })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject confidence below 0', async () => {
        await expect(
          client.submitClaim({ ...validClaim, confidence: -0.1 })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject invalid claim type', async () => {
        await expect(
          client.submitClaim({ ...validClaim, claim_type: 'Invalid' as any })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject epistemic values above 1', async () => {
        await expect(
          client.submitClaim({
            ...validClaim,
            classification: { empirical: 1.5, normative: 0.5, mythic: 0.2 },
          })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject epistemic values below 0', async () => {
        await expect(
          client.submitClaim({
            ...validClaim,
            classification: { empirical: -0.1, normative: 0.5, mythic: 0.2 },
          })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject negative version', async () => {
        await expect(
          client.submitClaim({ ...validClaim, version: 0 })
        ).rejects.toThrow(MycelixError);
      });
    });

    describe('getClaim', () => {
      it('should accept valid claim ID', async () => {
        await expect(client.getClaim('claim-1')).resolves.toBeDefined();
      });

      it('should reject empty claim ID', async () => {
        await expect(client.getClaim('')).rejects.toThrow(MycelixError);
      });
    });

    describe('getClaimsByAuthor', () => {
      it('should accept valid DID', async () => {
        await expect(client.getClaimsByAuthor('did:example:author')).resolves.toBeDefined();
      });

      it('should reject non-DID', async () => {
        await expect(client.getClaimsByAuthor('not-a-did')).rejects.toThrow(MycelixError);
      });
    });

    describe('getClaimsByTag', () => {
      it('should accept valid tag', async () => {
        await expect(client.getClaimsByTag('test')).resolves.toBeDefined();
      });

      it('should reject empty tag', async () => {
        await expect(client.getClaimsByTag('')).rejects.toThrow(MycelixError);
      });
    });

    describe('updateClaim', () => {
      it('should accept valid update input', async () => {
        await expect(client.updateClaim({ claim_id: 'claim-1', content: 'updated' })).resolves.toBeDefined();
      });

      it('should reject empty claim_id', async () => {
        await expect(client.updateClaim({ claim_id: '' })).rejects.toThrow(MycelixError);
      });

      it('should accept partial update', async () => {
        await expect(client.updateClaim({ claim_id: 'claim-1', confidence: 0.8 })).resolves.toBeDefined();
      });

      it('should reject invalid confidence', async () => {
        await expect(client.updateClaim({ claim_id: 'claim-1', confidence: 2.0 })).rejects.toThrow(
          MycelixError
        );
      });
    });

    describe('addEvidence', () => {
      const validEvidence = {
        id: 'evidence-1',
        claim_id: 'claim-1',
        evidence_type: 'Observation' as const,
        source_uri: 'https://example.com',
        content: 'Evidence content',
        strength: 0.8,
        submitted_by: 'did:example:submitter',
        submitted_at: Date.now(),
      };

      it('should accept valid evidence', async () => {
        await expect(client.addEvidence(validEvidence)).resolves.toBeDefined();
      });

      it('should reject empty content', async () => {
        await expect(client.addEvidence({ ...validEvidence, content: '' })).rejects.toThrow(
          MycelixError
        );
      });

      it('should reject invalid evidence type', async () => {
        await expect(
          client.addEvidence({ ...validEvidence, evidence_type: 'Invalid' as any })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject strength above 1', async () => {
        await expect(client.addEvidence({ ...validEvidence, strength: 1.5 })).rejects.toThrow(
          MycelixError
        );
      });

      it('should reject invalid submitter DID', async () => {
        await expect(
          client.addEvidence({ ...validEvidence, submitted_by: 'not-a-did' })
        ).rejects.toThrow(MycelixError);
      });
    });

    describe('getClaimEvidence', () => {
      it('should accept valid claim ID', async () => {
        await expect(client.getClaimEvidence('claim-1')).resolves.toBeDefined();
      });

      it('should reject empty claim ID', async () => {
        await expect(client.getClaimEvidence('')).rejects.toThrow(MycelixError);
      });
    });

    describe('challengeClaim', () => {
      const validChallenge = {
        claim_id: 'claim-1',
        challenger_did: 'did:example:challenger',
        reason: 'This claim is incorrect',
        counter_evidence: ['evidence-1'],
      };

      it('should accept valid challenge', async () => {
        await expect(client.challengeClaim(validChallenge)).resolves.toBeDefined();
      });

      it('should reject empty reason', async () => {
        await expect(client.challengeClaim({ ...validChallenge, reason: '' })).rejects.toThrow(
          MycelixError
        );
      });

      it('should reject invalid challenger DID', async () => {
        await expect(
          client.challengeClaim({ ...validChallenge, challenger_did: 'not-a-did' })
        ).rejects.toThrow(MycelixError);
      });
    });

    describe('getClaimChallenges', () => {
      it('should accept valid claim ID', async () => {
        await expect(client.getClaimChallenges('claim-1')).resolves.toBeDefined();
      });

      it('should reject empty claim ID', async () => {
        await expect(client.getClaimChallenges('')).rejects.toThrow(MycelixError);
      });
    });

    describe('searchByEpistemicRange', () => {
      const validRange = {
        e_min: 0.2,
        e_max: 0.8,
        n_min: 0.1,
        n_max: 0.9,
        m_min: 0.0,
        m_max: 1.0,
      };

      it('should accept valid range', async () => {
        await expect(client.searchByEpistemicRange(validRange)).resolves.toBeDefined();
      });

      it('should reject min > max', async () => {
        await expect(
          client.searchByEpistemicRange({ ...validRange, e_min: 0.9, e_max: 0.1 })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject values above 1', async () => {
        await expect(
          client.searchByEpistemicRange({ ...validRange, e_max: 1.5 })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject values below 0', async () => {
        await expect(
          client.searchByEpistemicRange({ ...validRange, e_min: -0.1 })
        ).rejects.toThrow(MycelixError);
      });
    });
  });

  describe('ValidatedGraphClient', () => {
    let client: ValidatedGraphClient;

    beforeEach(() => {
      client = new ValidatedGraphClient(mockZomeClient);
    });

    describe('createRelationship', () => {
      const validRelationship = {
        id: 'rel-1',
        source: 'claim-1',
        target: 'claim-2',
        relationship_type: 'Supports' as const,
        weight: 0.8,
        creator: 'did:example:creator',
        created: Date.now(),
      };

      it('should accept valid relationship', async () => {
        await expect(client.createRelationship(validRelationship)).resolves.toBeDefined();
      });

      it('should reject empty source', async () => {
        await expect(
          client.createRelationship({ ...validRelationship, source: '' })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject empty target', async () => {
        await expect(
          client.createRelationship({ ...validRelationship, target: '' })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject weight above 1', async () => {
        await expect(
          client.createRelationship({ ...validRelationship, weight: 1.5 })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject invalid creator DID', async () => {
        await expect(
          client.createRelationship({ ...validRelationship, creator: 'not-a-did' })
        ).rejects.toThrow(MycelixError);
      });

      it('should accept custom relationship type', async () => {
        await expect(
          client.createRelationship({
            ...validRelationship,
            relationship_type: { Custom: 'MyCustomType' },
          })
        ).resolves.toBeDefined();
      });
    });

    describe('getOutgoingRelationships', () => {
      it('should accept valid claim ID', async () => {
        await expect(client.getOutgoingRelationships('claim-1')).resolves.toBeDefined();
      });

      it('should reject empty claim ID', async () => {
        await expect(client.getOutgoingRelationships('')).rejects.toThrow(MycelixError);
      });
    });

    describe('getIncomingRelationships', () => {
      it('should accept valid claim ID', async () => {
        await expect(client.getIncomingRelationships('claim-1')).resolves.toBeDefined();
      });

      it('should reject empty claim ID', async () => {
        await expect(client.getIncomingRelationships('')).rejects.toThrow(MycelixError);
      });
    });

    describe('findPath', () => {
      it('should accept valid input', async () => {
        await expect(
          client.findPath({ source: 'claim-1', target: 'claim-2', max_depth: 5 })
        ).resolves.toBeDefined();
      });

      it('should reject empty source', async () => {
        await expect(
          client.findPath({ source: '', target: 'claim-2', max_depth: 5 })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject max_depth below 1', async () => {
        await expect(
          client.findPath({ source: 'claim-1', target: 'claim-2', max_depth: 0 })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject max_depth above 20', async () => {
        await expect(
          client.findPath({ source: 'claim-1', target: 'claim-2', max_depth: 21 })
        ).rejects.toThrow(MycelixError);
      });
    });

    describe('createOntology', () => {
      const validOntology = {
        id: 'ont-1',
        name: 'Test Ontology',
        description: 'Test description',
        namespace: 'test.ontology',
        schema: 'schema-content',
        version: '1.0.0',
        creator: 'did:example:creator',
        created: Date.now(),
        updated: Date.now(),
      };

      it('should accept valid ontology', async () => {
        await expect(client.createOntology(validOntology)).resolves.toBeDefined();
      });

      it('should reject empty name', async () => {
        await expect(
          client.createOntology({ ...validOntology, name: '' })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject invalid creator DID', async () => {
        await expect(
          client.createOntology({ ...validOntology, creator: 'not-a-did' })
        ).rejects.toThrow(MycelixError);
      });
    });

    describe('createConcept', () => {
      const validConcept = {
        id: 'concept-1',
        ontology_id: 'ont-1',
        name: 'Test Concept',
        definition: 'Test definition',
        synonyms: ['alt-name'],
        created: Date.now(),
      };

      it('should accept valid concept', async () => {
        await expect(client.createConcept(validConcept)).resolves.toBeDefined();
      });

      it('should reject empty name', async () => {
        await expect(
          client.createConcept({ ...validConcept, name: '' })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject empty definition', async () => {
        await expect(
          client.createConcept({ ...validConcept, definition: '' })
        ).rejects.toThrow(MycelixError);
      });

      it('should accept optional parent', async () => {
        await expect(
          client.createConcept({ ...validConcept, parent: 'parent-concept' })
        ).resolves.toBeDefined();
      });
    });

    describe('getOntologyConcepts', () => {
      it('should accept valid ontology ID', async () => {
        await expect(client.getOntologyConcepts('ont-1')).resolves.toBeDefined();
      });

      it('should reject empty ontology ID', async () => {
        await expect(client.getOntologyConcepts('')).rejects.toThrow(MycelixError);
      });
    });

    describe('getChildConcepts', () => {
      it('should accept valid concept ID', async () => {
        await expect(client.getChildConcepts('concept-1')).resolves.toBeDefined();
      });

      it('should reject empty concept ID', async () => {
        await expect(client.getChildConcepts('')).rejects.toThrow(MycelixError);
      });
    });

    describe('getGraphStats', () => {
      it('should not require validation', async () => {
        await expect(client.getGraphStats()).resolves.toBeDefined();
      });
    });
  });

  describe('ValidatedQueryClient', () => {
    let client: ValidatedQueryClient;

    beforeEach(() => {
      client = new ValidatedQueryClient(mockZomeClient);
    });

    describe('executeQuery', () => {
      const validInput = {
        query: 'SELECT * FROM claims',
        use_cache: true,
      };

      it('should accept valid input', async () => {
        await expect(client.executeQuery(validInput)).resolves.toBeDefined();
      });

      it('should reject empty query', async () => {
        await expect(client.executeQuery({ ...validInput, query: '' })).rejects.toThrow(
          MycelixError
        );
      });

      it('should accept optional limit', async () => {
        await expect(
          client.executeQuery({ ...validInput, limit: 100 })
        ).resolves.toBeDefined();
      });

      it('should reject limit above 10000', async () => {
        await expect(
          client.executeQuery({ ...validInput, limit: 10001 })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject limit below 1', async () => {
        await expect(
          client.executeQuery({ ...validInput, limit: 0 })
        ).rejects.toThrow(MycelixError);
      });

      it('should accept optional offset', async () => {
        await expect(
          client.executeQuery({ ...validInput, offset: 50 })
        ).resolves.toBeDefined();
      });

      it('should reject negative offset', async () => {
        await expect(
          client.executeQuery({ ...validInput, offset: -1 })
        ).rejects.toThrow(MycelixError);
      });
    });

    describe('saveQuery', () => {
      const validQuery = {
        id: 'query-1',
        name: 'Test Query',
        description: 'Test description',
        query: 'SELECT * FROM claims',
        creator: 'did:example:creator',
        public: true,
        created: Date.now(),
        updated: Date.now(),
      };

      it('should accept valid query', async () => {
        await expect(client.saveQuery(validQuery)).resolves.toBeDefined();
      });

      it('should reject empty name', async () => {
        await expect(client.saveQuery({ ...validQuery, name: '' })).rejects.toThrow(
          MycelixError
        );
      });

      it('should reject empty query string', async () => {
        await expect(client.saveQuery({ ...validQuery, query: '' })).rejects.toThrow(
          MycelixError
        );
      });

      it('should reject invalid creator DID', async () => {
        await expect(
          client.saveQuery({ ...validQuery, creator: 'not-a-did' })
        ).rejects.toThrow(MycelixError);
      });
    });

    describe('getMyQueries', () => {
      it('should accept valid DID', async () => {
        await expect(client.getMyQueries('did:example:user')).resolves.toBeDefined();
      });

      it('should reject non-DID', async () => {
        await expect(client.getMyQueries('not-a-did')).rejects.toThrow(MycelixError);
      });
    });

    describe('getPublicQueries', () => {
      it('should not require validation', async () => {
        await expect(client.getPublicQueries()).resolves.toBeDefined();
      });
    });
  });

  describe('ValidatedInferenceClient', () => {
    let client: ValidatedInferenceClient;

    beforeEach(() => {
      client = new ValidatedInferenceClient(mockZomeClient);
    });

    describe('createInference', () => {
      const validInference = {
        id: 'inf-1',
        inference_type: 'Implication' as const,
        source_claims: ['claim-1', 'claim-2'],
        conclusion: 'Test conclusion',
        confidence: 0.8,
        reasoning: 'Test reasoning',
        model: 'test-model',
        created: Date.now(),
        verified: false,
      };

      it('should accept valid inference', async () => {
        await expect(client.createInference(validInference)).resolves.toBeDefined();
      });

      it('should reject empty source_claims', async () => {
        await expect(
          client.createInference({ ...validInference, source_claims: [] })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject empty conclusion', async () => {
        await expect(
          client.createInference({ ...validInference, conclusion: '' })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject invalid inference type', async () => {
        await expect(
          client.createInference({ ...validInference, inference_type: 'Invalid' as any })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject confidence above 1', async () => {
        await expect(
          client.createInference({ ...validInference, confidence: 1.5 })
        ).rejects.toThrow(MycelixError);
      });
    });

    describe('getClaimInferences', () => {
      it('should accept valid claim ID', async () => {
        await expect(client.getClaimInferences('claim-1')).resolves.toBeDefined();
      });

      it('should reject empty claim ID', async () => {
        await expect(client.getClaimInferences('')).rejects.toThrow(MycelixError);
      });
    });

    describe('assessCredibility', () => {
      it('should accept valid input', async () => {
        await expect(
          client.assessCredibility({ subject: 'claim-1', subject_type: 'Claim' })
        ).resolves.toBeDefined();
      });

      it('should reject empty subject', async () => {
        await expect(
          client.assessCredibility({ subject: '', subject_type: 'Claim' })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject invalid subject_type', async () => {
        await expect(
          client.assessCredibility({ subject: 'claim-1', subject_type: 'Invalid' as any })
        ).rejects.toThrow(MycelixError);
      });

      it('should accept optional expires_at', async () => {
        await expect(
          client.assessCredibility({ subject: 'claim-1', subject_type: 'Author', expires_at: Date.now() + 86400000 })
        ).resolves.toBeDefined();
      });
    });

    describe('getCredibility', () => {
      it('should accept valid subject', async () => {
        await expect(client.getCredibility('claim-1')).resolves.toBeDefined();
      });

      it('should reject empty subject', async () => {
        await expect(client.getCredibility('')).rejects.toThrow(MycelixError);
      });
    });

    describe('detectPatterns', () => {
      it('should accept valid input', async () => {
        await expect(
          client.detectPatterns({ claims: ['claim-1', 'claim-2'] })
        ).resolves.toBeDefined();
      });

      it('should reject fewer than 2 claims', async () => {
        await expect(
          client.detectPatterns({ claims: ['claim-1'] })
        ).rejects.toThrow(MycelixError);
      });

      it('should accept optional pattern_types', async () => {
        await expect(
          client.detectPatterns({ claims: ['claim-1', 'claim-2'], pattern_types: ['Cluster', 'Trend'] })
        ).resolves.toBeDefined();
      });
    });

    describe('findContradictions', () => {
      it('should accept valid claim IDs', async () => {
        await expect(client.findContradictions(['claim-1', 'claim-2'])).resolves.toBeDefined();
      });

      it('should reject fewer than 2 claims', async () => {
        await expect(client.findContradictions(['claim-1'])).rejects.toThrow(MycelixError);
      });

      it('should reject empty claim IDs', async () => {
        await expect(client.findContradictions(['claim-1', ''])).rejects.toThrow(MycelixError);
      });
    });

    describe('verifyInference', () => {
      it('should accept valid input', async () => {
        await expect(
          client.verifyInference({
            inference_id: 'inf-1',
            is_correct: true,
            verifier_did: 'did:example:verifier',
          })
        ).resolves.toBeDefined();
      });

      it('should reject empty inference_id', async () => {
        await expect(
          client.verifyInference({
            inference_id: '',
            is_correct: true,
            verifier_did: 'did:example:verifier',
          })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject invalid verifier DID', async () => {
        await expect(
          client.verifyInference({
            inference_id: 'inf-1',
            is_correct: false,
            verifier_did: 'not-a-did',
          })
        ).rejects.toThrow(MycelixError);
      });

      it('should accept optional comment', async () => {
        await expect(
          client.verifyInference({
            inference_id: 'inf-1',
            is_correct: true,
            verifier_did: 'did:example:verifier',
            comment: 'Looks correct',
          })
        ).resolves.toBeDefined();
      });
    });
  });

  describe('ValidatedKnowledgeBridgeClient', () => {
    let client: ValidatedKnowledgeBridgeClient;

    beforeEach(() => {
      client = new ValidatedKnowledgeBridgeClient(mockZomeClient);
    });

    describe('queryKnowledge', () => {
      it('should accept valid input', async () => {
        await expect(
          client.queryKnowledge({
            source_happ: 'marketplace',
            query_type: 'VerifyClaim',
            parameters: { claim_id: 'claim-1' },
          })
        ).resolves.toBeDefined();
      });

      it('should reject empty source_happ', async () => {
        await expect(
          client.queryKnowledge({
            source_happ: '',
            query_type: 'VerifyClaim',
            parameters: {},
          })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject invalid query_type', async () => {
        await expect(
          client.queryKnowledge({
            source_happ: 'marketplace',
            query_type: 'Invalid' as any,
            parameters: {},
          })
        ).rejects.toThrow(MycelixError);
      });
    });

    describe('factCheck', () => {
      it('should accept valid input', async () => {
        await expect(
          client.factCheck({
            source_happ: 'marketplace',
            statement: 'Test statement to fact check',
          })
        ).resolves.toBeDefined();
      });

      it('should reject empty statement', async () => {
        await expect(
          client.factCheck({
            source_happ: 'marketplace',
            statement: '',
          })
        ).rejects.toThrow(MycelixError);
      });

      it('should accept optional context', async () => {
        await expect(
          client.factCheck({
            source_happ: 'marketplace',
            statement: 'Test statement',
            context: 'Additional context',
          })
        ).resolves.toBeDefined();
      });
    });

    describe('registerExternalClaim', () => {
      it('should accept valid input', async () => {
        await expect(
          client.registerExternalClaim({
            source_happ: 'marketplace',
            subject: 'Product',
            predicate: 'hasQuality',
            object: 'high',
          })
        ).resolves.toBeDefined();
      });

      it('should reject empty subject', async () => {
        await expect(
          client.registerExternalClaim({
            source_happ: 'marketplace',
            subject: '',
            predicate: 'hasQuality',
            object: 'high',
          })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject empty predicate', async () => {
        await expect(
          client.registerExternalClaim({
            source_happ: 'marketplace',
            subject: 'Product',
            predicate: '',
            object: 'high',
          })
        ).rejects.toThrow(MycelixError);
      });

      it('should accept optional epistemic values', async () => {
        await expect(
          client.registerExternalClaim({
            source_happ: 'marketplace',
            subject: 'Product',
            predicate: 'hasQuality',
            object: 'high',
            epistemic_e: 0.8,
            epistemic_n: 0.5,
            epistemic_m: 0.3,
          })
        ).resolves.toBeDefined();
      });

      it('should reject epistemic values above 1', async () => {
        await expect(
          client.registerExternalClaim({
            source_happ: 'marketplace',
            subject: 'Product',
            predicate: 'hasQuality',
            object: 'high',
            epistemic_e: 1.5,
          })
        ).rejects.toThrow(MycelixError);
      });
    });

    describe('broadcastKnowledgeEvent', () => {
      it('should accept valid input', async () => {
        await expect(
          client.broadcastKnowledgeEvent({
            event_type: 'ClaimCreated',
            subject: 'Test subject',
            payload: '{}',
          })
        ).resolves.toBeDefined();
      });

      it('should reject invalid event_type', async () => {
        await expect(
          client.broadcastKnowledgeEvent({
            event_type: 'Invalid' as any,
            subject: 'Test subject',
            payload: '{}',
          })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject empty subject', async () => {
        await expect(
          client.broadcastKnowledgeEvent({
            event_type: 'ClaimUpdated',
            subject: '',
            payload: '{}',
          })
        ).rejects.toThrow(MycelixError);
      });

      it('should accept optional claim_id', async () => {
        await expect(
          client.broadcastKnowledgeEvent({
            event_type: 'ClaimVerified',
            subject: 'Test subject',
            payload: '{}',
            claim_id: 'claim-1',
          })
        ).resolves.toBeDefined();
      });
    });

    describe('getRecentKnowledgeEvents', () => {
      it('should work without since parameter', async () => {
        await expect(client.getRecentKnowledgeEvents()).resolves.toBeDefined();
      });

      it('should accept valid since timestamp', async () => {
        await expect(client.getRecentKnowledgeEvents(Date.now() - 86400000)).resolves.toBeDefined();
      });

      it('should reject non-positive since', async () => {
        await expect(client.getRecentKnowledgeEvents(0)).rejects.toThrow(MycelixError);
      });

      it('should reject negative since', async () => {
        await expect(client.getRecentKnowledgeEvents(-1)).rejects.toThrow(MycelixError);
      });
    });
  });

  describe('Factory function', () => {
    it('should create all validated clients', () => {
      const clients = createValidatedKnowledgeClients(mockZomeClient);
      expect(clients.claims).toBeInstanceOf(ValidatedClaimsClient);
      expect(clients.graph).toBeInstanceOf(ValidatedGraphClient);
      expect(clients.query).toBeInstanceOf(ValidatedQueryClient);
      expect(clients.inference).toBeInstanceOf(ValidatedInferenceClient);
      expect(clients.bridge).toBeInstanceOf(ValidatedKnowledgeBridgeClient);
    });
  });

  describe('Error handling', () => {
    it('should have INVALID_ARGUMENT error code', async () => {
      const client = new ValidatedClaimsClient(mockZomeClient);
      try {
        await client.getClaim('');
      } catch (e) {
        expect(e).toBeInstanceOf(MycelixError);
        expect((e as MycelixError).code).toBe(ErrorCode.INVALID_ARGUMENT);
      }
    });

    it('should include context in error message', async () => {
      const client = new ValidatedClaimsClient(mockZomeClient);
      try {
        await client.getClaim('');
      } catch (e) {
        expect((e as MycelixError).message).toContain('claimId');
      }
    });
  });
});
