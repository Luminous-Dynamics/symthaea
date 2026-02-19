/**
 * Knowledge Client Tests
 *
 * Verifies zome call arguments and response pass-through for
 * ClaimsClient, GraphClient, QueryClient, InferenceClient, and KnowledgeBridgeClient.
 *
 * Knowledge clients use ZomeCallable interface and return HolochainRecord<T>
 * or plain data depending on the method.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  ClaimsClient,
  GraphClient,
  QueryClient,
  InferenceClient,
  KnowledgeBridgeClient,
  type ZomeCallable,
} from '../index';

// ============================================================================
// MOCK HELPERS
// ============================================================================

function createMockCallable(): ZomeCallable {
  return {
    callZome: vi.fn(),
  } as unknown as ZomeCallable;
}

function mockRecord(entry: Record<string, unknown>) {
  return {
    entry: { Present: entry },
    signed_action: { hashed: { hash: 'mock-hash', content: {} }, signature: 'mock-sig' },
  };
}

// ============================================================================
// MOCK ENTRIES
// ============================================================================

const CLAIM_ENTRY = {
  id: 'claim-001',
  content: 'The Earth is approximately 4.5 billion years old',
  classification: { empirical: 0.95, normative: 0.1, mythic: 0.2 },
  author: 'did:mycelix:geologist',
  sources: ['https://science.nasa.gov/earth'],
  tags: ['geology', 'earth-science'],
  claim_type: 'Fact',
  confidence: 0.99,
  created: 1708200000,
  updated: 1708200000,
  version: 1,
};

const EVIDENCE_ENTRY = {
  id: 'ev-001',
  claim_id: 'claim-001',
  evidence_type: 'Observation',
  source_uri: 'https://example.com/radiometric-dating',
  content: 'Radiometric dating of meteorites',
  strength: 0.95,
  submitted_by: 'did:mycelix:geologist',
  submitted_at: 1708200000,
};

const CHALLENGE_ENTRY = {
  id: 'ch-001',
  claim_id: 'claim-001',
  challenger: 'did:mycelix:skeptic',
  reason: 'Requires independent verification',
  counter_evidence: ['https://example.com/counter'],
  status: 'Pending',
  created: 1708200000,
};

const RELATIONSHIP_ENTRY = {
  id: 'rel-001',
  source: 'claim-001',
  target: 'claim-002',
  relationship_type: 'Supports',
  weight: 0.8,
  creator: 'did:mycelix:geologist',
  created: 1708200000,
};

const ONTOLOGY_ENTRY = {
  id: 'onto-001',
  name: 'Earth Sciences',
  description: 'Ontology for earth science claims',
  namespace: 'https://mycelix.net/ontology/earth-sciences',
  schema: '{}',
  version: '1.0.0',
  creator: 'did:mycelix:geologist',
  created: 1708200000,
  updated: 1708200000,
};

const CONCEPT_ENTRY = {
  id: 'concept-001',
  ontology_id: 'onto-001',
  name: 'Geological Age',
  definition: 'Time period in geological history',
  synonyms: ['era', 'period'],
  created: 1708200000,
};

const SAVED_QUERY_ENTRY = {
  id: 'q-001',
  name: 'High Empirical Claims',
  description: 'Find claims with high empirical validity',
  query: "SELECT * FROM claims WHERE e > 0.8",
  creator: 'did:mycelix:researcher',
  public: true,
  created: 1708200000,
  updated: 1708200000,
};

const INFERENCE_ENTRY = {
  id: 'inf-001',
  inference_type: 'Implication',
  source_claims: ['claim-001', 'claim-002'],
  conclusion: 'Earth formed during early solar system formation',
  confidence: 0.92,
  reasoning: 'Both claims support the same timeline',
  model: 'graph-reasoning-v1',
  created: 1708200000,
  verified: false,
};

const CREDIBILITY_ENTRY = {
  id: 'cred-001',
  subject: 'claim-001',
  subject_type: 'Claim',
  score: 0.91,
  components: {
    accuracy: 0.95,
    consistency: 0.9,
    transparency: 0.88,
    track_record: 0.92,
    corroboration: 0.9,
  },
  factors: ['peer-reviewed sources', 'high corroboration'],
  assessed_at: 1708200000,
};

const PATTERN_ENTRY = {
  id: 'pat-001',
  pattern_type: 'Cluster',
  description: 'Cluster of related geological age claims',
  claims: ['claim-001', 'claim-002', 'claim-003'],
  strength: 0.85,
  detected_at: 1708200000,
};

const BRIDGE_EVENT_ENTRY = {
  id: 'evt-001',
  event_type: 'ClaimCreated',
  claim_id: 'claim-001',
  subject: 'geology',
  payload: '{}',
  source_happ: 'mycelix-desci',
  timestamp: 1708200000,
};

// ============================================================================
// CLAIMS CLIENT TESTS
// ============================================================================

describe('ClaimsClient', () => {
  let mockCallable: ZomeCallable;
  let claims: ClaimsClient;

  beforeEach(() => {
    mockCallable = createMockCallable();
    claims = new ClaimsClient(mockCallable);
  });

  it('submitClaim calls correct zome', async () => {
    const record = mockRecord(CLAIM_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const result = await claims.submitClaim(CLAIM_ENTRY as any);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'knowledge',
      zome_name: 'claims',
      fn_name: 'submit_claim',
      payload: CLAIM_ENTRY,
    });
    expect(result.entry.Present.content).toBe('The Earth is approximately 4.5 billion years old');
    expect(result.entry.Present.classification.empirical).toBe(0.95);
  });

  it('getClaimsByAuthor returns claim list', async () => {
    const record = mockRecord(CLAIM_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([record]);

    const result = await claims.getClaimsByAuthor('did:mycelix:geologist');

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'knowledge',
      zome_name: 'claims',
      fn_name: 'get_claims_by_author',
      payload: 'did:mycelix:geologist',
    });
    expect(result).toHaveLength(1);
  });

  it('addEvidence calls correct zome', async () => {
    const record = mockRecord(EVIDENCE_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const result = await claims.addEvidence(EVIDENCE_ENTRY as any);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'knowledge',
      zome_name: 'claims',
      fn_name: 'add_evidence',
      payload: EVIDENCE_ENTRY,
    });
    expect(result.entry.Present.evidence_type).toBe('Observation');
  });

  it('challengeClaim returns challenge record', async () => {
    const record = mockRecord(CHALLENGE_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = {
      claim_id: 'claim-001',
      challenger_did: 'did:mycelix:skeptic',
      reason: 'Requires independent verification',
      counter_evidence: ['https://example.com/counter'],
    };
    const result = await claims.challengeClaim(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'knowledge',
      zome_name: 'claims',
      fn_name: 'challenge_claim',
      payload: input,
    });
    expect(result.entry.Present.status).toBe('Pending');
  });
});

// ============================================================================
// GRAPH CLIENT TESTS
// ============================================================================

describe('GraphClient', () => {
  let mockCallable: ZomeCallable;
  let graph: GraphClient;

  beforeEach(() => {
    mockCallable = createMockCallable();
    graph = new GraphClient(mockCallable);
  });

  it('createRelationship calls correct zome', async () => {
    const record = mockRecord(RELATIONSHIP_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const result = await graph.createRelationship(RELATIONSHIP_ENTRY as any);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'knowledge',
      zome_name: 'graph',
      fn_name: 'create_relationship',
      payload: RELATIONSHIP_ENTRY,
    });
    expect(result.entry.Present.relationship_type).toBe('Supports');
    expect(result.entry.Present.weight).toBe(0.8);
  });

  it('createOntology returns ontology record', async () => {
    const record = mockRecord(ONTOLOGY_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const result = await graph.createOntology(ONTOLOGY_ENTRY as any);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'knowledge',
      zome_name: 'graph',
      fn_name: 'create_ontology',
      payload: ONTOLOGY_ENTRY,
    });
    expect(result.entry.Present.name).toBe('Earth Sciences');
  });

  it('findPath returns path array', async () => {
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
      'claim-001', 'claim-003', 'claim-005',
    ]);

    const input = { source: 'claim-001', target: 'claim-005', max_depth: 5 };
    const result = await graph.findPath(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'knowledge',
      zome_name: 'graph',
      fn_name: 'find_path',
      payload: input,
    });
    expect(result).toHaveLength(3);
    expect(result[0]).toBe('claim-001');
  });

  it('getGraphStats returns stats', async () => {
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      relationship_count: 42,
    });

    const result = await graph.getGraphStats();

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'knowledge',
      zome_name: 'graph',
      fn_name: 'get_graph_stats',
      payload: null,
    });
    expect(result.relationship_count).toBe(42);
  });
});

// ============================================================================
// QUERY CLIENT TESTS
// ============================================================================

describe('QueryClient', () => {
  let mockCallable: ZomeCallable;
  let query: QueryClient;

  beforeEach(() => {
    mockCallable = createMockCallable();
    query = new QueryClient(mockCallable);
  });

  it('executeQuery returns execution result', async () => {
    const mockResult = {
      results: ['claim-001', 'claim-002'],
      count: 2,
      execution_time_ms: 15,
    };
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockResult);

    const input = {
      query: "SELECT * FROM claims WHERE tag = 'geology'",
      use_cache: true,
      limit: 100,
    };
    const result = await query.executeQuery(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'knowledge',
      zome_name: 'query',
      fn_name: 'execute_query',
      payload: input,
    });
    expect(result.count).toBe(2);
    expect(result.execution_time_ms).toBe(15);
  });

  it('saveQuery returns saved query record', async () => {
    const record = mockRecord(SAVED_QUERY_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const result = await query.saveQuery(SAVED_QUERY_ENTRY as any);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'knowledge',
      zome_name: 'query',
      fn_name: 'save_query',
      payload: SAVED_QUERY_ENTRY,
    });
    expect(result.entry.Present.name).toBe('High Empirical Claims');
    expect(result.entry.Present.public).toBe(true);
  });
});

// ============================================================================
// INFERENCE CLIENT TESTS
// ============================================================================

describe('InferenceClient', () => {
  let mockCallable: ZomeCallable;
  let inference: InferenceClient;

  beforeEach(() => {
    mockCallable = createMockCallable();
    inference = new InferenceClient(mockCallable);
  });

  it('createInference calls correct zome', async () => {
    const record = mockRecord(INFERENCE_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const result = await inference.createInference(INFERENCE_ENTRY as any);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'knowledge',
      zome_name: 'inference',
      fn_name: 'create_inference',
      payload: INFERENCE_ENTRY,
    });
    expect(result.entry.Present.inference_type).toBe('Implication');
    expect(result.entry.Present.confidence).toBe(0.92);
  });

  it('assessCredibility returns credibility record', async () => {
    const record = mockRecord(CREDIBILITY_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = { subject: 'claim-001', subject_type: 'Claim' as const };
    const result = await inference.assessCredibility(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'knowledge',
      zome_name: 'inference',
      fn_name: 'assess_credibility',
      payload: input,
    });
    expect(result.entry.Present.score).toBe(0.91);
    expect(result.entry.Present.components.accuracy).toBe(0.95);
  });

  it('detectPatterns returns pattern list', async () => {
    const record = mockRecord(PATTERN_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([record]);

    const input = {
      claims: ['claim-001', 'claim-002', 'claim-003'],
      pattern_types: ['Cluster' as const],
    };
    const result = await inference.detectPatterns(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'knowledge',
      zome_name: 'inference',
      fn_name: 'detect_patterns',
      payload: input,
    });
    expect(result).toHaveLength(1);
    expect(result[0].entry.Present.pattern_type).toBe('Cluster');
  });
});

// ============================================================================
// KNOWLEDGE BRIDGE CLIENT TESTS
// ============================================================================

describe('KnowledgeBridgeClient', () => {
  let mockCallable: ZomeCallable;
  let bridge: KnowledgeBridgeClient;

  beforeEach(() => {
    mockCallable = createMockCallable();
    bridge = new KnowledgeBridgeClient(mockCallable);
  });

  it('factCheck returns fact-check result', async () => {
    const mockResult = {
      statement: 'The economy grew by 3% last quarter',
      verdict: 'PartiallyTrue',
      confidence: 0.72,
      supporting_claims: ['claim-010'],
      contradicting_claims: ['claim-011'],
    };
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockResult);

    const input = {
      source_happ: 'mycelix-news',
      statement: 'The economy grew by 3% last quarter',
    };
    const result = await bridge.factCheck(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'knowledge',
      zome_name: 'knowledge_bridge',
      fn_name: 'fact_check',
      payload: input,
    });
    expect(result.verdict).toBe('PartiallyTrue');
    expect(result.confidence).toBe(0.72);
  });

  it('broadcastKnowledgeEvent returns event record', async () => {
    const record = mockRecord(BRIDGE_EVENT_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = {
      event_type: 'ClaimCreated' as const,
      claim_id: 'claim-001',
      subject: 'geology',
      payload: '{}',
    };
    const result = await bridge.broadcastKnowledgeEvent(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'knowledge',
      zome_name: 'knowledge_bridge',
      fn_name: 'broadcast_knowledge_event',
      payload: input,
    });
    expect(result.entry.Present.event_type).toBe('ClaimCreated');
    expect(result.entry.Present.source_happ).toBe('mycelix-desci');
  });
});
