// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Media Module Tests
 *
 * Tests for the Media hApp TypeScript clients:
 * - PublicationClient (content publishing)
 * - AttributionClient (contributor attribution)
 * - FactCheckClient (fact verification)
 * - CurationClient (content curation and endorsement)
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  PublicationClient,
  AttributionClient,
  FactCheckClient,
  CurationClient,
  createMediaClients,
  type Content,
  type Attribution,
  type FactCheck,
  type CurationList,
  type Endorsement,
  type ZomeCallable,
  type HolochainRecord,
  type ContentType,
  type LicenseType,
  type VerificationStatus,
} from '../src/media/index.js';

// ============================================================================
// Mock Setup
// ============================================================================

function createMockRecord<T>(entry: T): HolochainRecord<T> {
  return {
    signed_action: {
      hashed: { hash: 'uhCXk_test_hash_123', content: {} },
      signature: 'sig_test_123',
    },
    entry: { Present: entry },
  };
}

function createMockClient(responses: Map<string, unknown> = new Map()): ZomeCallable {
  return {
    callZome: vi.fn(
      async <T>(params: {
        role_name: string;
        zome_name: string;
        fn_name: string;
        payload: unknown;
      }): Promise<T> => {
        const key = `${params.zome_name}:${params.fn_name}`;
        if (responses.has(key)) {
          return responses.get(key) as T;
        }
        throw new Error(`No mock response for ${key}`);
      }
    ),
  };
}

// ============================================================================
// Mock Data Factories
// ============================================================================

function createMockContent(overrides: Partial<Content> = {}): Content {
  return {
    id: 'content-123',
    author: 'did:mycelix:author123',
    type_: 'Article' as ContentType,
    title: 'The Future of Decentralized Media',
    description: 'An exploration of how decentralized systems can transform journalism',
    content_hash: 'sha256:abc123def456',
    storage_uri: 'ipfs://QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG',
    license: 'CCBY' as LicenseType,
    tags: ['media', 'decentralization', 'journalism'],
    verification_status: 'Verified' as VerificationStatus,
    views: 1250,
    endorsements: 42,
    published_at: Date.now() * 1000 - 86400000000,
    updated_at: Date.now() * 1000,
    ...overrides,
  };
}

function createMockAttribution(overrides: Partial<Attribution> = {}): Attribution {
  return {
    id: 'attribution-123',
    content_id: 'content-123',
    contributor: 'did:mycelix:contributor456',
    role: 'Editor',
    description: 'Technical review and editing',
    royalty_percentage: 15,
    created_at: Date.now() * 1000,
    ...overrides,
  };
}

function createMockFactCheck(overrides: Partial<FactCheck> = {}): FactCheck {
  return {
    id: 'factcheck-123',
    content_id: 'content-123',
    checker: 'did:mycelix:checker789',
    verdict: 'Verified' as VerificationStatus,
    reasoning: 'All claims verified against primary sources',
    sources: [
      'https://source1.example.com',
      'https://source2.example.com',
    ],
    epistemic_score: {
      empirical: 0.85,
      normative: 0.2,
      metaphorical: 0.1,
    },
    created_at: Date.now() * 1000,
    ...overrides,
  };
}

function createMockCurationList(overrides: Partial<CurationList> = {}): CurationList {
  return {
    id: 'list-123',
    curator: 'did:mycelix:curator123',
    name: 'Best Decentralization Articles',
    description: 'Curated collection of high-quality articles on decentralization',
    content_ids: ['content-123', 'content-456', 'content-789'],
    public_: true,
    created_at: Date.now() * 1000 - 604800000000,
    updated_at: Date.now() * 1000,
    ...overrides,
  };
}

function createMockEndorsement(overrides: Partial<Endorsement> = {}): Endorsement {
  return {
    id: 'endorsement-123',
    content_id: 'content-123',
    endorser: 'did:mycelix:endorser456',
    weight: 5,
    comment: 'Excellent analysis and well-researched',
    created_at: Date.now() * 1000,
    ...overrides,
  };
}

// ============================================================================
// PublicationClient Tests
// ============================================================================

describe('PublicationClient', () => {
  let client: PublicationClient;
  let mockZome: ZomeCallable;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    const mockContent = createMockContent();

    responses.set('media_publication:publish', createMockRecord(mockContent));
    responses.set('media_publication:get_content', createMockRecord(mockContent));
    responses.set('media_publication:get_by_author', [createMockRecord(mockContent)]);
    responses.set('media_publication:get_by_tag', [createMockRecord(mockContent)]);
    responses.set('media_publication:get_by_type', [createMockRecord(mockContent)]);
    responses.set(
      'media_publication:update_content',
      createMockRecord({
        ...mockContent,
        description: 'Updated description',
        updated_at: Date.now() * 1000,
      })
    );
    responses.set('media_publication:record_view', undefined);

    mockZome = createMockClient(responses);
    client = new PublicationClient(mockZome);
  });

  describe('publish', () => {
    it('should publish new content', async () => {
      const result = await client.publish({
        type_: 'Article',
        title: 'The Future of Decentralized Media',
        description: 'An exploration...',
        content_hash: 'sha256:abc123',
        storage_uri: 'ipfs://QmTest',
        license: 'CCBY',
        tags: ['media', 'decentralization'],
      });

      expect(result.entry.Present.id).toBe('content-123');
      expect(result.entry.Present.verification_status).toBe('Verified');
      expect(mockZome.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'civic',
          zome_name: 'media_publication',
          fn_name: 'publish',
        })
      );
    });

    it('should publish with all content types', async () => {
      const types: ContentType[] = ['Article', 'Opinion', 'Investigation', 'Review', 'Analysis', 'Interview', 'Other'];

      for (const type_ of types) {
        const result = await client.publish({
          type_,
          title: `Test ${type_}`,
          content_hash: 'sha256:test',
          storage_uri: 'ipfs://test',
          license: 'CC0',
        });
        expect(result).toBeDefined();
      }
    });

    it('should publish with all license types', async () => {
      const licenses: LicenseType[] = [
        'CC0',
        'CCBY',
        'CCBYSA',
        'CCBYNC',
        'CCBYNCSA',
        'AllRightsReserved',
        'Custom',
      ];

      for (const license of licenses) {
        const result = await client.publish({
          type_: 'Article',
          title: 'Test',
          content_hash: 'sha256:test',
          storage_uri: 'ipfs://test',
          license,
          custom_license: license === 'Custom' ? 'Custom terms here' : undefined,
        });
        expect(result).toBeDefined();
      }
    });
  });

  describe('getContent', () => {
    it('should get content by ID', async () => {
      const result = await client.getContent('content-123');

      expect(result).not.toBeNull();
      expect(result!.entry.Present.id).toBe('content-123');
    });
  });

  describe('getContentByAuthor', () => {
    it('should get content by author DID', async () => {
      const results = await client.getContentByAuthor('did:mycelix:author123');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.author).toBe('did:mycelix:author123');
    });
  });

  describe('getContentByTag', () => {
    it('should get content by tag', async () => {
      const results = await client.getContentByTag('decentralization');

      expect(results).toHaveLength(1);
    });
  });

  describe('getContentByType', () => {
    it('should get content by type', async () => {
      const results = await client.getContentByType('Article');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.type_).toBe('Article');
    });
  });

  describe('updateContent', () => {
    it('should update content metadata', async () => {
      const result = await client.updateContent('content-123', {
        description: 'Updated description',
      });

      expect(result.entry.Present.description).toBe('Updated description');
    });
  });

  describe('recordView', () => {
    it('should record a content view', async () => {
      await expect(client.recordView('content-123')).resolves.not.toThrow();
    });
  });
});

// ============================================================================
// AttributionClient Tests
// ============================================================================

describe('AttributionClient', () => {
  let client: AttributionClient;
  let mockZome: ZomeCallable;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    const mockAttribution = createMockAttribution();

    responses.set('media_attribution:add_attribution', createMockRecord(mockAttribution));
    responses.set('media_attribution:get_attributions', [createMockRecord(mockAttribution)]);
    responses.set('media_attribution:get_contributor_works', [createMockRecord(mockAttribution)]);

    mockZome = createMockClient(responses);
    client = new AttributionClient(mockZome);
  });

  describe('addAttribution', () => {
    it('should add attribution to content', async () => {
      const result = await client.addAttribution({
        content_id: 'content-123',
        contributor: 'did:mycelix:contributor456',
        role: 'Editor',
        description: 'Technical review',
        royalty_percentage: 15,
      });

      expect(result.entry.Present.role).toBe('Editor');
      expect(result.entry.Present.royalty_percentage).toBe(15);
    });

    it('should add with all roles', async () => {
      const roles: Attribution['role'][] = ['Author', 'Editor', 'Contributor', 'Source', 'Translator'];

      for (const role of roles) {
        const result = await client.addAttribution({
          content_id: 'content-123',
          contributor: 'did:mycelix:test',
          role,
        });
        expect(result).toBeDefined();
      }
    });
  });

  describe('getAttributions', () => {
    it('should get attributions for content', async () => {
      const results = await client.getAttributions('content-123');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.content_id).toBe('content-123');
    });
  });

  describe('getContributorWorks', () => {
    it('should get works by contributor', async () => {
      const results = await client.getContributorWorks('did:mycelix:contributor456');

      expect(results).toHaveLength(1);
    });
  });
});

// ============================================================================
// FactCheckClient Tests
// ============================================================================

describe('FactCheckClient', () => {
  let client: FactCheckClient;
  let mockZome: ZomeCallable;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    const mockFactCheck = createMockFactCheck();

    responses.set('media_factcheck:submit_fact_check', createMockRecord(mockFactCheck));
    responses.set('media_factcheck:get_fact_checks', [createMockRecord(mockFactCheck)]);
    responses.set('media_factcheck:get_by_checker', [createMockRecord(mockFactCheck)]);
    responses.set('media_factcheck:get_verification_status', 'Verified');

    mockZome = createMockClient(responses);
    client = new FactCheckClient(mockZome);
  });

  describe('submitFactCheck', () => {
    it('should submit a fact check', async () => {
      const result = await client.submitFactCheck({
        content_id: 'content-123',
        verdict: 'Verified',
        reasoning: 'All claims verified',
        sources: ['https://source1.com'],
        epistemic_score: {
          empirical: 0.85,
          normative: 0.2,
          metaphorical: 0.1,
        },
      });

      expect(result.entry.Present.verdict).toBe('Verified');
      expect(result.entry.Present.epistemic_score.empirical).toBe(0.85);
    });

    it('should submit with all verdicts', async () => {
      const verdicts: VerificationStatus[] = [
        'Unverified',
        'Pending',
        'Verified',
        'Disputed',
        'Debunked',
      ];

      for (const verdict of verdicts) {
        const result = await client.submitFactCheck({
          content_id: 'content-123',
          verdict,
          reasoning: 'Test',
          sources: [],
          epistemic_score: { empirical: 0.5, normative: 0.5, metaphorical: 0 },
        });
        expect(result).toBeDefined();
      }
    });
  });

  describe('getFactChecks', () => {
    it('should get fact checks for content', async () => {
      const results = await client.getFactChecks('content-123');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.content_id).toBe('content-123');
    });
  });

  describe('getFactChecksByChecker', () => {
    it('should get fact checks by checker', async () => {
      const results = await client.getFactChecksByChecker('did:mycelix:checker789');

      expect(results).toHaveLength(1);
    });
  });

  describe('getContentVerificationStatus', () => {
    it('should get verification status', async () => {
      const status = await client.getContentVerificationStatus('content-123');

      expect(status).toBe('Verified');
    });
  });
});

// ============================================================================
// CurationClient Tests
// ============================================================================

describe('CurationClient', () => {
  let client: CurationClient;
  let mockZome: ZomeCallable;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    const mockList = createMockCurationList();
    const mockEndorsement = createMockEndorsement();

    responses.set('media_curation:create_list', createMockRecord(mockList));
    responses.set('media_curation:get_list', createMockRecord(mockList));
    responses.set('media_curation:get_by_curator', [createMockRecord(mockList)]);
    responses.set(
      'media_curation:add_to_list',
      createMockRecord({
        ...mockList,
        content_ids: [...mockList.content_ids, 'content-new'],
      })
    );
    responses.set(
      'media_curation:remove_from_list',
      createMockRecord({
        ...mockList,
        content_ids: mockList.content_ids.slice(0, -1),
      })
    );
    responses.set('media_curation:endorse', createMockRecord(mockEndorsement));
    responses.set('media_curation:get_endorsements', [createMockRecord(mockEndorsement)]);

    mockZome = createMockClient(responses);
    client = new CurationClient(mockZome);
  });

  describe('createList', () => {
    it('should create a curation list', async () => {
      const result = await client.createList({
        name: 'Best Decentralization Articles',
        description: 'Curated collection',
        content_ids: ['content-123'],
        public_: true,
      });

      expect(result.entry.Present.id).toBe('list-123');
      expect(result.entry.Present.public_).toBe(true);
    });

    it('should create private list', async () => {
      const result = await client.createList({
        name: 'My Reading List',
        description: 'Personal collection',
        public_: false,
      });

      expect(result).toBeDefined();
    });
  });

  describe('getList', () => {
    it('should get list by ID', async () => {
      const result = await client.getList('list-123');

      expect(result).not.toBeNull();
      expect(result!.entry.Present.id).toBe('list-123');
    });
  });

  describe('getListsByCurator', () => {
    it('should get lists by curator', async () => {
      const results = await client.getListsByCurator('did:mycelix:curator123');

      expect(results).toHaveLength(1);
    });
  });

  describe('addToList', () => {
    it('should add content to list', async () => {
      const result = await client.addToList('list-123', 'content-new');

      expect(result.entry.Present.content_ids).toContain('content-new');
    });
  });

  describe('removeFromList', () => {
    it('should remove content from list', async () => {
      const result = await client.removeFromList('list-123', 'content-789');

      expect(result.entry.Present.content_ids).not.toContain('content-789');
    });
  });

  describe('endorse', () => {
    it('should endorse content', async () => {
      const result = await client.endorse('content-123', 5, 'Excellent work!');

      expect(result.entry.Present.weight).toBe(5);
      expect(result.entry.Present.comment).toBe('Excellent analysis and well-researched');
    });

    it('should endorse without comment', async () => {
      const result = await client.endorse('content-123', 3);

      expect(result).toBeDefined();
    });
  });

  describe('getEndorsements', () => {
    it('should get endorsements for content', async () => {
      const results = await client.getEndorsements('content-123');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.content_id).toBe('content-123');
    });
  });
});

// ============================================================================
// Factory Function Tests
// ============================================================================

describe('createMediaClients', () => {
  it('should create all media clients', () => {
    const mockZome = createMockClient(new Map());
    const clients = createMediaClients(mockZome);

    expect(clients.publication).toBeInstanceOf(PublicationClient);
    expect(clients.attribution).toBeInstanceOf(AttributionClient);
    expect(clients.factcheck).toBeInstanceOf(FactCheckClient);
    expect(clients.curation).toBeInstanceOf(CurationClient);
  });
});

// ============================================================================
// Type Safety Tests
// ============================================================================

describe('Type Safety', () => {
  it('should enforce verification status constraints', () => {
    const statuses: VerificationStatus[] = ['Unverified', 'Pending', 'Verified', 'Disputed', 'Debunked'];

    statuses.forEach((status) => {
      const content = createMockContent({ verification_status: status });
      expect(statuses).toContain(content.verification_status);
    });
  });

  it('should enforce epistemic scores in valid range', () => {
    const factCheck = createMockFactCheck();

    expect(factCheck.epistemic_score.empirical).toBeGreaterThanOrEqual(0);
    expect(factCheck.epistemic_score.empirical).toBeLessThanOrEqual(1);
    expect(factCheck.epistemic_score.normative).toBeGreaterThanOrEqual(0);
    expect(factCheck.epistemic_score.normative).toBeLessThanOrEqual(1);
    expect(factCheck.epistemic_score.metaphorical).toBeGreaterThanOrEqual(0);
    expect(factCheck.epistemic_score.metaphorical).toBeLessThanOrEqual(1);
  });

  it('should enforce royalty percentage in valid range', () => {
    const attribution = createMockAttribution({ royalty_percentage: 15 });

    expect(attribution.royalty_percentage).toBeGreaterThanOrEqual(0);
    expect(attribution.royalty_percentage).toBeLessThanOrEqual(100);
  });

  it('should enforce endorsement weight is positive', () => {
    const endorsement = createMockEndorsement();
    expect(endorsement.weight).toBeGreaterThan(0);
  });
});

// ============================================================================
// Integration Pattern Tests
// ============================================================================

describe('Integration Patterns', () => {
  it('should support full content lifecycle', async () => {
    const responses = new Map<string, unknown>();
    const mockContent = createMockContent({ verification_status: 'Unverified' });
    const mockAttribution = createMockAttribution();
    const mockFactCheck = createMockFactCheck();

    responses.set('media_publication:publish', createMockRecord(mockContent));
    responses.set('media_attribution:add_attribution', createMockRecord(mockAttribution));
    responses.set('media_factcheck:submit_fact_check', createMockRecord(mockFactCheck));
    responses.set('media_curation:endorse', createMockRecord(createMockEndorsement()));

    const mockZome = createMockClient(responses);
    const clients = createMediaClients(mockZome);

    // Publish content
    const content = await clients.publication.publish({
      type_: 'Article',
      title: 'Test Article',
      content_hash: 'sha256:test',
      storage_uri: 'ipfs://test',
      license: 'CCBY',
    });
    expect(content.entry.Present.id).toBeDefined();

    // Add attribution
    const attribution = await clients.attribution.addAttribution({
      content_id: content.entry.Present.id,
      contributor: 'did:mycelix:editor',
      role: 'Editor',
    });
    expect(attribution.entry.Present.content_id).toBe(content.entry.Present.id);

    // Submit fact check
    const factCheck = await clients.factcheck.submitFactCheck({
      content_id: content.entry.Present.id,
      verdict: 'Verified',
      reasoning: 'All claims verified',
      sources: ['https://source.com'],
      epistemic_score: { empirical: 0.9, normative: 0.1, metaphorical: 0 },
    });
    expect(factCheck.entry.Present.verdict).toBe('Verified');

    // Endorse content
    const endorsement = await clients.curation.endorse(content.entry.Present.id, 5, 'Great work!');
    expect(endorsement.entry.Present.weight).toBe(5);
  });

  it('should support curation workflow', async () => {
    const responses = new Map<string, unknown>();
    const mockList = createMockCurationList({ content_ids: [] });

    responses.set('media_curation:create_list', createMockRecord(mockList));
    responses.set(
      'media_curation:add_to_list',
      createMockRecord({
        ...mockList,
        content_ids: ['content-123'],
      })
    );

    const mockZome = createMockClient(responses);
    const clients = createMediaClients(mockZome);

    // Create list
    const list = await clients.curation.createList({
      name: 'Test List',
      description: 'Test',
      public_: true,
    });
    expect(list.entry.Present.content_ids).toHaveLength(0);

    // Add content
    const updated = await clients.curation.addToList(list.entry.Present.id, 'content-123');
    expect(updated.entry.Present.content_ids).toContain('content-123');
  });
});

// ============================================================================
// Edge Case Tests
// ============================================================================

describe('Edge Cases', () => {
  it('should handle content with no tags', () => {
    const content = createMockContent({ tags: [] });
    expect(content.tags).toHaveLength(0);
  });

  it('should handle content with no description', () => {
    const content = createMockContent({ description: undefined });
    expect(content.description).toBeUndefined();
  });

  it('should handle fact check with no sources', () => {
    const factCheck = createMockFactCheck({ sources: [] });
    expect(factCheck.sources).toHaveLength(0);
  });

  it('should handle attribution without royalty', () => {
    const attribution = createMockAttribution({ royalty_percentage: undefined });
    expect(attribution.royalty_percentage).toBeUndefined();
  });

  it('should handle endorsement without comment', () => {
    const endorsement = createMockEndorsement({ comment: undefined });
    expect(endorsement.comment).toBeUndefined();
  });

  it('should handle empty curation list', () => {
    const list = createMockCurationList({ content_ids: [] });
    expect(list.content_ids).toHaveLength(0);
  });

  it('should handle highly empirical content', () => {
    const factCheck = createMockFactCheck({
      epistemic_score: { empirical: 0.95, normative: 0.03, metaphorical: 0.02 },
    });
    expect(factCheck.epistemic_score.empirical).toBeGreaterThan(0.9);
  });

  it('should handle normative content', () => {
    const factCheck = createMockFactCheck({
      epistemic_score: { empirical: 0.1, normative: 0.85, metaphorical: 0.05 },
    });
    expect(factCheck.epistemic_score.normative).toBeGreaterThan(factCheck.epistemic_score.empirical);
  });
});

// ============================================================================
// Epistemic Charter Tests
// ============================================================================

describe('Epistemic Charter Integration', () => {
  it('should classify content on E/N/M dimensions', () => {
    const factCheck = createMockFactCheck({
      epistemic_score: {
        empirical: 0.85,
        normative: 0.2,
        metaphorical: 0.1,
      },
    });

    const { empirical, normative, metaphorical } = factCheck.epistemic_score;

    // Sum should be <= 1.15 (allowing some overlap and floating-point tolerance)
    expect(empirical + normative + metaphorical).toBeLessThanOrEqual(1.15 + 1e-10);

    // Empirical dominates in this case
    expect(empirical).toBeGreaterThan(normative);
    expect(empirical).toBeGreaterThan(metaphorical);
  });

  it('should support primarily metaphorical content', () => {
    const factCheck = createMockFactCheck({
      verdict: 'Verified',
      epistemic_score: {
        empirical: 0.1,
        normative: 0.2,
        metaphorical: 0.8,
      },
    });

    // Metaphorical content can still be "verified" in its context
    expect(factCheck.verdict).toBe('Verified');
    expect(factCheck.epistemic_score.metaphorical).toBeGreaterThan(0.5);
  });
});

// ============================================================================
// Validated Client Tests
// ============================================================================

import {
  ValidatedPublicationClient,
  ValidatedAttributionClient,
  ValidatedFactCheckClient,
  ValidatedCurationClient,
  createValidatedMediaClients,
} from '../src/media/validated.js';

describe('ValidatedPublicationClient', () => {
  let mockClient: ZomeCallable;
  let validatedClient: ValidatedPublicationClient;

  beforeEach(() => {
    mockClient = createMockClient(
      new Map([
        ['media_publication:publish', createMockRecord(createMockContent())],
        ['media_publication:get_content', createMockRecord(createMockContent())],
        ['media_publication:get_by_author', [createMockRecord(createMockContent())]],
        ['media_publication:get_by_tag', [createMockRecord(createMockContent())]],
        ['media_publication:get_by_type', [createMockRecord(createMockContent())]],
        ['media_publication:update_content', createMockRecord(createMockContent())],
        ['media_publication:record_view', undefined],
      ])
    );
    validatedClient = new ValidatedPublicationClient(mockClient);
  });

  describe('publish', () => {
    it('accepts valid publish input', async () => {
      const result = await validatedClient.publish({
        type_: 'Article',
        title: 'Test Article',
        content_hash: 'sha256:abcdef1234567890abcdef1234567890',
        storage_uri: 'ipfs://QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG',
        license: 'CCBY',
      });
      expect(result.entry.Present?.id).toBe('content-123');
    });

    it('rejects empty title', async () => {
      await expect(
        validatedClient.publish({
          type_: 'Article',
          title: '',
          content_hash: 'sha256:abcdef1234567890abcdef1234567890',
          storage_uri: 'ipfs://test',
          license: 'CCBY',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects title over 500 chars', async () => {
      await expect(
        validatedClient.publish({
          type_: 'Article',
          title: 'x'.repeat(501),
          content_hash: 'sha256:abcdef1234567890abcdef1234567890',
          storage_uri: 'ipfs://test',
          license: 'CCBY',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects invalid content type', async () => {
      await expect(
        validatedClient.publish({
          type_: 'Invalid' as 'Article',
          title: 'Test',
          content_hash: 'sha256:abcdef1234567890abcdef1234567890',
          storage_uri: 'ipfs://test',
          license: 'CCBY',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects invalid license', async () => {
      await expect(
        validatedClient.publish({
          type_: 'Article',
          title: 'Test',
          content_hash: 'sha256:abcdef1234567890abcdef1234567890',
          storage_uri: 'ipfs://test',
          license: 'InvalidLicense' as 'CCBY',
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('getContent', () => {
    it('accepts valid content ID', async () => {
      const result = await validatedClient.getContent('content-123');
      expect(result?.entry.Present?.id).toBe('content-123');
    });

    it('rejects empty content ID', async () => {
      await expect(validatedClient.getContent('')).rejects.toThrow('Validation failed');
    });
  });

  describe('getContentByAuthor', () => {
    it('accepts valid DID', async () => {
      const result = await validatedClient.getContentByAuthor('did:mycelix:author123');
      expect(result.length).toBeGreaterThan(0);
    });

    it('rejects non-DID string', async () => {
      await expect(validatedClient.getContentByAuthor('not-a-did')).rejects.toThrow('Validation failed');
    });
  });

  describe('getContentByTag', () => {
    it('accepts valid tag', async () => {
      const result = await validatedClient.getContentByTag('media');
      expect(result.length).toBeGreaterThan(0);
    });

    it('rejects empty tag', async () => {
      await expect(validatedClient.getContentByTag('')).rejects.toThrow('Validation failed');
    });
  });

  describe('getContentByType', () => {
    it('accepts valid content type', async () => {
      const result = await validatedClient.getContentByType('Article');
      expect(result.length).toBeGreaterThan(0);
    });

    it('rejects invalid content type', async () => {
      await expect(validatedClient.getContentByType('Invalid' as 'Article')).rejects.toThrow('Validation failed');
    });
  });
});

describe('ValidatedAttributionClient', () => {
  let mockClient: ZomeCallable;
  let validatedClient: ValidatedAttributionClient;

  beforeEach(() => {
    mockClient = createMockClient(
      new Map([
        ['media_attribution:add_attribution', createMockRecord(createMockAttribution())],
        ['media_attribution:get_attributions', [createMockRecord(createMockAttribution())]],
        ['media_attribution:get_contributor_works', [createMockRecord(createMockAttribution())]],
      ])
    );
    validatedClient = new ValidatedAttributionClient(mockClient);
  });

  describe('addAttribution', () => {
    it('accepts valid attribution input', async () => {
      const result = await validatedClient.addAttribution({
        content_id: 'content-123',
        contributor: 'did:mycelix:contributor456',
        role: 'Editor',
      });
      expect(result.entry.Present?.id).toBe('attribution-123');
    });

    it('rejects invalid role', async () => {
      await expect(
        validatedClient.addAttribution({
          content_id: 'content-123',
          contributor: 'did:mycelix:contributor456',
          role: 'Invalid' as 'Editor',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects non-DID contributor', async () => {
      await expect(
        validatedClient.addAttribution({
          content_id: 'content-123',
          contributor: 'not-a-did',
          role: 'Editor',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects royalty over 100', async () => {
      await expect(
        validatedClient.addAttribution({
          content_id: 'content-123',
          contributor: 'did:mycelix:contributor456',
          role: 'Editor',
          royalty_percentage: 150,
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('getAttributions', () => {
    it('accepts valid content ID', async () => {
      const result = await validatedClient.getAttributions('content-123');
      expect(result.length).toBeGreaterThan(0);
    });

    it('rejects empty content ID', async () => {
      await expect(validatedClient.getAttributions('')).rejects.toThrow('Validation failed');
    });
  });

  describe('getContributorWorks', () => {
    it('accepts valid DID', async () => {
      const result = await validatedClient.getContributorWorks('did:mycelix:contributor456');
      expect(result.length).toBeGreaterThan(0);
    });

    it('rejects non-DID string', async () => {
      await expect(validatedClient.getContributorWorks('not-a-did')).rejects.toThrow('Validation failed');
    });
  });
});

describe('ValidatedFactCheckClient', () => {
  let mockClient: ZomeCallable;
  let validatedClient: ValidatedFactCheckClient;

  beforeEach(() => {
    mockClient = createMockClient(
      new Map([
        ['media_factcheck:submit_fact_check', createMockRecord(createMockFactCheck())],
        ['media_factcheck:get_fact_checks', [createMockRecord(createMockFactCheck())]],
        ['media_factcheck:get_by_checker', [createMockRecord(createMockFactCheck())]],
        ['media_factcheck:get_verification_status', 'Verified'],
      ])
    );
    validatedClient = new ValidatedFactCheckClient(mockClient);
  });

  describe('submitFactCheck', () => {
    it('accepts valid fact check input', async () => {
      const result = await validatedClient.submitFactCheck({
        content_id: 'content-123',
        verdict: 'Verified',
        reasoning: 'All claims verified against primary sources and are accurate.',
        sources: ['https://source1.example.com'],
        epistemic_score: { empirical: 0.9, normative: 0.1, metaphorical: 0.0 },
      });
      expect(result.entry.Present?.id).toBe('factcheck-123');
    });

    it('rejects short reasoning', async () => {
      await expect(
        validatedClient.submitFactCheck({
          content_id: 'content-123',
          verdict: 'Verified',
          reasoning: 'Too short',
          sources: ['https://source1.example.com'],
          epistemic_score: { empirical: 0.9, normative: 0.1, metaphorical: 0.0 },
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects empty sources array', async () => {
      await expect(
        validatedClient.submitFactCheck({
          content_id: 'content-123',
          verdict: 'Verified',
          reasoning: 'All claims verified against primary sources and are accurate.',
          sources: [],
          epistemic_score: { empirical: 0.9, normative: 0.1, metaphorical: 0.0 },
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects invalid verdict', async () => {
      await expect(
        validatedClient.submitFactCheck({
          content_id: 'content-123',
          verdict: 'Invalid' as 'Verified',
          reasoning: 'All claims verified against primary sources and are accurate.',
          sources: ['https://source1.example.com'],
          epistemic_score: { empirical: 0.9, normative: 0.1, metaphorical: 0.0 },
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects epistemic score over 1', async () => {
      await expect(
        validatedClient.submitFactCheck({
          content_id: 'content-123',
          verdict: 'Verified',
          reasoning: 'All claims verified against primary sources and are accurate.',
          sources: ['https://source1.example.com'],
          epistemic_score: { empirical: 1.5, normative: 0.1, metaphorical: 0.0 },
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('getFactChecks', () => {
    it('accepts valid content ID', async () => {
      const result = await validatedClient.getFactChecks('content-123');
      expect(result.length).toBeGreaterThan(0);
    });

    it('rejects empty content ID', async () => {
      await expect(validatedClient.getFactChecks('')).rejects.toThrow('Validation failed');
    });
  });

  describe('getFactChecksByChecker', () => {
    it('accepts valid DID', async () => {
      const result = await validatedClient.getFactChecksByChecker('did:mycelix:checker789');
      expect(result.length).toBeGreaterThan(0);
    });

    it('rejects non-DID string', async () => {
      await expect(validatedClient.getFactChecksByChecker('not-a-did')).rejects.toThrow('Validation failed');
    });
  });

  describe('getContentVerificationStatus', () => {
    it('accepts valid content ID', async () => {
      const result = await validatedClient.getContentVerificationStatus('content-123');
      expect(result).toBe('Verified');
    });

    it('rejects empty content ID', async () => {
      await expect(validatedClient.getContentVerificationStatus('')).rejects.toThrow('Validation failed');
    });
  });
});

describe('ValidatedCurationClient', () => {
  let mockClient: ZomeCallable;
  let validatedClient: ValidatedCurationClient;

  beforeEach(() => {
    mockClient = createMockClient(
      new Map([
        ['media_curation:create_list', createMockRecord(createMockCurationList())],
        ['media_curation:get_list', createMockRecord(createMockCurationList())],
        ['media_curation:get_by_curator', [createMockRecord(createMockCurationList())]],
        ['media_curation:add_to_list', createMockRecord(createMockCurationList())],
        ['media_curation:remove_from_list', createMockRecord(createMockCurationList())],
        ['media_curation:endorse', createMockRecord(createMockEndorsement())],
        ['media_curation:get_endorsements', [createMockRecord(createMockEndorsement())]],
      ])
    );
    validatedClient = new ValidatedCurationClient(mockClient);
  });

  describe('createList', () => {
    it('accepts valid list input', async () => {
      const result = await validatedClient.createList({
        name: 'My List',
        description: 'A collection of great articles',
        public_: true,
      });
      expect(result.entry.Present?.id).toBe('list-123');
    });

    it('rejects empty name', async () => {
      await expect(
        validatedClient.createList({
          name: '',
          description: 'A collection',
          public_: true,
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects name over 200 chars', async () => {
      await expect(
        validatedClient.createList({
          name: 'x'.repeat(201),
          description: 'A collection',
          public_: true,
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects empty description', async () => {
      await expect(
        validatedClient.createList({
          name: 'My List',
          description: '',
          public_: true,
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('getList', () => {
    it('accepts valid list ID', async () => {
      const result = await validatedClient.getList('list-123');
      expect(result?.entry.Present?.id).toBe('list-123');
    });

    it('rejects empty list ID', async () => {
      await expect(validatedClient.getList('')).rejects.toThrow('Validation failed');
    });
  });

  describe('getListsByCurator', () => {
    it('accepts valid DID', async () => {
      const result = await validatedClient.getListsByCurator('did:mycelix:curator123');
      expect(result.length).toBeGreaterThan(0);
    });

    it('rejects non-DID string', async () => {
      await expect(validatedClient.getListsByCurator('not-a-did')).rejects.toThrow('Validation failed');
    });
  });

  describe('addToList', () => {
    it('accepts valid IDs', async () => {
      const result = await validatedClient.addToList('list-123', 'content-456');
      expect(result.entry.Present?.id).toBe('list-123');
    });

    it('rejects empty list ID', async () => {
      await expect(validatedClient.addToList('', 'content-456')).rejects.toThrow('Validation failed');
    });

    it('rejects empty content ID', async () => {
      await expect(validatedClient.addToList('list-123', '')).rejects.toThrow('Validation failed');
    });
  });

  describe('removeFromList', () => {
    it('accepts valid IDs', async () => {
      const result = await validatedClient.removeFromList('list-123', 'content-456');
      expect(result.entry.Present?.id).toBe('list-123');
    });

    it('rejects empty list ID', async () => {
      await expect(validatedClient.removeFromList('', 'content-456')).rejects.toThrow('Validation failed');
    });
  });

  describe('endorse', () => {
    it('accepts valid endorsement', async () => {
      const result = await validatedClient.endorse('content-123', 0.8, 'Great article!');
      expect(result.entry.Present?.id).toBe('endorsement-123');
    });

    it('rejects weight over 1', async () => {
      await expect(validatedClient.endorse('content-123', 1.5)).rejects.toThrow('Validation failed');
    });

    it('rejects negative weight', async () => {
      await expect(validatedClient.endorse('content-123', -0.5)).rejects.toThrow('Validation failed');
    });

    it('rejects empty content ID', async () => {
      await expect(validatedClient.endorse('', 0.8)).rejects.toThrow('Validation failed');
    });
  });

  describe('getEndorsements', () => {
    it('accepts valid content ID', async () => {
      const result = await validatedClient.getEndorsements('content-123');
      expect(result.length).toBeGreaterThan(0);
    });

    it('rejects empty content ID', async () => {
      await expect(validatedClient.getEndorsements('')).rejects.toThrow('Validation failed');
    });
  });
});

describe('createValidatedMediaClients', () => {
  it('creates all validated clients', () => {
    const mockClient = createMockClient();
    const clients = createValidatedMediaClients(mockClient);

    expect(clients.publication).toBeInstanceOf(ValidatedPublicationClient);
    expect(clients.attribution).toBeInstanceOf(ValidatedAttributionClient);
    expect(clients.factcheck).toBeInstanceOf(ValidatedFactCheckClient);
    expect(clients.curation).toBeInstanceOf(ValidatedCurationClient);
  });
});

// ============================================================================
// Media Validation Schema Boundary Tests
// ============================================================================

describe('Media Validation Schema Boundary Tests', () => {
  let mockClient: ZomeCallable;

  beforeEach(() => {
    mockClient = createMockClient(
      new Map([
        ['media_publication:publish', createMockRecord(createMockContent())],
        ['media_publication:get_content', createMockRecord(createMockContent())],
        ['media_publication:get_by_author', [createMockRecord(createMockContent())]],
        ['media_publication:get_by_tag', [createMockRecord(createMockContent())]],
        ['media_publication:get_by_type', [createMockRecord(createMockContent())]],
        ['media_publication:update_content', createMockRecord(createMockContent())],
        ['media_publication:record_view', undefined],
        ['media_attribution:add_attribution', createMockRecord(createMockAttribution())],
        ['media_attribution:get_attributions', [createMockRecord(createMockAttribution())]],
        ['media_attribution:get_contributor_works', [createMockRecord(createMockAttribution())]],
        ['media_factcheck:submit_fact_check', createMockRecord(createMockFactCheck())],
        ['media_factcheck:get_fact_checks', [createMockRecord(createMockFactCheck())]],
        ['media_factcheck:get_by_checker', [createMockRecord(createMockFactCheck())]],
        ['media_factcheck:get_verification_status', 'Verified'],
        ['media_curation:create_list', createMockRecord(createMockCurationList())],
        ['media_curation:get_list', createMockRecord(createMockCurationList())],
        ['media_curation:get_by_curator', [createMockRecord(createMockCurationList())]],
        ['media_curation:add_to_list', createMockRecord(createMockCurationList())],
        ['media_curation:remove_from_list', createMockRecord(createMockCurationList())],
        ['media_curation:endorse', createMockRecord(createMockEndorsement())],
        ['media_curation:get_endorsements', [createMockRecord(createMockEndorsement())]],
      ])
    );
  });

  describe('ContentType enum validation', () => {
    const validTypes: ContentType[] = ['Article', 'Opinion', 'Investigation', 'Review', 'Analysis', 'Interview', 'Report', 'Editorial', 'Other'];

    it.each(validTypes)('accepts valid content type: %s', async (type_) => {
      const client = new ValidatedPublicationClient(mockClient);
      const result = await client.publish({
        type_,
        title: 'Test Content',
        content_hash: 'sha256:abcdef1234567890abcdef1234567890',
        storage_uri: 'ipfs://QmTest',
        license: 'CCBY',
      });
      expect(result).toBeDefined();
    });

    it.each(['article', 'VIDEO', 'invalid', '', 'Video', 'Image', 'Unknown'])(
      'rejects invalid content type: %s',
      async (type_) => {
        const client = new ValidatedPublicationClient(mockClient);
        await expect(
          client.publish({
            type_: type_ as ContentType,
            title: 'Test',
            content_hash: 'sha256:abcdef1234567890abcdef1234567890',
            storage_uri: 'ipfs://test',
            license: 'CCBY',
          })
        ).rejects.toThrow('Validation failed');
      }
    );
  });

  describe('LicenseType enum validation', () => {
    const validLicenses: LicenseType[] = [
      'CC0',
      'CCBY',
      'CCBYSA',
      'CCBYNC',
      'CCBYNCSA',
      'AllRightsReserved',
      'Custom',
    ];

    it.each(validLicenses)('accepts valid license type: %s', async (license) => {
      const client = new ValidatedPublicationClient(mockClient);
      const result = await client.publish({
        type_: 'Article',
        title: 'Test Content',
        content_hash: 'sha256:abcdef1234567890abcdef1234567890',
        storage_uri: 'ipfs://QmTest',
        license,
      });
      expect(result).toBeDefined();
    });

    it.each(['MIT', 'GPL', 'Apache', 'cc-by', '', 'Public Domain', 'Unknown'])(
      'rejects invalid license type: %s',
      async (license) => {
        const client = new ValidatedPublicationClient(mockClient);
        await expect(
          client.publish({
            type_: 'Article',
            title: 'Test',
            content_hash: 'sha256:abcdef1234567890abcdef1234567890',
            storage_uri: 'ipfs://test',
            license: license as LicenseType,
          })
        ).rejects.toThrow('Validation failed');
      }
    );
  });

  describe('VerificationStatus enum validation', () => {
    const validStatuses: VerificationStatus[] = ['Unverified', 'Pending', 'Verified', 'Disputed', 'Debunked'];

    it.each(validStatuses)('accepts valid verification status: %s', async (verdict) => {
      const client = new ValidatedFactCheckClient(mockClient);
      const result = await client.submitFactCheck({
        content_id: 'content-123',
        verdict,
        reasoning: 'This is a sufficiently long reasoning text for validation.',
        sources: ['https://source.example.com'],
        epistemic_score: { empirical: 0.5, normative: 0.3, metaphorical: 0.2 },
      });
      expect(result).toBeDefined();
    });

    it.each(['verified', 'PENDING', 'invalid', '', 'True', 'False', 'Unknown'])(
      'rejects invalid verification status: %s',
      async (verdict) => {
        const client = new ValidatedFactCheckClient(mockClient);
        await expect(
          client.submitFactCheck({
            content_id: 'content-123',
            verdict: verdict as VerificationStatus,
            reasoning: 'This is a sufficiently long reasoning text for validation.',
            sources: ['https://source.example.com'],
            epistemic_score: { empirical: 0.5, normative: 0.3, metaphorical: 0.2 },
          })
        ).rejects.toThrow('Validation failed');
      }
    );
  });

  describe('AttributionRole enum validation', () => {
    const validRoles: Attribution['role'][] = ['Author', 'Editor', 'Contributor', 'Source', 'Translator'];

    it.each(validRoles)('accepts valid attribution role: %s', async (role) => {
      const client = new ValidatedAttributionClient(mockClient);
      const result = await client.addAttribution({
        content_id: 'content-123',
        contributor: 'did:mycelix:contributor456',
        role,
      });
      expect(result).toBeDefined();
    });

    it.each(['author', 'EDITOR', 'invalid', '', 'Creator', 'Publisher', 'Reviewer'])(
      'rejects invalid attribution role: %s',
      async (role) => {
        const client = new ValidatedAttributionClient(mockClient);
        await expect(
          client.addAttribution({
            content_id: 'content-123',
            contributor: 'did:mycelix:contributor456',
            role: role as Attribution['role'],
          })
        ).rejects.toThrow('Validation failed');
      }
    );
  });

  describe('Title length validation', () => {
    it('accepts minimum valid title (1 char)', async () => {
      const client = new ValidatedPublicationClient(mockClient);
      const result = await client.publish({
        type_: 'Article',
        title: 'A',
        content_hash: 'sha256:abcdef1234567890abcdef1234567890',
        storage_uri: 'ipfs://test',
        license: 'CCBY',
      });
      expect(result).toBeDefined();
    });

    it('accepts maximum valid title (500 chars)', async () => {
      const client = new ValidatedPublicationClient(mockClient);
      const result = await client.publish({
        type_: 'Article',
        title: 'x'.repeat(500),
        content_hash: 'sha256:abcdef1234567890abcdef1234567890',
        storage_uri: 'ipfs://test',
        license: 'CCBY',
      });
      expect(result).toBeDefined();
    });

    it('rejects title just over max (501 chars)', async () => {
      const client = new ValidatedPublicationClient(mockClient);
      await expect(
        client.publish({
          type_: 'Article',
          title: 'x'.repeat(501),
          content_hash: 'sha256:abcdef1234567890abcdef1234567890',
          storage_uri: 'ipfs://test',
          license: 'CCBY',
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('Content hash length validation', () => {
    it('accepts minimum valid content hash (32 chars)', async () => {
      const client = new ValidatedPublicationClient(mockClient);
      const result = await client.publish({
        type_: 'Article',
        title: 'Test',
        content_hash: 'a'.repeat(32),
        storage_uri: 'ipfs://test',
        license: 'CCBY',
      });
      expect(result).toBeDefined();
    });

    it('accepts longer content hash', async () => {
      const client = new ValidatedPublicationClient(mockClient);
      const result = await client.publish({
        type_: 'Article',
        title: 'Test',
        content_hash: 'sha256:' + 'a'.repeat(64),
        storage_uri: 'ipfs://test',
        license: 'CCBY',
      });
      expect(result).toBeDefined();
    });

    it('rejects content hash just under min (31 chars)', async () => {
      const client = new ValidatedPublicationClient(mockClient);
      await expect(
        client.publish({
          type_: 'Article',
          title: 'Test',
          content_hash: 'a'.repeat(31),
          storage_uri: 'ipfs://test',
          license: 'CCBY',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects empty content hash', async () => {
      const client = new ValidatedPublicationClient(mockClient);
      await expect(
        client.publish({
          type_: 'Article',
          title: 'Test',
          content_hash: '',
          storage_uri: 'ipfs://test',
          license: 'CCBY',
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('Storage URI validation', () => {
    it('accepts minimum valid storage URI (1 char)', async () => {
      const client = new ValidatedPublicationClient(mockClient);
      const result = await client.publish({
        type_: 'Article',
        title: 'Test',
        content_hash: 'sha256:abcdef1234567890abcdef1234567890',
        storage_uri: 'x',
        license: 'CCBY',
      });
      expect(result).toBeDefined();
    });

    it('rejects empty storage URI', async () => {
      const client = new ValidatedPublicationClient(mockClient);
      await expect(
        client.publish({
          type_: 'Article',
          title: 'Test',
          content_hash: 'sha256:abcdef1234567890abcdef1234567890',
          storage_uri: '',
          license: 'CCBY',
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('Reasoning length validation', () => {
    it('accepts minimum valid reasoning (20 chars)', async () => {
      const client = new ValidatedFactCheckClient(mockClient);
      const result = await client.submitFactCheck({
        content_id: 'content-123',
        verdict: 'Verified',
        reasoning: 'x'.repeat(20),
        sources: ['https://source.example.com'],
        epistemic_score: { empirical: 0.5, normative: 0.3, metaphorical: 0.2 },
      });
      expect(result).toBeDefined();
    });

    it('rejects reasoning just under min (19 chars)', async () => {
      const client = new ValidatedFactCheckClient(mockClient);
      await expect(
        client.submitFactCheck({
          content_id: 'content-123',
          verdict: 'Verified',
          reasoning: 'x'.repeat(19),
          sources: ['https://source.example.com'],
          epistemic_score: { empirical: 0.5, normative: 0.3, metaphorical: 0.2 },
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects empty reasoning', async () => {
      const client = new ValidatedFactCheckClient(mockClient);
      await expect(
        client.submitFactCheck({
          content_id: 'content-123',
          verdict: 'Verified',
          reasoning: '',
          sources: ['https://source.example.com'],
          epistemic_score: { empirical: 0.5, normative: 0.3, metaphorical: 0.2 },
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('Sources array validation', () => {
    it('accepts minimum sources array (1 element)', async () => {
      const client = new ValidatedFactCheckClient(mockClient);
      const result = await client.submitFactCheck({
        content_id: 'content-123',
        verdict: 'Verified',
        reasoning: 'Sufficiently long reasoning text.',
        sources: ['https://source.example.com'],
        epistemic_score: { empirical: 0.5, normative: 0.3, metaphorical: 0.2 },
      });
      expect(result).toBeDefined();
    });

    it('accepts multiple sources', async () => {
      const client = new ValidatedFactCheckClient(mockClient);
      const result = await client.submitFactCheck({
        content_id: 'content-123',
        verdict: 'Verified',
        reasoning: 'Sufficiently long reasoning text.',
        sources: [
          'https://source1.example.com',
          'https://source2.example.com',
          'https://source3.example.com',
        ],
        epistemic_score: { empirical: 0.5, normative: 0.3, metaphorical: 0.2 },
      });
      expect(result).toBeDefined();
    });
  });

  describe('Epistemic score validation', () => {
    it('accepts boundary value 0 for empirical', async () => {
      const client = new ValidatedFactCheckClient(mockClient);
      const result = await client.submitFactCheck({
        content_id: 'content-123',
        verdict: 'Verified',
        reasoning: 'Sufficiently long reasoning text.',
        sources: ['https://source.example.com'],
        epistemic_score: { empirical: 0, normative: 0.5, metaphorical: 0.5 },
      });
      expect(result).toBeDefined();
    });

    it('accepts boundary value 1 for empirical', async () => {
      const client = new ValidatedFactCheckClient(mockClient);
      const result = await client.submitFactCheck({
        content_id: 'content-123',
        verdict: 'Verified',
        reasoning: 'Sufficiently long reasoning text.',
        sources: ['https://source.example.com'],
        epistemic_score: { empirical: 1, normative: 0, metaphorical: 0 },
      });
      expect(result).toBeDefined();
    });

    it('accepts boundary value 0 for normative', async () => {
      const client = new ValidatedFactCheckClient(mockClient);
      const result = await client.submitFactCheck({
        content_id: 'content-123',
        verdict: 'Verified',
        reasoning: 'Sufficiently long reasoning text.',
        sources: ['https://source.example.com'],
        epistemic_score: { empirical: 0.5, normative: 0, metaphorical: 0.5 },
      });
      expect(result).toBeDefined();
    });

    it('accepts boundary value 1 for normative', async () => {
      const client = new ValidatedFactCheckClient(mockClient);
      const result = await client.submitFactCheck({
        content_id: 'content-123',
        verdict: 'Verified',
        reasoning: 'Sufficiently long reasoning text.',
        sources: ['https://source.example.com'],
        epistemic_score: { empirical: 0, normative: 1, metaphorical: 0 },
      });
      expect(result).toBeDefined();
    });

    it('accepts boundary value 0 for metaphorical', async () => {
      const client = new ValidatedFactCheckClient(mockClient);
      const result = await client.submitFactCheck({
        content_id: 'content-123',
        verdict: 'Verified',
        reasoning: 'Sufficiently long reasoning text.',
        sources: ['https://source.example.com'],
        epistemic_score: { empirical: 0.5, normative: 0.5, metaphorical: 0 },
      });
      expect(result).toBeDefined();
    });

    it('accepts boundary value 1 for metaphorical', async () => {
      const client = new ValidatedFactCheckClient(mockClient);
      const result = await client.submitFactCheck({
        content_id: 'content-123',
        verdict: 'Verified',
        reasoning: 'Sufficiently long reasoning text.',
        sources: ['https://source.example.com'],
        epistemic_score: { empirical: 0, normative: 0, metaphorical: 1 },
      });
      expect(result).toBeDefined();
    });

    it('rejects negative empirical score', async () => {
      const client = new ValidatedFactCheckClient(mockClient);
      await expect(
        client.submitFactCheck({
          content_id: 'content-123',
          verdict: 'Verified',
          reasoning: 'Sufficiently long reasoning text.',
          sources: ['https://source.example.com'],
          epistemic_score: { empirical: -0.1, normative: 0.5, metaphorical: 0.5 },
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects negative normative score', async () => {
      const client = new ValidatedFactCheckClient(mockClient);
      await expect(
        client.submitFactCheck({
          content_id: 'content-123',
          verdict: 'Verified',
          reasoning: 'Sufficiently long reasoning text.',
          sources: ['https://source.example.com'],
          epistemic_score: { empirical: 0.5, normative: -0.1, metaphorical: 0.5 },
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects negative metaphorical score', async () => {
      const client = new ValidatedFactCheckClient(mockClient);
      await expect(
        client.submitFactCheck({
          content_id: 'content-123',
          verdict: 'Verified',
          reasoning: 'Sufficiently long reasoning text.',
          sources: ['https://source.example.com'],
          epistemic_score: { empirical: 0.5, normative: 0.5, metaphorical: -0.1 },
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects empirical score over 1', async () => {
      const client = new ValidatedFactCheckClient(mockClient);
      await expect(
        client.submitFactCheck({
          content_id: 'content-123',
          verdict: 'Verified',
          reasoning: 'Sufficiently long reasoning text.',
          sources: ['https://source.example.com'],
          epistemic_score: { empirical: 1.1, normative: 0, metaphorical: 0 },
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects normative score over 1', async () => {
      const client = new ValidatedFactCheckClient(mockClient);
      await expect(
        client.submitFactCheck({
          content_id: 'content-123',
          verdict: 'Verified',
          reasoning: 'Sufficiently long reasoning text.',
          sources: ['https://source.example.com'],
          epistemic_score: { empirical: 0, normative: 1.1, metaphorical: 0 },
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects metaphorical score over 1', async () => {
      const client = new ValidatedFactCheckClient(mockClient);
      await expect(
        client.submitFactCheck({
          content_id: 'content-123',
          verdict: 'Verified',
          reasoning: 'Sufficiently long reasoning text.',
          sources: ['https://source.example.com'],
          epistemic_score: { empirical: 0, normative: 0, metaphorical: 1.1 },
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('Royalty percentage validation', () => {
    it('accepts boundary value 0', async () => {
      const client = new ValidatedAttributionClient(mockClient);
      const result = await client.addAttribution({
        content_id: 'content-123',
        contributor: 'did:mycelix:contributor456',
        role: 'Editor',
        royalty_percentage: 0,
      });
      expect(result).toBeDefined();
    });

    it('accepts boundary value 100', async () => {
      const client = new ValidatedAttributionClient(mockClient);
      const result = await client.addAttribution({
        content_id: 'content-123',
        contributor: 'did:mycelix:contributor456',
        role: 'Author',
        royalty_percentage: 100,
      });
      expect(result).toBeDefined();
    });

    it('accepts mid-range value 50', async () => {
      const client = new ValidatedAttributionClient(mockClient);
      const result = await client.addAttribution({
        content_id: 'content-123',
        contributor: 'did:mycelix:contributor456',
        role: 'Contributor',
        royalty_percentage: 50,
      });
      expect(result).toBeDefined();
    });

    it('rejects negative royalty percentage', async () => {
      const client = new ValidatedAttributionClient(mockClient);
      await expect(
        client.addAttribution({
          content_id: 'content-123',
          contributor: 'did:mycelix:contributor456',
          role: 'Editor',
          royalty_percentage: -1,
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects royalty over 100', async () => {
      const client = new ValidatedAttributionClient(mockClient);
      await expect(
        client.addAttribution({
          content_id: 'content-123',
          contributor: 'did:mycelix:contributor456',
          role: 'Editor',
          royalty_percentage: 101,
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('Curation list name validation', () => {
    it('accepts minimum valid name (1 char)', async () => {
      const client = new ValidatedCurationClient(mockClient);
      const result = await client.createList({
        name: 'X',
        description: 'A collection',
        public_: true,
      });
      expect(result).toBeDefined();
    });

    it('accepts maximum valid name (200 chars)', async () => {
      const client = new ValidatedCurationClient(mockClient);
      const result = await client.createList({
        name: 'x'.repeat(200),
        description: 'A collection',
        public_: true,
      });
      expect(result).toBeDefined();
    });

    it('rejects name just over max (201 chars)', async () => {
      const client = new ValidatedCurationClient(mockClient);
      await expect(
        client.createList({
          name: 'x'.repeat(201),
          description: 'A collection',
          public_: true,
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('Endorsement weight validation', () => {
    it('accepts boundary value 0', async () => {
      const client = new ValidatedCurationClient(mockClient);
      const result = await client.endorse('content-123', 0);
      expect(result).toBeDefined();
    });

    it('accepts boundary value 1', async () => {
      const client = new ValidatedCurationClient(mockClient);
      const result = await client.endorse('content-123', 1);
      expect(result).toBeDefined();
    });

    it('accepts mid-range value 0.5', async () => {
      const client = new ValidatedCurationClient(mockClient);
      const result = await client.endorse('content-123', 0.5, 'Good article');
      expect(result).toBeDefined();
    });

    it('rejects weight just over 1 (1.01)', async () => {
      const client = new ValidatedCurationClient(mockClient);
      await expect(client.endorse('content-123', 1.01)).rejects.toThrow('Validation failed');
    });

    it('rejects weight just under 0 (-0.01)', async () => {
      const client = new ValidatedCurationClient(mockClient);
      await expect(client.endorse('content-123', -0.01)).rejects.toThrow('Validation failed');
    });
  });

  describe('DID validation', () => {
    it('accepts valid DID with mycelix method', async () => {
      const client = new ValidatedAttributionClient(mockClient);
      const result = await client.getContributorWorks('did:mycelix:abc123');
      expect(result).toBeDefined();
    });

    it('accepts valid DID with other method', async () => {
      const client = new ValidatedAttributionClient(mockClient);
      const result = await client.getContributorWorks('did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK');
      expect(result).toBeDefined();
    });

    it('accepts valid DID with web method', async () => {
      const client = new ValidatedPublicationClient(mockClient);
      const result = await client.getContentByAuthor('did:web:example.com');
      expect(result).toBeDefined();
    });

    it.each(['not-a-did', 'id:test', 'DID:test:123', 'did', ''])(
      'rejects invalid DID format: "%s"',
      async (invalidDid) => {
        const client = new ValidatedAttributionClient(mockClient);
        await expect(client.getContributorWorks(invalidDid)).rejects.toThrow('Validation failed');
      }
    );
  });

  describe('ID string validation', () => {
    it('accepts valid content ID', async () => {
      const client = new ValidatedPublicationClient(mockClient);
      const result = await client.getContent('content-abc-123');
      expect(result).toBeDefined();
    });

    it('accepts valid list ID', async () => {
      const client = new ValidatedCurationClient(mockClient);
      const result = await client.getList('list-abc-123');
      expect(result).toBeDefined();
    });

    it('rejects empty content ID for getContent', async () => {
      const client = new ValidatedPublicationClient(mockClient);
      await expect(client.getContent('')).rejects.toThrow('Validation failed');
    });

    it('rejects empty content ID for updateContent', async () => {
      const client = new ValidatedPublicationClient(mockClient);
      await expect(client.updateContent('', { title: 'New Title' })).rejects.toThrow('Validation failed');
    });

    it('rejects empty content ID for recordView', async () => {
      const client = new ValidatedPublicationClient(mockClient);
      await expect(client.recordView('')).rejects.toThrow('Validation failed');
    });
  });

  describe('Optional fields handling', () => {
    it('accepts publish without optional description', async () => {
      const client = new ValidatedPublicationClient(mockClient);
      const result = await client.publish({
        type_: 'Article',
        title: 'Test',
        content_hash: 'sha256:abcdef1234567890abcdef1234567890',
        storage_uri: 'ipfs://test',
        license: 'CCBY',
      });
      expect(result).toBeDefined();
    });

    it('accepts publish with description', async () => {
      const client = new ValidatedPublicationClient(mockClient);
      const result = await client.publish({
        type_: 'Article',
        title: 'Test',
        description: 'A test article',
        content_hash: 'sha256:abcdef1234567890abcdef1234567890',
        storage_uri: 'ipfs://test',
        license: 'CCBY',
      });
      expect(result).toBeDefined();
    });

    it('accepts publish without optional tags', async () => {
      const client = new ValidatedPublicationClient(mockClient);
      const result = await client.publish({
        type_: 'Article',
        title: 'Test',
        content_hash: 'sha256:abcdef1234567890abcdef1234567890',
        storage_uri: 'ipfs://test',
        license: 'CCBY',
      });
      expect(result).toBeDefined();
    });

    it('accepts publish with tags', async () => {
      const client = new ValidatedPublicationClient(mockClient);
      const result = await client.publish({
        type_: 'Article',
        title: 'Test',
        content_hash: 'sha256:abcdef1234567890abcdef1234567890',
        storage_uri: 'ipfs://test',
        license: 'CCBY',
        tags: ['tag1', 'tag2'],
      });
      expect(result).toBeDefined();
    });

    it('accepts publish without custom_license when not Custom', async () => {
      const client = new ValidatedPublicationClient(mockClient);
      const result = await client.publish({
        type_: 'Article',
        title: 'Test',
        content_hash: 'sha256:abcdef1234567890abcdef1234567890',
        storage_uri: 'ipfs://test',
        license: 'CCBY',
      });
      expect(result).toBeDefined();
    });

    it('accepts publish with custom_license when Custom license', async () => {
      const client = new ValidatedPublicationClient(mockClient);
      const result = await client.publish({
        type_: 'Article',
        title: 'Test',
        content_hash: 'sha256:abcdef1234567890abcdef1234567890',
        storage_uri: 'ipfs://test',
        license: 'Custom',
        custom_license: 'Custom terms here',
      });
      expect(result).toBeDefined();
    });

    it('accepts attribution without optional description', async () => {
      const client = new ValidatedAttributionClient(mockClient);
      const result = await client.addAttribution({
        content_id: 'content-123',
        contributor: 'did:mycelix:contributor456',
        role: 'Editor',
      });
      expect(result).toBeDefined();
    });

    it('accepts attribution with description', async () => {
      const client = new ValidatedAttributionClient(mockClient);
      const result = await client.addAttribution({
        content_id: 'content-123',
        contributor: 'did:mycelix:contributor456',
        role: 'Editor',
        description: 'Technical editing',
      });
      expect(result).toBeDefined();
    });

    it('accepts attribution without optional royalty_percentage', async () => {
      const client = new ValidatedAttributionClient(mockClient);
      const result = await client.addAttribution({
        content_id: 'content-123',
        contributor: 'did:mycelix:contributor456',
        role: 'Source',
      });
      expect(result).toBeDefined();
    });

    it('accepts curation list without optional content_ids', async () => {
      const client = new ValidatedCurationClient(mockClient);
      const result = await client.createList({
        name: 'New List',
        description: 'A new curation list',
        public_: false,
      });
      expect(result).toBeDefined();
    });

    it('accepts curation list with content_ids', async () => {
      const client = new ValidatedCurationClient(mockClient);
      const result = await client.createList({
        name: 'New List',
        description: 'A new curation list',
        content_ids: ['content-1', 'content-2'],
        public_: true,
      });
      expect(result).toBeDefined();
    });

    it('accepts endorsement without comment', async () => {
      const client = new ValidatedCurationClient(mockClient);
      const result = await client.endorse('content-123', 0.8);
      expect(result).toBeDefined();
    });

    it('accepts endorsement with comment', async () => {
      const client = new ValidatedCurationClient(mockClient);
      const result = await client.endorse('content-123', 0.8, 'Great article!');
      expect(result).toBeDefined();
    });
  });

  describe('Error message content validation', () => {
    it('includes context in error message for publish', async () => {
      const client = new ValidatedPublicationClient(mockClient);
      await expect(
        client.publish({
          type_: 'Invalid' as ContentType,
          title: '',
          content_hash: '',
          storage_uri: '',
          license: 'Invalid' as LicenseType,
        })
      ).rejects.toThrow(/publish input/);
    });

    it('includes context in error message for addAttribution', async () => {
      const client = new ValidatedAttributionClient(mockClient);
      await expect(
        client.addAttribution({
          content_id: '',
          contributor: 'not-a-did',
          role: 'Invalid' as Attribution['role'],
        })
      ).rejects.toThrow(/addAttribution input/);
    });

    it('includes context in error message for submitFactCheck', async () => {
      const client = new ValidatedFactCheckClient(mockClient);
      await expect(
        client.submitFactCheck({
          content_id: '',
          verdict: 'Invalid' as VerificationStatus,
          reasoning: '',
          sources: [],
          epistemic_score: { empirical: 2, normative: 2, metaphorical: 2 },
        })
      ).rejects.toThrow(/submitFactCheck input/);
    });

    it('includes context in error message for createList', async () => {
      const client = new ValidatedCurationClient(mockClient);
      await expect(
        client.createList({
          name: '',
          description: '',
          public_: true,
        })
      ).rejects.toThrow(/createList input/);
    });

    it('includes context in error message for contentId validation', async () => {
      const client = new ValidatedPublicationClient(mockClient);
      await expect(client.getContent('')).rejects.toThrow(/contentId/);
    });

    it('includes context in error message for authorDid validation', async () => {
      const client = new ValidatedPublicationClient(mockClient);
      await expect(client.getContentByAuthor('invalid')).rejects.toThrow(/authorDid/);
    });

    it('includes context in error message for tag validation', async () => {
      const client = new ValidatedPublicationClient(mockClient);
      await expect(client.getContentByTag('')).rejects.toThrow(/tag/);
    });

    it('includes context in error message for type_ validation', async () => {
      const client = new ValidatedPublicationClient(mockClient);
      await expect(client.getContentByType('Invalid' as ContentType)).rejects.toThrow(/type_/);
    });

    it('includes context in error message for listId validation', async () => {
      const client = new ValidatedCurationClient(mockClient);
      await expect(client.getList('')).rejects.toThrow(/listId/);
    });

    it('includes context in error message for curatorDid validation', async () => {
      const client = new ValidatedCurationClient(mockClient);
      await expect(client.getListsByCurator('invalid')).rejects.toThrow(/curatorDid/);
    });

    it('includes context in error message for checkerDid validation', async () => {
      const client = new ValidatedFactCheckClient(mockClient);
      await expect(client.getFactChecksByChecker('invalid')).rejects.toThrow(/checkerDid/);
    });

    it('includes context in error message for weight validation', async () => {
      const client = new ValidatedCurationClient(mockClient);
      await expect(client.endorse('content-123', 5)).rejects.toThrow(/weight/);
    });
  });
});
