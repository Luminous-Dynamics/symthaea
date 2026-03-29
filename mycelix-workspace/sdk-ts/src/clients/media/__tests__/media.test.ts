// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Media Client Tests
 *
 * Verifies zome call arguments and response pass-through for
 * PublicationClient, AttributionClient, FactCheckClient, and CurationClient.
 *
 * Media clients use ZomeCallable interface and return HolochainRecord<T> directly.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  PublicationClient,
  AttributionClient,
  FactCheckClient,
  CurationClient,
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

const CONTENT_ENTRY = {
  id: 'content-001',
  author: 'did:mycelix:journalist',
  type_: 'Investigation',
  title: 'Community Energy Audit 2026',
  description: 'Comprehensive audit of local energy projects',
  content_hash: 'sha256:abc123',
  storage_uri: 'ipfs://Qm123',
  license: 'CCBY',
  tags: ['energy', 'audit', 'community'],
  verification_status: 'Verified',
  views: 142,
  endorsements: 8,
  published_at: 1708200000,
  updated_at: 1708200000,
};

const ATTRIBUTION_ENTRY = {
  id: 'attr-001',
  content_id: 'content-001',
  contributor: 'did:mycelix:editor',
  role: 'Editor',
  description: 'Reviewed and fact-checked',
  royalty_percentage: 10,
  created_at: 1708200000,
};

const FACT_CHECK_ENTRY = {
  id: 'fc-001',
  content_id: 'content-001',
  checker: 'did:mycelix:checker',
  verdict: 'Verified',
  reasoning: 'All claims backed by primary sources',
  sources: ['https://example.com/report'],
  epistemic_score: { empirical: 0.9, normative: 0.2, metaphorical: 0.1 },
  created_at: 1708200000,
};

const CURATION_LIST_ENTRY = {
  id: 'list-001',
  curator: 'did:mycelix:curator',
  name: 'Best Energy Investigations',
  description: 'Top investigative pieces on energy',
  content_ids: ['content-001', 'content-002'],
  public_: true,
  created_at: 1708200000,
  updated_at: 1708200000,
};

// ============================================================================
// PUBLICATION CLIENT TESTS
// ============================================================================

describe('PublicationClient', () => {
  let mockCallable: ZomeCallable;
  let publication: PublicationClient;

  beforeEach(() => {
    mockCallable = createMockCallable();
    publication = new PublicationClient(mockCallable);
  });

  it('publish calls correct zome', async () => {
    const record = mockRecord(CONTENT_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = {
      type_: 'Investigation' as const,
      title: 'Community Energy Audit 2026',
      description: 'Comprehensive audit of local energy projects',
      content_hash: 'sha256:abc123',
      storage_uri: 'ipfs://Qm123',
      license: 'CCBY' as const,
      tags: ['energy', 'audit', 'community'],
    };
    const result = await publication.publish(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'civic',
      zome_name: 'media_publication',
      fn_name: 'publish',
      payload: input,
    });
    expect(result.entry.Present.title).toBe('Community Energy Audit 2026');
    expect(result.entry.Present.verification_status).toBe('Verified');
  });

  it('getContentByAuthor returns content list', async () => {
    const record = mockRecord(CONTENT_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([record]);

    const result = await publication.getContentByAuthor('did:mycelix:journalist');

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'civic',
      zome_name: 'media_publication',
      fn_name: 'get_by_author',
      payload: 'did:mycelix:journalist',
    });
    expect(result).toHaveLength(1);
  });
});

// ============================================================================
// ATTRIBUTION CLIENT TESTS
// ============================================================================

describe('AttributionClient', () => {
  let mockCallable: ZomeCallable;
  let attribution: AttributionClient;

  beforeEach(() => {
    mockCallable = createMockCallable();
    attribution = new AttributionClient(mockCallable);
  });

  it('addAttribution calls correct zome', async () => {
    const record = mockRecord(ATTRIBUTION_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = {
      content_id: 'content-001',
      contributor: 'did:mycelix:editor',
      role: 'Editor' as const,
      description: 'Reviewed and fact-checked',
      royalty_percentage: 10,
    };
    const result = await attribution.addAttribution(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'civic',
      zome_name: 'media_attribution',
      fn_name: 'add_attribution',
      payload: input,
    });
    expect(result.entry.Present.role).toBe('Editor');
  });

  it('getAttributions returns attribution list', async () => {
    const record = mockRecord(ATTRIBUTION_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([record]);

    const result = await attribution.getAttributions('content-001');

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'civic',
      zome_name: 'media_attribution',
      fn_name: 'get_attributions',
      payload: 'content-001',
    });
    expect(result).toHaveLength(1);
  });
});

// ============================================================================
// FACT CHECK CLIENT TESTS
// ============================================================================

describe('FactCheckClient', () => {
  let mockCallable: ZomeCallable;
  let factCheck: FactCheckClient;

  beforeEach(() => {
    mockCallable = createMockCallable();
    factCheck = new FactCheckClient(mockCallable);
  });

  it('submitFactCheck calls correct zome', async () => {
    const record = mockRecord(FACT_CHECK_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = {
      content_id: 'content-001',
      verdict: 'Verified' as const,
      reasoning: 'All claims backed by primary sources',
      sources: ['https://example.com/report'],
      epistemic_score: { empirical: 0.9, normative: 0.2, metaphorical: 0.1 },
    };
    const result = await factCheck.submitFactCheck(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'civic',
      zome_name: 'media_factcheck',
      fn_name: 'submit_fact_check',
      payload: input,
    });
    expect(result.entry.Present.verdict).toBe('Verified');
    expect(result.entry.Present.epistemic_score.empirical).toBe(0.9);
  });
});

// ============================================================================
// CURATION CLIENT TESTS
// ============================================================================

describe('CurationClient', () => {
  let mockCallable: ZomeCallable;
  let curation: CurationClient;

  beforeEach(() => {
    mockCallable = createMockCallable();
    curation = new CurationClient(mockCallable);
  });

  it('createList calls correct zome', async () => {
    const record = mockRecord(CURATION_LIST_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = {
      name: 'Best Energy Investigations',
      description: 'Top investigative pieces on energy',
      public_: true,
    };
    const result = await curation.createList(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'civic',
      zome_name: 'media_curation',
      fn_name: 'create_list',
      payload: input,
    });
    expect(result.entry.Present.name).toBe('Best Energy Investigations');
    expect(result.entry.Present.content_ids).toHaveLength(2);
  });
});
