// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Media hApp Conductor Integration Tests
 *
 * These tests verify the Media clients work correctly with a real
 * Holochain conductor. Provides content publication, attribution,
 * fact-checking, and curation functionality.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  PublicationClient,
  AttributionClient,
  FactCheckClient,
  CurationClient,
  createMediaClients,
  type ZomeCallable,
  type Content,
  type Attribution,
  type FactCheck,
  type CurationList,
  type Endorsement,
} from '../../src/media/index.js';
import { createValidatedMediaClients } from '../../src/media/validated.js';
import { MycelixError } from '../../src/errors.js';
import { CONDUCTOR_ENABLED, generateTestAgentId } from './conductor-harness.js';

function createMockClient(): ZomeCallable {
  const content = new Map<string, Content>();
  const attributions = new Map<string, Attribution[]>();
  const factChecks = new Map<string, FactCheck[]>();
  const curationLists = new Map<string, CurationList>();
  const endorsements = new Map<string, Endorsement[]>();
  let idCounter = 0;

  return {
    async callZome({
      fn_name,
      payload,
    }: {
      role_name: string;
      zome_name: string;
      fn_name: string;
      payload: unknown;
    }) {
      const agentId = `did:mycelix:${generateTestAgentId()}`;

      switch (fn_name) {
        case 'publish': {
          const input = payload as {
            type_: string;
            title: string;
            description?: string;
            content_hash: string;
            storage_uri: string;
            license: string;
            custom_license?: string;
            tags?: string[];
          };
          const c: Content = {
            id: `content-${Date.now()}-${++idCounter}`,
            author: agentId,
            type_: input.type_ as Content['type_'],
            title: input.title,
            description: input.description,
            content_hash: input.content_hash,
            storage_uri: input.storage_uri,
            license: input.license as Content['license'],
            custom_license: input.custom_license,
            tags: input.tags || [],
            verification_status: 'Unverified',
            views: 0,
            endorsements: 0,
            published_at: Date.now(),
            updated_at: Date.now(),
          };
          content.set(c.id, c);
          return {
            signed_action: { hashed: { hash: c.id, content: {} }, signature: 'sig' },
            entry: { Present: c },
          };
        }

        case 'get_content': {
          const id = payload as string;
          const c = content.get(id);
          return c
            ? {
                signed_action: { hashed: { hash: id, content: {} }, signature: 'sig' },
                entry: { Present: c },
              }
            : null;
        }

        case 'get_by_author': {
          const authorDid = payload as string;
          return Array.from(content.values())
            .filter((c) => c.author === authorDid)
            .map((c) => ({
              signed_action: { hashed: { hash: c.id, content: {} }, signature: 'sig' },
              entry: { Present: c },
            }));
        }

        case 'get_by_tag': {
          const tag = payload as string;
          return Array.from(content.values())
            .filter((c) => c.tags.includes(tag))
            .map((c) => ({
              signed_action: { hashed: { hash: c.id, content: {} }, signature: 'sig' },
              entry: { Present: c },
            }));
        }

        case 'get_by_type': {
          const type = payload as string;
          return Array.from(content.values())
            .filter((c) => c.type_ === type)
            .map((c) => ({
              signed_action: { hashed: { hash: c.id, content: {} }, signature: 'sig' },
              entry: { Present: c },
            }));
        }

        case 'update_content': {
          const { content_id, ...updates } = payload as {
            content_id: string;
            [key: string]: unknown;
          };
          const c = content.get(content_id);
          if (c) {
            Object.assign(c, updates);
            c.updated_at = Date.now();
          }
          return c
            ? {
                signed_action: { hashed: { hash: content_id, content: {} }, signature: 'sig' },
                entry: { Present: c },
              }
            : null;
        }

        case 'record_view': {
          const contentId = payload as string;
          const c = content.get(contentId);
          if (c) {
            c.views++;
          }
          return undefined;
        }

        case 'add_attribution': {
          const input = payload as {
            content_id: string;
            contributor: string;
            role: string;
            description?: string;
            royalty_percentage?: number;
          };
          const attr: Attribution = {
            id: `attr-${Date.now()}-${++idCounter}`,
            content_id: input.content_id,
            contributor: input.contributor,
            role: input.role as Attribution['role'],
            description: input.description,
            royalty_percentage: input.royalty_percentage,
            created_at: Date.now(),
          };
          const existing = attributions.get(input.content_id) || [];
          existing.push(attr);
          attributions.set(input.content_id, existing);
          return {
            signed_action: { hashed: { hash: attr.id, content: {} }, signature: 'sig' },
            entry: { Present: attr },
          };
        }

        case 'get_attributions': {
          const contentId = payload as string;
          const attrs = attributions.get(contentId) || [];
          return attrs.map((a) => ({
            signed_action: { hashed: { hash: a.id, content: {} }, signature: 'sig' },
            entry: { Present: a },
          }));
        }

        case 'get_contributor_works': {
          const contributorDid = payload as string;
          const works: Attribution[] = [];
          for (const attrs of attributions.values()) {
            for (const a of attrs) {
              if (a.contributor === contributorDid) {
                works.push(a);
              }
            }
          }
          return works.map((a) => ({
            signed_action: { hashed: { hash: a.id, content: {} }, signature: 'sig' },
            entry: { Present: a },
          }));
        }

        case 'submit_fact_check': {
          const input = payload as {
            content_id: string;
            verdict: string;
            reasoning: string;
            sources: string[];
            epistemic_score: { empirical: number; normative: number; metaphorical: number };
          };
          const fc: FactCheck = {
            id: `fc-${Date.now()}-${++idCounter}`,
            content_id: input.content_id,
            checker: agentId,
            verdict: input.verdict as FactCheck['verdict'],
            reasoning: input.reasoning,
            sources: input.sources,
            epistemic_score: input.epistemic_score,
            created_at: Date.now(),
          };
          const existing = factChecks.get(input.content_id) || [];
          existing.push(fc);
          factChecks.set(input.content_id, existing);
          // Update content verification status
          const c = content.get(input.content_id);
          if (c) {
            c.verification_status = input.verdict as Content['verification_status'];
          }
          return {
            signed_action: { hashed: { hash: fc.id, content: {} }, signature: 'sig' },
            entry: { Present: fc },
          };
        }

        case 'get_fact_checks': {
          const contentId = payload as string;
          const fcs = factChecks.get(contentId) || [];
          return fcs.map((f) => ({
            signed_action: { hashed: { hash: f.id, content: {} }, signature: 'sig' },
            entry: { Present: f },
          }));
        }

        case 'get_by_checker': {
          const checkerDid = payload as string;
          const checks: FactCheck[] = [];
          for (const fcs of factChecks.values()) {
            for (const fc of fcs) {
              if (fc.checker === checkerDid) {
                checks.push(fc);
              }
            }
          }
          return checks.map((f) => ({
            signed_action: { hashed: { hash: f.id, content: {} }, signature: 'sig' },
            entry: { Present: f },
          }));
        }

        case 'get_verification_status': {
          const contentId = payload as string;
          const c = content.get(contentId);
          return c?.verification_status || 'Unverified';
        }

        case 'create_list': {
          const input = payload as {
            name: string;
            description: string;
            content_ids?: string[];
            public_: boolean;
          };
          const list: CurationList = {
            id: `list-${Date.now()}-${++idCounter}`,
            curator: agentId,
            name: input.name,
            description: input.description,
            content_ids: input.content_ids || [],
            public_: input.public_,
            created_at: Date.now(),
            updated_at: Date.now(),
          };
          curationLists.set(list.id, list);
          return {
            signed_action: { hashed: { hash: list.id, content: {} }, signature: 'sig' },
            entry: { Present: list },
          };
        }

        case 'get_list': {
          const id = payload as string;
          const list = curationLists.get(id);
          return list
            ? {
                signed_action: { hashed: { hash: id, content: {} }, signature: 'sig' },
                entry: { Present: list },
              }
            : null;
        }

        case 'get_by_curator': {
          const curatorDid = payload as string;
          return Array.from(curationLists.values())
            .filter((l) => l.curator === curatorDid)
            .map((l) => ({
              signed_action: { hashed: { hash: l.id, content: {} }, signature: 'sig' },
              entry: { Present: l },
            }));
        }

        case 'add_to_list': {
          const { list_id, content_id } = payload as { list_id: string; content_id: string };
          const list = curationLists.get(list_id);
          if (list && !list.content_ids.includes(content_id)) {
            list.content_ids.push(content_id);
            list.updated_at = Date.now();
          }
          return list
            ? {
                signed_action: { hashed: { hash: list_id, content: {} }, signature: 'sig' },
                entry: { Present: list },
              }
            : null;
        }

        case 'remove_from_list': {
          const { list_id, content_id } = payload as { list_id: string; content_id: string };
          const list = curationLists.get(list_id);
          if (list) {
            list.content_ids = list.content_ids.filter((id) => id !== content_id);
            list.updated_at = Date.now();
          }
          return list
            ? {
                signed_action: { hashed: { hash: list_id, content: {} }, signature: 'sig' },
                entry: { Present: list },
              }
            : null;
        }

        case 'endorse': {
          const { content_id, weight, comment } = payload as {
            content_id: string;
            weight: number;
            comment?: string;
          };
          const endorsement: Endorsement = {
            id: `end-${Date.now()}-${++idCounter}`,
            content_id,
            endorser: agentId,
            weight,
            comment,
            created_at: Date.now(),
          };
          const existing = endorsements.get(content_id) || [];
          existing.push(endorsement);
          endorsements.set(content_id, existing);
          // Update content endorsement count
          const c = content.get(content_id);
          if (c) {
            c.endorsements++;
          }
          return {
            signed_action: { hashed: { hash: endorsement.id, content: {} }, signature: 'sig' },
            entry: { Present: endorsement },
          };
        }

        case 'get_endorsements': {
          const contentId = payload as string;
          const ends = endorsements.get(contentId) || [];
          return ends.map((e) => ({
            signed_action: { hashed: { hash: e.id, content: {} }, signature: 'sig' },
            entry: { Present: e },
          }));
        }

        default:
          throw new Error(`Unknown function: ${fn_name}`);
      }
    },
  };
}

const describeConductor = CONDUCTOR_ENABLED ? describe : describe.skip;

describe('Media Clients (Mock)', () => {
  let mockClient: ZomeCallable;
  let publicationClient: PublicationClient;
  let attributionClient: AttributionClient;
  let factCheckClient: FactCheckClient;
  let curationClient: CurationClient;

  beforeAll(() => {
    mockClient = createMockClient();
    const clients = createMediaClients(mockClient);
    publicationClient = clients.publication;
    attributionClient = clients.attribution;
    factCheckClient = clients.factcheck;
    curationClient = clients.curation;
  });

  describe('PublicationClient', () => {
    it('should publish content', async () => {
      const result = await publicationClient.publish({
        type_: 'Article',
        title: 'Introduction to Decentralized Media',
        description: 'A comprehensive guide to decentralized publishing',
        content_hash: 'QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG',
        storage_uri: 'ipfs://QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG',
        license: 'CC-BY',
        tags: ['decentralization', 'media', 'tutorial'],
      });

      expect(result).toBeDefined();
      const c = result.entry.Present as Content;
      expect(c.title).toBe('Introduction to Decentralized Media');
      expect(c.type_).toBe('Article');
      expect(c.license).toBe('CC-BY');
    });

    it('should get content by tag', async () => {
      await publicationClient.publish({
        type_: 'Video',
        title: 'Tech Tutorial',
        content_hash: 'hash1',
        storage_uri: 'ipfs://hash1',
        license: 'CC0',
        tags: ['technology', 'tutorial'],
      });

      const result = await publicationClient.getContentByTag('technology');

      expect(result.length).toBeGreaterThan(0);
    });

    it('should record views', async () => {
      const published = await publicationClient.publish({
        type_: 'Article',
        title: 'Popular Article',
        content_hash: 'hash2',
        storage_uri: 'ipfs://hash2',
        license: 'CC-BY-SA',
      });
      const contentId = (published.entry.Present as Content).id;

      await publicationClient.recordView(contentId);
      await publicationClient.recordView(contentId);

      const fetched = await publicationClient.getContent(contentId);
      expect((fetched!.entry.Present as Content).views).toBe(2);
    });
  });

  describe('AttributionClient', () => {
    it('should add attribution', async () => {
      const published = await publicationClient.publish({
        type_: 'Document',
        title: 'Collaborative Paper',
        content_hash: 'hash3',
        storage_uri: 'ipfs://hash3',
        license: 'CC-BY',
      });
      const contentId = (published.entry.Present as Content).id;

      const result = await attributionClient.addAttribution({
        content_id: contentId,
        contributor: 'did:mycelix:coauthor',
        role: 'Editor',
        description: 'Edited and reviewed the manuscript',
        royalty_percentage: 20,
      });

      expect(result).toBeDefined();
      const attr = result.entry.Present as Attribution;
      expect(attr.role).toBe('Editor');
      expect(attr.royalty_percentage).toBe(20);
    });

    it('should get attributions for content', async () => {
      const published = await publicationClient.publish({
        type_: 'Audio',
        title: 'Podcast Episode',
        content_hash: 'hash4',
        storage_uri: 'ipfs://hash4',
        license: 'CC-BY-NC',
      });
      const contentId = (published.entry.Present as Content).id;

      await attributionClient.addAttribution({
        content_id: contentId,
        contributor: 'did:mycelix:host',
        role: 'Author',
      });
      await attributionClient.addAttribution({
        content_id: contentId,
        contributor: 'did:mycelix:guest',
        role: 'Contributor',
      });

      const result = await attributionClient.getAttributions(contentId);

      expect(result).toHaveLength(2);
    });
  });

  describe('FactCheckClient', () => {
    it('should submit a fact check', async () => {
      const published = await publicationClient.publish({
        type_: 'Article',
        title: 'Climate Report',
        content_hash: 'hash5',
        storage_uri: 'ipfs://hash5',
        license: 'CC0',
        tags: ['climate', 'science'],
      });
      const contentId = (published.entry.Present as Content).id;

      const result = await factCheckClient.submitFactCheck({
        content_id: contentId,
        verdict: 'Verified',
        reasoning: 'Claims are supported by peer-reviewed research from IPCC and NASA.',
        sources: ['https://ipcc.ch/report', 'https://nasa.gov/climate'],
        epistemic_score: { empirical: 0.95, normative: 0.1, metaphorical: 0.05 },
      });

      expect(result).toBeDefined();
      const fc = result.entry.Present as FactCheck;
      expect(fc.verdict).toBe('Verified');
      expect(fc.epistemic_score.empirical).toBe(0.95);
    });

    it('should get verification status', async () => {
      const published = await publicationClient.publish({
        type_: 'Article',
        title: 'Unverified Claim',
        content_hash: 'hash6',
        storage_uri: 'ipfs://hash6',
        license: 'CC-BY',
      });
      const contentId = (published.entry.Present as Content).id;

      const initial = await factCheckClient.getContentVerificationStatus(contentId);
      expect(initial).toBe('Unverified');

      await factCheckClient.submitFactCheck({
        content_id: contentId,
        verdict: 'Disputed',
        reasoning: 'Multiple sources contradict the main claims.',
        sources: ['https://factcheck.org/example'],
        epistemic_score: { empirical: 0.3, normative: 0.5, metaphorical: 0.2 },
      });

      const updated = await factCheckClient.getContentVerificationStatus(contentId);
      expect(updated).toBe('Disputed');
    });
  });

  describe('CurationClient', () => {
    it('should create a curation list', async () => {
      const result = await curationClient.createList({
        name: 'Best of 2026',
        description: 'Top articles from 2026',
        public_: true,
      });

      expect(result).toBeDefined();
      const list = result.entry.Present as CurationList;
      expect(list.name).toBe('Best of 2026');
      expect(list.public_).toBe(true);
    });

    it('should manage list contents', async () => {
      const content1 = await publicationClient.publish({
        type_: 'Article',
        title: 'Article 1',
        content_hash: 'hash7',
        storage_uri: 'ipfs://hash7',
        license: 'CC0',
      });
      const content2 = await publicationClient.publish({
        type_: 'Article',
        title: 'Article 2',
        content_hash: 'hash8',
        storage_uri: 'ipfs://hash8',
        license: 'CC0',
      });

      const list = await curationClient.createList({
        name: 'Reading List',
        description: 'My personal reading list',
        public_: false,
      });
      const listId = (list.entry.Present as CurationList).id;

      await curationClient.addToList(listId, (content1.entry.Present as Content).id);
      await curationClient.addToList(listId, (content2.entry.Present as Content).id);

      let updated = await curationClient.getList(listId);
      expect((updated!.entry.Present as CurationList).content_ids).toHaveLength(2);

      await curationClient.removeFromList(listId, (content1.entry.Present as Content).id);

      updated = await curationClient.getList(listId);
      expect((updated!.entry.Present as CurationList).content_ids).toHaveLength(1);
    });

    it('should endorse content', async () => {
      const published = await publicationClient.publish({
        type_: 'Article',
        title: 'Excellent Article',
        content_hash: 'hash9',
        storage_uri: 'ipfs://hash9',
        license: 'CC-BY',
      });
      const contentId = (published.entry.Present as Content).id;

      const result = await curationClient.endorse(contentId, 0.9, 'Highly recommended!');

      expect(result).toBeDefined();
      const endorsement = result.entry.Present as Endorsement;
      expect(endorsement.weight).toBe(0.9);

      const content = await publicationClient.getContent(contentId);
      expect((content!.entry.Present as Content).endorsements).toBe(1);
    });
  });
});

describe('Validated Media Clients', () => {
  let mockClient: ZomeCallable;
  let validatedClients: ReturnType<typeof createValidatedMediaClients>;

  beforeAll(() => {
    mockClient = createMockClient();
    validatedClients = createValidatedMediaClients(mockClient);
  });

  it('should reject invalid publication', async () => {
    await expect(
      validatedClients.publication.publish({
        type_: 'Article',
        title: '', // Empty title
        content_hash: 'short', // Too short hash
        storage_uri: '',
        license: 'CC-BY',
      })
    ).rejects.toThrow(MycelixError);
  });

  it('should reject invalid attribution', async () => {
    await expect(
      validatedClients.attribution.addAttribution({
        content_id: '',
        contributor: 'not-a-did',
        role: 'Author',
        royalty_percentage: 150, // Over 100%
      })
    ).rejects.toThrow(MycelixError);
  });

  it('should reject invalid fact check', async () => {
    await expect(
      validatedClients.factcheck.submitFactCheck({
        content_id: '',
        verdict: 'Verified',
        reasoning: 'Too short', // Less than 20 chars
        sources: [], // Empty sources
        epistemic_score: { empirical: 2, normative: 0, metaphorical: 0 }, // Invalid score > 1
      })
    ).rejects.toThrow(MycelixError);
  });

  it('should reject invalid curation list', async () => {
    await expect(
      validatedClients.curation.createList({
        name: '', // Empty name
        description: '',
        public_: true,
      })
    ).rejects.toThrow(MycelixError);
  });
});

describeConductor('Media Conductor Integration Tests', () => {
  it.todo('should publish content with content addressing');
  it.todo('should verify attributions on-chain');
  it.todo('should aggregate fact-check consensus');
  it.todo('should weight endorsements by reputation');
});
