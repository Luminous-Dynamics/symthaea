// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Media Integration Tests
 *
 * Tests for MediaService - content publishing, moderation, and royalties
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  MediaService,
  getMediaService,
  resetMediaService,
  type Content,
  type CreatorProfile,
  type ModerationReport,
  type LicenseGrant,
} from '../../src/integrations/media/index.js';

describe('Media Integration', () => {
  let service: MediaService;

  beforeEach(() => {
    resetMediaService();
    service = new MediaService();
  });

  describe('MediaService', () => {
    describe('publishContent', () => {
      it('should publish new content', () => {
        const content = service.publishContent(
          'did:mycelix:creator1',
          'Article',
          'Introduction to Mycelix',
          'Learn about decentralized systems...',
          'ipfs://Qm...',
          'storage://abc123',
          'CCBY',
          ['technology', 'decentralization']
        );

        expect(content).toBeDefined();
        expect(content.id).toMatch(/^content-/);
        expect(content.title).toBe('Introduction to Mycelix');
        expect(content.type).toBe('Article');
        expect(content.license).toBe('CCBY');
        expect(content.status).toBe('published');
        expect(content.tags).toContain('technology');
      });

      it('should create creator as primary contributor', () => {
        const content = service.publishContent(
          'did:mycelix:creator',
          'Report',
          'Tutorial',
          'How to build...',
          'ipfs://video',
          'storage://video'
        );

        expect(content.contributors.length).toBe(1);
        expect(content.contributors[0].did).toBe('did:mycelix:creator');
        expect(content.contributors[0].role).toBe('creator');
        expect(content.contributors[0].royaltyShare).toBe(100);
      });

      it('should initialize with zero engagement', () => {
        const content = service.publishContent(
          'did:mycelix:creator',
          'Article',
          'Artwork',
          'Digital art',
          'ipfs://art',
          'storage://art'
        );

        expect(content.views).toBe(0);
        expect(content.likes).toBe(0);
        expect(content.shares).toBe(0);
      });

      it('should create and update creator profile', () => {
        service.publishContent('did:mycelix:newcreator', 'Article', 'Article 1', 'Content', 'h1', 's1');
        service.publishContent('did:mycelix:newcreator', 'Article', 'Article 2', 'Content', 'h2', 's2');

        const profile = service.getCreator('did:mycelix:newcreator');

        expect(profile).toBeDefined();
        expect(profile!.contentCount).toBe(2);
      });

      it('should support different content types', () => {
        const article = service.publishContent('did:mycelix:c', 'Article', 'Title', 'Desc', 'h', 's');
        const report = service.publishContent('did:mycelix:c', 'Report', 'Title', 'Desc', 'h', 's');
        const interview = service.publishContent('did:mycelix:c', 'Interview', 'Title', 'Desc', 'h', 's');
        const analysis = service.publishContent('did:mycelix:c', 'Analysis', 'Title', 'Desc', 'h', 's');

        expect(article.type).toBe('Article');
        expect(report.type).toBe('Report');
        expect(interview.type).toBe('Interview');
        expect(analysis.type).toBe('Analysis');
      });
    });

    describe('addContributor', () => {
      it('should add a contributor to content', () => {
        const content = service.publishContent(
          'did:mycelix:main',
          'Report',
          'Collaboration',
          'Joint project',
          'ipfs://collab',
          'storage://collab'
        );

        const updated = service.addContributor(
          content.id,
          'did:mycelix:coauthor',
          'editor',
          20
        );

        expect(updated.contributors.length).toBe(2);
        expect(updated.contributors[1].did).toBe('did:mycelix:coauthor');
        expect(updated.contributors[1].role).toBe('editor');
        expect(updated.contributors[1].royaltyShare).toBe(20);
        expect(updated.contributors[0].royaltyShare).toBe(80); // Reduced from 100
      });

      it('should throw if royalty shares exceed 100%', () => {
        const content = service.publishContent(
          'did:mycelix:main',
          'Report',
          'Title',
          'Desc',
          'h',
          's'
        );

        expect(() => {
          service.addContributor(content.id, 'did:mycelix:c1', 'role', 110);
        }).toThrow('Royalty shares exceed 100%');
      });

      it('should throw for non-existent content', () => {
        expect(() => {
          service.addContributor('content-fake', 'did:mycelix:c', 'role', 10);
        }).toThrow('Content not found');
      });
    });

    describe('recordView', () => {
      it('should increment view count', () => {
        const content = service.publishContent('did:mycelix:c', 'Article', 'Title', 'Desc', 'h', 's');

        service.recordView(content.id, 'did:mycelix:viewer1');
        service.recordView(content.id, 'did:mycelix:viewer2');
        service.recordView(content.id, 'did:mycelix:viewer3');

        const updated = service.getContent(content.id);
        expect(updated!.views).toBe(3);
      });

      it('should update creator total views', () => {
        service.publishContent('did:mycelix:creator', 'Article', 'Title', 'Desc', 'h', 's');
        const content = service.publishContent('did:mycelix:creator', 'Article', 'Title 2', 'Desc', 'h', 's');

        service.recordView(content.id, 'did:mycelix:viewer');

        const profile = service.getCreator('did:mycelix:creator');
        expect(profile!.totalViews).toBe(1);
      });

      it('should throw for non-existent content', () => {
        expect(() => {
          service.recordView('content-fake', 'did:mycelix:viewer');
        }).toThrow('Content not found');
      });
    });

    describe('likeContent', () => {
      it('should increment like count', () => {
        const content = service.publishContent('did:mycelix:c', 'Article', 'Art', 'Digital', 'h', 's');

        service.likeContent(content.id, 'did:mycelix:liker1');
        service.likeContent(content.id, 'did:mycelix:liker2');

        const updated = service.getContent(content.id);
        expect(updated!.likes).toBe(2);
      });

      it('should boost creator reputation', () => {
        const content = service.publishContent('did:mycelix:creator', 'Article', 'Title', 'Desc', 'h', 's');
        const profileBefore = service.getCreator('did:mycelix:creator')!;
        const repBefore = profileBefore.reputation.positiveCount;

        service.likeContent(content.id, 'did:mycelix:liker');

        const profileAfter = service.getCreator('did:mycelix:creator')!;
        expect(profileAfter.reputation.positiveCount).toBeGreaterThan(repBefore);
      });
    });

    describe('reportContent', () => {
      it('should create a moderation report', () => {
        const content = service.publishContent('did:mycelix:c', 'Article', 'Title', 'Desc', 'h', 's');

        const report = service.reportContent(
          content.id,
          'did:mycelix:reporter',
          'Contains misinformation about health',
          'misinformation'
        );

        expect(report).toBeDefined();
        expect(report.id).toMatch(/^report-/);
        expect(report.contentId).toBe(content.id);
        expect(report.category).toBe('misinformation');
        expect(report.status).toBe('pending');
      });

      it('should auto-flag content after 3 reports', () => {
        const content = service.publishContent('did:mycelix:c', 'Article', 'Title', 'Desc', 'h', 's');

        service.reportContent(content.id, 'did:mycelix:r1', 'Spam', 'spam');
        service.reportContent(content.id, 'did:mycelix:r2', 'Spam', 'spam');
        service.reportContent(content.id, 'did:mycelix:r3', 'Spam', 'spam');

        const updated = service.getContent(content.id);
        expect(updated!.status).toBe('flagged');
      });

      it('should throw for non-existent content', () => {
        expect(() => {
          service.reportContent('content-fake', 'did:mycelix:r', 'Reason', 'spam');
        }).toThrow('Content not found');
      });
    });

    describe('moderateContent', () => {
      it('should approve flagged content', () => {
        const content = service.publishContent('did:mycelix:c', 'Article', 'Title', 'Desc', 'h', 's');
        const report = service.reportContent(content.id, 'did:mycelix:r', 'False positive', 'spam');

        const resolved = service.moderateContent(report.id, 'did:mycelix:mod', 'approve');

        expect(resolved.status).toBe('resolved');
        expect(resolved.action).toBe('approve');

        const updated = service.getContent(content.id);
        expect(updated!.status).toBe('published');
      });

      it('should hide harmful content and penalize creator', () => {
        const content = service.publishContent('did:mycelix:offender', 'Article', 'Bad', 'Harmful', 'h', 's');
        const report = service.reportContent(content.id, 'did:mycelix:r', 'Harmful content', 'harmful');

        service.moderateContent(report.id, 'did:mycelix:mod', 'hide');

        const updated = service.getContent(content.id);
        expect(updated!.status).toBe('hidden');

        const profile = service.getCreator('did:mycelix:offender');
        expect(profile!.reputation.negativeCount).toBeGreaterThan(1);
      });

      it('should throw for non-existent report', () => {
        expect(() => {
          service.moderateContent('report-fake', 'did:mycelix:mod', 'approve');
        }).toThrow('Report not found');
      });
    });

    describe('grantLicense', () => {
      it('should grant a license for content', () => {
        const content = service.publishContent('did:mycelix:c', 'Analysis', 'Data', 'Valuable', 'h', 's');

        const grant = service.grantLicense(
          content.id,
          'did:mycelix:licensee',
          'CCBYNC',
          'did:mycelix:c', // grantorId - must be content owner
          100, // $100 fee
          'Attribution required'
        );

        expect(grant).toBeDefined();
        expect(grant.id).toMatch(/^license-/);
        expect(grant.license).toBe('CCBYNC');
        expect(grant.fee).toBe(100);
        expect(grant.customTerms).toBe('Attribution required');
      });

      it('should distribute royalties when fee is paid', () => {
        // Creator publishes content
        const content = service.publishContent('did:mycelix:creator', 'Review', 'Tool', 'Utility', 'h', 's');

        // Co-creator publishes their own content first (creates their profile)
        service.publishContent('did:mycelix:co', 'Review', 'Other Tool', 'Helper', 'h2', 's2');

        // Add co-creator as contributor to original content
        service.addContributor(content.id, 'did:mycelix:co', 'developer', 30);

        // Grant license which distributes royalties (must pass owner as grantorId)
        service.grantLicense(content.id, 'did:mycelix:buyer', 'AllRightsReserved', 'did:mycelix:creator', 1000);

        const creator = service.getCreator('did:mycelix:creator');
        const coCreator = service.getCreator('did:mycelix:co');

        // Creator gets 70% of 1000 = 700
        expect(creator!.totalEarnings).toBe(700);
        // Co-creator gets 30% of 1000 = 300
        expect(coCreator!.totalEarnings).toBe(300);
      });

      it('should support optional expiration', () => {
        const content = service.publishContent('did:mycelix:c', 'Article', 'Title', 'Desc', 'h', 's');
        const expiresAt = Date.now() + 30 * 24 * 60 * 60 * 1000; // 30 days

        const grant = service.grantLicense(
          content.id,
          'did:mycelix:licensee',
          'CCBY',
          'did:mycelix:c', // grantorId - must be content owner
          0,
          undefined,
          expiresAt
        );

        expect(grant.expiresAt).toBe(expiresAt);
      });

      it('should throw for non-existent content', () => {
        expect(() => {
          service.grantLicense('content-fake', 'did:mycelix:l', 'CCBY', 'did:mycelix:owner');
        }).toThrow('Content not found');
      });

      it('should throw if grantor is not content owner', () => {
        const content = service.publishContent('did:mycelix:owner', 'Analysis', 'Data', 'Desc', 'h', 's');
        expect(() => {
          service.grantLicense(content.id, 'did:mycelix:licensee', 'CCBY', 'did:mycelix:notowner');
        }).toThrow('Only the content owner can grant licenses');
      });
    });

    describe('distributeRoyalties', () => {
      it('should distribute to all contributors proportionally', () => {
        // Main creator publishes content
        const content = service.publishContent('did:mycelix:main', 'Report', 'Film', 'Documentary', 'h', 's');

        // Create profiles for contributors by having them publish their own content
        service.publishContent('did:mycelix:editor', 'Report', 'Reel', 'Demo reel', 'h2', 's2');
        service.publishContent('did:mycelix:musician', 'Interview', 'Track', 'Music track', 'h3', 's3');

        // Add contributors to the main content
        service.addContributor(content.id, 'did:mycelix:editor', 'editor', 20);
        service.addContributor(content.id, 'did:mycelix:musician', 'music', 10);

        service.distributeRoyalties(content.id, 1000, 'subscription');

        const main = service.getCreator('did:mycelix:main');
        const editor = service.getCreator('did:mycelix:editor');
        const musician = service.getCreator('did:mycelix:musician');

        expect(main!.totalEarnings).toBe(700); // 70%
        expect(editor!.totalEarnings).toBe(200); // 20%
        expect(musician!.totalEarnings).toBe(100); // 10%
      });
    });

    describe('searchContent', () => {
      it('should find content by title', () => {
        service.publishContent('did:mycelix:c', 'Article', 'Blockchain Guide', 'Content', 'h', 's', 'CCBY', ['tech']);
        service.publishContent('did:mycelix:c', 'Article', 'AI Basics', 'Content', 'h', 's', 'CCBY', ['ai']);

        const results = service.searchContent('blockchain');

        expect(results.length).toBe(1);
        expect(results[0].title).toContain('Blockchain');
      });

      it('should find content by description', () => {
        service.publishContent('did:mycelix:c', 'Analysis', 'Climate Data', 'Temperature measurements over 50 years', 'h', 's');

        const results = service.searchContent('temperature');

        expect(results.length).toBe(1);
      });

      it('should filter by tags', () => {
        service.publishContent('did:mycelix:c', 'Article', 'Tech Article', 'Desc', 'h', 's', 'CCBY', ['tech', 'tutorial']);
        service.publishContent('did:mycelix:c', 'Article', 'Art Article', 'Desc', 'h', 's', 'CCBY', ['art']);

        const results = service.searchContent('Article', ['tech']);

        expect(results.length).toBe(1);
        expect(results[0].title).toBe('Tech Article');
      });

      it('should filter by type', () => {
        service.publishContent('did:mycelix:c', 'Report', 'Tutorial Video', 'Desc', 'h', 's');
        service.publishContent('did:mycelix:c', 'Article', 'Tutorial Article', 'Desc', 'h', 's');

        const results = service.searchContent('Tutorial', undefined, 'Report');

        expect(results.length).toBe(1);
        expect(results[0].type).toBe('Report');
      });

      it('should exclude non-published content', () => {
        const content = service.publishContent('did:mycelix:c', 'Article', 'Searchable', 'Desc', 'h', 's');
        service.reportContent(content.id, 'did:mycelix:r1', 'Report', 'spam');
        service.reportContent(content.id, 'did:mycelix:r2', 'Report', 'spam');
        service.reportContent(content.id, 'did:mycelix:r3', 'Report', 'spam'); // Now flagged

        const results = service.searchContent('Searchable');

        expect(results.length).toBe(0);
      });
    });

    describe('getContent', () => {
      it('should retrieve existing content', () => {
        const created = service.publishContent('did:mycelix:c', 'Article', 'Title', 'Desc', 'h', 's');
        const retrieved = service.getContent(created.id);

        expect(retrieved).toBeDefined();
        expect(retrieved!.id).toBe(created.id);
      });

      it('should return undefined for non-existent content', () => {
        const result = service.getContent('content-fake');
        expect(result).toBeUndefined();
      });
    });

    describe('getCreator', () => {
      it('should retrieve creator profile', () => {
        service.publishContent('did:mycelix:creator', 'Article', 'Title', 'Desc', 'h', 's');

        const profile = service.getCreator('did:mycelix:creator');

        expect(profile).toBeDefined();
        expect(profile!.did).toBe('did:mycelix:creator');
      });

      it('should return undefined for non-existent creator', () => {
        const result = service.getCreator('did:mycelix:unknown');
        expect(result).toBeUndefined();
      });
    });
  });

  describe('getMediaService', () => {
    it('should return singleton instance', () => {
      const service1 = getMediaService();
      const service2 = getMediaService();

      expect(service1).toBe(service2);
    });
  });
});
