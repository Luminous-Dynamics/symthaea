// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Property Integration Tests
 *
 * Tests for PropertyService - assets, transfers, and commons management
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  PropertyService,
  getPropertyService,
  resetPropertyService,
  type Asset,
  type TransferProposal,
  type CommonsMembership,
} from '../../src/integrations/property/index.js';

describe('Property Integration', () => {
  let service: PropertyService;

  beforeEach(() => {
    resetPropertyService();
    service = new PropertyService();
  });

  describe('PropertyService', () => {
    describe('registerAsset', () => {
      it('should register a new asset', () => {
        const asset = service.registerAsset(
          'real_estate',
          'Community Garden',
          'Shared urban garden space',
          'did:mycelix:owner1'
        );

        expect(asset).toBeDefined();
        expect(asset.id).toMatch(/^asset-/);
        expect(asset.type).toBe('real_estate');
        expect(asset.name).toBe('Community Garden');
        expect(asset.ownershipModel).toBe('sole');
        expect(asset.owners.length).toBe(1);
        expect(asset.owners[0].percentage).toBe(100);
      });

      it('should support different ownership models', () => {
        const joint = service.registerAsset('equipment', 'Tractor', 'Farm equipment', 'did:mycelix:o1', 'joint');
        const fractional = service.registerAsset('digital', 'NFT Art', 'Digital art', 'did:mycelix:o2', 'fractional');
        const coop = service.registerAsset('commons', 'Water Well', 'Shared well', 'did:mycelix:o3', 'cooperative');

        expect(joint.ownershipModel).toBe('joint');
        expect(fractional.ownershipModel).toBe('fractional');
        expect(coop.ownershipModel).toBe('cooperative');
      });

      it('should set steward role for cooperative ownership', () => {
        const asset = service.registerAsset(
          'commons',
          'Forest',
          'Community forest',
          'did:mycelix:steward',
          'cooperative'
        );

        expect(asset.owners[0].role).toBe('steward');
      });

      it('should support optional metadata', () => {
        const asset = service.registerAsset(
          'real_estate',
          'House',
          'Residential property',
          'did:mycelix:owner',
          'sole',
          { address: '123 Main St', sqft: 2000, bedrooms: 3 }
        );

        expect(asset.metadata.address).toBe('123 Main St');
        expect(asset.metadata.sqft).toBe(2000);
      });
    });

    describe('proposeTransfer', () => {
      it('should create a transfer proposal', () => {
        const asset = service.registerAsset(
          'real_estate',
          'Property',
          'Description',
          'did:mycelix:seller'
        );

        const proposal = service.proposeTransfer(
          asset.id,
          'did:mycelix:seller',
          'did:mycelix:buyer',
          100,
          50000
        );

        expect(proposal).toBeDefined();
        expect(proposal.id).toMatch(/^transfer-/);
        expect(proposal.assetId).toBe(asset.id);
        expect(proposal.sharePercentage).toBe(100);
        expect(proposal.consideration).toBe(50000);
      });

      it('should auto-approve for sole ownership', () => {
        const asset = service.registerAsset('equipment', 'Tool', 'Description', 'did:mycelix:owner');

        const proposal = service.proposeTransfer(
          asset.id,
          'did:mycelix:owner',
          'did:mycelix:buyer',
          100
        );

        expect(proposal.status).toBe('approved');
        expect(proposal.requiredApprovals).toBe(1);
      });

      it('should require multiple approvals for joint ownership', () => {
        const asset = service.registerAsset('equipment', 'Tool', 'Description', 'did:mycelix:o1', 'joint');

        // Add more owners
        asset.owners.push(
          { ownerId: 'did:mycelix:o2', percentage: 0, acquiredAt: Date.now() },
          { ownerId: 'did:mycelix:o3', percentage: 0, acquiredAt: Date.now() }
        );
        asset.owners[0].percentage = 34;
        asset.owners[1].percentage = 33;
        asset.owners[2].percentage = 33;

        const proposal = service.proposeTransfer(
          asset.id,
          'did:mycelix:o1',
          'did:mycelix:buyer',
          34
        );

        expect(proposal.status).toBe('pending_approval');
        expect(proposal.requiredApprovals).toBe(2); // Majority of 3
      });

      it('should throw for share percentage exceeding 100%', () => {
        const asset = service.registerAsset('digital', 'Token', 'Description', 'did:mycelix:owner');

        expect(() => {
          service.proposeTransfer(asset.id, 'did:mycelix:owner', 'did:mycelix:buyer', 150);
        }).toThrow('Transfer share percentage cannot exceed 100%');
      });

      it('should throw for non-existent asset', () => {
        expect(() => {
          service.proposeTransfer('asset-fake', 'did:mycelix:owner', 'did:mycelix:buyer', 50);
        }).toThrow('Asset not found');
      });
    });

    describe('approveTransfer', () => {
      it('should add approval to transfer', () => {
        const asset = service.registerAsset('equipment', 'Equipment', 'Desc', 'did:mycelix:o1', 'joint');
        asset.owners.push({ ownerId: 'did:mycelix:o2', percentage: 50, acquiredAt: Date.now() });
        asset.owners[0].percentage = 50;

        const proposal = service.proposeTransfer(asset.id, 'did:mycelix:o1', 'did:mycelix:buyer', 50);

        const approved = service.approveTransfer(proposal.id, 'did:mycelix:o2');

        expect(approved.approvals).toContain('did:mycelix:o2');
        expect(approved.status).toBe('approved');
      });

      it('should throw for non-owner approval', () => {
        const asset = service.registerAsset('equipment', 'Equipment', 'Desc', 'did:mycelix:owner');
        const proposal = service.proposeTransfer(asset.id, 'did:mycelix:owner', 'did:mycelix:buyer', 100);

        expect(() => {
          service.approveTransfer(proposal.id, 'did:mycelix:unauthorized');
        }).toThrow('Approver is not an owner');
      });

      it('should not duplicate approvals', () => {
        const asset = service.registerAsset('equipment', 'Equipment', 'Desc', 'did:mycelix:o1', 'joint');
        asset.owners.push({ ownerId: 'did:mycelix:o2', percentage: 50, acquiredAt: Date.now() });
        asset.owners[0].percentage = 50;

        const proposal = service.proposeTransfer(asset.id, 'did:mycelix:o1', 'did:mycelix:buyer', 50);
        service.approveTransfer(proposal.id, 'did:mycelix:o1');
        service.approveTransfer(proposal.id, 'did:mycelix:o1');

        expect(proposal.approvals.filter(a => a === 'did:mycelix:o1').length).toBe(1);
      });
    });

    describe('executeTransfer', () => {
      it('should execute an approved transfer', () => {
        const asset = service.registerAsset('vehicle', 'Car', 'Description', 'did:mycelix:seller');
        const proposal = service.proposeTransfer(asset.id, 'did:mycelix:seller', 'did:mycelix:buyer', 100);

        const updated = service.executeTransfer(proposal.id);

        expect(updated.owners.length).toBe(1);
        expect(updated.owners[0].ownerId).toBe('did:mycelix:buyer');
        expect(updated.owners[0].percentage).toBe(100);
      });

      it('should handle partial transfers', () => {
        const asset = service.registerAsset('digital', 'Token', 'Description', 'did:mycelix:owner');
        const proposal = service.proposeTransfer(asset.id, 'did:mycelix:owner', 'did:mycelix:buyer', 40);

        const updated = service.executeTransfer(proposal.id);

        expect(updated.owners.length).toBe(2);
        expect(updated.owners.find(o => o.ownerId === 'did:mycelix:owner')!.percentage).toBe(60);
        expect(updated.owners.find(o => o.ownerId === 'did:mycelix:buyer')!.percentage).toBe(40);
      });

      it('should throw for unapproved transfer', () => {
        // Create a joint ownership asset with 3 owners (requires 2 approvals)
        const asset = service.registerAsset('equipment', 'Tool', 'Desc', 'did:mycelix:o1', 'joint');
        asset.owners.push(
          { ownerId: 'did:mycelix:o2', percentage: 33, acquiredAt: Date.now() },
          { ownerId: 'did:mycelix:o3', percentage: 33, acquiredAt: Date.now() }
        );
        asset.owners[0].percentage = 34;

        // Create proposal - requires 2 approvals (majority of 3)
        // Only proposer (o1) has approved at this point
        const proposal = service.proposeTransfer(asset.id, 'did:mycelix:o1', 'did:mycelix:buyer', 34);

        // Proposal should be pending_approval since only 1 of 2 required approvals
        expect(proposal.status).toBe('pending_approval');

        expect(() => {
          service.executeTransfer(proposal.id);
        }).toThrow('Transfer not approved');
      });

      it('should mark transfer as completed', () => {
        const asset = service.registerAsset('vehicle', 'Bike', 'Description', 'did:mycelix:seller');
        const proposal = service.proposeTransfer(asset.id, 'did:mycelix:seller', 'did:mycelix:buyer', 100);

        service.executeTransfer(proposal.id);

        expect(proposal.status).toBe('completed');
        expect(proposal.completedAt).toBeDefined();
      });
    });

    describe('joinCommons', () => {
      it('should create a commons membership', () => {
        const membership = service.joinCommons(
          'commons-forest',
          'did:mycelix:member1',
          ['read', 'harvest']
        );

        expect(membership).toBeDefined();
        expect(membership.commonsId).toBe('commons-forest');
        expect(membership.memberId).toBe('did:mycelix:member1');
        expect(membership.accessRights).toContain('read');
        expect(membership.accessRights).toContain('harvest');
        expect(membership.contributionCredits).toBe(0);
      });

      it('should default to read access', () => {
        const membership = service.joinCommons('commons-garden', 'did:mycelix:member');

        expect(membership.accessRights).toContain('read');
      });

      it('should initialize reputation for member', () => {
        const membership = service.joinCommons('commons-well', 'did:mycelix:member');

        expect(membership.reputation).toBeDefined();
        expect(membership.reputation.agentId).toBe('did:mycelix:member');
      });
    });

    describe('recordUsage', () => {
      it('should record usage of commons asset', () => {
        const usage = service.recordUsage(
          'asset-forest',
          'did:mycelix:user',
          'Collecting firewood'
        );

        expect(usage).toBeDefined();
        expect(usage.id).toMatch(/^usage-/);
        expect(usage.assetId).toBe('asset-forest');
        expect(usage.purpose).toBe('Collecting firewood');
        expect(usage.approved).toBe(true);
      });

      it('should track start time', () => {
        const usage = service.recordUsage('asset-garden', 'did:mycelix:user', 'Planting');

        expect(usage.startTime).toBeDefined();
        expect(usage.startTime).toBeLessThanOrEqual(Date.now());
      });
    });

    describe('getAsset', () => {
      it('should retrieve an existing asset', () => {
        const created = service.registerAsset('real_estate', 'Land', 'Description', 'did:mycelix:owner');
        const retrieved = service.getAsset(created.id);

        expect(retrieved).toBeDefined();
        expect(retrieved!.id).toBe(created.id);
      });

      it('should return undefined for non-existent asset', () => {
        const result = service.getAsset('asset-fake');
        expect(result).toBeUndefined();
      });
    });

    describe('getOwnedAssets', () => {
      it('should return all assets owned by a DID', () => {
        service.registerAsset('real_estate', 'House', 'Desc', 'did:mycelix:owner');
        service.registerAsset('vehicle', 'Car', 'Desc', 'did:mycelix:owner');
        service.registerAsset('equipment', 'Tractor', 'Desc', 'did:mycelix:other');

        const assets = service.getOwnedAssets('did:mycelix:owner');

        expect(assets.length).toBe(2);
      });

      it('should return empty array for owner with no assets', () => {
        const assets = service.getOwnedAssets('did:mycelix:noassets');
        expect(assets).toEqual([]);
      });
    });

    describe('verifyOwnership', () => {
      it('should verify ownership claim', async () => {
        const asset = service.registerAsset('digital', 'Token', 'Description', 'did:mycelix:owner');

        const result = await service.verifyOwnership(asset.id, 'did:mycelix:owner');

        expect(result.verified).toBe(true);
        expect(result.share).toBe(100);
      });

      it('should return false for non-owner', async () => {
        const asset = service.registerAsset('digital', 'Token', 'Description', 'did:mycelix:owner');

        const result = await service.verifyOwnership(asset.id, 'did:mycelix:notowner');

        expect(result.verified).toBe(false);
        expect(result.share).toBe(0);
      });

      it('should return false for non-existent asset', async () => {
        const result = await service.verifyOwnership('asset-fake', 'did:mycelix:anyone');

        expect(result.verified).toBe(false);
        expect(result.share).toBe(0);
      });
    });
  });

  describe('getPropertyService', () => {
    it('should return singleton instance', () => {
      const service1 = getPropertyService();
      const service2 = getPropertyService();

      expect(service1).toBe(service2);
    });
  });
});
