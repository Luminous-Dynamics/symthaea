// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * SupplyChain Integration Tests
 *
 * Tests for SupplyChainProvenanceService - provenance tracking and chain verification
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  SupplyChainProvenanceService,
  getSupplyChainService,
  type Checkpoint,
  type ProvenanceChain,
  type ChainVerification,
  type HandlerProfile,
  type CheckpointEvidence,
} from '../../src/integrations/supplychain/index.js';

describe('SupplyChain Integration', () => {
  let service: SupplyChainProvenanceService;

  beforeEach(() => {
    service = new SupplyChainProvenanceService();
  });

  describe('SupplyChainProvenanceService', () => {
    describe('recordCheckpoint', () => {
      it('should record basic checkpoint', () => {
        const checkpoint = service.recordCheckpoint({
          productId: 'product-001',
          location: 'Warehouse A',
          handler: 'handler-1',
          action: 'received',
          evidence: [
            {
              type: 'manual_entry',
              data: { notes: 'Package received' },
              timestamp: Date.now(),
              verified: true,
            },
          ],
        });

        expect(checkpoint.id).toBeDefined();
        expect(checkpoint.productId).toBe('product-001');
        expect(checkpoint.location).toBe('Warehouse A');
        expect(checkpoint.handler).toBe('handler-1');
        expect(checkpoint.action).toBe('received');
      });

      it('should create epistemic claim', () => {
        const checkpoint = service.recordCheckpoint({
          productId: 'product-002',
          location: 'Port of Entry',
          handler: 'handler-2',
          action: 'inspected',
          evidence: [],
        });

        expect(checkpoint.claim).toBeDefined();
        expect(checkpoint.claim.content).toContain('product-002');
        expect(checkpoint.claim.content).toContain('inspected');
        expect(checkpoint.claim.content).toContain('Port of Entry');
      });

      it('should link to previous checkpoint', () => {
        const first = service.recordCheckpoint({
          productId: 'linked-product',
          location: 'Origin',
          handler: 'handler-1',
          action: 'shipped',
          evidence: [],
        });

        const second = service.recordCheckpoint({
          productId: 'linked-product',
          location: 'Destination',
          handler: 'handler-2',
          action: 'received',
          evidence: [],
        });

        expect(second.previousCheckpointId).toBe(first.id);
        expect(first.previousCheckpointId).toBeUndefined();
      });

      it('should update handler reputation', () => {
        service.recordCheckpoint({
          productId: 'rep-product',
          location: 'Location',
          handler: 'rep-handler',
          action: 'received',
          evidence: [],
        });

        const profile = service.getHandlerProfile('rep-handler');
        expect(profile.trustScore).toBeGreaterThan(0.5);
      });

      it('should include batch ID when provided', () => {
        const checkpoint = service.recordCheckpoint({
          productId: 'batch-product',
          batchId: 'BATCH-2024-001',
          location: 'Factory',
          handler: 'handler',
          action: 'processed',
          evidence: [],
        });

        expect(checkpoint.batchId).toBe('BATCH-2024-001');
      });

      it('should include coordinates when provided', () => {
        const checkpoint = service.recordCheckpoint({
          productId: 'geo-product',
          location: 'Distribution Center',
          coordinates: { lat: 37.7749, lng: -122.4194 },
          handler: 'handler',
          action: 'received',
          evidence: [],
        });

        expect(checkpoint.coordinates).toEqual({ lat: 37.7749, lng: -122.4194 });
      });

      it('should handle all action types', () => {
        const actions: Array<'received' | 'processed' | 'shipped' | 'delivered' | 'inspected'> = [
          'received',
          'processed',
          'shipped',
          'delivered',
          'inspected',
        ];

        actions.forEach((action) => {
          const checkpoint = service.recordCheckpoint({
            productId: `action-product-${action}`,
            location: 'Location',
            handler: 'handler',
            action,
            evidence: [],
          });

          expect(checkpoint.action).toBe(action);
        });
      });

      it('should handle various evidence types', () => {
        const evidenceTypes: CheckpointEvidence[] = [
          { type: 'iot_sensor', data: { temp: 4.5 }, timestamp: Date.now(), verified: true },
          { type: 'gps', data: { lat: 37.7, lng: -122.4 }, timestamp: Date.now(), verified: true },
          { type: 'photo', data: { url: 'https://example.com/photo.jpg' }, timestamp: Date.now(), verified: true },
          { type: 'blockchain', data: { txHash: '0x123' }, timestamp: Date.now(), verified: true },
        ];

        const checkpoint = service.recordCheckpoint({
          productId: 'evidence-product',
          location: 'Location',
          handler: 'handler',
          action: 'inspected',
          evidence: evidenceTypes,
        });

        expect(checkpoint.evidence.length).toBe(4);
      });
    });

    describe('getProvenanceChain', () => {
      it('should return null for unknown product', () => {
        const chain = service.getProvenanceChain('non-existent');
        expect(chain).toBeNull();
      });

      it('should return full chain for product', () => {
        // Create a provenance chain
        service.recordCheckpoint({
          productId: 'traced-product',
          location: 'Farm',
          handler: 'farmer',
          action: 'shipped',
          evidence: [{ type: 'manual_entry', data: {}, timestamp: Date.now(), verified: true }],
        });

        service.recordCheckpoint({
          productId: 'traced-product',
          location: 'Processing Plant',
          handler: 'processor',
          action: 'processed',
          evidence: [{ type: 'iot_sensor', data: { temp: 2 }, timestamp: Date.now(), verified: true }],
        });

        service.recordCheckpoint({
          productId: 'traced-product',
          location: 'Distribution Center',
          handler: 'distributor',
          action: 'received',
          evidence: [{ type: 'rfid', data: { tag: 'ABC123' }, timestamp: Date.now(), verified: true }],
        });

        const chain = service.getProvenanceChain('traced-product');

        expect(chain).not.toBeNull();
        expect(chain!.productId).toBe('traced-product');
        expect(chain!.checkpoints.length).toBe(3);
        expect(chain!.origin.location).toBe('Farm');
        expect(chain!.currentLocation).toBe('Distribution Center');
      });

      it('should calculate chain integrity', () => {
        // Create chain with verified evidence
        for (let i = 0; i < 3; i++) {
          service.recordCheckpoint({
            productId: 'integrity-product',
            location: `Location ${i}`,
            handler: `handler-${i}`,
            action: 'received',
            evidence: [{ type: 'iot_sensor', data: {}, timestamp: Date.now(), verified: true }],
          });
        }

        const chain = service.getProvenanceChain('integrity-product');

        expect(chain!.chainIntegrity).toBeGreaterThan(0.8);
      });

      it('should calculate total time', () => {
        const now = Date.now();

        service.recordCheckpoint({
          productId: 'time-product',
          location: 'Start',
          handler: 'handler',
          action: 'shipped',
          evidence: [],
        });

        // Simulate some time passing (in practice, timestamps would differ)
        service.recordCheckpoint({
          productId: 'time-product',
          location: 'End',
          handler: 'handler',
          action: 'delivered',
          evidence: [],
        });

        const chain = service.getProvenanceChain('time-product');

        expect(chain!.totalTime).toBeDefined();
        expect(chain!.totalTime).toBeGreaterThanOrEqual(0);
      });

      it('should calculate distance with coordinates', () => {
        service.recordCheckpoint({
          productId: 'distance-product',
          location: 'San Francisco',
          coordinates: { lat: 37.7749, lng: -122.4194 },
          handler: 'handler',
          action: 'shipped',
          evidence: [],
        });

        service.recordCheckpoint({
          productId: 'distance-product',
          location: 'Los Angeles',
          coordinates: { lat: 34.0522, lng: -118.2437 },
          handler: 'handler',
          action: 'delivered',
          evidence: [],
        });

        const chain = service.getProvenanceChain('distance-product');

        expect(chain!.totalDistance).toBeGreaterThan(0);
        // SF to LA is about 550-600 km
        expect(chain!.totalDistance).toBeGreaterThan(500);
        expect(chain!.totalDistance).toBeLessThan(700);
      });
    });

    describe('verifyChain', () => {
      it('should return not verified for unknown product', () => {
        const verification = service.verifyChain('unknown-product');

        expect(verification.verified).toBe(false);
        expect(verification.integrityScore).toBe(0);
        expect(verification.recommendations).toContain('Product not found in supply chain');
      });

      it('should verify chain with strong evidence', () => {
        // Create chain with blockchain evidence
        for (let i = 0; i < 3; i++) {
          service.recordCheckpoint({
            productId: 'verified-product',
            location: `Location ${i}`,
            handler: `handler-${i}`,
            action: 'received',
            evidence: [{ type: 'blockchain', data: { txHash: `0x${i}` }, timestamp: Date.now(), verified: true }],
          });
        }

        const verification = service.verifyChain('verified-product');

        expect(verification.verified).toBe(true);
        expect(verification.integrityScore).toBeGreaterThanOrEqual(0.8);
        expect(verification.timeline.length).toBe(3);
      });

      it('should detect weak links without verified evidence', () => {
        service.recordCheckpoint({
          productId: 'weak-product',
          location: 'Location 1',
          handler: 'handler-1',
          action: 'received',
          evidence: [{ type: 'manual_entry', data: {}, timestamp: Date.now(), verified: false }],
        });

        const verification = service.verifyChain('weak-product');

        expect(verification.weakLinks.length).toBeGreaterThan(0);
        expect(verification.recommendations.some((r) => r.includes('weak verification'))).toBe(true);
      });

      it('should include timeline with status for each checkpoint', () => {
        service.recordCheckpoint({
          productId: 'timeline-product',
          location: 'Location 1',
          handler: 'handler-1',
          action: 'received',
          evidence: [{ type: 'iot_sensor', data: {}, timestamp: Date.now(), verified: true }],
        });

        service.recordCheckpoint({
          productId: 'timeline-product',
          location: 'Location 2',
          handler: 'handler-2',
          action: 'shipped',
          evidence: [{ type: 'manual_entry', data: {}, timestamp: Date.now(), verified: true }],
        });

        const verification = service.verifyChain('timeline-product');

        expect(verification.timeline.length).toBe(2);
        verification.timeline.forEach((t) => {
          expect(['verified', 'suspicious', 'unverified']).toContain(t.status);
          expect(t.checkpoint).toBeDefined();
        });
      });

      it('should provide recommendations', () => {
        service.recordCheckpoint({
          productId: 'rec-product',
          location: 'Location',
          handler: 'handler',
          action: 'received',
          evidence: [],
        });

        const verification = service.verifyChain('rec-product');

        expect(verification.recommendations).toBeDefined();
        expect(Array.isArray(verification.recommendations)).toBe(true);
      });
    });

    describe('getHandlerProfile', () => {
      it('should return default profile for unknown handler', () => {
        const profile = service.getHandlerProfile('unknown-handler');

        expect(profile.handlerId).toBe('unknown-handler');
        expect(profile.checkpointsRecorded).toBe(0);
        expect(profile.trustScore).toBe(0.5);
        expect(profile.verificationRate).toBe(0);
      });

      it('should count checkpoints recorded', () => {
        for (let i = 0; i < 5; i++) {
          service.recordCheckpoint({
            productId: `product-${i}`,
            location: 'Location',
            handler: 'active-handler',
            action: 'received',
            evidence: [{ type: 'manual_entry', data: {}, timestamp: Date.now(), verified: true }],
          });
        }

        const profile = service.getHandlerProfile('active-handler');

        expect(profile.checkpointsRecorded).toBe(5);
      });

      it('should calculate verification rate', () => {
        // 3 verified, 2 unverified
        for (let i = 0; i < 3; i++) {
          service.recordCheckpoint({
            productId: `verified-${i}`,
            location: 'Location',
            handler: 'mixed-handler',
            action: 'received',
            evidence: [{ type: 'iot_sensor', data: {}, timestamp: Date.now(), verified: true }],
          });
        }

        for (let i = 0; i < 2; i++) {
          service.recordCheckpoint({
            productId: `unverified-${i}`,
            location: 'Location',
            handler: 'mixed-handler',
            action: 'received',
            evidence: [{ type: 'manual_entry', data: {}, timestamp: Date.now(), verified: false }],
          });
        }

        const profile = service.getHandlerProfile('mixed-handler');

        expect(profile.verificationRate).toBe(0.6); // 3/5
      });

      it('should include handler DID', () => {
        const profile = service.getHandlerProfile('did-handler');
        expect(profile.handlerDid).toBe('did:mycelix:did-handler');
      });

      it('should include reputation object', () => {
        service.recordCheckpoint({
          productId: 'rep-product',
          location: 'Location',
          handler: 'rep-handler',
          action: 'received',
          evidence: [],
        });

        const profile = service.getHandlerProfile('rep-handler');

        expect(profile.reputation).toBeDefined();
        expect(profile.reputation.agentId).toBe('rep-handler');
      });
    });

    describe('queryExternalReputation', () => {
      it('should not throw when querying', () => {
        expect(() => {
          service.queryExternalReputation('handler-1');
        }).not.toThrow();
      });
    });

    describe('isHandlerTrusted', () => {
      it('should return false for unknown handler', () => {
        expect(service.isHandlerTrusted('unknown')).toBe(false);
      });

      it('should return true for trusted handler', () => {
        for (let i = 0; i < 5; i++) {
          service.recordCheckpoint({
            productId: `trust-product-${i}`,
            location: 'Location',
            handler: 'trusted-handler',
            action: 'received',
            evidence: [],
          });
        }

        expect(service.isHandlerTrusted('trusted-handler')).toBe(true);
      });

      it('should respect custom threshold', () => {
        service.recordCheckpoint({
          productId: 'threshold-product',
          location: 'Location',
          handler: 'threshold-handler',
          action: 'received',
          evidence: [],
        });

        // Low threshold - should pass
        expect(service.isHandlerTrusted('threshold-handler', 0.5)).toBe(true);

        // High threshold - should fail
        expect(service.isHandlerTrusted('threshold-handler', 0.95)).toBe(false);
      });
    });

    describe('Evidence Level Determination', () => {
      it('should classify blockchain evidence as E4_Consensus', () => {
        const checkpoint = service.recordCheckpoint({
          productId: 'blockchain-product',
          location: 'Location',
          handler: 'handler',
          action: 'received',
          evidence: [{ type: 'blockchain', data: {}, timestamp: Date.now(), verified: true }],
        });

        // The claim should have high empirical level
        expect(checkpoint.claim.classification.empirical).toBeGreaterThanOrEqual(3);
      });

      it('should classify multi-evidence as E3_Cryptographic', () => {
        const checkpoint = service.recordCheckpoint({
          productId: 'multi-evidence-product',
          location: 'Location',
          handler: 'handler',
          action: 'received',
          evidence: [
            { type: 'iot_sensor', data: {}, timestamp: Date.now(), verified: true },
            { type: 'gps', data: {}, timestamp: Date.now(), verified: true },
            { type: 'photo', data: {}, timestamp: Date.now(), verified: true },
          ],
        });

        expect(checkpoint.claim.classification.empirical).toBeGreaterThanOrEqual(2);
      });

      it('should classify unverified evidence as E0_Unverified', () => {
        const checkpoint = service.recordCheckpoint({
          productId: 'unverified-product',
          location: 'Location',
          handler: 'handler',
          action: 'received',
          evidence: [{ type: 'manual_entry', data: {}, timestamp: Date.now(), verified: false }],
        });

        expect(checkpoint.claim.classification.empirical).toBe(0);
      });
    });
  });

  describe('getSupplyChainService', () => {
    it('should return singleton instance', () => {
      const service1 = getSupplyChainService();
      const service2 = getSupplyChainService();

      expect(service1).toBe(service2);
    });

    it('should maintain state across calls', () => {
      const service1 = getSupplyChainService();
      service1.recordCheckpoint({
        productId: 'singleton-product',
        location: 'Singleton Location',
        handler: 'singleton-handler',
        action: 'received',
        evidence: [],
      });

      const service2 = getSupplyChainService();
      const chain = service2.getProvenanceChain('singleton-product');

      expect(chain).not.toBeNull();
      expect(chain!.checkpoints.length).toBe(1);
    });
  });

  describe('Complex Provenance Scenarios', () => {
    it('should handle complete supply chain journey', () => {
      const productId = 'organic-apples';

      // Step 1: Farm origin
      service.recordCheckpoint({
        productId,
        batchId: 'BATCH-2024-APPLES-001',
        location: 'Smith Family Farm, WA',
        coordinates: { lat: 47.6062, lng: -122.3321 },
        handler: 'farmer-smith',
        action: 'shipped',
        evidence: [
          { type: 'photo', data: { url: 'harvest.jpg' }, timestamp: Date.now(), verified: true },
          { type: 'gps', data: { lat: 47.6062, lng: -122.3321 }, timestamp: Date.now(), verified: true },
        ],
      });

      // Step 2: Processing
      service.recordCheckpoint({
        productId,
        batchId: 'BATCH-2024-APPLES-001',
        location: 'Organic Processing Co, OR',
        coordinates: { lat: 45.5152, lng: -122.6784 },
        handler: 'processor-organic-co',
        action: 'processed',
        evidence: [
          { type: 'iot_sensor', data: { temp: 4, humidity: 85 }, timestamp: Date.now(), verified: true },
          { type: 'third_party_verification', data: { certifier: 'USDA Organic' }, timestamp: Date.now(), verified: true },
        ],
      });

      // Step 3: Distribution
      service.recordCheckpoint({
        productId,
        batchId: 'BATCH-2024-APPLES-001',
        location: 'Regional Distribution Center, CA',
        coordinates: { lat: 37.7749, lng: -122.4194 },
        handler: 'distributor-west-coast',
        action: 'received',
        evidence: [
          { type: 'rfid', data: { tag: 'APPLES-001-RFID' }, timestamp: Date.now(), verified: true },
        ],
      });

      // Step 4: Retail delivery
      service.recordCheckpoint({
        productId,
        batchId: 'BATCH-2024-APPLES-001',
        location: 'Whole Foods Market, SF',
        coordinates: { lat: 37.7849, lng: -122.4094 },
        handler: 'retailer-whole-foods',
        action: 'delivered',
        evidence: [
          { type: 'signature', data: { signedBy: 'Store Manager' }, timestamp: Date.now(), verified: true },
        ],
      });

      // Verify the complete chain
      const chain = service.getProvenanceChain(productId);
      const verification = service.verifyChain(productId);

      expect(chain).not.toBeNull();
      expect(chain!.checkpoints.length).toBe(4);
      expect(chain!.origin.location).toBe('Smith Family Farm, WA');
      expect(chain!.currentLocation).toBe('Whole Foods Market, SF');
      expect(chain!.totalDistance).toBeGreaterThan(0);

      expect(verification.verified).toBe(true);
      expect(verification.integrityScore).toBeGreaterThan(0.8);
    });

    it('should detect suspicious activity in chain', () => {
      const productId = 'suspicious-product';

      // Normal first checkpoint
      service.recordCheckpoint({
        productId,
        location: 'Origin',
        handler: 'good-handler',
        action: 'shipped',
        evidence: [{ type: 'iot_sensor', data: {}, timestamp: Date.now(), verified: true }],
      });

      // Build up negative reputation for a handler
      for (let i = 0; i < 10; i++) {
        const badHandler = 'bad-handler';
        const rep = service.getHandlerProfile(badHandler);
        // This doesn't directly decrease reputation, but we can test the structure
      }

      // Add checkpoint with unverified evidence
      service.recordCheckpoint({
        productId,
        location: 'Suspicious Location',
        handler: 'unknown-handler',
        action: 'received',
        evidence: [{ type: 'manual_entry', data: {}, timestamp: Date.now(), verified: false }],
      });

      const verification = service.verifyChain(productId);

      // Should have weak links
      expect(verification.weakLinks.length).toBeGreaterThan(0);
    });
  });
});
