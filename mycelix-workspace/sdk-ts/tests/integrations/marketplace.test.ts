/**
 * Marketplace Integration Tests
 *
 * Tests for MarketplaceReputationService - transaction trust and seller/buyer profiles
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  MarketplaceReputationService,
  getMarketplaceService,
  type Transaction,
  type SellerProfile,
  type BuyerProfile,
  type ListingVerification,
} from '../../src/integrations/marketplace/index.js';

describe('Marketplace Integration', () => {
  let service: MarketplaceReputationService;

  beforeEach(() => {
    service = new MarketplaceReputationService();
  });

  describe('MarketplaceReputationService', () => {
    describe('recordTransaction', () => {
      it('should record successful transaction', () => {
        const tx: Transaction = {
          id: 'tx-001',
          type: 'purchase',
          buyerId: 'buyer-1',
          sellerId: 'seller-1',
          amount: 100,
          currency: 'USD',
          itemId: 'item-1',
          timestamp: Date.now(),
          success: true,
        };

        const result = service.recordTransaction(tx);

        expect(result.sellerScore).toBeGreaterThan(0.5);
        expect(result.buyerScore).toBeGreaterThan(0.5);
      });

      it('should record failed transaction', () => {
        const tx: Transaction = {
          id: 'tx-002',
          type: 'purchase',
          buyerId: 'buyer-2',
          sellerId: 'seller-2',
          amount: 50,
          currency: 'USD',
          itemId: 'item-2',
          timestamp: Date.now(),
          success: false,
        };

        const result = service.recordTransaction(tx);

        // Failed transaction reduces reputation
        expect(result.sellerScore).toBeLessThan(0.6);
        expect(result.buyerScore).toBeLessThan(0.6);
      });

      it('should accumulate reputation over multiple transactions', () => {
        // Multiple successful transactions
        for (let i = 0; i < 5; i++) {
          service.recordTransaction({
            id: `tx-${i}`,
            type: 'purchase',
            buyerId: 'buyer-regular',
            sellerId: 'seller-regular',
            amount: 100,
            currency: 'USD',
            itemId: `item-${i}`,
            timestamp: Date.now(),
            success: true,
          });
        }

        const seller = service.getSellerProfile('seller-regular');
        const buyer = service.getBuyerProfile('buyer-regular');

        expect(seller.transactionCount).toBe(5);
        expect(buyer.transactionCount).toBe(5);
        expect(seller.trustScore).toBeGreaterThan(0.7);
      });

      it('should handle different transaction types', () => {
        const types: Transaction['type'][] = ['purchase', 'sale', 'exchange', 'rental'];

        types.forEach((type, i) => {
          service.recordTransaction({
            id: `tx-type-${i}`,
            type,
            buyerId: 'buyer-types',
            sellerId: 'seller-types',
            amount: 100,
            currency: 'USD',
            itemId: `item-${i}`,
            timestamp: Date.now(),
            success: true,
          });
        });

        const seller = service.getSellerProfile('seller-types');
        expect(seller.transactionCount).toBe(4);
      });
    });

    describe('getSellerProfile', () => {
      it('should return default profile for new seller', () => {
        const profile = service.getSellerProfile('new-seller');

        expect(profile.sellerId).toBe('new-seller');
        expect(profile.transactionCount).toBe(0);
        expect(profile.trustScore).toBe(0.5);
        expect(profile.verified).toBe(false);
      });

      it('should calculate success rate correctly', () => {
        // 3 successful, 1 failed
        for (let i = 0; i < 3; i++) {
          service.recordTransaction({
            id: `tx-success-${i}`,
            type: 'purchase',
            buyerId: 'buyer',
            sellerId: 'seller-mixed',
            amount: 100,
            currency: 'USD',
            itemId: `item-${i}`,
            timestamp: Date.now(),
            success: true,
          });
        }
        service.recordTransaction({
          id: 'tx-fail',
          type: 'purchase',
          buyerId: 'buyer',
          sellerId: 'seller-mixed',
          amount: 100,
          currency: 'USD',
          itemId: 'item-fail',
          timestamp: Date.now(),
          success: false,
        });

        const profile = service.getSellerProfile('seller-mixed');

        // 4 positive (1 prior + 3) / 6 total (2 prior + 4) = 0.667
        expect(profile.successRate).toBeGreaterThan(0.6);
        expect(profile.successRate).toBeLessThan(0.8);
      });

      it('should mark seller as verified with high trust and volume', () => {
        // Need 10+ transactions and 90%+ trust
        for (let i = 0; i < 15; i++) {
          service.recordTransaction({
            id: `tx-verify-${i}`,
            type: 'purchase',
            buyerId: 'buyer',
            sellerId: 'top-seller',
            amount: 100,
            currency: 'USD',
            itemId: `item-${i}`,
            timestamp: Date.now(),
            success: true,
          });
        }

        const profile = service.getSellerProfile('top-seller');

        expect(profile.verified).toBe(true);
        expect(profile.trustScore).toBeGreaterThanOrEqual(0.9);
        expect(profile.transactionCount).toBeGreaterThanOrEqual(10);
      });

      it('should include lastActive timestamp', () => {
        service.recordTransaction({
          id: 'tx-active',
          type: 'purchase',
          buyerId: 'buyer',
          sellerId: 'active-seller',
          amount: 100,
          currency: 'USD',
          itemId: 'item',
          timestamp: Date.now(),
          success: true,
        });

        const profile = service.getSellerProfile('active-seller');

        expect(profile.lastActive).toBeDefined();
        expect(profile.lastActive).toBeLessThanOrEqual(Date.now());
      });
    });

    describe('getBuyerProfile', () => {
      it('should return default profile for new buyer', () => {
        const profile = service.getBuyerProfile('new-buyer');

        expect(profile.buyerId).toBe('new-buyer');
        expect(profile.transactionCount).toBe(0);
        expect(profile.trustScore).toBe(0.5);
      });

      it('should calculate payment reliability', () => {
        for (let i = 0; i < 5; i++) {
          service.recordTransaction({
            id: `tx-reliable-${i}`,
            type: 'purchase',
            buyerId: 'reliable-buyer',
            sellerId: 'seller',
            amount: 100,
            currency: 'USD',
            itemId: `item-${i}`,
            timestamp: Date.now(),
            success: true,
          });
        }

        const profile = service.getBuyerProfile('reliable-buyer');

        expect(profile.paymentReliability).toBeGreaterThan(0.7);
      });

      it('should calculate dispute rate', () => {
        // Some successful, some failed
        service.recordTransaction({
          id: 'tx-ok',
          type: 'purchase',
          buyerId: 'disputed-buyer',
          sellerId: 'seller',
          amount: 100,
          currency: 'USD',
          itemId: 'item-1',
          timestamp: Date.now(),
          success: true,
        });
        service.recordTransaction({
          id: 'tx-dispute',
          type: 'purchase',
          buyerId: 'disputed-buyer',
          sellerId: 'seller',
          amount: 100,
          currency: 'USD',
          itemId: 'item-2',
          timestamp: Date.now(),
          success: false,
        });

        const profile = service.getBuyerProfile('disputed-buyer');

        expect(profile.disputeRate).toBeGreaterThan(0);
        expect(profile.disputeRate).toBeLessThan(1);
      });
    });

    describe('verifyListing', () => {
      it('should verify listing from trusted seller', () => {
        // Build up seller reputation
        for (let i = 0; i < 10; i++) {
          service.recordTransaction({
            id: `tx-trust-${i}`,
            type: 'sale',
            buyerId: 'buyer',
            sellerId: 'trusted-seller',
            amount: 100,
            currency: 'USD',
            itemId: `item-${i}`,
            timestamp: Date.now(),
            success: true,
          });
        }

        const verification = service.verifyListing('listing-1', 'trusted-seller');

        expect(verification.verified).toBe(true);
        expect(verification.sellerTrust).toBeGreaterThan(0.7);
        expect(verification.scamRiskScore).toBeLessThan(0.3);
      });

      it('should flag listing from untrusted seller', () => {
        // Build up negative reputation
        for (let i = 0; i < 5; i++) {
          service.recordTransaction({
            id: `tx-untrust-${i}`,
            type: 'sale',
            buyerId: 'buyer',
            sellerId: 'scam-seller',
            amount: 100,
            currency: 'USD',
            itemId: `item-${i}`,
            timestamp: Date.now(),
            success: false,
          });
        }

        const verification = service.verifyListing('listing-2', 'scam-seller');

        expect(verification.verified).toBe(false);
        expect(verification.scamRiskScore).toBeGreaterThan(0.3);
        expect(verification.recommendations.length).toBeGreaterThan(0);
      });

      it('should provide recommendations', () => {
        const verification = service.verifyListing('listing-3', 'new-seller');

        expect(verification.recommendations).toBeDefined();
        expect(Array.isArray(verification.recommendations)).toBe(true);
      });

      it('should include listing ID in result', () => {
        const verification = service.verifyListing('my-listing-id', 'seller');

        expect(verification.listingId).toBe('my-listing-id');
      });
    });

    describe('isSellerTrustworthy', () => {
      it('should return false for unknown seller', () => {
        expect(service.isSellerTrustworthy('unknown-seller')).toBe(false);
      });

      it('should return true for trusted seller', () => {
        for (let i = 0; i < 5; i++) {
          service.recordTransaction({
            id: `tx-${i}`,
            type: 'sale',
            buyerId: 'buyer',
            sellerId: 'good-seller',
            amount: 100,
            currency: 'USD',
            itemId: `item-${i}`,
            timestamp: Date.now(),
            success: true,
          });
        }

        expect(service.isSellerTrustworthy('good-seller')).toBe(true);
      });

      it('should respect custom threshold', () => {
        service.recordTransaction({
          id: 'tx-1',
          type: 'sale',
          buyerId: 'buyer',
          sellerId: 'mid-seller',
          amount: 100,
          currency: 'USD',
          itemId: 'item',
          timestamp: Date.now(),
          success: true,
        });

        // Low threshold - should pass
        expect(service.isSellerTrustworthy('mid-seller', 0.5)).toBe(true);

        // High threshold - should fail
        expect(service.isSellerTrustworthy('mid-seller', 0.9)).toBe(false);
      });
    });

    describe('queryExternalReputation', () => {
      it('should not throw when querying', () => {
        expect(() => {
          service.queryExternalReputation('seller-1');
        }).not.toThrow();
      });
    });

    describe('getFLCoordinator', () => {
      it('should return FL coordinator instance', () => {
        const coordinator = service.getFLCoordinator();

        expect(coordinator).toBeDefined();
        expect(typeof coordinator.registerParticipant).toBe('function');
      });
    });

    describe('getFLStats', () => {
      it('should return FL statistics', () => {
        const stats = service.getFLStats();

        expect(stats).toHaveProperty('totalRounds');
        expect(stats).toHaveProperty('participantCount');
        expect(stats).toHaveProperty('averageParticipation');
      });
    });
  });

  describe('getMarketplaceService', () => {
    it('should return singleton instance', () => {
      const service1 = getMarketplaceService();
      const service2 = getMarketplaceService();

      expect(service1).toBe(service2);
    });
  });

  describe('Cross-Service Reputation', () => {
    it('should share reputation via bridge', () => {
      // Record transaction
      service.recordTransaction({
        id: 'tx-bridge',
        type: 'purchase',
        buyerId: 'buyer-bridge',
        sellerId: 'seller-bridge',
        amount: 100,
        currency: 'USD',
        itemId: 'item',
        timestamp: Date.now(),
        success: true,
      });

      // Seller profile should reflect the transaction
      const seller = service.getSellerProfile('seller-bridge');
      expect(seller.transactionCount).toBe(1);
    });
  });
});
