/**
 * Energy Integration Tests
 *
 * Tests for EnergyService - grid management, trading, and credits
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  EnergyService,
  getEnergyService,
  resetEnergyService,
  type EnergyParticipant,
  type EnergyReading,
  type EnergyTransaction,
  type GridBalanceRequest,
  type EnergyCredit,
} from '../../src/integrations/energy/index.js';

describe('Energy Integration', () => {
  let service: EnergyService;

  beforeEach(() => {
    resetEnergyService();
    service = new EnergyService();
  });

  describe('EnergyService', () => {
    describe('registerParticipant', () => {
      it('should register a new energy participant', () => {
        const participant = service.registerParticipant(
          'did:mycelix:prosumer1',
          'prosumer',
          ['solar', 'battery'],
          10 // 10 kWh capacity
        );

        expect(participant).toBeDefined();
        expect(participant.id).toMatch(/^participant-/);
        expect(participant.did).toBe('did:mycelix:prosumer1');
        expect(participant.role).toBe('prosumer');
        expect(participant.sources).toContain('solar');
        expect(participant.sources).toContain('battery');
        expect(participant.capacity).toBe(10);
      });

      it('should support different roles', () => {
        const producer = service.registerParticipant('did:mycelix:p1', 'producer', ['solar'], 100);
        const consumer = service.registerParticipant('did:mycelix:p2', 'consumer', [], 0);
        const storage = service.registerParticipant('did:mycelix:p3', 'storage', ['battery'], 50);
        const operator = service.registerParticipant('did:mycelix:p4', 'grid_operator', ['grid'], 1000);

        expect(producer.role).toBe('producer');
        expect(consumer.role).toBe('consumer');
        expect(storage.role).toBe('storage');
        expect(operator.role).toBe('grid_operator');
      });

      it('should initialize with zero production and consumption', () => {
        const participant = service.registerParticipant('did:mycelix:user', 'prosumer', ['solar'], 10);

        expect(participant.totalProduced).toBe(0);
        expect(participant.totalConsumed).toBe(0);
        expect(participant.carbonOffset).toBe(0);
      });

      it('should support optional location', () => {
        const participant = service.registerParticipant(
          'did:mycelix:located',
          'producer',
          ['wind'],
          50,
          { lat: 37.7749, lng: -122.4194 }
        );

        expect(participant.location).toBeDefined();
        expect(participant.location!.lat).toBe(37.7749);
        expect(participant.location!.lng).toBe(-122.4194);
      });

      it('should initialize empty credits', () => {
        const participant = service.registerParticipant('did:mycelix:user', 'prosumer', ['solar'], 10);
        const credits = service.getCredits(participant.id);

        expect(credits).toEqual([]);
      });
    });

    describe('submitReading', () => {
      it('should submit an energy reading', () => {
        const participant = service.registerParticipant('did:mycelix:user', 'prosumer', ['solar'], 10);

        const reading = service.submitReading(
          participant.id,
          5, // 5 kWh produced
          2, // 2 kWh consumed
          'solar'
        );

        expect(reading).toBeDefined();
        expect(reading.id).toMatch(/^reading-/);
        expect(reading.production).toBe(5);
        expect(reading.consumption).toBe(2);
        expect(reading.netExport).toBe(3);
        expect(reading.source).toBe('solar');
      });

      it('should update participant totals', () => {
        const participant = service.registerParticipant('did:mycelix:user', 'prosumer', ['solar'], 10);

        service.submitReading(participant.id, 10, 4, 'solar');
        service.submitReading(participant.id, 8, 3, 'solar');

        const updated = service.getParticipant(participant.id);
        expect(updated!.totalProduced).toBe(18);
        expect(updated!.totalConsumed).toBe(7);
      });

      it('should calculate carbon offset for renewable sources', () => {
        const participant = service.registerParticipant('did:mycelix:user', 'producer', ['solar'], 100);

        service.submitReading(participant.id, 100, 0, 'solar');

        const updated = service.getParticipant(participant.id);
        expect(updated!.carbonOffset).toBe(50); // 100 kWh * 0.5 kg/kWh
      });

      it('should not calculate carbon offset for grid source', () => {
        const participant = service.registerParticipant('did:mycelix:user', 'consumer', ['grid'], 0);

        service.submitReading(participant.id, 0, 50, 'grid');

        const updated = service.getParticipant(participant.id);
        expect(updated!.carbonOffset).toBe(0);
      });

      it('should issue credits for net export', () => {
        const participant = service.registerParticipant('did:mycelix:user', 'prosumer', ['solar'], 10);

        service.submitReading(participant.id, 10, 3, 'solar'); // Net export: 7

        const credits = service.getCredits(participant.id);
        expect(credits.length).toBe(1);
        expect(credits[0].amount).toBe(7);
        expect(credits[0].source).toBe('solar');
      });

      it('should throw for non-existent participant', () => {
        expect(() => {
          service.submitReading('participant-fake', 10, 5, 'solar');
        }).toThrow('Participant not found');
      });
    });

    describe('issueCredit', () => {
      it('should issue energy credits', () => {
        const participant = service.registerParticipant('did:mycelix:user', 'producer', ['wind'], 50);

        const credit = service.issueCredit(participant.id, 25, 'wind');

        expect(credit).toBeDefined();
        expect(credit.id).toMatch(/^credit-/);
        expect(credit.amount).toBe(25);
        expect(credit.source).toBe('wind');
        expect(credit.transferable).toBe(true);
      });

      it('should calculate carbon offset for renewable credits', () => {
        const participant = service.registerParticipant('did:mycelix:user', 'producer', ['solar'], 100);

        const credit = service.issueCredit(participant.id, 100, 'solar');

        expect(credit.carbonOffset).toBe(50); // 100 kWh * 0.5 kg/kWh
      });

      it('should not add carbon offset for grid credits', () => {
        const participant = service.registerParticipant('did:mycelix:user', 'consumer', ['grid'], 0);

        const credit = service.issueCredit(participant.id, 50, 'grid');

        expect(credit.carbonOffset).toBe(0);
      });
    });

    describe('tradeEnergy', () => {
      it('should trade energy between participants', () => {
        const seller = service.registerParticipant('did:mycelix:seller', 'producer', ['solar'], 100);
        const buyer = service.registerParticipant('did:mycelix:buyer', 'consumer', [], 0);

        // Give seller some credits
        service.submitReading(seller.id, 50, 0, 'solar'); // Creates 50 credits

        const tx = service.tradeEnergy(seller.id, buyer.id, 30, 'solar', 0.10);

        expect(tx).toBeDefined();
        expect(tx.id).toMatch(/^tx-/);
        expect(tx.type).toBe('sale');
        expect(tx.amount).toBe(30);
        expect(tx.pricePerKwh).toBe(0.10);
        expect(tx.totalPrice).toBe(3); // 30 * 0.10
        expect(tx.settlementStatus).toBe('settled');
      });

      it('should transfer credits from seller to buyer', () => {
        const seller = service.registerParticipant('did:mycelix:seller', 'producer', ['solar'], 100);
        const buyer = service.registerParticipant('did:mycelix:buyer', 'consumer', [], 0);

        service.submitReading(seller.id, 50, 0, 'solar');
        service.tradeEnergy(seller.id, buyer.id, 30, 'solar');

        const sellerCredits = service.getCredits(seller.id);
        const buyerCredits = service.getCredits(buyer.id);

        const sellerTotal = sellerCredits.reduce((sum, c) => sum + c.amount, 0);
        const buyerTotal = buyerCredits.reduce((sum, c) => sum + c.amount, 0);

        expect(sellerTotal).toBe(20); // 50 - 30
        expect(buyerTotal).toBe(30);
      });

      it('should throw for insufficient credits', () => {
        const seller = service.registerParticipant('did:mycelix:seller', 'producer', ['solar'], 10);
        const buyer = service.registerParticipant('did:mycelix:buyer', 'consumer', [], 0);

        service.submitReading(seller.id, 10, 5, 'solar'); // Only 5 net export

        expect(() => {
          service.tradeEnergy(seller.id, buyer.id, 100, 'solar');
        }).toThrow('Insufficient energy credits');
      });

      it('should throw for non-existent participant', () => {
        const seller = service.registerParticipant('did:mycelix:seller', 'producer', ['solar'], 10);

        expect(() => {
          service.tradeEnergy(seller.id, 'participant-fake', 10, 'solar');
        }).toThrow('Participant not found');
      });

      it('should calculate carbon credits for renewable trades', () => {
        const seller = service.registerParticipant('did:mycelix:seller', 'producer', ['wind'], 100);
        const buyer = service.registerParticipant('did:mycelix:buyer', 'consumer', [], 0);

        service.submitReading(seller.id, 100, 0, 'wind');
        const tx = service.tradeEnergy(seller.id, buyer.id, 50, 'wind');

        expect(tx.carbonCredits).toBe(25); // 50 * 0.5
      });
    });

    describe('requestGridBalance', () => {
      it('should create a grid balance request', () => {
        const participant = service.registerParticipant('did:mycelix:user', 'consumer', [], 0);

        const request = service.requestGridBalance(
          participant.id,
          'demand',
          100, // Need 100 kWh
          0.12, // Max price $0.12/kWh
          'high'
        );

        expect(request).toBeDefined();
        expect(request.id).toMatch(/^balance-/);
        expect(request.type).toBe('demand');
        expect(request.amountNeeded).toBe(100);
        expect(request.maxPrice).toBe(0.12);
        expect(request.urgency).toBe('high');
        expect(request.fulfilled).toBe(false);
      });

      it('should set expiry based on urgency', () => {
        const participant = service.registerParticipant('did:mycelix:user', 'grid_operator', ['grid'], 1000);

        const critical = service.requestGridBalance(participant.id, 'supply', 500, undefined, 'critical');
        const low = service.requestGridBalance(participant.id, 'supply', 500, undefined, 'low');

        const criticalExpiry = critical.expiresAt - Date.now();
        const lowExpiry = low.expiresAt - Date.now();

        expect(criticalExpiry).toBeLessThan(lowExpiry);
        expect(criticalExpiry).toBeLessThanOrEqual(1 * 60 * 60 * 1000); // 1 hour
        expect(lowExpiry).toBeLessThanOrEqual(24 * 60 * 60 * 1000); // 24 hours
      });
    });

    describe('getParticipant', () => {
      it('should retrieve an existing participant', () => {
        const created = service.registerParticipant('did:mycelix:user', 'prosumer', ['solar'], 10);
        const retrieved = service.getParticipant(created.id);

        expect(retrieved).toBeDefined();
        expect(retrieved!.id).toBe(created.id);
      });

      it('should return undefined for non-existent participant', () => {
        const result = service.getParticipant('participant-fake');
        expect(result).toBeUndefined();
      });
    });

    describe('getCredits', () => {
      it('should return all credits for a participant', () => {
        const participant = service.registerParticipant('did:mycelix:user', 'producer', ['solar', 'wind'], 100);

        service.issueCredit(participant.id, 50, 'solar');
        service.issueCredit(participant.id, 30, 'wind');

        const credits = service.getCredits(participant.id);

        expect(credits.length).toBe(2);
      });

      it('should return empty array for participant with no credits', () => {
        const participant = service.registerParticipant('did:mycelix:user', 'consumer', [], 0);
        const credits = service.getCredits(participant.id);

        expect(credits).toEqual([]);
      });
    });

    describe('error paths and edge cases', () => {
      it('should throw for negative production in submitReading', () => {
        const participant = service.registerParticipant('did:mycelix:user', 'prosumer', ['solar'], 10);

        expect(() => {
          service.submitReading(participant.id, -5, 2, 'solar');
        }).toThrow('Production cannot be negative');
      });

      it('should throw for negative consumption in submitReading', () => {
        const participant = service.registerParticipant('did:mycelix:user', 'prosumer', ['solar'], 10);

        expect(() => {
          service.submitReading(participant.id, 5, -2, 'solar');
        }).toThrow('Consumption cannot be negative');
      });

      it('should throw for NaN production in submitReading', () => {
        const participant = service.registerParticipant('did:mycelix:user', 'prosumer', ['solar'], 10);

        expect(() => {
          service.submitReading(participant.id, NaN, 2, 'solar');
        }).toThrow('Production must be a valid number');
      });

      it('should throw for negative trade price', () => {
        const seller = service.registerParticipant('did:mycelix:seller', 'producer', ['solar'], 100);
        const buyer = service.registerParticipant('did:mycelix:buyer', 'consumer', [], 0);

        service.submitReading(seller.id, 50, 0, 'solar');

        expect(() => {
          service.tradeEnergy(seller.id, buyer.id, 30, 'solar', -0.10);
        }).toThrow('Energy price cannot be negative');
      });

      it('should throw when seller has no credits of the requested source type', () => {
        const seller = service.registerParticipant('did:mycelix:seller', 'producer', ['solar'], 100);
        const buyer = service.registerParticipant('did:mycelix:buyer', 'consumer', [], 0);

        service.submitReading(seller.id, 50, 0, 'solar'); // Credits are solar

        expect(() => {
          service.tradeEnergy(seller.id, buyer.id, 30, 'wind'); // Request wind credits
        }).toThrow('Insufficient energy credits');
      });

      it('should handle zero production and zero consumption reading', () => {
        const participant = service.registerParticipant('did:mycelix:user', 'prosumer', ['solar'], 10);

        const reading = service.submitReading(participant.id, 0, 0, 'solar');

        expect(reading.production).toBe(0);
        expect(reading.consumption).toBe(0);
        expect(reading.netExport).toBe(0);

        // Should not issue credits for zero net export
        const credits = service.getCredits(participant.id);
        expect(credits.length).toBe(0);
      });

      it('should not issue credits when consumption exceeds production', () => {
        const participant = service.registerParticipant('did:mycelix:user', 'prosumer', ['solar'], 10);

        const reading = service.submitReading(participant.id, 2, 8, 'solar');

        expect(reading.netExport).toBe(-6);

        // No credits should be issued for negative net export
        const credits = service.getCredits(participant.id);
        expect(credits.length).toBe(0);
      });

      it('should throw when trading with non-existent seller', () => {
        const buyer = service.registerParticipant('did:mycelix:buyer', 'consumer', [], 0);

        expect(() => {
          service.tradeEnergy('participant-fake', buyer.id, 10, 'solar');
        }).toThrow('Participant not found');
      });
    });

    describe('getGridStats', () => {
      it('should return aggregated grid statistics', () => {
        const p1 = service.registerParticipant('did:mycelix:p1', 'producer', ['solar'], 100);
        const p2 = service.registerParticipant('did:mycelix:p2', 'prosumer', ['wind'], 50);

        service.submitReading(p1.id, 80, 10, 'solar');
        service.submitReading(p2.id, 40, 30, 'wind');

        const stats = service.getGridStats();

        expect(stats.totalProduction).toBe(120); // 80 + 40
        expect(stats.totalConsumption).toBe(40); // 10 + 30
        expect(stats.netBalance).toBe(80); // 120 - 40
        expect(stats.participantCount).toBe(2);
        expect(stats.carbonOffset).toBe(60); // 80*0.5 + 40*0.5 (both renewable)
      });

      it('should return zeros for empty grid', () => {
        const stats = service.getGridStats();

        expect(stats.totalProduction).toBe(0);
        expect(stats.totalConsumption).toBe(0);
        expect(stats.netBalance).toBe(0);
        expect(stats.participantCount).toBe(0);
        expect(stats.carbonOffset).toBe(0);
      });
    });
  });

  describe('getEnergyService', () => {
    it('should return singleton instance', () => {
      const service1 = getEnergyService();
      const service2 = getEnergyService();

      expect(service1).toBe(service2);
    });
  });
});
