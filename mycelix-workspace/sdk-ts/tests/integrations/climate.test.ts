// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Climate Integration Tests
 *
 * Tests for ClimateService - carbon credit issuance and retirement,
 * Renewable Energy Certificate tracking, project registration and
 * verification, and MATL-based project reputation.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  ClimateService,
  getClimateService,
  resetClimateService,
  type ClimateProject,
  type CarbonCreditBatch,
  type RenewableEnergyCertificate,
} from '../../src/integrations/climate/index.js';

describe('Climate Integration', () => {
  let service: ClimateService;

  beforeEach(() => {
    resetClimateService();
    service = new ClimateService();
  });

  describe('registerProject', () => {
    it('should register a new climate project', () => {
      const project = service.registerProject({
        id: 'proj-001',
        name: 'Solar Farm Offset',
        type: 'renewable_energy',
        location: { lat: 32.7, lng: -96.8 },
        estimatedCreditsPerYear: 5000,
        verifier: 'verra',
      });

      expect(project).toBeDefined();
      expect(project.id).toBe('proj-001');
      expect(project.name).toBe('Solar Farm Offset');
      expect(project.verified).toBe(false);
      expect(project.registeredAt).toBeGreaterThan(0);
    });

    it('should support all project types', () => {
      const types = [
        'renewable_energy',
        'reforestation',
        'methane_capture',
        'direct_air_capture',
        'energy_efficiency',
        'blue_carbon',
        'soil_carbon',
      ] as const;

      for (const type of types) {
        const project = service.registerProject({
          id: `proj-${type}`,
          name: `${type} project`,
          type,
          location: { lat: 0, lng: 0 },
          estimatedCreditsPerYear: 100,
          verifier: 'verra',
        });
        expect(project.type).toBe(type);
      }
    });

    it('should initialize project reputation', () => {
      service.registerProject({
        id: 'proj-rep',
        name: 'Rep Test',
        type: 'reforestation',
        location: { lat: 0, lng: 0 },
        estimatedCreditsPerYear: 100,
        verifier: 'gold_standard',
      });

      const rep = service.getProjectReputation('proj-rep');
      expect(rep).toBeGreaterThan(0);
    });
  });

  describe('verifyProject', () => {
    it('should mark project as verified', () => {
      service.registerProject({
        id: 'proj-v',
        name: 'Verify Test',
        type: 'reforestation',
        location: { lat: 0, lng: 0 },
        estimatedCreditsPerYear: 100,
        verifier: 'verra',
      });

      const verified = service.verifyProject('proj-v');

      expect(verified.verified).toBe(true);
    });

    it('should boost project reputation on verification', () => {
      service.registerProject({
        id: 'proj-rb',
        name: 'Rep Boost',
        type: 'blue_carbon',
        location: { lat: 0, lng: 0 },
        estimatedCreditsPerYear: 500,
        verifier: 'gold_standard',
      });

      const repBefore = service.getProjectReputation('proj-rb');
      service.verifyProject('proj-rb');
      const repAfter = service.getProjectReputation('proj-rb');

      expect(repAfter).toBeGreaterThanOrEqual(repBefore);
    });

    it('should throw for non-existent project', () => {
      expect(() => service.verifyProject('proj-nonexistent')).toThrow('Project not found');
    });
  });

  describe('issueCredits', () => {
    it('should issue carbon credits from a verified project', () => {
      service.registerProject({
        id: 'proj-ic',
        name: 'Issue Credits Test',
        type: 'methane_capture',
        location: { lat: 10, lng: 20 },
        estimatedCreditsPerYear: 1000,
        verifier: 'acr',
      });
      service.verifyProject('proj-ic');

      const batch = service.issueCredits('proj-ic', 500, 'tCO2e');

      expect(batch).toBeDefined();
      expect(batch.id).toMatch(/^credit-/);
      expect(batch.projectId).toBe('proj-ic');
      expect(batch.amount).toBe(500);
      expect(batch.unit).toBe('tCO2e');
      expect(batch.status).toBe('active');
      expect(batch.serialStart).toBeDefined();
      expect(batch.serialEnd).toBeDefined();
    });

    it('should set vintage to current year by default', () => {
      service.registerProject({
        id: 'proj-vy',
        name: 'Vintage Year',
        type: 'solar_carbon' as any,
        location: { lat: 0, lng: 0 },
        estimatedCreditsPerYear: 100,
        verifier: 'verra',
      });
      service.verifyProject('proj-vy');

      const batch = service.issueCredits('proj-vy', 100, 'tCO2e');
      expect(batch.vintage).toBe(new Date().getFullYear());
    });

    it('should accept a custom vintage year', () => {
      service.registerProject({
        id: 'proj-cv',
        name: 'Custom Vintage',
        type: 'reforestation',
        location: { lat: 0, lng: 0 },
        estimatedCreditsPerYear: 100,
        verifier: 'verra',
      });
      service.verifyProject('proj-cv');

      const batch = service.issueCredits('proj-cv', 50, 'tCO2e', 2023);
      expect(batch.vintage).toBe(2023);
    });

    it('should throw for unverified project', () => {
      service.registerProject({
        id: 'proj-uv',
        name: 'Unverified',
        type: 'reforestation',
        location: { lat: 0, lng: 0 },
        estimatedCreditsPerYear: 100,
        verifier: 'verra',
      });

      expect(() => service.issueCredits('proj-uv', 100, 'tCO2e')).toThrow(
        'Project not verified',
      );
    });

    it('should throw for non-existent project', () => {
      expect(() => service.issueCredits('proj-none', 100, 'tCO2e')).toThrow(
        'Project not found',
      );
    });
  });

  describe('retireCredits', () => {
    it('should retire active credits', () => {
      service.registerProject({
        id: 'proj-ret',
        name: 'Retire Test',
        type: 'direct_air_capture',
        location: { lat: 0, lng: 0 },
        estimatedCreditsPerYear: 100,
        verifier: 'verra',
      });
      service.verifyProject('proj-ret');
      const batch = service.issueCredits('proj-ret', 200, 'tCO2e');

      const retired = service.retireCredits(batch.id, 'company-abc', 'Annual offset');

      expect(retired.status).toBe('retired');
      expect(retired.retiredBy).toBe('company-abc');
      expect(retired.retirementReason).toBe('Annual offset');
      expect(retired.retiredAt).toBeGreaterThan(0);
    });

    it('should throw when retiring non-existent batch', () => {
      expect(() => service.retireCredits('fake-batch', 'buyer')).toThrow(
        'Credit batch not found',
      );
    });

    it('should throw when retiring already-retired credits', () => {
      service.registerProject({
        id: 'proj-rr',
        name: 'Double Retire',
        type: 'reforestation',
        location: { lat: 0, lng: 0 },
        estimatedCreditsPerYear: 100,
        verifier: 'verra',
      });
      service.verifyProject('proj-rr');
      const batch = service.issueCredits('proj-rr', 100, 'tCO2e');
      service.retireCredits(batch.id, 'buyer-1');

      expect(() => service.retireCredits(batch.id, 'buyer-2')).toThrow(
        'Credit batch not active',
      );
    });
  });

  describe('issueREC', () => {
    it('should issue a Renewable Energy Certificate', () => {
      const rec = service.issueREC({
        projectId: 'proj-rec',
        amountMwh: 100,
        generationPeriodStart: Date.now() - 30 * 24 * 60 * 60 * 1000,
        generationPeriodEnd: Date.now(),
        source: 'solar',
      });

      expect(rec).toBeDefined();
      expect(rec.id).toMatch(/^rec-/);
      expect(rec.amountMwh).toBe(100);
      expect(rec.source).toBe('solar');
      expect(rec.status).toBe('active');
    });
  });

  describe('redeemREC', () => {
    it('should redeem an active REC', () => {
      const rec = service.issueREC({
        projectId: 'proj-redeem',
        amountMwh: 50,
        generationPeriodStart: Date.now() - 86400000,
        generationPeriodEnd: Date.now(),
        source: 'wind',
      });

      const redeemed = service.redeemREC(rec.id, 'company-xyz');

      expect(redeemed.status).toBe('redeemed');
      expect(redeemed.redeemedBy).toBe('company-xyz');
      expect(redeemed.redeemedAt).toBeGreaterThan(0);
    });

    it('should throw when redeeming non-existent REC', () => {
      expect(() => service.redeemREC('fake-rec', 'buyer')).toThrow('REC not found');
    });

    it('should throw when redeeming already-redeemed REC', () => {
      const rec = service.issueREC({
        projectId: 'proj-rr2',
        amountMwh: 25,
        generationPeriodStart: Date.now() - 86400000,
        generationPeriodEnd: Date.now(),
        source: 'hydro',
      });
      service.redeemREC(rec.id, 'buyer-1');

      expect(() => service.redeemREC(rec.id, 'buyer-2')).toThrow('REC not active');
    });
  });

  describe('getProject / getAllProjects', () => {
    it('should retrieve a project by ID', () => {
      service.registerProject({
        id: 'proj-get',
        name: 'Get Test',
        type: 'reforestation',
        location: { lat: 0, lng: 0 },
        estimatedCreditsPerYear: 100,
        verifier: 'verra',
      });

      expect(service.getProject('proj-get')).toBeDefined();
      expect(service.getProject('proj-get')!.name).toBe('Get Test');
    });

    it('should return undefined for unknown project', () => {
      expect(service.getProject('nonexistent')).toBeUndefined();
    });

    it('should return all projects', () => {
      service.registerProject({
        id: 'p1',
        name: 'P1',
        type: 'reforestation',
        location: { lat: 0, lng: 0 },
        estimatedCreditsPerYear: 100,
        verifier: 'verra',
      });
      service.registerProject({
        id: 'p2',
        name: 'P2',
        type: 'blue_carbon',
        location: { lat: 1, lng: 1 },
        estimatedCreditsPerYear: 200,
        verifier: 'gold_standard',
      });

      expect(service.getAllProjects()).toHaveLength(2);
    });
  });

  describe('getActiveCredits', () => {
    it('should return only active credits for a project', () => {
      service.registerProject({
        id: 'proj-ac',
        name: 'Active Credits',
        type: 'reforestation',
        location: { lat: 0, lng: 0 },
        estimatedCreditsPerYear: 1000,
        verifier: 'verra',
      });
      service.verifyProject('proj-ac');

      const b1 = service.issueCredits('proj-ac', 100, 'tCO2e');
      const b2 = service.issueCredits('proj-ac', 200, 'tCO2e');
      service.retireCredits(b1.id, 'buyer');

      const active = service.getActiveCredits('proj-ac');
      expect(active).toHaveLength(1);
      expect(active[0].id).toBe(b2.id);
    });
  });

  describe('getTotalRetired', () => {
    it('should sum all retired tCO2e credits', () => {
      service.registerProject({
        id: 'proj-tr',
        name: 'Total Retired',
        type: 'methane_capture',
        location: { lat: 0, lng: 0 },
        estimatedCreditsPerYear: 1000,
        verifier: 'verra',
      });
      service.verifyProject('proj-tr');

      const b1 = service.issueCredits('proj-tr', 300, 'tCO2e');
      const b2 = service.issueCredits('proj-tr', 200, 'tCO2e');
      service.retireCredits(b1.id, 'buyer');
      service.retireCredits(b2.id, 'buyer');

      expect(service.getTotalRetired()).toBe(500);
    });

    it('should return 0 when no credits have been retired', () => {
      expect(service.getTotalRetired()).toBe(0);
    });
  });

  describe('getClimateService', () => {
    it('should return singleton instance', () => {
      const s1 = getClimateService();
      const s2 = getClimateService();
      expect(s1).toBe(s2);
    });

    it('should return a fresh instance after reset', () => {
      const s1 = getClimateService();
      resetClimateService();
      const s2 = getClimateService();
      expect(s1).not.toBe(s2);
    });
  });
});
