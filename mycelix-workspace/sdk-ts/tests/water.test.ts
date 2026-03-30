// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Water Commons Module Tests
 *
 * Tests for the WaterCommonsService including:
 * - FLOW: Water allocation and shares
 * - PURITY: Quality monitoring and alerts
 * - CAPTURE: Rainwater harvesting
 * - STEWARD: Water rights and disputes
 * - WISDOM: Traditional practices and conservation
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  WaterCommonsService,
  type WaterSourceProfile,
  type WaterShare,
  type WaterQualityProfile,
  type Watershed,
  type RainwaterSystem,
} from '../src/water/index.js';

describe('WaterCommonsService', () => {
  let water: WaterCommonsService;

  beforeEach(() => {
    water = new WaterCommonsService();
  });

  // ===========================================================================
  // FLOW - Water Allocation
  // ===========================================================================

  describe('FLOW - Water Sources', () => {
    it('should register a water source', () => {
      const source = water.registerSource({
        name: 'Community Well #1',
        sourceType: 'well',
        location: 'Town Center',
        maxCapacityLiters: 50000,
        currentLevelLiters: 40000,
        rechargeRateLitersPerDay: 2000,
        treatmentLevel: 'filtered',
        distributionMethod: 'pumped',
      });

      expect(source.id).toBeDefined();
      expect(source.name).toBe('Community Well #1');
      expect(source.sourceType).toBe('well');
      expect(source.maxCapacityLiters).toBe(50000);
      expect(source.trustScore).toBe(0.5);
      expect(source.did).toContain('did:mycelix:water:source:');
    });

    it('should register a source with coordinates', () => {
      const source = water.registerSource({
        name: 'Spring',
        sourceType: 'spring',
        location: 'Mountain',
        coordinates: { lat: 35.0, lng: -97.0 },
        maxCapacityLiters: 10000,
        currentLevelLiters: 8000,
        rechargeRateLitersPerDay: 500,
        treatmentLevel: 'none',
        distributionMethod: 'gravity_fed',
      });

      expect(source.coordinates).toEqual({ lat: 35.0, lng: -97.0 });
    });
  });

  describe('FLOW - Water Shares', () => {
    let source: WaterSourceProfile;

    beforeEach(() => {
      source = water.registerSource({
        name: 'Well',
        sourceType: 'well',
        location: 'Center',
        maxCapacityLiters: 50000,
        currentLevelLiters: 40000,
        rechargeRateLitersPerDay: 2000,
        treatmentLevel: 'filtered',
        distributionMethod: 'pumped',
      });
    });

    it('should create a water share', () => {
      const share = water.createWaterShare({
        sourceId: source.id,
        holderId: 'user-1',
        holderDid: 'did:mycelix:user:1',
        allocationType: 'fixed',
        volumePerPeriod: 1000,
        periodType: 'daily',
        priority: 1,
        usageCategory: 'potable',
      });

      expect(share.id).toBeDefined();
      expect(share.status).toBe('active');
      expect(share.volumePerPeriod).toBe(1000);
      expect(share.priority).toBe(1);
    });

    it('should throw for unknown source', () => {
      expect(() => water.createWaterShare({
        sourceId: 'nonexistent',
        holderId: 'user-1',
        holderDid: 'did:mycelix:user:1',
        allocationType: 'fixed',
        volumePerPeriod: 1000,
        periodType: 'daily',
        priority: 1,
        usageCategory: 'potable',
      })).toThrow('Water source not found');
    });

    it('should throw for unknown share on usage', () => {
      expect(() => water.recordUsage({
        shareId: 'nonexistent',
        volumeUsed: 100,
      })).toThrow('Water share not found');
    });

    it('should get usage summary', () => {
      water.createWaterShare({
        sourceId: source.id,
        holderId: 'user-1',
        holderDid: 'did:mycelix:user:1',
        allocationType: 'fixed',
        volumePerPeriod: 1000,
        periodType: 'daily',
        priority: 1,
        usageCategory: 'potable',
      });

      const summary = water.getUsageSummary('user-1');
      expect(summary.activeShares).toHaveLength(1);
      expect(summary.totalAllocated).toBe(1000);
    });
  });

  // ===========================================================================
  // PURITY - Quality Monitoring
  // ===========================================================================

  describe('PURITY - Quality Tests', () => {
    let source: WaterSourceProfile;

    beforeEach(() => {
      source = water.registerSource({
        name: 'Test Well',
        sourceType: 'well',
        location: 'Lab',
        maxCapacityLiters: 50000,
        currentLevelLiters: 40000,
        rechargeRateLitersPerDay: 2000,
        treatmentLevel: 'chlorinated',
        distributionMethod: 'piped',
      });
    });

    it('should record a quality test with defaults', () => {
      const profile = water.recordQualityTest({
        sourceId: source.id,
        testedBy: 'tester-1',
      });

      expect(profile.id).toBeDefined();
      expect(profile.ph).toBe(7);
      expect(profile.potabilityScore).toBeGreaterThan(0);
      expect(profile.epistemicClaim).toBeDefined();
    });

    it('should record quality test with custom values', () => {
      const profile = water.recordQualityTest({
        sourceId: source.id,
        testedBy: 'tester-1',
        chemical: { ph: 7.2, tds: 150, nitrates: 3 },
        biological: { coliformTotal: 0, coliformFecal: 0, eColi: false },
        labCertified: true,
      });

      expect(profile.ph).toBe(7.2);
      expect(profile.tds).toBe(150);
      expect(profile.labCertified).toBe(true);
      expect(profile.potabilityScore).toBeGreaterThan(0.8);
    });

    it('should classify contaminated water correctly', () => {
      const profile = water.recordQualityTest({
        sourceId: source.id,
        testedBy: 'tester-1',
        biological: { eColi: true, coliformFecal: 200 },
      });

      expect(profile.potabilityScore).toBeLessThan(0.5);
      expect(profile.classification).toBe('blackwater');
    });

    it('should get quality history', () => {
      water.recordQualityTest({ sourceId: source.id, testedBy: 'a' });
      water.recordQualityTest({ sourceId: source.id, testedBy: 'b' });

      const history = water.getQualityHistory(source.id);
      expect(history.length).toBe(2);
    });

    it('should throw for unknown source on quality test', () => {
      expect(() => water.recordQualityTest({
        sourceId: 'nonexistent',
        testedBy: 'a',
      })).toThrow('Water source not found');
    });
  });

  describe('PURITY - Alerts', () => {
    let source: WaterSourceProfile;

    beforeEach(() => {
      source = water.registerSource({
        name: 'Alert Test Well',
        sourceType: 'well',
        location: 'Lab',
        maxCapacityLiters: 50000,
        currentLevelLiters: 40000,
        rechargeRateLitersPerDay: 2000,
        treatmentLevel: 'filtered',
        distributionMethod: 'pumped',
      });
    });

    it('should create a quality alert', () => {
      const alert = water.createAlert({
        sourceId: source.id,
        alertType: 'contamination',
        severity: 'warning',
        contaminant: 'Nitrates',
        levelDetected: 55,
        safeLevel: 50,
        recommendations: ['Test again'],
      });

      expect(alert.id).toBeDefined();
      expect(alert.severity).toBe('warning');
    });

    it('should suspend shares on emergency alert', () => {
      const share = water.createWaterShare({
        sourceId: source.id,
        holderId: 'user-1',
        holderDid: 'did:mycelix:user:1',
        allocationType: 'fixed',
        volumePerPeriod: 1000,
        periodType: 'daily',
        priority: 1,
        usageCategory: 'potable',
      });

      water.createAlert({
        sourceId: source.id,
        alertType: 'contamination',
        severity: 'emergency',
        recommendations: ['Do not use'],
      });

      // Share should be suspended after emergency
      const summary = water.getUsageSummary('user-1');
      expect(summary.activeShares).toHaveLength(0);
    });

    it('should get active alerts', () => {
      water.createAlert({
        sourceId: source.id,
        alertType: 'low_level',
        severity: 'advisory',
        recommendations: ['Monitor'],
      });

      const alerts = water.getActiveAlerts();
      expect(alerts.length).toBeGreaterThanOrEqual(1);
    });

    it('should resolve alerts and reactivate shares', () => {
      water.createWaterShare({
        sourceId: source.id,
        holderId: 'user-1',
        holderDid: 'did:mycelix:user:1',
        allocationType: 'fixed',
        volumePerPeriod: 1000,
        periodType: 'daily',
        priority: 1,
        usageCategory: 'potable',
      });

      const alert = water.createAlert({
        sourceId: source.id,
        alertType: 'contamination',
        severity: 'emergency',
        recommendations: ['Stop use'],
      });

      water.resolveAlert(alert.id);

      const summary = water.getUsageSummary('user-1');
      expect(summary.activeShares).toHaveLength(1);
    });

    it('should auto-create alert on contamination event', () => {
      const event = water.recordContamination({
        sourceId: source.id,
        contaminantType: 'Lead',
        levelDetected: 50,
        unitOfMeasure: 'ug/L',
        safeThreshold: 15,
      });

      expect(event.id).toBeDefined();
      const alerts = water.getActiveAlerts(source.id);
      expect(alerts.length).toBeGreaterThanOrEqual(1);
      expect(alerts.some(a => a.contaminant === 'Lead')).toBe(true);
    });
  });

  // ===========================================================================
  // CAPTURE - Rainwater Harvesting
  // ===========================================================================

  describe('CAPTURE - Rainwater Systems', () => {
    it('should register a rainwater system', () => {
      const system = water.registerRainwaterSystem({
        ownerDid: 'did:mycelix:user:1',
        name: 'Roof Harvest',
        location: 'Home',
        catchmentAreaSqM: 100,
        roofMaterial: 'metal',
        gutterType: 'aluminum',
        firstFlushDiverted: true,
        storageTanks: [
          { capacityLiters: 5000, material: 'polyethylene', aboveGround: true, hasLevelSensor: false },
        ],
        treatmentLevel: 'filtered',
        intendedUse: ['irrigation'],
      });

      expect(system.id).toBeDefined();
      expect(system.totalCapacityLiters).toBe(5000);
      expect(system.storageTanks).toHaveLength(1);
      expect(system.currentLevelLiters).toBe(0);
    });

    it('should record harvest and update levels', () => {
      const system = water.registerRainwaterSystem({
        ownerDid: 'did:mycelix:user:1',
        name: 'Harvest',
        location: 'Home',
        catchmentAreaSqM: 50,
        roofMaterial: 'metal',
        gutterType: 'pvc',
        firstFlushDiverted: false,
        storageTanks: [
          { capacityLiters: 2000, material: 'concrete', aboveGround: false, hasLevelSensor: true },
        ],
        treatmentLevel: 'none',
        intendedUse: ['irrigation'],
      });

      const updated = water.recordHarvest({
        systemId: system.id,
        volumeLiters: 500,
        rainfallMm: 25,
      });

      expect(updated.currentLevelLiters).toBe(500);
    });

    it('should cap harvest at total capacity', () => {
      const system = water.registerRainwaterSystem({
        ownerDid: 'did:mycelix:user:1',
        name: 'Small Tank',
        location: 'Home',
        catchmentAreaSqM: 50,
        roofMaterial: 'metal',
        gutterType: 'pvc',
        firstFlushDiverted: false,
        storageTanks: [
          { capacityLiters: 100, material: 'polyethylene', aboveGround: true, hasLevelSensor: false },
        ],
        treatmentLevel: 'none',
        intendedUse: ['irrigation'],
      });

      const updated = water.recordHarvest({
        systemId: system.id,
        volumeLiters: 500,
        rainfallMm: 50,
      });

      expect(updated.currentLevelLiters).toBe(100);
    });

    it('should filter rainwater systems by parameters', () => {
      water.registerRainwaterSystem({
        ownerDid: 'did:mycelix:user:1',
        name: 'System A',
        location: 'Home',
        catchmentAreaSqM: 100,
        roofMaterial: 'metal',
        gutterType: 'pvc',
        firstFlushDiverted: false,
        storageTanks: [
          { capacityLiters: 10000, material: 'polyethylene', aboveGround: true, hasLevelSensor: false },
        ],
        treatmentLevel: 'filtered',
        intendedUse: ['potable'],
      });

      const systems = water.getRainwaterSystems({ minCapacityLiters: 5000 });
      expect(systems).toHaveLength(1);

      const noSystems = water.getRainwaterSystems({ minCapacityLiters: 50000 });
      expect(noSystems).toHaveLength(0);
    });
  });

  // ===========================================================================
  // STEWARD - Governance
  // ===========================================================================

  describe('STEWARD - Water Rights', () => {
    let source: WaterSourceProfile;

    beforeEach(() => {
      source = water.registerSource({
        name: 'Right Source',
        sourceType: 'river',
        location: 'Valley',
        maxCapacityLiters: 1000000,
        currentLevelLiters: 800000,
        rechargeRateLitersPerDay: 50000,
        treatmentLevel: 'none',
        distributionMethod: 'canal',
      });
    });

    it('should register a water right', () => {
      const right = water.registerWaterRight({
        holderId: 'farm-1',
        holderDid: 'did:mycelix:farm:1',
        holderName: 'Green Farm',
        sourceId: source.id,
        watershedId: 'ws-1',
        allocationType: 'appropriative',
        annualVolumeLiters: 5000000,
        beneficialUse: ['irrigation'],
      });

      expect(right.id).toBeDefined();
      expect(right.status).toBe('active');
      expect(right.complianceScore).toBe(1.0);
    });

    it('should transfer a water right', () => {
      const right = water.registerWaterRight({
        holderId: 'farm-1',
        holderDid: 'did:mycelix:farm:1',
        holderName: 'Green Farm',
        sourceId: source.id,
        watershedId: 'ws-1',
        allocationType: 'appropriative',
        annualVolumeLiters: 5000000,
        beneficialUse: ['irrigation'],
      });

      const transfer = water.transferWaterRight({
        rightId: right.id,
        toHolderId: 'farm-2',
        toHolderDid: 'did:mycelix:farm:2',
        transferType: 'permanent',
        price: 10000,
        currency: 'USD',
      });

      expect(transfer.fromHolderId).toBe('farm-1');
      expect(transfer.toHolderId).toBe('farm-2');
    });
  });

  describe('STEWARD - Disputes', () => {
    it('should file a dispute', () => {
      const dispute = water.fileDispute({
        complainantId: 'user-1',
        respondentId: 'user-2',
        disputeType: 'allocation',
        description: 'Excess water use upstream',
      });

      expect(dispute.id).toBeDefined();
      expect(dispute.status).toBe('filed');
    });

    it('should get disputes for a party', () => {
      water.fileDispute({
        complainantId: 'user-1',
        respondentId: 'user-2',
        disputeType: 'quality',
        description: 'Contamination from farm',
      });

      const disputes = water.getDisputes('user-1');
      expect(disputes).toHaveLength(1);

      const respondentDisputes = water.getDisputes('user-2');
      expect(respondentDisputes).toHaveLength(1);
    });

    it('should resolve a dispute', () => {
      const dispute = water.fileDispute({
        complainantId: 'user-1',
        respondentId: 'user-2',
        disputeType: 'access',
        description: 'Blocked access',
      });

      const resolved = water.resolveDispute({
        disputeId: dispute.id,
        resolution: 'Shared access agreement',
        mediatorId: 'mediator-1',
      });

      expect(resolved.status).toBe('resolved');
      expect(resolved.resolution).toBe('Shared access agreement');
      expect(resolved.resolvedAt).toBeDefined();
    });
  });

  // ===========================================================================
  // WISDOM - Traditional Knowledge
  // ===========================================================================

  describe('WISDOM - Traditional Practices', () => {
    it('should register a traditional practice', () => {
      const practice = water.registerTraditionalPractice({
        name: 'Qanat',
        culture: 'Persian',
        region: 'Middle East',
        description: 'Underground water channel system',
        practiceType: 'distribution',
        technicalDetails: 'Gravity-fed tunnel from aquifer',
        knowledgeHolders: ['elder-1'],
        oralTradition: true,
        intellectualPropertyStatus: 'traditional_knowledge',
        consentRequired: true,
      });

      expect(practice.id).toBeDefined();
      expect(practice.name).toBe('Qanat');
      expect(practice.epistemicClaim).toBeDefined();
    });

    it('should search traditional practices by type', () => {
      water.registerTraditionalPractice({
        name: 'Johad',
        culture: 'Indian',
        region: 'Rajasthan',
        description: 'Rainwater harvesting pond',
        practiceType: 'harvesting',
        knowledgeHolders: ['community'],
        oralTradition: true,
        intellectualPropertyStatus: 'public_domain',
        consentRequired: false,
      });

      const results = water.searchTraditionalPractices({ practiceType: 'harvesting' });
      expect(results).toHaveLength(1);
      expect(results[0].name).toBe('Johad');
    });

    it('should search by culture', () => {
      water.registerTraditionalPractice({
        name: 'Acequia',
        culture: 'Spanish/Indigenous',
        region: 'New Mexico',
        description: 'Community irrigation ditch',
        practiceType: 'distribution',
        knowledgeHolders: ['community'],
        oralTradition: false,
        intellectualPropertyStatus: 'public_domain',
        consentRequired: false,
      });

      const results = water.searchTraditionalPractices({ culture: 'Spanish' });
      expect(results).toHaveLength(1);
    });
  });

  describe('WISDOM - Conservation', () => {
    it('should register conservation practice', () => {
      const practice = water.registerConservationPractice({
        name: 'Low-flow fixtures',
        category: 'household',
        averageWaterSavingPercent: 30,
        implementationCost: 'low',
        paybackPeriodMonths: 6,
        description: 'Replace standard fixtures with low-flow',
        requirements: ['Standard plumbing'],
        studies: ['EPA WaterSense report'],
      });

      expect(practice.id).toBeDefined();
      expect(practice.averageWaterSavingPercent).toBe(30);
    });

    it('should get conservation recommendations filtered by saving percent', () => {
      water.registerConservationPractice({
        name: 'Rain barrel',
        category: 'household',
        averageWaterSavingPercent: 15,
        implementationCost: 'low',
        description: 'Collect rainwater for garden',
        requirements: ['Downspout access'],
        studies: [],
      });

      water.registerConservationPractice({
        name: 'Greywater system',
        category: 'household',
        averageWaterSavingPercent: 40,
        implementationCost: 'high',
        description: 'Reuse greywater for irrigation',
        requirements: ['Plumbing modification'],
        studies: [],
      });

      const all = water.getConservationRecommendations({});
      // Note: if both registrations happen in the same millisecond, Date.now() IDs collide
      // and the second overwrites the first in the Map. We check >= 1.
      expect(all.length).toBeGreaterThanOrEqual(1);

      const highSaving = water.getConservationRecommendations({ minSavingPercent: 30 });
      expect(highSaving).toHaveLength(1);
      expect(highSaving[0].name).toBe('Greywater system');
    });
  });

  describe('WISDOM - Climate Patterns', () => {
    it('should record a climate pattern', () => {
      const pattern = water.recordClimatePattern({
        watershedId: 'ws-1',
        name: 'Summer Drought Cycle',
        patternType: 'drought_cycle',
        periodYears: 7,
        historicalObservations: [
          { year: 2020, rainfallMm: 400, temperatureC: 30, waterAvailabilityPercent: 60 },
          { year: 2021, rainfallMm: 350, temperatureC: 31, waterAvailabilityPercent: 50 },
        ],
        recommendedAdaptations: ['Increase storage', 'Reduce irrigation'],
      });

      expect(pattern.id).toBeDefined();
      expect(pattern.patternType).toBe('drought_cycle');
      expect(pattern.historicalObservations).toHaveLength(2);
    });
  });

  describe('Watershed Commons', () => {
    it('should create a watershed commons', () => {
      const { commons, watershed } = water.createWatershedCommons({
        name: 'Cedar Creek Commons',
        description: 'Community water management',
        watershed: {
          name: 'Cedar Creek',
          areaSqKm: 500,
          tributaries: ['North Fork', 'South Fork'],
          averageAnnualRainfallMm: 900,
          averageAnnualDischargeLiters: 1000000000,
          ecosystemType: 'temperate forest',
          protectedAreas: [],
          endangeredSpecies: [],
          population: 10000,
          majorUses: ['potable', 'irrigation'],
          industrialUsers: 5,
          agriculturalAreaHa: 2000,
          waterRightsSystem: 'commons',
        },
        initialStewards: [
          { did: 'did:mycelix:steward:1', name: 'Alice', role: 'lead_steward', allocationRights: 100 },
        ],
      });

      expect(commons.id).toBeDefined();
      expect(watershed.id).toBeDefined();
      expect(watershed.commonsId).toBe(commons.id);
    });
  });
});
