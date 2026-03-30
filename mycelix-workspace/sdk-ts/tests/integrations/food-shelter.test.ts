// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Food-Shelter Integration Tests
 *
 * Tests for FoodShelterBridge - household management, resource allocation,
 * security assessment, cross-domain reputation, and utility functions
 * (isInCrisis, householdSize, qualifiesForEmergencyAssistance).
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  FoodShelterBridge,
  createFoodShelterBridge,
  isInCrisis,
  householdSize,
  qualifiesForEmergencyAssistance,
  type Household,
  type HouseholdSecurityAssessment,
  type CreateHouseholdInput,
} from '../../src/integrations/food-shelter/index.js';

/** Helper: create a basic household input */
function makeHouseholdInput(overrides: Partial<CreateHouseholdInput> = {}): CreateHouseholdInput {
  return {
    members: [new Uint8Array(39), new Uint8Array(39)],
    headOfHousehold: new Uint8Array(39),
    type: 'Family',
    dependentCount: 2,
    monthlyIncome: 5000,
    ...overrides,
  };
}

describe('Food-Shelter Integration', () => {
  let bridge: FoodShelterBridge;

  beforeEach(() => {
    bridge = new FoodShelterBridge();
  });

  describe('createHousehold', () => {
    it('should create a household with default security statuses', async () => {
      const household = await bridge.createHousehold(makeHouseholdInput());

      expect(household).toBeDefined();
      expect(household.householdId).toMatch(/^hh-/);
      expect(household.type).toBe('Family');
      expect(household.dependentCount).toBe(2);
      expect(household.foodSecurityStatus).toBe('HighFoodSecurity');
      expect(household.housingSecurityStatus).toBe('StableHousing');
      expect(household.createdAt).toBeGreaterThan(0);
    });

    it('should respect provided security statuses', async () => {
      const household = await bridge.createHousehold(
        makeHouseholdInput({
          foodSecurityStatus: 'LowFoodSecurity',
          housingSecurityStatus: 'AtRisk',
        }),
      );

      expect(household.foodSecurityStatus).toBe('LowFoodSecurity');
      expect(household.housingSecurityStatus).toBe('AtRisk');
    });

    it('should default dependentCount to 0 when not specified', async () => {
      const household = await bridge.createHousehold({
        members: [new Uint8Array(39)],
        headOfHousehold: new Uint8Array(39),
        type: 'Single',
      });

      expect(household.dependentCount).toBe(0);
    });
  });

  describe('getHousehold', () => {
    it('should retrieve an existing household', async () => {
      const created = await bridge.createHousehold(makeHouseholdInput());
      const retrieved = await bridge.getHousehold(created.householdId);

      expect(retrieved).toBeDefined();
      expect(retrieved!.householdId).toBe(created.householdId);
    });

    it('should return null for non-existent household', async () => {
      const result = await bridge.getHousehold('hh-nonexistent');
      expect(result).toBeNull();
    });
  });

  describe('updateSecurityStatus', () => {
    it('should update food security status', async () => {
      const household = await bridge.createHousehold(makeHouseholdInput());

      const updated = await bridge.updateSecurityStatus(
        household.householdId,
        'VeryLowFoodSecurity',
      );

      expect(updated).toBeDefined();
      expect(updated!.foodSecurityStatus).toBe('VeryLowFoodSecurity');
      expect(updated!.housingSecurityStatus).toBe('StableHousing');
    });

    it('should update housing security status', async () => {
      const household = await bridge.createHousehold(makeHouseholdInput());

      const updated = await bridge.updateSecurityStatus(
        household.householdId,
        undefined,
        'Homeless',
      );

      expect(updated).toBeDefined();
      expect(updated!.housingSecurityStatus).toBe('Homeless');
      expect(updated!.foodSecurityStatus).toBe('HighFoodSecurity');
    });

    it('should update both statuses simultaneously', async () => {
      const household = await bridge.createHousehold(makeHouseholdInput());

      const updated = await bridge.updateSecurityStatus(
        household.householdId,
        'LowFoodSecurity',
        'Unstable',
      );

      expect(updated!.foodSecurityStatus).toBe('LowFoodSecurity');
      expect(updated!.housingSecurityStatus).toBe('Unstable');
    });

    it('should return null for non-existent household', async () => {
      const result = await bridge.updateSecurityStatus('hh-fake', 'LowFoodSecurity');
      expect(result).toBeNull();
    });
  });

  describe('calculateResourceAllocation', () => {
    it('should calculate allocation for a family with dependents', async () => {
      const household = await bridge.createHousehold(makeHouseholdInput());
      const allocation = bridge.calculateResourceAllocation(household, 6000);

      expect(allocation.householdId).toBe(household.householdId);
      expect(allocation.totalBudget).toBe(6000);
      expect(allocation.allocations.Food).toBeDefined();
      expect(allocation.allocations.Shelter).toBeDefined();
      expect(allocation.allocations.Food.budgeted).toBeGreaterThan(0);
      expect(allocation.allocations.Shelter.budgeted).toBeGreaterThan(0);
    });

    it('should flag below-recommended allocations with warnings', async () => {
      const household = await bridge.createHousehold(makeHouseholdInput());
      // Very low budget should trigger warnings for a family of 4
      const allocation = bridge.calculateResourceAllocation(household, 1000);

      expect(allocation.warnings.length).toBeGreaterThan(0);
      const criticalWarnings = allocation.warnings.filter((w) => w.severity === 'Critical');
      expect(criticalWarnings.length).toBeGreaterThan(0);
    });

    it('should allocate more for food in food-insecure households', async () => {
      const secure = await bridge.createHousehold(makeHouseholdInput());
      const insecure = await bridge.createHousehold(
        makeHouseholdInput({ foodSecurityStatus: 'VeryLowFoodSecurity' }),
      );

      const secureAlloc = bridge.calculateResourceAllocation(secure, 6000);
      const insecureAlloc = bridge.calculateResourceAllocation(insecure, 6000);

      expect(insecureAlloc.allocations.Food.percentOfBudget).toBeGreaterThan(
        secureAlloc.allocations.Food.percentOfBudget,
      );
    });

    it('should allocate childcare for households with dependents', async () => {
      const withDeps = await bridge.createHousehold(
        makeHouseholdInput({ dependentCount: 2 }),
      );
      const noDeps = await bridge.createHousehold(
        makeHouseholdInput({ dependentCount: 0 }),
      );

      const withDepsAlloc = bridge.calculateResourceAllocation(withDeps, 6000);
      const noDepsAlloc = bridge.calculateResourceAllocation(noDeps, 6000);

      expect(withDepsAlloc.allocations.Childcare.budgeted).toBeGreaterThan(0);
      expect(noDepsAlloc.allocations.Childcare.budgeted).toBe(0);
    });
  });

  describe('assessHouseholdSecurity', () => {
    it('should return high scores for a secure household', async () => {
      // monthlyIncome must exceed 2000 * householdSize to avoid income risk factor
      // householdSize = 2 members + 2 dependents = 4, so need > 8000
      const household = await bridge.createHousehold(
        makeHouseholdInput({ monthlyIncome: 10000 }),
      );
      const assessment = bridge.assessHouseholdSecurity(household);

      expect(assessment.foodSecurityScore).toBe(100);
      expect(assessment.housingSecurityScore).toBe(100);
      expect(assessment.combinedSecurityIndex).toBe(100);
      expect(assessment.riskFactors).toHaveLength(0);
    });

    it('should identify food insecurity risk factor', async () => {
      const household = await bridge.createHousehold(
        makeHouseholdInput({ foodSecurityStatus: 'VeryLowFoodSecurity' }),
      );
      const assessment = bridge.assessHouseholdSecurity(household);

      expect(assessment.foodSecurityScore).toBe(10);
      const foodFactors = assessment.riskFactors.filter((f) => f.factorType === 'Food');
      expect(foodFactors.length).toBeGreaterThan(0);
      expect(foodFactors[0].severity).toBe('Critical');
    });

    it('should identify housing insecurity risk factor', async () => {
      const household = await bridge.createHousehold(
        makeHouseholdInput({ housingSecurityStatus: 'Homeless' }),
      );
      const assessment = bridge.assessHouseholdSecurity(household);

      expect(assessment.housingSecurityScore).toBe(5);
      const shelterFactors = assessment.riskFactors.filter(
        (f) => f.factorType === 'Shelter',
      );
      expect(shelterFactors.length).toBeGreaterThan(0);
      expect(shelterFactors[0].severity).toBe('Critical');
    });

    it('should suggest SNAP for food-insecure households', async () => {
      const household = await bridge.createHousehold(
        makeHouseholdInput({ foodSecurityStatus: 'LowFoodSecurity' }),
      );
      const assessment = bridge.assessHouseholdSecurity(household);

      const snap = assessment.availablePrograms.find((p) => p.programId === 'snap');
      expect(snap).toBeDefined();
      expect(snap!.category).toBe('FoodAssistance');
    });

    it('should suggest emergency assistance for severely insecure households', async () => {
      const household = await bridge.createHousehold(
        makeHouseholdInput({
          foodSecurityStatus: 'VeryLowFoodSecurity',
          housingSecurityStatus: 'Homeless',
        }),
      );
      const assessment = bridge.assessHouseholdSecurity(household);

      const emergency = assessment.availablePrograms.find(
        (p) => p.programId === 'emergency',
      );
      expect(emergency).toBeDefined();
    });

    it('should include income risk factor for low-income households', async () => {
      const household = await bridge.createHousehold(
        makeHouseholdInput({ monthlyIncome: 1000 }),
      );
      const assessment = bridge.assessHouseholdSecurity(household);

      const incomeFactors = assessment.riskFactors.filter(
        (f) => f.factorType === 'Income',
      );
      expect(incomeFactors.length).toBeGreaterThan(0);
    });
  });

  describe('getCrossReputationion', () => {
    it('should return default reputation for new agent', async () => {
      const agentKey = new Uint8Array(39);
      const reputation = await bridge.getCrossReputationion(agentKey);

      expect(reputation).toBeDefined();
      expect(reputation.foodReputation.domain).toBe('Food');
      expect(reputation.shelterReputation.domain).toBe('Shelter');
      expect(reputation.combinedScore).toBeGreaterThan(0);
    });
  });

  describe('recordContribution', () => {
    it('should record a food domain contribution', async () => {
      const agentKey = new Uint8Array(39);
      await bridge.recordContribution(agentKey, 'Food', {
        type: 'FoodDonation',
        description: 'Donated 50 lbs of food',
        impactScore: 0.8,
        timestamp: Date.now(),
      });

      const reputation = await bridge.getCrossReputationion(agentKey);
      expect(reputation.foodReputation.contributions).toHaveLength(1);
      expect(reputation.foodReputation.interactionCount).toBe(1);
    });

    it('should record a shelter domain contribution', async () => {
      const agentKey = new Uint8Array(39);
      await bridge.recordContribution(agentKey, 'Shelter', {
        type: 'HousingVolunteer',
        description: 'Helped build shelter',
        impactScore: 0.9,
        timestamp: Date.now(),
      });

      const reputation = await bridge.getCrossReputationion(agentKey);
      expect(reputation.shelterReputation.contributions).toHaveLength(1);
    });
  });

  describe('isInCrisis (utility)', () => {
    it('should return true when both food and housing are insecure', async () => {
      const household = await bridge.createHousehold(
        makeHouseholdInput({
          foodSecurityStatus: 'VeryLowFoodSecurity',
          housingSecurityStatus: 'Homeless',
        }),
      );
      expect(isInCrisis(household)).toBe(true);
    });

    it('should return false for high food security even with homeless status', async () => {
      const household = await bridge.createHousehold(
        makeHouseholdInput({
          foodSecurityStatus: 'HighFoodSecurity',
          housingSecurityStatus: 'Homeless',
        }),
      );
      expect(isInCrisis(household)).toBe(false);
    });

    it('should return false for a stable household', async () => {
      const household = await bridge.createHousehold(makeHouseholdInput());
      expect(isInCrisis(household)).toBe(false);
    });
  });

  describe('householdSize (utility)', () => {
    it('should return members + dependents', async () => {
      const household = await bridge.createHousehold(
        makeHouseholdInput({ dependentCount: 3 }),
      );
      expect(householdSize(household)).toBe(household.members.length + 3);
    });
  });

  describe('qualifiesForEmergencyAssistance (utility)', () => {
    it('should qualify when combined index is below 30', () => {
      const assessment: HouseholdSecurityAssessment = {
        householdId: 'test',
        assessmentDate: Date.now(),
        foodSecurityScore: 10,
        housingSecurityScore: 5,
        combinedSecurityIndex: 7.5,
        riskFactors: [],
        availablePrograms: [],
        recommendations: [],
      };
      expect(qualifiesForEmergencyAssistance(assessment)).toBe(true);
    });

    it('should qualify when any risk factor is Critical', () => {
      const assessment: HouseholdSecurityAssessment = {
        householdId: 'test',
        assessmentDate: Date.now(),
        foodSecurityScore: 70,
        housingSecurityScore: 60,
        combinedSecurityIndex: 65,
        riskFactors: [
          { factorType: 'Food', severity: 'Critical', description: 'Hunger' },
        ],
        availablePrograms: [],
        recommendations: [],
      };
      expect(qualifiesForEmergencyAssistance(assessment)).toBe(true);
    });

    it('should not qualify for a secure household assessment', () => {
      const assessment: HouseholdSecurityAssessment = {
        householdId: 'test',
        assessmentDate: Date.now(),
        foodSecurityScore: 100,
        housingSecurityScore: 100,
        combinedSecurityIndex: 100,
        riskFactors: [],
        availablePrograms: [],
        recommendations: [],
      };
      expect(qualifiesForEmergencyAssistance(assessment)).toBe(false);
    });
  });

  describe('createFoodShelterBridge (factory)', () => {
    it('should create a bridge with default config', () => {
      const b = createFoodShelterBridge();
      expect(b).toBeInstanceOf(FoodShelterBridge);
    });

    it('should create a bridge with custom config', () => {
      const b = createFoodShelterBridge({ cacheTtlMs: 60000 });
      expect(b).toBeInstanceOf(FoodShelterBridge);
    });
  });
});
