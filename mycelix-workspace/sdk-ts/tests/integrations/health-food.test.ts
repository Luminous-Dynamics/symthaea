/**
 * Health-Food Integration Tests
 *
 * Tests for the health-food integration module including:
 * - Utility functions (allergen mapping, nutrition scoring)
 * - Type validation
 * - Bridge functionality
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  mapAllergySeverity,
  getAllergenCategories,
  calculateNutritionScore,
  checkConditionSafety,
  HealthFoodBridge,
  type NutritionGoal,
  type MealLog,
  type DietaryRestriction,
  type FoodCategory,
} from '../../src/integrations/health-food/index.js';

describe('Health-Food Integration', () => {
  describe('mapAllergySeverity', () => {
    it('should map LifeThreatening correctly', () => {
      expect(mapAllergySeverity('LifeThreatening')).toBe('LifeThreatening');
    });

    it('should map Severe correctly', () => {
      expect(mapAllergySeverity('Severe')).toBe('Severe');
    });

    it('should map Moderate correctly', () => {
      expect(mapAllergySeverity('Moderate')).toBe('Moderate');
    });

    it('should map Mild correctly', () => {
      expect(mapAllergySeverity('Mild')).toBe('Mild');
    });
  });

  describe('getAllergenCategories', () => {
    it('should return Dairy and Lactose for milk allergen', () => {
      const categories = getAllergenCategories('milk');
      expect(categories).toContain('Dairy');
      expect(categories).toContain('Lactose');
    });

    it('should return Eggs for egg allergen', () => {
      const categories = getAllergenCategories('eggs');
      expect(categories).toContain('Eggs');
    });

    it('should return TreeNuts for tree nut allergen', () => {
      const categories = getAllergenCategories('almond');
      expect(categories).toContain('TreeNuts');
    });

    it('should return Peanuts for peanut allergen', () => {
      const categories = getAllergenCategories('peanut');
      expect(categories).toContain('Peanuts');
    });

    it('should return Gluten and Wheat for gluten allergen', () => {
      const categories = getAllergenCategories('gluten');
      expect(categories).toContain('Gluten');
      expect(categories).toContain('Wheat');
    });

    it('should return Shellfish for shellfish allergen', () => {
      const categories = getAllergenCategories('shrimp');
      expect(categories).toContain('Shellfish');
    });

    it('should return Other for unknown allergen', () => {
      const categories = getAllergenCategories('unknown_allergen');
      expect(categories).toEqual(['Other']);
    });

    it('should be case insensitive', () => {
      expect(getAllergenCategories('MILK')).toEqual(getAllergenCategories('milk'));
    });
  });

  describe('calculateNutritionScore', () => {
    const baseGoal: NutritionGoal = {
      goalId: 'goal-1',
      patientHash: new Uint8Array(39) as any,
      goalType: 'General',
      targetCalories: 2000,
      targetProteinG: 50,
      targetFiberG: 25,
      targetSodiumMg: 2300,
      restrictions: [],
      startDate: Date.now(),
      active: true,
    };

    it('should return perfect score for exact matches', () => {
      const meal: MealLog = {
        logId: 'meal-1',
        patientHash: new Uint8Array(39) as any,
        mealType: 'Lunch',
        timestamp: Date.now(),
        foods: [],
        totalCalories: 2000,
        totalProteinG: 50,
        totalFiberG: 25,
        totalSodiumMg: 2300,
      };

      const result = calculateNutritionScore(meal, baseGoal);
      expect(result.overall).toBe(100);
    });

    it('should penalize exceeding calorie target', () => {
      const meal: MealLog = {
        logId: 'meal-1',
        patientHash: new Uint8Array(39) as any,
        mealType: 'Lunch',
        timestamp: Date.now(),
        foods: [],
        totalCalories: 3000, // 150% of target
        totalProteinG: 50,
        totalFiberG: 25,
        totalSodiumMg: 2300,
      };

      const result = calculateNutritionScore(meal, baseGoal);
      expect(result.breakdown.calories?.score).toBeLessThan(100);
    });

    it('should reward meeting protein goals', () => {
      const meal: MealLog = {
        logId: 'meal-1',
        patientHash: new Uint8Array(39) as any,
        mealType: 'Lunch',
        timestamp: Date.now(),
        foods: [],
        totalCalories: 2000,
        totalProteinG: 60, // Exceeding target is good for protein
        totalFiberG: 25,
        totalSodiumMg: 2300,
      };

      const result = calculateNutritionScore(meal, baseGoal);
      expect(result.breakdown.protein?.score).toBe(100); // Capped at 100
    });

    it('should penalize exceeding sodium target', () => {
      const meal: MealLog = {
        logId: 'meal-1',
        patientHash: new Uint8Array(39) as any,
        mealType: 'Lunch',
        timestamp: Date.now(),
        foods: [],
        totalCalories: 2000,
        totalProteinG: 50,
        totalFiberG: 25,
        totalSodiumMg: 4600, // Double the target
      };

      const result = calculateNutritionScore(meal, baseGoal);
      expect(result.breakdown.sodium?.score).toBeLessThan(100);
    });

    it('should return 0 when no metrics match goals', () => {
      const emptyGoal: NutritionGoal = {
        goalId: 'goal-2',
        patientHash: new Uint8Array(39) as any,
        goalType: 'General',
        restrictions: [],
        startDate: Date.now(),
        active: true,
      };

      const meal: MealLog = {
        logId: 'meal-1',
        patientHash: new Uint8Array(39) as any,
        mealType: 'Lunch',
        timestamp: Date.now(),
        foods: [],
      };

      const result = calculateNutritionScore(meal, emptyGoal);
      expect(result.overall).toBe(0);
    });
  });

  describe('checkConditionSafety', () => {
    it('should flag gluten as unsafe for celiac', () => {
      const result = checkConditionSafety('celiac', 'Gluten');
      expect(result.safe).toBe(false);
      expect(result.warning).toContain('gluten');
    });

    it('should flag wheat as unsafe for celiac', () => {
      const result = checkConditionSafety('celiac', 'Wheat');
      expect(result.safe).toBe(false);
    });

    it('should return safe for non-restricted foods', () => {
      const result = checkConditionSafety('celiac', 'Dairy');
      expect(result.safe).toBe(true);
    });

    it('should be case insensitive for conditions', () => {
      const result1 = checkConditionSafety('CELIAC', 'Gluten');
      const result2 = checkConditionSafety('celiac', 'Gluten');
      expect(result1.safe).toBe(result2.safe);
    });

    it('should return safe for unknown conditions', () => {
      const result = checkConditionSafety('unknown_condition', 'Dairy');
      expect(result.safe).toBe(true);
    });
  });

  describe('HealthFoodBridge', () => {
    let mockClient: any;
    let bridge: HealthFoodBridge;

    beforeEach(() => {
      mockClient = {
        callZome: vi.fn(),
      };
      bridge = new HealthFoodBridge(mockClient);
    });

    describe('restrictions client', () => {
      it('should call correct zome for getPatientRestrictions', async () => {
        const patientHash = new Uint8Array(39);
        mockClient.callZome.mockResolvedValue([]);

        await bridge.restrictions.getPatientRestrictions(patientHash as any);

        expect(mockClient.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'nutrition',
          fn_name: 'get_patient_restrictions',
          payload: patientHash,
        });
      });

      it('should call correct zome for addRestriction', async () => {
        const restriction = {
          patientHash: new Uint8Array(39) as any,
          type: 'Allergy' as const,
          severity: 'Severe' as const,
          foodCategory: 'Peanuts' as FoodCategory,
          active: true,
        };
        mockClient.callZome.mockResolvedValue({ restrictionId: 'r-1' });

        await bridge.restrictions.addRestriction(restriction);

        expect(mockClient.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'nutrition',
          fn_name: 'add_dietary_restriction',
          payload: restriction,
        });
      });
    });

    describe('interactions client', () => {
      it('should call correct zome for checkInteractions', async () => {
        mockClient.callZome.mockResolvedValue([]);

        await bridge.interactions.checkInteractions('warfarin');

        expect(mockClient.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'nutrition',
          fn_name: 'get_drug_food_interactions',
          payload: 'warfarin',
        });
      });

      it('should call correct zome for checkFoodSafety', async () => {
        const patientHash = new Uint8Array(39);
        mockClient.callZome.mockResolvedValue({
          safe: true,
          warnings: [],
          contraindicated: [],
        });

        await bridge.interactions.checkFoodSafety(
          patientHash as any,
          ['Dairy', 'Eggs']
        );

        expect(mockClient.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'nutrition',
          fn_name: 'check_food_safety',
          payload: {
            patient_hash: patientHash,
            food_categories: ['Dairy', 'Eggs'],
          },
        });
      });
    });

    describe('tracking client', () => {
      it('should call correct zome for logMeal', async () => {
        const meal = {
          patientHash: new Uint8Array(39) as any,
          mealType: 'Lunch' as const,
          timestamp: Date.now(),
          foods: [],
        };
        mockClient.callZome.mockResolvedValue({ logId: 'meal-1' });

        await bridge.tracking.logMeal(meal);

        expect(mockClient.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'nutrition',
          fn_name: 'log_meal',
          payload: meal,
        });
      });

      it('should call correct zome for getDailySummary', async () => {
        const patientHash = new Uint8Array(39);
        const date = Date.now();
        mockClient.callZome.mockResolvedValue({
          totalCalories: 1500,
          meals: [],
          goalProgress: {},
          restrictionViolations: [],
        });

        await bridge.tracking.getDailySummary(patientHash as any, date);

        expect(mockClient.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'nutrition',
          fn_name: 'get_daily_nutrition_summary',
          payload: { patient_hash: patientHash, date },
        });
      });
    });

    describe('security client', () => {
      it('should call correct zome for getLatestAssessment', async () => {
        const patientHash = new Uint8Array(39);
        mockClient.callZome.mockResolvedValue(null);

        await bridge.security.getLatestAssessment(patientHash as any);

        expect(mockClient.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'sdoh',
          fn_name: 'get_latest_food_security',
          payload: patientHash,
        });
      });

      it('should call correct zome for findFoodResources', async () => {
        mockClient.callZome.mockResolvedValue({
          foodBanks: [],
          snapOffices: [],
          mealPrograms: [],
        });

        await bridge.security.findFoodResources(32.9, -96.7, 10);

        expect(mockClient.callZome).toHaveBeenCalledWith({
          role_name: 'health',
          zome_name: 'sdoh',
          fn_name: 'find_food_resources',
          payload: { lat: 32.9, lng: -96.7, radius_miles: 10 },
        });
      });
    });

    describe('getPatientNutritionProfile', () => {
      it('should aggregate data from all sources', async () => {
        const patientHash = new Uint8Array(39) as any;

        mockClient.callZome
          .mockResolvedValueOnce([]) // restrictions
          .mockResolvedValueOnce([]) // drug food interactions
          .mockResolvedValueOnce([]) // active goals
          .mockResolvedValueOnce(null) // food security
          .mockResolvedValueOnce([]); // recommendations

        const profile = await bridge.getPatientNutritionProfile(patientHash);

        expect(profile.restrictions).toEqual([]);
        expect(profile.drugFoodInteractions).toEqual([]);
        expect(profile.activeGoals).toEqual([]);
        expect(profile.foodSecurityStatus).toBeNull();
        expect(profile.activeRecommendations).toEqual([]);
        expect(profile.summary).toBeDefined();
      });

      it('should calculate summary correctly', async () => {
        const patientHash = new Uint8Array(39) as any;

        const restrictions: DietaryRestriction[] = [
          {
            restrictionId: 'r-1',
            patientHash,
            type: 'Allergy',
            severity: 'LifeThreatening',
            foodCategory: 'Peanuts',
            active: true,
            createdAt: Date.now(),
            updatedAt: Date.now(),
          },
          {
            restrictionId: 'r-2',
            patientHash,
            type: 'Intolerance',
            severity: 'Moderate',
            foodCategory: 'Lactose',
            active: true,
            createdAt: Date.now(),
            updatedAt: Date.now(),
          },
        ];

        mockClient.callZome
          .mockResolvedValueOnce(restrictions)
          .mockResolvedValueOnce([{ medication: 'warfarin', interactions: [{ severity: 'Contraindicated' }] }])
          .mockResolvedValueOnce([])
          .mockResolvedValueOnce({ securityLevel: 'VeryLow' })
          .mockResolvedValueOnce([]);

        const profile = await bridge.getPatientNutritionProfile(patientHash);

        expect(profile.summary.totalRestrictions).toBe(2);
        expect(profile.summary.lifeThreatening).toBe(1);
        expect(profile.summary.contraindicated).toBe(1);
        expect(profile.summary.foodInsecure).toBe(true);
      });
    });
  });
});
