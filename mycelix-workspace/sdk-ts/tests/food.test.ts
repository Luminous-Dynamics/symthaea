// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Food Sovereignty Module Tests
 *
 * Tests for FoodSovereigntyService:
 * - Consumer profile management
 * - Food safety checking (allergens, drug interactions, dietary conflicts)
 * - Farm and market discovery
 * - CSA membership management
 * - Seasonal data
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  FoodSovereigntyService,
  type ConsumerProfile,
  type Allergy,
  type Medication,
  type FoodProduct,
  type SafetyCheckResult,
  type Farm,
} from '../src/food/index.js';

describe('FoodSovereigntyService', () => {
  let service: FoodSovereigntyService;

  beforeEach(() => {
    service = new FoodSovereigntyService();
  });

  // ==========================================================================
  // Profile Management
  // ==========================================================================

  describe('profile management', () => {
    it('should set a consumer profile and return it with generated fields', () => {
      const profile = service.setProfile({
        name: 'Test User',
        zipCode: '75080',
        allergies: [],
        dietaryPreferences: [],
        medications: [],
        householdSize: 2,
        preferWorkTrade: false,
        organicPreferred: true,
        localRadiusMiles: 25,
      });

      expect(profile.id).toMatch(/^profile-/);
      expect(profile.name).toBe('Test User');
      expect(profile.zipCode).toBe('75080');
      expect(profile.createdAt).toBeGreaterThan(0);
      expect(profile.updatedAt).toBeGreaterThan(0);
    });

    it('should return null when no profile is set', () => {
      expect(service.getProfile()).toBeNull();
    });

    it('should return the profile after setting it', () => {
      service.setProfile({
        name: 'Jane',
        zipCode: '75002',
        allergies: [],
        dietaryPreferences: [],
        medications: [],
        householdSize: 1,
        preferWorkTrade: false,
        organicPreferred: false,
        localRadiusMiles: 10,
      });

      const profile = service.getProfile();
      expect(profile).not.toBeNull();
      expect(profile!.name).toBe('Jane');
    });

    it('should add an allergy to the profile', () => {
      service.setProfile({
        name: 'Test',
        zipCode: '75080',
        allergies: [],
        dietaryPreferences: [],
        medications: [],
        householdSize: 1,
        preferWorkTrade: false,
        organicPreferred: false,
        localRadiusMiles: 10,
      });

      const allergy: Allergy = {
        allergen: 'peanut',
        severity: 'severe',
        diagnosed: true,
      };

      const updated = service.addAllergy(allergy);
      expect(updated.allergies).toHaveLength(1);
      expect(updated.allergies[0].allergen).toBe('peanut');
    });

    it('should not add duplicate allergies', () => {
      service.setProfile({
        name: 'Test',
        zipCode: '75080',
        allergies: [],
        dietaryPreferences: [],
        medications: [],
        householdSize: 1,
        preferWorkTrade: false,
        organicPreferred: false,
        localRadiusMiles: 10,
      });

      const allergy: Allergy = {
        allergen: 'Peanut',
        severity: 'severe',
        diagnosed: true,
      };

      service.addAllergy(allergy);
      service.addAllergy({ ...allergy, allergen: 'peanut' }); // different case
      expect(service.getProfile()!.allergies).toHaveLength(1);
    });

    it('should throw when adding allergy without profile', () => {
      expect(() =>
        service.addAllergy({ allergen: 'milk', severity: 'mild', diagnosed: false })
      ).toThrow('Set up your profile first');
    });

    it('should add medication with enriched food interactions', () => {
      service.setProfile({
        name: 'Test',
        zipCode: '75080',
        allergies: [],
        dietaryPreferences: [],
        medications: [],
        householdSize: 1,
        preferWorkTrade: false,
        organicPreferred: false,
        localRadiusMiles: 10,
      });

      const updated = service.addMedication({
        name: 'Warfarin',
        genericName: 'warfarin',
      });

      const med = updated.medications[0];
      expect(med.name).toBe('Warfarin');
      // Should be enriched with known interactions
      expect(med.foodsToLimit).toBeDefined();
      expect(med.foodsToLimit!.length).toBeGreaterThan(0);
    });

    it('should throw when adding medication without profile', () => {
      expect(() =>
        service.addMedication({ name: 'Aspirin' })
      ).toThrow('Set up your profile first');
    });
  });

  // ==========================================================================
  // Food Safety Checking
  // ==========================================================================

  describe('food safety checking', () => {
    it('should return safe verdict when no profile is set', () => {
      const result = service.checkFood('apple');
      expect(result.verdict).toBe('safe');
      expect(result.safe).toBe(true);
      expect(result.summary).toContain('Set up your profile');
    });

    it('should detect allergen in known foods', () => {
      service.setProfile({
        name: 'Test',
        zipCode: '75080',
        allergies: [{ allergen: 'peanut', severity: 'life_threatening', diagnosed: true }],
        dietaryPreferences: [],
        medications: [],
        householdSize: 1,
        preferWorkTrade: false,
        organicPreferred: false,
        localRadiusMiles: 10,
      });

      const result = service.checkFood('peanut butter');
      expect(result.verdict).toBe('danger');
      expect(result.safe).toBe(false);
      expect(result.allergenAlerts.length).toBeGreaterThan(0);
      expect(result.summary).toContain('DANGER');
    });

    it('should detect severe allergen and return danger verdict', () => {
      service.setProfile({
        name: 'Test',
        zipCode: '75080',
        allergies: [{ allergen: 'milk', severity: 'severe', diagnosed: true }],
        dietaryPreferences: [],
        medications: [],
        householdSize: 1,
        preferWorkTrade: false,
        organicPreferred: false,
        localRadiusMiles: 10,
      });

      const result = service.checkFood('cheese');
      expect(result.verdict).toBe('danger');
      expect(result.allergenAlerts.some(a => a.allergen === 'milk')).toBe(true);
    });

    it('should detect traces (may contain) allergens', () => {
      service.setProfile({
        name: 'Test',
        zipCode: '75080',
        allergies: [{ allergen: 'tree nuts', severity: 'moderate', diagnosed: true }],
        dietaryPreferences: [],
        medications: [],
        householdSize: 1,
        preferWorkTrade: false,
        organicPreferred: false,
        localRadiusMiles: 10,
      });

      const result = service.checkFood('peanut butter');
      // Peanut butter has traces of tree nuts
      expect(result.allergenAlerts.some(a => a.source === 'traces')).toBe(true);
    });

    it('should detect drug interactions', () => {
      service.setProfile({
        name: 'Test',
        zipCode: '75080',
        allergies: [],
        dietaryPreferences: [],
        medications: [],
        householdSize: 1,
        preferWorkTrade: false,
        organicPreferred: false,
        localRadiusMiles: 10,
      });

      service.addMedication({ name: 'Atorvastatin', genericName: 'atorvastatin' });

      const result = service.checkFood('grapefruit');
      expect(result.drugInteractions.length).toBeGreaterThan(0);
      expect(result.drugInteractions[0].severity).toBe('avoid');
      expect(result.verdict).toBe('avoid');
    });

    it('should detect drug interactions with limit severity', () => {
      service.setProfile({
        name: 'Test',
        zipCode: '75080',
        allergies: [],
        dietaryPreferences: [],
        medications: [],
        householdSize: 1,
        preferWorkTrade: false,
        organicPreferred: false,
        localRadiusMiles: 10,
      });

      service.addMedication({ name: 'Warfarin' });

      const result = service.checkFood('kale');
      expect(result.drugInteractions.length).toBeGreaterThan(0);
      expect(result.verdict).toBe('caution');
    });

    it('should detect dietary conflicts', () => {
      service.setProfile({
        name: 'Test',
        zipCode: '75080',
        allergies: [],
        dietaryPreferences: ['vegan'],
        medications: [],
        householdSize: 1,
        preferWorkTrade: false,
        organicPreferred: false,
        localRadiusMiles: 10,
      });

      const result = service.checkFood('cheese', {
        name: 'Cheddar',
        ingredients: ['milk', 'salt', 'enzymes'],
        allergens: ['milk'],
        traces: [],
        labels: [],
        dataSource: 'manual',
      });

      expect(result.dietaryConflicts.length).toBeGreaterThan(0);
      expect(result.dietaryConflicts[0].preference).toBe('vegan');
    });

    it('should accept custom product data for safety check', () => {
      service.setProfile({
        name: 'Test',
        zipCode: '75080',
        allergies: [{ allergen: 'soy', severity: 'moderate', diagnosed: true }],
        dietaryPreferences: [],
        medications: [],
        householdSize: 1,
        preferWorkTrade: false,
        organicPreferred: false,
        localRadiusMiles: 10,
      });

      const result = service.checkFood('Custom Product', {
        name: 'Custom Product',
        allergens: ['soy'],
        traces: [],
        ingredients: ['soybean oil'],
        labels: [],
        dataSource: 'manual',
      });

      expect(result.allergenAlerts.length).toBeGreaterThan(0);
    });

    it('should return safe for food with no conflicts', () => {
      service.setProfile({
        name: 'Test',
        zipCode: '75080',
        allergies: [{ allergen: 'shellfish', severity: 'severe', diagnosed: true }],
        dietaryPreferences: [],
        medications: [],
        householdSize: 1,
        preferWorkTrade: false,
        organicPreferred: false,
        localRadiusMiles: 10,
      });

      const result = service.checkFood('apple');
      expect(result.verdict).toBe('safe');
      expect(result.safe).toBe(true);
    });
  });

  // ==========================================================================
  // Farm Discovery
  // ==========================================================================

  describe('farm discovery', () => {
    it('should return all farms when no profile is set', () => {
      const farms = service.findFarms();
      expect(farms.length).toBeGreaterThan(0);
    });

    it('should calculate distance when profile is set', () => {
      service.setProfile({
        name: 'Test',
        zipCode: '75080',
        coordinates: { lat: 32.9483, lng: -96.7299 },
        allergies: [],
        dietaryPreferences: [],
        medications: [],
        householdSize: 1,
        preferWorkTrade: false,
        organicPreferred: false,
        localRadiusMiles: 50,
      });

      const farms = service.findFarms();
      expect(farms.length).toBeGreaterThan(0);
      expect(farms[0].distanceMiles).toBeDefined();
      expect(farms[0].distanceMiles).toBeGreaterThan(0);
    });

    it('should sort farms by distance', () => {
      service.setProfile({
        name: 'Test',
        zipCode: '75080',
        coordinates: { lat: 32.9483, lng: -96.7299 },
        allergies: [],
        dietaryPreferences: [],
        medications: [],
        householdSize: 1,
        preferWorkTrade: false,
        organicPreferred: false,
        localRadiusMiles: 100,
      });

      const farms = service.findFarms();
      for (let i = 1; i < farms.length; i++) {
        expect(farms[i].distanceMiles!).toBeGreaterThanOrEqual(farms[i - 1].distanceMiles!);
      }
    });

    it('should filter farms by CSA availability', () => {
      service.setProfile({
        name: 'Test',
        zipCode: '75080',
        allergies: [],
        dietaryPreferences: [],
        medications: [],
        householdSize: 1,
        preferWorkTrade: false,
        organicPreferred: false,
        localRadiusMiles: 100,
      });

      const farms = service.findFarms({ hasCSA: true });
      expect(farms.every(f => f.csa?.available)).toBe(true);
    });

    it('should filter farms by work trade availability', () => {
      service.setProfile({
        name: 'Test',
        zipCode: '75080',
        allergies: [],
        dietaryPreferences: [],
        medications: [],
        householdSize: 1,
        preferWorkTrade: true,
        organicPreferred: false,
        localRadiusMiles: 100,
      });

      const farms = service.findFarms({ acceptsWorkTrade: true });
      expect(farms.every(f => f.csa?.workTradeAvailable)).toBe(true);
    });

    it('should get a specific farm by ID', () => {
      const farm = service.getFarm('farm-1');
      expect(farm).toBeDefined();
      expect(farm!.name).toBe("Bois d'Arc Farm");
    });

    it('should return undefined for unknown farm ID', () => {
      expect(service.getFarm('farm-nonexistent')).toBeUndefined();
    });
  });

  // ==========================================================================
  // Market Discovery
  // ==========================================================================

  describe('market discovery', () => {
    it('should return all markets when no profile is set', () => {
      const markets = service.findMarkets();
      expect(markets.length).toBeGreaterThan(0);
    });

    it('should filter markets by day of week', () => {
      service.setProfile({
        name: 'Test',
        zipCode: '75080',
        allergies: [],
        dietaryPreferences: [],
        medications: [],
        householdSize: 1,
        preferWorkTrade: false,
        organicPreferred: false,
        localRadiusMiles: 100,
      });

      const markets = service.findMarkets({ dayOfWeek: 'Saturday' });
      expect(markets.length).toBeGreaterThan(0);
      expect(markets.every(m => m.dayOfWeek === 'Saturday')).toBe(true);
    });

    it('should filter markets by SNAP acceptance', () => {
      service.setProfile({
        name: 'Test',
        zipCode: '75080',
        allergies: [],
        dietaryPreferences: [],
        medications: [],
        householdSize: 1,
        preferWorkTrade: false,
        organicPreferred: false,
        localRadiusMiles: 100,
      });

      const markets = service.findMarkets({ acceptsSNAP: true });
      expect(markets.every(m => m.acceptsSNAP)).toBe(true);
    });
  });

  // ==========================================================================
  // CSA Membership
  // ==========================================================================

  describe('CSA membership', () => {
    beforeEach(() => {
      service.setProfile({
        name: 'Test',
        zipCode: '75080',
        allergies: [],
        dietaryPreferences: [],
        medications: [],
        householdSize: 2,
        preferWorkTrade: false,
        organicPreferred: false,
        localRadiusMiles: 50,
      });
    });

    it('should join a CSA and create a membership', async () => {
      const membership = await service.joinCSA({
        farmId: 'farm-1',
        shareName: 'Full Share',
        paymentMethod: 'credit_card',
        paymentSchedule: 'monthly',
        pickupLocationIndex: 0,
      });

      expect(membership.id).toMatch(/^membership-/);
      expect(membership.farmName).toBe("Bois d'Arc Farm");
      expect(membership.shareName).toBe('Full Share');
      expect(membership.status).toBe('active');
      expect(membership.totalDue).toBeGreaterThan(0);
    });

    it('should throw when farm not found', async () => {
      await expect(
        service.joinCSA({
          farmId: 'nonexistent',
          shareName: 'Full Share',
          paymentMethod: 'credit_card',
          paymentSchedule: 'monthly',
          pickupLocationIndex: 0,
        })
      ).rejects.toThrow('Farm not found');
    });

    it('should throw when share type not found', async () => {
      await expect(
        service.joinCSA({
          farmId: 'farm-1',
          shareName: 'Nonexistent Share',
          paymentMethod: 'credit_card',
          paymentSchedule: 'monthly',
          pickupLocationIndex: 0,
        })
      ).rejects.toThrow('Share type not found');
    });

    it('should throw when pickup location index invalid', async () => {
      await expect(
        service.joinCSA({
          farmId: 'farm-1',
          shareName: 'Full Share',
          paymentMethod: 'credit_card',
          paymentSchedule: 'monthly',
          pickupLocationIndex: 99,
        })
      ).rejects.toThrow('Pickup location not found');
    });

    it('should apply work trade discount to membership', async () => {
      const membership = await service.joinCSA({
        farmId: 'farm-1',
        shareName: 'Full Share',
        paymentMethod: 'cash',
        paymentSchedule: 'monthly',
        pickupLocationIndex: 0,
        workTradeHours: 10,
      });

      expect(membership.workTradeHours).toBe(10);
      expect(membership.workTradeValue).toBe(150); // 10 * $15/hr
    });

    it('should include add-ons in membership pricing', async () => {
      const membership = await service.joinCSA({
        farmId: 'farm-1',
        shareName: 'Full Share',
        paymentMethod: 'credit_card',
        paymentSchedule: 'monthly',
        pickupLocationIndex: 0,
        addOns: ['Egg Share'],
      });

      expect(membership.addOns.length).toBe(1);
      expect(membership.addOns[0].name).toBe('Egg Share');
      expect(membership.pricePerWeek).toBeGreaterThan(35); // base + add-on
    });

    it('should record payments on membership', async () => {
      const membership = await service.joinCSA({
        farmId: 'farm-1',
        shareName: 'Full Share',
        paymentMethod: 'credit_card',
        paymentSchedule: 'monthly',
        pickupLocationIndex: 0,
      });

      const updated = service.recordPayment(membership.id, 100);
      expect(updated.totalPaid).toBe(100);
    });

    it('should throw when recording payment for nonexistent membership', () => {
      expect(() => service.recordPayment('nonexistent', 100)).toThrow('Membership not found');
    });

    it('should log work trade hours and reduce total due', async () => {
      const membership = await service.joinCSA({
        farmId: 'farm-1',
        shareName: 'Full Share',
        paymentMethod: 'cash',
        paymentSchedule: 'monthly',
        pickupLocationIndex: 0,
        workTradeHours: 20,
      });

      const originalTotalDue = membership.totalDue;
      const updated = service.logWorkTrade(membership.id, 4, 'Weeding');
      expect(updated.workTradeCompleted).toBe(4);
      // 4 hours * $15/hr = $60 reduction
      expect(updated.totalDue).toBe(Math.max(0, originalTotalDue - 60));
    });

    it('should get weekly box for membership', async () => {
      const membership = await service.joinCSA({
        farmId: 'farm-1',
        shareName: 'Full Share',
        paymentMethod: 'credit_card',
        paymentSchedule: 'monthly',
        pickupLocationIndex: 0,
      });

      const box = service.getWeeklyBox(membership.id);
      expect(box.farmName).toBe("Bois d'Arc Farm");
      expect(box.items.length).toBeGreaterThan(0);
      expect(box.totalValue).toBeGreaterThan(0);
      expect(box.storageGuide.length).toBeGreaterThan(0);
    });

    it('should throw when getting weekly box for nonexistent membership', () => {
      expect(() => service.getWeeklyBox('nonexistent')).toThrow('Membership not found');
    });

    it('should list memberships', async () => {
      await service.joinCSA({
        farmId: 'farm-1',
        shareName: 'Full Share',
        paymentMethod: 'credit_card',
        paymentSchedule: 'monthly',
        pickupLocationIndex: 0,
      });

      expect(service.getMemberships()).toHaveLength(1);
    });
  });

  // ==========================================================================
  // Food Story
  // ==========================================================================

  describe('food story', () => {
    it('should generate a food story for an item and farm', () => {
      const story = service.getFoodStory('Tomatoes', 'farm-1');
      expect(story.farmName).toBe("Bois d'Arc Farm");
      expect(story.item).toBe('Tomatoes');
      expect(story.daysSinceHarvest).toBeGreaterThanOrEqual(1);
      expect(story.typicalGroceryMiles).toBe(1500);
      expect(story.practices.length).toBeGreaterThan(0);
    });

    it('should throw for unknown farm', () => {
      expect(() => service.getFoodStory('Tomatoes', 'nonexistent')).toThrow('Farm not found');
    });
  });

  // ==========================================================================
  // Seasonal Data
  // ==========================================================================

  describe('seasonal data', () => {
    it('should return seasonal guide with current month', () => {
      const guide = service.getInSeason();
      expect(guide.region).toBeDefined();
      expect(guide.currentMonth).toBeGreaterThanOrEqual(1);
      expect(guide.currentMonth).toBeLessThanOrEqual(12);
      expect(Array.isArray(guide.inSeason)).toBe(true);
      expect(Array.isArray(guide.comingSoon)).toBe(true);
      expect(Array.isArray(guide.endingSoon)).toBe(true);
    });

    it('should check if an item is in season', () => {
      const result = service.isInSeason('Eggs');
      // Eggs are year-round
      expect(result.inSeason).toBe(true);
    });

    it('should return default for unknown seasonal items', () => {
      const result = service.isInSeason('Dragon Fruit');
      // Unknown items assumed available
      expect(result.inSeason).toBe(true);
      expect(result.peakSeason).toBe(false);
    });
  });
});
