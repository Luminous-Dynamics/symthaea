// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Food ↔ Shelter Integration Module
 *
 * Provides cross-domain bridges between food security and housing/property systems.
 * Key use cases:
 * - Household resource allocation (food + housing budgets)
 * - Food insecurity linked to housing insecurity
 * - Community food resources tied to community housing
 * - Cross-domain reputation for food/shelter stewardship
 *
 * @module @mycelix/sdk/integrations/food-shelter
 */

import type { ActionHash, AgentPubKey } from '@holochain/client';

// ============================================================================
// Core Types
// ============================================================================

/**
 * Household food security level (USDA-aligned simple enum)
 * Note: Different from health-food's FoodSecurityStatus which is a full assessment record
 */
export type HouseholdFoodSecurityLevel =
  | 'HighFoodSecurity' // No issues accessing food
  | 'MarginalFoodSecurity' // Anxiety about food supply
  | 'LowFoodSecurity' // Reduced quality/variety
  | 'VeryLowFoodSecurity'; // Disrupted eating, hunger

/**
 * Housing security status
 */
export type HousingSecurityStatus =
  | 'StableHousing' // Secure, long-term housing
  | 'AtRisk' // Facing potential displacement
  | 'Unstable' // Frequently moving, temporary housing
  | 'Homeless' // Currently without shelter
  | 'Transitional'; // In transitional housing program

/**
 * Resource category for allocation
 */
export type ResourceCategory =
  | 'Food'
  | 'Shelter'
  | 'Utilities'
  | 'Healthcare'
  | 'Transportation'
  | 'Childcare'
  | 'Other';

/**
 * Household type
 */
export type HouseholdType =
  | 'Single'
  | 'Couple'
  | 'Family'
  | 'SingleParent'
  | 'Multigenerational'
  | 'SharedHousing'
  | 'Collective'; // e.g., commune, co-op

// ============================================================================
// Household Resource Model
// ============================================================================

/**
 * Represents a household unit for resource tracking
 */
export interface Household {
  /** Unique household identifier */
  householdId: string;
  /** Primary property/shelter hash (from property hApp) */
  propertyHash?: ActionHash;
  /** Household members (agent pubkeys) */
  members: AgentPubKey[];
  /** Head of household (for decision-making) */
  headOfHousehold: AgentPubKey;
  /** Household type */
  type: HouseholdType;
  /** Number of dependents (children, elderly) */
  dependentCount: number;
  /** Monthly income (in local currency units) */
  monthlyIncome?: number;
  /** Food security status */
  foodSecurityStatus: HouseholdFoodSecurityLevel;
  /** Housing security status */
  housingSecurityStatus: HousingSecurityStatus;
  /** Created timestamp */
  createdAt: number;
  /** Last updated */
  updatedAt: number;
}

/**
 * Monthly resource allocation for a household
 */
export interface ResourceAllocation {
  /** Reference to household */
  householdId: string;
  /** Month (YYYY-MM format) */
  month: string;
  /** Allocations by category */
  allocations: Record<ResourceCategory, AllocationEntry>;
  /** Total monthly budget */
  totalBudget: number;
  /** Remaining unallocated */
  unallocated: number;
  /** Warnings/flags for the month */
  warnings: AllocationWarning[];
}

/**
 * Single category allocation entry
 */
export interface AllocationEntry {
  /** Budgeted amount */
  budgeted: number;
  /** Actual spent (if tracking) */
  spent?: number;
  /** Percentage of total budget */
  percentOfBudget: number;
  /** Is this below recommended threshold? */
  isBelowRecommended: boolean;
  /** Recommended minimum (based on household size) */
  recommendedMinimum: number;
}

/**
 * Warning for resource allocation issues
 */
export interface AllocationWarning {
  category: ResourceCategory;
  severity: 'Info' | 'Warning' | 'Critical';
  message: string;
  suggestedAction?: string;
}

// ============================================================================
// Cross-Domain Security Assessment
// ============================================================================

/**
 * Combined food + shelter security assessment
 */
export interface HouseholdSecurityAssessment {
  householdId: string;
  assessmentDate: number;
  /** Food security score (0-100) */
  foodSecurityScore: number;
  /** Housing security score (0-100) */
  housingSecurityScore: number;
  /** Combined security index */
  combinedSecurityIndex: number;
  /** Risk factors identified */
  riskFactors: SecurityRiskFactor[];
  /** Available support programs */
  availablePrograms: SupportProgram[];
  /** Recommended interventions */
  recommendations: SecurityRecommendation[];
}

/**
 * Identified risk factor
 */
export interface SecurityRiskFactor {
  factorType: 'Food' | 'Shelter' | 'Income' | 'Health' | 'Employment';
  severity: 'Low' | 'Medium' | 'High' | 'Critical';
  description: string;
  /** How long has this been a factor? */
  durationMonths?: number;
}

/**
 * Available support program
 */
export interface SupportProgram {
  programId: string;
  name: string;
  category: 'FoodAssistance' | 'HousingAssistance' | 'Combined' | 'Emergency';
  description: string;
  eligibilityMet: boolean;
  estimatedBenefit?: number;
  applicationUrl?: string;
}

/**
 * Recommended action
 */
export interface SecurityRecommendation {
  priority: number; // 1 = highest
  action: string;
  category: ResourceCategory;
  potentialImpact: 'Low' | 'Medium' | 'High';
  requiredResources?: string[];
}

// ============================================================================
// Community Resource Sharing
// ============================================================================

/**
 * Community resource hub (shared food/shelter resources)
 */
export interface CommunityResourceHub {
  hubId: string;
  name: string;
  /** Geographic center (from property hApp) */
  locationPropertyHash?: ActionHash;
  /** Service area radius in km */
  serviceRadiusKm: number;
  /** Available food resources */
  foodResources: CommunityFoodResource[];
  /** Available shelter resources */
  shelterResources: CommunityShelterResource[];
  /** Member households */
  memberHouseholds: string[];
  /** Operating schedule */
  operatingHours?: string;
  /** Contact information */
  contactInfo?: ContactInfo;
  /** Status */
  status: 'Active' | 'Paused' | 'Closed';
}

/**
 * Community food resource (pantry, garden, kitchen, etc.)
 */
export interface CommunityFoodResource {
  resourceId: string;
  type: 'FoodPantry' | 'CommunityGarden' | 'CommunityKitchen' | 'MealProgram' | 'FoodRescue';
  name: string;
  description?: string;
  /** Capacity (meals per week, pounds of food, etc.) */
  capacity?: number;
  capacityUnit?: string;
  /** Current utilization percentage */
  utilizationPercent?: number;
  /** Property where located */
  propertyHash?: ActionHash;
  /** Steward/manager */
  stewardAgent?: AgentPubKey;
}

/**
 * Community shelter resource
 */
export interface CommunityShelterResource {
  resourceId: string;
  type: 'EmergencyShelter' | 'TransitionalHousing' | 'SharedHousing' | 'CoLiving' | 'LandTrust';
  name: string;
  description?: string;
  /** Available units/beds */
  availableUnits?: number;
  totalUnits?: number;
  /** Property reference */
  propertyHash?: ActionHash;
  /** Steward/manager */
  stewardAgent?: AgentPubKey;
}

/**
 * Contact information
 */
export interface ContactInfo {
  phone?: string;
  email?: string;
  website?: string;
}

// ============================================================================
// Cross-Domain Reputation
// ============================================================================

/**
 * Cross-domain reputation for food/shelter stewardship
 */
export interface FoodShelterReputation {
  agentPubKey: AgentPubKey;
  /** Food domain reputation */
  foodReputation: DomainReputation;
  /** Shelter domain reputation */
  shelterReputation: DomainReputation;
  /** Combined cross-domain score */
  combinedScore: number;
  /** Last updated */
  lastUpdated: number;
}

/**
 * Single domain reputation
 */
export interface DomainReputation {
  domain: 'Food' | 'Shelter';
  score: number; // 0-1
  /** Number of interactions in this domain */
  interactionCount: number;
  /** Notable contributions */
  contributions: ReputationContribution[];
  /** Trust level based on score */
  trustLevel: 'Trusted' | 'Conditional' | 'Suspicious' | 'Unknown';
}

/**
 * Reputation contribution entry
 */
export interface ReputationContribution {
  type: string;
  description: string;
  impactScore: number;
  timestamp: number;
}

// ============================================================================
// Bridge Client
// ============================================================================

/**
 * Configuration for FoodShelterBridge
 */
export interface FoodShelterBridgeConfig {
  /** Food hApp cell ID */
  foodCellId?: string;
  /** Property hApp cell ID */
  propertyCellId?: string;
  /** Cache TTL in milliseconds */
  cacheTtlMs?: number;
}

/**
 * Food-Shelter Bridge Client
 *
 * Provides methods for cross-domain operations between food and property/shelter systems.
 */
export class FoodShelterBridge {
  private config: FoodShelterBridgeConfig;
  private householdCache: Map<string, Household> = new Map();
  private reputationCache: Map<string, FoodShelterReputation> = new Map();

  constructor(config: FoodShelterBridgeConfig = {}) {
    this.config = {
      cacheTtlMs: 5 * 60 * 1000, // 5 minutes default
      ...config,
    };
  }

  // --------------------------------------------------------------------------
  // Household Management
  // --------------------------------------------------------------------------

  /**
   * Create a new household linking food and shelter resources
   */
  async createHousehold(input: CreateHouseholdInput): Promise<Household> {
    const now = Date.now();
    const household: Household = {
      householdId: this.generateId('hh'),
      propertyHash: input.propertyHash,
      members: input.members,
      headOfHousehold: input.headOfHousehold,
      type: input.type,
      dependentCount: input.dependentCount ?? 0,
      monthlyIncome: input.monthlyIncome,
      foodSecurityStatus: input.foodSecurityStatus ?? 'HighFoodSecurity',
      housingSecurityStatus: input.housingSecurityStatus ?? 'StableHousing',
      createdAt: now,
      updatedAt: now,
    };

    this.householdCache.set(household.householdId, household);
    return household;
  }

  /**
   * Get household by ID
   */
  async getHousehold(householdId: string): Promise<Household | null> {
    return this.householdCache.get(householdId) ?? null;
  }

  /**
   * Update household security status
   */
  async updateSecurityStatus(
    householdId: string,
    foodStatus?: HouseholdFoodSecurityLevel,
    housingStatus?: HousingSecurityStatus
  ): Promise<Household | null> {
    const household = this.householdCache.get(householdId);
    if (!household) return null;

    if (foodStatus) {
      household.foodSecurityStatus = foodStatus;
    }
    if (housingStatus) {
      household.housingSecurityStatus = housingStatus;
    }
    household.updatedAt = Date.now();

    this.householdCache.set(householdId, household);
    return household;
  }

  // --------------------------------------------------------------------------
  // Resource Allocation
  // --------------------------------------------------------------------------

  /**
   * Calculate recommended resource allocation for a household
   */
  calculateResourceAllocation(
    household: Household,
    totalBudget: number
  ): ResourceAllocation {
    const month = new Date().toISOString().slice(0, 7);
    const allocations: Record<ResourceCategory, AllocationEntry> = {} as any;
    const warnings: AllocationWarning[] = [];

    // Recommended percentages based on household size and type
    const recommendations = this.getRecommendedAllocations(household);

    let allocated = 0;

    for (const [category, percent] of Object.entries(recommendations)) {
      const budgeted = Math.round(totalBudget * percent);
      const recommended = this.calculateRecommendedMinimum(
        category as ResourceCategory,
        household
      );
      const isBelowRecommended = budgeted < recommended;

      allocations[category as ResourceCategory] = {
        budgeted,
        percentOfBudget: percent * 100,
        isBelowRecommended,
        recommendedMinimum: recommended,
      };

      allocated += budgeted;

      if (isBelowRecommended) {
        warnings.push({
          category: category as ResourceCategory,
          severity: category === 'Food' || category === 'Shelter' ? 'Critical' : 'Warning',
          message: `${category} allocation ($${budgeted}) is below recommended minimum ($${recommended})`,
          suggestedAction: `Consider reallocating from other categories or seeking assistance`,
        });
      }
    }

    return {
      householdId: household.householdId,
      month,
      allocations,
      totalBudget,
      unallocated: totalBudget - allocated,
      warnings,
    };
  }

  /**
   * Get recommended allocation percentages based on household characteristics
   */
  private getRecommendedAllocations(household: Household): Record<ResourceCategory, number> {
    const hasChildren = household.dependentCount > 0;
    const isLargeFamily = household.members.length + household.dependentCount >= 5;

    // Base recommendations (federal guidelines suggest 30% housing, 10-15% food)
    const base: Record<ResourceCategory, number> = {
      Shelter: 0.30,
      Food: hasChildren ? 0.15 : 0.12,
      Utilities: 0.10,
      Healthcare: 0.08,
      Transportation: 0.12,
      Childcare: hasChildren ? 0.10 : 0.0,
      Other: 0.15,
    };

    // Adjust for large families
    if (isLargeFamily) {
      base.Food += 0.03;
      base.Shelter += 0.02;
      base.Other -= 0.05;
    }

    // Adjust for insecurity
    if (household.foodSecurityStatus === 'VeryLowFoodSecurity') {
      base.Food += 0.05;
      base.Other -= 0.05;
    }

    if (household.housingSecurityStatus === 'AtRisk' || household.housingSecurityStatus === 'Unstable') {
      base.Shelter += 0.05;
      base.Other -= 0.05;
    }

    return base;
  }

  /**
   * Calculate recommended minimum for a category
   */
  private calculateRecommendedMinimum(
    category: ResourceCategory,
    household: Household
  ): number {
    const householdSize = household.members.length + household.dependentCount;

    // USDA thrifty food plan: ~$250/person/month
    // HUD fair market rent varies by location, use $1200 base
    switch (category) {
      case 'Food':
        return householdSize * 250;
      case 'Shelter':
        return 1200 + (householdSize > 2 ? (householdSize - 2) * 200 : 0);
      case 'Utilities':
        return 150 + householdSize * 25;
      case 'Healthcare':
        return householdSize * 100;
      case 'Transportation':
        return 200;
      case 'Childcare':
        return household.dependentCount * 500;
      default:
        return 0;
    }
  }

  // --------------------------------------------------------------------------
  // Security Assessment
  // --------------------------------------------------------------------------

  /**
   * Perform comprehensive household security assessment
   */
  assessHouseholdSecurity(household: Household): HouseholdSecurityAssessment {
    const riskFactors = this.identifyRiskFactors(household);
    const foodScore = this.calculateFoodSecurityScore(household);
    const housingScore = this.calculateHousingSecurityScore(household);
    const combinedIndex = (foodScore + housingScore) / 2;

    return {
      householdId: household.householdId,
      assessmentDate: Date.now(),
      foodSecurityScore: foodScore,
      housingSecurityScore: housingScore,
      combinedSecurityIndex: combinedIndex,
      riskFactors,
      availablePrograms: this.findAvailablePrograms(household, combinedIndex),
      recommendations: this.generateRecommendations(household, riskFactors),
    };
  }

  private calculateFoodSecurityScore(household: Household): number {
    const statusScores: Record<HouseholdFoodSecurityLevel, number> = {
      HighFoodSecurity: 100,
      MarginalFoodSecurity: 70,
      LowFoodSecurity: 40,
      VeryLowFoodSecurity: 10,
    };
    return statusScores[household.foodSecurityStatus];
  }

  private calculateHousingSecurityScore(household: Household): number {
    const statusScores: Record<HousingSecurityStatus, number> = {
      StableHousing: 100,
      AtRisk: 60,
      Unstable: 40,
      Transitional: 30,
      Homeless: 5,
    };
    return statusScores[household.housingSecurityStatus];
  }

  private identifyRiskFactors(household: Household): SecurityRiskFactor[] {
    const factors: SecurityRiskFactor[] = [];

    if (household.foodSecurityStatus !== 'HighFoodSecurity') {
      factors.push({
        factorType: 'Food',
        severity: household.foodSecurityStatus === 'VeryLowFoodSecurity' ? 'Critical' : 'Medium',
        description: `Food security status: ${household.foodSecurityStatus}`,
      });
    }

    if (household.housingSecurityStatus !== 'StableHousing') {
      factors.push({
        factorType: 'Shelter',
        severity: household.housingSecurityStatus === 'Homeless' ? 'Critical' :
          household.housingSecurityStatus === 'Unstable' ? 'High' : 'Medium',
        description: `Housing status: ${household.housingSecurityStatus}`,
      });
    }

    if (household.monthlyIncome && household.monthlyIncome < 2000 * (household.members.length + household.dependentCount)) {
      factors.push({
        factorType: 'Income',
        severity: 'High',
        description: 'Income below poverty threshold for household size',
      });
    }

    return factors;
  }

  private findAvailablePrograms(
    household: Household,
    securityIndex: number
  ): SupportProgram[] {
    const programs: SupportProgram[] = [];

    // SNAP eligibility (simplified)
    if (household.foodSecurityStatus !== 'HighFoodSecurity') {
      programs.push({
        programId: 'snap',
        name: 'SNAP (Food Stamps)',
        category: 'FoodAssistance',
        description: 'Federal nutrition assistance program',
        eligibilityMet: securityIndex < 80,
        estimatedBenefit: 250 * (household.members.length + household.dependentCount),
      });
    }

    // Section 8 (simplified)
    if (household.housingSecurityStatus !== 'StableHousing') {
      programs.push({
        programId: 'section8',
        name: 'Section 8 Housing Choice Voucher',
        category: 'HousingAssistance',
        description: 'Federal rental assistance program',
        eligibilityMet: securityIndex < 60,
      });
    }

    // Emergency assistance
    if (securityIndex < 30) {
      programs.push({
        programId: 'emergency',
        name: 'Emergency Assistance Program',
        category: 'Emergency',
        description: 'Immediate crisis intervention services',
        eligibilityMet: true,
      });
    }

    return programs;
  }

  private generateRecommendations(
    household: Household,
    riskFactors: SecurityRiskFactor[]
  ): SecurityRecommendation[] {
    const recommendations: SecurityRecommendation[] = [];
    let priority = 1;

    for (const factor of riskFactors) {
      if (factor.severity === 'Critical') {
        recommendations.push({
          priority: priority++,
          action: `Address critical ${factor.factorType.toLowerCase()} insecurity immediately`,
          category: factor.factorType === 'Food' ? 'Food' : 'Shelter',
          potentialImpact: 'High',
        });
      }
    }

    if (household.foodSecurityStatus === 'VeryLowFoodSecurity') {
      recommendations.push({
        priority: priority++,
        action: 'Connect with local food pantry or community meal program',
        category: 'Food',
        potentialImpact: 'High',
      });
    }

    if (household.housingSecurityStatus === 'AtRisk') {
      recommendations.push({
        priority: priority++,
        action: 'Contact housing counselor for eviction prevention resources',
        category: 'Shelter',
        potentialImpact: 'High',
      });
    }

    return recommendations;
  }

  // --------------------------------------------------------------------------
  // Cross-Domain Reputation
  // --------------------------------------------------------------------------

  /**
   * Get cross-domain reputation for an agent
   */
  async getCrossReputationion(agentPubKey: AgentPubKey): Promise<FoodShelterReputation> {
    const cached = this.reputationCache.get(agentPubKey.toString());
    if (cached && Date.now() - cached.lastUpdated < this.config.cacheTtlMs!) {
      return cached;
    }

    // In production, this would query both food and property hApps
    const reputation: FoodShelterReputation = {
      agentPubKey,
      foodReputation: {
        domain: 'Food',
        score: 0.75,
        interactionCount: 0,
        contributions: [],
        trustLevel: 'Conditional',
      },
      shelterReputation: {
        domain: 'Shelter',
        score: 0.75,
        interactionCount: 0,
        contributions: [],
        trustLevel: 'Conditional',
      },
      combinedScore: 0.75,
      lastUpdated: Date.now(),
    };

    this.reputationCache.set(agentPubKey.toString(), reputation);
    return reputation;
  }

  /**
   * Record a cross-domain contribution
   */
  async recordContribution(
    agentPubKey: AgentPubKey,
    domain: 'Food' | 'Shelter',
    contribution: ReputationContribution
  ): Promise<void> {
    const reputation = await this.getCrossReputationion(agentPubKey);

    if (domain === 'Food') {
      reputation.foodReputation.contributions.push(contribution);
      reputation.foodReputation.interactionCount++;
      reputation.foodReputation.score = this.recalculateScore(reputation.foodReputation);
    } else {
      reputation.shelterReputation.contributions.push(contribution);
      reputation.shelterReputation.interactionCount++;
      reputation.shelterReputation.score = this.recalculateScore(reputation.shelterReputation);
    }

    reputation.combinedScore = (reputation.foodReputation.score + reputation.shelterReputation.score) / 2;
    reputation.lastUpdated = Date.now();

    this.reputationCache.set(agentPubKey.toString(), reputation);
  }

  private recalculateScore(domainRep: DomainReputation): number {
    if (domainRep.contributions.length === 0) return 0.5;

    const avgImpact =
      domainRep.contributions.reduce((sum, c) => sum + c.impactScore, 0) /
      domainRep.contributions.length;

    // Combine interaction count and impact
    const interactionBoost = Math.min(domainRep.interactionCount * 0.01, 0.2);
    return Math.min(avgImpact + interactionBoost, 1.0);
  }

  // --------------------------------------------------------------------------
  // Utilities
  // --------------------------------------------------------------------------

  private generateId(prefix: string): string {
    return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
  }
}

// ============================================================================
// Input Types
// ============================================================================

/**
 * Input for creating a household
 */
export interface CreateHouseholdInput {
  propertyHash?: ActionHash;
  members: AgentPubKey[];
  headOfHousehold: AgentPubKey;
  type: HouseholdType;
  dependentCount?: number;
  monthlyIncome?: number;
  foodSecurityStatus?: HouseholdFoodSecurityLevel;
  housingSecurityStatus?: HousingSecurityStatus;
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a new FoodShelterBridge instance
 */
export function createFoodShelterBridge(
  config?: FoodShelterBridgeConfig
): FoodShelterBridge {
  return new FoodShelterBridge(config);
}

/**
 * Check if a household is in crisis (both food and housing insecure)
 */
export function isInCrisis(household: Household): boolean {
  const foodCrisis =
    household.foodSecurityStatus === 'VeryLowFoodSecurity' ||
    household.foodSecurityStatus === 'LowFoodSecurity';

  const housingCrisis =
    household.housingSecurityStatus === 'Homeless' ||
    household.housingSecurityStatus === 'Unstable';

  return foodCrisis && housingCrisis;
}

/**
 * Calculate household size including dependents
 */
export function householdSize(household: Household): number {
  return household.members.length + household.dependentCount;
}

/**
 * Check if household qualifies for emergency assistance
 */
export function qualifiesForEmergencyAssistance(
  assessment: HouseholdSecurityAssessment
): boolean {
  return (
    assessment.combinedSecurityIndex < 30 ||
    assessment.riskFactors.some((f) => f.severity === 'Critical')
  );
}

// ============================================================================
// Holochain Conductor Bridge Client
// ============================================================================

/** Holochain conductor bridge client for Food-Shelter cross-domain operations */
export class FoodShelterBridgeClient {
  constructor(
    private client: {
      callZome(input: {
        role_name: string;
        zome_name: string;
        fn_name: string;
        payload: any;
      }): Promise<any>;
    }
  ) {}

  // ---- Food Production (commons_land role) ----

  async createFoodProduction(payload: any): Promise<any> {
    return this.client.callZome({
      role_name: 'commons_land',
      zome_name: 'food-production',
      fn_name: 'create_production',
      payload,
    });
  }

  async getFoodProduction(payload: any): Promise<any> {
    return this.client.callZome({
      role_name: 'commons_land',
      zome_name: 'food-production',
      fn_name: 'get_production',
      payload,
    });
  }

  async listFoodProductions(payload?: any): Promise<any> {
    return this.client.callZome({
      role_name: 'commons_land',
      zome_name: 'food-production',
      fn_name: 'list_productions',
      payload: payload ?? null,
    });
  }

  async updateFoodProduction(payload: any): Promise<any> {
    return this.client.callZome({
      role_name: 'commons_land',
      zome_name: 'food-production',
      fn_name: 'update_production',
      payload,
    });
  }

  // ---- Food Distribution (commons_land role) ----

  async createDistribution(payload: any): Promise<any> {
    return this.client.callZome({
      role_name: 'commons_land',
      zome_name: 'food-distribution',
      fn_name: 'create_distribution',
      payload,
    });
  }

  async getDistribution(payload: any): Promise<any> {
    return this.client.callZome({
      role_name: 'commons_land',
      zome_name: 'food-distribution',
      fn_name: 'get_distribution',
      payload,
    });
  }

  async listDistributions(payload?: any): Promise<any> {
    return this.client.callZome({
      role_name: 'commons_land',
      zome_name: 'food-distribution',
      fn_name: 'list_distributions',
      payload: payload ?? null,
    });
  }

  // ---- Food Preservation (commons_land role) ----

  async createPreservation(payload: any): Promise<any> {
    return this.client.callZome({
      role_name: 'commons_land',
      zome_name: 'food-preservation',
      fn_name: 'create_preservation',
      payload,
    });
  }

  async getPreservation(payload: any): Promise<any> {
    return this.client.callZome({
      role_name: 'commons_land',
      zome_name: 'food-preservation',
      fn_name: 'get_preservation',
      payload,
    });
  }

  async listPreservations(payload?: any): Promise<any> {
    return this.client.callZome({
      role_name: 'commons_land',
      zome_name: 'food-preservation',
      fn_name: 'list_preservations',
      payload: payload ?? null,
    });
  }

  // ---- Food Knowledge (commons_land role) ----

  async createKnowledgeEntry(payload: any): Promise<any> {
    return this.client.callZome({
      role_name: 'commons_land',
      zome_name: 'food-knowledge',
      fn_name: 'create_knowledge_entry',
      payload,
    });
  }

  async getKnowledgeEntry(payload: any): Promise<any> {
    return this.client.callZome({
      role_name: 'commons_land',
      zome_name: 'food-knowledge',
      fn_name: 'get_knowledge_entry',
      payload,
    });
  }

  async listKnowledgeEntries(payload?: any): Promise<any> {
    return this.client.callZome({
      role_name: 'commons_land',
      zome_name: 'food-knowledge',
      fn_name: 'list_knowledge_entries',
      payload: payload ?? null,
    });
  }

  // ---- Emergency Shelters (civic role) ----

  async registerShelter(payload: any): Promise<any> {
    return this.client.callZome({
      role_name: 'civic',
      zome_name: 'emergency-shelters',
      fn_name: 'register_shelter',
      payload,
    });
  }

  async getShelter(payload: any): Promise<any> {
    return this.client.callZome({
      role_name: 'civic',
      zome_name: 'emergency-shelters',
      fn_name: 'get_shelter',
      payload,
    });
  }

  async listShelters(payload?: any): Promise<any> {
    return this.client.callZome({
      role_name: 'civic',
      zome_name: 'emergency-shelters',
      fn_name: 'list_shelters',
      payload: payload ?? null,
    });
  }

  async updateShelterStatus(payload: any): Promise<any> {
    return this.client.callZome({
      role_name: 'civic',
      zome_name: 'emergency-shelters',
      fn_name: 'update_shelter_status',
      payload,
    });
  }

  async updateShelterCapacity(payload: any): Promise<any> {
    return this.client.callZome({
      role_name: 'civic',
      zome_name: 'emergency-shelters',
      fn_name: 'update_shelter_capacity',
      payload,
    });
  }

  async findNearbyShelters(payload: any): Promise<any> {
    return this.client.callZome({
      role_name: 'civic',
      zome_name: 'emergency-shelters',
      fn_name: 'find_nearby_shelters',
      payload,
    });
  }
}
