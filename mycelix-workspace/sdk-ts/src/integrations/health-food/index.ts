/**
 * Health ↔ Food Integration Module
 *
 * Provides cross-module bridges between mycelix-health and food/nutrition systems.
 * Key use cases:
 * - Patient dietary restrictions and allergies
 * - Drug-food interactions
 * - Nutrition tracking and meal planning
 * - SDOH food security integration
 *
 * @module @mycelix/sdk/integrations/health-food
 */

import type { ActionHash } from '@holochain/client';

// ============================================================================
// Types
// ============================================================================

/**
 * Dietary restriction types
 */
export type DietaryRestrictionType =
  | 'Allergy' // Immune-mediated (IgE)
  | 'Intolerance' // Non-immune (lactose, etc.)
  | 'MedicalCondition' // Disease-related (celiac, PKU)
  | 'DrugInteraction' // Medication-related
  | 'Religious' // Faith-based restrictions
  | 'Ethical' // Personal choice (vegan, etc.)
  | 'Other';

/**
 * Severity levels for dietary restrictions
 */
export type RestrictionSeverity =
  | 'LifeThreatening' // Anaphylaxis risk
  | 'Severe' // Serious reaction
  | 'Moderate' // Significant discomfort
  | 'Mild' // Minor symptoms
  | 'Preference'; // No physical reaction

/**
 * Food categories for restriction matching
 */
export type FoodCategory =
  | 'Dairy'
  | 'Eggs'
  | 'Fish'
  | 'Shellfish'
  | 'TreeNuts'
  | 'Peanuts'
  | 'Wheat'
  | 'Soy'
  | 'Sesame'
  | 'Gluten'
  | 'Lactose'
  | 'Fructose'
  | 'Sulfites'
  | 'Nightshades'
  | 'Citrus'
  | 'Meat'
  | 'Pork'
  | 'Alcohol'
  | 'Caffeine'
  | 'Other';

/**
 * Dietary restriction entry
 */
export interface DietaryRestriction {
  restrictionId: string;
  patientHash: ActionHash;
  type: DietaryRestrictionType;
  severity: RestrictionSeverity;
  foodCategory: FoodCategory;
  specificFoods?: string[]; // Specific items within category
  clinicalNotes?: string;
  diagnosedBy?: ActionHash; // Provider who diagnosed
  diagnosedAt?: number;
  verifiedBy?: ActionHash; // Provider who verified
  verifiedAt?: number;
  linkedAllergyHash?: ActionHash; // Link to patient allergy record
  linkedConditionHash?: ActionHash; // Link to diagnosis (e.g., celiac)
  linkedMedicationHash?: ActionHash; // Link to medication causing interaction
  active: boolean;
  createdAt: number;
  updatedAt: number;
}

/**
 * Drug-food interaction
 */
export interface DrugFoodInteraction {
  interactionId: string;
  medicationName: string;
  medicationHash?: ActionHash; // Link to prescription if available
  foodCategory: FoodCategory;
  specificFoods: string[];
  interactionType: 'Avoid' | 'Limit' | 'TimeSeparate' | 'MonitorClosely';
  severity: 'Contraindicated' | 'Major' | 'Moderate' | 'Minor';
  description: string;
  mechanism?: string;
  clinicalEffect?: string;
  recommendation: string;
  evidenceLevel: 'Established' | 'Probable' | 'Suspected' | 'Theoretical';
  sources?: string[];
}

/**
 * Nutrition goal
 */
export interface NutritionGoal {
  goalId: string;
  patientHash: ActionHash;
  goalType: 'WeightManagement' | 'GlucoseControl' | 'HeartHealth' | 'RenalDiet' | 'GIHealth' | 'General';
  targetCalories?: number;
  targetProteinG?: number;
  targetCarbsG?: number;
  targetFatG?: number;
  targetFiberG?: number;
  targetSodiumMg?: number;
  targetPotassiumMg?: number;
  restrictions: string[];
  prescribedBy?: ActionHash;
  startDate: number;
  endDate?: number;
  notes?: string;
  active: boolean;
}

/**
 * Meal log entry
 */
export interface MealLog {
  logId: string;
  patientHash: ActionHash;
  mealType: 'Breakfast' | 'Lunch' | 'Dinner' | 'Snack' | 'Supplement';
  timestamp: number;
  foods: MealItem[];
  totalCalories?: number;
  totalProteinG?: number;
  totalCarbsG?: number;
  totalFatG?: number;
  totalFiberG?: number;
  totalSodiumMg?: number;
  notes?: string;
  photoHash?: string; // Optional photo for meal logging
  location?: string;
  flaggedRestrictions?: string[]; // Any restriction violations
}

/**
 * Individual food item in a meal
 */
export interface MealItem {
  name: string;
  quantity: number;
  unit: string;
  calories?: number;
  proteinG?: number;
  carbsG?: number;
  fatG?: number;
  fiberG?: number;
  sodiumMg?: number;
  categories: FoodCategory[];
  barcode?: string;
  brandName?: string;
}

/**
 * Food security assessment (SDOH)
 */
export interface FoodSecurityStatus {
  assessmentId: string;
  patientHash: ActionHash;
  assessedAt: number;
  assessedBy?: ActionHash;
  securityLevel: 'High' | 'Marginal' | 'Low' | 'VeryLow';
  accessToHealthyFood: boolean;
  affordabilityScore: number; // 0-100
  proximityToGrocery: 'Easy' | 'Moderate' | 'Difficult' | 'NoAccess';
  hasTransportation: boolean;
  receivesAssistance: boolean;
  assistancePrograms?: string[]; // SNAP, WIC, etc.
  barriersToHealthyEating: string[];
  referralsMade?: string[];
  followUpDate?: number;
  notes?: string;
}

/**
 * Nutrition recommendation from AI/provider
 */
export interface NutritionRecommendation {
  recommendationId: string;
  patientHash: ActionHash;
  source: 'Provider' | 'AI' | 'HealthTwin' | 'System';
  sourceHash?: ActionHash; // Provider who made recommendation
  type: 'MealPlan' | 'FoodSwap' | 'Supplement' | 'Avoidance' | 'Timing' | 'Portion' | 'General';
  title: string;
  description: string;
  rationale?: string;
  linkedConditions?: string[];
  linkedMedications?: string[];
  priority: 'Critical' | 'High' | 'Medium' | 'Low';
  createdAt: number;
  expiresAt?: number;
  acknowledged: boolean;
  acknowledgedAt?: number;
}

// ============================================================================
// Zome Callable Interface
// ============================================================================

export interface ZomeCallable {
  callZome(args: {
    role_name: string;
    zome_name: string;
    fn_name: string;
    payload: unknown;
  }): Promise<unknown>;
}

// ============================================================================
// Client Classes
// ============================================================================

const HEALTH_ROLE = 'health';

/**
 * Client for managing dietary restrictions
 */
export class DietaryRestrictionClient {
  constructor(readonly client: ZomeCallable) {}

  /**
   * Add a dietary restriction for a patient
   */
  async addRestriction(
    input: Omit<DietaryRestriction, 'restrictionId' | 'createdAt' | 'updatedAt'>
  ): Promise<DietaryRestriction> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'nutrition',
      fn_name: 'add_dietary_restriction',
      payload: input,
    }) as Promise<DietaryRestriction>;
  }

  /**
   * Get all restrictions for a patient
   */
  async getPatientRestrictions(patientHash: ActionHash): Promise<DietaryRestriction[]> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'nutrition',
      fn_name: 'get_patient_restrictions',
      payload: patientHash,
    }) as Promise<DietaryRestriction[]>;
  }

  /**
   * Get active restrictions only
   */
  async getActiveRestrictions(patientHash: ActionHash): Promise<DietaryRestriction[]> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'nutrition',
      fn_name: 'get_active_restrictions',
      payload: patientHash,
    }) as Promise<DietaryRestriction[]>;
  }

  /**
   * Update restriction status
   */
  async updateRestriction(
    restrictionId: string,
    updates: Partial<DietaryRestriction>
  ): Promise<DietaryRestriction> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'nutrition',
      fn_name: 'update_restriction',
      payload: { restriction_id: restrictionId, updates },
    }) as Promise<DietaryRestriction>;
  }

  /**
   * Link restriction to patient allergy record
   */
  async linkToAllergy(
    restrictionId: string,
    allergyHash: ActionHash
  ): Promise<DietaryRestriction> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'nutrition',
      fn_name: 'link_restriction_to_allergy',
      payload: { restriction_id: restrictionId, allergy_hash: allergyHash },
    }) as Promise<DietaryRestriction>;
  }

  /**
   * Deactivate a restriction
   */
  async deactivateRestriction(restrictionId: string): Promise<void> {
    await this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'nutrition',
      fn_name: 'deactivate_restriction',
      payload: restrictionId,
    });
  }
}

/**
 * Client for drug-food interaction checking
 */
export class DrugFoodInteractionClient {
  constructor(private readonly client: ZomeCallable) {}

  /**
   * Check for drug-food interactions given a medication
   */
  async checkInteractions(medicationName: string): Promise<DrugFoodInteraction[]> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'nutrition',
      fn_name: 'get_drug_food_interactions',
      payload: medicationName,
    }) as Promise<DrugFoodInteraction[]>;
  }

  /**
   * Check all interactions for a patient's current medications
   */
  async checkPatientInteractions(patientHash: ActionHash): Promise<{
    medication: string;
    interactions: DrugFoodInteraction[];
  }[]> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'nutrition',
      fn_name: 'get_patient_drug_food_interactions',
      payload: patientHash,
    }) as Promise<{ medication: string; interactions: DrugFoodInteraction[] }[]>;
  }

  /**
   * Check if a specific food is safe with patient's medications
   */
  async checkFoodSafety(
    patientHash: ActionHash,
    foodCategories: FoodCategory[]
  ): Promise<{
    safe: boolean;
    warnings: DrugFoodInteraction[];
    contraindicated: DrugFoodInteraction[];
  }> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'nutrition',
      fn_name: 'check_food_safety',
      payload: { patient_hash: patientHash, food_categories: foodCategories },
    }) as Promise<{
      safe: boolean;
      warnings: DrugFoodInteraction[];
      contraindicated: DrugFoodInteraction[];
    }>;
  }

  /**
   * Register a new drug-food interaction (admin/provider only)
   */
  async registerInteraction(interaction: Omit<DrugFoodInteraction, 'interactionId'>): Promise<DrugFoodInteraction> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'nutrition',
      fn_name: 'register_drug_food_interaction',
      payload: interaction,
    }) as Promise<DrugFoodInteraction>;
  }
}

/**
 * Client for nutrition goals and meal tracking
 */
export class NutritionTrackingClient {
  constructor(private readonly client: ZomeCallable) {}

  /**
   * Set nutrition goal for a patient
   */
  async setGoal(goal: Omit<NutritionGoal, 'goalId'>): Promise<NutritionGoal> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'nutrition',
      fn_name: 'set_nutrition_goal',
      payload: goal,
    }) as Promise<NutritionGoal>;
  }

  /**
   * Get active nutrition goals for a patient
   */
  async getActiveGoals(patientHash: ActionHash): Promise<NutritionGoal[]> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'nutrition',
      fn_name: 'get_active_nutrition_goals',
      payload: patientHash,
    }) as Promise<NutritionGoal[]>;
  }

  /**
   * Log a meal
   */
  async logMeal(meal: Omit<MealLog, 'logId'>): Promise<MealLog> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'nutrition',
      fn_name: 'log_meal',
      payload: meal,
    }) as Promise<MealLog>;
  }

  /**
   * Get meal history for a patient
   */
  async getMealHistory(
    patientHash: ActionHash,
    startDate: number,
    endDate: number
  ): Promise<MealLog[]> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'nutrition',
      fn_name: 'get_meal_history',
      payload: { patient_hash: patientHash, start_date: startDate, end_date: endDate },
    }) as Promise<MealLog[]>;
  }

  /**
   * Get daily nutrition summary
   */
  async getDailySummary(
    patientHash: ActionHash,
    date: number
  ): Promise<{
    totalCalories: number;
    totalProteinG: number;
    totalCarbsG: number;
    totalFatG: number;
    totalFiberG: number;
    totalSodiumMg: number;
    meals: MealLog[];
    goalProgress: Record<string, number>; // percentage of goal achieved
    restrictionViolations: string[];
  }> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'nutrition',
      fn_name: 'get_daily_nutrition_summary',
      payload: { patient_hash: patientHash, date },
    }) as Promise<{
      totalCalories: number;
      totalProteinG: number;
      totalCarbsG: number;
      totalFatG: number;
      totalFiberG: number;
      totalSodiumMg: number;
      meals: MealLog[];
      goalProgress: Record<string, number>;
      restrictionViolations: string[];
    }>;
  }
}

/**
 * Client for food security (SDOH) assessments
 */
export class FoodSecurityClient {
  constructor(private readonly client: ZomeCallable) {}

  /**
   * Record food security assessment
   */
  async recordAssessment(
    assessment: Omit<FoodSecurityStatus, 'assessmentId'>
  ): Promise<FoodSecurityStatus> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'sdoh',
      fn_name: 'record_food_security_assessment',
      payload: assessment,
    }) as Promise<FoodSecurityStatus>;
  }

  /**
   * Get latest food security status for a patient
   */
  async getLatestAssessment(patientHash: ActionHash): Promise<FoodSecurityStatus | null> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'sdoh',
      fn_name: 'get_latest_food_security',
      payload: patientHash,
    }) as Promise<FoodSecurityStatus | null>;
  }

  /**
   * Get assessment history
   */
  async getAssessmentHistory(patientHash: ActionHash): Promise<FoodSecurityStatus[]> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'sdoh',
      fn_name: 'get_food_security_history',
      payload: patientHash,
    }) as Promise<FoodSecurityStatus[]>;
  }

  /**
   * Find local food assistance resources
   */
  async findFoodResources(
    lat: number,
    lng: number,
    radiusMiles: number
  ): Promise<{
    foodBanks: Array<{ name: string; address: string; phone?: string; hours?: string }>;
    snapOffices: Array<{ name: string; address: string; phone?: string }>;
    mealPrograms: Array<{ name: string; address: string; eligibility?: string }>;
  }> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'sdoh',
      fn_name: 'find_food_resources',
      payload: { lat, lng, radius_miles: radiusMiles },
    }) as Promise<{
      foodBanks: Array<{ name: string; address: string; phone?: string; hours?: string }>;
      snapOffices: Array<{ name: string; address: string; phone?: string }>;
      mealPrograms: Array<{ name: string; address: string; eligibility?: string }>;
    }>;
  }
}

/**
 * Client for nutrition recommendations
 */
export class NutritionRecommendationClient {
  constructor(private readonly client: ZomeCallable) {}

  /**
   * Create a recommendation
   */
  async createRecommendation(
    recommendation: Omit<NutritionRecommendation, 'recommendationId' | 'createdAt' | 'acknowledged' | 'acknowledgedAt'>
  ): Promise<NutritionRecommendation> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'nutrition',
      fn_name: 'create_nutrition_recommendation',
      payload: recommendation,
    }) as Promise<NutritionRecommendation>;
  }

  /**
   * Get active recommendations for a patient
   */
  async getActiveRecommendations(patientHash: ActionHash): Promise<NutritionRecommendation[]> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'nutrition',
      fn_name: 'get_active_recommendations',
      payload: patientHash,
    }) as Promise<NutritionRecommendation[]>;
  }

  /**
   * Acknowledge a recommendation
   */
  async acknowledgeRecommendation(recommendationId: string): Promise<void> {
    await this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'nutrition',
      fn_name: 'acknowledge_recommendation',
      payload: recommendationId,
    });
  }

  /**
   * Generate AI-based recommendations from Health Twin
   */
  async generateRecommendations(patientHash: ActionHash): Promise<NutritionRecommendation[]> {
    return this.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'twin',
      fn_name: 'generate_nutrition_recommendations',
      payload: patientHash,
    }) as Promise<NutritionRecommendation[]>;
  }
}

// ============================================================================
// Unified Health-Food Bridge
// ============================================================================

/**
 * Unified interface for health-food integration
 */
export class HealthFoodBridge {
  readonly restrictions: DietaryRestrictionClient;
  readonly interactions: DrugFoodInteractionClient;
  readonly tracking: NutritionTrackingClient;
  readonly security: FoodSecurityClient;
  readonly recommendations: NutritionRecommendationClient;

  constructor(client: ZomeCallable) {
    this.restrictions = new DietaryRestrictionClient(client);
    this.interactions = new DrugFoodInteractionClient(client);
    this.tracking = new NutritionTrackingClient(client);
    this.security = new FoodSecurityClient(client);
    this.recommendations = new NutritionRecommendationClient(client);
  }

  /**
   * Get comprehensive nutrition profile for a patient
   *
   * Combines restrictions, interactions, goals, and food security status
   */
  async getPatientNutritionProfile(patientHash: ActionHash): Promise<{
    restrictions: DietaryRestriction[];
    drugFoodInteractions: { medication: string; interactions: DrugFoodInteraction[] }[];
    activeGoals: NutritionGoal[];
    foodSecurityStatus: FoodSecurityStatus | null;
    activeRecommendations: NutritionRecommendation[];
    summary: {
      totalRestrictions: number;
      lifeThreatening: number;
      totalInteractions: number;
      contraindicated: number;
      foodInsecure: boolean;
    };
  }> {
    // Fetch all data in parallel
    const [
      restrictions,
      drugFoodInteractions,
      activeGoals,
      foodSecurityStatus,
      activeRecommendations,
    ] = await Promise.all([
      this.restrictions.getActiveRestrictions(patientHash),
      this.interactions.checkPatientInteractions(patientHash),
      this.tracking.getActiveGoals(patientHash),
      this.security.getLatestAssessment(patientHash),
      this.recommendations.getActiveRecommendations(patientHash),
    ]);

    // Calculate summary
    const lifeThreatening = restrictions.filter(r => r.severity === 'LifeThreatening').length;
    const allInteractions = drugFoodInteractions.flatMap(d => d.interactions);
    const contraindicated = allInteractions.filter(i => i.severity === 'Contraindicated').length;
    const foodInsecure = foodSecurityStatus
      ? foodSecurityStatus.securityLevel === 'Low' || foodSecurityStatus.securityLevel === 'VeryLow'
      : false;

    return {
      restrictions,
      drugFoodInteractions,
      activeGoals,
      foodSecurityStatus,
      activeRecommendations,
      summary: {
        totalRestrictions: restrictions.length,
        lifeThreatening,
        totalInteractions: allInteractions.length,
        contraindicated,
        foodInsecure,
      },
    };
  }

  /**
   * Check if a meal is safe for a patient
   *
   * Validates against restrictions and drug-food interactions
   */
  async validateMeal(
    patientHash: ActionHash,
    meal: Omit<MealLog, 'logId' | 'flaggedRestrictions'>
  ): Promise<{
    safe: boolean;
    restrictionViolations: { restriction: DietaryRestriction; violatingFood: string }[];
    interactionWarnings: { interaction: DrugFoodInteraction; violatingFood: string }[];
    recommendations: string[];
  }> {
    // Get patient restrictions and interactions
    const [restrictions, interactionCheck] = await Promise.all([
      this.restrictions.getActiveRestrictions(patientHash),
      this.interactions.checkFoodSafety(
        patientHash,
        meal.foods.flatMap(f => f.categories)
      ),
    ]);

    const restrictionViolations: { restriction: DietaryRestriction; violatingFood: string }[] = [];
    const interactionWarnings: { interaction: DrugFoodInteraction; violatingFood: string }[] = [];
    const recommendations: string[] = [];

    // Check each food against restrictions
    for (const food of meal.foods) {
      for (const restriction of restrictions) {
        if (food.categories.includes(restriction.foodCategory)) {
          restrictionViolations.push({
            restriction,
            violatingFood: food.name,
          });
        }
      }
    }

    // Check drug-food interactions
    for (const food of meal.foods) {
      for (const interaction of [...interactionCheck.warnings, ...interactionCheck.contraindicated]) {
        if (food.categories.some(c => interaction.foodCategory === c)) {
          interactionWarnings.push({
            interaction,
            violatingFood: food.name,
          });
        }
      }
    }

    // Generate recommendations based on violations
    if (restrictionViolations.length > 0) {
      const lifeThreatening = restrictionViolations.filter(
        v => v.restriction.severity === 'LifeThreatening'
      );
      if (lifeThreatening.length > 0) {
        recommendations.push(
          `CRITICAL: Contains ${lifeThreatening.map(v => v.violatingFood).join(', ')} which may cause severe allergic reaction`
        );
      }
    }

    if (interactionWarnings.length > 0) {
      const contraindicated = interactionWarnings.filter(
        w => w.interaction.severity === 'Contraindicated'
      );
      if (contraindicated.length > 0) {
        recommendations.push(
          `WARNING: ${contraindicated.map(w => w.violatingFood).join(', ')} may interact with your medication`
        );
      }
    }

    return {
      safe: restrictionViolations.length === 0 && interactionCheck.contraindicated.length === 0,
      restrictionViolations,
      interactionWarnings,
      recommendations,
    };
  }

  /**
   * Sync patient allergies to dietary restrictions
   *
   * Automatically creates dietary restrictions from patient allergy records
   */
  async syncAllergiesFromHealth(patientHash: ActionHash): Promise<{
    synced: number;
    skipped: number;
    errors: string[];
  }> {
    // This would call the patient zome to get allergies and create corresponding restrictions
    return this.restrictions.client.callZome({
      role_name: HEALTH_ROLE,
      zome_name: 'nutrition',
      fn_name: 'sync_allergies_to_restrictions',
      payload: patientHash,
    }) as Promise<{ synced: number; skipped: number; errors: string[] }>;
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create a Health-Food bridge instance
 */
export function createHealthFoodBridge(client: ZomeCallable): HealthFoodBridge {
  return new HealthFoodBridge(client);
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Map allergy severity to dietary restriction severity
 */
export function mapAllergySeverity(
  allergySeverity: 'Mild' | 'Moderate' | 'Severe' | 'LifeThreatening'
): RestrictionSeverity {
  switch (allergySeverity) {
    case 'LifeThreatening':
      return 'LifeThreatening';
    case 'Severe':
      return 'Severe';
    case 'Moderate':
      return 'Moderate';
    case 'Mild':
      return 'Mild';
    default:
      return 'Moderate';
  }
}

/**
 * Get food categories that contain a specific allergen
 */
export function getAllergenCategories(allergen: string): FoodCategory[] {
  const allergenMap: Record<string, FoodCategory[]> = {
    milk: ['Dairy', 'Lactose'],
    dairy: ['Dairy', 'Lactose'],
    lactose: ['Lactose', 'Dairy'],
    egg: ['Eggs'],
    eggs: ['Eggs'],
    fish: ['Fish'],
    shellfish: ['Shellfish'],
    shrimp: ['Shellfish'],
    crab: ['Shellfish'],
    lobster: ['Shellfish'],
    'tree nut': ['TreeNuts'],
    almond: ['TreeNuts'],
    walnut: ['TreeNuts'],
    cashew: ['TreeNuts'],
    peanut: ['Peanuts'],
    wheat: ['Wheat', 'Gluten'],
    gluten: ['Gluten', 'Wheat'],
    soy: ['Soy'],
    soybean: ['Soy'],
    sesame: ['Sesame'],
    sulfite: ['Sulfites'],
  };

  const lower = allergen.toLowerCase();
  return allergenMap[lower] || ['Other'];
}

/**
 * Calculate nutrition score for a meal relative to goals
 */
export function calculateNutritionScore(
  meal: MealLog,
  goals: NutritionGoal
): {
  overall: number;
  breakdown: Record<string, { actual: number; target: number; score: number }>;
} {
  const breakdown: Record<string, { actual: number; target: number; score: number }> = {};
  let totalScore = 0;
  let metrics = 0;

  if (goals.targetCalories && meal.totalCalories) {
    const ratio = Math.min(meal.totalCalories / goals.targetCalories, 1.5);
    const score = ratio <= 1 ? ratio * 100 : Math.max(0, 100 - (ratio - 1) * 100);
    breakdown.calories = { actual: meal.totalCalories, target: goals.targetCalories, score };
    totalScore += score;
    metrics++;
  }

  if (goals.targetProteinG && meal.totalProteinG) {
    const ratio = meal.totalProteinG / goals.targetProteinG;
    const score = Math.min(ratio * 100, 100);
    breakdown.protein = { actual: meal.totalProteinG, target: goals.targetProteinG, score };
    totalScore += score;
    metrics++;
  }

  if (goals.targetFiberG && meal.totalFiberG) {
    const ratio = meal.totalFiberG / goals.targetFiberG;
    const score = Math.min(ratio * 100, 100);
    breakdown.fiber = { actual: meal.totalFiberG, target: goals.targetFiberG, score };
    totalScore += score;
    metrics++;
  }

  if (goals.targetSodiumMg && meal.totalSodiumMg) {
    // Lower is better for sodium
    const ratio = meal.totalSodiumMg / goals.targetSodiumMg;
    const score = ratio <= 1 ? 100 : Math.max(0, 100 - (ratio - 1) * 50);
    breakdown.sodium = { actual: meal.totalSodiumMg, target: goals.targetSodiumMg, score };
    totalScore += score;
    metrics++;
  }

  return {
    overall: metrics > 0 ? Math.round(totalScore / metrics) : 0,
    breakdown,
  };
}

/**
 * Check if a food category is safe for common medical conditions
 */
export function checkConditionSafety(
  condition: string,
  foodCategory: FoodCategory
): { safe: boolean; warning?: string } {
  const conditionRestrictions: Record<string, { unsafe: FoodCategory[]; warning: string }> = {
    celiac: {
      unsafe: ['Gluten', 'Wheat'],
      warning: 'Contains gluten which can damage intestinal lining',
    },
    diabetes: {
      unsafe: [], // More nuanced - about portions not categories
      warning: 'Monitor carbohydrate intake',
    },
    hypertension: {
      unsafe: [],
      warning: 'Monitor sodium intake',
    },
    'chronic kidney disease': {
      unsafe: [],
      warning: 'Monitor potassium, phosphorus, and sodium',
    },
    phenylketonuria: {
      unsafe: [],
      warning: 'Avoid high-protein foods and aspartame',
    },
  };

  const lower = condition.toLowerCase();
  const restrictions = conditionRestrictions[lower];

  if (restrictions && restrictions.unsafe.includes(foodCategory)) {
    return { safe: false, warning: restrictions.warning };
  }

  return { safe: true };
}

// ============================================================================
// Exports
// ============================================================================

export default {
  HealthFoodBridge,
  DietaryRestrictionClient,
  DrugFoodInteractionClient,
  NutritionTrackingClient,
  FoodSecurityClient,
  NutritionRecommendationClient,
  createHealthFoodBridge,
  mapAllergySeverity,
  getAllergenCategories,
  calculateNutritionScore,
  checkConditionSafety,
};
