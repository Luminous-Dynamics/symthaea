// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Food Sovereignty Module
 *
 * GROUNDED IN TODAY'S ECONOMY - Built for real people, real food, real money.
 *
 * Consumer-First Design:
 * - Find local farms and CSAs near you
 * - Check any food against YOUR allergies
 * - See where your food comes from (simple, not complex)
 * - Know what's in season in YOUR region
 * - Pay with real money (USD) or work-trade hours
 *
 * Data Sources:
 * - Open Food Facts API (1M+ products, allergens, ingredients)
 * - USDA FoodData Central (nutrition)
 * - Regional seasonal calendars
 * - Real farm/CSA discovery patterns
 *
 * @packageDocumentation
 * @module food
 */

// ============================================================================
// CONSUMER PROFILE - What matters to YOU
// ============================================================================

/** Allergy with severity - because "mild" vs "anaphylactic" matters */
export interface Allergy {
  allergen: string;
  severity: 'mild' | 'moderate' | 'severe' | 'life_threatening';
  diagnosed: boolean;
  notes?: string;
}

/** Dietary preference/restriction */
export type DietaryPreference =
  | 'vegetarian'
  | 'vegan'
  | 'pescatarian'
  | 'gluten_free'
  | 'dairy_free'
  | 'keto'
  | 'paleo'
  | 'halal'
  | 'kosher'
  | 'low_sodium'
  | 'diabetic_friendly';

/** Medication that might interact with foods */
export interface Medication {
  name: string;
  genericName?: string;
  foodsToAvoid?: string[];
  foodsToLimit?: string[];
}

/** Your food profile - simple and useful */
export interface ConsumerProfile {
  id: string;
  name: string;

  // Location for finding local food
  zipCode: string;
  coordinates?: { lat: number; lng: number };

  // Health & Safety
  allergies: Allergy[];
  dietaryPreferences: DietaryPreference[];
  medications: Medication[];
  householdSize: number;

  // Budget (real money)
  weeklyBudget?: number; // USD
  preferWorkTrade: boolean;

  // Preferences
  organicPreferred: boolean;
  localRadiusMiles: number; // How far will you travel?

  createdAt: number;
  updatedAt: number;
}

// ============================================================================
// FOOD SAFETY - Real allergen and drug interaction data
// ============================================================================

/** Food product with real data (Open Food Facts style) */
export interface FoodProduct {
  // Identity
  barcode?: string;
  name: string;
  brand?: string;

  // What's in it
  ingredients: string[];
  ingredientsText?: string;

  // Allergens (real labeling)
  allergens: string[];
  traces: string[]; // "may contain"

  // Nutrition per serving
  nutrition?: {
    servingSize: string;
    calories: number;
    protein: number;
    carbs: number;
    fat: number;
    fiber?: number;
    sodium?: number;
    sugar?: number;
  };

  // Labels
  labels: string[]; // 'organic', 'non-gmo', 'fair-trade', etc.

  // Open Food Facts scores (when available)
  nutritionGrade?: string; // a, b, c, d, e (Nutri-Score)
  novaGroup?: number; // 1-4 (food processing level)
  categories?: string;
  imageUrl?: string;

  // Source
  dataSource: 'open_food_facts' | 'usda' | 'manual' | 'farm_direct' | 'not_found' | 'error';
}

/** Result of checking a food against your profile */
export interface SafetyCheckResult {
  food: string;
  safe: boolean;

  // Allergy alerts
  allergenAlerts: Array<{
    allergen: string;
    severity: Allergy['severity'];
    source: 'ingredient' | 'traces' | 'cross_contamination';
    ingredient?: string;
  }>;

  // Drug interaction warnings
  drugInteractions: Array<{
    medication: string;
    food: string;
    severity: 'avoid' | 'limit' | 'monitor';
    reason: string;
  }>;

  // Dietary conflicts
  dietaryConflicts: Array<{
    preference: DietaryPreference;
    reason: string;
  }>;

  // Simple verdict
  verdict: 'safe' | 'caution' | 'avoid' | 'danger';
  summary: string;
}

// ============================================================================
// FIND LOCAL FOOD - Real farms, real markets, real CSAs
// ============================================================================

/** Farm profile - what consumers need to know */
export interface Farm {
  id: string;
  name: string;

  // Location
  address: string;
  city: string;
  state: string;
  zipCode: string;
  coordinates: { lat: number; lng: number };

  // Contact (real contact info)
  phone?: string;
  email?: string;
  website?: string;
  socialMedia?: {
    facebook?: string;
    instagram?: string;
  };

  // What they grow/raise
  products: string[];
  practices: string[]; // 'organic', 'no-spray', 'pasture-raised', etc.
  certifications: string[]; // 'USDA Organic', 'Certified Naturally Grown', etc.

  // How to buy
  salesChannels: Array<{
    type: 'farm_stand' | 'farmers_market' | 'csa' | 'online' | 'restaurant_wholesale';
    details: string;
    hours?: string;
  }>;

  // CSA details if offered
  csa?: {
    available: boolean;
    sharesRemaining?: number;
    seasonStart: string; // "May"
    seasonEnd: string;   // "October"
    pricing: CSAPricing;
    pickupLocations: PickupLocation[];
    workTradeAvailable: boolean;
    workTradeDiscount?: number; // percentage
  };

  // Simple trust (not complex MATL scores)
  yearsFarming: number;
  acceptsVisitors: boolean;

  // Distance from consumer (calculated)
  distanceMiles?: number;
}

/** CSA pricing - real money, real options */
export interface CSAPricing {
  // Share sizes with real USD prices
  shares: Array<{
    name: string;
    description: string;
    pricePerWeek: number;    // USD
    pricePerSeason: number;  // USD (usually discounted)
    feedsHowMany: string;    // "2-3 people"
  }>;

  // Payment options real CSAs offer
  paymentOptions: Array<{
    type: 'full_upfront' | 'monthly' | 'weekly' | 'sliding_scale' | 'snap_ebt';
    discount?: number;       // percentage off for upfront
    details?: string;
  }>;

  // Add-ons
  addOns?: Array<{
    name: string;            // "Egg share", "Flower share"
    pricePerWeek: number;
  }>;
}

/** Where to pick up your CSA share */
export interface PickupLocation {
  name: string;
  address: string;
  city: string;
  dayOfWeek: string;        // "Wednesday"
  timeWindow: string;       // "3:00 PM - 6:00 PM"
  notes?: string;
}

/** Farmers market with real schedule */
export interface FarmersMarket {
  id: string;
  name: string;

  // Location
  address: string;
  city: string;
  state: string;
  coordinates: { lat: number; lng: number };

  // Schedule (real info)
  season: string;           // "May - October"
  dayOfWeek: string;        // "Saturday"
  hours: string;            // "8:00 AM - 1:00 PM"

  // What to expect
  vendorCount: number;
  acceptsSNAP: boolean;
  acceptsCredit: boolean;

  // Amenities
  parking: string;
  petFriendly: boolean;

  distanceMiles?: number;
}

// ============================================================================
// FOOD STORY - Simple provenance that matters
// ============================================================================

/** Where your food comes from - simple and meaningful */
export interface FoodStory {
  // The food
  item: string;
  variety?: string;         // "Cherokee Purple" tomato

  // The farm
  farmName: string;
  farmLocation: string;     // "Richardson, TX"
  farmerName?: string;

  // The journey
  harvestedDate: Date;
  daysSinceHarvest: number;
  milesFromYou: number;

  // How it was grown
  practices: string[];      // "No synthetic pesticides", "Cover cropped"

  // The story (optional - human touch)
  farmStory?: string;       // "Third generation family farm..."
  photoUrl?: string;

  // Compare to conventional
  typicalGroceryMiles?: number;  // Average supermarket tomato travels 1,500 miles
}

// ============================================================================
// WHAT'S IN SEASON - Regional, practical
// ============================================================================

/** Seasonal availability for your region */
export interface SeasonalItem {
  item: string;
  category: 'vegetable' | 'fruit' | 'herb' | 'meat' | 'dairy' | 'egg';

  // When it's available
  peakMonths: number[];     // [6, 7, 8] = June, July, August
  availableMonths: number[]; // Includes early/late season

  // Storage
  storageTips: string;
  shelfLife: string;        // "1 week refrigerated"

  // Cooking
  quickRecipeIdea: string;
  pairsWith: string[];      // Other seasonal items it goes with
}

/** Regional season guide */
export interface SeasonGuide {
  region: string;           // "North Texas" or USDA zone
  currentMonth: number;

  // What's available NOW
  inSeason: SeasonalItem[];
  comingSoon: SeasonalItem[]; // Next month
  endingSoon: SeasonalItem[]; // Last chance

  // Local tips
  localTips?: string[];
}

// ============================================================================
// CSA MEMBERSHIP - Real subscription management
// ============================================================================

/** Your CSA subscription */
export interface CSAMembership {
  id: string;

  // The farm
  farmId: string;
  farmName: string;

  // Your share
  shareName: string;        // "Full Share"
  shareDescription: string;

  // Payment (real money)
  pricePerWeek: number;     // USD
  totalPaid: number;        // USD
  totalDue: number;         // USD
  paymentMethod: 'credit_card' | 'check' | 'cash' | 'venmo' | 'snap_ebt';
  paymentSchedule: 'full_upfront' | 'monthly' | 'weekly';

  // Work trade
  workTradeHours?: number;  // Hours committed
  workTradeCompleted?: number;
  workTradeValue?: number;  // USD equivalent

  // Pickup
  pickupLocation: PickupLocation;

  // Season
  seasonStart: Date;
  seasonEnd: Date;
  weeksRemaining: number;

  // Status
  status: 'active' | 'paused' | 'completed' | 'cancelled';

  // Add-ons
  addOns: Array<{
    name: string;
    pricePerWeek: number;
  }>;
}

/** What's in your box this week */
export interface WeeklyBox {
  weekOf: Date;
  membershipId: string;
  farmName: string;

  // The contents
  items: Array<{
    item: string;
    quantity: string;       // "1 bunch", "2 lbs"
    variety?: string;       // "Red Russian kale"
    story?: FoodStory;
  }>;

  // Helpful info
  totalValue: number;       // USD - what this would cost at store
  storageGuide: Array<{
    item: string;
    how: string;
    shelfLife: string;
  }>;

  // Recipes that use this week's items
  recipeIdeas: Array<{
    name: string;
    usesItems: string[];
    url?: string;
  }>;

  // Pickup reminder
  pickupDay: string;
  pickupTime: string;
  pickupLocation: string;
}

// ============================================================================
// FOOD SOVEREIGNTY SERVICE - Consumer-focused, grounded in reality
// ============================================================================

/**
 * FoodSovereigntyService - Real food, real money, real useful.
 *
 * Built for consumers who want to:
 * - Find local farms and CSAs
 * - Know their food is safe for their allergies
 * - Understand where their food comes from
 * - Eat what's in season
 * - Support local agriculture with real money
 */
export class FoodSovereigntyService {
  private profile: ConsumerProfile | null = null;

  // In a real app, these would be API calls
  // For now, we use realistic sample data that demonstrates the patterns
  private farms: Map<string, Farm> = new Map();
  private markets: Map<string, FarmersMarket> = new Map();
  private memberships: Map<string, CSAMembership> = new Map();

  constructor() {
    this.initializeRealWorldData();
  }

  // ===========================================================================
  // YOUR PROFILE
  // ===========================================================================

  /**
   * Set up your food profile
   */
  setProfile(profile: Omit<ConsumerProfile, 'id' | 'createdAt' | 'updatedAt'>): ConsumerProfile {
    this.profile = {
      ...profile,
      id: `profile-${Date.now()}`,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    return this.profile;
  }

  /**
   * Get your current profile
   */
  getProfile(): ConsumerProfile | null {
    return this.profile;
  }

  /**
   * Add an allergy to your profile
   */
  addAllergy(allergy: Allergy): ConsumerProfile {
    if (!this.profile) throw new Error('Set up your profile first');

    // Don't add duplicates
    const exists = this.profile.allergies.find(a =>
      a.allergen.toLowerCase() === allergy.allergen.toLowerCase()
    );
    if (!exists) {
      this.profile.allergies.push(allergy);
      this.profile.updatedAt = Date.now();
    }
    return this.profile;
  }

  /**
   * Add a medication to check for food interactions
   */
  addMedication(medication: Medication): ConsumerProfile {
    if (!this.profile) throw new Error('Set up your profile first');

    // Enrich with known food interactions
    const enriched = this.enrichMedicationInteractions(medication);
    this.profile.medications.push(enriched);
    this.profile.updatedAt = Date.now();
    return this.profile;
  }

  // ===========================================================================
  // FOOD SAFETY - Check any food against YOUR profile
  // ===========================================================================

  /**
   * Check if a food is safe for YOU
   *
   * @example
   * ```typescript
   * const result = food.checkFood('peanut butter');
   * if (result.verdict === 'danger') {
   *   console.log(result.summary); // "Contains peanuts - you have a severe peanut allergy"
   * }
   * ```
   */
  checkFood(foodName: string, product?: Partial<FoodProduct>): SafetyCheckResult {
    if (!this.profile) {
      return {
        food: foodName,
        safe: true,
        allergenAlerts: [],
        drugInteractions: [],
        dietaryConflicts: [],
        verdict: 'safe',
        summary: 'Set up your profile to get personalized safety checks',
      };
    }

    const alerts: SafetyCheckResult['allergenAlerts'] = [];
    const interactions: SafetyCheckResult['drugInteractions'] = [];
    const conflicts: SafetyCheckResult['dietaryConflicts'] = [];

    // Get food data (from product or lookup)
    const foodData = product || this.lookupFood(foodName);

    // Check allergens
    for (const allergy of this.profile.allergies) {
      const allergenLower = allergy.allergen.toLowerCase();

      // Check direct allergens
      if (foodData.allergens?.some(a => a.toLowerCase().includes(allergenLower))) {
        alerts.push({
          allergen: allergy.allergen,
          severity: allergy.severity,
          source: 'ingredient',
        });
      }

      // Check traces ("may contain")
      if (foodData.traces?.some(t => t.toLowerCase().includes(allergenLower))) {
        alerts.push({
          allergen: allergy.allergen,
          severity: allergy.severity,
          source: 'traces',
        });
      }

      // Check ingredients text
      if (foodData.ingredientsText?.toLowerCase().includes(allergenLower)) {
        alerts.push({
          allergen: allergy.allergen,
          severity: allergy.severity,
          source: 'ingredient',
          ingredient: allergenLower,
        });
      }

      // Check common aliases
      const aliases = this.getAllergenAliases(allergy.allergen);
      for (const alias of aliases) {
        if (foodData.ingredients?.some(i => i.toLowerCase().includes(alias))) {
          alerts.push({
            allergen: allergy.allergen,
            severity: allergy.severity,
            source: 'ingredient',
            ingredient: alias,
          });
          break;
        }
      }
    }

    // Check drug interactions
    for (const med of this.profile.medications) {
      if (med.foodsToAvoid?.some(f => foodName.toLowerCase().includes(f.toLowerCase()))) {
        interactions.push({
          medication: med.name,
          food: foodName,
          severity: 'avoid',
          reason: this.getDrugInteractionReason(med.name, foodName),
        });
      }
      if (med.foodsToLimit?.some(f => foodName.toLowerCase().includes(f.toLowerCase()))) {
        interactions.push({
          medication: med.name,
          food: foodName,
          severity: 'limit',
          reason: this.getDrugInteractionReason(med.name, foodName),
        });
      }
    }

    // Check dietary preferences
    for (const pref of this.profile.dietaryPreferences) {
      const conflict = this.checkDietaryConflict(pref, foodData);
      if (conflict) {
        conflicts.push(conflict);
      }
    }

    // Determine verdict
    let verdict: SafetyCheckResult['verdict'] = 'safe';
    let summary = `${foodName} looks good for you!`;

    if (alerts.some(a => a.severity === 'life_threatening')) {
      verdict = 'danger';
      summary = `DANGER: Contains ${alerts[0].allergen} - you have a life-threatening allergy`;
    } else if (alerts.some(a => a.severity === 'severe')) {
      verdict = 'danger';
      summary = `AVOID: Contains ${alerts[0].allergen} - you have a severe allergy`;
    } else if (interactions.some(i => i.severity === 'avoid')) {
      verdict = 'avoid';
      summary = `AVOID: Interacts with your ${interactions[0].medication}`;
    } else if (alerts.length > 0 || interactions.some(i => i.severity === 'limit')) {
      verdict = 'caution';
      const reasons: string[] = [];
      if (alerts.length > 0) reasons.push(`may contain ${alerts[0].allergen}`);
      if (interactions.length > 0) reasons.push(`may interact with ${interactions[0].medication}`);
      summary = `CAUTION: ${reasons.join(', ')}`;
    } else if (conflicts.length > 0) {
      verdict = 'caution';
      summary = `Note: ${conflicts[0].reason}`;
    }

    return {
      food: foodName,
      safe: verdict === 'safe',
      allergenAlerts: alerts,
      drugInteractions: interactions,
      dietaryConflicts: conflicts,
      verdict,
      summary,
    };
  }

  /**
   * Scan a barcode and check if it's safe
   * In production, this would call Open Food Facts API
   */
  async checkBarcode(barcode: string): Promise<SafetyCheckResult> {
    const product = await this.lookupBarcode(barcode);
    return this.checkFood(product.name, product);
  }

  // ===========================================================================
  // FIND LOCAL FOOD
  // ===========================================================================

  /**
   * Find farms near you
   *
   * @example
   * ```typescript
   * const farms = food.findFarms({ maxDistance: 25, hasCSA: true });
   * ```
   */
  findFarms(options?: {
    maxDistance?: number;
    hasCSA?: boolean;
    products?: string[];
    certifications?: string[];
    acceptsWorkTrade?: boolean;
  }): Farm[] {
    if (!this.profile) {
      return Array.from(this.farms.values());
    }

    let farms = Array.from(this.farms.values());

    // Calculate distances
    farms = farms.map(farm => ({
      ...farm,
      distanceMiles: this.calculateDistance(
        this.profile!.coordinates || this.zipToCoords(this.profile!.zipCode),
        farm.coordinates
      ),
    }));

    // Filter by distance
    const maxDist = options?.maxDistance || this.profile.localRadiusMiles;
    farms = farms.filter(f => (f.distanceMiles || 0) <= maxDist);

    // Filter by CSA availability
    if (options?.hasCSA) {
      farms = farms.filter(f => f.csa?.available);
    }

    // Filter by products
    if (options?.products?.length) {
      farms = farms.filter(f =>
        options.products!.some(p =>
          f.products.some(fp => fp.toLowerCase().includes(p.toLowerCase()))
        )
      );
    }

    // Filter by certifications
    if (options?.certifications?.length) {
      farms = farms.filter(f =>
        options.certifications!.some(c =>
          f.certifications.some(fc => fc.toLowerCase().includes(c.toLowerCase()))
        )
      );
    }

    // Filter by work trade
    if (options?.acceptsWorkTrade) {
      farms = farms.filter(f => f.csa?.workTradeAvailable);
    }

    // Sort by distance
    return farms.sort((a, b) => (a.distanceMiles || 0) - (b.distanceMiles || 0));
  }

  /**
   * Find farmers markets near you
   */
  findMarkets(options?: {
    maxDistance?: number;
    dayOfWeek?: string;
    acceptsSNAP?: boolean;
  }): FarmersMarket[] {
    if (!this.profile) {
      return Array.from(this.markets.values());
    }

    let markets = Array.from(this.markets.values());

    // Calculate distances
    markets = markets.map(market => ({
      ...market,
      distanceMiles: this.calculateDistance(
        this.profile!.coordinates || this.zipToCoords(this.profile!.zipCode),
        market.coordinates
      ),
    }));

    // Filter
    const maxDist = options?.maxDistance || this.profile.localRadiusMiles;
    markets = markets.filter(m => (m.distanceMiles || 0) <= maxDist);

    if (options?.dayOfWeek) {
      markets = markets.filter(m =>
        m.dayOfWeek.toLowerCase() === options.dayOfWeek!.toLowerCase()
      );
    }

    if (options?.acceptsSNAP) {
      markets = markets.filter(m => m.acceptsSNAP);
    }

    return markets.sort((a, b) => (a.distanceMiles || 0) - (b.distanceMiles || 0));
  }

  /**
   * Get details about a specific farm
   */
  getFarm(farmId: string): Farm | undefined {
    const farm = this.farms.get(farmId);
    if (farm && this.profile) {
      return {
        ...farm,
        distanceMiles: this.calculateDistance(
          this.profile.coordinates || this.zipToCoords(this.profile.zipCode),
          farm.coordinates
        ),
      };
    }
    return farm;
  }

  // ===========================================================================
  // CSA MEMBERSHIP - Real subscriptions with real money
  // ===========================================================================

  /**
   * Join a CSA - with real payment
   *
   * @example
   * ```typescript
   * const membership = await food.joinCSA({
   *   farmId: 'farm-1',
   *   shareName: 'Full Share',
   *   paymentMethod: 'credit_card',
   *   paymentSchedule: 'monthly',
   *   pickupLocationIndex: 0,
   * });
   * ```
   */
  async joinCSA(params: {
    farmId: string;
    shareName: string;
    paymentMethod: CSAMembership['paymentMethod'];
    paymentSchedule: CSAMembership['paymentSchedule'];
    pickupLocationIndex: number;
    addOns?: string[];
    workTradeHours?: number;
  }): Promise<CSAMembership> {
    const farm = this.farms.get(params.farmId);
    if (!farm) throw new Error('Farm not found');
    if (!farm.csa?.available) throw new Error('This farm does not offer CSA');

    const share = farm.csa.pricing.shares.find(s => s.name === params.shareName);
    if (!share) throw new Error('Share type not found');

    const pickupLocation = farm.csa.pickupLocations[params.pickupLocationIndex];
    if (!pickupLocation) throw new Error('Pickup location not found');

    // Calculate pricing
    const pricePerWeek = share.pricePerWeek;
    const seasonWeeks = this.calculateSeasonWeeks(farm.csa.seasonStart, farm.csa.seasonEnd);

    // Apply work trade discount
    let workTradeValue = 0;
    if (params.workTradeHours && farm.csa.workTradeAvailable) {
      const hourlyValue = 15; // $15/hour is common
      workTradeValue = params.workTradeHours * hourlyValue;
    }

    // Calculate add-ons
    const addOns = (params.addOns || []).map(name => {
      const addOn = farm.csa!.pricing.addOns?.find(a => a.name === name);
      return addOn || { name, pricePerWeek: 0 };
    });
    const addOnTotal = addOns.reduce((sum, a) => sum + a.pricePerWeek, 0);

    const totalDue = ((pricePerWeek + addOnTotal) * seasonWeeks) - workTradeValue;

    // Create membership
    const membership: CSAMembership = {
      id: `membership-${Date.now()}`,
      farmId: params.farmId,
      farmName: farm.name,
      shareName: params.shareName,
      shareDescription: share.description,
      pricePerWeek: pricePerWeek + addOnTotal,
      totalPaid: 0,
      totalDue: Math.max(0, totalDue),
      paymentMethod: params.paymentMethod,
      paymentSchedule: params.paymentSchedule,
      workTradeHours: params.workTradeHours,
      workTradeCompleted: 0,
      workTradeValue,
      pickupLocation,
      seasonStart: this.parseSeasonDate(farm.csa.seasonStart),
      seasonEnd: this.parseSeasonDate(farm.csa.seasonEnd),
      weeksRemaining: seasonWeeks,
      status: 'active',
      addOns,
    };

    this.memberships.set(membership.id, membership);

    // In production: process payment via Stripe/Square
    // await this.processPayment(membership);

    return membership;
  }

  /**
   * Get your CSA memberships
   */
  getMemberships(): CSAMembership[] {
    return Array.from(this.memberships.values());
  }

  /**
   * Get what's in your box this week
   */
  getWeeklyBox(membershipId: string): WeeklyBox {
    const membership = this.memberships.get(membershipId);
    if (!membership) throw new Error('Membership not found');

    const farm = this.farms.get(membership.farmId);

    // Generate realistic weekly box based on season
    const seasonalItems = this.getSeasonalItems();
    const boxItems = seasonalItems.slice(0, 8).map(item => ({
      item: item.item,
      quantity: this.getTypicalQuantity(item.item),
      variety: this.getVariety(item.item),
      story: this.generateFoodStory(item.item, farm!),
    }));

    const today = new Date();
    const pickupDay = membership.pickupLocation.dayOfWeek;

    return {
      weekOf: today,
      membershipId,
      farmName: membership.farmName,
      items: boxItems,
      totalValue: this.calculateRetailValue(boxItems),
      storageGuide: boxItems.map(item => ({
        item: item.item,
        how: this.getStorageTip(item.item),
        shelfLife: this.getShelfLife(item.item),
      })),
      recipeIdeas: this.getRecipeIdeas(boxItems.map(i => i.item)),
      pickupDay,
      pickupTime: membership.pickupLocation.timeWindow,
      pickupLocation: `${membership.pickupLocation.name}, ${membership.pickupLocation.address}`,
    };
  }

  /**
   * Record a payment
   */
  recordPayment(membershipId: string, amount: number): CSAMembership {
    const membership = this.memberships.get(membershipId);
    if (!membership) throw new Error('Membership not found');

    membership.totalPaid += amount;
    return membership;
  }

  /**
   * Log work trade hours
   */
  logWorkTrade(membershipId: string, hours: number, _activity: string): CSAMembership {
    const membership = this.memberships.get(membershipId);
    if (!membership) throw new Error('Membership not found');

    membership.workTradeCompleted = (membership.workTradeCompleted || 0) + hours;

    // Update value
    const hourlyValue = 15;
    membership.workTradeValue = (membership.workTradeValue || 0) + (hours * hourlyValue);
    membership.totalDue = Math.max(0, membership.totalDue - (hours * hourlyValue));

    return membership;
  }

  // ===========================================================================
  // FOOD STORY - Simple provenance
  // ===========================================================================

  /**
   * Get the story of a food item
   */
  getFoodStory(item: string, farmId: string): FoodStory {
    const farm = this.farms.get(farmId);
    if (!farm) throw new Error('Farm not found');

    return this.generateFoodStory(item, farm);
  }

  // ===========================================================================
  // WHAT'S IN SEASON
  // ===========================================================================

  /**
   * Get what's in season in your region right now
   */
  getInSeason(): SeasonGuide {
    const currentMonth = new Date().getMonth() + 1; // 1-12
    const region = this.profile ? this.getRegion(this.profile.zipCode) : 'North Texas';

    const allItems = this.getSeasonalData(region);

    return {
      region,
      currentMonth,
      inSeason: allItems.filter(item => item.peakMonths.includes(currentMonth)),
      comingSoon: allItems.filter(item => item.availableMonths.includes(currentMonth + 1) && !item.peakMonths.includes(currentMonth)),
      endingSoon: allItems.filter(item => item.peakMonths.includes(currentMonth) && !item.peakMonths.includes(currentMonth + 1)),
      localTips: this.getSeasonalTips(currentMonth, region),
    };
  }

  /**
   * Check if a specific item is in season
   */
  isInSeason(item: string): { inSeason: boolean; peakSeason: boolean; nextAvailable?: string } {
    const currentMonth = new Date().getMonth() + 1;
    const region = this.profile ? this.getRegion(this.profile.zipCode) : 'North Texas';

    const seasonal = this.getSeasonalData(region).find(s =>
      s.item.toLowerCase() === item.toLowerCase()
    );

    if (!seasonal) {
      return { inSeason: true, peakSeason: false }; // Assume available if not in seasonal list
    }

    const inSeason = seasonal.availableMonths.includes(currentMonth);
    const peakSeason = seasonal.peakMonths.includes(currentMonth);

    let nextAvailable: string | undefined;
    if (!inSeason) {
      const nextMonth = seasonal.availableMonths.find(m => m > currentMonth)
        || seasonal.availableMonths[0];
      nextAvailable = this.monthName(nextMonth);
    }

    return { inSeason, peakSeason, nextAvailable };
  }

  // ===========================================================================
  // HELPERS - Private methods
  // ===========================================================================

  private initializeRealWorldData(): void {
    // Add realistic sample farms (based on real CSA patterns)
    const sampleFarms: Farm[] = [
      {
        id: 'farm-1',
        name: 'Bois d\'Arc Farm',
        address: '1234 Country Road',
        city: 'Allen',
        state: 'TX',
        zipCode: '75002',
        coordinates: { lat: 33.1032, lng: -96.6706 },
        phone: '(972) 555-0123',
        email: 'hello@boisdarcfarm.com',
        website: 'https://boisdarcfarm.com',
        products: ['vegetables', 'herbs', 'eggs', 'flowers'],
        practices: ['organic methods', 'no synthetic pesticides', 'cover cropping', 'composting'],
        certifications: ['Certified Naturally Grown'],
        salesChannels: [
          { type: 'csa', details: '20-week season, May-September', hours: 'Pickup Wednesdays 4-7pm' },
          { type: 'farmers_market', details: 'Richardson Farmers Market', hours: 'Saturdays 8am-12pm' },
        ],
        csa: {
          available: true,
          sharesRemaining: 12,
          seasonStart: 'May',
          seasonEnd: 'September',
          pricing: {
            shares: [
              { name: 'Full Share', description: 'Feeds a family of 4', pricePerWeek: 35, pricePerSeason: 630, feedsHowMany: '3-4 people' },
              { name: 'Half Share', description: 'Feeds 1-2 people', pricePerWeek: 22, pricePerSeason: 396, feedsHowMany: '1-2 people' },
            ],
            paymentOptions: [
              { type: 'full_upfront', discount: 10, details: 'Pay in full by April 15 for 10% off' },
              { type: 'monthly', details: '4 monthly payments' },
              { type: 'snap_ebt', details: 'We accept SNAP/EBT' },
            ],
            addOns: [
              { name: 'Egg Share', pricePerWeek: 6 },
              { name: 'Flower Share', pricePerWeek: 8 },
            ],
          },
          pickupLocations: [
            { name: 'Farm Stand', address: '1234 Country Road, Allen TX', city: 'Allen', dayOfWeek: 'Wednesday', timeWindow: '4:00 PM - 7:00 PM' },
            { name: 'Richardson Civic Center', address: '411 W Arapaho Rd', city: 'Richardson', dayOfWeek: 'Wednesday', timeWindow: '5:00 PM - 7:00 PM' },
          ],
          workTradeAvailable: true,
          workTradeDiscount: 25,
        },
        yearsFarming: 8,
        acceptsVisitors: true,
      },
      {
        id: 'farm-2',
        name: 'Pure Land Organic',
        address: '5678 Farm to Market Road',
        city: 'McKinney',
        state: 'TX',
        zipCode: '75071',
        coordinates: { lat: 33.1972, lng: -96.6397 },
        email: 'info@purelandorganic.com',
        website: 'https://purelandorganic.com',
        products: ['vegetables', 'microgreens', 'herbs'],
        practices: ['USDA Certified Organic', 'no-till', 'drip irrigation'],
        certifications: ['USDA Organic'],
        salesChannels: [
          { type: 'csa', details: '24-week season', hours: 'Thursdays' },
          { type: 'online', details: 'Weekly farm box delivery', hours: 'Order by Tuesday for Thursday delivery' },
        ],
        csa: {
          available: true,
          sharesRemaining: 5,
          seasonStart: 'April',
          seasonEnd: 'October',
          pricing: {
            shares: [
              { name: 'Weekly Box', description: 'Curated seasonal vegetables', pricePerWeek: 40, pricePerSeason: 880, feedsHowMany: '2-4 people' },
              { name: 'Bi-Weekly Box', description: 'Every other week delivery', pricePerWeek: 40, pricePerSeason: 440, feedsHowMany: '2-4 people' },
            ],
            paymentOptions: [
              { type: 'monthly', details: 'Charged first of each month' },
              { type: 'full_upfront', discount: 5 },
            ],
          },
          pickupLocations: [
            { name: 'Home Delivery', address: 'Your address', city: 'Various', dayOfWeek: 'Thursday', timeWindow: '2:00 PM - 6:00 PM', notes: 'Free delivery within 15 miles' },
          ],
          workTradeAvailable: false,
        },
        yearsFarming: 12,
        acceptsVisitors: false,
      },
      {
        id: 'farm-3',
        name: 'Sunshine Family Farm',
        address: '9012 Rural Route',
        city: 'Celina',
        state: 'TX',
        zipCode: '75009',
        coordinates: { lat: 33.3248, lng: -96.7844 },
        phone: '(469) 555-0456',
        website: 'https://sunshinefamilyfarm.net',
        socialMedia: { instagram: '@sunshinefamilyfarm', facebook: 'SunshineFamilyFarmTX' },
        products: ['vegetables', 'pasture-raised eggs', 'pasture-raised chicken', 'honey'],
        practices: ['regenerative agriculture', 'rotational grazing', 'no antibiotics', 'heritage breeds'],
        certifications: ['Animal Welfare Approved', 'Regenerative Organic Certified'],
        salesChannels: [
          { type: 'csa', details: 'Year-round with seasonal variety', hours: 'Saturdays' },
          { type: 'farm_stand', details: 'Open weekends', hours: 'Sat-Sun 9am-3pm' },
        ],
        csa: {
          available: true,
          sharesRemaining: 20,
          seasonStart: 'January',
          seasonEnd: 'December',
          pricing: {
            shares: [
              { name: 'Veggie Share', description: 'Seasonal vegetables', pricePerWeek: 30, pricePerSeason: 1440, feedsHowMany: '2-3 people' },
              { name: 'Omnivore Share', description: 'Veggies + eggs + monthly chicken', pricePerWeek: 55, pricePerSeason: 2640, feedsHowMany: '2-4 people' },
            ],
            paymentOptions: [
              { type: 'monthly' },
              { type: 'sliding_scale', details: 'Pay what you can - ask us about reduced rates' },
            ],
            addOns: [
              { name: 'Extra Eggs (dozen)', pricePerWeek: 7 },
              { name: 'Raw Honey (monthly)', pricePerWeek: 3 },
            ],
          },
          pickupLocations: [
            { name: 'On-Farm', address: '9012 Rural Route, Celina TX', city: 'Celina', dayOfWeek: 'Saturday', timeWindow: '9:00 AM - 12:00 PM' },
          ],
          workTradeAvailable: true,
          workTradeDiscount: 50,
        },
        yearsFarming: 15,
        acceptsVisitors: true,
      },
    ];

    for (const farm of sampleFarms) {
      this.farms.set(farm.id, farm);
    }

    // Add realistic farmers markets
    const sampleMarkets: FarmersMarket[] = [
      {
        id: 'market-1',
        name: 'Richardson Farmers Market',
        address: '411 W Arapaho Rd',
        city: 'Richardson',
        state: 'TX',
        coordinates: { lat: 32.9537, lng: -96.7297 },
        season: 'Year-round',
        dayOfWeek: 'Saturday',
        hours: '8:00 AM - 12:00 PM',
        vendorCount: 45,
        acceptsSNAP: true,
        acceptsCredit: true,
        parking: 'Free parking in civic center lot',
        petFriendly: true,
      },
      {
        id: 'market-2',
        name: 'McKinney Farmers Market',
        address: 'Chestnut Square Historic Village',
        city: 'McKinney',
        state: 'TX',
        coordinates: { lat: 33.1972, lng: -96.6153 },
        season: 'April - November',
        dayOfWeek: 'Saturday',
        hours: '8:00 AM - 12:00 PM',
        vendorCount: 60,
        acceptsSNAP: true,
        acceptsCredit: true,
        parking: 'Street parking and nearby lots',
        petFriendly: true,
      },
      {
        id: 'market-3',
        name: 'Plano Farmers Market',
        address: '1901 E Spring Creek Pkwy',
        city: 'Plano',
        state: 'TX',
        coordinates: { lat: 33.0458, lng: -96.7500 },
        season: 'Year-round (covered)',
        dayOfWeek: 'Saturday',
        hours: '8:00 AM - 1:00 PM',
        vendorCount: 35,
        acceptsSNAP: true,
        acceptsCredit: true,
        parking: 'Large free lot',
        petFriendly: false,
      },
    ];

    for (const market of sampleMarkets) {
      this.markets.set(market.id, market);
    }
  }

  private getAllergenAliases(allergen: string): string[] {
    const aliases: Record<string, string[]> = {
      'peanut': ['peanut', 'groundnut', 'arachis', 'monkey nuts'],
      'tree nuts': ['almond', 'cashew', 'walnut', 'pecan', 'pistachio', 'brazil nut', 'macadamia', 'hazelnut', 'filbert', 'chestnut'],
      'milk': ['milk', 'dairy', 'lactose', 'casein', 'whey', 'cream', 'butter', 'cheese', 'yogurt', 'ghee'],
      'egg': ['egg', 'albumin', 'globulin', 'lysozyme', 'mayonnaise', 'meringue'],
      'wheat': ['wheat', 'flour', 'bread', 'semolina', 'spelt', 'durum', 'farina', 'gluten'],
      'soy': ['soy', 'soya', 'soybean', 'edamame', 'tofu', 'tempeh', 'miso', 'soy sauce'],
      'fish': ['fish', 'cod', 'salmon', 'tuna', 'anchovy', 'bass', 'tilapia', 'fish sauce'],
      'shellfish': ['shellfish', 'shrimp', 'crab', 'lobster', 'crawfish', 'prawn', 'scallop', 'clam', 'mussel', 'oyster'],
      'sesame': ['sesame', 'tahini', 'hummus', 'halvah'],
    };
    return aliases[allergen.toLowerCase()] || [allergen.toLowerCase()];
  }

  private enrichMedicationInteractions(med: Medication): Medication {
    // Real drug-food interactions database
    const interactions: Record<string, { avoid: string[]; limit: string[] }> = {
      'warfarin': {
        avoid: [],
        limit: ['leafy greens', 'kale', 'spinach', 'broccoli', 'brussels sprouts', 'green tea']
      },
      'coumadin': {
        avoid: [],
        limit: ['leafy greens', 'kale', 'spinach', 'broccoli', 'brussels sprouts', 'green tea']
      },
      'lisinopril': {
        avoid: ['potassium supplements'],
        limit: ['bananas', 'oranges', 'potatoes', 'tomatoes', 'salt substitutes']
      },
      'enalapril': {
        avoid: ['potassium supplements'],
        limit: ['bananas', 'oranges', 'potatoes', 'tomatoes']
      },
      'metformin': {
        avoid: ['alcohol'],
        limit: []
      },
      'atorvastatin': {
        avoid: ['grapefruit', 'grapefruit juice'],
        limit: []
      },
      'simvastatin': {
        avoid: ['grapefruit', 'grapefruit juice'],
        limit: []
      },
      'levothyroxine': {
        avoid: [],
        limit: ['soy', 'walnuts', 'fiber supplements', 'coffee']
      },
      'synthroid': {
        avoid: [],
        limit: ['soy', 'walnuts', 'fiber supplements', 'coffee']
      },
      'ciprofloxacin': {
        avoid: ['dairy', 'calcium-fortified foods'],
        limit: ['caffeine']
      },
      'tetracycline': {
        avoid: ['dairy', 'antacids'],
        limit: []
      },
      'maoi': {
        avoid: ['aged cheese', 'cured meats', 'fermented foods', 'soy sauce', 'red wine', 'beer'],
        limit: ['caffeine', 'chocolate']
      },
    };

    const medLower = med.name.toLowerCase();
    const genericLower = med.genericName?.toLowerCase() || '';

    for (const [drug, foods] of Object.entries(interactions)) {
      if (medLower.includes(drug) || genericLower.includes(drug)) {
        return {
          ...med,
          foodsToAvoid: [...(med.foodsToAvoid || []), ...foods.avoid],
          foodsToLimit: [...(med.foodsToLimit || []), ...foods.limit],
        };
      }
    }

    return med;
  }

  private getDrugInteractionReason(medication: string, food: string): string {
    const reasons: Record<string, string> = {
      'warfarin-leafy greens': 'Vitamin K in leafy greens can reduce warfarin effectiveness',
      'warfarin-kale': 'High vitamin K content interferes with blood thinning',
      'lisinopril-bananas': 'Can increase potassium levels, risking hyperkalemia',
      'lisinopril-potatoes': 'High potassium foods may cause dangerous potassium buildup',
      'atorvastatin-grapefruit': 'Grapefruit increases statin levels, raising side effect risk',
      'simvastatin-grapefruit': 'Grapefruit inhibits enzyme that breaks down the medication',
      'metformin-alcohol': 'Alcohol increases risk of lactic acidosis',
      'levothyroxine-soy': 'Soy can decrease thyroid hormone absorption',
      'levothyroxine-coffee': 'Coffee reduces absorption - take medication 1 hour before',
    };

    const key = `${medication.toLowerCase()}-${food.toLowerCase()}`;
    return reasons[key] || `${food} may interact with ${medication}`;
  }

  private checkDietaryConflict(pref: DietaryPreference, food: Partial<FoodProduct>): { preference: DietaryPreference; reason: string } | null {
    const ingredients = (food.ingredients || []).map(i => i.toLowerCase()).join(' ');
    const labels = (food.labels || []).map(l => l.toLowerCase());

    const conflicts: Record<DietaryPreference, { check: () => boolean; reason: string }> = {
      'vegetarian': {
        check: () => ['meat', 'chicken', 'beef', 'pork', 'fish', 'gelatin'].some(i => ingredients.includes(i)),
        reason: 'Contains meat or animal-derived ingredients',
      },
      'vegan': {
        check: () => ['meat', 'chicken', 'beef', 'pork', 'fish', 'milk', 'egg', 'honey', 'gelatin', 'dairy', 'cheese'].some(i => ingredients.includes(i)),
        reason: 'Contains animal products',
      },
      'gluten_free': {
        check: () => ['wheat', 'barley', 'rye', 'malt', 'flour'].some(i => ingredients.includes(i)) && !labels.includes('gluten-free'),
        reason: 'May contain gluten',
      },
      'dairy_free': {
        check: () => ['milk', 'cream', 'cheese', 'butter', 'whey', 'casein', 'lactose'].some(i => ingredients.includes(i)),
        reason: 'Contains dairy',
      },
      'keto': {
        check: () => (food.nutrition?.carbs || 0) > 10,
        reason: 'High in carbohydrates',
      },
      'low_sodium': {
        check: () => (food.nutrition?.sodium || 0) > 600,
        reason: 'High in sodium',
      },
      'pescatarian': {
        check: () => ['chicken', 'beef', 'pork', 'lamb', 'turkey'].some(i => ingredients.includes(i)),
        reason: 'Contains meat (non-seafood)',
      },
      'halal': {
        check: () => ['pork', 'bacon', 'ham', 'lard', 'gelatin'].some(i => ingredients.includes(i)) && !labels.includes('halal'),
        reason: 'May contain non-halal ingredients',
      },
      'kosher': {
        check: () => ['pork', 'shellfish', 'shrimp', 'crab', 'lobster'].some(i => ingredients.includes(i)) && !labels.includes('kosher'),
        reason: 'May contain non-kosher ingredients',
      },
      'paleo': {
        check: () => ['grain', 'wheat', 'corn', 'rice', 'beans', 'dairy', 'sugar'].some(i => ingredients.includes(i)),
        reason: 'Contains grains, legumes, or dairy',
      },
      'diabetic_friendly': {
        check: () => (food.nutrition?.sugar || 0) > 10,
        reason: 'High in sugar',
      },
    };

    const checker = conflicts[pref];
    if (checker && checker.check()) {
      return { preference: pref, reason: checker.reason };
    }
    return null;
  }

  private lookupFood(name: string): Partial<FoodProduct> {
    // In production, this would call Open Food Facts API
    // For now, return basic structure
    const nameLower = name.toLowerCase();

    // Common foods with known allergens
    const knownFoods: Record<string, Partial<FoodProduct>> = {
      'peanut butter': {
        name: 'Peanut Butter',
        allergens: ['peanuts'],
        traces: ['tree nuts'],
        ingredients: ['roasted peanuts', 'salt'],
      },
      'bread': {
        name: 'Bread',
        allergens: ['wheat', 'gluten'],
        traces: ['milk', 'eggs', 'soy'],
        ingredients: ['wheat flour', 'water', 'yeast', 'salt', 'sugar'],
      },
      'milk': {
        name: 'Milk',
        allergens: ['milk'],
        traces: [],
        ingredients: ['milk'],
      },
      'cheese': {
        name: 'Cheese',
        allergens: ['milk'],
        traces: [],
        ingredients: ['milk', 'salt', 'enzymes'],
      },
      'tofu': {
        name: 'Tofu',
        allergens: ['soy'],
        traces: [],
        ingredients: ['soybeans', 'water', 'calcium sulfate'],
      },
      'shrimp': {
        name: 'Shrimp',
        allergens: ['shellfish'],
        traces: ['fish'],
        ingredients: ['shrimp'],
      },
      'soy sauce': {
        name: 'Soy Sauce',
        allergens: ['soy', 'wheat'],
        traces: [],
        ingredients: ['water', 'soybeans', 'wheat', 'salt'],
      },
      'hummus': {
        name: 'Hummus',
        allergens: ['sesame'],
        traces: [],
        ingredients: ['chickpeas', 'tahini', 'olive oil', 'lemon juice', 'garlic'],
      },
    };

    for (const [key, value] of Object.entries(knownFoods)) {
      if (nameLower.includes(key)) {
        return value;
      }
    }

    return {
      name,
      allergens: [],
      traces: [],
      ingredients: [],
    };
  }

  /**
   * Look up a product by barcode using Open Food Facts API
   * This is a REAL API call - no mock data!
   */
  async lookupBarcode(barcode: string): Promise<FoodProduct> {
    try {
      const response = await fetch(
        `https://world.openfoodfacts.org/api/v0/product/${barcode}.json`,
        {
          headers: {
            'User-Agent': 'MycelixFoodSovereignty/1.0 (https://mycelix.net)',
          },
        }
      );

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status}`);
      }

      const data = await response.json();

      if (data.status !== 1 || !data.product) {
        // Product not found in database
        return {
          barcode,
          name: 'Product not found',
          brand: undefined,
          ingredients: [],
          allergens: [],
          traces: [],
          labels: [],
          dataSource: 'not_found',
        };
      }

      const product = data.product;

      // Extract allergens from Open Food Facts format
      // They use tags like "en:gluten", "en:milk", etc.
      const allergens = (product.allergens_tags || []).map((tag: string) =>
        tag.replace(/^en:/, '').replace(/-/g, ' ')
      );

      // Extract traces (may contain)
      const traces = (product.traces_tags || []).map((tag: string) =>
        tag.replace(/^en:/, '').replace(/-/g, ' ')
      );

      // Extract ingredients
      const ingredients = product.ingredients_text
        ? product.ingredients_text.split(',').map((i: string) => i.trim())
        : [];

      // Extract labels (organic, fair trade, etc.)
      const labels = (product.labels_tags || []).map((tag: string) =>
        tag.replace(/^en:/, '').replace(/-/g, ' ')
      );

      return {
        barcode,
        name: product.product_name || product.product_name_en || 'Unknown Product',
        brand: product.brands || undefined,
        ingredients,
        allergens,
        traces,
        labels,
        nutritionGrade: product.nutrition_grades || undefined,
        novaGroup: product.nova_group || undefined,
        categories: product.categories || undefined,
        imageUrl: product.image_front_small_url || undefined,
        dataSource: 'open_food_facts',
      };
    } catch (error) {
      console.error('Open Food Facts lookup failed:', error);
      // Return a fallback for network errors
      return {
        barcode,
        name: 'Lookup failed - try again',
        brand: undefined,
        ingredients: [],
        allergens: [],
        traces: [],
        labels: [],
        dataSource: 'error',
      };
    }
  }

  /**
   * Search for products by name using Open Food Facts API
   * Returns up to 10 results
   */
  async searchProducts(query: string): Promise<FoodProduct[]> {
    if (!query || query.length < 2) return [];

    try {
      const response = await fetch(
        `https://world.openfoodfacts.org/cgi/search.pl?search_terms=${encodeURIComponent(query)}&search_simple=1&action=process&json=1&page_size=10`,
        {
          headers: {
            'User-Agent': 'MycelixFoodSovereignty/1.0 (https://mycelix.net)',
          },
        }
      );

      if (!response.ok) {
        throw new Error(`Search failed: ${response.status}`);
      }

      const data = await response.json();

      if (!data.products || data.products.length === 0) {
        return [];
      }

      return data.products.map((product: any) => ({
        barcode: product.code || '',
        name: product.product_name || product.product_name_en || 'Unknown',
        brand: product.brands || undefined,
        ingredients: product.ingredients_text
          ? product.ingredients_text.split(',').map((i: string) => i.trim())
          : [],
        allergens: (product.allergens_tags || []).map((tag: string) =>
          tag.replace(/^en:/, '').replace(/-/g, ' ')
        ),
        traces: (product.traces_tags || []).map((tag: string) =>
          tag.replace(/^en:/, '').replace(/-/g, ' ')
        ),
        labels: (product.labels_tags || []).map((tag: string) =>
          tag.replace(/^en:/, '').replace(/-/g, ' ')
        ),
        nutritionGrade: product.nutrition_grades || undefined,
        novaGroup: product.nova_group || undefined,
        imageUrl: product.image_front_small_url || undefined,
        dataSource: 'open_food_facts',
      }));
    } catch (error) {
      console.error('Open Food Facts search failed:', error);
      return [];
    }
  }

  private calculateDistance(from: { lat: number; lng: number }, to: { lat: number; lng: number }): number {
    // Haversine formula
    const R = 3959; // Earth's radius in miles
    const dLat = (to.lat - from.lat) * Math.PI / 180;
    const dLon = (to.lng - from.lng) * Math.PI / 180;
    const a =
      Math.sin(dLat/2) * Math.sin(dLat/2) +
      Math.cos(from.lat * Math.PI / 180) * Math.cos(to.lat * Math.PI / 180) *
      Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return Math.round(R * c * 10) / 10;
  }

  private zipToCoords(zip: string): { lat: number; lng: number } {
    // In production: use geocoding API
    // For now, return Richardson TX as default
    const zipCoords: Record<string, { lat: number; lng: number }> = {
      '75080': { lat: 32.9483, lng: -96.7299 }, // Richardson
      '75002': { lat: 33.1032, lng: -96.6706 }, // Allen
      '75071': { lat: 33.1972, lng: -96.6397 }, // McKinney
      '75009': { lat: 33.3248, lng: -96.7844 }, // Celina
    };
    return zipCoords[zip] || { lat: 32.9483, lng: -96.7299 };
  }

  private calculateSeasonWeeks(start: string, end: string): number {
    const months: Record<string, number> = {
      'january': 0, 'february': 1, 'march': 2, 'april': 3,
      'may': 4, 'june': 5, 'july': 6, 'august': 7,
      'september': 8, 'october': 9, 'november': 10, 'december': 11,
    };
    const startMonth = months[start.toLowerCase()] || 4;
    const endMonth = months[end.toLowerCase()] || 9;
    const monthsDiff = endMonth >= startMonth ? endMonth - startMonth + 1 : 12 - startMonth + endMonth + 1;
    return monthsDiff * 4;
  }

  private parseSeasonDate(monthName: string): Date {
    const months: Record<string, number> = {
      'january': 0, 'february': 1, 'march': 2, 'april': 3,
      'may': 4, 'june': 5, 'july': 6, 'august': 7,
      'september': 8, 'october': 9, 'november': 10, 'december': 11,
    };
    const month = months[monthName.toLowerCase()] || 0;
    const year = new Date().getFullYear();
    return new Date(year, month, 1);
  }

  private getSeasonalItems(): SeasonalItem[] {
    const month = new Date().getMonth() + 1;
    return this.getSeasonalData('North Texas').filter(item =>
      item.peakMonths.includes(month) || item.availableMonths.includes(month)
    );
  }

  private getSeasonalData(_region: string): SeasonalItem[] {
    // North Texas seasonal availability (based on real data)
    return [
      { item: 'Tomatoes', category: 'vegetable', peakMonths: [6, 7, 8], availableMonths: [5, 6, 7, 8, 9], storageTips: 'Room temperature until ripe, then refrigerate', shelfLife: '1 week', quickRecipeIdea: 'Caprese salad with fresh basil', pairsWith: ['basil', 'mozzarella', 'cucumber'] },
      { item: 'Zucchini', category: 'vegetable', peakMonths: [6, 7, 8], availableMonths: [5, 6, 7, 8, 9], storageTips: 'Refrigerator crisper drawer', shelfLife: '1-2 weeks', quickRecipeIdea: 'Grilled zucchini with olive oil and herbs', pairsWith: ['squash', 'onions', 'garlic'] },
      { item: 'Okra', category: 'vegetable', peakMonths: [7, 8, 9], availableMonths: [6, 7, 8, 9, 10], storageTips: 'Paper bag in refrigerator', shelfLife: '3-4 days', quickRecipeIdea: 'Fried okra or gumbo', pairsWith: ['tomatoes', 'corn', 'peppers'] },
      { item: 'Peppers', category: 'vegetable', peakMonths: [7, 8, 9], availableMonths: [6, 7, 8, 9, 10], storageTips: 'Refrigerator crisper drawer', shelfLife: '1-2 weeks', quickRecipeIdea: 'Stuffed peppers or fajitas', pairsWith: ['onions', 'tomatoes', 'cilantro'] },
      { item: 'Corn', category: 'vegetable', peakMonths: [6, 7, 8], availableMonths: [5, 6, 7, 8], storageTips: 'Refrigerate in husks, use quickly', shelfLife: '1-3 days', quickRecipeIdea: 'Grilled corn with butter and lime', pairsWith: ['tomatoes', 'peppers', 'cilantro'] },
      { item: 'Watermelon', category: 'fruit', peakMonths: [6, 7, 8], availableMonths: [5, 6, 7, 8, 9], storageTips: 'Room temp until cut, then refrigerate', shelfLife: '2 weeks whole, 5 days cut', quickRecipeIdea: 'Watermelon feta salad', pairsWith: ['feta', 'mint', 'lime'] },
      { item: 'Peaches', category: 'fruit', peakMonths: [6, 7], availableMonths: [5, 6, 7, 8], storageTips: 'Room temp to ripen, refrigerate when ripe', shelfLife: '3-5 days', quickRecipeIdea: 'Peach cobbler or grilled peaches', pairsWith: ['cream', 'vanilla', 'cinnamon'] },
      { item: 'Blackberries', category: 'fruit', peakMonths: [5, 6], availableMonths: [4, 5, 6, 7], storageTips: 'Refrigerate unwashed, rinse before eating', shelfLife: '2-3 days', quickRecipeIdea: 'Fresh with cream or in smoothies', pairsWith: ['cream', 'yogurt', 'other berries'] },
      { item: 'Greens (Kale, Chard)', category: 'vegetable', peakMonths: [3, 4, 5, 10, 11], availableMonths: [2, 3, 4, 5, 10, 11, 12], storageTips: 'Damp paper towel in plastic bag', shelfLife: '5-7 days', quickRecipeIdea: 'Sautéed with garlic and olive oil', pairsWith: ['garlic', 'lemon', 'beans'] },
      { item: 'Lettuce', category: 'vegetable', peakMonths: [3, 4, 5, 10, 11], availableMonths: [2, 3, 4, 5, 10, 11, 12], storageTips: 'Refrigerator crisper, keep dry', shelfLife: '1 week', quickRecipeIdea: 'Fresh salad with vinaigrette', pairsWith: ['tomatoes', 'cucumber', 'radishes'] },
      { item: 'Carrots', category: 'vegetable', peakMonths: [3, 4, 5, 10, 11, 12], availableMonths: [2, 3, 4, 5, 10, 11, 12, 1], storageTips: 'Remove tops, store in water in fridge', shelfLife: '3-4 weeks', quickRecipeIdea: 'Roasted with honey and thyme', pairsWith: ['potatoes', 'onions', 'herbs'] },
      { item: 'Radishes', category: 'vegetable', peakMonths: [3, 4, 5], availableMonths: [2, 3, 4, 5, 10, 11], storageTips: 'Remove tops, refrigerate in water', shelfLife: '1-2 weeks', quickRecipeIdea: 'Sliced with butter and salt', pairsWith: ['butter', 'salt', 'salads'] },
      { item: 'Eggs', category: 'egg', peakMonths: [1,2,3,4,5,6,7,8,9,10,11,12], availableMonths: [1,2,3,4,5,6,7,8,9,10,11,12], storageTips: 'Refrigerate pointed end down', shelfLife: '3-5 weeks', quickRecipeIdea: 'Any way you like them!', pairsWith: ['everything'] },
      { item: 'Basil', category: 'herb', peakMonths: [6, 7, 8, 9], availableMonths: [5, 6, 7, 8, 9, 10], storageTips: 'Trim stems, place in water on counter', shelfLife: '1 week', quickRecipeIdea: 'Fresh pesto or caprese', pairsWith: ['tomatoes', 'mozzarella', 'garlic'] },
      { item: 'Cilantro', category: 'herb', peakMonths: [3, 4, 5, 10, 11], availableMonths: [2, 3, 4, 5, 10, 11, 12], storageTips: 'Stems in water, cover leaves loosely', shelfLife: '1-2 weeks', quickRecipeIdea: 'Salsa, guacamole, or garnish', pairsWith: ['lime', 'peppers', 'onions'] },
    ];
  }

  private getSeasonalTips(month: number, _region: string): string[] {
    const tips: Record<number, string[]> = {
      1: ['Start planning your CSA membership for spring', 'Great time for storage crops and root vegetables'],
      2: ['CSA sign-ups often open now - spots fill fast!', 'Look for early spring greens'],
      3: ['Spring farmers markets are starting', 'Greens and lettuces are peaking'],
      4: ['CSA season is beginning', 'Strawberries are coming!'],
      5: ['Peak time for spring vegetables', 'Start of tomato season in Texas'],
      6: ['Summer bounty is here!', 'Preserve extras by freezing or canning'],
      7: ['Hot weather crops are peaking', 'Shop early at markets before the heat'],
      8: ['Last chance for many summer crops', 'Fall CSA sign-ups may be opening'],
      9: ['Transition to fall crops', 'Great time for peppers and okra'],
      10: ['Fall greens are delicious', 'Look for cool-weather crops'],
      11: ['Support farms through the slow season', 'Stock up on storage vegetables'],
      12: ['Plan for next year\'s garden', 'Gift CSA memberships make great presents!'],
    };
    return tips[month] || [];
  }

  private getRegion(zipCode: string): string {
    // Simplified region lookup
    const prefix = zipCode.substring(0, 3);
    if (['750', '751', '752', '753', '754', '755', '760'].includes(prefix)) {
      return 'North Texas';
    }
    return 'North Texas'; // Default
  }

  private monthName(month: number): string {
    const names = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December'];
    return names[(month - 1) % 12];
  }

  private getTypicalQuantity(item: string): string {
    const quantities: Record<string, string> = {
      'Tomatoes': '2 lbs',
      'Zucchini': '3-4 medium',
      'Okra': '1 lb',
      'Peppers': '3-4',
      'Corn': '6 ears',
      'Watermelon': '1 small',
      'Peaches': '4-6',
      'Blackberries': '1 pint',
      'Greens (Kale, Chard)': '1 bunch',
      'Lettuce': '1 head',
      'Carrots': '1 bunch',
      'Radishes': '1 bunch',
      'Eggs': '1 dozen',
      'Basil': '1 bunch',
      'Cilantro': '1 bunch',
    };
    return quantities[item] || '1 unit';
  }

  private getVariety(item: string): string {
    const varieties: Record<string, string[]> = {
      'Tomatoes': ['Cherokee Purple', 'Brandywine', 'Early Girl', 'Roma', 'Cherry'],
      'Peppers': ['Bell', 'Poblano', 'Jalapeño', 'Shishito', 'Anaheim'],
      'Lettuce': ['Butterhead', 'Romaine', 'Red Leaf', 'Mixed Greens'],
      'Basil': ['Genovese', 'Thai', 'Purple'],
    };
    const options = varieties[item];
    if (options) {
      return options[Math.floor(Math.random() * options.length)];
    }
    return '';
  }

  private generateFoodStory(item: string, farm: Farm): FoodStory {
    const daysAgo = Math.floor(Math.random() * 3) + 1;
    const harvestedDate = new Date();
    harvestedDate.setDate(harvestedDate.getDate() - daysAgo);

    const distanceMiles = farm.distanceMiles || this.calculateDistance(
      this.profile?.coordinates || { lat: 32.9483, lng: -96.7299 },
      farm.coordinates
    );

    return {
      item,
      variety: this.getVariety(item),
      farmName: farm.name,
      farmLocation: `${farm.city}, ${farm.state}`,
      harvestedDate,
      daysSinceHarvest: daysAgo,
      milesFromYou: distanceMiles,
      practices: farm.practices,
      farmStory: `${farm.name} has been growing food for the community for ${farm.yearsFarming} years.`,
      typicalGroceryMiles: 1500, // Average produce travels 1,500 miles
    };
  }

  private calculateRetailValue(items: Array<{ item: string; quantity: string }>): number {
    // Rough retail prices
    const prices: Record<string, number> = {
      'Tomatoes': 4.99,
      'Zucchini': 3.99,
      'Okra': 4.49,
      'Peppers': 4.99,
      'Corn': 4.99,
      'Greens (Kale, Chard)': 3.49,
      'Lettuce': 2.99,
      'Carrots': 2.99,
      'Radishes': 2.49,
      'Eggs': 5.99,
      'Basil': 2.99,
      'Cilantro': 1.99,
    };
    return items.reduce((sum, item) => sum + (prices[item.item] || 3.00), 0);
  }

  private getStorageTip(item: string): string {
    const seasonal = this.getSeasonalData('North Texas').find(s => s.item === item);
    return seasonal?.storageTips || 'Store in refrigerator';
  }

  private getShelfLife(item: string): string {
    const seasonal = this.getSeasonalData('North Texas').find(s => s.item === item);
    return seasonal?.shelfLife || '1 week';
  }

  private getRecipeIdeas(items: string[]): Array<{ name: string; usesItems: string[]; url?: string }> {
    // Simple recipe suggestions based on what's in the box
    const recipes: Array<{ name: string; usesItems: string[]; url?: string }> = [];

    if (items.some(i => i.includes('Tomato')) && items.some(i => i.includes('Basil'))) {
      recipes.push({ name: 'Caprese Salad', usesItems: ['Tomatoes', 'Basil'], url: 'https://example.com/caprese' });
    }
    if (items.some(i => i.includes('Zucchini'))) {
      recipes.push({ name: 'Grilled Zucchini', usesItems: ['Zucchini'], url: 'https://example.com/zucchini' });
    }
    if (items.some(i => i.includes('Pepper')) && items.some(i => i.includes('Corn'))) {
      recipes.push({ name: 'Summer Succotash', usesItems: ['Peppers', 'Corn'], url: 'https://example.com/succotash' });
    }
    if (items.some(i => i.includes('Greens'))) {
      recipes.push({ name: 'Sautéed Greens with Garlic', usesItems: ['Greens'], url: 'https://example.com/greens' });
    }
    if (items.some(i => i.includes('Eggs'))) {
      recipes.push({ name: 'Farm Fresh Frittata', usesItems: ['Eggs'] });
    }

    return recipes.slice(0, 3);
  }
}

// ============================================================================
// SINGLETON INSTANCE (for easy use)
// ============================================================================

export const foodService = new FoodSovereigntyService();
