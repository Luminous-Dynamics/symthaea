// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Energy Sovereignty Module
 *
 * GROUNDED IN TODAY'S ECONOMY - Built for real communities, real energy, real money.
 *
 * Five Pillars:
 * - GENERATE: Production assets (solar, wind, batteries, generators)
 * - STORE: Energy storage and resilience (batteries, thermal, pumped hydro)
 * - SHARE: Peer-to-peer energy trading and surplus allocation
 * - CONSERVE: Efficiency tracking, demand response, usage optimization
 * - GOVERN: Emergency protocols, investment decisions, grid coordination
 *
 * This extends the base energy module with consumer-facing features.
 *
 * @packageDocumentation
 * @module energy/sovereignty
 */

// ============================================================================
// GENERATE - Energy Production Assets
// ============================================================================

/** Energy source types (extended from base) */
export type EnergySourceType =
  | 'solar_pv'          // Photovoltaic panels
  | 'solar_thermal'     // Solar water heating
  | 'wind_turbine'      // Small wind
  | 'micro_hydro'       // Stream/river turbine
  | 'generator_gas'     // Natural gas/propane generator
  | 'generator_diesel'  // Diesel backup
  | 'generator_dual'    // Dual-fuel generator
  | 'fuel_cell'         // Hydrogen fuel cell
  | 'grid'              // Utility connection
  | 'community_solar'   // Shared solar subscription
  | 'geothermal';       // Ground-source heat pump

/** Storage types */
export type StorageType =
  | 'battery_lithium'     // Li-ion, LiFePO4
  | 'battery_lead_acid'   // Traditional lead-acid
  | 'battery_flow'        // Vanadium redox, etc.
  | 'thermal_water'       // Hot water tank
  | 'thermal_ice'         // Ice storage for cooling
  | 'pumped_hydro'        // Water reservoir
  | 'flywheel'            // Mechanical storage
  | 'hydrogen';           // H2 storage

/** Energy asset (something that generates or stores energy) */
export interface EnergyAsset {
  id: string;
  ownerId: string;
  ownerDid: string;

  // What is it?
  type: EnergySourceType | StorageType;
  make?: string;
  model?: string;
  serialNumber?: string;

  // Location
  location: {
    address?: string;
    zipCode: string;
    coordinates?: { lat: number; lng: number };
    installationType: 'rooftop' | 'ground_mount' | 'carport' | 'building_integrated' | 'portable';
  };

  // Capacity
  capacityKw: number;
  capacityKwh?: number;

  // Performance
  efficiencyPercent: number;
  degradationPerYear: number;
  currentAge: number;

  // Economics (real money)
  purchaseCost: number;
  installationCost: number;
  annualMaintenance: number;
  warrantyYears: number;

  // Grid connection
  gridConnected: boolean;
  netMeteringEnabled: boolean;
  utilityAccountId?: string;

  // Status
  status: 'active' | 'maintenance' | 'offline' | 'decommissioned';
  verified: boolean;
  verificationMethod?: 'utility_bill' | 'photo' | 'installer_cert' | 'inspection';

  createdAt: number;
  updatedAt: number;
}

/** Battery storage asset */
export interface StorageAsset extends EnergyAsset {
  storageType: StorageType;
  usableCapacityKwh: number;
  depthOfDischarge: number;
  roundTripEfficiency: number;
  maxChargeRateKw: number;
  maxDischargeRateKw: number;
  cycleLife: number;
  currentCycles: number;
  stateOfCharge: number;
  stateOfHealth: number;
  canProvideBackup: boolean;
  islandingCapable: boolean;
}

// ============================================================================
// SHARE - Peer-to-Peer Energy Trading
// ============================================================================

/** Energy priority levels */
export type EnergyPriority =
  | 'critical_medical'
  | 'essential_safety'
  | 'food_preservation'
  | 'water_systems'
  | 'climate_extreme'
  | 'standard'
  | 'deferrable'
  | 'luxury';

/** Energy offer (surplus to share) */
export interface EnergyOffer {
  id: string;
  sellerId: string;
  sellerDid: string;
  assetId: string;
  sourceType: EnergySourceType;
  amountKwh: number;
  availableFrom: number;
  availableUntil: number;
  pricePerKwh: number;
  currency: 'USD' | 'energy_credits' | 'mutual_credit';
  status: 'open' | 'matched' | 'delivering' | 'completed' | 'cancelled';
  createdAt: number;
}

/** Energy request (need energy) */
export interface EnergyRequest {
  id: string;
  buyerId: string;
  buyerDid: string;
  amountKwh: number;
  neededBy: number;
  maxPricePerKwh: number;
  currency: 'USD' | 'energy_credits' | 'mutual_credit';
  priority: EnergyPriority;
  reason?: string;
  status: 'open' | 'matched' | 'receiving' | 'completed' | 'cancelled';
  createdAt: number;
}

// ============================================================================
// CONSERVE - Efficiency & Demand Response
// ============================================================================

/** Meter reading */
export interface MeterReading {
  id: string;
  householdId: string;
  timestamp: number;
  readingKwh: number;
  readingType: 'manual' | 'smart_meter' | 'estimated';
  direction: 'consumption' | 'generation' | 'net';
  ratePeriod?: 'peak' | 'off_peak' | 'mid_peak' | 'super_off_peak';
}

/** Conservation tip */
export interface ConservationTip {
  id: string;
  category: 'hvac' | 'lighting' | 'appliances' | 'water_heating' | 'behavioral' | 'upgrade';
  title: string;
  description: string;
  estimatedSavingsKwh: number;
  estimatedSavingsUsd: number;
  implementationCost: number;
  paybackMonths: number;
  difficulty: 'easy' | 'moderate' | 'professional';
  diyPossible: boolean;
}

// ============================================================================
// GOVERN - Emergency Protocols & Community Decisions
// ============================================================================

/** Medical equipment requiring power */
export interface MedicalEquipment {
  type: 'oxygen_concentrator' | 'cpap' | 'bipap' | 'ventilator' | 'dialysis' |
        'infusion_pump' | 'refrigerated_meds' | 'electric_wheelchair' | 'other';
  powerRequirementWatts: number;
  batteryBackupHours?: number;
  criticality: 'life_sustaining' | 'important' | 'convenience';
}

/** Household energy profile */
export interface HouseholdProfile {
  id: string;
  ownerDid: string;
  name: string;
  zipCode: string;
  coordinates?: { lat: number; lng: number };
  utilityProvider?: string;
  ratePlan?: string;
  buildingType: 'single_family' | 'multi_family' | 'apartment' | 'condo' | 'mobile' | 'commercial';
  squareFeet?: number;
  heatingType: 'electric' | 'gas' | 'oil' | 'propane' | 'heat_pump' | 'wood' | 'none';
  coolingType: 'central_ac' | 'window_ac' | 'heat_pump' | 'evaporative' | 'none';
  waterHeatingType: 'electric' | 'gas' | 'solar' | 'heat_pump' | 'tankless';
  occupants: number;
  medicalEquipment: MedicalEquipment[];
  evCharging: boolean;
  assets: EnergyAsset[];
  emergencyPriority: EnergyPriority;
  createdAt: number;
  updatedAt: number;
}

// ============================================================================
// REAL WORLD DATA - Utility Rates & Solar Potential
// ============================================================================

/** Utility rate structure */
export interface UtilityRate {
  utilityName: string;
  ratePlanName: string;
  effectiveDate: number;
  rates: {
    period: 'flat' | 'peak' | 'off_peak' | 'mid_peak' | 'super_off_peak';
    season: 'summer' | 'winter' | 'all';
    ratePerKwh: number;
    hoursApplicable?: string;
  }[];
  monthlyServiceCharge: number;
  netMeteringAvailable: boolean;
  netMeteringRate?: number;
}

/** Solar potential for a location */
export interface SolarPotential {
  zipCode: string;
  coordinates: { lat: number; lng: number };
  annualGhi: number;
  annualDni: number;
  annualDhi: number;
  capacityFactor: number;
  annualProductionPerKw: number;
  monthlyProduction: { month: number; productionPerKw: number }[];
  federalItcPercent: number;
  averageCostPerWatt: number;
  estimatedSystemCost: number;
}

/** Texas utility rates (major providers) */
export const TEXAS_UTILITY_RATES: Record<string, UtilityRate> = {
  'oncor_tdu': {
    utilityName: 'Oncor (TDU charges only)',
    ratePlanName: 'Standard Delivery',
    effectiveDate: Date.now(),
    rates: [{ period: 'flat', season: 'all', ratePerKwh: 0.0422 }],
    monthlyServiceCharge: 3.42,
    netMeteringAvailable: false,
  },
  'txu_free_nights': {
    utilityName: 'TXU Energy',
    ratePlanName: 'Free Nights',
    effectiveDate: Date.now(),
    rates: [
      { period: 'peak', season: 'all', ratePerKwh: 0.189, hoursApplicable: '6am-9pm' },
      { period: 'off_peak', season: 'all', ratePerKwh: 0.0, hoursApplicable: '9pm-6am' }
    ],
    monthlyServiceCharge: 9.95,
    netMeteringAvailable: false,
  },
  'reliant_truly_free': {
    utilityName: 'Reliant',
    ratePlanName: 'Truly Free Weekends',
    effectiveDate: Date.now(),
    rates: [
      { period: 'peak', season: 'all', ratePerKwh: 0.165, hoursApplicable: 'Mon-Fri' },
      { period: 'off_peak', season: 'all', ratePerKwh: 0.0, hoursApplicable: 'Sat-Sun' }
    ],
    monthlyServiceCharge: 4.95,
    netMeteringAvailable: false,
  },
  'green_mountain': {
    utilityName: 'Green Mountain Energy',
    ratePlanName: 'Pollution Free',
    effectiveDate: Date.now(),
    rates: [{ period: 'flat', season: 'all', ratePerKwh: 0.149 }],
    monthlyServiceCharge: 0,
    netMeteringAvailable: true,
    netMeteringRate: 0.08,
  },
};

/** Average Texas electricity costs by city ($/kWh, all-in) */
export const TEXAS_AVERAGE_RATES: Record<string, number> = {
  'Dallas': 0.142,
  'Richardson': 0.138,
  'Plano': 0.141,
  'Frisco': 0.145,
  'McKinney': 0.139,
  'Houston': 0.135,
  'Austin': 0.128,
  'San Antonio': 0.119,
  'Fort Worth': 0.140,
};

/** DFW Solar potential (NREL data) */
export const DFW_SOLAR_POTENTIAL: SolarPotential = {
  zipCode: '75080',
  coordinates: { lat: 32.95, lng: -96.73 },
  annualGhi: 5.2,
  annualDni: 5.8,
  annualDhi: 2.1,
  capacityFactor: 0.18,
  annualProductionPerKw: 1580,
  monthlyProduction: [
    { month: 1, productionPerKw: 105 }, { month: 2, productionPerKw: 115 },
    { month: 3, productionPerKw: 140 }, { month: 4, productionPerKw: 150 },
    { month: 5, productionPerKw: 160 }, { month: 6, productionPerKw: 165 },
    { month: 7, productionPerKw: 170 }, { month: 8, productionPerKw: 165 },
    { month: 9, productionPerKw: 145 }, { month: 10, productionPerKw: 130 },
    { month: 11, productionPerKw: 110 }, { month: 12, productionPerKw: 95 },
  ],
  federalItcPercent: 30,
  averageCostPerWatt: 2.75,
  estimatedSystemCost: 22000,
};

/** Battery storage costs (2024 market) */
export const BATTERY_COSTS = {
  tesla_powerwall_3: {
    name: 'Tesla Powerwall 3',
    capacityKwh: 13.5,
    powerKw: 11.5,
    cost: 9200,
    installationCost: 2500,
    warrantyYears: 10,
  },
  enphase_iq_5p: {
    name: 'Enphase IQ Battery 5P',
    capacityKwh: 5,
    powerKw: 3.84,
    cost: 5500,
    installationCost: 1500,
    warrantyYears: 15,
  },
  lg_resu_16h: {
    name: 'LG RESU 16H Prime',
    capacityKwh: 16,
    powerKw: 7,
    cost: 8500,
    installationCost: 2000,
    warrantyYears: 10,
  },
  generac_pwrcell: {
    name: 'Generac PWRcell',
    capacityKwh: 9,
    powerKw: 4.5,
    cost: 7500,
    installationCost: 2000,
    warrantyYears: 10,
  },
};

/** Solar panel costs (2024 market, per watt installed) */
export const SOLAR_COSTS = {
  budget: { name: 'Budget Tier', costPerWatt: 2.40, brands: ['Hanwha Q Cells', 'Canadian Solar'] },
  midRange: { name: 'Mid-Range', costPerWatt: 2.75, brands: ['REC', 'LG', 'Panasonic'] },
  premium: { name: 'Premium', costPerWatt: 3.20, brands: ['SunPower', 'REC Alpha'] },
};

// ============================================================================
// COMMUNITY ENERGY STATUS
// ============================================================================

/** Community energy summary */
export interface CommunityEnergySummary {
  communityId: string;
  timestamp: number;
  totalGenerationCapacityKw: number;
  totalStorageCapacityKwh: number;
  householdsWithSolar: number;
  householdsWithStorage: number;
  totalHouseholds: number;
  currentGenerationKw: number;
  currentConsumptionKw: number;
  currentStorageKwh: number;
  gridImportKw: number;
  gridExportKw: number;
  todayGenerationKwh: number;
  todayConsumptionKwh: number;
  status: 'surplus' | 'balanced' | 'deficit' | 'emergency';
  activeOffers: number;
  activeRequests: number;
  todayTrades: number;
  todayTradedKwh: number;
}

/** Grid status for area */
export interface GridStatus {
  region: string;
  timestamp: number;
  gridFrequencyHz: number;
  systemLoadMw: number;
  availableCapacityMw: number;
  reserveMarginPercent: number;
  status: 'normal' | 'conservation' | 'watch' | 'emergency' | 'blackout';
  alertLevel?: number;
  message?: string;
  currentPricePerMwh?: number;
}

// ============================================================================
// ENERGY SOVEREIGNTY SERVICE
// ============================================================================

/**
 * Energy Sovereignty Service
 *
 * Consumer-first interface for community energy management.
 * Five Pillars: GENERATE, STORE, SHARE, CONSERVE, GOVERN
 */
export class EnergySovereigntyService {
  private householdProfile: HouseholdProfile | null = null;
  private assets: EnergyAsset[] = [];

  // ============================================================================
  // PROFILE MANAGEMENT
  // ============================================================================

  createProfile(params: {
    name: string;
    zipCode: string;
    buildingType: HouseholdProfile['buildingType'];
    squareFeet?: number;
    heatingType: HouseholdProfile['heatingType'];
    coolingType: HouseholdProfile['coolingType'];
    waterHeatingType: HouseholdProfile['waterHeatingType'];
    occupants: number;
    medicalEquipment?: MedicalEquipment[];
    evCharging?: boolean;
  }): HouseholdProfile {
    const medicalEquipment = params.medicalEquipment || [];

    let emergencyPriority: EnergyPriority = 'standard';
    if (medicalEquipment.some(e => e.criticality === 'life_sustaining')) {
      emergencyPriority = 'critical_medical';
    } else if (medicalEquipment.some(e => e.criticality === 'important')) {
      emergencyPriority = 'essential_safety';
    }

    const profile: HouseholdProfile = {
      id: `household-${Date.now()}`,
      ownerDid: `did:mycelix:${Date.now()}`,
      name: params.name,
      zipCode: params.zipCode,
      buildingType: params.buildingType,
      squareFeet: params.squareFeet,
      heatingType: params.heatingType,
      coolingType: params.coolingType,
      waterHeatingType: params.waterHeatingType,
      occupants: params.occupants,
      medicalEquipment,
      evCharging: params.evCharging || false,
      assets: [],
      emergencyPriority,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };

    this.householdProfile = profile;
    return profile;
  }

  getProfile(): HouseholdProfile | null {
    return this.householdProfile;
  }

  updateMedicalEquipment(equipment: MedicalEquipment[]): void {
    if (!this.householdProfile) return;

    this.householdProfile.medicalEquipment = equipment;

    if (equipment.some(e => e.criticality === 'life_sustaining')) {
      this.householdProfile.emergencyPriority = 'critical_medical';
    } else if (equipment.some(e => e.criticality === 'important')) {
      this.householdProfile.emergencyPriority = 'essential_safety';
    } else {
      this.householdProfile.emergencyPriority = 'standard';
    }

    this.householdProfile.updatedAt = Date.now();
  }

  // ============================================================================
  // GENERATE - Asset Registration
  // ============================================================================

  registerSolarSystem(params: {
    make: string;
    model: string;
    panelCount: number;
    panelWattage: number;
    installationType: EnergyAsset['location']['installationType'];
    purchaseCost: number;
    installationCost: number;
    installDate: Date;
  }): EnergyAsset {
    const capacityKw = (params.panelCount * params.panelWattage) / 1000;
    const ageYears = (Date.now() - params.installDate.getTime()) / (365.25 * 24 * 60 * 60 * 1000);

    const asset: EnergyAsset = {
      id: `solar-${Date.now()}`,
      ownerId: this.householdProfile?.id || 'unknown',
      ownerDid: this.householdProfile?.ownerDid || 'unknown',
      type: 'solar_pv',
      make: params.make,
      model: params.model,
      location: {
        zipCode: this.householdProfile?.zipCode || '00000',
        installationType: params.installationType,
      },
      capacityKw,
      efficiencyPercent: 21,
      degradationPerYear: 0.5,
      currentAge: ageYears,
      purchaseCost: params.purchaseCost,
      installationCost: params.installationCost,
      annualMaintenance: 150,
      warrantyYears: 25,
      gridConnected: true,
      netMeteringEnabled: false,
      status: 'active',
      verified: false,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };

    this.assets.push(asset);
    if (this.householdProfile) {
      this.householdProfile.assets.push(asset);
    }

    return asset;
  }

  registerBattery(params: {
    make: string;
    model: string;
    capacityKwh: number;
    powerKw: number;
    purchaseCost: number;
    installationCost: number;
    installDate: Date;
  }): StorageAsset {
    const ageYears = (Date.now() - params.installDate.getTime()) / (365.25 * 24 * 60 * 60 * 1000);

    const asset: StorageAsset = {
      id: `battery-${Date.now()}`,
      ownerId: this.householdProfile?.id || 'unknown',
      ownerDid: this.householdProfile?.ownerDid || 'unknown',
      type: 'battery_lithium',
      storageType: 'battery_lithium',
      make: params.make,
      model: params.model,
      location: {
        zipCode: this.householdProfile?.zipCode || '00000',
        installationType: 'ground_mount',
      },
      capacityKw: params.powerKw,
      capacityKwh: params.capacityKwh,
      usableCapacityKwh: params.capacityKwh * 0.9,
      depthOfDischarge: 0.9,
      roundTripEfficiency: 0.9,
      maxChargeRateKw: params.powerKw,
      maxDischargeRateKw: params.powerKw,
      cycleLife: 6000,
      currentCycles: Math.round(ageYears * 365),
      stateOfCharge: 0.5,
      stateOfHealth: Math.max(0.8, 1 - (ageYears * 0.02)),
      efficiencyPercent: 90,
      degradationPerYear: 2,
      currentAge: ageYears,
      purchaseCost: params.purchaseCost,
      installationCost: params.installationCost,
      annualMaintenance: 50,
      warrantyYears: 10,
      gridConnected: true,
      netMeteringEnabled: false,
      status: 'active',
      verified: false,
      canProvideBackup: true,
      islandingCapable: true,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };

    this.assets.push(asset);
    if (this.householdProfile) {
      this.householdProfile.assets.push(asset);
    }

    return asset;
  }

  getAssets(): EnergyAsset[] {
    return this.assets;
  }

  getTotalCapacity(): { generationKw: number; storageKwh: number } {
    let generationKw = 0;
    let storageKwh = 0;

    for (const asset of this.assets) {
      if (['solar_pv', 'wind_turbine', 'micro_hydro'].includes(asset.type as string)) {
        const degradedCapacity = asset.capacityKw * Math.pow(1 - asset.degradationPerYear / 100, asset.currentAge);
        generationKw += degradedCapacity;
      }
      if (asset.capacityKwh) {
        storageKwh += asset.capacityKwh;
      }
    }

    return { generationKw, storageKwh };
  }

  // ============================================================================
  // CONSERVE - Efficiency & Tracking
  // ============================================================================

  estimateMonthlyBill(usageKwh: number): {
    energyCost: number;
    deliveryCost: number;
    fixedCharges: number;
    totalCost: number;
    costPerKwh: number;
  } {
    const city = this.householdProfile?.zipCode?.startsWith('75') ? 'Richardson' : 'Dallas';
    const avgRate = TEXAS_AVERAGE_RATES[city] || 0.14;

    const energyCost = usageKwh * (avgRate - 0.042);
    const deliveryCost = usageKwh * 0.042;
    const fixedCharges = 9.95;
    const totalCost = energyCost + deliveryCost + fixedCharges;

    return {
      energyCost: Math.round(energyCost * 100) / 100,
      deliveryCost: Math.round(deliveryCost * 100) / 100,
      fixedCharges,
      totalCost: Math.round(totalCost * 100) / 100,
      costPerKwh: Math.round((totalCost / usageKwh) * 1000) / 1000,
    };
  }

  getConservationTips(): ConservationTip[] {
    const tips: ConservationTip[] = [];

    if (this.householdProfile?.heatingType === 'electric') {
      tips.push({
        id: 'tip-heat-pump',
        category: 'hvac',
        title: 'Consider a Heat Pump',
        description: 'Heat pumps are 2-3x more efficient than electric resistance heating.',
        estimatedSavingsKwh: 3000,
        estimatedSavingsUsd: 420,
        implementationCost: 8000,
        paybackMonths: 228,
        difficulty: 'professional',
        diyPossible: false,
      });
    }

    if (this.householdProfile?.waterHeatingType === 'electric') {
      tips.push({
        id: 'tip-hpwh',
        category: 'water_heating',
        title: 'Heat Pump Water Heater',
        description: 'Uses 70% less energy than standard electric water heaters.',
        estimatedSavingsKwh: 2000,
        estimatedSavingsUsd: 280,
        implementationCost: 2500,
        paybackMonths: 107,
        difficulty: 'professional',
        diyPossible: false,
      });
    }

    tips.push(
      {
        id: 'tip-led',
        category: 'lighting',
        title: 'Switch to LED Bulbs',
        description: 'LED bulbs use 75% less energy and last 25x longer than incandescent.',
        estimatedSavingsKwh: 500,
        estimatedSavingsUsd: 70,
        implementationCost: 50,
        paybackMonths: 9,
        difficulty: 'easy',
        diyPossible: true,
      },
      {
        id: 'tip-thermostat',
        category: 'hvac',
        title: 'Smart Thermostat',
        description: 'Automatically adjusts temperature when you\'re away or sleeping.',
        estimatedSavingsKwh: 1000,
        estimatedSavingsUsd: 140,
        implementationCost: 200,
        paybackMonths: 17,
        difficulty: 'easy',
        diyPossible: true,
      },
      {
        id: 'tip-phantom',
        category: 'behavioral',
        title: 'Eliminate Phantom Loads',
        description: 'Unplug devices or use smart power strips. Electronics draw power even when "off".',
        estimatedSavingsKwh: 300,
        estimatedSavingsUsd: 42,
        implementationCost: 30,
        paybackMonths: 9,
        difficulty: 'easy',
        diyPossible: true,
      }
    );

    return tips;
  }

  // ============================================================================
  // SOLAR ECONOMICS
  // ============================================================================

  calculateSolarEconomics(systemSizeKw: number): {
    systemCost: number;
    federalTaxCredit: number;
    netCost: number;
    annualProduction: number;
    annualSavings: number;
    paybackYears: number;
    twentyFiveYearSavings: number;
    co2AvoidedTons: number;
  } {
    const potential = DFW_SOLAR_POTENTIAL;
    const tier = SOLAR_COSTS.midRange;

    const systemCost = systemSizeKw * 1000 * tier.costPerWatt;
    const federalTaxCredit = systemCost * (potential.federalItcPercent / 100);
    const netCost = systemCost - federalTaxCredit;

    const annualProduction = systemSizeKw * potential.annualProductionPerKw;

    const city = this.householdProfile?.zipCode?.startsWith('75') ? 'Richardson' : 'Dallas';
    const rate = TEXAS_AVERAGE_RATES[city] || 0.14;
    const annualSavings = annualProduction * rate;

    const lifetimeProduction = annualProduction * 25 * 0.88;
    const twentyFiveYearSavings = lifetimeProduction * rate - netCost;
    const paybackYears = netCost / annualSavings;
    const co2AvoidedTons = (lifetimeProduction * 0.9) / 2000;

    return {
      systemCost: Math.round(systemCost),
      federalTaxCredit: Math.round(federalTaxCredit),
      netCost: Math.round(netCost),
      annualProduction: Math.round(annualProduction),
      annualSavings: Math.round(annualSavings),
      paybackYears: Math.round(paybackYears * 10) / 10,
      twentyFiveYearSavings: Math.round(twentyFiveYearSavings),
      co2AvoidedTons: Math.round(co2AvoidedTons),
    };
  }

  calculateBatteryEconomics(batteryType: keyof typeof BATTERY_COSTS): {
    totalCost: number;
    capacityKwh: number;
    costPerKwh: number;
    backupHours: number;
    annualArbitrageSavings: number;
    paybackYears: number;
    valueProposition: string;
  } {
    const battery = BATTERY_COSTS[batteryType];
    const totalCost = battery.cost + battery.installationCost;
    const costPerKwh = totalCost / battery.capacityKwh;

    const avgHomeLoadKw = 2;
    const backupHours = battery.capacityKwh / avgHomeLoadKw;

    const cyclesPerYear = 300;
    const arbitragePerCycle = battery.capacityKwh * 0.9 * 0.10;
    const annualArbitrageSavings = cyclesPerYear * arbitragePerCycle;
    const backupValuePerYear = 100;
    const paybackYears = totalCost / (annualArbitrageSavings + backupValuePerYear);

    let valueProposition: string;
    if (this.householdProfile?.medicalEquipment.some(e => e.criticality === 'life_sustaining')) {
      valueProposition = 'Critical for medical equipment backup - consider primary investment';
    } else if (paybackYears < 10) {
      valueProposition = 'Good economics with TOU arbitrage - reasonable investment';
    } else {
      valueProposition = 'Best for backup power peace of mind rather than pure economics';
    }

    return {
      totalCost,
      capacityKwh: battery.capacityKwh,
      costPerKwh: Math.round(costPerKwh),
      backupHours: Math.round(backupHours * 10) / 10,
      annualArbitrageSavings: Math.round(annualArbitrageSavings),
      paybackYears: Math.round(paybackYears * 10) / 10,
      valueProposition,
    };
  }

  // ============================================================================
  // COMMUNITY STATUS
  // ============================================================================

  getCommunityStatus(): CommunityEnergySummary {
    return {
      communityId: 'richardson-energy-coop',
      timestamp: Date.now(),
      totalGenerationCapacityKw: 450,
      totalStorageCapacityKwh: 280,
      householdsWithSolar: 45,
      householdsWithStorage: 18,
      totalHouseholds: 200,
      currentGenerationKw: 320,
      currentConsumptionKw: 380,
      currentStorageKwh: 180,
      gridImportKw: 60,
      gridExportKw: 0,
      todayGenerationKwh: 1800,
      todayConsumptionKwh: 2400,
      status: 'balanced',
      activeOffers: 3,
      activeRequests: 2,
      todayTrades: 5,
      todayTradedKwh: 45,
    };
  }

  getGridStatus(): GridStatus {
    return {
      region: 'ERCOT',
      timestamp: Date.now(),
      gridFrequencyHz: 60.01,
      systemLoadMw: 52000,
      availableCapacityMw: 75000,
      reserveMarginPercent: 18,
      status: 'normal',
      currentPricePerMwh: 35,
    };
  }
}

// ============================================================================
// SINGLETON EXPORT
// ============================================================================

export const energySovereignty = new EnergySovereigntyService();
