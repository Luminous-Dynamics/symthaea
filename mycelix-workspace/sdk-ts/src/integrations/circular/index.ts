// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * Circular Economy SDK Module
 *
 * TypeScript types for interacting with the waste/circular economy zomes:
 * waste-registry, waste-collection, compost-control, circular-marketplace.
 *
 * These mirror the Holochain entry types and match the Rust serde serialization format.
 */

// ============================================================================
// Waste Registry Types
// ============================================================================

export type WasteCategory =
  | 'Organic'
  | 'Recyclable'
  | 'Hazardous'
  | 'Electronic'
  | 'Mixed'
  | 'Inert';

export type ContaminationLevel =
  | 'Clean'
  | 'LightlyContaminated'
  | 'Contaminated'
  | 'Hazardous';

export type EndOfLifeStrategy =
  | 'MechanicalRecycling'
  | 'ChemicalRecycling'
  | 'Biodegradable'
  | 'IndustrialCompost'
  | 'Downcycle'
  | 'Landfill';

export type WasteStreamStatus =
  | 'Registered'
  | 'Classified'
  | 'Routed'
  | 'Collected'
  | 'Processing'
  | 'Completed';

export type ClassificationMethod =
  | 'Manual'
  | 'SensorBased'
  | 'VisionAI'
  | 'MaterialPassport';

export type FacilityType =
  | 'MRF'
  | 'Composter'
  | 'AnaerobicDigester'
  | 'ChemicalRecycler'
  | 'Landfill'
  | 'TransferStation';

export type FacilityStatus =
  | 'Active'
  | 'Maintenance'
  | 'AtCapacity'
  | 'Closed';

export interface WasteStream {
  id: string;
  source_did: string;
  material_passport_hash: Uint8Array | null;
  waste_category: WasteCategory;
  subcategory: string;
  quantity_kg: number;
  contamination_level: ContaminationLevel;
  recommended_eol: EndOfLifeStrategy;
  location_lat: number;
  location_lon: number;
  status: WasteStreamStatus;
  created_at: number;
}

export interface WasteClassification {
  stream_hash: Uint8Array;
  classifier_did: string;
  category: WasteCategory;
  subcategory: string;
  contamination_level: ContaminationLevel;
  confidence: number;
  method: ClassificationMethod;
  classified_at: number;
}

export interface WasteFacility {
  id: string;
  name: string;
  facility_type: FacilityType;
  accepts: WasteCategory[];
  capacity_kg_per_day: number;
  current_load_kg: number;
  location_lat: number;
  location_lon: number;
  operator_did: string;
  status: FacilityStatus;
}

export interface WasteRouteFactors {
  category_match: number;
  proximity_score: number;
  capacity_score: number;
  contamination_compatibility: number;
}

export interface DiversionRateResult {
  total_kg: number;
  composted_kg: number;
  recycled_kg: number;
  landfilled_kg: number;
  diversion_rate_pct: number;
}

export interface ContaminationFeedback {
  contamination_detected: boolean;
  severity: number;
  contaminants: string[];
  alert_emitted: boolean;
}

// ============================================================================
// Compost Control Types
// ============================================================================

export type CompostMethod =
  | 'Windrow'
  | 'StaticAerated'
  | 'InVessel'
  | 'Vermicompost'
  | 'Bokashi';

export type CompostBatchStatus =
  | 'Mixing'
  | 'ActiveComposting'
  | 'Curing'
  | 'Screening'
  | 'Complete'
  | 'Failed';

export type CompostActionType =
  | 'Turn'
  | 'AddWater'
  | 'AddBulking'
  | 'AdjustAeration'
  | 'HarvestScreen';

export type ActionRecommender = 'Sensor' | 'AI' | 'Manual';

export interface CompostInput {
  stream_hash: Uint8Array;
  quantity_kg: number;
  resource_type: string;
}

export interface CompostReading {
  batch_hash: Uint8Array;
  sensor_id: string;
  temperature_c: number;
  moisture_pct: number;
  oxygen_pct: number;
  ph: number;
  timestamp_us: number;
}

export interface RecommendedAction {
  action_type: CompostActionType;
  reason: string;
  urgency: string;
}

export interface BatchEvaluation {
  current_phase: string;
  recommended_actions: RecommendedAction[];
  phase_transition: string | null;
  temperature_status: string;
  moisture_status: string;
  oxygen_status: string;
  ph_status: string;
}

export interface CarbonAttribution {
  waste_diverted_kg: number;
  co2e_avoided_tonnes: number;
  methodology: string;
}

export interface NutrientEstimate {
  total_kg: number;
  nitrogen_pct: number;
  phosphorus_pct: number;
  potassium_pct: number;
}

// ============================================================================
// Marketplace Types
// ============================================================================

export type SecondaryMaterialType =
  | 'Compost'
  | 'Vermicompost'
  | 'Digestate'
  | 'Biochar'
  | 'RecycledPlastic'
  | 'RecycledMetal'
  | 'RecycledGlass'
  | 'RecycledFiber'
  | 'Mulch';

export type MaterialQualityGrade = 'Premium' | 'Standard' | 'Industrial';

export type ListingStatus = 'Available' | 'Reserved' | 'Sold' | 'Withdrawn';

export type OrderStatus =
  | 'Placed'
  | 'Confirmed'
  | 'InTransit'
  | 'Delivered'
  | 'Cancelled';

export interface NutrientInfo {
  nitrogen_pct: number;
  phosphorus_pct: number;
  potassium_pct: number;
}

export interface SecondaryMaterialListing {
  facility_hash: Uint8Array;
  batch_hash: Uint8Array | null;
  material_type: SecondaryMaterialType;
  quantity_kg: number;
  quality_grade: MaterialQualityGrade;
  nutrient_info: NutrientInfo | null;
  price_per_kg: number;
  location_lat: number;
  location_lon: number;
  available_from: number;
  status: ListingStatus;
  seller_did: string;
}

export interface DemandMatchInput {
  material_type: SecondaryMaterialType;
  min_quantity_kg: number;
  min_quality: MaterialQualityGrade;
  buyer_lat: number;
  buyer_lon: number;
  max_distance_km: number;
}

export interface ListingMatch {
  listing_hash: Uint8Array;
  material_type: string;
  quantity_kg: number;
  quality_grade: string;
  price_per_kg: number;
  distance_km: number;
  match_score: number;
}

// ============================================================================
// Composting Threshold Constants (EPA-cited)
// ============================================================================

/** Thermophilic phase minimum temperature (C) — EPA 40 CFR 503 */
export const THERMOPHILIC_MIN_C = 55.0;
/** Thermophilic phase maximum temperature (C) */
export const THERMOPHILIC_MAX_C = 65.0;
/** Moisture target range (%) — USCC guidelines */
export const MOISTURE_MIN_PCT = 40.0;
export const MOISTURE_MAX_PCT = 65.0;
export const MOISTURE_OPTIMAL_PCT = 55.0;
/** Minimum oxygen for aerobic decomposition (%) */
export const OXYGEN_MIN_PCT = 5.0;
/** Optimal C:N ratio range — Cornell Waste Management Institute */
export const CN_RATIO_MIN = 25.0;
export const CN_RATIO_MAX = 30.0;
/** EPA WARM model: composting 1 tonne food waste avoids ~0.06 tCO2e */
export const COMPOST_CO2E_AVOIDED_PER_TONNE = 0.06;
