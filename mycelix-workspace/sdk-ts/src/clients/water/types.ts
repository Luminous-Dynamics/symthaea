// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Water hApp Client Types
 *
 * Type definitions for the Water hApp Master SDK client,
 * providing coverage of flow, purity, and steward zomes.
 *
 * @module @mycelix/sdk/clients/water/types
 */

import type { ActionHash } from '../../generated/common';

// ============================================================================
// Common Types
// ============================================================================

export type Timestamp = number;

export interface PaginationFilter {
  limit?: number;
  offset?: number;
}

// ============================================================================
// Flow Types
// ============================================================================

/** Water source type */
export type WaterSourceType = 'Well' | 'Spring' | 'River' | 'Lake' | 'Reservoir' | 'Rainwater' | 'Desalination' | 'Recycled';

/** A registered water source */
export interface WaterSource {
  id: ActionHash;
  name: string;
  description: string;
  sourceType: WaterSourceType;
  ownerDid: string;
  latitude: number;
  longitude: number;
  watershedId?: ActionHash;
  /** Estimated yield in liters per day */
  estimatedYieldLitersPerDay: number;
  /** Current available allocation in liters */
  availableAllocation: number;
  status: 'Active' | 'Seasonal' | 'Depleted' | 'Contaminated' | 'Decommissioned';
  createdAt: Timestamp;
  updatedAt: Timestamp;
}

/** Input for registering a water source */
export interface RegisterWaterSourceInput {
  name: string;
  description: string;
  sourceType: WaterSourceType;
  latitude: number;
  longitude: number;
  watershedId?: ActionHash;
  estimatedYieldLitersPerDay: number;
}

/** A water share allocation */
export interface WaterShare {
  id: ActionHash;
  sourceId: ActionHash;
  holderDid: string;
  /** Allocated liters per day */
  dailyAllocationLiters: number;
  /** Priority level (lower = higher priority) */
  priority: number;
  purpose: 'Drinking' | 'Agriculture' | 'Industrial' | 'Ecological' | 'Community';
  active: boolean;
  grantedAt: Timestamp;
  expiresAt?: Timestamp;
}

/** Input for allocating water shares */
export interface AllocateSharesInput {
  sourceId: ActionHash;
  holderDid: string;
  dailyAllocationLiters: number;
  priority?: number;
  purpose: WaterShare['purpose'];
  expiresAt?: Timestamp;
}

/** Water credit transfer input */
export interface TransferCreditsInput {
  fromDid: string;
  toDid: string;
  sourceId: ActionHash;
  liters: number;
  reason: string;
}

/** Water usage record */
export interface WaterUsageRecord {
  id: ActionHash;
  sourceId: ActionHash;
  userDid: string;
  litersUsed: number;
  purpose: WaterShare['purpose'];
  recordedAt: Timestamp;
}

/** Water balance for a user */
export interface WaterBalance {
  did: string;
  /** Total daily allocation across all sources */
  totalDailyAllocation: number;
  /** Total used today */
  usedToday: number;
  /** Remaining for today */
  remainingToday: number;
  /** Credits available for transfer */
  transferableCredits: number;
}

// ============================================================================
// Purity Types
// ============================================================================

/** Water quality parameters */
export interface QualityReading {
  id: ActionHash;
  sourceId: ActionHash;
  testerDid: string;
  /** pH level (0-14) */
  ph: number;
  /** Turbidity in NTU */
  turbidity: number;
  /** Total dissolved solids in ppm */
  tds: number;
  /** Dissolved oxygen in mg/L */
  dissolvedOxygen?: number;
  /** E. coli count per 100mL */
  eColiCount?: number;
  /** Temperature in Celsius */
  temperature?: number;
  /** Additional contaminants as key-value */
  contaminants: Record<string, number>;
  /** Whether the water is potable */
  potable: boolean;
  /** Notes from tester */
  notes?: string;
  testedAt: Timestamp;
}

/** Input for submitting a quality reading */
export interface SubmitQualityReadingInput {
  sourceId: ActionHash;
  ph: number;
  turbidity: number;
  tds: number;
  dissolvedOxygen?: number;
  eColiCount?: number;
  temperature?: number;
  contaminants?: Record<string, number>;
  notes?: string;
}

/** A water quality alert */
export interface WaterAlert {
  id: ActionHash;
  sourceId: ActionHash;
  raisedByDid: string;
  severity: 'Notice' | 'Warning' | 'Critical';
  description: string;
  contaminant?: string;
  readingId?: ActionHash;
  status: 'Active' | 'Investigating' | 'Resolved';
  raisedAt: Timestamp;
  resolvedAt?: Timestamp;
}

/** Input for raising an alert */
export interface RaiseAlertInput {
  sourceId: ActionHash;
  severity: WaterAlert['severity'];
  description: string;
  contaminant?: string;
  readingId?: ActionHash;
}

// ============================================================================
// Steward Types
// ============================================================================

/** A watershed boundary */
export interface Watershed {
  id: ActionHash;
  name: string;
  description: string;
  stewardDid: string;
  /** GeoJSON boundary (stringified) */
  boundaryGeoJson: string;
  /** Total estimated annual yield in megaliters */
  annualYieldMegaliters: number;
  /** Number of registered sources */
  sourceCount: number;
  /** Number of active water rights */
  activeRightsCount: number;
  createdAt: Timestamp;
  updatedAt: Timestamp;
}

/** Input for defining a watershed */
export interface DefineWatershedInput {
  name: string;
  description: string;
  boundaryGeoJson: string;
  annualYieldMegaliters: number;
}

/** A water right */
export interface WaterRight {
  id: ActionHash;
  watershedId: ActionHash;
  holderDid: string;
  /** Annual allocation in megaliters */
  annualAllocationMegaliters: number;
  rightType: 'Riparian' | 'Appropriative' | 'Prescriptive' | 'Traditional' | 'Community';
  priority: number;
  purpose: string;
  conditions: string;
  active: boolean;
  grantedAt: Timestamp;
  expiresAt?: Timestamp;
}

/** Input for registering a water right */
export interface RegisterWaterRightInput {
  watershedId: ActionHash;
  holderDid: string;
  annualAllocationMegaliters: number;
  rightType: WaterRight['rightType'];
  priority?: number;
  purpose: string;
  conditions?: string;
  expiresAt?: Timestamp;
}

/** Input for transferring a water right */
export interface TransferWaterRightInput {
  rightId: ActionHash;
  toDid: string;
  reason: string;
}

/** A water dispute */
export interface WaterDispute {
  id: ActionHash;
  watershedId: ActionHash;
  filedByDid: string;
  againstDid: string;
  rightIds: ActionHash[];
  description: string;
  status: 'Filed' | 'UnderReview' | 'Mediation' | 'Resolved' | 'Escalated';
  resolution?: string;
  filedAt: Timestamp;
  resolvedAt?: Timestamp;
}

/** Input for filing a dispute */
export interface FileDisputeInput {
  watershedId: ActionHash;
  againstDid: string;
  rightIds: ActionHash[];
  description: string;
}

/** A traditional water practice */
export interface TraditionalPractice {
  id: ActionHash;
  watershedId: ActionHash;
  name: string;
  description: string;
  communityDid: string;
  /** Whether this practice has legal recognition */
  recognized: boolean;
  createdAt: Timestamp;
}

// ============================================================================
// Error Types
// ============================================================================

export type WaterErrorCode =
  | 'CONNECTION_ERROR'
  | 'ZOME_CALL_ERROR'
  | 'NOT_FOUND'
  | 'UNAUTHORIZED'
  | 'INVALID_INPUT'
  | 'INSUFFICIENT_ALLOCATION'
  | 'SOURCE_DEPLETED'
  | 'CONTAMINATION_ALERT'
  | 'RIGHT_EXPIRED'
  | 'DISPUTE_ERROR';

export class WaterError extends Error {
  constructor(
    public readonly code: WaterErrorCode,
    message: string,
    public readonly details?: unknown
  ) {
    super(message);
    this.name = 'WaterError';
  }
}
