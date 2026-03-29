// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Emergency hApp Client Types
 *
 * Type definitions for the Emergency hApp Master SDK client,
 * providing coverage of incidents, coordination, and shelters zomes.
 *
 * @module @mycelix/sdk/clients/emergency/types
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
// Incident Types
// ============================================================================

/** Disaster severity levels */
export type DisasterSeverity = 'Advisory' | 'Watch' | 'Warning' | 'Emergency' | 'Catastrophic';

/** Disaster type classification */
export type DisasterType =
  | 'Flood'
  | 'Fire'
  | 'Earthquake'
  | 'Hurricane'
  | 'Tornado'
  | 'Pandemic'
  | 'IndustrialAccident'
  | 'Infrastructure'
  | 'Other';

/** A declared disaster or emergency event */
export interface Disaster {
  id: ActionHash;
  title: string;
  description: string;
  disasterType: DisasterType;
  severity: DisasterSeverity;
  declaredByDid: string;
  /** Geographic center (lat, lon) */
  latitude: number;
  longitude: number;
  /** Affected radius in kilometers */
  radiusKm: number;
  /** Estimated affected population */
  estimatedAffected?: number;
  status: 'Declared' | 'Active' | 'Contained' | 'Recovery' | 'Ended';
  declaredAt: Timestamp;
  updatedAt: Timestamp;
  endedAt?: Timestamp;
}

/** Input for declaring a disaster */
export interface DeclareDisasterInput {
  title: string;
  description: string;
  disasterType: DisasterType;
  severity: DisasterSeverity;
  latitude: number;
  longitude: number;
  radiusKm: number;
  estimatedAffected?: number;
}

/** Input for updating disaster status */
export interface UpdateDisasterStatusInput {
  disasterId: ActionHash;
  status: Disaster['status'];
  severity?: DisasterSeverity;
  description?: string;
  radiusKm?: number;
  estimatedAffected?: number;
}

// ============================================================================
// Triage Types
// ============================================================================

/** Triage classification */
export type TriageLevel = 'Green' | 'Yellow' | 'Red' | 'Black';

/** A triage record for an individual */
export interface TriageRecord {
  id: ActionHash;
  disasterId: ActionHash;
  patientIdentifier: string;
  triageLevel: TriageLevel;
  notes: string;
  triageByDid: string;
  location: string;
  needsEvacuation: boolean;
  createdAt: Timestamp;
  updatedAt: Timestamp;
}

// ============================================================================
// Resource Types
// ============================================================================

/** Resource type classification */
export type ResourceType = 'Medical' | 'Food' | 'Water' | 'Shelter' | 'Transport' | 'Power' | 'Communication' | 'Personnel';

/** An emergency resource */
export interface EmergencyResource {
  id: ActionHash;
  disasterId: ActionHash;
  resourceType: ResourceType;
  name: string;
  description: string;
  quantity: number;
  unit: string;
  location: string;
  ownedByDid: string;
  status: 'Available' | 'Deployed' | 'InTransit' | 'Exhausted';
  createdAt: Timestamp;
  updatedAt: Timestamp;
}

// ============================================================================
// Coordination Types
// ============================================================================

/** A response team */
export interface Team {
  id: ActionHash;
  disasterId: ActionHash;
  name: string;
  leaderDid: string;
  members: string[];
  assignedZone?: string;
  specialization: string;
  status: 'Forming' | 'Active' | 'Stood Down';
  createdAt: Timestamp;
  updatedAt: Timestamp;
}

/** Input for forming a team */
export interface FormTeamInput {
  disasterId: ActionHash;
  name: string;
  specialization: string;
  initialMembers?: string[];
}

/** A situation report */
export interface SitRep {
  id: ActionHash;
  disasterId: ActionHash;
  teamId?: ActionHash;
  authorDid: string;
  summary: string;
  casualties?: number;
  rescued?: number;
  resourcesNeeded: string[];
  createdAt: Timestamp;
}

/** Input for submitting a situation report */
export interface SubmitSitRepInput {
  disasterId: ActionHash;
  teamId?: ActionHash;
  summary: string;
  casualties?: number;
  rescued?: number;
  resourcesNeeded?: string[];
}

/** A check-in from a team or individual */
export interface CheckIn {
  id: ActionHash;
  disasterId: ActionHash;
  agentDid: string;
  teamId?: ActionHash;
  latitude: number;
  longitude: number;
  status: 'OK' | 'NeedsAssistance' | 'Emergency';
  message?: string;
  checkedInAt: Timestamp;
}

/** Input for checking in */
export interface CheckInInput {
  disasterId: ActionHash;
  teamId?: ActionHash;
  latitude: number;
  longitude: number;
  status: CheckIn['status'];
  message?: string;
}

// ============================================================================
// Shelter Types
// ============================================================================

/** A registered shelter */
export interface Shelter {
  id: ActionHash;
  disasterId?: ActionHash;
  name: string;
  address: string;
  latitude: number;
  longitude: number;
  capacity: number;
  currentOccupancy: number;
  managerDid: string;
  amenities: string[];
  petFriendly: boolean;
  accessible: boolean;
  status: 'Open' | 'Full' | 'Closed';
  createdAt: Timestamp;
  updatedAt: Timestamp;
}

/** Input for registering a shelter */
export interface RegisterShelterInput {
  disasterId?: ActionHash;
  name: string;
  address: string;
  latitude: number;
  longitude: number;
  capacity: number;
  amenities?: string[];
  petFriendly?: boolean;
  accessible?: boolean;
}

/** An emergency message/broadcast */
export interface EmergencyMessage {
  id: ActionHash;
  disasterId: ActionHash;
  senderDid: string;
  priority: 'Info' | 'Warning' | 'Critical';
  subject: string;
  body: string;
  targetZone?: string;
  createdAt: Timestamp;
}

// ============================================================================
// Error Types
// ============================================================================

export type EmergencyErrorCode =
  | 'CONNECTION_ERROR'
  | 'ZOME_CALL_ERROR'
  | 'NOT_FOUND'
  | 'UNAUTHORIZED'
  | 'INVALID_INPUT'
  | 'DISASTER_NOT_ACTIVE'
  | 'SHELTER_FULL'
  | 'TEAM_ERROR'
  | 'COORDINATION_ERROR';

export class EmergencyError extends Error {
  constructor(
    public readonly code: EmergencyErrorCode,
    message: string,
    public readonly details?: unknown
  ) {
    super(message);
    this.name = 'EmergencyError';
  }
}
