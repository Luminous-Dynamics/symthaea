// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Epistemic Types
 *
 * Core type definitions for the epistemic charter.
 * This file has no imports to avoid circular dependencies.
 *
 * @packageDocumentation
 * @module epistemic/types
 */

/**
 * Empirical verification levels (E-axis)
 */
export enum EmpiricalLevel {
  E0_Unverified = 0, // No verification
  E1_Testimonial = 1, // First-person testimony
  E2_PrivateVerify = 2, // Privately verifiable (shared secret)
  E3_Cryptographic = 3, // Cryptographic proof
  E4_Consensus = 4, // Network consensus
}

/**
 * Normative scope levels (N-axis)
 */
export enum NormativeLevel {
  N0_Personal = 0, // Individual preference
  N1_Communal = 1, // Small group consensus
  N2_Network = 2, // Network-wide agreement
  N3_Universal = 3, // Cross-network standard
}

/**
 * Materiality levels (M-axis)
 */
export enum MaterialityLevel {
  M0_Ephemeral = 0, // Temporary, in-memory
  M1_Temporal = 1, // Time-bounded persistence
  M2_Persistent = 2, // Long-term storage
  M3_Immutable = 3, // Permanent record
}

/**
 * 3D epistemic classification
 */
export interface EpistemicClassification {
  empirical: EmpiricalLevel;
  normative: NormativeLevel;
  materiality: MaterialityLevel;
}

/**
 * Evidence supporting a claim
 */
export interface Evidence {
  type: string; // Evidence type (signature, hash, testimony, etc.)
  data: string; // Evidence data
  source: string; // Source identifier
  timestamp: number; // When evidence was collected
}

/**
 * Epistemic claim with classification and evidence
 */
export interface EpistemicClaim {
  id: string;
  content: string;
  classification: EpistemicClassification;
  evidence: Evidence[];
  issuer: string;
  issuedAt: number;
  expiresAt?: number;
}
