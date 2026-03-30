// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix ZK-Tax TypeScript Type Definitions
 *
 * Zero-knowledge tax bracket proofs for privacy-preserving compliance.
 */

// =============================================================================
// Core Types
// =============================================================================

/**
 * Supported regions for jurisdictions.
 */
export type Region = 'Americas' | 'Europe' | 'Asia-Pacific' | 'Middle East' | 'Africa';

/**
 * Filing status codes.
 */
export type FilingStatusCode = 'single' | 'mfj' | 'mfs' | 'hoh';

/**
 * All supported jurisdiction codes (58 total).
 */
export type JurisdictionCode =
  // Americas (10)
  | 'US' | 'CA' | 'MX' | 'BR' | 'AR' | 'CL' | 'CO' | 'PE' | 'EC' | 'UY'
  // Europe (23)
  | 'UK' | 'DE' | 'FR' | 'IT' | 'ES' | 'NL' | 'BE' | 'AT' | 'PT' | 'IE'
  | 'PL' | 'SE' | 'DK' | 'FI' | 'NO' | 'CH' | 'CZ' | 'GR' | 'HU' | 'RO'
  | 'RU' | 'TR' | 'UA'
  // Asia-Pacific (15)
  | 'JP' | 'CN' | 'IN' | 'KR' | 'ID' | 'AU' | 'NZ' | 'SG' | 'HK' | 'TW'
  | 'MY' | 'TH' | 'VN' | 'PH' | 'PK'
  // Middle East (5)
  | 'SA' | 'AE' | 'IL' | 'EG' | 'QA'
  // Africa (5)
  | 'ZA' | 'NG' | 'KE' | 'MA' | 'GH';

/**
 * Supported tax years.
 */
export type TaxYear = 2020 | 2021 | 2022 | 2023 | 2024 | 2025;

// =============================================================================
// Proof Types
// =============================================================================

/**
 * A zero-knowledge tax bracket proof.
 * Proves income falls within a specific tax bracket without revealing exact amount.
 */
export interface TaxBracketProof {
  /** Jurisdiction code (e.g., "US", "UK") */
  jurisdiction: JurisdictionCode;
  /** Filing status code */
  filing_status: FilingStatusCode;
  /** Tax year */
  tax_year: TaxYear;
  /** Index of the tax bracket (0-based) */
  bracket_index: number;
  /** Tax rate in basis points (e.g., 2200 = 22%) */
  rate_bps: number;
  /** Lower bound of the bracket in currency units */
  bracket_lower: number;
  /** Upper bound of the bracket (u64::MAX for unbounded) */
  bracket_upper: number;
  /** Cryptographic commitment to the bracket */
  commitment: number[];
  /** RISC Zero receipt bytes (or dev mode marker) */
  receipt_bytes: number[];
  /** RISC Zero image ID */
  image_id: number[];
}

/**
 * Human-readable proof information.
 */
export interface ProofInfo {
  jurisdiction: string;
  jurisdiction_name: string;
  filing_status: string;
  tax_year: number;
  bracket_index: number;
  rate_percent: number;
  bracket_lower: number;
  bracket_upper: number;
  commitment: string;
  is_dev_mode: boolean;
}

/**
 * Range proof - proves income is within bounds without revealing exact value.
 */
export interface RangeProof {
  income_commitment: number[];
  min_bound: number | null;
  max_bound: number | null;
  tax_year: number;
  receipt_bytes: number[];
  image_id: number[];
}

/**
 * Batch proof for multiple years.
 */
export interface BatchProof {
  jurisdiction: JurisdictionCode;
  filing_status: FilingStatusCode;
  proofs: TaxBracketProof[];
  aggregation_commitment: number[];
}

/**
 * Compressed proof for efficient storage/transmission.
 */
export interface CompressedProof {
  compression_type: 'rle_base64' | 'zstd' | 'raw';
  original_size: number;
  compressed_data: string;
}

// =============================================================================
// Bracket Types
// =============================================================================

/**
 * Tax bracket information.
 */
export interface BracketInfo {
  index: number;
  lower: number;
  upper: number;
  rate_percent: number;
}

/**
 * Jurisdiction metadata.
 */
export interface JurisdictionInfo {
  code: JurisdictionCode;
  name: string;
  currency: string;
  region: Region;
}

// =============================================================================
// Configuration Types
// =============================================================================

/**
 * Options for proof generation.
 */
export interface ProveOptions {
  /** Use dev mode (fast, not cryptographically secure) */
  devMode?: boolean;
  /** Enable proof caching */
  cache?: boolean;
  /** Compression type for proof */
  compress?: 'rle_base64' | 'zstd' | 'raw' | false;
}

/**
 * Options for batch proof generation.
 */
export interface BatchProveOptions extends ProveOptions {
  /** Years and incomes to prove */
  years: Array<{ year: TaxYear; income: number }>;
}

/**
 * Options for range proof generation.
 */
export interface RangeProveOptions extends ProveOptions {
  /** Minimum income bound (optional) */
  minIncome?: number;
  /** Maximum income bound (optional) */
  maxIncome?: number;
}

// =============================================================================
// Result Types
// =============================================================================

/**
 * Result of proof verification.
 */
export interface VerificationResult {
  valid: boolean;
  info?: ProofInfo;
  error?: string;
}

/**
 * SDK initialization status.
 */
export interface InitStatus {
  initialized: boolean;
  version: string;
  wasmLoaded: boolean;
  jurisdictions: number;
}

// =============================================================================
// Utility Types
// =============================================================================

/**
 * Callback for async operations.
 */
export type ProgressCallback = (progress: number, message: string) => void;

/**
 * Error thrown by ZK-Tax SDK.
 */
export class ZkTaxError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly cause?: unknown
  ) {
    super(message);
    this.name = 'ZkTaxError';
  }
}
