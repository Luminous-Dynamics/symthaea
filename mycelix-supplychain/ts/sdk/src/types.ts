// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Type definitions for Mycelix Supply Chain
 */

export type EventType = 'PRODUCED' | 'TRANSFORMED' | 'SHIPPED' | 'RECEIVED' | 'CERTIFIED';

export interface Location {
  lat?: number;
  lon?: number;
  address?: string;
  country?: string;
}

export interface Facility {
  id: string;
  name: string;
  location?: Location;
}

export interface Shipment {
  shipmentId: string;
  carrier?: string;
  trackingNumber?: string;
  origin?: string;
  destination?: string;
}

export interface Certification {
  certType: string;
  certBody: string;
  certId: string;
  validFrom: string;
  validUntil: string;
}

export interface CredentialSubject {
  eventType: EventType;
  productId: string;
  batchId: string;
  prevBatchIds?: string[];
  quantity: number;
  unit: string;
  facility: Facility;
  timestamp: string;
  shipment?: Shipment;
  certification?: Certification;
  metadata?: Record<string, any>;
}

export interface SupplyEventVC {
  '@context': string[];
  type: string[];
  issuer: string;
  issuanceDate: string;
  expirationDate?: string;
  credentialSubject: CredentialSubject;
  proof?: any;
}

export interface Lineage {
  hash: string;
  previousClaims?: string[];
}

export interface Subject {
  batchId: string;
  productId: string;
}

export interface Assertion {
  eventType: EventType;
  quantity?: number;
  unit?: string;
  facilityId?: string;
}

export interface Evidence {
  vcJwt: string;
  additionalDocuments?: Array<{
    type: string;
    uri: string;
    hash?: string;
  }>;
}

export interface DkgClaim {
  id: string;
  type: string;
  issuer: string;
  subject: Subject;
  assertion: Assertion;
  evidence: Evidence;
  lineage: Lineage;
  timestamp: string;
  confidence?: number;
  metadata?: Record<string, any>;
}

export interface EventResponse {
  vc_jwt: string;
  claim_id: string;
  lineage_hash: string;
  previous_claims?: string[];
}

export interface ClaimResponse {
  claim: DkgClaim;
  lineage?: DkgClaim[];
}

export interface VerifyRequest {
  vc_jwt: string;
  expected_product_id?: string;
  check_lineage?: boolean;
}

export interface VerifyResponse {
  valid: boolean;
  signature_valid: boolean;
  lineage_valid?: boolean;
  issuer?: string;
  errors?: string[];
}

export interface HealthResponse {
  status: string;
  version: string;
}

// Batch operation types
export interface BatchRequest {
  events: SupplyEventVC[];
  mode?: 'best-effort' | 'atomic';
}

export interface BatchResult {
  index: number;
  status: 'success' | 'error';
  claim_id?: string;
  vc_jwt?: string;
  lineage_hash?: string;
  error?: string;
}

export interface BatchResponse {
  total: number;
  succeeded: number;
  failed: number;
  results: BatchResult[];
  duration_ms: number;
}

// Lineage query types
export interface LineageBatch {
  batch_id: string;
  claim_count: number;
  depth: number;
}

export interface LineageResponse {
  batch_id: string;
  claims: DkgClaim[];
  upstream?: LineageBatch[];
  downstream?: LineageBatch[];
  total_claims: number;
  depth: number;
}

export interface BatchClaimsResponse {
  batch_id: string;
  claims: DkgClaim[];
  total_claims: number;
}

// Search and filter types
export interface ClaimFilters {
  product_id?: string;
  batch_id?: string;
  facility_id?: string;
  event_type?: EventType;
  from?: string; // ISO 8601
  to?: string;   // ISO 8601
  limit?: number;
  offset?: number;
}

export interface SearchResponse {
  claims: DkgClaim[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}
