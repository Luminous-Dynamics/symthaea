/**
 * Care hApp Client Types
 *
 * Type definitions for the Care hApp Master SDK client,
 * providing coverage of timebank, circles, and matching zomes.
 *
 * @module @mycelix/sdk/clients/care/types
 */

import type { ActionHash, AgentPubKey } from '../../generated/common';

// ============================================================================
// Common Types
// ============================================================================

/** Timestamp in microseconds (Holochain format) */
export type Timestamp = number;

/** Base filter for list queries */
export interface PaginationFilter {
  /** Maximum results to return */
  limit?: number;
  /** Offset for pagination */
  offset?: number;
}

// ============================================================================
// Timebank Types
// ============================================================================

/** A service offer from a community member */
export interface ServiceOffer {
  id: ActionHash;
  providerDid: string;
  providerPubKey: AgentPubKey;
  title: string;
  description: string;
  category: string;
  /** Estimated time in minutes per session */
  estimatedMinutes: number;
  /** Skills or tags */
  tags: string[];
  /** Geographic availability (optional) */
  location?: string;
  /** Whether the offer is currently available */
  available: boolean;
  /** Average rating (0.0-5.0) */
  averageRating: number;
  /** Number of completed exchanges */
  completedCount: number;
  createdAt: Timestamp;
  updatedAt: Timestamp;
}

/** Input for creating a service offer */
export interface CreateServiceOfferInput {
  title: string;
  description: string;
  category: string;
  estimatedMinutes: number;
  tags?: string[];
  location?: string;
}

/** A request for a service from a community member */
export interface ServiceRequest {
  id: ActionHash;
  requesterDid: string;
  requesterPubKey: AgentPubKey;
  title: string;
  description: string;
  category: string;
  /** Urgency level */
  urgency: 'Low' | 'Medium' | 'High' | 'Critical';
  /** Preferred time range */
  preferredTimeStart?: Timestamp;
  preferredTimeEnd?: Timestamp;
  /** Geographic constraint */
  location?: string;
  /** Status of the request */
  status: 'Open' | 'Matched' | 'InProgress' | 'Completed' | 'Cancelled';
  createdAt: Timestamp;
  updatedAt: Timestamp;
}

/** Input for creating a service request */
export interface CreateServiceRequestInput {
  title: string;
  description: string;
  category: string;
  urgency?: 'Low' | 'Medium' | 'High' | 'Critical';
  preferredTimeStart?: Timestamp;
  preferredTimeEnd?: Timestamp;
  location?: string;
}

/** A completed time exchange between two members */
export interface TimeExchange {
  id: ActionHash;
  offerId: ActionHash;
  requestId?: ActionHash;
  providerDid: string;
  receiverDid: string;
  /** Actual time spent in minutes */
  minutes: number;
  description: string;
  /** Provider's rating of receiver (0.0-5.0) */
  providerRating?: number;
  /** Receiver's rating of provider (0.0-5.0) */
  receiverRating?: number;
  status: 'Pending' | 'Confirmed' | 'Disputed' | 'Completed';
  completedAt?: Timestamp;
  createdAt: Timestamp;
}

/** Input for completing an exchange */
export interface CompleteExchangeInput {
  offerId: ActionHash;
  requestId?: ActionHash;
  receiverDid: string;
  minutes: number;
  description: string;
}

/** Input for rating an exchange */
export interface RateExchangeInput {
  exchangeId: ActionHash;
  rating: number;
  comment?: string;
}

/** Time balance for a member */
export interface TimeBalance {
  did: string;
  /** Hours earned providing services */
  earned: number;
  /** Hours spent receiving services */
  spent: number;
  /** Net balance (earned - spent) */
  balance: number;
  /** Total exchanges participated in */
  totalExchanges: number;
  /** Average rating as provider */
  averageProviderRating: number;
  /** Average rating as receiver */
  averageReceiverRating: number;
}

// ============================================================================
// Circle Types
// ============================================================================

/** A care circle - a group of members providing mutual support */
export interface CareCircle {
  id: ActionHash;
  name: string;
  description: string;
  founderDid: string;
  /** Maximum number of members */
  maxMembers: number;
  /** Current member count */
  memberCount: number;
  /** Geographic area (optional) */
  area?: string;
  /** Categories of care this circle focuses on */
  categories: string[];
  /** Whether the circle is open to new members */
  open: boolean;
  active: boolean;
  createdAt: Timestamp;
  updatedAt: Timestamp;
}

/** Input for creating a care circle */
export interface CreateCareCircleInput {
  name: string;
  description: string;
  maxMembers?: number;
  area?: string;
  categories?: string[];
  open?: boolean;
}

/** A circle member record */
export interface CircleMember {
  id: ActionHash;
  circleId: ActionHash;
  memberDid: string;
  memberPubKey: AgentPubKey;
  role: 'Member' | 'Coordinator' | 'Founder';
  joinedAt: Timestamp;
  active: boolean;
}

// ============================================================================
// Matching Types
// ============================================================================

/** A match between a service request and an offer */
export interface CareMatch {
  id: ActionHash;
  requestId: ActionHash;
  offerId: ActionHash;
  requesterDid: string;
  providerDid: string;
  /** Match confidence score (0.0-1.0) */
  confidence: number;
  /** Reason for match suggestion */
  reason: string;
  status: 'Suggested' | 'Accepted' | 'Declined' | 'Expired';
  createdAt: Timestamp;
}

/** A structured care plan for ongoing support */
export interface CarePlan {
  id: ActionHash;
  recipientDid: string;
  circleId?: ActionHash;
  title: string;
  description: string;
  /** Scheduled sessions */
  sessions: CarePlanSession[];
  status: 'Draft' | 'Active' | 'Completed' | 'Paused';
  createdAt: Timestamp;
  updatedAt: Timestamp;
}

/** A session within a care plan */
export interface CarePlanSession {
  providerDid: string;
  category: string;
  scheduledAt: Timestamp;
  durationMinutes: number;
  completed: boolean;
}

/** A credential attesting to care service quality */
export interface CareCredential {
  id: ActionHash;
  holderDid: string;
  issuerDid: string;
  credentialType: 'ServiceProvider' | 'CircleCoordinator' | 'CommunityPillar';
  category: string;
  hoursCompleted: number;
  averageRating: number;
  issuedAt: Timestamp;
  expiresAt?: Timestamp;
}

// ============================================================================
// Error Types
// ============================================================================

export type CareErrorCode =
  | 'CONNECTION_ERROR'
  | 'ZOME_CALL_ERROR'
  | 'NOT_FOUND'
  | 'UNAUTHORIZED'
  | 'INVALID_INPUT'
  | 'INSUFFICIENT_BALANCE'
  | 'CIRCLE_FULL'
  | 'ALREADY_MEMBER'
  | 'NOT_MEMBER'
  | 'EXCHANGE_ERROR'
  | 'MATCH_ERROR';

export class CareError extends Error {
  constructor(
    public readonly code: CareErrorCode,
    message: string,
    public readonly details?: unknown
  ) {
    super(message);
    this.name = 'CareError';
  }
}
