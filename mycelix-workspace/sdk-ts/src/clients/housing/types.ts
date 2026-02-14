/**
 * Housing hApp Client Types
 *
 * Type definitions for the Housing hApp Master SDK client,
 * providing coverage of units, membership, finances, and governance zomes.
 *
 * @module @mycelix/sdk/clients/housing/types
 */

import type { ActionHash, AgentPubKey } from '../../generated/common';

// ============================================================================
// Common Types
// ============================================================================

export type Timestamp = number;

export interface PaginationFilter {
  limit?: number;
  offset?: number;
}

// ============================================================================
// Building & Unit Types
// ============================================================================

/** A registered building or housing complex */
export interface Building {
  id: ActionHash;
  name: string;
  address: string;
  latitude?: number;
  longitude?: number;
  ownerDid: string;
  buildingType: 'CoOp' | 'LandTrust' | 'MutualHousing' | 'CommunityLand' | 'PublicHousing';
  totalUnits: number;
  occupiedUnits: number;
  yearBuilt?: number;
  amenities: string[];
  active: boolean;
  createdAt: Timestamp;
  updatedAt: Timestamp;
}

/** Input for registering a building */
export interface RegisterBuildingInput {
  name: string;
  address: string;
  latitude?: number;
  longitude?: number;
  buildingType: Building['buildingType'];
  totalUnits: number;
  yearBuilt?: number;
  amenities?: string[];
}

/** A housing unit within a building */
export interface Unit {
  id: ActionHash;
  buildingId: ActionHash;
  unitNumber: string;
  floorPlan: string;
  squareMeters: number;
  bedrooms: number;
  bathrooms: number;
  monthlyCharge: number;
  occupantDid?: string;
  status: 'Available' | 'Occupied' | 'Reserved' | 'Maintenance' | 'Decommissioned';
  createdAt: Timestamp;
  updatedAt: Timestamp;
}

/** Input for registering a unit */
export interface RegisterUnitInput {
  buildingId: ActionHash;
  unitNumber: string;
  floorPlan: string;
  squareMeters: number;
  bedrooms: number;
  bathrooms: number;
  monthlyCharge: number;
}

/** Input for assigning an occupant */
export interface AssignOccupantInput {
  unitId: ActionHash;
  occupantDid: string;
  moveInDate: Timestamp;
}

// ============================================================================
// Membership Types
// ============================================================================

/** A housing cooperative member */
export interface Member {
  id: ActionHash;
  buildingId: ActionHash;
  memberDid: string;
  memberPubKey: AgentPubKey;
  name: string;
  unitId?: ActionHash;
  membershipType: 'Full' | 'Associate' | 'Provisional' | 'RentToOwn';
  /** Equity share in the cooperative (0.0-1.0) */
  equityShare: number;
  /** Total equity accumulated */
  equityAccumulated: number;
  votingWeight: number;
  status: 'Active' | 'Suspended' | 'Former';
  joinedAt: Timestamp;
  updatedAt: Timestamp;
}

/** Input for submitting a membership application */
export interface SubmitApplicationInput {
  buildingId: ActionHash;
  name: string;
  membershipType?: Member['membershipType'];
  statement?: string;
}

/** Input for approving a member */
export interface ApproveMemberInput {
  applicationId: ActionHash;
  unitId?: ActionHash;
  equityShare?: number;
  votingWeight?: number;
}

/** Rent-to-own agreement */
export interface RentToOwnAgreement {
  id: ActionHash;
  memberId: ActionHash;
  unitId: ActionHash;
  totalEquityTarget: number;
  monthlyEquityContribution: number;
  equityAccumulated: number;
  termMonths: number;
  startDate: Timestamp;
  projectedCompletionDate: Timestamp;
  status: 'Active' | 'Completed' | 'Defaulted' | 'Cancelled';
  createdAt: Timestamp;
}

/** Input for creating a rent-to-own agreement */
export interface CreateRentToOwnInput {
  memberId: ActionHash;
  unitId: ActionHash;
  totalEquityTarget: number;
  monthlyEquityContribution: number;
  termMonths: number;
}

/** Input for recording a rent payment */
export interface RecordRentPaymentInput {
  memberId: ActionHash;
  amount: number;
  period: string;
  notes?: string;
}

// ============================================================================
// Finance Types
// ============================================================================

/** A monthly charge record */
export interface MonthlyCharge {
  id: ActionHash;
  buildingId: ActionHash;
  unitId: ActionHash;
  memberDid: string;
  period: string;
  baseRent: number;
  maintenanceFee: number;
  utilities: number;
  specialAssessment: number;
  totalDue: number;
  totalPaid: number;
  status: 'Pending' | 'Partial' | 'Paid' | 'Overdue' | 'Waived';
  dueDate: Timestamp;
  createdAt: Timestamp;
}

/** Input for generating monthly charges */
export interface GenerateMonthlyChargesInput {
  buildingId: ActionHash;
  period: string;
  specialAssessment?: number;
  specialAssessmentReason?: string;
}

/** Payment record */
export interface PaymentRecord {
  id: ActionHash;
  chargeId: ActionHash;
  memberDid: string;
  amount: number;
  method: 'Bank' | 'Check' | 'Cash' | 'Crypto' | 'TimeBank';
  reference?: string;
  paidAt: Timestamp;
}

/** Input for recording a payment */
export interface RecordPaymentInput {
  chargeId: ActionHash;
  amount: number;
  method: PaymentRecord['method'];
  reference?: string;
}

/** A maintenance request */
export interface MaintenanceRequest {
  id: ActionHash;
  buildingId: ActionHash;
  unitId?: ActionHash;
  requesterDid: string;
  title: string;
  description: string;
  priority: 'Low' | 'Medium' | 'High' | 'Emergency';
  estimatedCost?: number;
  actualCost?: number;
  status: 'Submitted' | 'Approved' | 'InProgress' | 'Completed' | 'Rejected';
  createdAt: Timestamp;
  completedAt?: Timestamp;
}

/** Financial summary for a building */
export interface FinancialSummary {
  buildingId: ActionHash;
  period: string;
  totalRevenue: number;
  totalExpenses: number;
  netIncome: number;
  outstandingCharges: number;
  reserveFund: number;
  occupancyRate: number;
}

// ============================================================================
// Housing Governance Types
// ============================================================================

/** A scheduled meeting */
export interface Meeting {
  id: ActionHash;
  buildingId: ActionHash;
  title: string;
  description: string;
  meetingType: 'Regular' | 'Special' | 'Emergency' | 'Annual';
  scheduledAt: Timestamp;
  location: string;
  agenda: string[];
  minutesTakerDid?: string;
  quorumRequired: number;
  status: 'Scheduled' | 'InProgress' | 'Completed' | 'Cancelled';
  createdAt: Timestamp;
}

/** Input for scheduling a meeting */
export interface ScheduleMeetingInput {
  buildingId: ActionHash;
  title: string;
  description: string;
  meetingType: Meeting['meetingType'];
  scheduledAt: Timestamp;
  location: string;
  agenda?: string[];
  quorumRequired?: number;
}

/** A resolution proposed at a meeting */
export interface Resolution {
  id: ActionHash;
  buildingId: ActionHash;
  meetingId?: ActionHash;
  title: string;
  description: string;
  proposerDid: string;
  approveVotes: number;
  rejectVotes: number;
  abstainVotes: number;
  quorumRequired: number;
  threshold: number;
  status: 'Proposed' | 'Voting' | 'Passed' | 'Rejected' | 'Tabled';
  votingEndsAt: Timestamp;
  createdAt: Timestamp;
}

/** Input for proposing a resolution */
export interface ProposeResolutionInput {
  buildingId: ActionHash;
  meetingId?: ActionHash;
  title: string;
  description: string;
  quorumRequired?: number;
  threshold?: number;
  votingPeriodHours?: number;
}

/** Input for voting on a resolution */
export interface VoteOnResolutionInput {
  resolutionId: ActionHash;
  choice: 'Approve' | 'Reject' | 'Abstain';
  reason?: string;
}

/** A board election */
export interface Election {
  id: ActionHash;
  buildingId: ActionHash;
  title: string;
  positions: string[];
  candidates: ElectionCandidate[];
  status: 'Nominations' | 'Voting' | 'Completed' | 'Cancelled';
  votingStartsAt: Timestamp;
  votingEndsAt: Timestamp;
  createdAt: Timestamp;
}

/** A candidate in an election */
export interface ElectionCandidate {
  did: string;
  name: string;
  position: string;
  statement: string;
  votes: number;
}

/** Input for creating an election */
export interface CreateElectionInput {
  buildingId: ActionHash;
  title: string;
  positions: string[];
  votingStartsAt: Timestamp;
  votingEndsAt: Timestamp;
}

/** Input for casting a ballot */
export interface CastBallotInput {
  electionId: ActionHash;
  /** Map of position to candidate DID */
  selections: Record<string, string>;
}

/** A land trust */
export interface LandTrust {
  id: ActionHash;
  name: string;
  description: string;
  trusteeDids: string[];
  /** GeoJSON boundary */
  boundaryGeoJson?: string;
  buildingIds: ActionHash[];
  active: boolean;
  createdAt: Timestamp;
  updatedAt: Timestamp;
}

/** A ground lease */
export interface GroundLease {
  id: ActionHash;
  landTrustId: ActionHash;
  buildingId: ActionHash;
  leaseholderDid: string;
  termYears: number;
  annualFee: number;
  resaleRestrictions: string;
  startDate: Timestamp;
  expiresAt: Timestamp;
  status: 'Active' | 'Expired' | 'Terminated';
  createdAt: Timestamp;
}

// ============================================================================
// Error Types
// ============================================================================

export type HousingErrorCode =
  | 'CONNECTION_ERROR'
  | 'ZOME_CALL_ERROR'
  | 'NOT_FOUND'
  | 'UNAUTHORIZED'
  | 'INVALID_INPUT'
  | 'UNIT_UNAVAILABLE'
  | 'NOT_MEMBER'
  | 'ALREADY_MEMBER'
  | 'PAYMENT_ERROR'
  | 'GOVERNANCE_ERROR'
  | 'ELECTION_ERROR';

export class HousingError extends Error {
  constructor(
    public readonly code: HousingErrorCode,
    message: string,
    public readonly details?: unknown
  ) {
    super(message);
    this.name = 'HousingError';
  }
}
