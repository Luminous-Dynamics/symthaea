/**
 * Hearth hApp Client Types
 *
 * Type definitions for the Hearth hApp Master SDK client,
 * covering kinship, gratitude, stories, care, autonomy, emergency,
 * decisions, resources, milestones, rhythms, and bridge zomes.
 *
 * @module @mycelix/sdk/clients/hearth/types
 */

import type { ActionHash, AgentPubKey } from '../../generated/common';

// ============================================================================
// Common Types
// ============================================================================

/** Timestamp in microseconds (Holochain format) */
export type Timestamp = number;

/** Bond strength in basis points (0-10000) */
export type BondStrength = number;

// ============================================================================
// Hearth Core Enums (match Rust hearth-types exactly)
// ============================================================================

export type HearthType = 'Nuclear' | 'Extended' | 'Chosen' | 'Blended' | 'Multigenerational' | 'Intentional' | 'CoPod' | { Custom: string };
export type MemberRole = 'Founder' | 'Elder' | 'Adult' | 'Youth' | 'Child' | 'Guest' | 'Ancestor';
export type MembershipStatus = 'Active' | 'Invited' | 'Departed' | 'Ancestral';
export type BondType = 'Parent' | 'Child' | 'Sibling' | 'Partner' | 'Grandparent' | 'Grandchild' | 'AuntUncle' | 'NieceNephew' | 'Cousin' | 'ChosenFamily' | 'Guardian' | 'Ward' | { Custom: string };
export type AutonomyTier = 'Dependent' | 'Supervised' | 'Guided' | 'SemiAutonomous' | 'Autonomous';
export type HearthVisibility = 'AllMembers' | 'AdultsOnly' | 'GuardiansOnly' | { Specified: AgentPubKey[] };
export type CareType = 'Childcare' | 'Eldercare' | 'PetCare' | 'Chore' | 'MealPrep' | 'Medical' | 'Emotional' | { Custom: string };
export type Recurrence = 'Daily' | 'Weekly' | 'Monthly' | { Custom: string };
export type AlertSeverity = 'Low' | 'Medium' | 'High' | 'Critical';
export type AlertType = 'Medical' | 'Natural' | 'Security' | 'Missing' | 'Fire' | { Custom: string };
export type GratitudeType = 'Appreciation' | 'Acknowledgment' | 'Celebration' | 'Blessing' | { Custom: string };
export type StoryType = 'Memory' | 'Tradition' | 'Recipe' | 'Wisdom' | 'Origin' | 'Migration' | { Custom: string };
export type MilestoneType = 'Birth' | 'Birthday' | 'FirstStep' | 'SchoolStart' | 'Graduation' | 'Engagement' | 'Marriage' | 'NewHome' | 'Retirement' | 'Passing' | { Custom: string };
export type TransitionType = 'JoiningHearth' | 'LeavingHearth' | 'ComingOfAge' | 'Retirement' | 'Bereavement' | { Custom: string };
export type TransitionPhase = 'PreLiminal' | 'Liminal' | 'PostLiminal' | 'Integrated';
export type DecisionType = 'Consensus' | 'MajorityVote' | 'ElderDecision' | 'GuardianDecision';
export type DecisionStatus = 'Open' | 'Closed' | 'Finalized';
export type ResourceType = 'Tool' | 'Vehicle' | 'Book' | 'Kitchen' | 'Electronics' | 'Clothing' | { Custom: string };
export type LoanStatus = 'Active' | 'Returned' | 'Overdue';
export type RhythmType = 'Morning' | 'Evening' | 'Weekly' | 'Seasonal' | { Custom: string };
export type PresenceStatusType = 'Home' | 'Away' | 'Working' | 'Sleeping' | 'DoNotDisturb';
export type InvitationStatus = 'Pending' | 'Accepted' | 'Declined' | 'Expired';
export type SafetyStatus = 'Safe' | 'NeedHelp' | 'NoResponse';
export type SwapStatus = 'Proposed' | 'Accepted' | 'Declined' | 'Completed';
export type CareScheduleStatus = 'Active' | 'Paused' | 'Completed';
export type CircleStatus = 'Open' | 'InProgress' | 'Completed';
export type AutonomyRequestStatus = 'Pending' | 'Approved' | 'Denied';

// ============================================================================
// Kinship Types
// ============================================================================

export interface Hearth {
  name: string;
  description: string;
  hearth_type: HearthType;
  created_by: AgentPubKey;
  created_at: Timestamp;
  max_members: number;
}

export interface CreateHearthInput {
  name: string;
  description: string;
  hearth_type: HearthType;
  max_members?: number;
}

export interface HearthMembership {
  hearth_hash: ActionHash;
  agent: AgentPubKey;
  role: MemberRole;
  status: MembershipStatus;
  display_name: string;
  joined_at: Timestamp;
}

export interface KinshipBond {
  hearth_hash: ActionHash;
  member_a: AgentPubKey;
  member_b: AgentPubKey;
  bond_type: BondType;
  strength: BondStrength;
  last_tended: Timestamp;
}

export interface HearthInvitation {
  hearth_hash: ActionHash;
  inviter: AgentPubKey;
  invitee_key: AgentPubKey;
  proposed_role: MemberRole;
  message: string;
  expires_at: Timestamp;
  status: InvitationStatus;
}

export interface InviteMemberInput {
  hearth_hash: ActionHash;
  invitee_key: AgentPubKey;
  proposed_role: MemberRole;
  message: string;
  expires_at: Timestamp;
}

export interface CreateBondInput {
  hearth_hash: ActionHash;
  member_a: AgentPubKey;
  member_b: AgentPubKey;
  bond_type: BondType;
  initial_strength?: BondStrength;
}

export interface TendBondInput {
  bond_hash: ActionHash;
  description: string;
  quality: BondStrength;
}

export interface BondHealthResult {
  bond: KinshipBond;
  current_strength: BondStrength;
  days_since_tended: number;
  neglected: boolean;
}

export interface UpdateMemberRoleInput {
  membership_hash: ActionHash;
  new_role: MemberRole;
}

// ============================================================================
// Gratitude Types
// ============================================================================

export interface GratitudeExpression {
  from_agent: AgentPubKey;
  to_agent: AgentPubKey;
  message: string;
  gratitude_type: GratitudeType;
  visibility: HearthVisibility;
  hearth_hash: ActionHash;
  created_at: Timestamp;
}

export interface ExpressGratitudeInput {
  to_agent: AgentPubKey;
  message: string;
  gratitude_type: GratitudeType;
  visibility: HearthVisibility;
  hearth_hash: ActionHash;
}

export interface AppreciationCircle {
  hearth_hash: ActionHash;
  theme: string;
  participants: AgentPubKey[];
  started_at: Timestamp;
  completed_at?: Timestamp;
  status: CircleStatus;
}

export interface StartCircleInput {
  hearth_hash: ActionHash;
  theme: string;
}

export interface GratitudeAnchor {
  agent: AgentPubKey;
  hearth_hash: ActionHash;
  total_given: number;
  total_received: number;
  current_streak_days: number;
}

// ============================================================================
// Stories Types
// ============================================================================

export interface FamilyStory {
  title: string;
  content: string;
  storyteller: AgentPubKey;
  story_type: StoryType;
  media_hashes: ActionHash[];
  tags: string[];
  visibility: HearthVisibility;
  hearth_hash: ActionHash;
  created_at: Timestamp;
}

export interface CreateStoryInput {
  title: string;
  content: string;
  story_type: StoryType;
  media_hashes?: ActionHash[];
  tags?: string[];
  visibility: HearthVisibility;
  hearth_hash: ActionHash;
}

export interface StoryCollection {
  name: string;
  description: string;
  story_hashes: ActionHash[];
  curator: AgentPubKey;
  hearth_hash: ActionHash;
}

export interface CreateCollectionInput {
  name: string;
  description: string;
  hearth_hash: ActionHash;
}

export interface FamilyTradition {
  name: string;
  description: string;
  frequency: Recurrence;
  season?: string;
  instructions: string;
  last_observed?: Timestamp;
  next_due?: Timestamp;
  hearth_hash: ActionHash;
}

export interface CreateTraditionInput {
  name: string;
  description: string;
  frequency: Recurrence;
  season?: string;
  instructions: string;
  hearth_hash: ActionHash;
}

// ============================================================================
// Care Types
// ============================================================================

export interface CareSchedule {
  hearth_hash: ActionHash;
  care_type: CareType;
  title: string;
  description: string;
  assigned_to: AgentPubKey;
  recurrence: Recurrence;
  notes: string;
  status: CareScheduleStatus;
  created_at: Timestamp;
}

export interface CreateCareScheduleInput {
  hearth_hash: ActionHash;
  care_type: CareType;
  title: string;
  description: string;
  assigned_to: AgentPubKey;
  recurrence: Recurrence;
  notes?: string;
}

export interface CareSwap {
  requester: AgentPubKey;
  responder?: AgentPubKey;
  original_schedule_hash: ActionHash;
  swap_date: Timestamp;
  status: SwapStatus;
}

export interface ProposeSwapInput {
  original_schedule_hash: ActionHash;
  swap_date: Timestamp;
}

export interface MealPlan {
  hearth_hash: ActionHash;
  week_start: Timestamp;
  meals: PlannedMeal[];
  shopper: AgentPubKey;
  cook: AgentPubKey;
  dietary_notes: string;
}

export interface PlannedMeal {
  day: string;
  meal_type: string;
  recipe_name: string;
  servings: number;
}

export interface CreateMealPlanInput {
  hearth_hash: ActionHash;
  week_start: Timestamp;
  meals: PlannedMeal[];
  shopper: AgentPubKey;
  cook: AgentPubKey;
  dietary_notes?: string;
}

// ============================================================================
// Autonomy Types
// ============================================================================

export interface AutonomyProfile {
  member_hash: AgentPubKey;
  guardian_hashes: AgentPubKey[];
  current_tier: AutonomyTier;
  capabilities: string[];
  restrictions: string[];
  review_schedule?: string;
  hearth_hash: ActionHash;
}

export interface CreateAutonomyProfileInput {
  member_hash: AgentPubKey;
  guardian_hashes: AgentPubKey[];
  initial_tier: AutonomyTier;
  hearth_hash: ActionHash;
}

export interface AutonomyRequest {
  requester: AgentPubKey;
  capability: string;
  justification: string;
  guardian_approvals: ActionHash[];
  status: AutonomyRequestStatus;
  hearth_hash: ActionHash;
}

export interface RequestCapabilityInput {
  capability: string;
  justification: string;
  hearth_hash: ActionHash;
}

export interface GuardianApproval {
  request_hash: ActionHash;
  guardian: AgentPubKey;
  approved: boolean;
  conditions?: string;
}

export interface ApproveCapabilityInput {
  request_hash: ActionHash;
  approved: boolean;
  conditions?: string;
}

export interface CheckCapabilityInput {
  member: AgentPubKey;
  capability: string;
}

export interface AdvanceTierInput {
  profile_hash: ActionHash;
  new_tier: AutonomyTier;
}

// ============================================================================
// Emergency Types
// ============================================================================

export interface EmergencyPlan {
  hearth_hash: ActionHash;
  contacts: EmergencyContact[];
  meeting_points: string[];
  medical_info_hashes: ActionHash[];
  last_reviewed: Timestamp;
}

export interface CreateEmergencyPlanInput {
  hearth_hash: ActionHash;
  contacts: EmergencyContact[];
  meeting_points: string[];
  medical_info_hashes?: ActionHash[];
}

export interface EmergencyContact {
  name: string;
  phone: string;
  relationship: string;
  priority_order: number;
}

export interface EmergencyAlert {
  hearth_hash: ActionHash;
  alert_type: AlertType;
  severity: AlertSeverity;
  message: string;
  reporter: AgentPubKey;
  resolved_at?: Timestamp;
  created_at: Timestamp;
}

export interface RaiseAlertInput {
  hearth_hash: ActionHash;
  alert_type: AlertType;
  severity: AlertSeverity;
  message: string;
}

export interface SafetyCheckIn {
  hearth_hash: ActionHash;
  member: AgentPubKey;
  status: SafetyStatus;
  location_hint?: string;
  checked_in_at: Timestamp;
}

export interface CheckInInput {
  hearth_hash: ActionHash;
  status: SafetyStatus;
  location_hint?: string;
}

// ============================================================================
// Decisions Types
// ============================================================================

export interface Decision {
  hearth_hash: ActionHash;
  title: string;
  description: string;
  decision_type: DecisionType;
  eligible_roles: MemberRole[];
  options: string[];
  deadline: Timestamp;
  status: DecisionStatus;
  created_at: Timestamp;
}

export interface CreateDecisionInput {
  hearth_hash: ActionHash;
  title: string;
  description: string;
  decision_type: DecisionType;
  eligible_roles: MemberRole[];
  options: string[];
  deadline: Timestamp;
}

export interface Vote {
  decision_hash: ActionHash;
  voter: AgentPubKey;
  choice: number;
  weight: BondStrength;
  reasoning?: string;
}

export interface CastVoteInput {
  decision_hash: ActionHash;
  choice: number;
  reasoning?: string;
}

export interface DecisionOutcome {
  decision_hash: ActionHash;
  chosen_option: number;
  participation_rate: BondStrength;
  resolved_at: Timestamp;
}

// ============================================================================
// Resources Types
// ============================================================================

export interface SharedResource {
  name: string;
  description: string;
  resource_type: ResourceType;
  current_holder?: AgentPubKey;
  condition: string;
  location: string;
  hearth_hash: ActionHash;
}

export interface RegisterResourceInput {
  name: string;
  description: string;
  resource_type: ResourceType;
  condition: string;
  location: string;
  hearth_hash: ActionHash;
}

export interface ResourceLoan {
  resource_hash: ActionHash;
  lender_hearth: ActionHash;
  borrower: AgentPubKey;
  due_date: Timestamp;
  status: LoanStatus;
}

export interface LendResourceInput {
  resource_hash: ActionHash;
  borrower: AgentPubKey;
  due_date: Timestamp;
}

export interface BudgetCategory {
  hearth_hash: ActionHash;
  category: string;
  monthly_target_cents: number;
  current_month_actual_cents: number;
}

export interface CreateBudgetInput {
  hearth_hash: ActionHash;
  category: string;
  monthly_target_cents: number;
}

export interface LogExpenseInput {
  budget_hash: ActionHash;
  amount_cents: number;
  description: string;
}

// ============================================================================
// Milestones Types
// ============================================================================

export interface Milestone {
  hearth_hash: ActionHash;
  member_hash: AgentPubKey;
  milestone_type: MilestoneType;
  date: Timestamp;
  description: string;
  witnesses: AgentPubKey[];
  media_hashes: ActionHash[];
}

export interface RecordMilestoneInput {
  hearth_hash: ActionHash;
  member_hash: AgentPubKey;
  milestone_type: MilestoneType;
  date: Timestamp;
  description: string;
  witnesses?: AgentPubKey[];
  media_hashes?: ActionHash[];
}

export interface LifeTransition {
  member_hash: AgentPubKey;
  transition_type: TransitionType;
  started_at: Timestamp;
  completed_at?: Timestamp;
  current_phase: TransitionPhase;
  supporting_members: AgentPubKey[];
  hearth_hash: ActionHash;
}

export interface BeginTransitionInput {
  hearth_hash: ActionHash;
  member_hash: AgentPubKey;
  transition_type: TransitionType;
  supporting_members?: AgentPubKey[];
}

export interface AdvanceTransitionInput {
  transition_hash: ActionHash;
}

// ============================================================================
// Rhythms Types
// ============================================================================

export interface Rhythm {
  hearth_hash: ActionHash;
  name: string;
  rhythm_type: RhythmType;
  schedule: string;
  participants: AgentPubKey[];
  description: string;
}

export interface CreateRhythmInput {
  hearth_hash: ActionHash;
  name: string;
  rhythm_type: RhythmType;
  schedule: string;
  participants: AgentPubKey[];
  description: string;
}

export interface RhythmOccurrence {
  rhythm_hash: ActionHash;
  date: Timestamp;
  participants_present: AgentPubKey[];
  notes: string;
  mood?: BondStrength;
}

export interface LogOccurrenceInput {
  rhythm_hash: ActionHash;
  participants_present: AgentPubKey[];
  notes: string;
  mood?: BondStrength;
}

export interface PresenceStatus {
  agent: AgentPubKey;
  status: PresenceStatusType;
  expected_return?: Timestamp;
  hearth_hash: ActionHash;
}

export interface SetPresenceInput {
  status: PresenceStatusType;
  expected_return?: Timestamp;
  hearth_hash: ActionHash;
}

// ============================================================================
// Bridge Types
// ============================================================================

export interface HearthQueryInput {
  domain: 'kinship' | 'gratitude' | 'stories' | 'care' | 'autonomy' | 'emergency' | 'decisions' | 'resources' | 'milestones' | 'rhythms';
  query_type: string;
  params: string;
}

export interface HearthEventInput {
  domain: 'kinship' | 'gratitude' | 'stories' | 'care' | 'autonomy' | 'emergency' | 'decisions' | 'resources' | 'milestones' | 'rhythms';
  event_type: string;
  payload: string;
  related_hashes?: string[];
}

// ============================================================================
// Error Types
// ============================================================================

export type HearthErrorCode =
  | 'CONNECTION_ERROR'
  | 'ZOME_CALL_ERROR'
  | 'NOT_FOUND'
  | 'UNAUTHORIZED'
  | 'INVALID_INPUT'
  | 'NOT_MEMBER'
  | 'HEARTH_FULL'
  | 'ALREADY_MEMBER'
  | 'BOND_ERROR'
  | 'AUTONOMY_ERROR'
  | 'EMERGENCY_ACTIVE';

export class HearthError extends Error {
  constructor(
    public readonly code: HearthErrorCode,
    message: string,
    public readonly details?: unknown
  ) {
    super(message);
    this.name = 'HearthError';
  }
}
