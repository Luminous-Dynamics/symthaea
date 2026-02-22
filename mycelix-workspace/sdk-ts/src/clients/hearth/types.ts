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
  strength_bp: number;
  last_tended: Timestamp;
  created_at: Timestamp;
}

export interface HearthInvitation {
  hearth_hash: ActionHash;
  inviter: AgentPubKey;
  invitee_agent: AgentPubKey;
  proposed_role: MemberRole;
  message: string;
  expires_at: Timestamp;
  status: InvitationStatus;
}

export interface InviteMemberInput {
  hearth_hash: ActionHash;
  invitee_agent: AgentPubKey;
  proposed_role: MemberRole;
  message: string;
  expires_at: Timestamp;
}

export interface CreateBondInput {
  hearth_hash: ActionHash;
  member_b: AgentPubKey;
  bond_type: BondType;
  initial_strength_bp?: number;
}

export interface TendBondInput {
  bond_hash: ActionHash;
  description: string;
  quality_bp: number;
}

export interface GetBondHealthInput {
  bond_hash: ActionHash;
}

export interface UpdateMemberRoleInput {
  membership_hash: ActionHash;
  new_role: MemberRole;
}

export interface BondUpdate {
  member_a: AgentPubKey;
  member_b: AgentPubKey;
  co_creation_count: number;
  quality_sum_bp: number;
}

export interface CareSummary {
  assignee: AgentPubKey;
  tasks_completed: number;
  hours_hundredths: number;
}

export interface GratitudeSummary {
  from_agent: AgentPubKey;
  to_agent: AgentPubKey;
  count: number;
}

export interface RhythmSummary {
  rhythm_hash: ActionHash;
  occurrences: number;
  avg_participation_bp: number;
}

export interface WeeklyDigest {
  hearth_hash: ActionHash;
  epoch_start: Timestamp;
  epoch_end: Timestamp;
  bond_updates: BondUpdate[];
  care_summary: CareSummary[];
  gratitude_summary: GratitudeSummary[];
  rhythm_summary: RhythmSummary[];
  created_by: AgentPubKey;
  created_at: Timestamp;
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

export interface UpdateStoryInput {
  story_hash: ActionHash;
  title: string;
  content: string;
  tags: string[];
}

export interface AddMediaInput {
  story_hash: ActionHash;
  media_hash: ActionHash;
}

export interface AddToCollectionInput {
  collection_hash: ActionHash;
  story_hash: ActionHash;
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

export interface UpdatePlanInput {
  plan_hash: ActionHash;
  input: CreateEmergencyPlanInput;
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

export interface DigestEpochInput {
  hearth_hash: ActionHash;
  epoch_start: Timestamp;
  epoch_end: Timestamp;
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
// Decision Input Types (extended)
// ============================================================================

export interface FinalizeDecisionInput {
  decision_hash: ActionHash;
}

export interface CloseDecisionInput {
  decision_hash: ActionHash;
  reason: string;
}

export interface AmendVoteInput {
  original_vote_hash: ActionHash;
  new_choice_index: number;
  reason?: string;
}

// ============================================================================
// Bridge Dispatch Types
// ============================================================================

export interface DispatchInput {
  target_zome: string;
  fn_name: string;
  payload: Uint8Array;
}

export interface DispatchResult {
  success: boolean;
  payload?: Uint8Array;
  error?: string;
}

export interface ResolveQueryInput {
  domain: string;
  query_type: string;
  params: Uint8Array;
}

export interface EventTypeQuery {
  event_type: string;
  limit?: number;
}

export interface CrossClusterDispatchInput {
  target_role: string;
  target_zome: string;
  fn_name: string;
  payload: Uint8Array;
}

export interface SeveranceInput {
  hearth_hash: ActionHash;
  reason: string;
}

export interface HearthSyncInput {
  hearth_hash: ActionHash;
  epoch_start: Timestamp;
  epoch_end: Timestamp;
}

export interface BridgeHealth {
  status: string;
  connected_zomes: number;
  last_dispatch_at?: Timestamp;
}

export interface GateAuditInput {
  action_name: string;
  zome_name: string;
  eligible: boolean;
  actual_tier: string;
  required_tier: string;
  weight_bp: number;
}

export interface GovernanceAuditFilter {
  action_name?: string;
  zome_name?: string;
  eligible?: boolean;
  from_us?: number;
  to_us?: number;
}

export interface GovernanceAuditResult {
  entries: GateAuditInput[];
  total_matched: number;
}

export type ConsciousnessTier = 'Dormant' | 'Awakening' | 'Aware' | 'Reflective' | 'Transcendent';

export interface ConsciousnessCredential {
  did: string;
  tier: ConsciousnessTier;
  phi_score: number;
  issued_at: Timestamp;
}

// ============================================================================
// Signal Types (match Rust HearthSignal enum in hearth-types)
// ============================================================================

/** Gratitude expressed between members */
export interface GratitudeExpressedSignal {
  from_agent: AgentPubKey;
  to_agent: AgentPubKey;
  message: string;
  gratitude_type: GratitudeType;
}

/** Care task completed by assignee */
export interface CareTaskCompletedSignal {
  assignee: AgentPubKey;
  schedule_hash: ActionHash;
  care_type: CareType;
}

/** Care swap accepted by responder */
export interface SwapAcceptedSignal {
  swap_hash: ActionHash;
  hearth_hash: ActionHash;
}

/** Care swap declined by responder */
export interface SwapDeclinedSignal {
  swap_hash: ActionHash;
  hearth_hash: ActionHash;
}

/** Rhythm occurrence logged */
export interface RhythmOccurredSignal {
  rhythm_hash: ActionHash;
  participants: AgentPubKey[];
}

/** Presence status changed */
export interface PresenceChangedSignal {
  agent: AgentPubKey;
  status: PresenceStatusType;
}

/** Emergency alert raised */
export interface EmergencyAlertSignal {
  alert_hash: ActionHash;
  severity: AlertSeverity;
  message: string;
}

/** Member joined a hearth */
export interface MemberJoinedSignal {
  hearth_hash: ActionHash;
  agent: AgentPubKey;
  role: MemberRole;
}

/** Member departed a hearth */
export interface MemberDepartedSignal {
  hearth_hash: ActionHash;
  agent: AgentPubKey;
}

/** Kinship bond tended */
export interface BondTendedSignal {
  member_a: AgentPubKey;
  member_b: AgentPubKey;
  quality_bp: number;
}

/** Cross-zome call failed (observability, non-blocking) */
export interface CrossZomeCallFailedSignal {
  zome: string;
  function: string;
  error: string;
}

/** Vote cast on a decision */
export interface VoteCastSignal {
  decision_hash: ActionHash;
  voter: AgentPubKey;
  choice: number;
}

/** Vote amended (old replaced with new) */
export interface VoteAmendedSignal {
  decision_hash: ActionHash;
  voter: AgentPubKey;
  old_choice: number;
  new_choice: number;
}

/** Decision manually closed before deadline */
export interface DecisionClosedSignal {
  decision_hash: ActionHash;
  closed_by: AgentPubKey;
}

/** Decision finalized with a winning option */
export interface DecisionFinalizedSignal {
  decision_hash: ActionHash;
  chosen_option: number;
  participation_rate_bp: number;
}

/** Family story created */
export interface StoryCreatedSignal {
  hearth_hash: ActionHash;
  story_hash: ActionHash;
  storyteller: AgentPubKey;
  story_type: StoryType;
}

/** Story updated by storyteller */
export interface StoryUpdatedSignal {
  story_hash: ActionHash;
  updated_by: AgentPubKey;
}

/** Family tradition observed */
export interface TraditionObservedSignal {
  tradition_hash: ActionHash;
  observed_by: AgentPubKey;
}

/** Shared resource lent to member */
export interface ResourceLentSignal {
  resource_hash: ActionHash;
  borrower: AgentPubKey;
  due_date: Timestamp;
}

/** Borrowed resource returned */
export interface ResourceReturnedSignal {
  loan_hash: ActionHash;
  borrower: AgentPubKey;
}

/** Expense logged against budget */
export interface ExpenseLoggedSignal {
  budget_hash: ActionHash;
  amount_cents: number;
}

/** Milestone recorded in family timeline */
export interface MilestoneRecordedSignal {
  hearth_hash: ActionHash;
  milestone_hash: ActionHash;
  milestone_type: MilestoneType;
}

/** Life transition advanced to new phase */
export interface TransitionAdvancedSignal {
  transition_hash: ActionHash;
  new_phase: TransitionPhase;
}

/** Autonomy tier advanced (guardian action) */
export interface TierAdvancedSignal {
  profile_hash: ActionHash;
  from_tier: AutonomyTier;
  to_tier: AutonomyTier;
}

/** Capability request approved */
export interface CapabilityApprovedSignal {
  request_hash: ActionHash;
  capability: string;
}

/** Capability request denied */
export interface CapabilityDeniedSignal {
  request_hash: ActionHash;
  capability: string;
}

/** Tier transition progressed to next phase */
export interface TransitionProgressedSignal {
  transition_hash: ActionHash;
  new_phase: TransitionPhase;
}

/** Bridge event signal (separate from HearthSignal enum) */
export interface BridgeEventSignal {
  signal_type: string;
  domain: string;
  event_type: string;
  payload: string;
  action_hash: ActionHash;
}

/**
 * Discriminated union of all HearthSignal variants.
 *
 * Matches Rust's serde externally-tagged enum representation:
 * `{ "VariantName": { field1: ..., field2: ... } }`
 */
export type HearthSignal =
  | { GratitudeExpressed: GratitudeExpressedSignal }
  | { CareTaskCompleted: CareTaskCompletedSignal }
  | { SwapAccepted: SwapAcceptedSignal }
  | { SwapDeclined: SwapDeclinedSignal }
  | { RhythmOccurred: RhythmOccurredSignal }
  | { PresenceChanged: PresenceChangedSignal }
  | { EmergencyAlert: EmergencyAlertSignal }
  | { MemberJoined: MemberJoinedSignal }
  | { MemberDeparted: MemberDepartedSignal }
  | { BondTended: BondTendedSignal }
  | { CrossZomeCallFailed: CrossZomeCallFailedSignal }
  | { VoteCast: VoteCastSignal }
  | { VoteAmended: VoteAmendedSignal }
  | { DecisionClosed: DecisionClosedSignal }
  | { DecisionFinalized: DecisionFinalizedSignal }
  | { StoryCreated: StoryCreatedSignal }
  | { StoryUpdated: StoryUpdatedSignal }
  | { TraditionObserved: TraditionObservedSignal }
  | { ResourceLent: ResourceLentSignal }
  | { ResourceReturned: ResourceReturnedSignal }
  | { ExpenseLogged: ExpenseLoggedSignal }
  | { MilestoneRecorded: MilestoneRecordedSignal }
  | { TransitionAdvanced: TransitionAdvancedSignal }
  | { TierAdvanced: TierAdvancedSignal }
  | { CapabilityApproved: CapabilityApprovedSignal }
  | { CapabilityDenied: CapabilityDeniedSignal }
  | { TransitionProgressed: TransitionProgressedSignal };

/** All possible HearthSignal variant names */
export type HearthSignalType =
  | 'GratitudeExpressed'
  | 'CareTaskCompleted'
  | 'SwapAccepted'
  | 'SwapDeclined'
  | 'RhythmOccurred'
  | 'PresenceChanged'
  | 'EmergencyAlert'
  | 'MemberJoined'
  | 'MemberDeparted'
  | 'BondTended'
  | 'CrossZomeCallFailed'
  | 'VoteCast'
  | 'VoteAmended'
  | 'DecisionClosed'
  | 'DecisionFinalized'
  | 'StoryCreated'
  | 'StoryUpdated'
  | 'TraditionObserved'
  | 'ResourceLent'
  | 'ResourceReturned'
  | 'ExpenseLogged'
  | 'MilestoneRecorded'
  | 'TransitionAdvanced'
  | 'TierAdvanced'
  | 'CapabilityApproved'
  | 'CapabilityDenied'
  | 'TransitionProgressed';

/** Extract the variant name from a HearthSignal */
export function getSignalType(signal: HearthSignal): HearthSignalType {
  return Object.keys(signal)[0] as HearthSignalType;
}

/** Signal types emitted by each zome */
export const ZOME_SIGNAL_MAP: Record<string, readonly HearthSignalType[]> = {
  hearth_kinship: ['MemberJoined', 'MemberDeparted', 'BondTended', 'CrossZomeCallFailed'],
  hearth_gratitude: ['GratitudeExpressed'],
  hearth_stories: ['StoryCreated', 'StoryUpdated', 'TraditionObserved'],
  hearth_care: ['CareTaskCompleted', 'SwapAccepted', 'SwapDeclined'],
  hearth_autonomy: ['TierAdvanced', 'CapabilityApproved', 'CapabilityDenied', 'TransitionProgressed', 'CrossZomeCallFailed'],
  hearth_emergency: ['EmergencyAlert'],
  hearth_decisions: ['VoteCast', 'VoteAmended', 'DecisionClosed', 'DecisionFinalized'],
  hearth_resources: ['ResourceLent', 'ResourceReturned', 'ExpenseLogged'],
  hearth_milestones: ['MilestoneRecorded', 'TransitionAdvanced'],
  hearth_rhythms: ['RhythmOccurred', 'PresenceChanged'],
} as const;

// ============================================================================
// Record Decoding Helpers
// ============================================================================

import type { Record as HolochainRecord, RecordEntry } from '@holochain/client';

/**
 * Extract and cast the entry from a Holochain Record.
 *
 * @throws {HearthError} if the entry is not present
 *
 * @example
 * ```typescript
 * const record = await hearth.kinship.createHearth(input);
 * const hearth = decodeRecord<Hearth>(record);
 * console.log(hearth.name);
 * ```
 */
export function decodeRecord<T>(record: HolochainRecord): T {
  const entry = record.entry as RecordEntry;
  if (!entry || !('Present' in entry)) {
    throw new HearthError('NOT_FOUND', 'Record entry is not present');
  }
  return (entry as { Present: T }).Present;
}

/**
 * Extract and cast entries from an array of Holochain Records.
 * Records without present entries are silently skipped.
 *
 * @example
 * ```typescript
 * const records = await hearth.care.getHearthSchedule(hearthHash);
 * const schedules = decodeRecords<CareSchedule>(records);
 * ```
 */
export function decodeRecords<T>(records: HolochainRecord[]): T[] {
  const results: T[] = [];
  for (const record of records) {
    const entry = record.entry as RecordEntry;
    if (entry && 'Present' in entry) {
      results.push((entry as { Present: T }).Present);
    }
  }
  return results;
}

/**
 * Extract the ActionHash from a Record's signed action.
 *
 * @example
 * ```typescript
 * const record = await hearth.kinship.createHearth(input);
 * const hearthHash = getRecordActionHash(record);
 * ```
 */
export function getRecordActionHash(record: HolochainRecord): Uint8Array {
  return record.signed_action.hashed.hash;
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
