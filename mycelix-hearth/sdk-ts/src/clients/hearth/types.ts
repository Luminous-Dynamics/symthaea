/**
 * TypeScript types for the Mycelix Hearth cluster.
 * Mirrors Rust entry types, input structs, and signal enums from all 11 zomes.
 */

import type { ActionHash, AgentPubKey, Timestamp } from '@holochain/client';

// ============================================================================
// Enums — Hearth Core
// ============================================================================

export type HearthType =
  | 'Nuclear'
  | 'Extended'
  | 'Chosen'
  | 'Blended'
  | 'Multigenerational'
  | 'Intentional'
  | 'CoPod'
  | { Custom: string };

export type MemberRole =
  | 'Founder'
  | 'Elder'
  | 'Adult'
  | 'Youth'
  | 'Child'
  | 'Guest'
  | 'Ancestor';

export type MembershipStatus = 'Active' | 'Invited' | 'Departed' | 'Ancestral';

export type BondType =
  | 'Parent'
  | 'Child'
  | 'Sibling'
  | 'Partner'
  | 'Grandparent'
  | 'Grandchild'
  | 'AuntUncle'
  | 'NieceNephew'
  | 'Cousin'
  | 'ChosenFamily'
  | 'Guardian'
  | 'Ward'
  | { Custom: string };

export type InvitationStatus = 'Pending' | 'Accepted' | 'Declined' | 'Expired';

// ============================================================================
// Enums — Decisions
// ============================================================================

export type DecisionType =
  | 'Consensus'
  | 'MajorityVote'
  | 'ElderDecision'
  | 'GuardianDecision';

export type DecisionStatus = 'Open' | 'Closed' | 'Finalized';

// ============================================================================
// Enums — Autonomy
// ============================================================================

export type AutonomyTier =
  | 'Dependent'
  | 'Supervised'
  | 'Guided'
  | 'SemiAutonomous'
  | 'Autonomous';

export type AutonomyRequestStatus = 'Pending' | 'Approved' | 'Denied';

// ============================================================================
// Enums — Visibility
// ============================================================================

export type HearthVisibility =
  | 'AllMembers'
  | 'AdultsOnly'
  | 'GuardiansOnly'
  | { Specified: AgentPubKey[] };

// ============================================================================
// Enums — Care
// ============================================================================

export type CareType =
  | 'Childcare'
  | 'Eldercare'
  | 'PetCare'
  | 'Chore'
  | 'MealPrep'
  | 'Medical'
  | 'Emotional'
  | { Custom: string };

export type CareScheduleStatus = 'Active' | 'Paused' | 'Completed';

export type SwapStatus = 'Proposed' | 'Accepted' | 'Declined' | 'Completed';

export type Recurrence =
  | 'Daily'
  | 'Weekly'
  | 'Monthly'
  | { Custom: string };

// ============================================================================
// Enums — Emergency
// ============================================================================

export type AlertSeverity = 'Low' | 'Medium' | 'High' | 'Critical';

export type AlertType =
  | 'Medical'
  | 'Natural'
  | 'Security'
  | 'Missing'
  | 'Fire'
  | { Custom: string };

export type SafetyStatus = 'Safe' | 'NeedHelp' | 'NoResponse';

// ============================================================================
// Enums — Gratitude
// ============================================================================

export type GratitudeType =
  | 'Appreciation'
  | 'Acknowledgment'
  | 'Celebration'
  | 'Blessing'
  | { Custom: string };

export type CircleStatus = 'Open' | 'InProgress' | 'Completed';

// ============================================================================
// Enums — Stories
// ============================================================================

export type StoryType =
  | 'Memory'
  | 'Tradition'
  | 'Recipe'
  | 'Wisdom'
  | 'Origin'
  | 'Migration'
  | { Custom: string };

// ============================================================================
// Enums — Milestones & Transitions
// ============================================================================

export type MilestoneType =
  | 'Birth'
  | 'Birthday'
  | 'FirstStep'
  | 'SchoolStart'
  | 'Graduation'
  | 'Engagement'
  | 'Marriage'
  | 'NewHome'
  | 'Retirement'
  | 'Passing'
  | { Custom: string };

export type TransitionType =
  | 'JoiningHearth'
  | 'LeavingHearth'
  | 'ComingOfAge'
  | 'Retirement'
  | 'Bereavement'
  | { Custom: string };

export type TransitionPhase =
  | 'PreLiminal'
  | 'Liminal'
  | 'PostLiminal'
  | 'Integrated';

// ============================================================================
// Enums — Resources
// ============================================================================

export type ResourceType =
  | 'Tool'
  | 'Vehicle'
  | 'Book'
  | 'Kitchen'
  | 'Electronics'
  | 'Clothing'
  | { Custom: string };

export type LoanStatus = 'Active' | 'Returned' | 'Overdue';

// ============================================================================
// Enums — Rhythms
// ============================================================================

export type RhythmType =
  | 'Morning'
  | 'Evening'
  | 'Weekly'
  | 'Seasonal'
  | { Custom: string };

export type PresenceStatusType =
  | 'Home'
  | 'Away'
  | 'Working'
  | 'Sleeping'
  | 'DoNotDisturb';

// ============================================================================
// Entry Types — Kinship
// ============================================================================

export interface Hearth {
  name: string;
  description: string;
  hearth_type: HearthType;
  created_by: AgentPubKey;
  created_at: Timestamp;
  max_members: number;
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

// ============================================================================
// Entry Types — Decisions
// ============================================================================

export interface Decision {
  hearth_hash: ActionHash;
  title: string;
  description: string;
  decision_type: DecisionType;
  eligible_roles: MemberRole[];
  options: string[];
  deadline: Timestamp;
  quorum_bp?: number;
  status: DecisionStatus;
  created_by: AgentPubKey;
  created_at: Timestamp;
}

export interface Vote {
  decision_hash: ActionHash;
  voter: AgentPubKey;
  choice: number;
  weight_bp: number;
  reasoning?: string;
  created_at: Timestamp;
}

export interface DecisionOutcome {
  decision_hash: ActionHash;
  chosen_option: number;
  participation_rate_bp: number;
  resolved_at: Timestamp;
  quorum_bp?: number;
}

// ============================================================================
// Entry Types — Gratitude
// ============================================================================

export interface GratitudeExpression {
  hearth_hash: ActionHash;
  from_agent: AgentPubKey;
  to_agent: AgentPubKey;
  message: string;
  gratitude_type: GratitudeType;
  visibility: HearthVisibility;
  created_at: Timestamp;
}

export interface AppreciationCircle {
  hearth_hash: ActionHash;
  theme: string;
  participants: AgentPubKey[];
  started_at: Timestamp;
  completed_at?: Timestamp;
  status: CircleStatus;
}

export interface GratitudeAnchor {
  agent: AgentPubKey;
  hearth_hash: ActionHash;
  total_given: number;
  total_received: number;
  current_streak_days: number;
}

// ============================================================================
// Entry Types — Stories
// ============================================================================

export interface FamilyStory {
  hearth_hash: ActionHash;
  title: string;
  content: string;
  storyteller: AgentPubKey;
  story_type: StoryType;
  media_hashes: ActionHash[];
  tags: string[];
  visibility: HearthVisibility;
  created_at: Timestamp;
}

export interface StoryCollection {
  hearth_hash: ActionHash;
  name: string;
  description: string;
  story_hashes: ActionHash[];
  curator: AgentPubKey;
}

export interface FamilyTradition {
  hearth_hash: ActionHash;
  name: string;
  description: string;
  frequency: Recurrence;
  season?: string;
  instructions: string;
  last_observed?: Timestamp;
  next_due?: Timestamp;
}

// ============================================================================
// Entry Types — Care
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
  completed_at?: Timestamp;
}

export interface CareSwap {
  hearth_hash: ActionHash;
  requester: AgentPubKey;
  responder: AgentPubKey;
  original_schedule_hash: ActionHash;
  swap_date: Timestamp;
  status: SwapStatus;
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
  recipe: string;
  servings: number;
}

// ============================================================================
// Entry Types — Autonomy
// ============================================================================

export interface AutonomyProfile {
  hearth_hash: ActionHash;
  member: AgentPubKey;
  guardian_agents: AgentPubKey[];
  current_tier: AutonomyTier;
  capabilities: string[];
  restrictions: string[];
  review_schedule?: string;
  created_at: Timestamp;
}

export interface AutonomyRequest {
  hearth_hash: ActionHash;
  requester: AgentPubKey;
  capability: string;
  justification: string;
  status: AutonomyRequestStatus;
  created_at: Timestamp;
}

export interface GuardianApproval {
  request_hash: ActionHash;
  guardian: AgentPubKey;
  approved: boolean;
  conditions?: string;
  created_at: Timestamp;
}

export interface TierTransition {
  hearth_hash: ActionHash;
  member: AgentPubKey;
  from_tier: AutonomyTier;
  to_tier: AutonomyTier;
  transition_phase: TransitionPhase;
  recategorization_blocked: boolean;
  started_at: Timestamp;
  completed_at?: Timestamp;
}

// ============================================================================
// Entry Types — Emergency
// ============================================================================

export interface EmergencyPlan {
  hearth_hash: ActionHash;
  contacts: EmergencyContact[];
  meeting_points: string[];
  medical_info_hashes: ActionHash[];
  last_reviewed: Timestamp;
}

export interface EmergencyAlert {
  hearth_hash: ActionHash;
  alert_type: AlertType;
  severity: AlertSeverity;
  message: string;
  reporter: AgentPubKey;
  location_hint?: string;
  created_at: Timestamp;
  resolved_at?: Timestamp;
}

export interface SafetyCheckIn {
  hearth_hash: ActionHash;
  alert_hash: ActionHash;
  member: AgentPubKey;
  status: SafetyStatus;
  location_hint?: string;
  checked_in_at: Timestamp;
}

export interface EmergencyContact {
  name: string;
  phone: string;
  relationship: string;
  priority_order: number;
}

// ============================================================================
// Entry Types — Resources
// ============================================================================

export interface SharedResource {
  hearth_hash: ActionHash;
  name: string;
  description: string;
  resource_type: ResourceType;
  current_holder?: AgentPubKey;
  condition: string;
  location: string;
}

export interface ResourceLoan {
  resource_hash: ActionHash;
  lender_hearth: ActionHash;
  borrower: AgentPubKey;
  due_date: Timestamp;
  status: LoanStatus;
  created_at: Timestamp;
}

export interface BudgetCategory {
  hearth_hash: ActionHash;
  category: string;
  monthly_target_cents: number;
  current_month_actual_cents: number;
}

// ============================================================================
// Entry Types — Milestones
// ============================================================================

export interface Milestone {
  hearth_hash: ActionHash;
  member: AgentPubKey;
  milestone_type: MilestoneType;
  date: Timestamp;
  description: string;
  witnesses: AgentPubKey[];
  media_hashes: ActionHash[];
  created_at: Timestamp;
}

export interface LifeTransition {
  hearth_hash: ActionHash;
  member: AgentPubKey;
  transition_type: TransitionType;
  current_phase: TransitionPhase;
  supporting_members: AgentPubKey[];
  recategorization_blocked: boolean;
  started_at: Timestamp;
  completed_at?: Timestamp;
}

// ============================================================================
// Entry Types — Rhythms
// ============================================================================

export interface Rhythm {
  hearth_hash: ActionHash;
  name: string;
  rhythm_type: RhythmType;
  schedule: string;
  participants: AgentPubKey[];
  description: string;
  created_at: Timestamp;
}

export interface RhythmOccurrence {
  rhythm_hash: ActionHash;
  date: Timestamp;
  participants_present: AgentPubKey[];
  notes: string;
  mood_bp?: number;
  created_at: Timestamp;
}

export interface PresenceStatus {
  hearth_hash: ActionHash;
  agent: AgentPubKey;
  status: PresenceStatusType;
  expected_return?: Timestamp;
  updated_at: Timestamp;
}

// ============================================================================
// Entry Types — Epoch Rollups
// ============================================================================

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

// ============================================================================
// Input Types — Kinship
// ============================================================================

export interface CreateHearthInput {
  name: string;
  description: string;
  hearth_type: HearthType;
  max_members?: number;
}

export interface InviteMemberInput {
  hearth_hash: ActionHash;
  invitee_agent: AgentPubKey;
  proposed_role: MemberRole;
  message: string;
  expires_at: Timestamp;
}

export interface AcceptInvitationInput {
  invitation_hash: ActionHash;
  display_name: string;
}

export interface UpdateMemberRoleInput {
  membership_hash: ActionHash;
  new_role: MemberRole;
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

// ============================================================================
// Input Types — Decisions
// ============================================================================

export interface CreateDecisionInput {
  hearth_hash: ActionHash;
  title: string;
  description: string;
  decision_type: DecisionType;
  eligible_roles: MemberRole[];
  options: string[];
  deadline: Timestamp;
  quorum_bp?: number;
}

export interface CastVoteInput {
  decision_hash: ActionHash;
  choice: number;
  reasoning?: string;
}

export interface FinalizeDecisionInput {
  decision_hash: ActionHash;
}

export interface CloseDecisionInput {
  decision_hash: ActionHash;
}

export interface AmendVoteInput {
  decision_hash: ActionHash;
  choice: number;
  reasoning?: string;
}

// ============================================================================
// Input Types — Gratitude
// ============================================================================

export interface ExpressGratitudeInput {
  hearth_hash: ActionHash;
  to_agent: AgentPubKey;
  message: string;
  gratitude_type: GratitudeType;
  visibility: HearthVisibility;
}

export interface StartCircleInput {
  hearth_hash: ActionHash;
  theme: string;
  participants: AgentPubKey[];
}

// ============================================================================
// Input Types — Stories
// ============================================================================

export interface CreateStoryInput {
  hearth_hash: ActionHash;
  title: string;
  content: string;
  story_type: StoryType;
  media_hashes: ActionHash[];
  tags: string[];
  visibility: HearthVisibility;
}

export interface UpdateStoryInput {
  story_hash: ActionHash;
  title: string;
  content: string;
  tags: string[];
}

export interface CreateCollectionInput {
  hearth_hash: ActionHash;
  name: string;
  description: string;
}

export interface CreateTraditionInput {
  hearth_hash: ActionHash;
  name: string;
  description: string;
  frequency: Recurrence;
  season?: string;
  instructions: string;
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
// Input Types — Care
// ============================================================================

export interface CreateCareScheduleInput {
  hearth_hash: ActionHash;
  care_type: CareType;
  title: string;
  description: string;
  assigned_to: AgentPubKey;
  recurrence: Recurrence;
  notes: string;
}

export interface CompleteTaskInput {
  schedule_hash: ActionHash;
}

export interface ProposeSwapInput {
  hearth_hash: ActionHash;
  original_schedule_hash: ActionHash;
  swap_date: Timestamp;
}

export interface CreateMealPlanInput {
  hearth_hash: ActionHash;
  week_start: Timestamp;
  meals: PlannedMeal[];
  shopper: AgentPubKey;
  cook: AgentPubKey;
  dietary_notes: string;
}

// ============================================================================
// Input Types — Autonomy
// ============================================================================

export interface CreateAutonomyProfileInput {
  hearth_hash: ActionHash;
  member: AgentPubKey;
  guardian_agents: AgentPubKey[];
  initial_tier: AutonomyTier;
  capabilities: string[];
  restrictions: string[];
  review_schedule?: string;
}

export interface RequestCapabilityInput {
  hearth_hash: ActionHash;
  capability: string;
  justification: string;
}

export interface ApproveCapabilityInput {
  request_hash: ActionHash;
  approved: boolean;
  conditions?: string;
}

export interface AdvanceTierInput {
  profile_hash: ActionHash;
  new_tier: AutonomyTier;
}

export interface CheckCapabilityInput {
  member: AgentPubKey;
  capability: string;
}

// ============================================================================
// Input Types — Emergency
// ============================================================================

export interface CreateEmergencyPlanInput {
  hearth_hash: ActionHash;
  contacts: EmergencyContact[];
  meeting_points: string[];
  medical_info_hashes: ActionHash[];
}

export interface UpdatePlanInput {
  plan_hash: ActionHash;
  input: CreateEmergencyPlanInput;
}

export interface RaiseAlertInput {
  hearth_hash: ActionHash;
  alert_type: AlertType;
  severity: AlertSeverity;
  message: string;
  location_hint?: string;
}

export interface CheckInInput {
  alert_hash: ActionHash;
  status: SafetyStatus;
  location_hint?: string;
}

// ============================================================================
// Input Types — Resources
// ============================================================================

export interface RegisterResourceInput {
  hearth_hash: ActionHash;
  name: string;
  description: string;
  resource_type: ResourceType;
  condition: string;
  location: string;
}

export interface LendResourceInput {
  resource_hash: ActionHash;
  borrower: AgentPubKey;
  due_date: Timestamp;
}

export interface CreateBudgetInput {
  hearth_hash: ActionHash;
  category: string;
  monthly_target_cents: number;
}

export interface LogExpenseInput {
  budget_hash: ActionHash;
  amount_cents: number;
}

// ============================================================================
// Input Types — Milestones
// ============================================================================

export interface RecordMilestoneInput {
  hearth_hash: ActionHash;
  member: AgentPubKey;
  milestone_type: MilestoneType;
  date: Timestamp;
  description: string;
  witnesses: AgentPubKey[];
  media_hashes: ActionHash[];
}

export interface BeginTransitionInput {
  hearth_hash: ActionHash;
  member: AgentPubKey;
  transition_type: TransitionType;
  supporting_members: AgentPubKey[];
}

// ============================================================================
// Input Types — Rhythms
// ============================================================================

export interface CreateRhythmInput {
  hearth_hash: ActionHash;
  name: string;
  rhythm_type: RhythmType;
  schedule: string;
  participants: AgentPubKey[];
  description: string;
}

export interface LogOccurrenceInput {
  rhythm_hash: ActionHash;
  participants_present: AgentPubKey[];
  notes: string;
  mood_bp?: number;
}

export interface SetPresenceInput {
  hearth_hash: ActionHash;
  status: PresenceStatusType;
  expected_return?: Timestamp;
}

// ============================================================================
// Input Types — Shared / Bridge
// ============================================================================

export interface DigestEpochInput {
  hearth_hash: ActionHash;
  epoch_start: Timestamp;
  epoch_end: Timestamp;
}

export interface SeveranceInput {
  hearth_hash: ActionHash;
  member_hash: ActionHash;
  export_milestones: boolean;
  export_care_history: boolean;
  export_bond_snapshot: boolean;
  new_role: MemberRole;
}

export interface SeveranceSummaryData {
  hearth_hash: ActionHash;
  member: AgentPubKey;
  milestones_exported: number;
  care_records_exported: number;
  bond_snapshot_exported: boolean;
  new_role: MemberRole;
  completed_at: Timestamp;
}

export interface HearthSyncInput {
  hearth_hash: ActionHash;
  epoch_start: Timestamp;
  epoch_end: Timestamp;
}

export interface DispatchInput {
  zome: string;
  fn_name: string;
  payload: Uint8Array;
}

export interface DispatchResult {
  success: boolean;
  response?: Uint8Array;
  error?: string;
}

export interface CrossClusterDispatchInput {
  role: string;
  zome: string;
  fn_name: string;
  payload: Uint8Array;
}

export interface EventTypeQuery {
  domain: string;
  event_type: string;
}

export interface ResolveQueryInput {
  query_hash: ActionHash;
  result: string;
  success: boolean;
}

export interface BridgeHealth {
  healthy: boolean;
  agent: string;
  total_events: number;
  total_queries: number;
  domains: string[];
}

// ============================================================================
// Signal Types — Decision Signals
// ============================================================================

export interface VoteCastSignal {
  type: 'VoteCast';
  decision_hash: ActionHash;
  voter: AgentPubKey;
  choice: number;
}

export interface VoteAmendedSignal {
  type: 'VoteAmended';
  decision_hash: ActionHash;
  voter: AgentPubKey;
  old_choice: number;
  new_choice: number;
}

export interface DecisionClosedSignal {
  type: 'DecisionClosed';
  decision_hash: ActionHash;
  closed_by: AgentPubKey;
}

export interface DecisionFinalizedSignal {
  type: 'DecisionFinalized';
  decision_hash: ActionHash;
  chosen_option: number;
  participation_rate_bp: number;
}

export type DecisionSignal =
  | VoteCastSignal
  | VoteAmendedSignal
  | DecisionClosedSignal
  | DecisionFinalizedSignal;

export type DecisionSignalType = DecisionSignal['type'];

// ============================================================================
// Signal Types — Kinship Signals
// ============================================================================

export interface MemberJoinedSignal {
  type: 'MemberJoined';
  hearth_hash: ActionHash;
  agent: AgentPubKey;
  role: MemberRole;
}

export interface MemberDepartedSignal {
  type: 'MemberDeparted';
  hearth_hash: ActionHash;
  agent: AgentPubKey;
}

export interface BondTendedSignal {
  type: 'BondTended';
  member_a: AgentPubKey;
  member_b: AgentPubKey;
  quality_bp: number;
}

export type KinshipSignal =
  | MemberJoinedSignal
  | MemberDepartedSignal
  | BondTendedSignal;

export type KinshipSignalType = KinshipSignal['type'];

// ============================================================================
// Signal Types — Domain Signals
// ============================================================================

export interface GratitudeExpressedSignal {
  type: 'GratitudeExpressed';
  from_agent: AgentPubKey;
  to_agent: AgentPubKey;
  message: string;
  gratitude_type: GratitudeType;
}

export interface CareTaskCompletedSignal {
  type: 'CareTaskCompleted';
  assignee: AgentPubKey;
  schedule_hash: ActionHash;
  care_type: CareType;
}

export interface EmergencyAlertSignal {
  type: 'EmergencyAlert';
  alert_hash: ActionHash;
  severity: AlertSeverity;
  message: string;
}

export interface RhythmOccurredSignal {
  type: 'RhythmOccurred';
  rhythm_hash: ActionHash;
  participants: AgentPubKey[];
}

export interface PresenceChangedSignal {
  type: 'PresenceChanged';
  agent: AgentPubKey;
  status: PresenceStatusType;
}

export interface CrossZomeCallFailedSignal {
  type: 'CrossZomeCallFailed';
  zome: string;
  function: string;
  error: string;
}

// ============================================================================
// Signal Types — Story Signals
// ============================================================================

export interface StoryCreatedSignal {
  type: 'StoryCreated';
  hearth_hash: ActionHash;
  story_hash: ActionHash;
  storyteller: AgentPubKey;
  story_type: StoryType;
}

export interface StoryUpdatedSignal {
  type: 'StoryUpdated';
  story_hash: ActionHash;
  updated_by: AgentPubKey;
}

export interface TraditionObservedSignal {
  type: 'TraditionObserved';
  tradition_hash: ActionHash;
  observed_by: AgentPubKey;
}

export type StorySignal =
  | StoryCreatedSignal
  | StoryUpdatedSignal
  | TraditionObservedSignal;

export type StorySignalType = StorySignal['type'];

// ============================================================================
// Signal Types — Resource Signals
// ============================================================================

export interface ResourceLentSignal {
  type: 'ResourceLent';
  resource_hash: ActionHash;
  borrower: AgentPubKey;
  due_date: Timestamp;
}

export interface ResourceReturnedSignal {
  type: 'ResourceReturned';
  loan_hash: ActionHash;
  borrower: AgentPubKey;
}

export interface ExpenseLoggedSignal {
  type: 'ExpenseLogged';
  budget_hash: ActionHash;
  amount_cents: number;
}

export type ResourceSignal =
  | ResourceLentSignal
  | ResourceReturnedSignal
  | ExpenseLoggedSignal;

export type ResourceSignalType = ResourceSignal['type'];

// ============================================================================
// Signal Types — Milestone Signals
// ============================================================================

export interface MilestoneRecordedSignal {
  type: 'MilestoneRecorded';
  hearth_hash: ActionHash;
  milestone_hash: ActionHash;
  milestone_type: MilestoneType;
}

export interface TransitionAdvancedSignal {
  type: 'TransitionAdvanced';
  transition_hash: ActionHash;
  new_phase: TransitionPhase;
}

export type MilestoneSignal =
  | MilestoneRecordedSignal
  | TransitionAdvancedSignal;

export type MilestoneSignalType = MilestoneSignal['type'];

// ============================================================================
// Signal Types — Autonomy Signals
// ============================================================================

export interface TierAdvancedSignal {
  type: 'TierAdvanced';
  profile_hash: ActionHash;
  from_tier: AutonomyTier;
  to_tier: AutonomyTier;
}

export interface CapabilityApprovedSignal {
  type: 'CapabilityApproved';
  request_hash: ActionHash;
  capability: string;
}

export interface CapabilityDeniedSignal {
  type: 'CapabilityDenied';
  request_hash: ActionHash;
  capability: string;
}

export interface TransitionProgressedSignal {
  type: 'TransitionProgressed';
  transition_hash: ActionHash;
  new_phase: TransitionPhase;
}

export type AutonomySignal =
  | TierAdvancedSignal
  | CapabilityApprovedSignal
  | CapabilityDeniedSignal
  | TransitionProgressedSignal;

export type AutonomySignalType = AutonomySignal['type'];

// ============================================================================
// Signal Types — Care Swap Signals
// ============================================================================

export interface SwapAcceptedSignal {
  type: 'SwapAccepted';
  swap_hash: ActionHash;
  hearth_hash: ActionHash;
}

export interface SwapDeclinedSignal {
  type: 'SwapDeclined';
  swap_hash: ActionHash;
  hearth_hash: ActionHash;
}

export type SwapSignal =
  | SwapAcceptedSignal
  | SwapDeclinedSignal;

export type SwapSignalType = SwapSignal['type'];

// ============================================================================
// Unified HearthSignal
// ============================================================================

export type HearthSignal =
  | DecisionSignal
  | KinshipSignal
  | StorySignal
  | ResourceSignal
  | MilestoneSignal
  | AutonomySignal
  | SwapSignal
  | GratitudeExpressedSignal
  | CareTaskCompletedSignal
  | EmergencyAlertSignal
  | RhythmOccurredSignal
  | PresenceChangedSignal
  | CrossZomeCallFailedSignal;

export type HearthSignalType = HearthSignal['type'];

// ============================================================================
// Consciousness Profile Types (from mycelix-bridge-common)
// ============================================================================

/** Governance tiers derived from combined consciousness score. */
export type ConsciousnessTier =
  | 'Observer'
  | 'Participant'
  | 'Citizen'
  | 'Steward'
  | 'Guardian';

/** 4-dimensional consciousness profile. Each dimension is 0.0-1.0. */
export interface ConsciousnessProfile {
  /** Identity verification strength (Anonymous=0.0 to Critical=1.0). */
  identity: number;
  /** Cross-hApp reputation with exponential decay. */
  reputation: number;
  /** Community trust attestations, weighted by attestor tier. */
  community: number;
  /** Domain-specific engagement, computed locally per bridge. */
  engagement: number;
}

/** Time-limited credential containing a ConsciousnessProfile. */
export interface ConsciousnessCredential {
  /** Agent's DID (e.g., "did:mycelix:<pubkey>"). */
  did: string;
  /** The multi-dimensional profile. */
  profile: ConsciousnessProfile;
  /** Derived tier at issuance time. */
  tier: ConsciousnessTier;
  /** Issuance timestamp (microseconds since epoch). */
  issued_at: number;
  /** Expiry timestamp (microseconds since epoch). */
  expires_at: number;
  /** DID of the issuing bridge. */
  issuer: string;
}

/** Audit input for logging governance gate decisions. */
export interface GateAuditInput {
  /** The extern function that triggered the gate check. */
  action_name: string;
  /** The zome that performed the check. */
  zome_name: string;
  /** Whether the agent met all requirements. */
  eligible: boolean;
  /** The agent's derived consciousness tier. */
  actual_tier: string;
  /** The minimum tier required by the governance action. */
  required_tier: string;
  /** Progressive vote weight in basis points (0-10000). */
  weight_bp: number;
}
