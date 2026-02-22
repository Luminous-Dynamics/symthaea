/**
 * Hearth hApp Client Tests
 *
 * Comprehensive unit tests for all 11 Hearth SDK client classes:
 * - KinshipClient (hearth_kinship) - 20 methods
 * - DecisionsClient (hearth_decisions) - 12 methods
 * - GratitudeClient (hearth_gratitude) - 8 methods
 * - StoriesClient (hearth_stories) - 11 methods
 * - HearthCareClient (hearth_care) - 10 methods
 * - AutonomyClient (hearth_autonomy) - 10 methods
 * - EmergencyClient (hearth_emergency) - 8 methods
 * - ResourcesClient (hearth_resources) - 8 methods
 * - MilestonesClient (hearth_milestones) - 7 methods
 * - RhythmsClient (hearth_rhythms) - 7 methods
 * - BridgeClient (hearth_bridge) - 22 methods
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  KinshipClient,
  DecisionsClient,
  GratitudeClient,
  StoriesClient,
  HearthCareClient,
  AutonomyClient,
  EmergencyClient,
  ResourcesClient,
  MilestonesClient,
  RhythmsClient,
  BridgeClient,
  HearthClient,
  createHearthClient,
  getSignalType,
  HearthError,
  type HearthSignal,
} from '../src/clients/hearth/index.js';

// ============================================================================
// Mock Setup
// ============================================================================

function createMockClient(responses: Map<string, unknown> = new Map()) {
  return {
    callZome: vi.fn(
      async <T>(params: {
        role_name: string;
        zome_name: string;
        fn_name: string;
        payload: unknown;
      }): Promise<T> => {
        const key = `${params.zome_name}:${params.fn_name}`;
        if (responses.has(key)) {
          return responses.get(key) as T;
        }
        throw new Error(`No mock response for ${key}`);
      }
    ),
  };
}

const MOCK_HASH = new Uint8Array(39);
const MOCK_AGENT = new Uint8Array(39);
const MOCK_RECORD = {
  signed_action: {
    hashed: { hash: MOCK_HASH, content: {} },
    signature: new Uint8Array(64),
  },
  entry: { Present: { entry_type: 'test' } },
};

const BASE_CONFIG = { roleName: 'hearth', timeout: 30000 };

// ============================================================================
// Helper: assert callZome was invoked with the expected shape
// ============================================================================

function expectZomeCall(
  mockClient: ReturnType<typeof createMockClient>,
  zomeName: string,
  fnName: string,
  payload: unknown,
) {
  expect(mockClient.callZome).toHaveBeenCalledWith({
    role_name: 'hearth',
    zome_name: zomeName,
    fn_name: fnName,
    payload,
  });
}

// ============================================================================
// KinshipClient Tests (hearth_kinship) - 20 methods
// ============================================================================

describe('KinshipClient', () => {
  let client: KinshipClient;
  let mock: ReturnType<typeof createMockClient>;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    responses.set('hearth_kinship:create_hearth', MOCK_RECORD);
    responses.set('hearth_kinship:get_my_hearths', [MOCK_RECORD]);
    responses.set('hearth_kinship:invite_member', MOCK_RECORD);
    responses.set('hearth_kinship:accept_invitation', MOCK_RECORD);
    responses.set('hearth_kinship:decline_invitation', MOCK_RECORD);
    responses.set('hearth_kinship:leave_hearth', MOCK_RECORD);
    responses.set('hearth_kinship:update_member_role', MOCK_RECORD);
    responses.set('hearth_kinship:get_hearth_members', [MOCK_RECORD]);
    responses.set('hearth_kinship:is_guardian', true);
    responses.set('hearth_kinship:get_caller_vote_weight', 100);
    responses.set('hearth_kinship:get_caller_role', 'Elder');
    responses.set('hearth_kinship:get_active_member_count', 5);
    responses.set('hearth_kinship:create_kinship_bond', MOCK_RECORD);
    responses.set('hearth_kinship:tend_bond', MOCK_RECORD);
    responses.set('hearth_kinship:get_bond_health', 8500);
    responses.set('hearth_kinship:get_kinship_graph', [MOCK_RECORD]);
    responses.set('hearth_kinship:get_neglected_bonds', [MOCK_RECORD]);
    responses.set('hearth_kinship:get_bond_snapshots', []);
    responses.set('hearth_kinship:create_weekly_digest', MOCK_RECORD);
    responses.set('hearth_kinship:get_weekly_digests', [MOCK_RECORD]);

    mock = createMockClient(responses);
    client = new KinshipClient(mock, BASE_CONFIG);
  });

  it('createHearth calls correct zome function', async () => {
    const input = { name: 'Stoltz Family', description: 'Our hearth', hearth_type: 'Nuclear' as const };
    await client.createHearth(input);
    expectZomeCall(mock, 'hearth_kinship', 'create_hearth', input);
  });

  it('getMyHearths calls correct zome function with null payload', async () => {
    await client.getMyHearths();
    expectZomeCall(mock, 'hearth_kinship', 'get_my_hearths', null);
  });

  it('inviteMember calls correct zome function', async () => {
    const input = { hearth_hash: MOCK_HASH, invitee_agent: MOCK_AGENT, proposed_role: 'Adult' as const, message: 'Welcome!', expires_at: 1000000 };
    await client.inviteMember(input);
    expectZomeCall(mock, 'hearth_kinship', 'invite_member', input);
  });

  it('acceptInvitation calls correct zome function', async () => {
    await client.acceptInvitation(MOCK_HASH);
    expectZomeCall(mock, 'hearth_kinship', 'accept_invitation', MOCK_HASH);
  });

  it('declineInvitation calls correct zome function', async () => {
    await client.declineInvitation(MOCK_HASH);
    expectZomeCall(mock, 'hearth_kinship', 'decline_invitation', MOCK_HASH);
  });

  it('leaveHearth calls correct zome function', async () => {
    await client.leaveHearth(MOCK_HASH);
    expectZomeCall(mock, 'hearth_kinship', 'leave_hearth', MOCK_HASH);
  });

  it('updateMemberRole calls correct zome function', async () => {
    const input = { membership_hash: MOCK_HASH, new_role: 'Elder' as const };
    await client.updateMemberRole(input);
    expectZomeCall(mock, 'hearth_kinship', 'update_member_role', input);
  });

  it('getHearthMembers calls correct zome function', async () => {
    await client.getHearthMembers(MOCK_HASH);
    expectZomeCall(mock, 'hearth_kinship', 'get_hearth_members', MOCK_HASH);
  });

  it('isGuardian calls correct zome function', async () => {
    const result = await client.isGuardian(MOCK_HASH);
    expectZomeCall(mock, 'hearth_kinship', 'is_guardian', MOCK_HASH);
    expect(result).toBe(true);
  });

  it('getCallerVoteWeight calls correct zome function', async () => {
    const result = await client.getCallerVoteWeight(MOCK_HASH);
    expectZomeCall(mock, 'hearth_kinship', 'get_caller_vote_weight', MOCK_HASH);
    expect(result).toBe(100);
  });

  it('getCallerRole calls correct zome function', async () => {
    const result = await client.getCallerRole(MOCK_HASH);
    expectZomeCall(mock, 'hearth_kinship', 'get_caller_role', MOCK_HASH);
    expect(result).toBe('Elder');
  });

  it('getActiveMemberCount calls correct zome function', async () => {
    const result = await client.getActiveMemberCount(MOCK_HASH);
    expectZomeCall(mock, 'hearth_kinship', 'get_active_member_count', MOCK_HASH);
    expect(result).toBe(5);
  });

  it('createBond calls correct zome function', async () => {
    const input = { hearth_hash: MOCK_HASH, member_b: MOCK_AGENT, bond_type: 'Sibling' as const };
    await client.createBond(input);
    expectZomeCall(mock, 'hearth_kinship', 'create_kinship_bond', input);
  });

  it('tendBond calls correct zome function', async () => {
    const input = { bond_hash: MOCK_HASH, description: 'Shared dinner', quality_bp: 8000 };
    await client.tendBond(input);
    expectZomeCall(mock, 'hearth_kinship', 'tend_bond', input);
  });

  it('getBondHealth calls correct zome function', async () => {
    const input = { bond_hash: MOCK_HASH };
    const result = await client.getBondHealth(input);
    expectZomeCall(mock, 'hearth_kinship', 'get_bond_health', input);
    expect(result).toBe(8500);
  });

  it('getKinshipGraph calls correct zome function', async () => {
    await client.getKinshipGraph(MOCK_HASH);
    expectZomeCall(mock, 'hearth_kinship', 'get_kinship_graph', MOCK_HASH);
  });

  it('getNeglectedBonds calls correct zome function', async () => {
    await client.getNeglectedBonds(MOCK_HASH);
    expectZomeCall(mock, 'hearth_kinship', 'get_neglected_bonds', MOCK_HASH);
  });

  it('getBondSnapshots calls correct zome function', async () => {
    await client.getBondSnapshots(MOCK_HASH);
    expectZomeCall(mock, 'hearth_kinship', 'get_bond_snapshots', MOCK_HASH);
  });

  it('createWeeklyDigest calls correct zome function', async () => {
    const input = {
      hearth_hash: MOCK_HASH,
      epoch_start: 1000000,
      epoch_end: 2000000,
      bond_updates: [],
      care_summary: [],
      gratitude_summary: [],
      rhythm_summary: [],
      created_by: MOCK_AGENT,
      created_at: 1500000,
    };
    await client.createWeeklyDigest(input);
    expectZomeCall(mock, 'hearth_kinship', 'create_weekly_digest', input);
  });

  it('getWeeklyDigests calls correct zome function', async () => {
    await client.getWeeklyDigests(MOCK_HASH);
    expectZomeCall(mock, 'hearth_kinship', 'get_weekly_digests', MOCK_HASH);
  });
});

// ============================================================================
// DecisionsClient Tests (hearth_decisions) - 12 methods
// ============================================================================

describe('DecisionsClient', () => {
  let client: DecisionsClient;
  let mock: ReturnType<typeof createMockClient>;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    responses.set('hearth_decisions:create_decision', MOCK_RECORD);
    responses.set('hearth_decisions:get_decision', MOCK_RECORD);
    responses.set('hearth_decisions:get_hearth_decisions', [MOCK_RECORD]);
    responses.set('hearth_decisions:finalize_decision', MOCK_RECORD);
    responses.set('hearth_decisions:close_decision', MOCK_RECORD);
    responses.set('hearth_decisions:get_decision_outcome', MOCK_RECORD);
    responses.set('hearth_decisions:cast_vote', MOCK_RECORD);
    responses.set('hearth_decisions:amend_vote', MOCK_RECORD);
    responses.set('hearth_decisions:tally_votes', [[0, 3], [1, 2]]);
    responses.set('hearth_decisions:get_decision_votes', [MOCK_RECORD]);
    responses.set('hearth_decisions:get_vote_history', [MOCK_RECORD]);
    responses.set('hearth_decisions:get_my_pending_votes', [MOCK_RECORD]);

    mock = createMockClient(responses);
    client = new DecisionsClient(mock, BASE_CONFIG);
  });

  it('createDecision calls correct zome function', async () => {
    const input = {
      hearth_hash: MOCK_HASH,
      title: 'Vacation destination',
      description: 'Where should we go?',
      decision_type: 'MajorityVote' as const,
      eligible_roles: ['Adult' as const, 'Elder' as const],
      options: ['Beach', 'Mountains'],
      deadline: 2000000,
    };
    await client.createDecision(input);
    expectZomeCall(mock, 'hearth_decisions', 'create_decision', input);
  });

  it('getDecision calls correct zome function', async () => {
    await client.getDecision(MOCK_HASH);
    expectZomeCall(mock, 'hearth_decisions', 'get_decision', MOCK_HASH);
  });

  it('getHearthDecisions calls correct zome function', async () => {
    await client.getHearthDecisions(MOCK_HASH);
    expectZomeCall(mock, 'hearth_decisions', 'get_hearth_decisions', MOCK_HASH);
  });

  it('finalizeDecision calls correct zome function', async () => {
    const input = { decision_hash: MOCK_HASH };
    await client.finalizeDecision(input);
    expectZomeCall(mock, 'hearth_decisions', 'finalize_decision', input);
  });

  it('closeDecision calls correct zome function', async () => {
    const input = { decision_hash: MOCK_HASH, reason: 'No longer relevant' };
    await client.closeDecision(input);
    expectZomeCall(mock, 'hearth_decisions', 'close_decision', input);
  });

  it('getDecisionOutcome calls correct zome function', async () => {
    await client.getDecisionOutcome(MOCK_HASH);
    expectZomeCall(mock, 'hearth_decisions', 'get_decision_outcome', MOCK_HASH);
  });

  it('castVote calls correct zome function', async () => {
    const input = { decision_hash: MOCK_HASH, choice: 0, reasoning: 'I prefer the beach' };
    await client.castVote(input);
    expectZomeCall(mock, 'hearth_decisions', 'cast_vote', input);
  });

  it('amendVote calls correct zome function', async () => {
    const input = { original_vote_hash: MOCK_HASH, new_choice_index: 1, reason: 'Changed my mind' };
    await client.amendVote(input);
    expectZomeCall(mock, 'hearth_decisions', 'amend_vote', input);
  });

  it('tallyVotes calls correct zome function', async () => {
    const result = await client.tallyVotes(MOCK_HASH);
    expectZomeCall(mock, 'hearth_decisions', 'tally_votes', MOCK_HASH);
    expect(result).toEqual([[0, 3], [1, 2]]);
  });

  it('getDecisionVotes calls correct zome function', async () => {
    await client.getDecisionVotes(MOCK_HASH);
    expectZomeCall(mock, 'hearth_decisions', 'get_decision_votes', MOCK_HASH);
  });

  it('getVoteHistory calls correct zome function', async () => {
    await client.getVoteHistory(MOCK_HASH);
    expectZomeCall(mock, 'hearth_decisions', 'get_vote_history', MOCK_HASH);
  });

  it('getMyPendingVotes calls correct zome function', async () => {
    await client.getMyPendingVotes(MOCK_HASH);
    expectZomeCall(mock, 'hearth_decisions', 'get_my_pending_votes', MOCK_HASH);
  });
});

// ============================================================================
// GratitudeClient Tests (hearth_gratitude) - 8 methods
// ============================================================================

describe('GratitudeClient', () => {
  let client: GratitudeClient;
  let mock: ReturnType<typeof createMockClient>;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    responses.set('hearth_gratitude:express_gratitude', MOCK_RECORD);
    responses.set('hearth_gratitude:get_gratitude_stream', [MOCK_RECORD]);
    responses.set('hearth_gratitude:get_gratitude_balance', {
      agent: MOCK_AGENT,
      hearth_hash: MOCK_HASH,
      total_given: 12,
      total_received: 8,
      current_streak_days: 5,
    });
    responses.set('hearth_gratitude:start_appreciation_circle', MOCK_RECORD);
    responses.set('hearth_gratitude:join_circle', MOCK_RECORD);
    responses.set('hearth_gratitude:complete_circle', MOCK_RECORD);
    responses.set('hearth_gratitude:get_hearth_circles', [MOCK_RECORD]);
    responses.set('hearth_gratitude:create_gratitude_digest', MOCK_RECORD);

    mock = createMockClient(responses);
    client = new GratitudeClient(mock, BASE_CONFIG);
  });

  it('expressGratitude calls correct zome function', async () => {
    const input = {
      to_agent: MOCK_AGENT,
      message: 'Thank you for dinner!',
      gratitude_type: 'Appreciation' as const,
      visibility: 'AllMembers' as const,
      hearth_hash: MOCK_HASH,
    };
    await client.expressGratitude(input);
    expectZomeCall(mock, 'hearth_gratitude', 'express_gratitude', input);
  });

  it('getGratitudeStream calls correct zome function', async () => {
    await client.getGratitudeStream(MOCK_HASH);
    expectZomeCall(mock, 'hearth_gratitude', 'get_gratitude_stream', MOCK_HASH);
  });

  it('getGratitudeBalance calls correct zome function', async () => {
    const result = await client.getGratitudeBalance(MOCK_HASH);
    expectZomeCall(mock, 'hearth_gratitude', 'get_gratitude_balance', MOCK_HASH);
    expect(result.total_given).toBe(12);
    expect(result.current_streak_days).toBe(5);
  });

  it('startCircle calls correct zome function', async () => {
    const input = { hearth_hash: MOCK_HASH, theme: 'Gratitude for spring' };
    await client.startCircle(input);
    expectZomeCall(mock, 'hearth_gratitude', 'start_appreciation_circle', input);
  });

  it('joinCircle calls correct zome function', async () => {
    await client.joinCircle(MOCK_HASH);
    expectZomeCall(mock, 'hearth_gratitude', 'join_circle', MOCK_HASH);
  });

  it('completeCircle calls correct zome function', async () => {
    await client.completeCircle(MOCK_HASH);
    expectZomeCall(mock, 'hearth_gratitude', 'complete_circle', MOCK_HASH);
  });

  it('getHearthCircles calls correct zome function', async () => {
    await client.getHearthCircles(MOCK_HASH);
    expectZomeCall(mock, 'hearth_gratitude', 'get_hearth_circles', MOCK_HASH);
  });

  it('createGratitudeDigest calls correct zome function', async () => {
    const input = { hearth_hash: MOCK_HASH, epoch_start: 1000000, epoch_end: 2000000 };
    await client.createGratitudeDigest(input);
    expectZomeCall(mock, 'hearth_gratitude', 'create_gratitude_digest', input);
  });
});

// ============================================================================
// StoriesClient Tests (hearth_stories) - 11 methods
// ============================================================================

describe('StoriesClient', () => {
  let client: StoriesClient;
  let mock: ReturnType<typeof createMockClient>;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    responses.set('hearth_stories:create_story', MOCK_RECORD);
    responses.set('hearth_stories:update_story', MOCK_RECORD);
    responses.set('hearth_stories:add_media_to_story', MOCK_RECORD);
    responses.set('hearth_stories:get_hearth_stories', [MOCK_RECORD]);
    responses.set('hearth_stories:search_stories_by_tag', [MOCK_RECORD]);
    responses.set('hearth_stories:create_collection', MOCK_RECORD);
    responses.set('hearth_stories:add_to_collection', MOCK_RECORD);
    responses.set('hearth_stories:get_hearth_collections', [MOCK_RECORD]);
    responses.set('hearth_stories:create_tradition', MOCK_RECORD);
    responses.set('hearth_stories:observe_tradition', MOCK_RECORD);
    responses.set('hearth_stories:get_hearth_traditions', [MOCK_RECORD]);

    mock = createMockClient(responses);
    client = new StoriesClient(mock, BASE_CONFIG);
  });

  it('createStory calls correct zome function', async () => {
    const input = {
      title: 'How we met',
      content: 'It was a sunny day...',
      story_type: 'Memory' as const,
      visibility: 'AllMembers' as const,
      hearth_hash: MOCK_HASH,
    };
    await client.createStory(input);
    expectZomeCall(mock, 'hearth_stories', 'create_story', input);
  });

  it('updateStory calls correct zome function', async () => {
    const input = { story_hash: MOCK_HASH, title: 'Updated title', content: 'New content', tags: ['family'] };
    await client.updateStory(input);
    expectZomeCall(mock, 'hearth_stories', 'update_story', input);
  });

  it('addMediaToStory calls correct zome function', async () => {
    const input = { story_hash: MOCK_HASH, media_hash: MOCK_HASH };
    await client.addMediaToStory(input);
    expectZomeCall(mock, 'hearth_stories', 'add_media_to_story', input);
  });

  it('getHearthStories calls correct zome function', async () => {
    await client.getHearthStories(MOCK_HASH);
    expectZomeCall(mock, 'hearth_stories', 'get_hearth_stories', MOCK_HASH);
  });

  it('searchStoriesByTag calls correct zome function', async () => {
    await client.searchStoriesByTag(MOCK_HASH, 'holiday');
    expectZomeCall(mock, 'hearth_stories', 'search_stories_by_tag', { hearth_hash: MOCK_HASH, tag: 'holiday' });
  });

  it('createCollection calls correct zome function', async () => {
    const input = { name: 'Summer Memories', description: 'Stories from summer', hearth_hash: MOCK_HASH };
    await client.createCollection(input);
    expectZomeCall(mock, 'hearth_stories', 'create_collection', input);
  });

  it('addToCollection calls correct zome function', async () => {
    const input = { collection_hash: MOCK_HASH, story_hash: MOCK_HASH };
    await client.addToCollection(input);
    expectZomeCall(mock, 'hearth_stories', 'add_to_collection', input);
  });

  it('getHearthCollections calls correct zome function', async () => {
    await client.getHearthCollections(MOCK_HASH);
    expectZomeCall(mock, 'hearth_stories', 'get_hearth_collections', MOCK_HASH);
  });

  it('createTradition calls correct zome function', async () => {
    const input = {
      name: 'Sunday Dinner',
      description: 'Weekly family dinner',
      frequency: 'Weekly' as const,
      instructions: 'Everyone gathers at 6pm',
      hearth_hash: MOCK_HASH,
    };
    await client.createTradition(input);
    expectZomeCall(mock, 'hearth_stories', 'create_tradition', input);
  });

  it('observeTradition calls correct zome function', async () => {
    await client.observeTradition(MOCK_HASH);
    expectZomeCall(mock, 'hearth_stories', 'observe_tradition', MOCK_HASH);
  });

  it('getHearthTraditions calls correct zome function', async () => {
    await client.getHearthTraditions(MOCK_HASH);
    expectZomeCall(mock, 'hearth_stories', 'get_hearth_traditions', MOCK_HASH);
  });
});

// ============================================================================
// HearthCareClient Tests (hearth_care) - 10 methods
// ============================================================================

describe('HearthCareClient', () => {
  let client: HearthCareClient;
  let mock: ReturnType<typeof createMockClient>;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    responses.set('hearth_care:create_care_schedule', MOCK_RECORD);
    responses.set('hearth_care:complete_task', MOCK_RECORD);
    responses.set('hearth_care:get_my_care_duties', [MOCK_RECORD]);
    responses.set('hearth_care:get_hearth_schedule', [MOCK_RECORD]);
    responses.set('hearth_care:propose_swap', MOCK_RECORD);
    responses.set('hearth_care:accept_swap', MOCK_RECORD);
    responses.set('hearth_care:decline_swap', MOCK_RECORD);
    responses.set('hearth_care:create_meal_plan', MOCK_RECORD);
    responses.set('hearth_care:get_hearth_meal_plans', [MOCK_RECORD]);
    responses.set('hearth_care:create_care_digest', MOCK_RECORD);

    mock = createMockClient(responses);
    client = new HearthCareClient(mock, BASE_CONFIG);
  });

  it('createCareSchedule calls correct zome function', async () => {
    const input = {
      hearth_hash: MOCK_HASH,
      care_type: 'Childcare' as const,
      title: 'Morning school run',
      description: 'Drop kids at school',
      assigned_to: MOCK_AGENT,
      recurrence: 'Daily' as const,
    };
    await client.createCareSchedule(input);
    expectZomeCall(mock, 'hearth_care', 'create_care_schedule', input);
  });

  it('completeTask calls correct zome function', async () => {
    await client.completeTask(MOCK_HASH);
    expectZomeCall(mock, 'hearth_care', 'complete_task', MOCK_HASH);
  });

  it('getMyCareDuties calls correct zome function with null payload', async () => {
    await client.getMyCareDuties();
    expectZomeCall(mock, 'hearth_care', 'get_my_care_duties', null);
  });

  it('getHearthSchedule calls correct zome function', async () => {
    await client.getHearthSchedule(MOCK_HASH);
    expectZomeCall(mock, 'hearth_care', 'get_hearth_schedule', MOCK_HASH);
  });

  it('proposeSwap calls correct zome function', async () => {
    const input = { original_schedule_hash: MOCK_HASH, swap_date: 1500000 };
    await client.proposeSwap(input);
    expectZomeCall(mock, 'hearth_care', 'propose_swap', input);
  });

  it('acceptSwap calls correct zome function', async () => {
    await client.acceptSwap(MOCK_HASH);
    expectZomeCall(mock, 'hearth_care', 'accept_swap', MOCK_HASH);
  });

  it('declineSwap calls correct zome function', async () => {
    await client.declineSwap(MOCK_HASH);
    expectZomeCall(mock, 'hearth_care', 'decline_swap', MOCK_HASH);
  });

  it('createMealPlan calls correct zome function', async () => {
    const input = {
      hearth_hash: MOCK_HASH,
      week_start: 1000000,
      meals: [{ day: 'Monday', meal_type: 'Dinner', recipe_name: 'Pasta', servings: 4 }],
      shopper: MOCK_AGENT,
      cook: MOCK_AGENT,
    };
    await client.createMealPlan(input);
    expectZomeCall(mock, 'hearth_care', 'create_meal_plan', input);
  });

  it('getHearthMealPlans calls correct zome function', async () => {
    await client.getHearthMealPlans(MOCK_HASH);
    expectZomeCall(mock, 'hearth_care', 'get_hearth_meal_plans', MOCK_HASH);
  });

  it('createCareDigest calls correct zome function', async () => {
    const input = { hearth_hash: MOCK_HASH, epoch_start: 1000000, epoch_end: 2000000 };
    await client.createCareDigest(input);
    expectZomeCall(mock, 'hearth_care', 'create_care_digest', input);
  });
});

// ============================================================================
// AutonomyClient Tests (hearth_autonomy) - 10 methods
// ============================================================================

describe('AutonomyClient', () => {
  let client: AutonomyClient;
  let mock: ReturnType<typeof createMockClient>;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    responses.set('hearth_autonomy:create_autonomy_profile', MOCK_RECORD);
    responses.set('hearth_autonomy:get_autonomy_profile', MOCK_RECORD);
    responses.set('hearth_autonomy:check_capability', true);
    responses.set('hearth_autonomy:advance_tier', MOCK_RECORD);
    responses.set('hearth_autonomy:request_capability', MOCK_RECORD);
    responses.set('hearth_autonomy:approve_capability', MOCK_RECORD);
    responses.set('hearth_autonomy:deny_capability', MOCK_RECORD);
    responses.set('hearth_autonomy:get_pending_requests', [MOCK_RECORD]);
    responses.set('hearth_autonomy:progress_transition', MOCK_RECORD);
    responses.set('hearth_autonomy:get_active_transitions', [MOCK_RECORD]);

    mock = createMockClient(responses);
    client = new AutonomyClient(mock, BASE_CONFIG);
  });

  it('createProfile calls correct zome function', async () => {
    const input = {
      member_hash: MOCK_AGENT,
      guardian_hashes: [MOCK_AGENT],
      initial_tier: 'Supervised' as const,
      hearth_hash: MOCK_HASH,
    };
    await client.createProfile(input);
    expectZomeCall(mock, 'hearth_autonomy', 'create_autonomy_profile', input);
  });

  it('getAutonomyProfile calls correct zome function', async () => {
    await client.getAutonomyProfile(MOCK_AGENT);
    expectZomeCall(mock, 'hearth_autonomy', 'get_autonomy_profile', MOCK_AGENT);
  });

  it('checkCapability calls correct zome function', async () => {
    const input = { member: MOCK_AGENT, capability: 'use_stove' };
    const result = await client.checkCapability(input);
    expectZomeCall(mock, 'hearth_autonomy', 'check_capability', input);
    expect(result).toBe(true);
  });

  it('advanceTier calls correct zome function', async () => {
    const input = { profile_hash: MOCK_HASH, new_tier: 'SemiAutonomous' as const };
    await client.advanceTier(input);
    expectZomeCall(mock, 'hearth_autonomy', 'advance_tier', input);
  });

  it('requestCapability calls correct zome function', async () => {
    const input = { capability: 'drive_car', justification: 'Passed driving test', hearth_hash: MOCK_HASH };
    await client.requestCapability(input);
    expectZomeCall(mock, 'hearth_autonomy', 'request_capability', input);
  });

  it('approveCapability calls correct zome function', async () => {
    const input = { request_hash: MOCK_HASH, approved: true, conditions: 'Only during daytime' };
    await client.approveCapability(input);
    expectZomeCall(mock, 'hearth_autonomy', 'approve_capability', input);
  });

  it('denyCapability calls correct zome function', async () => {
    const input = { request_hash: MOCK_HASH, approved: false };
    await client.denyCapability(input);
    expectZomeCall(mock, 'hearth_autonomy', 'deny_capability', input);
  });

  it('getPendingRequests calls correct zome function', async () => {
    await client.getPendingRequests(MOCK_HASH);
    expectZomeCall(mock, 'hearth_autonomy', 'get_pending_requests', MOCK_HASH);
  });

  it('progressTransition calls correct zome function', async () => {
    await client.progressTransition(MOCK_HASH);
    expectZomeCall(mock, 'hearth_autonomy', 'progress_transition', MOCK_HASH);
  });

  it('getActiveTransitions calls correct zome function', async () => {
    await client.getActiveTransitions(MOCK_HASH);
    expectZomeCall(mock, 'hearth_autonomy', 'get_active_transitions', MOCK_HASH);
  });
});

// ============================================================================
// EmergencyClient Tests (hearth_emergency) - 8 methods
// ============================================================================

describe('EmergencyClient', () => {
  let client: EmergencyClient;
  let mock: ReturnType<typeof createMockClient>;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    responses.set('hearth_emergency:create_emergency_plan', MOCK_RECORD);
    responses.set('hearth_emergency:update_emergency_plan', MOCK_RECORD);
    responses.set('hearth_emergency:get_emergency_plan', MOCK_RECORD);
    responses.set('hearth_emergency:raise_alert', MOCK_RECORD);
    responses.set('hearth_emergency:resolve_alert', MOCK_RECORD);
    responses.set('hearth_emergency:get_active_alerts', [MOCK_RECORD]);
    responses.set('hearth_emergency:check_in', MOCK_RECORD);
    responses.set('hearth_emergency:get_alert_checkins', [MOCK_RECORD]);

    mock = createMockClient(responses);
    client = new EmergencyClient(mock, BASE_CONFIG);
  });

  it('createEmergencyPlan calls correct zome function', async () => {
    const input = {
      hearth_hash: MOCK_HASH,
      contacts: [{ name: 'Dr. Smith', phone: '555-0100', relationship: 'Doctor', priority_order: 1 }],
      meeting_points: ['Front yard', 'Neighbor house'],
    };
    await client.createEmergencyPlan(input);
    expectZomeCall(mock, 'hearth_emergency', 'create_emergency_plan', input);
  });

  it('updateEmergencyPlan calls correct zome function', async () => {
    const input = {
      plan_hash: MOCK_HASH,
      input: {
        hearth_hash: MOCK_HASH,
        contacts: [{ name: 'Dr. Jones', phone: '555-0200', relationship: 'Doctor', priority_order: 1 }],
        meeting_points: ['Back yard'],
      },
    };
    await client.updateEmergencyPlan(input);
    expectZomeCall(mock, 'hearth_emergency', 'update_emergency_plan', input);
  });

  it('getEmergencyPlan calls correct zome function', async () => {
    await client.getEmergencyPlan(MOCK_HASH);
    expectZomeCall(mock, 'hearth_emergency', 'get_emergency_plan', MOCK_HASH);
  });

  it('raiseAlert calls correct zome function', async () => {
    const input = {
      hearth_hash: MOCK_HASH,
      alert_type: 'Medical' as const,
      severity: 'High' as const,
      message: 'Need immediate help',
    };
    await client.raiseAlert(input);
    expectZomeCall(mock, 'hearth_emergency', 'raise_alert', input);
  });

  it('resolveAlert calls correct zome function', async () => {
    await client.resolveAlert(MOCK_HASH);
    expectZomeCall(mock, 'hearth_emergency', 'resolve_alert', MOCK_HASH);
  });

  it('getActiveAlerts calls correct zome function', async () => {
    await client.getActiveAlerts(MOCK_HASH);
    expectZomeCall(mock, 'hearth_emergency', 'get_active_alerts', MOCK_HASH);
  });

  it('checkIn calls correct zome function', async () => {
    const input = { hearth_hash: MOCK_HASH, status: 'Safe' as const, location_hint: 'At home' };
    await client.checkIn(input);
    expectZomeCall(mock, 'hearth_emergency', 'check_in', input);
  });

  it('getAlertCheckins calls correct zome function', async () => {
    await client.getAlertCheckins(MOCK_HASH);
    expectZomeCall(mock, 'hearth_emergency', 'get_alert_checkins', MOCK_HASH);
  });
});

// ============================================================================
// ResourcesClient Tests (hearth_resources) - 8 methods
// ============================================================================

describe('ResourcesClient', () => {
  let client: ResourcesClient;
  let mock: ReturnType<typeof createMockClient>;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    responses.set('hearth_resources:register_resource', MOCK_RECORD);
    responses.set('hearth_resources:get_hearth_inventory', [MOCK_RECORD]);
    responses.set('hearth_resources:lend_resource', MOCK_RECORD);
    responses.set('hearth_resources:return_resource', MOCK_RECORD);
    responses.set('hearth_resources:get_resource_loans', [MOCK_RECORD]);
    responses.set('hearth_resources:create_budget_category', MOCK_RECORD);
    responses.set('hearth_resources:log_expense', MOCK_RECORD);
    responses.set('hearth_resources:get_budget_summary', [MOCK_RECORD]);

    mock = createMockClient(responses);
    client = new ResourcesClient(mock, BASE_CONFIG);
  });

  it('registerResource calls correct zome function', async () => {
    const input = {
      name: 'Family Van',
      description: '2020 Toyota Sienna',
      resource_type: 'Vehicle' as const,
      condition: 'Good',
      location: 'Garage',
      hearth_hash: MOCK_HASH,
    };
    await client.registerResource(input);
    expectZomeCall(mock, 'hearth_resources', 'register_resource', input);
  });

  it('getHearthInventory calls correct zome function', async () => {
    await client.getHearthInventory(MOCK_HASH);
    expectZomeCall(mock, 'hearth_resources', 'get_hearth_inventory', MOCK_HASH);
  });

  it('lendResource calls correct zome function', async () => {
    const input = { resource_hash: MOCK_HASH, borrower: MOCK_AGENT, due_date: 2000000 };
    await client.lendResource(input);
    expectZomeCall(mock, 'hearth_resources', 'lend_resource', input);
  });

  it('returnResource calls correct zome function', async () => {
    await client.returnResource(MOCK_HASH);
    expectZomeCall(mock, 'hearth_resources', 'return_resource', MOCK_HASH);
  });

  it('getResourceLoans calls correct zome function', async () => {
    await client.getResourceLoans(MOCK_HASH);
    expectZomeCall(mock, 'hearth_resources', 'get_resource_loans', MOCK_HASH);
  });

  it('createBudgetCategory calls correct zome function', async () => {
    const input = { hearth_hash: MOCK_HASH, category: 'Groceries', monthly_target_cents: 60000 };
    await client.createBudgetCategory(input);
    expectZomeCall(mock, 'hearth_resources', 'create_budget_category', input);
  });

  it('logExpense calls correct zome function', async () => {
    const input = { budget_hash: MOCK_HASH, amount_cents: 4550, description: 'Weekly grocery run' };
    await client.logExpense(input);
    expectZomeCall(mock, 'hearth_resources', 'log_expense', input);
  });

  it('getBudgetSummary calls correct zome function', async () => {
    await client.getBudgetSummary(MOCK_HASH);
    expectZomeCall(mock, 'hearth_resources', 'get_budget_summary', MOCK_HASH);
  });
});

// ============================================================================
// MilestonesClient Tests (hearth_milestones) - 7 methods
// ============================================================================

describe('MilestonesClient', () => {
  let client: MilestonesClient;
  let mock: ReturnType<typeof createMockClient>;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    responses.set('hearth_milestones:record_milestone', MOCK_RECORD);
    responses.set('hearth_milestones:get_family_timeline', [MOCK_RECORD]);
    responses.set('hearth_milestones:get_member_milestones', [MOCK_RECORD]);
    responses.set('hearth_milestones:begin_transition', MOCK_RECORD);
    responses.set('hearth_milestones:advance_transition', MOCK_RECORD);
    responses.set('hearth_milestones:complete_transition', MOCK_RECORD);
    responses.set('hearth_milestones:get_active_transitions', [MOCK_RECORD]);

    mock = createMockClient(responses);
    client = new MilestonesClient(mock, BASE_CONFIG);
  });

  it('recordMilestone calls correct zome function', async () => {
    const input = {
      hearth_hash: MOCK_HASH,
      member_hash: MOCK_AGENT,
      milestone_type: 'Birthday' as const,
      date: 1000000,
      description: 'Turned 10!',
    };
    await client.recordMilestone(input);
    expectZomeCall(mock, 'hearth_milestones', 'record_milestone', input);
  });

  it('getFamilyTimeline calls correct zome function', async () => {
    await client.getFamilyTimeline(MOCK_HASH);
    expectZomeCall(mock, 'hearth_milestones', 'get_family_timeline', MOCK_HASH);
  });

  it('getMemberMilestones calls correct zome function', async () => {
    await client.getMemberMilestones(MOCK_HASH);
    expectZomeCall(mock, 'hearth_milestones', 'get_member_milestones', MOCK_HASH);
  });

  it('beginTransition calls correct zome function', async () => {
    const input = {
      hearth_hash: MOCK_HASH,
      member_hash: MOCK_AGENT,
      transition_type: 'ComingOfAge' as const,
    };
    await client.beginTransition(input);
    expectZomeCall(mock, 'hearth_milestones', 'begin_transition', input);
  });

  it('advanceTransition calls correct zome function', async () => {
    const input = { transition_hash: MOCK_HASH };
    await client.advanceTransition(input);
    expectZomeCall(mock, 'hearth_milestones', 'advance_transition', input);
  });

  it('completeTransition calls correct zome function', async () => {
    await client.completeTransition(MOCK_HASH);
    expectZomeCall(mock, 'hearth_milestones', 'complete_transition', MOCK_HASH);
  });

  it('getActiveTransitions calls correct zome function', async () => {
    await client.getActiveTransitions(MOCK_HASH);
    expectZomeCall(mock, 'hearth_milestones', 'get_active_transitions', MOCK_HASH);
  });
});

// ============================================================================
// RhythmsClient Tests (hearth_rhythms) - 7 methods
// ============================================================================

describe('RhythmsClient', () => {
  let client: RhythmsClient;
  let mock: ReturnType<typeof createMockClient>;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    responses.set('hearth_rhythms:create_rhythm', MOCK_RECORD);
    responses.set('hearth_rhythms:get_hearth_rhythms', [MOCK_RECORD]);
    responses.set('hearth_rhythms:log_occurrence', MOCK_RECORD);
    responses.set('hearth_rhythms:get_rhythm_occurrences', [MOCK_RECORD]);
    responses.set('hearth_rhythms:set_presence', MOCK_RECORD);
    responses.set('hearth_rhythms:get_hearth_presence', [MOCK_RECORD]);
    responses.set('hearth_rhythms:create_rhythm_digest', MOCK_RECORD);

    mock = createMockClient(responses);
    client = new RhythmsClient(mock, BASE_CONFIG);
  });

  it('createRhythm calls correct zome function', async () => {
    const input = {
      hearth_hash: MOCK_HASH,
      name: 'Morning Gathering',
      rhythm_type: 'Morning' as const,
      schedule: '07:00 daily',
      participants: [MOCK_AGENT],
      description: 'Breakfast together',
    };
    await client.createRhythm(input);
    expectZomeCall(mock, 'hearth_rhythms', 'create_rhythm', input);
  });

  it('getHearthRhythms calls correct zome function', async () => {
    await client.getHearthRhythms(MOCK_HASH);
    expectZomeCall(mock, 'hearth_rhythms', 'get_hearth_rhythms', MOCK_HASH);
  });

  it('logOccurrence calls correct zome function', async () => {
    const input = {
      rhythm_hash: MOCK_HASH,
      participants_present: [MOCK_AGENT],
      notes: 'Great morning!',
      mood: 9000,
    };
    await client.logOccurrence(input);
    expectZomeCall(mock, 'hearth_rhythms', 'log_occurrence', input);
  });

  it('getRhythmOccurrences calls correct zome function', async () => {
    await client.getRhythmOccurrences(MOCK_HASH);
    expectZomeCall(mock, 'hearth_rhythms', 'get_rhythm_occurrences', MOCK_HASH);
  });

  it('setPresence calls correct zome function', async () => {
    const input = { status: 'Home' as const, hearth_hash: MOCK_HASH };
    await client.setPresence(input);
    expectZomeCall(mock, 'hearth_rhythms', 'set_presence', input);
  });

  it('getHearthPresence calls correct zome function', async () => {
    await client.getHearthPresence(MOCK_HASH);
    expectZomeCall(mock, 'hearth_rhythms', 'get_hearth_presence', MOCK_HASH);
  });

  it('createRhythmDigest calls correct zome function', async () => {
    const input = { hearth_hash: MOCK_HASH, epoch_start: 1000000, epoch_end: 2000000 };
    await client.createRhythmDigest(input);
    expectZomeCall(mock, 'hearth_rhythms', 'create_rhythm_digest', input);
  });
});

// ============================================================================
// BridgeClient Tests (hearth_bridge) - 22 methods
// ============================================================================

describe('BridgeClient', () => {
  let client: BridgeClient;
  let mock: ReturnType<typeof createMockClient>;

  const MOCK_DISPATCH_RESULT = { success: true, payload: new Uint8Array(8) };
  const MOCK_HEALTH: { status: string; connected_zomes: number } = { status: 'healthy', connected_zomes: 10 };

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    responses.set('hearth_bridge:dispatch_call', MOCK_DISPATCH_RESULT);
    responses.set('hearth_bridge:query_hearth', MOCK_RECORD);
    responses.set('hearth_bridge:resolve_query', MOCK_RECORD);
    responses.set('hearth_bridge:broadcast_event', MOCK_RECORD);
    responses.set('hearth_bridge:dispatch_personal_call', MOCK_DISPATCH_RESULT);
    responses.set('hearth_bridge:dispatch_identity_call', MOCK_DISPATCH_RESULT);
    responses.set('hearth_bridge:dispatch_commons_call', MOCK_DISPATCH_RESULT);
    responses.set('hearth_bridge:dispatch_civic_call', MOCK_DISPATCH_RESULT);
    responses.set('hearth_bridge:verify_member_identity', MOCK_DISPATCH_RESULT);
    responses.set('hearth_bridge:escalate_emergency', MOCK_DISPATCH_RESULT);
    responses.set('hearth_bridge:query_timebank_balance', MOCK_DISPATCH_RESULT);
    responses.set('hearth_bridge:get_consciousness_credential', {
      did: 'did:mycelix:test',
      tier: 'Aware',
      phi_score: 0.75,
      issued_at: 1000000,
    });
    responses.set('hearth_bridge:log_governance_gate', undefined);
    responses.set('hearth_bridge:get_governance_audit_trail', { entries: [], total_matched: 0 });
    responses.set('hearth_bridge:get_domain_events', [MOCK_RECORD]);
    responses.set('hearth_bridge:get_all_events', [MOCK_RECORD]);
    responses.set('hearth_bridge:get_events_by_type', [MOCK_RECORD]);
    responses.set('hearth_bridge:get_my_queries', [MOCK_RECORD]);
    responses.set('hearth_bridge:initiate_severance', MOCK_RECORD);
    responses.set('hearth_bridge:hearth_sync', MOCK_RECORD);
    responses.set('hearth_bridge:get_weekly_digests', [MOCK_RECORD]);
    responses.set('hearth_bridge:health_check', MOCK_HEALTH);

    mock = createMockClient(responses);
    client = new BridgeClient(mock, BASE_CONFIG);
  });

  it('dispatchCall calls correct zome function', async () => {
    const input = { target_zome: 'hearth_kinship', fn_name: 'get_my_hearths', payload: new Uint8Array(0) };
    await client.dispatchCall(input);
    expectZomeCall(mock, 'hearth_bridge', 'dispatch_call', input);
  });

  it('queryHearth calls correct zome function', async () => {
    const input = { target_zome: 'hearth_care', fn_name: 'get_my_care_duties', payload: new Uint8Array(0) };
    await client.queryHearth(input);
    expectZomeCall(mock, 'hearth_bridge', 'query_hearth', input);
  });

  it('resolveQuery calls correct zome function', async () => {
    const input = { domain: 'kinship', query_type: 'members', params: new Uint8Array(4) };
    await client.resolveQuery(input);
    expectZomeCall(mock, 'hearth_bridge', 'resolve_query', input);
  });

  it('broadcastEvent calls correct zome function', async () => {
    const input = { target_zome: 'hearth_kinship', fn_name: 'member_joined', payload: new Uint8Array(4) };
    await client.broadcastEvent(input);
    expectZomeCall(mock, 'hearth_bridge', 'broadcast_event', input);
  });

  it('dispatchPersonalCall calls correct zome function', async () => {
    const input = { target_role: 'personal', target_zome: 'profile', fn_name: 'get_profile', payload: new Uint8Array(0) };
    await client.dispatchPersonalCall(input);
    expectZomeCall(mock, 'hearth_bridge', 'dispatch_personal_call', input);
  });

  it('dispatchIdentityCall calls correct zome function', async () => {
    const input = { target_role: 'identity', target_zome: 'identity_core', fn_name: 'get_did', payload: new Uint8Array(0) };
    await client.dispatchIdentityCall(input);
    expectZomeCall(mock, 'hearth_bridge', 'dispatch_identity_call', input);
  });

  it('dispatchCommonsCall calls correct zome function', async () => {
    const input = { target_role: 'commons', target_zome: 'commons_bridge', fn_name: 'query', payload: new Uint8Array(0) };
    await client.dispatchCommonsCall(input);
    expectZomeCall(mock, 'hearth_bridge', 'dispatch_commons_call', input);
  });

  it('dispatchCivicCall calls correct zome function', async () => {
    const input = { target_role: 'civic', target_zome: 'civic_bridge', fn_name: 'query', payload: new Uint8Array(0) };
    await client.dispatchCivicCall(input);
    expectZomeCall(mock, 'hearth_bridge', 'dispatch_civic_call', input);
  });

  it('verifyMemberIdentity calls correct zome function', async () => {
    await client.verifyMemberIdentity(MOCK_AGENT);
    expectZomeCall(mock, 'hearth_bridge', 'verify_member_identity', MOCK_AGENT);
  });

  it('escalateEmergency calls correct zome function', async () => {
    const alertData = '{"type":"Medical","severity":"Critical"}';
    await client.escalateEmergency(alertData);
    expectZomeCall(mock, 'hearth_bridge', 'escalate_emergency', alertData);
  });

  it('queryTimebankBalance calls correct zome function', async () => {
    await client.queryTimebankBalance(MOCK_AGENT);
    expectZomeCall(mock, 'hearth_bridge', 'query_timebank_balance', MOCK_AGENT);
  });

  it('getConsciousnessCredential calls correct zome function', async () => {
    const result = await client.getConsciousnessCredential('did:mycelix:test');
    expectZomeCall(mock, 'hearth_bridge', 'get_consciousness_credential', 'did:mycelix:test');
    expect(result.tier).toBe('Aware');
    expect(result.phi_score).toBe(0.75);
  });

  it('logGovernanceGate calls correct zome function', async () => {
    const input = {
      action_name: 'create_decision',
      zome_name: 'hearth_decisions',
      eligible: true,
      actual_tier: 'Aware',
      required_tier: 'Awakening',
      weight_bp: 8000,
    };
    await client.logGovernanceGate(input);
    expectZomeCall(mock, 'hearth_bridge', 'log_governance_gate', input);
  });

  it('getGovernanceAuditTrail calls correct zome function', async () => {
    const filter = { zome_name: 'hearth_decisions', eligible: true };
    const result = await client.getGovernanceAuditTrail(filter);
    expectZomeCall(mock, 'hearth_bridge', 'get_governance_audit_trail', filter);
    expect(result.total_matched).toBe(0);
  });

  it('getDomainEvents calls correct zome function', async () => {
    await client.getDomainEvents('kinship');
    expectZomeCall(mock, 'hearth_bridge', 'get_domain_events', 'kinship');
  });

  it('getAllEvents calls correct zome function with null payload', async () => {
    await client.getAllEvents();
    expectZomeCall(mock, 'hearth_bridge', 'get_all_events', null);
  });

  it('getEventsByType calls correct zome function', async () => {
    const query = { event_type: 'MemberJoined', limit: 10 };
    await client.getEventsByType(query);
    expectZomeCall(mock, 'hearth_bridge', 'get_events_by_type', query);
  });

  it('getMyQueries calls correct zome function with null payload', async () => {
    await client.getMyQueries();
    expectZomeCall(mock, 'hearth_bridge', 'get_my_queries', null);
  });

  it('initiateSeverance calls correct zome function', async () => {
    const input = { hearth_hash: MOCK_HASH, reason: 'Household dissolution' };
    await client.initiateSeverance(input);
    expectZomeCall(mock, 'hearth_bridge', 'initiate_severance', input);
  });

  it('hearthSync calls correct zome function', async () => {
    const input = { hearth_hash: MOCK_HASH, epoch_start: 1000000, epoch_end: 2000000 };
    await client.hearthSync(input);
    expectZomeCall(mock, 'hearth_bridge', 'hearth_sync', input);
  });

  it('getWeeklyDigests calls correct zome function', async () => {
    await client.getWeeklyDigests(MOCK_HASH);
    expectZomeCall(mock, 'hearth_bridge', 'get_weekly_digests', MOCK_HASH);
  });

  it('healthCheck calls correct zome function with null payload', async () => {
    const result = await client.healthCheck();
    expectZomeCall(mock, 'hearth_bridge', 'health_check', null);
    expect(result.status).toBe('healthy');
    expect(result.connected_zomes).toBe(10);
  });
});

// ============================================================================
// HearthClient Composition Tests
// ============================================================================

describe('HearthClient', () => {
  it('fromClient composes all 11 sub-clients', () => {
    const mock = createMockClient(new Map());
    // HearthClient.fromClient expects an AppClient; our mock satisfies the
    // ZomeCallable interface used by sub-clients, which is sufficient here.
    const hearth = HearthClient.fromClient(mock as any);

    expect(hearth.kinship).toBeInstanceOf(KinshipClient);
    expect(hearth.decisions).toBeInstanceOf(DecisionsClient);
    expect(hearth.gratitude).toBeInstanceOf(GratitudeClient);
    expect(hearth.stories).toBeInstanceOf(StoriesClient);
    expect(hearth.care).toBeInstanceOf(HearthCareClient);
    expect(hearth.autonomy).toBeInstanceOf(AutonomyClient);
    expect(hearth.emergency).toBeInstanceOf(EmergencyClient);
    expect(hearth.resources).toBeInstanceOf(ResourcesClient);
    expect(hearth.milestones).toBeInstanceOf(MilestonesClient);
    expect(hearth.rhythms).toBeInstanceOf(RhythmsClient);
    expect(hearth.bridge).toBeInstanceOf(BridgeClient);
  });

  it('createHearthClient factory returns HearthClient', () => {
    const mock = createMockClient(new Map());
    const hearth = createHearthClient(mock as any);

    expect(hearth).toBeInstanceOf(HearthClient);
    expect(hearth.kinship).toBeInstanceOf(KinshipClient);
    expect(hearth.bridge).toBeInstanceOf(BridgeClient);
  });

  it('respects custom config', () => {
    const mock = createMockClient(new Map());
    const hearth = createHearthClient(mock as any, {
      roleName: 'custom_role',
      timeout: 60000,
    });

    // The client is constructed; we verify the instance was created
    // (config is private, but sub-clients will use the provided roleName).
    expect(hearth).toBeInstanceOf(HearthClient);
  });
});

// ============================================================================
// Signal Handling Tests
// ============================================================================

describe('Signal Handling', () => {
  it('getSignalType extracts variant name from HearthSignal', () => {
    const signal: HearthSignal = {
      EmergencyAlert: {
        alert_hash: MOCK_HASH,
        severity: 'Critical',
        message: 'Fire alarm!',
      },
    };
    expect(getSignalType(signal)).toBe('EmergencyAlert');
  });

  it('getSignalType works for all signal variants', () => {
    const variants: Array<[HearthSignal, string]> = [
      [{ GratitudeExpressed: { from_agent: MOCK_AGENT, to_agent: MOCK_AGENT, message: 'Thanks', gratitude_type: 'Appreciation' } }, 'GratitudeExpressed'],
      [{ CareTaskCompleted: { assignee: MOCK_AGENT, schedule_hash: MOCK_HASH, care_type: 'Chore' } }, 'CareTaskCompleted'],
      [{ SwapAccepted: { swap_hash: MOCK_HASH, hearth_hash: MOCK_HASH } }, 'SwapAccepted'],
      [{ SwapDeclined: { swap_hash: MOCK_HASH, hearth_hash: MOCK_HASH } }, 'SwapDeclined'],
      [{ RhythmOccurred: { rhythm_hash: MOCK_HASH, participants: [MOCK_AGENT] } }, 'RhythmOccurred'],
      [{ PresenceChanged: { agent: MOCK_AGENT, status: 'Home' } }, 'PresenceChanged'],
      [{ MemberJoined: { hearth_hash: MOCK_HASH, agent: MOCK_AGENT, role: 'Adult' } }, 'MemberJoined'],
      [{ MemberDeparted: { hearth_hash: MOCK_HASH, agent: MOCK_AGENT } }, 'MemberDeparted'],
      [{ BondTended: { member_a: MOCK_AGENT, member_b: MOCK_AGENT, quality_bp: 8000 } }, 'BondTended'],
      [{ CrossZomeCallFailed: { zome: 'hearth_care', function: 'get_schedule', error: 'timeout' } }, 'CrossZomeCallFailed'],
      [{ VoteCast: { decision_hash: MOCK_HASH, voter: MOCK_AGENT, choice: 0 } }, 'VoteCast'],
      [{ VoteAmended: { decision_hash: MOCK_HASH, voter: MOCK_AGENT, old_choice: 0, new_choice: 1 } }, 'VoteAmended'],
      [{ DecisionClosed: { decision_hash: MOCK_HASH, closed_by: MOCK_AGENT } }, 'DecisionClosed'],
      [{ DecisionFinalized: { decision_hash: MOCK_HASH, chosen_option: 1, participation_rate_bp: 7500 } }, 'DecisionFinalized'],
      [{ StoryCreated: { hearth_hash: MOCK_HASH, story_hash: MOCK_HASH, storyteller: MOCK_AGENT, story_type: 'Memory' } }, 'StoryCreated'],
      [{ StoryUpdated: { story_hash: MOCK_HASH, updated_by: MOCK_AGENT } }, 'StoryUpdated'],
      [{ TraditionObserved: { tradition_hash: MOCK_HASH, observed_by: MOCK_AGENT } }, 'TraditionObserved'],
      [{ ResourceLent: { resource_hash: MOCK_HASH, borrower: MOCK_AGENT, due_date: 2000000 } }, 'ResourceLent'],
      [{ ResourceReturned: { loan_hash: MOCK_HASH, borrower: MOCK_AGENT } }, 'ResourceReturned'],
      [{ ExpenseLogged: { budget_hash: MOCK_HASH, amount_cents: 4500 } }, 'ExpenseLogged'],
      [{ MilestoneRecorded: { hearth_hash: MOCK_HASH, milestone_hash: MOCK_HASH, milestone_type: 'Birthday' } }, 'MilestoneRecorded'],
      [{ TransitionAdvanced: { transition_hash: MOCK_HASH, new_phase: 'Liminal' } }, 'TransitionAdvanced'],
      [{ TierAdvanced: { profile_hash: MOCK_HASH, from_tier: 'Supervised', to_tier: 'Guided' } }, 'TierAdvanced'],
      [{ CapabilityApproved: { request_hash: MOCK_HASH, capability: 'drive_car' } }, 'CapabilityApproved'],
      [{ CapabilityDenied: { request_hash: MOCK_HASH, capability: 'use_stove' } }, 'CapabilityDenied'],
      [{ TransitionProgressed: { transition_hash: MOCK_HASH, new_phase: 'PostLiminal' } }, 'TransitionProgressed'],
    ];

    for (const [signal, expectedType] of variants) {
      expect(getSignalType(signal)).toBe(expectedType);
    }
  });

  it('onSignal dispatches hearth signals to callback', () => {
    const mock = createMockClient(new Map());
    const listeners: Array<(signal: unknown) => void> = [];

    // Mock AppWebsocket.on('signal', ...)
    const appClientMock = {
      ...mock,
      myPubKey: MOCK_AGENT,
      appInfo: vi.fn(async () => ({})),
      on: vi.fn((event: string, cb: (signal: unknown) => void) => {
        if (event === 'signal') listeners.push(cb);
        return () => {};
      }),
    };

    const hearth = HearthClient.fromClient(appClientMock as any);
    const received: HearthSignal[] = [];

    hearth.onSignal((signal) => {
      received.push(signal);
    });

    // Simulate a signal from hearth_kinship
    const mockSignal = {
      zome_name: 'hearth_kinship',
      payload: { MemberJoined: { hearth_hash: MOCK_HASH, agent: MOCK_AGENT, role: 'Adult' } },
    };

    for (const listener of listeners) {
      listener(mockSignal);
    }

    expect(received).toHaveLength(1);
    expect(getSignalType(received[0])).toBe('MemberJoined');
  });

  it('onSignal filters by signal type when filter provided', () => {
    const listeners: Array<(signal: unknown) => void> = [];
    const appClientMock = {
      callZome: vi.fn(),
      myPubKey: MOCK_AGENT,
      appInfo: vi.fn(async () => ({})),
      on: vi.fn((event: string, cb: (signal: unknown) => void) => {
        if (event === 'signal') listeners.push(cb);
        return () => {};
      }),
    };

    const hearth = HearthClient.fromClient(appClientMock as any);
    const received: HearthSignal[] = [];

    hearth.onSignal(
      (signal) => received.push(signal),
      ['EmergencyAlert'],
    );

    // This should be filtered out
    for (const listener of listeners) {
      listener({
        zome_name: 'hearth_kinship',
        payload: { MemberJoined: { hearth_hash: MOCK_HASH, agent: MOCK_AGENT, role: 'Adult' } },
      });
    }

    // This should pass the filter
    for (const listener of listeners) {
      listener({
        zome_name: 'hearth_emergency',
        payload: { EmergencyAlert: { alert_hash: MOCK_HASH, severity: 'Critical', message: 'Fire!' } },
      });
    }

    expect(received).toHaveLength(1);
    expect(getSignalType(received[0])).toBe('EmergencyAlert');
  });

  it('onSignal ignores non-hearth zome signals', () => {
    const listeners: Array<(signal: unknown) => void> = [];
    const appClientMock = {
      callZome: vi.fn(),
      myPubKey: MOCK_AGENT,
      appInfo: vi.fn(async () => ({})),
      on: vi.fn((event: string, cb: (signal: unknown) => void) => {
        if (event === 'signal') listeners.push(cb);
        return () => {};
      }),
    };

    const hearth = HearthClient.fromClient(appClientMock as any);
    const received: HearthSignal[] = [];
    hearth.onSignal((signal) => received.push(signal));

    // Signal from non-hearth zome should be ignored
    for (const listener of listeners) {
      listener({
        zome_name: 'governance_proposals',
        payload: { SomethingHappened: {} },
      });
    }

    expect(received).toHaveLength(0);
  });

  it('onBridgeSignal dispatches bridge event signals', () => {
    const listeners: Array<(signal: unknown) => void> = [];
    const appClientMock = {
      callZome: vi.fn(),
      myPubKey: MOCK_AGENT,
      appInfo: vi.fn(async () => ({})),
      on: vi.fn((event: string, cb: (signal: unknown) => void) => {
        if (event === 'signal') listeners.push(cb);
        return () => {};
      }),
    };

    const hearth = HearthClient.fromClient(appClientMock as any);
    const received: unknown[] = [];
    hearth.onBridgeSignal((signal) => received.push(signal));

    // Simulate a bridge event signal
    for (const listener of listeners) {
      listener({
        zome_name: 'hearth_bridge',
        payload: {
          signal_type: 'DomainEvent',
          domain: 'kinship',
          event_type: 'MemberJoined',
          payload: '{}',
          action_hash: MOCK_HASH,
        },
      });
    }

    expect(received).toHaveLength(1);
    expect((received[0] as any).signal_type).toBe('DomainEvent');
  });
});

// ============================================================================
// Error Handling Tests
// ============================================================================

describe('Error Handling', () => {
  it('throws when no mock response is configured', async () => {
    const mock = createMockClient(new Map());
    const client = new KinshipClient(mock, BASE_CONFIG);

    await expect(client.createHearth({
      name: 'Test',
      description: 'Test',
      hearth_type: 'Nuclear',
    })).rejects.toThrow('No mock response for hearth_kinship:create_hearth');
  });

  it('HearthError has correct code and message', () => {
    const err = new HearthError('NOT_MEMBER', 'Agent is not a hearth member');

    expect(err.name).toBe('HearthError');
    expect(err.code).toBe('NOT_MEMBER');
    expect(err.message).toBe('Agent is not a hearth member');
  });
});
