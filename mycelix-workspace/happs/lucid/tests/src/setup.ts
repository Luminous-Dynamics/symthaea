/**
 * LUCID Integration Test Setup
 *
 * Provides utilities for setting up Tryorama conductors
 * and interacting with LUCID zomes.
 */

import { Scenario, runScenario, dhtSync, type PlayerApp } from '@holochain/tryorama';
import type { CellId } from '@holochain/client';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Path to the LUCID hApp bundle
export const LUCID_HAPP_PATH = join(__dirname, '../../lucid.happ');

// ============================================================================
// TYPES
// ============================================================================

export interface Thought {
  id: string;
  content: string;
  thought_type: string;
  confidence: number;
  tags: string[];
  created_at: number;
  updated_at: number;
}

export interface EpistemicClassification {
  empirical: number;
  normative: number;
  materiality: number;
  harmonic: number;
}

export interface CreateThoughtInput {
  content: string;
  thought_type: string;
  confidence?: number;
  tags?: string[];
  embedding?: number[];
  epistemic?: EpistemicClassification;
}

export interface SearchInput {
  query: string;
  limit?: number;
  threshold?: number;
}

// ============================================================================
// TEMPORAL TYPES
// ============================================================================

export interface RecordVersionInput {
  thought_id: string;
  version: number;
  content: string;
  confidence: number;
  epistemic_code: string;
  change_reason?: string;
  previous_version?: Uint8Array;
}

export interface CreateSnapshotInput {
  name: string;
  description?: string;
  thought_count: number;
  avg_confidence: number;
  tags?: string[];
}

export interface BeliefAtTimeInput {
  thought_id: string;
  timestamp: number;
}

export type SnapshotTrigger =
  | 'Created'
  | 'Edited'
  | 'Scheduled'
  | 'ContradictionDetected'
  | 'ReviewedConfirmed'
  | 'ReviewedModified'
  | 'EvidenceUpdated'
  | 'CoherenceChanged';

export interface RecordSnapshotInput {
  thought_id: string;
  epistemic_code: string;
  confidence: number;
  phi: number;
  coherence: number;
  trigger: SnapshotTrigger;
}

export interface RecordEvolutionInput {
  period_start: number;
  period_end: number;
  avg_phi: number;
  phi_trend: number;
  avg_coherence: number;
  coherence_trend: number;
  stable_belief_count: number;
  growing_belief_count: number;
  weakening_belief_count: number;
  entrenched_belief_count: number;
  insights: string[];
}

export interface ContradictionInput {
  thought_a: string;
  thought_b: string;
  thought_a_content: string;
  thought_b_content: string;
  description: string;
}

export interface BeliefShare {
  belief_content: string;
  stance: 'agree' | 'disagree' | 'neutral';
  confidence: number;
}

// ============================================================================
// TEST SETUP
// ============================================================================

// Re-export PlayerApp as LucidPlayer for compatibility
export type LucidPlayer = PlayerApp;

/**
 * Create a Tryorama scenario with LUCID
 */
export async function setupLucidScenario(numPlayers: number = 1): Promise<{
  scenario: Scenario;
  players: LucidPlayer[];
}> {
  const scenario = new Scenario();

  // Add players with the LUCID app
  const apps = Array(numPlayers).fill({
    appBundleSource: { path: LUCID_HAPP_PATH },
  });

  const players = await scenario.addPlayersWithApps(apps);

  return { scenario, players };
}

/**
 * Run a test with a LUCID scenario
 */
export async function runLucidTest(
  testFn: (players: LucidPlayer[], scenario: Scenario) => Promise<void>,
  numPlayers: number = 1
): Promise<void> {
  await runScenario(async (scenario) => {
    const apps = Array(numPlayers).fill({
      appBundleSource: { path: LUCID_HAPP_PATH },
    });

    const players = await scenario.addPlayersWithApps(apps);
    await testFn(players, scenario);
  });
}

// ============================================================================
// ZOME CALL HELPERS
// ============================================================================

/**
 * Get the CellId for the lucid role from a player
 */
export function getLucidCellId(player: LucidPlayer): CellId {
  const lucidCell = player.namedCells.get('lucid');
  if (!lucidCell) {
    throw new Error('lucid cell not found');
  }
  return lucidCell.cell_id;
}

/**
 * Call a LUCID zome function
 */
export async function callLucidZome<T>(
  player: LucidPlayer,
  zome: string,
  fn: string,
  payload: unknown
): Promise<T> {
  return player.appWs.callZome({
    role_name: 'lucid',
    zome_name: zome,
    fn_name: fn,
    payload,
  }) as Promise<T>;
}

// ============================================================================
// LUCID ZOME WRAPPERS
// ============================================================================

export const lucidZome = {
  async createThought(player: LucidPlayer, input: CreateThoughtInput): Promise<Thought> {
    return callLucidZome(player, 'lucid', 'create_thought', input);
  },

  async getThought(player: LucidPlayer, id: string): Promise<Thought | null> {
    return callLucidZome(player, 'lucid', 'get_thought', id);
  },

  async getAllThoughts(player: LucidPlayer): Promise<Thought[]> {
    return callLucidZome(player, 'lucid', 'get_all_thoughts', null);
  },

  async updateThought(player: LucidPlayer, id: string, input: Partial<CreateThoughtInput>): Promise<Thought> {
    return callLucidZome(player, 'lucid', 'update_thought', { id, ...input });
  },

  async deleteThought(player: LucidPlayer, id: string): Promise<void> {
    return callLucidZome(player, 'lucid', 'delete_thought', id);
  },

  async searchThoughts(player: LucidPlayer, input: SearchInput): Promise<Thought[]> {
    return callLucidZome(player, 'lucid', 'search_thoughts', input);
  },
};

export const reasoningZome = {
  async recordContradiction(player: LucidPlayer, input: ContradictionInput): Promise<string> {
    return callLucidZome(player, 'reasoning', 'record_contradiction_analyzed', input);
  },

  async getContradictions(player: LucidPlayer): Promise<any[]> {
    return callLucidZome(player, 'reasoning', 'get_all_contradictions', null);
  },

  async resolveContradiction(player: LucidPlayer, id: string, resolution: string): Promise<void> {
    return callLucidZome(player, 'reasoning', 'resolve_contradiction', { id, resolution });
  },
};

export const collectiveZome = {
  async shareBeliefToCollective(player: LucidPlayer, input: BeliefShare): Promise<string> {
    return callLucidZome(player, 'collective', 'share_belief', input);
  },

  async getSharedBeliefs(player: LucidPlayer): Promise<any[]> {
    return callLucidZome(player, 'collective', 'get_all_shared_beliefs', null);
  },

  async voteOnBelief(player: LucidPlayer, beliefHash: string, stance: string, confidence: number): Promise<void> {
    return callLucidZome(player, 'collective', 'vote_on_belief', { beliefHash, stance, confidence });
  },

  async detectPatterns(player: LucidPlayer): Promise<any[]> {
    return callLucidZome(player, 'collective', 'detect_patterns', null);
  },
};

export const bridgeZome = {
  async storeEmbedding(player: LucidPlayer, thoughtId: string, embedding: number[]): Promise<void> {
    return callLucidZome(player, 'bridge', 'store_embedding', { thoughtId, embedding });
  },

  async getEmbedding(player: LucidPlayer, thoughtId: string): Promise<number[] | null> {
    return callLucidZome(player, 'bridge', 'get_embedding', thoughtId);
  },

  async storeCoherenceAnalysis(player: LucidPlayer, thoughtIds: string[], result: any): Promise<void> {
    return callLucidZome(player, 'bridge', 'store_coherence_analysis', { thoughtIds, result });
  },
};

export const temporalZome = {
  async recordVersion(player: LucidPlayer, input: RecordVersionInput): Promise<any> {
    return callLucidZome(player, 'temporal', 'record_version', input);
  },

  async getThoughtHistory(player: LucidPlayer, thoughtId: string): Promise<any[]> {
    return callLucidZome(player, 'temporal', 'get_thought_history', thoughtId);
  },

  async getBeliefAtTime(player: LucidPlayer, input: BeliefAtTimeInput): Promise<any> {
    return callLucidZome(player, 'temporal', 'get_belief_at_time', input);
  },

  async createSnapshot(player: LucidPlayer, input: CreateSnapshotInput): Promise<any> {
    return callLucidZome(player, 'temporal', 'create_snapshot', input);
  },

  async getMySnapshots(player: LucidPlayer): Promise<any[]> {
    return callLucidZome(player, 'temporal', 'get_my_snapshots', null);
  },
};

export const temporalConsciousnessZome = {
  async recordBeliefSnapshot(player: LucidPlayer, input: RecordSnapshotInput): Promise<any> {
    return callLucidZome(player, 'temporal-consciousness', 'record_belief_snapshot', input);
  },

  async getBeliefTrajectory(player: LucidPlayer, thoughtId: string): Promise<any> {
    return callLucidZome(player, 'temporal-consciousness', 'get_belief_trajectory', thoughtId);
  },

  async analyzeBeliefTrajectory(player: LucidPlayer, thoughtId: string): Promise<any> {
    return callLucidZome(player, 'temporal-consciousness', 'analyze_belief_trajectory', thoughtId);
  },

  async recordConsciousnessEvolution(player: LucidPlayer, input: RecordEvolutionInput): Promise<any> {
    return callLucidZome(player, 'temporal-consciousness', 'record_consciousness_evolution', input);
  },

  async getMyEvolutionHistory(player: LucidPlayer): Promise<any[]> {
    return callLucidZome(player, 'temporal-consciousness', 'get_my_evolution_history', null);
  },
};

// ============================================================================
// UTILITIES
// ============================================================================

/**
 * Wait for DHT sync between players
 */
export async function syncPlayers(players: LucidPlayer[]): Promise<void> {
  const cellId = getLucidCellId(players[0]);
  await dhtSync(players, cellId[0]);
}

/**
 * Generate a random embedding for testing
 */
export function randomEmbedding(dimension: number = 16384): number[] {
  const embedding = new Array(dimension);
  for (let i = 0; i < dimension; i++) {
    embedding[i] = Math.random() * 2 - 1;
  }
  // Normalize
  const norm = Math.sqrt(embedding.reduce((sum, v) => sum + v * v, 0));
  return embedding.map((v) => v / norm);
}
