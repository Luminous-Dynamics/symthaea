// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Shared Tryorama test setup for Civic hApp
 */

import { Scenario, Player } from '@holochain/tryorama';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/** Path to the compiled civic.happ bundle */
export const HAPP_PATH = path.join(__dirname, '../civic-happ.happ');

/** Role name in happ.yaml */
export const ROLE_NAME = 'civic';

/** Zome names */
export const ZOMES = {
  KNOWLEDGE: 'civic_knowledge',
  REPUTATION: 'agent_reputation',
} as const;

/**
 * Add players and bootstrap DHT for a test scenario
 */
export async function setupPlayers(
  scenario: Scenario,
  count = 2,
): Promise<Player[]> {
  const apps = Array.from({ length: count }, () => ({
    appBundleSource: { type: 'path' as const, value: HAPP_PATH },
  }));

  const players = await scenario.addPlayersWithApps(apps);
  await scenario.shareAllAgents();
  return players;
}

/**
 * Helper to call a civic knowledge zome function
 */
export async function callKnowledge<T>(
  player: Player,
  fnName: string,
  payload: unknown,
): Promise<T> {
  return player.cells[0].callZome({
    zome_name: ZOMES.KNOWLEDGE,
    fn_name: fnName,
    payload,
  }) as Promise<T>;
}

/**
 * Helper to call an agent reputation zome function
 */
export async function callReputation<T>(
  player: Player,
  fnName: string,
  payload: unknown,
): Promise<T> {
  return player.cells[0].callZome({
    zome_name: ZOMES.REPUTATION,
    fn_name: fnName,
    payload,
  }) as Promise<T>;
}

/** Sample knowledge entry for tests */
export const sampleKnowledge = {
  domain: 'benefits',
  knowledge_type: 'eligibility_rule',
  title: 'Test SNAP Eligibility',
  content: 'SNAP provides food assistance to low-income households.',
  geographic_scope: 'Texas',
  keywords: ['snap', 'food stamps', 'benefits'],
  source: 'USDA',
  links: ['https://www.fns.usda.gov/snap'],
  contact_phone: '1-800-221-5689',
} as const;

/** Sample agent registration for tests */
export const sampleAgent = {
  agent_type: 'symthaea_ai',
  name: 'Test Civic Agent',
  specializations: ['benefits', 'general'],
  description: 'Test agent for civic knowledge assistance',
} as const;
