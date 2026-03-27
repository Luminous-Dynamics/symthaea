// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//
// Mesh test setup — configures 3-player Tryorama scenario for multi-node testing.
// Uses the resilience hApp (8-DNA subset) for faster startup.

import { Scenario, Player, dhtSync } from '@holochain/tryorama';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Use resilience hApp (smaller than 14-DNA unified) for mesh tests
export const HAPP_PATH = path.resolve(
  __dirname,
  '../../../../happs/mycelix-resilience.happ'
);

// Fallback: commons-only hApp for simpler tests
export const COMMONS_HAPP_PATH = path.resolve(
  __dirname,
  '../../../../mycelix-commons/mycelix-commons.happ'
);

/**
 * Create a 3-player mesh scenario.
 * Each player gets the same hApp installed, simulating 3 independent nodes.
 */
export async function createMeshScenario(
  happPath: string = HAPP_PATH
): Promise<{
  scenario: Scenario;
  alice: Player;
  bob: Player;
  carol: Player;
}> {
  const scenario = new Scenario();

  const [alice, bob, carol] = await scenario.addPlayersWithApps([
    { appBundleSource: { path: happPath } },
    { appBundleSource: { path: happPath } },
    { appBundleSource: { path: happPath } },
  ]);

  return { scenario, alice, bob, carol };
}

/**
 * Wait for DHT sync across all players for a specific cell.
 */
export async function syncAll(
  players: Player[],
  cellIndex: number = 0
): Promise<void> {
  const cellId = players[0].cells[cellIndex].cell_id[0];
  await dhtSync(players, cellId);
}

// Mirror types for care-circles zome (matching Rust serde output)
export interface CareCircle {
  name: string;
  description: string;
  location: string;
  max_members: number;
  circle_type: string | { Custom: string };
  active: boolean;
}

export interface JoinCircleInput {
  circle_hash: Uint8Array;
  role: string;
}

export interface RecordCareExchangeInput {
  circle_hash: Uint8Array;
  receiver: Uint8Array;
  hours: number;
  service_description: string;
}
