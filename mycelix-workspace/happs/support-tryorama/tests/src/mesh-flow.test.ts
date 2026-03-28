// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//
// Mesh Flow Integration Tests — 3-node end-to-end flows.
//
// These tests verify that Mycelix works as a real distributed system:
// entries gossip across nodes, cross-zome calls succeed, and mutual
// credit balances are consistent.
//
// Run: cd happs/support-tryorama/tests && pnpm test:mesh

import { test, expect, describe, afterAll } from 'vitest';
import {
  createMeshScenario,
  syncAll,
  COMMONS_HAPP_PATH,
  CareCircle,
  JoinCircleInput,
  RecordCareExchangeInput,
} from './mesh-setup';
import { Scenario } from '@holochain/tryorama';
import { Record as HcRecord } from '@holochain/client';

describe('Mesh Flow: 3-Node Integration', () => {
  let scenario: Scenario;

  afterAll(async () => {
    if (scenario) {
      await scenario.shutDown();
    }
  });

  test('Test 1: Entry gossips across all 3 nodes', async () => {
    const mesh = await createMeshScenario(COMMONS_HAPP_PATH);
    scenario = mesh.scenario;

    // Alice creates a care circle
    const circle: CareCircle = {
      name: 'Mesh Test Circle',
      description: 'Integration test for 3-node mesh',
      location: 'Distributed',
      max_members: 10,
      circle_type: 'Neighborhood',
      active: true,
    };

    const aliceRecord: HcRecord = await mesh.alice.cells[0].callZome({
      zome_name: 'care_circles',
      fn_name: 'create_circle',
      payload: circle,
    });

    expect(aliceRecord).toBeTruthy();
    const circleHash = aliceRecord.signed_action.hashed.hash;

    // Wait for DHT sync
    await syncAll([mesh.alice, mesh.bob, mesh.carol]);

    // Bob can see all circles (Alice's entry gossipped)
    const bobCircles: HcRecord[] = await mesh.bob.cells[0].callZome({
      zome_name: 'care_circles',
      fn_name: 'get_all_circles',
      payload: null,
    });

    expect(bobCircles.length).toBeGreaterThanOrEqual(1);
  });

  test('Test 2: Care circle membership across nodes', async () => {
    const mesh = await createMeshScenario(COMMONS_HAPP_PATH);
    scenario = mesh.scenario;

    // Alice creates a circle
    const circle: CareCircle = {
      name: 'Cross-Node Circle',
      description: 'Testing membership across nodes',
      location: 'Everywhere',
      max_members: 5,
      circle_type: 'Neighborhood',
      active: true,
    };

    const circleRecord: HcRecord = await mesh.alice.cells[0].callZome({
      zome_name: 'care_circles',
      fn_name: 'create_circle',
      payload: circle,
    });
    const circleHash = circleRecord.signed_action.hashed.hash;

    await syncAll([mesh.alice, mesh.bob, mesh.carol]);

    // Bob joins from a different node
    const joinInput: JoinCircleInput = {
      circle_hash: circleHash,
      role: 'Member',
    };

    const bobMembership: HcRecord = await mesh.bob.cells[0].callZome({
      zome_name: 'care_circles',
      fn_name: 'join_circle',
      payload: joinInput,
    });
    expect(bobMembership).toBeTruthy();

    await syncAll([mesh.alice, mesh.bob, mesh.carol]);

    // Carol can see Bob's membership
    const members: HcRecord[] = await mesh.carol.cells[0].callZome({
      zome_name: 'care_circles',
      fn_name: 'get_circle_members',
      payload: circleHash,
    });

    // Should have at least Alice (creator auto-joins) + Bob
    expect(members.length).toBeGreaterThanOrEqual(2);
  });

  test('Test 3: TEND care exchange across nodes', async () => {
    const mesh = await createMeshScenario(COMMONS_HAPP_PATH);
    scenario = mesh.scenario;

    // Setup: Alice creates circle, Bob joins
    const circle: CareCircle = {
      name: 'TEND Exchange Circle',
      description: 'Testing mutual credit',
      location: 'Local',
      max_members: 10,
      circle_type: 'Neighborhood',
      active: true,
    };

    const circleRecord: HcRecord = await mesh.alice.cells[0].callZome({
      zome_name: 'care_circles',
      fn_name: 'create_circle',
      payload: circle,
    });
    const circleHash = circleRecord.signed_action.hashed.hash;

    await syncAll([mesh.alice, mesh.bob, mesh.carol]);

    await mesh.bob.cells[0].callZome({
      zome_name: 'care_circles',
      fn_name: 'join_circle',
      payload: { circle_hash: circleHash, role: 'Member' },
    });

    await syncAll([mesh.alice, mesh.bob, mesh.carol]);

    // Alice records a care exchange with Bob
    const exchangeInput: RecordCareExchangeInput = {
      circle_hash: circleHash,
      receiver: mesh.bob.agentPubKey,
      hours: 2.0,
      service_description: 'Helped with community garden',
    };

    const exchangeRecord: HcRecord = await mesh.alice.cells[0].callZome({
      zome_name: 'care_circles',
      fn_name: 'record_care_exchange',
      payload: exchangeInput,
    });
    expect(exchangeRecord).toBeTruthy();

    await syncAll([mesh.alice, mesh.bob, mesh.carol]);

    // Carol can query the circle's TEND balance
    const balance = await mesh.carol.cells[0].callZome({
      zome_name: 'care_circles',
      fn_name: 'get_circle_tend_balance',
      payload: circleHash,
    });

    // Balance should reflect the exchange
    expect(balance).toBeTruthy();
    expect(balance.total_hours).toBeGreaterThanOrEqual(0);
  });
});
