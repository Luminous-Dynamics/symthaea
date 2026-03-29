// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Unified Hearth SDK client.
 * Composes all 11 domain zome clients into a single entry point.
 */

import type { AppClient } from '@holochain/client';
import { KinshipClient } from './kinship';
import { DecisionsClient } from './decisions';
import { GratitudeClient } from './gratitude';
import { StoriesClient } from './stories';
import { CareClient } from './care';
import { AutonomyClient } from './autonomy';
import { EmergencyClient } from './emergency';
import { ResourcesClient } from './resources';
import { MilestonesClient } from './milestones';
import { RhythmsClient } from './rhythms';
import { BridgeClient } from './bridge';

export class HearthClient {
  readonly kinship: KinshipClient;
  readonly decisions: DecisionsClient;
  readonly gratitude: GratitudeClient;
  readonly stories: StoriesClient;
  readonly care: CareClient;
  readonly autonomy: AutonomyClient;
  readonly emergency: EmergencyClient;
  readonly resources: ResourcesClient;
  readonly milestones: MilestonesClient;
  readonly rhythms: RhythmsClient;
  readonly bridge: BridgeClient;

  constructor(client: AppClient, roleName = 'hearth') {
    this.bridge = new BridgeClient(client, roleName);
    const refreshFn = () => this.bridge.refreshCredential();
    this.kinship = new KinshipClient(client, roleName, refreshFn);
    this.decisions = new DecisionsClient(client, roleName, refreshFn);
    this.gratitude = new GratitudeClient(client, roleName, refreshFn);
    this.stories = new StoriesClient(client, roleName, refreshFn);
    this.care = new CareClient(client, roleName, refreshFn);
    this.autonomy = new AutonomyClient(client, roleName, refreshFn);
    this.emergency = new EmergencyClient(client, roleName, refreshFn);
    this.resources = new ResourcesClient(client, roleName, refreshFn);
    this.milestones = new MilestonesClient(client, roleName, refreshFn);
    this.rhythms = new RhythmsClient(client, roleName, refreshFn);
  }
}
