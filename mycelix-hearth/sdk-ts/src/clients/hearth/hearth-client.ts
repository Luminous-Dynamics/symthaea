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
    this.kinship = new KinshipClient(client, roleName);
    this.decisions = new DecisionsClient(client, roleName);
    this.gratitude = new GratitudeClient(client, roleName);
    this.stories = new StoriesClient(client, roleName);
    this.care = new CareClient(client, roleName);
    this.autonomy = new AutonomyClient(client, roleName);
    this.emergency = new EmergencyClient(client, roleName);
    this.resources = new ResourcesClient(client, roleName);
    this.milestones = new MilestonesClient(client, roleName);
    this.rhythms = new RhythmsClient(client, roleName);
    this.bridge = new BridgeClient(client, roleName);
  }
}
