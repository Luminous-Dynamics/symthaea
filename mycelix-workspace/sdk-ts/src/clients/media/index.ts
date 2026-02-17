/**
 * Mycelix Media Client - Master SDK (Phase 4)
 *
 * Unified client for the mycelix-media hApp.
 * Provides content publication, attribution, fact-checking, and curation.
 *
 * Re-exports from existing comprehensive implementation.
 *
 * @module @mycelix/sdk/clients/media
 */

// Re-export all types and clients from the existing comprehensive implementation
export * from '../../media/index.js';
export * from '../../media/client.js';

// Re-export defaults (these don't conflict with named exports)
export { default as MediaClients } from '../../media/index.js';
export { default as MediaClientDefault } from '../../media/client.js';

// Note: Explicit re-exports removed to avoid duplicate identifier errors.
// All types are already exported via `export *` above:
// - HolochainRecord, ZomeCallable
// - ContentType, VerificationStatus, Content, PublishContentInput
// - Attribution, AddAttributionInput
// - FactCheck, SubmitFactCheckInput
// - CurationList, CreateCurationListInput, Endorsement
// - PublicationClient, AttributionClient, FactCheckClient, CurationClient
// - createMediaClients
// - MycelixMediaClient, MediaSdkError, MediaClientConfig, MediaConnectionOptions

/**
 * Media Client - Unified client for mycelix-media hApp
 *
 * Total: ~25 functions covered across 4 zome clients
 * - publication: Publish, get, update, search content (8 functions)
 * - attribution: Add, get, transfer attributions (5 functions)
 * - factCheck: Submit, get, verify fact-checks (6 functions)
 * - curation: Endorse, curate, collections (6 functions)
 */
import {
  PublicationClient,
  AttributionClient,
  FactCheckClient,
  CurationClient,
  type ZomeCallable,
} from '../../media/index.js';

import type { AppClient } from '@holochain/client';

export class MediaClient {
  /** Content publication operations */
  readonly publication: PublicationClient;

  /** Attribution management */
  readonly attribution: AttributionClient;

  /** Fact-checking operations */
  readonly factCheck: FactCheckClient;

  /** Curation and endorsement */
  readonly curation: CurationClient;

  constructor(
    client: AppClient | ZomeCallable,
    _appId: string = 'media'
  ) {
    const callable = client as ZomeCallable;
    this.publication = new PublicationClient(callable);
    this.attribution = new AttributionClient(callable);
    this.factCheck = new FactCheckClient(callable);
    this.curation = new CurationClient(callable);
  }
}
