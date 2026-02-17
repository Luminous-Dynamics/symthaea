/**
 * Mycelix Justice Client - Master SDK (Phase 4)
 *
 * Unified client for the mycelix-justice hApp.
 * Provides dispute resolution, arbitration, mediation, and enforcement.
 *
 * Re-exports from existing comprehensive implementation.
 *
 * @module @mycelix/sdk/clients/justice
 */

// Re-export all types and clients from the existing comprehensive implementation
export * from '../../justice/index.js';
export * from '../../justice/client.js';

// Re-export defaults (these don't conflict with named exports)
export { default as JusticeClients } from '../../justice/index.js';
export { default as JusticeClientDefault } from '../../justice/client.js';

// Note: Explicit re-exports removed to avoid duplicate identifier errors.
// All types are already exported via `export *` above:
// - HolochainRecord, ZomeCallable
// - CaseCategory, CasePhase, CaseStatus, Case, FileCaseInput
// - EvidenceType, Evidence, SubmitEvidenceInput
// - DecisionOutcome, Decision, MediatorProfile, ArbitratorProfile
// - Enforcement, EnforcementType, RequestEnforcementInput
// - CasesClient, EvidenceClient, ArbitrationClient, EnforcementClient
// - createJusticeClients
// - MycelixJusticeClient, JusticeSdkError, JusticeClientConfig, JusticeConnectionOptions

/**
 * Justice Client - Unified client for mycelix-justice hApp
 *
 * Total: ~30 functions covered across 4 zome clients
 * - cases: File, get, update, search cases (10 functions)
 * - evidence: Submit, get, verify evidence (6 functions)
 * - arbitration: Mediation, arbitration, decisions (8 functions)
 * - enforcement: Cross-hApp enforcement actions (6 functions)
 */
import {
  CasesClient,
  EvidenceClient,
  ArbitrationClient,
  EnforcementClient,
  type ZomeCallable,
} from '../../justice/index.js';

import type { AppClient } from '@holochain/client';

export class JusticeClient {
  /** Case filing and management */
  readonly cases: CasesClient;

  /** Evidence submission and verification */
  readonly evidence: EvidenceClient;

  /** Mediation and arbitration */
  readonly arbitration: ArbitrationClient;

  /** Cross-hApp enforcement */
  readonly enforcement: EnforcementClient;

  constructor(
    client: AppClient | ZomeCallable,
    _appId: string = 'justice'
  ) {
    const callable = client as ZomeCallable;
    this.cases = new CasesClient(callable);
    this.evidence = new EvidenceClient(callable);
    this.arbitration = new ArbitrationClient(callable);
    this.enforcement = new EnforcementClient(callable);
  }
}
