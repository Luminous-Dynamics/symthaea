// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Hearth Decisions SDK client.
 * Wraps zome calls to the hearth-decisions coordinator.
 */

import type { AppClient, Record as HolochainRecord, ActionHash } from '@holochain/client';
import type {
  CreateDecisionInput,
  CastVoteInput,
  FinalizeDecisionInput,
  CloseDecisionInput,
  AmendVoteInput,
  DecisionSignal,
  DecisionSignalType,
} from './types';
import { HearthError, classifyError } from './errors';
import { withGateRetry } from './consciousness-gate';

const ROLE_NAME = 'hearth';
const ZOME_NAME = 'hearth_decisions';

const DECISION_SIGNAL_TYPES: ReadonlySet<string> = new Set([
  'VoteCast',
  'VoteAmended',
  'DecisionClosed',
  'DecisionFinalized',
]);

export type DecisionSignalHandler = (signal: DecisionSignal) => void;

export class DecisionsClient {
  private signalHandlers: Map<string, Set<DecisionSignalHandler>> = new Map();
  private listening = false;
  private refreshFn?: () => Promise<void>;

  constructor(
    private readonly client: AppClient,
    private readonly roleName = ROLE_NAME,
    refreshFn?: () => Promise<void>,
  ) {
    this.refreshFn = refreshFn;
  }

  // ============================================================================
  // Zome Calls
  // ============================================================================

  private async callZome<T>(fnName: string, payload: unknown): Promise<T> {
    try {
      return await this.client.callZome({
        role_name: this.roleName,
        zome_name: ZOME_NAME,
        fn_name: fnName,
        payload,
      });
    } catch (err) {
      throw new HearthError({
        code: classifyError(err),
        message: `${ZOME_NAME}.${fnName} failed: ${err}`,
        zome: ZOME_NAME,
        fnName,
        cause: err,
      });
    }
  }

  /** Create a new decision for hearth members to vote on. */
  async createDecision(input: CreateDecisionInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('create_decision', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Cast a vote on an open decision. */
  async castVote(input: CastVoteInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('cast_vote', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Amend a previously cast vote. */
  async amendVote(input: AmendVoteInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('amend_vote', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Tally current votes for a decision. Returns array of [choice, weight] tuples. */
  async tallyVotes(decisionHash: ActionHash): Promise<Array<[number, number]>> {
    return this.callZome('tally_votes', decisionHash);
  }

  /** Finalize a decision after voting closes. */
  async finalizeDecision(input: FinalizeDecisionInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('finalize_decision', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Close a decision without finalizing (cancel). */
  async closeDecision(decisionHash: ActionHash): Promise<HolochainRecord> {
    const closeInput: CloseDecisionInput = { decision_hash: decisionHash };
    const call = () => this.callZome<HolochainRecord>('close_decision', closeInput);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Get a decision record by hash. Returns null if not found. */
  async getDecision(decisionHash: ActionHash): Promise<HolochainRecord | null> {
    return this.callZome('get_decision', decisionHash);
  }

  /** Get all decisions for a hearth. */
  async getHearthDecisions(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_hearth_decisions', hearthHash);
  }

  /** Get all votes cast on a decision. */
  async getDecisionVotes(decisionHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_decision_votes', decisionHash);
  }

  /** Get decisions in a hearth where the caller hasn't voted yet. */
  async getMyPendingVotes(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_my_pending_votes', hearthHash);
  }

  /** Get the vote amendment history for a decision. */
  async getVoteHistory(decisionHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_vote_history', decisionHash);
  }

  /** Get the outcome record for a finalized decision. */
  async getDecisionOutcome(decisionHash: ActionHash): Promise<HolochainRecord | null> {
    return this.callZome('get_decision_outcome', decisionHash);
  }

  // ============================================================================
  // Signal Handling
  // ============================================================================

  /**
   * Subscribe to decision signals. Returns an unsubscribe function.
   *
   * @param handler - Callback invoked for each matching signal
   * @param signalType - Optional filter: only receive signals of this type.
   *                     Pass '*' or omit to receive all decision signals.
   *
   * @example
   * ```ts
   * const unsub = client.onSignal((signal) => {
   *   if (signal.type === 'VoteCast') console.log('Vote!', signal.choice);
   * });
   * // Later:
   * unsub();
   * ```
   */
  onSignal(
    handler: DecisionSignalHandler,
    signalType: DecisionSignalType | '*' = '*',
  ): () => void {
    this.ensureListening();

    const key = signalType;
    if (!this.signalHandlers.has(key)) {
      this.signalHandlers.set(key, new Set());
    }
    this.signalHandlers.get(key)!.add(handler);

    return () => {
      const handlers = this.signalHandlers.get(key);
      if (handlers) {
        handlers.delete(handler);
        if (handlers.size === 0) {
          this.signalHandlers.delete(key);
        }
      }
    };
  }

  private ensureListening(): void {
    if (this.listening) return;
    this.listening = true;

    this.client.on('signal', (signal) => {
      try {
        if (signal.type !== 'app') return;
        const parsed = signal.value.payload as Record<string, unknown>;
        if (!parsed || typeof parsed !== 'object') return;

        // Rust enums serialize as { "VariantName": { fields... } }
        const variantName = Object.keys(parsed)[0];
        if (!variantName || !DECISION_SIGNAL_TYPES.has(variantName)) return;

        const fields = parsed[variantName] as Record<string, unknown>;
        const typedSignal = { type: variantName, ...fields } as DecisionSignal;

        // Notify type-specific handlers
        const typeHandlers = this.signalHandlers.get(variantName);
        if (typeHandlers) {
          typeHandlers.forEach((h) => h(typedSignal));
        }

        // Notify wildcard handlers
        const wildcardHandlers = this.signalHandlers.get('*');
        if (wildcardHandlers) {
          wildcardHandlers.forEach((h) => h(typedSignal));
        }
      } catch {
        // Ignore non-decision signals
      }
    });
  }
}
