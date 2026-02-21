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

  constructor(private readonly client: AppClient, private readonly roleName = ROLE_NAME) {}

  // ============================================================================
  // Zome Calls
  // ============================================================================

  async createDecision(input: CreateDecisionInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'create_decision',
      payload: input,
    });
  }

  async castVote(input: CastVoteInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'cast_vote',
      payload: input,
    });
  }

  async amendVote(input: AmendVoteInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'amend_vote',
      payload: input,
    });
  }

  async tallyVotes(decisionHash: ActionHash): Promise<Array<[number, number]>> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'tally_votes',
      payload: decisionHash,
    });
  }

  async finalizeDecision(input: FinalizeDecisionInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'finalize_decision',
      payload: input,
    });
  }

  async closeDecision(decisionHash: ActionHash): Promise<HolochainRecord> {
    const input: CloseDecisionInput = { decision_hash: decisionHash };
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'close_decision',
      payload: input,
    });
  }

  async getDecision(decisionHash: ActionHash): Promise<HolochainRecord | null> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_decision',
      payload: decisionHash,
    });
  }

  async getHearthDecisions(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_hearth_decisions',
      payload: hearthHash,
    });
  }

  async getDecisionVotes(decisionHash: ActionHash): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_decision_votes',
      payload: decisionHash,
    });
  }

  async getMyPendingVotes(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_my_pending_votes',
      payload: hearthHash,
    });
  }

  async getVoteHistory(decisionHash: ActionHash): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_vote_history',
      payload: decisionHash,
    });
  }

  async getDecisionOutcome(decisionHash: ActionHash): Promise<HolochainRecord | null> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_decision_outcome',
      payload: decisionHash,
    });
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
        const parsed = signal.payload as Record<string, unknown>;
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
