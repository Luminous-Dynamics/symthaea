/**
 * Hearth Milestones SDK client.
 * Wraps zome calls to the hearth-milestones coordinator.
 */

import type { AppClient, Record as HolochainRecord, ActionHash, AgentPubKey } from '@holochain/client';
import type {
  RecordMilestoneInput,
  BeginTransitionInput,
  MilestoneSignal,
  MilestoneSignalType,
} from './types';
import { HearthError, classifyError } from './errors';
import { withGateRetry } from './consciousness-gate';

const ROLE_NAME = 'hearth';
const ZOME_NAME = 'hearth_milestones';

const MILESTONE_SIGNAL_TYPES: ReadonlySet<string> = new Set([
  'MilestoneRecorded',
  'TransitionAdvanced',
]);

export type MilestoneSignalHandler = (signal: MilestoneSignal) => void;

export class MilestonesClient {
  private signalHandlers: Map<string, Set<MilestoneSignalHandler>> = new Map();
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
  // Private Helpers
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

  // ============================================================================
  // Zome Calls
  // ============================================================================

  /** Record a life milestone for a hearth member. */
  async recordMilestone(input: RecordMilestoneInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('record_milestone', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Begin a life transition (e.g., new school, moving). */
  async beginTransition(input: BeginTransitionInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('begin_transition', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Record progress on an active life transition. */
  async advanceTransition(transitionHash: ActionHash): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('advance_transition', transitionHash);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Mark a life transition as complete. */
  async completeTransition(transitionHash: ActionHash): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('complete_transition', transitionHash);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Get the full milestone timeline for a hearth. */
  async getFamilyTimeline(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_family_timeline', hearthHash);
  }

  /** Get all milestones for a specific member. */
  async getMemberMilestones(member: AgentPubKey): Promise<HolochainRecord[]> {
    return this.callZome('get_member_milestones', member);
  }

  /** Get all active life transitions for a hearth. */
  async getActiveTransitions(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_active_transitions', hearthHash);
  }

  // ============================================================================
  // Signal Handling
  // ============================================================================

  /**
   * Subscribe to milestone signals. Returns an unsubscribe function.
   *
   * @param handler - Callback invoked for each matching signal
   * @param signalType - Optional filter: only receive signals of this type.
   *                     Pass '*' or omit to receive all milestone signals.
   *
   * @example
   * ```ts
   * const unsub = client.onSignal((signal) => {
   *   if (signal.type === 'MilestoneRecorded') console.log('Milestone!', signal.milestone_hash);
   * });
   * // Later:
   * unsub();
   * ```
   */
  onSignal(
    handler: MilestoneSignalHandler,
    signalType: MilestoneSignalType | '*' = '*',
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
        if (!variantName || !MILESTONE_SIGNAL_TYPES.has(variantName)) return;

        const fields = parsed[variantName] as Record<string, unknown>;
        const typedSignal = { type: variantName, ...fields } as MilestoneSignal;

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
        // Ignore non-milestone signals
      }
    });
  }
}
