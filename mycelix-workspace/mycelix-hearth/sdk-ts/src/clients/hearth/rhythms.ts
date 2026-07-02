// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Hearth Rhythms SDK client.
 * Wraps zome calls to the hearth-rhythms coordinator.
 */

import type { AppClient, Record as HolochainRecord, ActionHash } from '@holochain/client';
import type {
  CreateRhythmInput,
  LogOccurrenceInput,
  SetPresenceInput,
  DigestEpochInput,
  RhythmSummary,
  RhythmOccurredSignal,
  PresenceChangedSignal,
} from './types';
import { HearthError, classifyError } from './errors';
import { withGateRetry } from './consciousness-gate';

const ROLE_NAME = 'hearth';
const ZOME_NAME = 'hearth_rhythms';

type RhythmsSignal = RhythmOccurredSignal | PresenceChangedSignal;
type RhythmsSignalType = RhythmsSignal['type'];

const RHYTHMS_SIGNAL_TYPES: ReadonlySet<string> = new Set([
  'RhythmOccurred',
  'PresenceChanged',
]);

export type RhythmsSignalHandler = (signal: RhythmsSignal) => void;

export class RhythmsClient {
  private signalHandlers: Map<string, Set<RhythmsSignalHandler>> = new Map();
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

  /** Create a new family rhythm or ritual. */
  async createRhythm(input: CreateRhythmInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('create_rhythm', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Log an occurrence of a rhythm. */
  async logOccurrence(input: LogOccurrenceInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('log_occurrence', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Set the caller's presence status for the hearth. */
  async setPresence(input: SetPresenceInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('set_presence', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Get all rhythms for a hearth. */
  async getHearthRhythms(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_hearth_rhythms', hearthHash);
  }

  /** Get all occurrences of a specific rhythm. */
  async getRhythmOccurrences(rhythmHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_rhythm_occurrences', rhythmHash);
  }

  /** Get current presence status for all hearth members. */
  async getHearthPresence(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_hearth_presence', hearthHash);
  }

  /** Create a rhythm digest for a time epoch. */
  async createRhythmDigest(input: DigestEpochInput): Promise<RhythmSummary[]> {
    const call = () => this.callZome<RhythmSummary[]>('create_rhythm_digest', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  // ============================================================================
  // Signal Handling
  // ============================================================================

  /**
   * Subscribe to rhythm signals. Returns an unsubscribe function.
   *
   * @param handler - Callback invoked for each matching signal
   * @param signalType - Optional filter: only receive signals of this type.
   *                     Pass '*' or omit to receive all rhythm signals.
   *
   * @example
   * ```ts
   * const unsub = client.onSignal((signal) => {
   *   if (signal.type === 'RhythmOccurred') console.log('Rhythm!', signal.rhythm_hash);
   *   if (signal.type === 'PresenceChanged') console.log('Presence!', signal.status);
   * });
   * // Later:
   * unsub();
   * ```
   */
  onSignal(
    handler: RhythmsSignalHandler,
    signalType: RhythmsSignalType | '*' = '*',
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
        if (!variantName || !RHYTHMS_SIGNAL_TYPES.has(variantName)) return;

        const fields = parsed[variantName] as Record<string, unknown>;
        const typedSignal = { type: variantName, ...fields } as RhythmsSignal;

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
        // Ignore non-rhythm signals
      }
    });
  }
}
