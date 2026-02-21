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

  constructor(private readonly client: AppClient, private readonly roleName = ROLE_NAME) {}

  // ============================================================================
  // Zome Calls
  // ============================================================================

  async createRhythm(input: CreateRhythmInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'create_rhythm',
      payload: input,
    });
  }

  async logOccurrence(input: LogOccurrenceInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'log_occurrence',
      payload: input,
    });
  }

  async setPresence(input: SetPresenceInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'set_presence',
      payload: input,
    });
  }

  async getHearthRhythms(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_hearth_rhythms',
      payload: hearthHash,
    });
  }

  async getRhythmOccurrences(rhythmHash: ActionHash): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_rhythm_occurrences',
      payload: rhythmHash,
    });
  }

  async getHearthPresence(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_hearth_presence',
      payload: hearthHash,
    });
  }

  async createRhythmDigest(input: DigestEpochInput): Promise<RhythmSummary[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'create_rhythm_digest',
      payload: input,
    });
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
        const parsed = signal.payload as Record<string, unknown>;
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
