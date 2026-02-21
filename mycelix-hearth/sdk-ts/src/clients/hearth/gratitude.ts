/**
 * Hearth Gratitude SDK client.
 * Wraps zome calls to the hearth-gratitude coordinator.
 */

import type { AppClient, Record as HolochainRecord, ActionHash, AgentPubKey } from '@holochain/client';
import type {
  ExpressGratitudeInput,
  StartCircleInput,
  DigestEpochInput,
  GratitudeSummary,
  GratitudeExpressedSignal,
} from './types';

const ROLE_NAME = 'hearth';
const ZOME_NAME = 'hearth_gratitude';

const GRATITUDE_SIGNAL_TYPES: ReadonlySet<string> = new Set([
  'GratitudeExpressed',
]);

export type GratitudeSignalHandler = (signal: GratitudeExpressedSignal) => void;

export class GratitudeClient {
  private signalHandlers: Map<string, Set<GratitudeSignalHandler>> = new Map();
  private listening = false;

  constructor(private readonly client: AppClient, private readonly roleName = ROLE_NAME) {}

  // ============================================================================
  // Zome Calls
  // ============================================================================

  async expressGratitude(input: ExpressGratitudeInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'express_gratitude',
      payload: input,
    });
  }

  async startAppreciationCircle(input: StartCircleInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'start_appreciation_circle',
      payload: input,
    });
  }

  async joinCircle(circleHash: ActionHash): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'join_circle',
      payload: circleHash,
    });
  }

  async completeCircle(circleHash: ActionHash): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'complete_circle',
      payload: circleHash,
    });
  }

  async getGratitudeStream(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_gratitude_stream',
      payload: hearthHash,
    });
  }

  async getGratitudeBalance(agent: AgentPubKey): Promise<HolochainRecord | null> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_gratitude_balance',
      payload: agent,
    });
  }

  async getHearthCircles(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_hearth_circles',
      payload: hearthHash,
    });
  }

  async createGratitudeDigest(input: DigestEpochInput): Promise<GratitudeSummary[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'create_gratitude_digest',
      payload: input,
    });
  }

  // ============================================================================
  // Signal Handling
  // ============================================================================

  /**
   * Subscribe to gratitude signals. Returns an unsubscribe function.
   *
   * @param handler - Callback invoked for each matching signal
   * @param signalType - Optional filter: only receive signals of this type.
   *                     Pass '*' or omit to receive all gratitude signals.
   *
   * @example
   * ```ts
   * const unsub = client.onSignal((signal) => {
   *   if (signal.type === 'GratitudeExpressed') console.log('Thanks!', signal.message);
   * });
   * // Later:
   * unsub();
   * ```
   */
  onSignal(
    handler: GratitudeSignalHandler,
    signalType: 'GratitudeExpressed' | '*' = '*',
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
        if (!variantName || !GRATITUDE_SIGNAL_TYPES.has(variantName)) return;

        const fields = parsed[variantName] as Record<string, unknown>;
        const typedSignal = { type: variantName, ...fields } as GratitudeExpressedSignal;

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
        // Ignore non-gratitude signals
      }
    });
  }
}
