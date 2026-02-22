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
import { HearthError, classifyError } from './errors';
import { withGateRetry } from './consciousness-gate';

const ROLE_NAME = 'hearth';
const ZOME_NAME = 'hearth_gratitude';

const GRATITUDE_SIGNAL_TYPES: ReadonlySet<string> = new Set([
  'GratitudeExpressed',
]);

export type GratitudeSignalHandler = (signal: GratitudeExpressedSignal) => void;

export class GratitudeClient {
  private signalHandlers: Map<string, Set<GratitudeSignalHandler>> = new Map();
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

  /** Express gratitude toward another hearth member. */
  async expressGratitude(input: ExpressGratitudeInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('express_gratitude', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Start a new appreciation circle in the hearth. */
  async startAppreciationCircle(input: StartCircleInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('start_appreciation_circle', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Join an active appreciation circle. */
  async joinCircle(circleHash: ActionHash): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('join_circle', circleHash);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Complete (close) an appreciation circle. */
  async completeCircle(circleHash: ActionHash): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('complete_circle', circleHash);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Get the gratitude stream for a hearth. */
  async getGratitudeStream(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_gratitude_stream', hearthHash);
  }

  /** Get the gratitude balance for an agent. */
  async getGratitudeBalance(agent: AgentPubKey): Promise<HolochainRecord | null> {
    return this.callZome('get_gratitude_balance', agent);
  }

  /** Get all appreciation circles for a hearth. */
  async getHearthCircles(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_hearth_circles', hearthHash);
  }

  /** Create a gratitude digest for a time epoch. */
  async createGratitudeDigest(input: DigestEpochInput): Promise<GratitudeSummary[]> {
    const call = () => this.callZome<GratitudeSummary[]>('create_gratitude_digest', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
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
        if (signal.type !== 'app') return;
        const parsed = signal.value.payload as Record<string, unknown>;
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
