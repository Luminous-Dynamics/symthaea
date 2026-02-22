/**
 * Hearth Emergency SDK client.
 * Wraps zome calls to the hearth-emergency coordinator.
 */

import type { AppClient, Record as HolochainRecord, ActionHash } from '@holochain/client';
import type {
  CreateEmergencyPlanInput,
  UpdatePlanInput,
  RaiseAlertInput,
  CheckInInput,
  EmergencyAlertSignal,
} from './types';
import { HearthError, classifyError } from './errors';
import { withGateRetry } from './consciousness-gate';

const ROLE_NAME = 'hearth';
const ZOME_NAME = 'hearth_emergency';

const EMERGENCY_SIGNAL_TYPES: ReadonlySet<string> = new Set([
  'EmergencyAlert',
]);

export type EmergencySignalHandler = (signal: EmergencyAlertSignal) => void;

export class EmergencyClient {
  private signalHandlers: Map<string, Set<EmergencySignalHandler>> = new Map();
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
  // Private helper
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

  /** Create an emergency plan for the hearth. */
  async createEmergencyPlan(input: CreateEmergencyPlanInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('create_emergency_plan', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Update an existing emergency plan. */
  async updateEmergencyPlan(input: UpdatePlanInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('update_emergency_plan', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Raise an emergency alert. */
  async raiseAlert(input: RaiseAlertInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('raise_alert', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Submit a safety check-in response to an alert. */
  async checkIn(input: CheckInInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('check_in', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Resolve (close) an active emergency alert. */
  async resolveAlert(alertHash: ActionHash): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('resolve_alert', alertHash);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Get all active emergency alerts for a hearth. */
  async getActiveAlerts(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_active_alerts', hearthHash);
  }

  /** Get all check-in responses for an alert. */
  async getAlertCheckins(alertHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_alert_checkins', alertHash);
  }

  /** Get the emergency plan for a hearth. */
  async getEmergencyPlan(hearthHash: ActionHash): Promise<HolochainRecord | null> {
    return this.callZome('get_emergency_plan', hearthHash);
  }

  // ============================================================================
  // Signal Handling
  // ============================================================================

  /**
   * Subscribe to emergency signals. Returns an unsubscribe function.
   *
   * @param handler - Callback invoked for each matching signal
   * @param signalType - Optional filter: only receive signals of this type.
   *                     Pass '*' or omit to receive all emergency signals.
   *
   * @example
   * ```ts
   * const unsub = client.onSignal((signal) => {
   *   console.log('Emergency!', signal.severity, signal.message);
   * });
   * // Later:
   * unsub();
   * ```
   */
  onSignal(
    handler: EmergencySignalHandler,
    signalType: 'EmergencyAlert' | '*' = '*',
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
        if (!variantName || !EMERGENCY_SIGNAL_TYPES.has(variantName)) return;

        const fields = parsed[variantName] as Record<string, unknown>;
        const typedSignal = { type: variantName, ...fields } as EmergencyAlertSignal;

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
        // Ignore non-emergency signals
      }
    });
  }
}
