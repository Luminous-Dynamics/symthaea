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

const ROLE_NAME = 'hearth';
const ZOME_NAME = 'hearth_emergency';

const EMERGENCY_SIGNAL_TYPES: ReadonlySet<string> = new Set([
  'EmergencyAlert',
]);

export type EmergencySignalHandler = (signal: EmergencyAlertSignal) => void;

export class EmergencyClient {
  private signalHandlers: Map<string, Set<EmergencySignalHandler>> = new Map();
  private listening = false;

  constructor(private readonly client: AppClient, private readonly roleName = ROLE_NAME) {}

  // ============================================================================
  // Zome Calls
  // ============================================================================

  async createEmergencyPlan(input: CreateEmergencyPlanInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'create_emergency_plan',
      payload: input,
    });
  }

  async updateEmergencyPlan(input: UpdatePlanInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'update_emergency_plan',
      payload: input,
    });
  }

  async raiseAlert(input: RaiseAlertInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'raise_alert',
      payload: input,
    });
  }

  async checkIn(input: CheckInInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'check_in',
      payload: input,
    });
  }

  async resolveAlert(alertHash: ActionHash): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'resolve_alert',
      payload: alertHash,
    });
  }

  async getActiveAlerts(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_active_alerts',
      payload: hearthHash,
    });
  }

  async getAlertCheckins(alertHash: ActionHash): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_alert_checkins',
      payload: alertHash,
    });
  }

  async getEmergencyPlan(hearthHash: ActionHash): Promise<HolochainRecord | null> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_emergency_plan',
      payload: hearthHash,
    });
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
        const parsed = signal.payload as Record<string, unknown>;
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
