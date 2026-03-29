// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Hearth Resources SDK client.
 * Wraps zome calls to the hearth-resources coordinator.
 */

import type { AppClient, Record as HolochainRecord, ActionHash } from '@holochain/client';
import type {
  RegisterResourceInput,
  LendResourceInput,
  CreateBudgetInput,
  LogExpenseInput,
  ResourceSignal,
  ResourceSignalType,
} from './types';
import { HearthError, classifyError } from './errors';
import { withGateRetry } from './consciousness-gate';

const ROLE_NAME = 'hearth';
const ZOME_NAME = 'hearth_resources';

const RESOURCE_SIGNAL_TYPES: ReadonlySet<string> = new Set([
  'ResourceLent',
  'ResourceReturned',
  'ExpenseLogged',
]);

export type ResourceSignalHandler = (signal: ResourceSignal) => void;

export class ResourcesClient {
  private signalHandlers: Map<string, Set<ResourceSignalHandler>> = new Map();
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

  /** Register a shared resource in the hearth inventory. */
  async registerResource(input: RegisterResourceInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('register_resource', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Lend a resource to another member. */
  async lendResource(input: LendResourceInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('lend_resource', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Return a borrowed resource. */
  async returnResource(loanHash: ActionHash): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('return_resource', loanHash);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Create a budget category for expense tracking. */
  async createBudgetCategory(input: CreateBudgetInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('create_budget_category', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Log an expense against a budget category. */
  async logExpense(input: LogExpenseInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('log_expense', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Get the full resource inventory for a hearth. */
  async getHearthInventory(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_hearth_inventory', hearthHash);
  }

  /** Get the budget summary for a hearth. */
  async getBudgetSummary(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_budget_summary', hearthHash);
  }

  /** Get all active resource loans for a hearth. */
  async getResourceLoans(resourceHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_resource_loans', resourceHash);
  }

  // ============================================================================
  // Signal Handling
  // ============================================================================

  /**
   * Subscribe to resource signals. Returns an unsubscribe function.
   *
   * @param handler - Callback invoked for each matching signal
   * @param signalType - Optional filter: only receive signals of this type.
   *                     Pass '*' or omit to receive all resource signals.
   *
   * @example
   * ```ts
   * const unsub = client.onSignal((signal) => {
   *   if (signal.type === 'ResourceLent') console.log('Lent!', signal.resource_hash);
   * });
   * // Later:
   * unsub();
   * ```
   */
  onSignal(
    handler: ResourceSignalHandler,
    signalType: ResourceSignalType | '*' = '*',
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
        if (!variantName || !RESOURCE_SIGNAL_TYPES.has(variantName)) return;

        const fields = parsed[variantName] as Record<string, unknown>;
        const typedSignal = { type: variantName, ...fields } as ResourceSignal;

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
        // Ignore non-resource signals
      }
    });
  }
}
