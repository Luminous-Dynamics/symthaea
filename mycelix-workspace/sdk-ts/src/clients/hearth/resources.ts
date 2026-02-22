/**
 * Resources Zome Client
 *
 * Shared resource inventory, loans, budgets, and expense tracking
 * for Hearth clusters.
 *
 * @module @mycelix/sdk/clients/hearth/resources
 */

import type {
  RegisterResourceInput,
  LendResourceInput,
  CreateBudgetInput,
  LogExpenseInput,
} from './types';
import type { ActionHash } from '../../generated/common';
import type { Record } from '@holochain/client';

export interface ResourcesClientConfig {
  roleName?: string;
  timeout?: number;
}

interface ZomeCallable {
  callZome<T>(params: { role_name: string; zome_name: string; fn_name: string; payload: unknown }): Promise<T>;
}

export class ResourcesClient {
  private readonly zomeName = 'hearth_resources';

  constructor(
    private readonly client: ZomeCallable,
    private readonly config: Required<Pick<ResourcesClientConfig, 'roleName' | 'timeout'>>,
  ) {}

  // ============================================================================
  // Resource Inventory
  // ============================================================================

  async registerResource(input: RegisterResourceInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'register_resource', payload: input });
  }

  async getHearthInventory(hearthHash: ActionHash): Promise<Record[]> {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_hearth_inventory', payload: hearthHash });
  }

  // ============================================================================
  // Loans
  // ============================================================================

  async lendResource(input: LendResourceInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'lend_resource', payload: input });
  }

  async returnResource(loanHash: ActionHash): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'return_resource', payload: loanHash });
  }

  async getResourceLoans(resourceHash: ActionHash): Promise<Record[]> {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_resource_loans', payload: resourceHash });
  }

  // ============================================================================
  // Budgets
  // ============================================================================

  async createBudgetCategory(input: CreateBudgetInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'create_budget_category', payload: input });
  }

  async logExpense(input: LogExpenseInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'log_expense', payload: input });
  }

  async getBudgetSummary(hearthHash: ActionHash): Promise<Record[]> {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_budget_summary', payload: hearthHash });
  }
}
