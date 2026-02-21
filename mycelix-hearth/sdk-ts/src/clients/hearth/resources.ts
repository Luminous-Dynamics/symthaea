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
} from './types';

const ROLE_NAME = 'hearth';
const ZOME_NAME = 'hearth_resources';

export class ResourcesClient {
  constructor(private readonly client: AppClient, private readonly roleName = ROLE_NAME) {}

  // ============================================================================
  // Zome Calls
  // ============================================================================

  async registerResource(input: RegisterResourceInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'register_resource',
      payload: input,
    });
  }

  async lendResource(input: LendResourceInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'lend_resource',
      payload: input,
    });
  }

  async returnResource(loanHash: ActionHash): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'return_resource',
      payload: loanHash,
    });
  }

  async createBudgetCategory(input: CreateBudgetInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'create_budget_category',
      payload: input,
    });
  }

  async logExpense(input: LogExpenseInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'log_expense',
      payload: input,
    });
  }

  async getHearthInventory(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_hearth_inventory',
      payload: hearthHash,
    });
  }

  async getBudgetSummary(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_budget_summary',
      payload: hearthHash,
    });
  }

  async getResourceLoans(resourceHash: ActionHash): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_resource_loans',
      payload: resourceHash,
    });
  }
}
