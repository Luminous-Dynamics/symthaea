/**
 * Tickets Zome Client
 *
 * Handles support tickets, comments, autonomous actions, and preemptive alerts.
 *
 * @module @mycelix/sdk/clients/support/tickets
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type {
  SupportTicketInput,
  UpdateTicketInput,
  TicketCommentInput,
  AutonomousActionInput,
  UndoActionInput,
  PreemptiveAlertInput,
  PromoteAlertInput,
  TicketStatus,
} from './types';
import type { ActionHash } from '../../generated/common';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

export interface TicketsClientConfig extends Partial<ZomeClientConfig> {
  roleName?: string;
}

const DEFAULT_CONFIG: TicketsClientConfig = {
  roleName: 'commons',
};

/**
 * Client for Support Tickets operations
 */
export class TicketsClient extends ZomeClient {
  protected readonly zomeName = 'support_tickets';

  constructor(client: AppClient, config: TicketsClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  // ============================================================================
  // Tickets
  // ============================================================================

  async createTicket(input: SupportTicketInput): Promise<Uint8Array> {
    const record = await this.callZomeOnce<HolochainRecord>('create_ticket', {
      title: input.title,
      description: input.description,
      category: input.category,
      priority: input.priority,
      status: input.status,
      requester: input.requester,
      assignee: input.assignee,
      autonomy_level: input.autonomyLevel,
      system_info: input.systemInfo,
      is_preemptive: input.isPreemptive,
      prediction_confidence: input.predictionConfidence,
      created_at: input.createdAt,
      updated_at: input.updatedAt,
    });
    return this.getActionHash(record);
  }

  async updateTicket(input: UpdateTicketInput): Promise<Uint8Array> {
    const record = await this.callZomeOnce<HolochainRecord>('update_ticket', {
      original_hash: input.originalHash,
      updated: {
        title: input.updated.title,
        description: input.updated.description,
        category: input.updated.category,
        priority: input.updated.priority,
        status: input.updated.status,
        requester: input.updated.requester,
        assignee: input.updated.assignee,
        autonomy_level: input.updated.autonomyLevel,
        system_info: input.updated.systemInfo,
        is_preemptive: input.updated.isPreemptive,
        prediction_confidence: input.updated.predictionConfidence,
        created_at: input.updated.createdAt,
        updated_at: input.updated.updatedAt,
      },
    });
    return this.getActionHash(record);
  }

  async getTicket(ticketHash: ActionHash): Promise<any | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_ticket', ticketHash);
    if (!record) return null;
    return this.mapTicket(record);
  }

  async listByStatus(status: TicketStatus): Promise<any[]> {
    const records = await this.callZome<HolochainRecord[]>('list_tickets_by_status', status);
    return records.map(r => this.mapTicket(r));
  }

  async listMyTickets(): Promise<any[]> {
    const records = await this.callZome<HolochainRecord[]>('list_my_tickets', null);
    return records.map(r => this.mapTicket(r));
  }

  async closeTicket(ticketHash: ActionHash): Promise<void> {
    await this.callZomeOnce<void>('close_ticket', ticketHash);
  }

  // ============================================================================
  // Comments
  // ============================================================================

  async addComment(input: TicketCommentInput): Promise<Uint8Array> {
    const record = await this.callZomeOnce<HolochainRecord>('add_comment', {
      ticket_hash: input.ticketHash,
      author: input.author,
      content: input.content,
      is_symthaea_response: input.isSymthaeaResponse,
      confidence: input.confidence,
      epistemic_status: input.epistemicStatus,
    });
    return this.getActionHash(record);
  }

  async getComments(ticketHash: ActionHash): Promise<any[]> {
    return this.callZome<any[]>('get_comments', ticketHash);
  }

  // ============================================================================
  // Autonomous Actions
  // ============================================================================

  async proposeAction(input: AutonomousActionInput): Promise<Uint8Array> {
    const record = await this.callZomeOnce<HolochainRecord>('propose_action', {
      ticket_hash: input.ticketHash,
      action_type: input.actionType,
      description: input.description,
      approved: input.approved,
      executed: input.executed,
      result: input.result,
      success: input.success,
      rollback_steps: input.rollbackSteps,
      rollback_state: input.rollbackState,
      rolled_back: input.rolledBack,
      created_at: input.createdAt,
    });
    return this.getActionHash(record);
  }

  async approveAction(actionHash: ActionHash): Promise<void> {
    await this.callZomeOnce<void>('approve_action', actionHash);
  }

  async executeAction(actionHash: ActionHash): Promise<void> {
    await this.callZomeOnce<void>('execute_action', actionHash);
  }

  async rollbackAction(actionHash: ActionHash): Promise<void> {
    await this.callZomeOnce<void>('rollback_action', actionHash);
  }

  async createUndo(input: UndoActionInput): Promise<Uint8Array> {
    const record = await this.callZomeOnce<HolochainRecord>('create_undo', {
      original_action_hash: input.originalActionHash,
      reason: input.reason,
      rollback_result: input.rollbackResult,
      created_at: input.createdAt,
    });
    return this.getActionHash(record);
  }

  // ============================================================================
  // Preemptive Alerts
  // ============================================================================

  async createPreemptiveAlert(input: PreemptiveAlertInput): Promise<Uint8Array> {
    const record = await this.callZomeOnce<HolochainRecord>('create_preemptive_alert', {
      predicted_failure: input.predictedFailure,
      expected_time_to_failure: input.expectedTimeToFailure,
      free_energy: input.freeEnergy,
      recommended_action: input.recommendedAction,
      auto_generated_ticket: input.autoGeneratedTicket,
      created_at: input.createdAt,
    });
    return this.getActionHash(record);
  }

  async listPreemptiveAlerts(): Promise<any[]> {
    return this.callZome<any[]>('list_preemptive_alerts', null);
  }

  async promoteAlertToTicket(input: PromoteAlertInput): Promise<Uint8Array> {
    const record = await this.callZomeOnce<HolochainRecord>('promote_alert_to_ticket', {
      alert_hash: input.alertHash,
      ticket: {
        title: input.ticket.title,
        description: input.ticket.description,
        category: input.ticket.category,
        priority: input.ticket.priority,
        status: input.ticket.status,
        requester: input.ticket.requester,
        assignee: input.ticket.assignee,
        autonomy_level: input.ticket.autonomyLevel,
        system_info: input.ticket.systemInfo,
        is_preemptive: input.ticket.isPreemptive,
        prediction_confidence: input.ticket.predictionConfidence,
        created_at: input.ticket.createdAt,
        updated_at: input.ticket.updatedAt,
      },
    });
    return this.getActionHash(record);
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private mapTicket(record: HolochainRecord): any {
    const entry = this.extractEntry<any>(record);
    return {
      actionHash: record.signed_action.hashed.hash,
      title: entry.title,
      description: entry.description,
      category: entry.category,
      priority: entry.priority,
      status: entry.status,
      requester: entry.requester,
      assignee: entry.assignee,
      autonomyLevel: entry.autonomy_level,
      systemInfo: entry.system_info,
      isPreemptive: entry.is_preemptive,
      predictionConfidence: entry.prediction_confidence,
      createdAt: entry.created_at,
      updatedAt: entry.updated_at,
    };
  }
}
