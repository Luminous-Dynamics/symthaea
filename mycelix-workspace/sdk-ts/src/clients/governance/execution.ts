/**
 * Execution Zome Client
 *
 * Handles proposal execution, scheduling, and cross-hApp action triggers
 * for the Governance hApp.
 *
 * @module @mycelix/sdk/clients/governance/execution
 */

import type { AppClient, Record as HolochainRecord } from '@holochain/client';
import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';
import type {
  ProposalExecution,
  RequestExecutionInput,
  AcknowledgeExecutionInput,
  ExecutionSchedule,
  ExecutionStatus,
} from './types';
// GovernanceError removed (unused)
import type { ActionHash } from '../../generated/common';

/**
 * Configuration for the Execution client
 */
export interface ExecutionClientConfig extends Partial<ZomeClientConfig> {
  /** Role name for governance DNA (default: 'governance') */
  roleName?: string;
}

const DEFAULT_CONFIG: ExecutionClientConfig = {
  roleName: 'governance',
};

/**
 * Client for Execution operations
 *
 * Handles the execution of passed proposals, including:
 * - Direct execution within the governance hApp
 * - Cross-hApp execution via bridge
 * - Scheduled/delayed execution
 * - Retry handling for failed executions
 *
 * @example
 * ```typescript
 * import { ExecutionClient } from '@mycelix/sdk/clients/governance';
 *
 * const execution = new ExecutionClient(appClient);
 *
 * // Request execution of a passed proposal
 * const exec = await execution.requestExecution({
 *   proposalId: 'uhCkkq...',
 * });
 *
 * // Monitor execution status
 * const status = await execution.getExecution(exec.id);
 * if (status.status === 'Completed') {
 *   console.log('Execution successful!');
 * }
 *
 * // Schedule delayed execution
 * const scheduled = await execution.scheduleExecution(
 *   'uhCkkq...',
 *   Date.now() + 86400000 // 24 hours from now
 * );
 * ```
 */
export class ExecutionClient extends ZomeClient {
  protected readonly zomeName = 'execution';
  

  constructor(client: AppClient, config: ExecutionClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
    
  }

  // ============================================================================
  // Execution Operations
  // ============================================================================

  /**
   * Request execution of a passed proposal
   *
   * Only proposals with status 'Passed' can be executed.
   * Creates an execution record and begins processing.
   *
   * @param input - Execution request parameters
   * @returns The execution record
   */
  async requestExecution(input: RequestExecutionInput): Promise<ProposalExecution> {
    const record = await this.callZomeOnce<HolochainRecord>('request_execution', {
      proposal_id: input.proposalId,
      target_happ: input.targetHapp,
      action: input.action,
      payload: input.payload,
    });
    return this.mapExecution(record);
  }

  /**
   * Execute a proposal immediately
   *
   * Combines request and execution in one call for immediate processing.
   *
   * @param proposalId - Proposal to execute
   * @returns The execution record
   */
  async executeNow(proposalId: ActionHash): Promise<ProposalExecution> {
    const record = await this.callZomeOnce<HolochainRecord>('execute_now', proposalId);
    return this.mapExecution(record);
  }

  /**
   * Get an execution by ID
   *
   * @param executionId - Execution identifier
   * @returns The execution or null
   */
  async getExecution(executionId: ActionHash): Promise<ProposalExecution | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_execution', executionId);
    if (!record) return null;
    return this.mapExecution(record);
  }

  /**
   * Get execution for a proposal
   *
   * @param proposalId - Proposal identifier
   * @returns The execution or null
   */
  async getExecutionForProposal(proposalId: ActionHash): Promise<ProposalExecution | null> {
    const record = await this.callZomeOrNull<HolochainRecord>(
      'get_execution_for_proposal',
      proposalId
    );
    if (!record) return null;
    return this.mapExecution(record);
  }

  /**
   * List all executions with optional status filter
   *
   * @param statusFilter - Optional status filter
   * @param limit - Maximum results
   * @returns Array of executions
   */
  async listExecutions(
    statusFilter?: ExecutionStatus,
    limit?: number
  ): Promise<ProposalExecution[]> {
    const records = await this.callZome<HolochainRecord[]>('list_executions', {
      status_filter: statusFilter,
      limit,
    });
    return records.map(r => this.mapExecution(r));
  }

  /**
   * Get pending executions
   *
   * @returns Array of pending executions
   */
  async getPendingExecutions(): Promise<ProposalExecution[]> {
    return this.listExecutions('Pending');
  }

  /**
   * Get in-progress executions
   *
   * @returns Array of in-progress executions
   */
  async getInProgressExecutions(): Promise<ProposalExecution[]> {
    return this.listExecutions('InProgress');
  }

  /**
   * Get failed executions
   *
   * @returns Array of failed executions
   */
  async getFailedExecutions(): Promise<ProposalExecution[]> {
    return this.listExecutions('Failed');
  }

  // ============================================================================
  // Execution Acknowledgment
  // ============================================================================

  /**
   * Acknowledge execution completion
   *
   * Called after cross-hApp execution completes (success or failure).
   *
   * @param input - Acknowledgment parameters
   * @returns Updated execution record
   */
  async acknowledgeExecution(input: AcknowledgeExecutionInput): Promise<ProposalExecution> {
    const record = await this.callZomeOnce<HolochainRecord>('acknowledge_execution', {
      execution_id: input.executionId,
      success: input.success,
      result_data: input.resultData,
      error_message: input.errorMessage,
    });
    return this.mapExecution(record);
  }

  /**
   * Mark execution as successful
   *
   * @param executionId - Execution identifier
   * @param resultData - Optional result data (JSON)
   * @returns Updated execution record
   */
  async markSuccess(executionId: ActionHash, resultData?: string): Promise<ProposalExecution> {
    return this.acknowledgeExecution({
      executionId,
      success: true,
      resultData,
    });
  }

  /**
   * Mark execution as failed
   *
   * @param executionId - Execution identifier
   * @param errorMessage - Error message
   * @returns Updated execution record
   */
  async markFailed(executionId: ActionHash, errorMessage: string): Promise<ProposalExecution> {
    return this.acknowledgeExecution({
      executionId,
      success: false,
      errorMessage,
    });
  }

  // ============================================================================
  // Retry & Cancel Operations
  // ============================================================================

  /**
   * Retry a failed execution
   *
   * @param executionId - Execution identifier
   * @returns Updated execution record
   */
  async retryExecution(executionId: ActionHash): Promise<ProposalExecution> {
    const record = await this.callZomeOnce<HolochainRecord>('retry_execution', executionId);
    return this.mapExecution(record);
  }

  /**
   * Cancel a pending execution
   *
   * @param executionId - Execution identifier
   * @param reason - Cancellation reason
   * @returns Updated execution record
   */
  async cancelExecution(executionId: ActionHash, reason: string): Promise<ProposalExecution> {
    const record = await this.callZomeOnce<HolochainRecord>('cancel_execution', {
      execution_id: executionId,
      reason,
    });
    return this.mapExecution(record);
  }

  /**
   * Revert a completed execution
   *
   * For executions that support rollback.
   *
   * @param executionId - Execution identifier
   * @param reason - Revert reason
   * @returns Updated execution record
   */
  async revertExecution(executionId: ActionHash, reason: string): Promise<ProposalExecution> {
    const record = await this.callZomeOnce<HolochainRecord>('revert_execution', {
      execution_id: executionId,
      reason,
    });
    return this.mapExecution(record);
  }

  // ============================================================================
  // Scheduled Execution
  // ============================================================================

  /**
   * Schedule execution for a future time
   *
   * @param proposalId - Proposal to execute
   * @param scheduledAt - Scheduled execution time (milliseconds)
   * @returns The schedule record
   */
  async scheduleExecution(
    proposalId: ActionHash,
    scheduledAt: number
  ): Promise<ExecutionSchedule> {
    const result = await this.callZomeOnce<any>('schedule_execution', {
      proposal_id: proposalId,
      scheduled_at: scheduledAt * 1000, // Convert to microseconds
    });
    return {
      id: result.id,
      proposalId: result.proposal_id,
      scheduledAt: result.scheduled_at,
      active: result.active,
      createdAt: result.created_at,
    };
  }

  /**
   * Cancel a scheduled execution
   *
   * @param scheduleId - Schedule identifier
   */
  async cancelSchedule(scheduleId: ActionHash): Promise<void> {
    await this.callZomeOnce('cancel_schedule', scheduleId);
  }

  /**
   * Get scheduled executions
   *
   * @returns Array of scheduled executions
   */
  async getScheduledExecutions(): Promise<ExecutionSchedule[]> {
    const results = await this.callZome<any[]>('get_scheduled_executions', null);
    return results.map(r => ({
      id: r.id,
      proposalId: r.proposal_id,
      scheduledAt: r.scheduled_at,
      active: r.active,
      createdAt: r.created_at,
    }));
  }

  /**
   * Get due scheduled executions
   *
   * Returns executions that are past their scheduled time.
   *
   * @returns Array of due executions
   */
  async getDueExecutions(): Promise<ExecutionSchedule[]> {
    const results = await this.callZome<any[]>('get_due_executions', null);
    return results.map(r => ({
      id: r.id,
      proposalId: r.proposal_id,
      scheduledAt: r.scheduled_at,
      active: r.active,
      createdAt: r.created_at,
    }));
  }

  // ============================================================================
  // Cross-hApp Execution
  // ============================================================================

  /**
   * Get executions targeting a specific hApp
   *
   * @param targetHapp - Target hApp identifier
   * @returns Array of executions
   */
  async getExecutionsForHapp(targetHapp: string): Promise<ProposalExecution[]> {
    const records = await this.callZome<HolochainRecord[]>('get_executions_for_happ', targetHapp);
    return records.map(r => this.mapExecution(r));
  }

  /**
   * Get pending executions for a target hApp
   *
   * Useful for hApps that need to process incoming governance actions.
   *
   * @param targetHapp - Target hApp identifier
   * @returns Array of pending executions
   */
  async getPendingExecutionsForHapp(targetHapp: string): Promise<ProposalExecution[]> {
    const records = await this.callZome<HolochainRecord[]>(
      'get_pending_executions_for_happ',
      targetHapp
    );
    return records.map(r => this.mapExecution(r));
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Get execution status description
   *
   * @param status - Execution status
   * @returns Human-readable description
   */
  getStatusDescription(status: ExecutionStatus): string {
    const descriptions: Record<ExecutionStatus, string> = {
      Pending: 'Execution is queued and waiting to be processed',
      InProgress: 'Execution is currently being processed',
      Completed: 'Execution completed successfully',
      Failed: 'Execution failed and may be retried',
      Reverted: 'Execution was completed but then reverted',
      Cancelled: 'Execution was cancelled before processing',
    };
    return descriptions[status];
  }

  /**
   * Check if execution can be retried
   *
   * @param execution - The execution record
   * @returns True if can be retried
   */
  canRetry(execution: ProposalExecution): boolean {
    return (
      execution.status === 'Failed' &&
      execution.retryCount < execution.maxRetries
    );
  }

  /**
   * Check if execution can be cancelled
   *
   * @param execution - The execution record
   * @returns True if can be cancelled
   */
  canCancel(execution: ProposalExecution): boolean {
    return execution.status === 'Pending';
  }

  /**
   * Get execution duration in seconds
   *
   * @param execution - The execution record
   * @returns Duration in seconds, or null if not started/completed
   */
  getExecutionDuration(execution: ProposalExecution): number | null {
    if (!execution.startedAt) return null;

    const endTime = execution.completedAt ?? Date.now() * 1000;
    return Math.floor((endTime - execution.startedAt) / 1000000);
  }

  /**
   * Get execution history for a proposal
   *
   * @param proposalId - Proposal identifier
   * @returns Array of executions (including retries)
   */
  async getExecutionHistory(proposalId: ActionHash): Promise<ProposalExecution[]> {
    const records = await this.callZome<HolochainRecord[]>(
      'get_execution_history',
      proposalId
    );
    return records.map(r => this.mapExecution(r));
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  /**
   * Map Holochain record to ProposalExecution type
   */
  private mapExecution(record: HolochainRecord): ProposalExecution {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      proposalId: entry.proposal_id,
      targetHapp: entry.target_happ,
      action: entry.action,
      payload: entry.payload,
      status: entry.status,
      executorDid: entry.executor_did,
      resultData: entry.result_data,
      errorMessage: entry.error_message,
      retryCount: entry.retry_count,
      maxRetries: entry.max_retries,
      requestedAt: entry.requested_at,
      startedAt: entry.started_at,
      completedAt: entry.completed_at,
    };
  }
}
