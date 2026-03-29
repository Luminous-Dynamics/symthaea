// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Workflow Engine
 *
 * Provides:
 * - Visual workflow builder
 * - Smart rules engine
 * - Auto-responders
 * - Integrations (webhooks, Slack, Discord)
 * - Scheduled actions
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// ============================================================================
// Types
// ============================================================================

export interface Workflow {
  id: string;
  name: string;
  description?: string;
  isActive: boolean;
  trigger: WorkflowTrigger;
  conditions: WorkflowCondition[];
  actions: WorkflowAction[];
  schedule?: WorkflowSchedule;
  stats: WorkflowStats;
  createdAt: Date;
  updatedAt: Date;
  lastTriggeredAt?: Date;
}

export type WorkflowTrigger =
  | { type: 'email_received'; filters?: EmailTriggerFilters }
  | { type: 'email_sent' }
  | { type: 'trust_changed'; threshold?: number }
  | { type: 'schedule'; cron: string }
  | { type: 'manual' }
  | { type: 'webhook'; webhookId: string };

export interface EmailTriggerFilters {
  from?: string[];
  fromDomain?: string[];
  to?: string[];
  subjectContains?: string[];
  hasAttachment?: boolean;
  trustScoreMin?: number;
  trustScoreMax?: number;
}

export interface WorkflowCondition {
  id: string;
  field: ConditionField;
  operator: ConditionOperator;
  value: string | number | boolean | string[];
  logic?: 'and' | 'or';
}

export type ConditionField =
  | 'from'
  | 'to'
  | 'subject'
  | 'body'
  | 'domain'
  | 'trust_score'
  | 'has_attachment'
  | 'attachment_type'
  | 'time_of_day'
  | 'day_of_week'
  | 'label'
  | 'folder';

export type ConditionOperator =
  | 'equals'
  | 'not_equals'
  | 'contains'
  | 'not_contains'
  | 'starts_with'
  | 'ends_with'
  | 'greater_than'
  | 'less_than'
  | 'in_list'
  | 'not_in_list'
  | 'matches_regex'
  | 'is_empty'
  | 'is_not_empty';

export interface WorkflowAction {
  id: string;
  type: ActionType;
  config: ActionConfig;
  delay?: number; // milliseconds
  order: number;
}

export type ActionType =
  | 'move_to_folder'
  | 'add_label'
  | 'remove_label'
  | 'mark_read'
  | 'mark_unread'
  | 'star'
  | 'unstar'
  | 'archive'
  | 'delete'
  | 'forward'
  | 'reply'
  | 'auto_respond'
  | 'send_notification'
  | 'webhook'
  | 'slack'
  | 'discord'
  | 'set_trust'
  | 'create_task'
  | 'add_to_calendar';

export type ActionConfig =
  | { folderId: string }
  | { labelId: string }
  | { forwardTo: string; includeOriginal?: boolean }
  | { template: string; variables?: Record<string, string> }
  | { message: string; channel?: string }
  | { webhookUrl: string; payload?: Record<string, unknown> }
  | { trustScore: number }
  | { taskTitle: string; dueDate?: string }
  | { eventTitle: string; duration?: number }
  | Record<string, unknown>;

export interface WorkflowSchedule {
  type: 'once' | 'recurring';
  cron?: string;
  timezone?: string;
  startDate?: Date;
  endDate?: Date;
  nextRun?: Date;
}

export interface WorkflowStats {
  totalRuns: number;
  successfulRuns: number;
  failedRuns: number;
  avgExecutionTime: number;
  lastRunDuration?: number;
}

export interface WorkflowExecution {
  id: string;
  workflowId: string;
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  triggeredBy: string;
  triggeredAt: Date;
  completedAt?: Date;
  executedActions: {
    actionId: string;
    status: 'success' | 'failed' | 'skipped';
    result?: unknown;
    error?: string;
    duration: number;
  }[];
  context: Record<string, unknown>;
}

export interface WorkflowTemplate {
  id: string;
  name: string;
  description: string;
  category: 'productivity' | 'organization' | 'security' | 'integration' | 'custom';
  workflow: Omit<Workflow, 'id' | 'stats' | 'createdAt' | 'updatedAt'>;
  popularity: number;
}

// ============================================================================
// Rule Engine
// ============================================================================

class RuleEngine {
  evaluate(conditions: WorkflowCondition[], context: Record<string, unknown>): boolean {
    if (conditions.length === 0) return true;

    let result = this.evaluateCondition(conditions[0], context);

    for (let i = 1; i < conditions.length; i++) {
      const condition = conditions[i];
      const conditionResult = this.evaluateCondition(condition, context);

      if (condition.logic === 'or') {
        result = result || conditionResult;
      } else {
        result = result && conditionResult;
      }
    }

    return result;
  }

  private evaluateCondition(
    condition: WorkflowCondition,
    context: Record<string, unknown>
  ): boolean {
    const fieldValue = this.getFieldValue(condition.field, context);
    const { operator, value } = condition;

    switch (operator) {
      case 'equals':
        return fieldValue === value;
      case 'not_equals':
        return fieldValue !== value;
      case 'contains':
        return String(fieldValue).toLowerCase().includes(String(value).toLowerCase());
      case 'not_contains':
        return !String(fieldValue).toLowerCase().includes(String(value).toLowerCase());
      case 'starts_with':
        return String(fieldValue).toLowerCase().startsWith(String(value).toLowerCase());
      case 'ends_with':
        return String(fieldValue).toLowerCase().endsWith(String(value).toLowerCase());
      case 'greater_than':
        return Number(fieldValue) > Number(value);
      case 'less_than':
        return Number(fieldValue) < Number(value);
      case 'in_list':
        return Array.isArray(value) && value.includes(fieldValue);
      case 'not_in_list':
        return Array.isArray(value) && !value.includes(fieldValue);
      case 'matches_regex':
        try {
          return new RegExp(String(value), 'i').test(String(fieldValue));
        } catch {
          return false;
        }
      case 'is_empty':
        return !fieldValue || fieldValue === '';
      case 'is_not_empty':
        return !!fieldValue && fieldValue !== '';
      default:
        return false;
    }
  }

  private getFieldValue(field: ConditionField, context: Record<string, unknown>): unknown {
    switch (field) {
      case 'from':
        return (context.email as { from?: { email?: string } })?.from?.email;
      case 'to':
        return (context.email as { to?: { email?: string }[] })?.to?.map((t) => t.email);
      case 'subject':
        return (context.email as { subject?: string })?.subject;
      case 'body':
        return (context.email as { body?: string })?.body;
      case 'domain':
        const from = (context.email as { from?: { email?: string } })?.from?.email || '';
        return from.split('@')[1];
      case 'trust_score':
        return context.trustScore;
      case 'has_attachment':
        return ((context.email as { attachments?: unknown[] })?.attachments?.length || 0) > 0;
      case 'attachment_type':
        return (context.email as { attachments?: { mimeType?: string }[] })?.attachments?.map(
          (a) => a.mimeType
        );
      case 'time_of_day':
        return new Date().getHours();
      case 'day_of_week':
        return new Date().getDay();
      case 'label':
        return (context.email as { labels?: string[] })?.labels;
      case 'folder':
        return (context.email as { folderId?: string })?.folderId;
      default:
        return context[field];
    }
  }
}

// ============================================================================
// Action Executor
// ============================================================================

class ActionExecutor {
  async execute(
    action: WorkflowAction,
    context: Record<string, unknown>
  ): Promise<{ success: boolean; result?: unknown; error?: string }> {
    try {
      // Apply delay if specified
      if (action.delay && action.delay > 0) {
        await new Promise((resolve) => setTimeout(resolve, action.delay));
      }

      const result = await this.executeAction(action, context);
      return { success: true, result };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  }

  private async executeAction(
    action: WorkflowAction,
    context: Record<string, unknown>
  ): Promise<unknown> {
    const config = action.config as Record<string, unknown>;

    switch (action.type) {
      case 'move_to_folder':
        return this.moveToFolder(context, config.folderId as string);

      case 'add_label':
        return this.addLabel(context, config.labelId as string);

      case 'remove_label':
        return this.removeLabel(context, config.labelId as string);

      case 'mark_read':
        return this.markRead(context, true);

      case 'mark_unread':
        return this.markRead(context, false);

      case 'star':
        return this.star(context, true);

      case 'unstar':
        return this.star(context, false);

      case 'archive':
        return this.archive(context);

      case 'delete':
        return this.delete(context);

      case 'forward':
        return this.forward(context, config.forwardTo as string, config.includeOriginal as boolean);

      case 'auto_respond':
        return this.autoRespond(
          context,
          config.template as string,
          config.variables as Record<string, string>
        );

      case 'send_notification':
        return this.sendNotification(config.message as string, config.channel as string);

      case 'webhook':
        return this.callWebhook(
          config.webhookUrl as string,
          config.payload as Record<string, unknown>,
          context
        );

      case 'slack':
        return this.sendSlack(config.message as string, config.channel as string);

      case 'discord':
        return this.sendDiscord(config.message as string, config.channel as string);

      case 'set_trust':
        return this.setTrust(context, config.trustScore as number);

      case 'create_task':
        return this.createTask(config.taskTitle as string, config.dueDate as string, context);

      case 'add_to_calendar':
        return this.addToCalendar(config.eventTitle as string, config.duration as number, context);

      default:
        throw new Error(`Unknown action type: ${action.type}`);
    }
  }

  // Action implementations (mocked for now)
  private async moveToFolder(context: Record<string, unknown>, folderId: string): Promise<unknown> {
    console.log(`Moving email to folder: ${folderId}`);
    return { folderId };
  }

  private async addLabel(context: Record<string, unknown>, labelId: string): Promise<unknown> {
    console.log(`Adding label: ${labelId}`);
    return { labelId };
  }

  private async removeLabel(context: Record<string, unknown>, labelId: string): Promise<unknown> {
    console.log(`Removing label: ${labelId}`);
    return { labelId };
  }

  private async markRead(context: Record<string, unknown>, isRead: boolean): Promise<unknown> {
    console.log(`Marking email as ${isRead ? 'read' : 'unread'}`);
    return { isRead };
  }

  private async star(context: Record<string, unknown>, isStarred: boolean): Promise<unknown> {
    console.log(`${isStarred ? 'Starring' : 'Unstarring'} email`);
    return { isStarred };
  }

  private async archive(context: Record<string, unknown>): Promise<unknown> {
    console.log('Archiving email');
    return { archived: true };
  }

  private async delete(context: Record<string, unknown>): Promise<unknown> {
    console.log('Deleting email');
    return { deleted: true };
  }

  private async forward(
    context: Record<string, unknown>,
    to: string,
    includeOriginal: boolean
  ): Promise<unknown> {
    console.log(`Forwarding email to: ${to}`);
    return { forwardedTo: to };
  }

  private async autoRespond(
    context: Record<string, unknown>,
    template: string,
    variables?: Record<string, string>
  ): Promise<unknown> {
    let response = template;
    if (variables) {
      for (const [key, value] of Object.entries(variables)) {
        response = response.replace(new RegExp(`{{${key}}}`, 'g'), value);
      }
    }
    console.log(`Auto-responding with: ${response}`);
    return { response };
  }

  private async sendNotification(message: string, channel?: string): Promise<unknown> {
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification('Mycelix Mail', { body: message });
    }
    return { notified: true };
  }

  private async callWebhook(
    url: string,
    payload: Record<string, unknown>,
    context: Record<string, unknown>
  ): Promise<unknown> {
    const body = { ...payload, context };
    console.log(`Calling webhook: ${url}`, body);
    // In production: await fetch(url, { method: 'POST', body: JSON.stringify(body) });
    return { webhookCalled: true };
  }

  private async sendSlack(message: string, channel?: string): Promise<unknown> {
    console.log(`Sending to Slack ${channel || 'default'}: ${message}`);
    return { slackSent: true };
  }

  private async sendDiscord(message: string, channel?: string): Promise<unknown> {
    console.log(`Sending to Discord ${channel || 'default'}: ${message}`);
    return { discordSent: true };
  }

  private async setTrust(context: Record<string, unknown>, score: number): Promise<unknown> {
    const from = (context.email as { from?: { email?: string } })?.from?.email;
    console.log(`Setting trust for ${from} to ${score}`);
    return { trustSet: true, email: from, score };
  }

  private async createTask(
    title: string,
    dueDate: string,
    context: Record<string, unknown>
  ): Promise<unknown> {
    console.log(`Creating task: ${title}`);
    return { taskCreated: true, title, dueDate };
  }

  private async addToCalendar(
    title: string,
    duration: number,
    context: Record<string, unknown>
  ): Promise<unknown> {
    console.log(`Adding to calendar: ${title}`);
    return { eventCreated: true, title, duration };
  }
}

// ============================================================================
// Workflow Manager
// ============================================================================

class WorkflowManager {
  private ruleEngine = new RuleEngine();
  private actionExecutor = new ActionExecutor();

  async executeWorkflow(
    workflow: Workflow,
    context: Record<string, unknown>
  ): Promise<WorkflowExecution> {
    const execution: WorkflowExecution = {
      id: `exec_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      workflowId: workflow.id,
      status: 'running',
      triggeredBy: (context.triggeredBy as string) || 'system',
      triggeredAt: new Date(),
      executedActions: [],
      context,
    };

    try {
      // Check conditions
      if (!this.ruleEngine.evaluate(workflow.conditions, context)) {
        execution.status = 'completed';
        execution.completedAt = new Date();
        return execution;
      }

      // Execute actions in order
      const sortedActions = [...workflow.actions].sort((a, b) => a.order - b.order);

      for (const action of sortedActions) {
        const startTime = Date.now();
        const result = await this.actionExecutor.execute(action, context);
        const duration = Date.now() - startTime;

        execution.executedActions.push({
          actionId: action.id,
          status: result.success ? 'success' : 'failed',
          result: result.result,
          error: result.error,
          duration,
        });

        if (!result.success) {
          execution.status = 'failed';
          break;
        }
      }

      if (execution.status === 'running') {
        execution.status = 'completed';
      }
    } catch (error) {
      execution.status = 'failed';
    }

    execution.completedAt = new Date();
    return execution;
  }

  evaluateConditions(conditions: WorkflowCondition[], context: Record<string, unknown>): boolean {
    return this.ruleEngine.evaluate(conditions, context);
  }
}

// ============================================================================
// Store
// ============================================================================

interface WorkflowState {
  workflows: Workflow[];
  executions: WorkflowExecution[];
  templates: WorkflowTemplate[];

  addWorkflow: (workflow: Workflow) => void;
  updateWorkflow: (id: string, updates: Partial<Workflow>) => void;
  deleteWorkflow: (id: string) => void;
  toggleWorkflow: (id: string) => void;
  addExecution: (execution: WorkflowExecution) => void;
  updateWorkflowStats: (id: string, execution: WorkflowExecution) => void;
}

export const useWorkflowStore = create<WorkflowState>()(
  persist(
    (set) => ({
      workflows: [],
      executions: [],
      templates: defaultTemplates,

      addWorkflow: (workflow) =>
        set((state) => ({
          workflows: [...state.workflows, workflow],
        })),

      updateWorkflow: (id, updates) =>
        set((state) => ({
          workflows: state.workflows.map((w) =>
            w.id === id ? { ...w, ...updates, updatedAt: new Date() } : w
          ),
        })),

      deleteWorkflow: (id) =>
        set((state) => ({
          workflows: state.workflows.filter((w) => w.id !== id),
        })),

      toggleWorkflow: (id) =>
        set((state) => ({
          workflows: state.workflows.map((w) =>
            w.id === id ? { ...w, isActive: !w.isActive, updatedAt: new Date() } : w
          ),
        })),

      addExecution: (execution) =>
        set((state) => ({
          executions: [execution, ...state.executions.slice(0, 99)],
        })),

      updateWorkflowStats: (id, execution) =>
        set((state) => ({
          workflows: state.workflows.map((w) => {
            if (w.id !== id) return w;
            const duration = execution.completedAt
              ? execution.completedAt.getTime() - execution.triggeredAt.getTime()
              : 0;
            return {
              ...w,
              lastTriggeredAt: execution.triggeredAt,
              stats: {
                totalRuns: w.stats.totalRuns + 1,
                successfulRuns:
                  w.stats.successfulRuns + (execution.status === 'completed' ? 1 : 0),
                failedRuns: w.stats.failedRuns + (execution.status === 'failed' ? 1 : 0),
                avgExecutionTime:
                  (w.stats.avgExecutionTime * w.stats.totalRuns + duration) /
                  (w.stats.totalRuns + 1),
                lastRunDuration: duration,
              },
            };
          }),
        })),
    }),
    {
      name: 'mycelix-workflows',
    }
  )
);

// ============================================================================
// Default Templates
// ============================================================================

const defaultTemplates: WorkflowTemplate[] = [
  {
    id: 'template_auto_organize',
    name: 'Auto-Organize by Sender',
    description: 'Automatically move emails to folders based on sender domain',
    category: 'organization',
    popularity: 95,
    workflow: {
      name: 'Auto-Organize',
      isActive: true,
      trigger: { type: 'email_received' },
      conditions: [],
      actions: [
        {
          id: 'action_1',
          type: 'move_to_folder',
          config: { folderId: 'auto' },
          order: 1,
        },
      ],
    },
  },
  {
    id: 'template_auto_respond_ooo',
    name: 'Out of Office Auto-Reply',
    description: 'Automatically respond when you are away',
    category: 'productivity',
    popularity: 90,
    workflow: {
      name: 'Out of Office',
      isActive: false,
      trigger: { type: 'email_received' },
      conditions: [],
      actions: [
        {
          id: 'action_1',
          type: 'auto_respond',
          config: {
            template:
              "Thank you for your email. I'm currently out of the office and will respond when I return.",
          },
          order: 1,
        },
      ],
    },
  },
  {
    id: 'template_slack_notify',
    name: 'Slack Notification for Important Emails',
    description: 'Get Slack notifications for high-trust emails',
    category: 'integration',
    popularity: 85,
    workflow: {
      name: 'Slack Alerts',
      isActive: true,
      trigger: { type: 'email_received', filters: { trustScoreMin: 80 } },
      conditions: [],
      actions: [
        {
          id: 'action_1',
          type: 'slack',
          config: {
            message: 'New important email from {{from}}',
            channel: '#email-alerts',
          },
          order: 1,
        },
      ],
    },
  },
  {
    id: 'template_security_quarantine',
    name: 'Quarantine Low-Trust Emails',
    description: 'Move suspicious emails to quarantine folder',
    category: 'security',
    popularity: 88,
    workflow: {
      name: 'Security Quarantine',
      isActive: true,
      trigger: { type: 'email_received' },
      conditions: [
        {
          id: 'cond_1',
          field: 'trust_score',
          operator: 'less_than',
          value: 30,
        },
      ],
      actions: [
        {
          id: 'action_1',
          type: 'move_to_folder',
          config: { folderId: 'quarantine' },
          order: 1,
        },
        {
          id: 'action_2',
          type: 'add_label',
          config: { labelId: 'suspicious' },
          order: 2,
        },
      ],
    },
  },
];

// ============================================================================
// Singleton Instance
// ============================================================================

const workflowManager = new WorkflowManager();

// ============================================================================
// React Hooks
// ============================================================================

import { useCallback, useMemo } from 'react';

export function useWorkflows() {
  const { workflows, addWorkflow, updateWorkflow, deleteWorkflow, toggleWorkflow, templates } =
    useWorkflowStore();

  const createWorkflow = useCallback(
    (workflow: Omit<Workflow, 'id' | 'stats' | 'createdAt' | 'updatedAt'>) => {
      const newWorkflow: Workflow = {
        ...workflow,
        id: `workflow_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        stats: {
          totalRuns: 0,
          successfulRuns: 0,
          failedRuns: 0,
          avgExecutionTime: 0,
        },
        createdAt: new Date(),
        updatedAt: new Date(),
      };
      addWorkflow(newWorkflow);
      return newWorkflow;
    },
    [addWorkflow]
  );

  const createFromTemplate = useCallback(
    (templateId: string) => {
      const template = templates.find((t) => t.id === templateId);
      if (!template) return null;
      return createWorkflow(template.workflow);
    },
    [templates, createWorkflow]
  );

  const activeWorkflows = useMemo(() => workflows.filter((w) => w.isActive), [workflows]);

  return {
    workflows,
    activeWorkflows,
    templates,
    createWorkflow,
    createFromTemplate,
    updateWorkflow,
    deleteWorkflow,
    toggleWorkflow,
  };
}

export function useWorkflowExecution() {
  const { addExecution, updateWorkflowStats, executions, workflows } = useWorkflowStore();

  const executeWorkflow = useCallback(
    async (workflowId: string, context: Record<string, unknown>) => {
      const workflow = workflows.find((w) => w.id === workflowId);
      if (!workflow) throw new Error('Workflow not found');

      const execution = await workflowManager.executeWorkflow(workflow, context);
      addExecution(execution);
      updateWorkflowStats(workflowId, execution);
      return execution;
    },
    [workflows, addExecution, updateWorkflowStats]
  );

  const executeMatchingWorkflows = useCallback(
    async (trigger: WorkflowTrigger['type'], context: Record<string, unknown>) => {
      const matchingWorkflows = workflows.filter(
        (w) => w.isActive && w.trigger.type === trigger
      );

      const results: WorkflowExecution[] = [];
      for (const workflow of matchingWorkflows) {
        const execution = await workflowManager.executeWorkflow(workflow, context);
        addExecution(execution);
        updateWorkflowStats(workflow.id, execution);
        results.push(execution);
      }

      return results;
    },
    [workflows, addExecution, updateWorkflowStats]
  );

  const recentExecutions = useMemo(() => executions.slice(0, 20), [executions]);

  return {
    executeWorkflow,
    executeMatchingWorkflows,
    recentExecutions,
    allExecutions: executions,
  };
}

export function useWorkflowBuilder() {
  const createCondition = useCallback(
    (
      field: ConditionField,
      operator: ConditionOperator,
      value: string | number | boolean | string[],
      logic?: 'and' | 'or'
    ): WorkflowCondition => ({
      id: `cond_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      field,
      operator,
      value,
      logic,
    }),
    []
  );

  const createAction = useCallback(
    (type: ActionType, config: ActionConfig, order: number, delay?: number): WorkflowAction => ({
      id: `action_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type,
      config,
      order,
      delay,
    }),
    []
  );

  const validateWorkflow = useCallback((workflow: Partial<Workflow>): string[] => {
    const errors: string[] = [];

    if (!workflow.name?.trim()) {
      errors.push('Workflow name is required');
    }

    if (!workflow.trigger) {
      errors.push('Workflow trigger is required');
    }

    if (!workflow.actions || workflow.actions.length === 0) {
      errors.push('At least one action is required');
    }

    return errors;
  }, []);

  const testConditions = useCallback(
    (conditions: WorkflowCondition[], context: Record<string, unknown>): boolean => {
      return workflowManager.evaluateConditions(conditions, context);
    },
    []
  );

  return {
    createCondition,
    createAction,
    validateWorkflow,
    testConditions,
  };
}
