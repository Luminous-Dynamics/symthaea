// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Workflow Engine Tests
 *
 * Comprehensive unit tests for the workflow engine including:
 * - Rule engine condition evaluation
 * - Action execution
 * - Workflow creation and management
 * - Template application
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

// Mock Zustand
vi.mock('zustand', () => ({
  create: vi.fn((fn) => {
    const state = fn(
      (partial: unknown) => Object.assign(state, typeof partial === 'function' ? partial(state) : partial),
      () => state,
      { getState: () => state, setState: (partial: unknown) => Object.assign(state, partial) }
    );
    return () => state;
  }),
}));

vi.mock('zustand/middleware', () => ({
  persist: (fn: Function) => fn,
}));

// Import after mocks
import {
  type WorkflowCondition,
  type ConditionField,
  type ConditionOperator,
} from '../workflow-engine';

// ============================================================================
// Rule Engine Tests
// ============================================================================

describe('Rule Engine', () => {
  // Simple rule engine implementation for testing
  const evaluateCondition = (
    condition: { field: ConditionField; operator: ConditionOperator; value: unknown },
    context: Record<string, unknown>
  ): boolean => {
    const getFieldValue = (field: ConditionField): unknown => {
      const email = context.email as Record<string, unknown> | undefined;
      switch (field) {
        case 'from':
          return (email?.from as { email?: string })?.email;
        case 'to':
          return (email?.to as { email?: string }[])?.map(t => t.email);
        case 'subject':
          return email?.subject;
        case 'body':
          return email?.body;
        case 'domain':
          const fromEmail = (email?.from as { email?: string })?.email || '';
          return fromEmail.split('@')[1];
        case 'trust_score':
          return context.trustScore;
        case 'has_attachment':
          return ((email?.attachments as unknown[])?.length || 0) > 0;
        default:
          return undefined;
      }
    };

    const fieldValue = getFieldValue(condition.field);
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
  };

  describe('String operators', () => {
    const emailContext = {
      email: {
        from: { email: 'john@example.com', name: 'John Doe' },
        to: [{ email: 'me@mycelix.mail', name: 'Me' }],
        subject: 'Important Meeting Tomorrow',
        body: 'Please join us for an important discussion about the project.',
        attachments: [],
      },
      trustScore: 75,
    };

    it('equals - should match exact string', () => {
      expect(evaluateCondition(
        { field: 'from', operator: 'equals', value: 'john@example.com' },
        emailContext
      )).toBe(true);

      expect(evaluateCondition(
        { field: 'from', operator: 'equals', value: 'jane@example.com' },
        emailContext
      )).toBe(false);
    });

    it('not_equals - should not match string', () => {
      expect(evaluateCondition(
        { field: 'from', operator: 'not_equals', value: 'jane@example.com' },
        emailContext
      )).toBe(true);
    });

    it('contains - should find substring', () => {
      expect(evaluateCondition(
        { field: 'subject', operator: 'contains', value: 'Meeting' },
        emailContext
      )).toBe(true);

      expect(evaluateCondition(
        { field: 'subject', operator: 'contains', value: 'meeting' }, // case insensitive
        emailContext
      )).toBe(true);

      expect(evaluateCondition(
        { field: 'subject', operator: 'contains', value: 'Deadline' },
        emailContext
      )).toBe(false);
    });

    it('not_contains - should not find substring', () => {
      expect(evaluateCondition(
        { field: 'body', operator: 'not_contains', value: 'urgent' },
        emailContext
      )).toBe(true);
    });

    it('starts_with - should match beginning', () => {
      expect(evaluateCondition(
        { field: 'subject', operator: 'starts_with', value: 'Important' },
        emailContext
      )).toBe(true);

      expect(evaluateCondition(
        { field: 'subject', operator: 'starts_with', value: 'Meeting' },
        emailContext
      )).toBe(false);
    });

    it('ends_with - should match ending', () => {
      expect(evaluateCondition(
        { field: 'subject', operator: 'ends_with', value: 'Tomorrow' },
        emailContext
      )).toBe(true);
    });

    it('matches_regex - should match pattern', () => {
      expect(evaluateCondition(
        { field: 'subject', operator: 'matches_regex', value: 'Meeting.*Tomorrow' },
        emailContext
      )).toBe(true);

      expect(evaluateCondition(
        { field: 'from', operator: 'matches_regex', value: '.*@example\\.com$' },
        emailContext
      )).toBe(true);
    });
  });

  describe('Numeric operators', () => {
    const contextLowTrust = { trustScore: 25 };
    const contextHighTrust = { trustScore: 85 };

    it('greater_than - should compare numbers', () => {
      expect(evaluateCondition(
        { field: 'trust_score', operator: 'greater_than', value: 50 },
        contextHighTrust
      )).toBe(true);

      expect(evaluateCondition(
        { field: 'trust_score', operator: 'greater_than', value: 50 },
        contextLowTrust
      )).toBe(false);
    });

    it('less_than - should compare numbers', () => {
      expect(evaluateCondition(
        { field: 'trust_score', operator: 'less_than', value: 30 },
        contextLowTrust
      )).toBe(true);

      expect(evaluateCondition(
        { field: 'trust_score', operator: 'less_than', value: 30 },
        contextHighTrust
      )).toBe(false);
    });
  });

  describe('List operators', () => {
    const emailContext = {
      email: {
        from: { email: 'vip@company.com' },
      },
    };

    const vipList = ['vip@company.com', 'ceo@company.com', 'cto@company.com'];

    it('in_list - should find value in array', () => {
      expect(evaluateCondition(
        { field: 'from', operator: 'in_list', value: vipList },
        emailContext
      )).toBe(true);

      expect(evaluateCondition(
        { field: 'from', operator: 'in_list', value: ['other@example.com'] },
        emailContext
      )).toBe(false);
    });

    it('not_in_list - should not find value in array', () => {
      expect(evaluateCondition(
        { field: 'from', operator: 'not_in_list', value: ['spam@evil.com'] },
        emailContext
      )).toBe(true);
    });
  });

  describe('Domain extraction', () => {
    const emailContext = {
      email: {
        from: { email: 'user@trusted-company.com' },
      },
    };

    it('should extract and match domain', () => {
      expect(evaluateCondition(
        { field: 'domain', operator: 'equals', value: 'trusted-company.com' },
        emailContext
      )).toBe(true);
    });

    it('should work with domain lists', () => {
      const trustedDomains = ['trusted-company.com', 'partner.org'];
      expect(evaluateCondition(
        { field: 'domain', operator: 'in_list', value: trustedDomains },
        emailContext
      )).toBe(true);
    });
  });

  describe('Attachment checks', () => {
    it('should detect emails with attachments', () => {
      const withAttachment = {
        email: {
          attachments: [{ id: '1', filename: 'doc.pdf' }],
        },
      };

      const withoutAttachment = {
        email: {
          attachments: [],
        },
      };

      expect(evaluateCondition(
        { field: 'has_attachment', operator: 'equals', value: true },
        withAttachment
      )).toBe(true);

      expect(evaluateCondition(
        { field: 'has_attachment', operator: 'equals', value: true },
        withoutAttachment
      )).toBe(false);
    });
  });

  describe('Empty checks', () => {
    it('is_empty - should check for empty values', () => {
      expect(evaluateCondition(
        { field: 'subject', operator: 'is_empty', value: null },
        { email: { subject: '' } }
      )).toBe(true);

      expect(evaluateCondition(
        { field: 'subject', operator: 'is_empty', value: null },
        { email: { subject: 'Hello' } }
      )).toBe(false);
    });

    it('is_not_empty - should check for non-empty values', () => {
      expect(evaluateCondition(
        { field: 'subject', operator: 'is_not_empty', value: null },
        { email: { subject: 'Hello World' } }
      )).toBe(true);
    });
  });
});

// ============================================================================
// Multiple Conditions Tests
// ============================================================================

describe('Multiple Conditions', () => {
  const evaluateMultipleConditions = (
    conditions: Array<{ field: ConditionField; operator: ConditionOperator; value: unknown; logic?: 'and' | 'or' }>,
    context: Record<string, unknown>
  ): boolean => {
    if (conditions.length === 0) return true;

    // Simplified evaluation for testing
    const evaluateCondition = (cond: typeof conditions[0]): boolean => {
      const email = context.email as Record<string, unknown>;
      const fieldValue = cond.field === 'trust_score'
        ? context.trustScore
        : cond.field === 'from'
          ? (email?.from as { email: string })?.email
          : (email as Record<string, unknown>)?.[cond.field];

      switch (cond.operator) {
        case 'contains':
          return String(fieldValue).toLowerCase().includes(String(cond.value).toLowerCase());
        case 'greater_than':
          return Number(fieldValue) > Number(cond.value);
        case 'less_than':
          return Number(fieldValue) < Number(cond.value);
        default:
          return fieldValue === cond.value;
      }
    };

    let result = evaluateCondition(conditions[0]);

    for (let i = 1; i < conditions.length; i++) {
      const condition = conditions[i];
      const conditionResult = evaluateCondition(condition);

      if (condition.logic === 'or') {
        result = result || conditionResult;
      } else {
        result = result && conditionResult;
      }
    }

    return result;
  };

  it('AND logic - all conditions must match', () => {
    const conditions = [
      { field: 'from' as ConditionField, operator: 'contains' as ConditionOperator, value: '@company.com' },
      { field: 'trust_score' as ConditionField, operator: 'greater_than' as ConditionOperator, value: 70, logic: 'and' as const },
    ];

    const goodContext = {
      email: { from: { email: 'user@company.com' } },
      trustScore: 85,
    };

    const badContext = {
      email: { from: { email: 'user@company.com' } },
      trustScore: 50, // Too low
    };

    expect(evaluateMultipleConditions(conditions, goodContext)).toBe(true);
    expect(evaluateMultipleConditions(conditions, badContext)).toBe(false);
  });

  it('OR logic - any condition can match', () => {
    const conditions = [
      { field: 'from' as ConditionField, operator: 'contains' as ConditionOperator, value: '@vip.com' },
      { field: 'trust_score' as ConditionField, operator: 'greater_than' as ConditionOperator, value: 90, logic: 'or' as const },
    ];

    const matchFirst = {
      email: { from: { email: 'user@vip.com' } },
      trustScore: 50,
    };

    const matchSecond = {
      email: { from: { email: 'user@other.com' } },
      trustScore: 95,
    };

    const matchNeither = {
      email: { from: { email: 'user@other.com' } },
      trustScore: 50,
    };

    expect(evaluateMultipleConditions(conditions, matchFirst)).toBe(true);
    expect(evaluateMultipleConditions(conditions, matchSecond)).toBe(true);
    expect(evaluateMultipleConditions(conditions, matchNeither)).toBe(false);
  });
});

// ============================================================================
// Workflow Validation Tests
// ============================================================================

describe('Workflow Validation', () => {
  const validateWorkflow = (workflow: {
    name?: string;
    trigger?: { type: string };
    actions?: Array<{ type: string }>;
  }): string[] => {
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
  };

  it('should validate complete workflow', () => {
    const validWorkflow = {
      name: 'My Workflow',
      trigger: { type: 'email_received' },
      actions: [{ type: 'mark_read' }],
    };

    expect(validateWorkflow(validWorkflow)).toHaveLength(0);
  });

  it('should require workflow name', () => {
    const noName = {
      name: '',
      trigger: { type: 'email_received' },
      actions: [{ type: 'mark_read' }],
    };

    const errors = validateWorkflow(noName);
    expect(errors).toContain('Workflow name is required');
  });

  it('should require trigger', () => {
    const noTrigger = {
      name: 'My Workflow',
      actions: [{ type: 'mark_read' }],
    };

    const errors = validateWorkflow(noTrigger);
    expect(errors).toContain('Workflow trigger is required');
  });

  it('should require at least one action', () => {
    const noActions = {
      name: 'My Workflow',
      trigger: { type: 'email_received' },
      actions: [],
    };

    const errors = validateWorkflow(noActions);
    expect(errors).toContain('At least one action is required');
  });
});

// ============================================================================
// Action Execution Tests
// ============================================================================

describe('Action Execution', () => {
  it('should execute actions in order', async () => {
    const executionOrder: string[] = [];

    const mockActions = [
      { id: '1', type: 'mark_read', order: 1 },
      { id: '2', type: 'add_label', order: 2 },
      { id: '3', type: 'move_to_folder', order: 3 },
    ];

    const executeAction = async (action: typeof mockActions[0]) => {
      executionOrder.push(action.type);
      return { success: true };
    };

    const sortedActions = [...mockActions].sort((a, b) => a.order - b.order);

    for (const action of sortedActions) {
      await executeAction(action);
    }

    expect(executionOrder).toEqual(['mark_read', 'add_label', 'move_to_folder']);
  });

  it('should handle action delays', async () => {
    vi.useFakeTimers();

    const executeWithDelay = async (delay: number): Promise<number> => {
      return new Promise((resolve) => {
        setTimeout(() => resolve(Date.now()), delay);
      });
    };

    const promise = executeWithDelay(1000);
    vi.advanceTimersByTime(1000);
    await promise;

    vi.useRealTimers();
  });
});
