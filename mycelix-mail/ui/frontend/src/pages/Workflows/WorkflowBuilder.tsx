// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Workflow Builder
 *
 * Visual workflow editor for creating email automation rules
 */

import React, { useState, useCallback } from 'react';

interface WorkflowNode {
  id: string;
  nodeType: NodeType;
  continueOnError?: boolean;
}

type NodeType =
  | { type: 'condition'; condition: Condition; thenBranch?: WorkflowNode[]; elseBranch?: WorkflowNode[] }
  | { type: 'action'; actionType: string; config: Record<string, unknown> }
  | { type: 'delay'; durationSeconds: number };

interface Condition {
  field: string;
  operator: ConditionOperator;
  value: string;
}

type ConditionOperator =
  | 'contains'
  | 'not_contains'
  | 'equals'
  | 'not_equals'
  | 'starts_with'
  | 'ends_with'
  | 'matches'
  | 'is_empty'
  | 'is_not_empty';

interface Workflow {
  id?: string;
  name: string;
  description?: string;
  trigger: {
    triggerType: 'email_received' | 'email_sent' | 'scheduled' | 'manual';
    conditions: Condition[];
  };
  nodes: WorkflowNode[];
  enabled: boolean;
}

interface WorkflowTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
}

const ACTION_TYPES = [
  { id: 'move_to_folder', name: 'Move to Folder', icon: '📁' },
  { id: 'apply_label', name: 'Apply Label', icon: '🏷️' },
  { id: 'mark_read', name: 'Mark as Read', icon: '✓' },
  { id: 'mark_starred', name: 'Star Email', icon: '⭐' },
  { id: 'forward', name: 'Forward', icon: '↗️' },
  { id: 'auto_reply', name: 'Auto Reply', icon: '↩️' },
  { id: 'notify', name: 'Send Notification', icon: '🔔' },
  { id: 'webhook', name: 'Call Webhook', icon: '🔗' },
];

const FIELDS = [
  { id: 'from', name: 'From' },
  { id: 'to', name: 'To' },
  { id: 'subject', name: 'Subject' },
  { id: 'body', name: 'Body' },
  { id: 'has_attachment', name: 'Has Attachment' },
];

const OPERATORS: { id: ConditionOperator; name: string }[] = [
  { id: 'contains', name: 'contains' },
  { id: 'not_contains', name: 'does not contain' },
  { id: 'equals', name: 'equals' },
  { id: 'not_equals', name: 'does not equal' },
  { id: 'starts_with', name: 'starts with' },
  { id: 'ends_with', name: 'ends with' },
  { id: 'matches', name: 'matches regex' },
  { id: 'is_empty', name: 'is empty' },
  { id: 'is_not_empty', name: 'is not empty' },
];

export default function WorkflowBuilder() {
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [templates, setTemplates] = useState<WorkflowTemplate[]>([]);
  const [editingWorkflow, setEditingWorkflow] = useState<Workflow | null>(null);
  const [view, setView] = useState<'list' | 'edit'>('list');

  React.useEffect(() => {
    fetchWorkflows();
    fetchTemplates();
  }, []);

  async function fetchWorkflows() {
    try {
      const response = await fetch('/api/workflows');
      if (response.ok) {
        setWorkflows(await response.json());
      }
    } catch (error) {
      console.error('Failed to fetch workflows:', error);
    }
  }

  async function fetchTemplates() {
    try {
      const response = await fetch('/api/workflows/templates');
      if (response.ok) {
        setTemplates(await response.json());
      }
    } catch (error) {
      console.error('Failed to fetch templates:', error);
    }
  }

  async function saveWorkflow(workflow: Workflow) {
    try {
      const method = workflow.id ? 'PUT' : 'POST';
      const url = workflow.id ? `/api/workflows/${workflow.id}` : '/api/workflows';

      const response = await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(workflow),
      });

      if (response.ok) {
        fetchWorkflows();
        setEditingWorkflow(null);
        setView('list');
      }
    } catch (error) {
      console.error('Failed to save workflow:', error);
    }
  }

  async function deleteWorkflow(id: string) {
    if (!confirm('Delete this workflow?')) return;
    try {
      await fetch(`/api/workflows/${id}`, { method: 'DELETE' });
      fetchWorkflows();
    } catch (error) {
      console.error('Failed to delete workflow:', error);
    }
  }

  async function toggleWorkflow(id: string, enabled: boolean) {
    try {
      await fetch(`/api/workflows/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      });
      fetchWorkflows();
    } catch (error) {
      console.error('Failed to toggle workflow:', error);
    }
  }

  function createNewWorkflow() {
    setEditingWorkflow({
      name: '',
      trigger: {
        triggerType: 'email_received',
        conditions: [],
      },
      nodes: [],
      enabled: true,
    });
    setView('edit');
  }

  function useTemplate(templateId: string) {
    // Would fetch template details and create workflow from it
    createNewWorkflow();
  }

  if (view === 'edit' && editingWorkflow) {
    return (
      <WorkflowEditor
        workflow={editingWorkflow}
        onSave={saveWorkflow}
        onCancel={() => {
          setEditingWorkflow(null);
          setView('list');
        }}
      />
    );
  }

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Workflow Automation</h1>
          <p className="text-muted">Automate email organization and responses</p>
        </div>
        <button
          onClick={createNewWorkflow}
          className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary/90"
        >
          Create Workflow
        </button>
      </div>

      {/* Templates */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold mb-4">Quick Start Templates</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {templates.map((template) => (
            <div
              key={template.id}
              className="bg-surface border border-border rounded-lg p-4 hover:border-primary cursor-pointer"
              onClick={() => useTemplate(template.id)}
            >
              <h3 className="font-medium">{template.name}</h3>
              <p className="text-sm text-muted mt-1">{template.description}</p>
              <span className="inline-block mt-2 px-2 py-0.5 bg-muted/30 rounded text-xs">
                {template.category}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Workflows List */}
      <div>
        <h2 className="text-lg font-semibold mb-4">Your Workflows</h2>
        {workflows.length === 0 ? (
          <div className="text-center py-12 bg-surface rounded-lg border border-border">
            <div className="text-4xl mb-4">⚡</div>
            <h3 className="text-lg font-semibold mb-2">No workflows yet</h3>
            <p className="text-muted mb-4">
              Create your first workflow to automate email handling
            </p>
            <button
              onClick={createNewWorkflow}
              className="px-4 py-2 bg-primary text-white rounded-lg"
            >
              Create Workflow
            </button>
          </div>
        ) : (
          <div className="space-y-3">
            {workflows.map((workflow) => (
              <div
                key={workflow.id}
                className={`bg-surface border rounded-lg p-4 ${
                  workflow.enabled ? 'border-border' : 'border-border/50 opacity-60'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <button
                      onClick={() => toggleWorkflow(workflow.id!, !workflow.enabled)}
                      className={`w-12 h-6 rounded-full transition-colors ${
                        workflow.enabled ? 'bg-primary' : 'bg-gray-300'
                      }`}
                    >
                      <div
                        className={`w-5 h-5 bg-white rounded-full shadow transform transition-transform ${
                          workflow.enabled ? 'translate-x-6' : 'translate-x-0.5'
                        }`}
                      />
                    </button>
                    <div>
                      <h3 className="font-medium">{workflow.name}</h3>
                      {workflow.description && (
                        <p className="text-sm text-muted">{workflow.description}</p>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-muted">
                      {workflow.trigger.triggerType.replace('_', ' ')}
                    </span>
                    <button
                      onClick={() => {
                        setEditingWorkflow(workflow);
                        setView('edit');
                      }}
                      className="p-2 hover:bg-muted/30 rounded"
                    >
                      ✏️
                    </button>
                    <button
                      onClick={() => deleteWorkflow(workflow.id!)}
                      className="p-2 hover:bg-red-50 text-red-600 rounded"
                    >
                      🗑️
                    </button>
                  </div>
                </div>
                <div className="mt-3 flex gap-2 flex-wrap">
                  {workflow.trigger.conditions.map((condition, idx) => (
                    <span
                      key={idx}
                      className="px-2 py-1 bg-blue-50 text-blue-700 rounded text-xs"
                    >
                      {condition.field} {condition.operator} "{condition.value}"
                    </span>
                  ))}
                  {workflow.nodes.map((node, idx) => (
                    <span
                      key={idx}
                      className="px-2 py-1 bg-green-50 text-green-700 rounded text-xs"
                    >
                      {node.nodeType.type === 'action'
                        ? (node.nodeType as { type: 'action'; actionType: string }).actionType.replace('_', ' ')
                        : node.nodeType.type}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

interface WorkflowEditorProps {
  workflow: Workflow;
  onSave: (workflow: Workflow) => void;
  onCancel: () => void;
}

function WorkflowEditor({ workflow: initialWorkflow, onSave, onCancel }: WorkflowEditorProps) {
  const [workflow, setWorkflow] = useState<Workflow>(initialWorkflow);

  function addCondition() {
    setWorkflow({
      ...workflow,
      trigger: {
        ...workflow.trigger,
        conditions: [
          ...workflow.trigger.conditions,
          { field: 'from', operator: 'contains', value: '' },
        ],
      },
    });
  }

  function updateCondition(index: number, condition: Condition) {
    const conditions = [...workflow.trigger.conditions];
    conditions[index] = condition;
    setWorkflow({
      ...workflow,
      trigger: { ...workflow.trigger, conditions },
    });
  }

  function removeCondition(index: number) {
    setWorkflow({
      ...workflow,
      trigger: {
        ...workflow.trigger,
        conditions: workflow.trigger.conditions.filter((_, i) => i !== index),
      },
    });
  }

  function addAction(actionType: string) {
    const newNode: WorkflowNode = {
      id: `node_${Date.now()}`,
      nodeType: {
        type: 'action',
        actionType,
        config: {},
      },
    };
    setWorkflow({
      ...workflow,
      nodes: [...workflow.nodes, newNode],
    });
  }

  function updateNode(index: number, node: WorkflowNode) {
    const nodes = [...workflow.nodes];
    nodes[index] = node;
    setWorkflow({ ...workflow, nodes });
  }

  function removeNode(index: number) {
    setWorkflow({
      ...workflow,
      nodes: workflow.nodes.filter((_, i) => i !== index),
    });
  }

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">
          {workflow.id ? 'Edit Workflow' : 'Create Workflow'}
        </h1>
        <div className="flex gap-2">
          <button
            onClick={onCancel}
            className="px-4 py-2 border border-border rounded-lg hover:bg-muted/30"
          >
            Cancel
          </button>
          <button
            onClick={() => onSave(workflow)}
            disabled={!workflow.name}
            className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary/90 disabled:opacity-50"
          >
            Save Workflow
          </button>
        </div>
      </div>

      <div className="space-y-6">
        {/* Basic Info */}
        <div className="bg-surface border border-border rounded-lg p-4">
          <h2 className="font-semibold mb-4">Basic Information</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Name</label>
              <input
                type="text"
                value={workflow.name}
                onChange={(e) => setWorkflow({ ...workflow, name: e.target.value })}
                placeholder="My Workflow"
                className="w-full px-3 py-2 border border-border rounded-lg"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Description (optional)</label>
              <textarea
                value={workflow.description || ''}
                onChange={(e) => setWorkflow({ ...workflow, description: e.target.value })}
                placeholder="What does this workflow do?"
                rows={2}
                className="w-full px-3 py-2 border border-border rounded-lg resize-none"
              />
            </div>
          </div>
        </div>

        {/* Trigger */}
        <div className="bg-surface border border-border rounded-lg p-4">
          <h2 className="font-semibold mb-4">Trigger</h2>
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">When</label>
            <select
              value={workflow.trigger.triggerType}
              onChange={(e) =>
                setWorkflow({
                  ...workflow,
                  trigger: {
                    ...workflow.trigger,
                    triggerType: e.target.value as Workflow['trigger']['triggerType'],
                  },
                })
              }
              className="w-full px-3 py-2 border border-border rounded-lg"
            >
              <option value="email_received">Email is received</option>
              <option value="email_sent">Email is sent</option>
              <option value="scheduled">On schedule</option>
              <option value="manual">Manual trigger</option>
            </select>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium">Conditions</label>
              <button
                onClick={addCondition}
                className="text-sm text-primary hover:underline"
              >
                + Add Condition
              </button>
            </div>
            {workflow.trigger.conditions.length === 0 ? (
              <p className="text-sm text-muted py-2">
                No conditions - workflow will run for all emails
              </p>
            ) : (
              <div className="space-y-2">
                {workflow.trigger.conditions.map((condition, index) => (
                  <div key={index} className="flex items-center gap-2">
                    <select
                      value={condition.field}
                      onChange={(e) =>
                        updateCondition(index, { ...condition, field: e.target.value })
                      }
                      className="px-3 py-2 border border-border rounded-lg"
                    >
                      {FIELDS.map((f) => (
                        <option key={f.id} value={f.id}>
                          {f.name}
                        </option>
                      ))}
                    </select>
                    <select
                      value={condition.operator}
                      onChange={(e) =>
                        updateCondition(index, {
                          ...condition,
                          operator: e.target.value as ConditionOperator,
                        })
                      }
                      className="px-3 py-2 border border-border rounded-lg"
                    >
                      {OPERATORS.map((op) => (
                        <option key={op.id} value={op.id}>
                          {op.name}
                        </option>
                      ))}
                    </select>
                    <input
                      type="text"
                      value={condition.value}
                      onChange={(e) =>
                        updateCondition(index, { ...condition, value: e.target.value })
                      }
                      placeholder="Value"
                      className="flex-1 px-3 py-2 border border-border rounded-lg"
                    />
                    <button
                      onClick={() => removeCondition(index)}
                      className="p-2 text-red-600 hover:bg-red-50 rounded"
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Actions */}
        <div className="bg-surface border border-border rounded-lg p-4">
          <h2 className="font-semibold mb-4">Actions</h2>

          {workflow.nodes.length > 0 && (
            <div className="space-y-2 mb-4">
              {workflow.nodes.map((node, index) => (
                <ActionNodeEditor
                  key={node.id}
                  node={node}
                  index={index}
                  onChange={(updated) => updateNode(index, updated)}
                  onRemove={() => removeNode(index)}
                />
              ))}
            </div>
          )}

          <div>
            <p className="text-sm text-muted mb-2">Add an action:</p>
            <div className="flex flex-wrap gap-2">
              {ACTION_TYPES.map((action) => (
                <button
                  key={action.id}
                  onClick={() => addAction(action.id)}
                  className="px-3 py-2 border border-border rounded-lg hover:bg-muted/30 text-sm"
                >
                  {action.icon} {action.name}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

interface ActionNodeEditorProps {
  node: WorkflowNode;
  index: number;
  onChange: (node: WorkflowNode) => void;
  onRemove: () => void;
}

function ActionNodeEditor({ node, index, onChange, onRemove }: ActionNodeEditorProps) {
  if (node.nodeType.type !== 'action') return null;

  const actionType = node.nodeType.actionType;
  const config = node.nodeType.config;
  const actionInfo = ACTION_TYPES.find((a) => a.id === actionType);

  function updateConfig(key: string, value: unknown) {
    onChange({
      ...node,
      nodeType: {
        ...node.nodeType,
        config: { ...config, [key]: value },
      },
    });
  }

  return (
    <div className="border border-border rounded-lg p-3 bg-background">
      <div className="flex items-center justify-between mb-2">
        <span className="font-medium">
          {actionInfo?.icon} {actionInfo?.name}
        </span>
        <button onClick={onRemove} className="text-red-600 hover:bg-red-50 p-1 rounded">
          ×
        </button>
      </div>

      {actionType === 'move_to_folder' && (
        <input
          type="text"
          value={(config.folder as string) || ''}
          onChange={(e) => updateConfig('folder', e.target.value)}
          placeholder="Folder name"
          className="w-full px-3 py-2 border border-border rounded text-sm"
        />
      )}

      {actionType === 'apply_label' && (
        <input
          type="text"
          value={(config.label as string) || ''}
          onChange={(e) => updateConfig('label', e.target.value)}
          placeholder="Label name"
          className="w-full px-3 py-2 border border-border rounded text-sm"
        />
      )}

      {actionType === 'forward' && (
        <input
          type="email"
          value={(config.to as string) || ''}
          onChange={(e) => updateConfig('to', e.target.value)}
          placeholder="Forward to email"
          className="w-full px-3 py-2 border border-border rounded text-sm"
        />
      )}

      {actionType === 'auto_reply' && (
        <div className="space-y-2">
          <input
            type="text"
            value={(config.subject as string) || ''}
            onChange={(e) => updateConfig('subject', e.target.value)}
            placeholder="Reply subject"
            className="w-full px-3 py-2 border border-border rounded text-sm"
          />
          <textarea
            value={(config.body as string) || ''}
            onChange={(e) => updateConfig('body', e.target.value)}
            placeholder="Reply body"
            rows={3}
            className="w-full px-3 py-2 border border-border rounded text-sm resize-none"
          />
        </div>
      )}

      {actionType === 'notify' && (
        <input
          type="text"
          value={(config.title as string) || ''}
          onChange={(e) => updateConfig('title', e.target.value)}
          placeholder="Notification title"
          className="w-full px-3 py-2 border border-border rounded text-sm"
        />
      )}

      {actionType === 'webhook' && (
        <input
          type="url"
          value={(config.url as string) || ''}
          onChange={(e) => updateConfig('url', e.target.value)}
          placeholder="https://..."
          className="w-full px-3 py-2 border border-border rounded text-sm"
        />
      )}
    </div>
  );
}
