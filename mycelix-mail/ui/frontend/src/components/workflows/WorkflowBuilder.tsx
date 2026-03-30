// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * WorkflowBuilder - Visual Drag-and-Drop Workflow Editor
 *
 * A visual interface for creating email automation workflows with:
 * - Drag-and-drop trigger, condition, and action nodes
 * - Visual connections between nodes
 * - Real-time validation
 * - Template library
 * - Test mode with sample data
 */

import React, { useState, useCallback, useRef, useMemo } from 'react';
import {
  Play,
  Pause,
  Save,
  Trash2,
  Plus,
  Settings,
  Zap,
  Filter,
  ArrowRight,
  Mail,
  Folder,
  Tag,
  Star,
  Archive,
  Forward,
  Reply,
  Bell,
  Webhook,
  Clock,
  Users,
  Shield,
  CheckCircle,
  XCircle,
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  GripVertical,
  Copy,
  MoreHorizontal,
  Search,
  X,
  MessageSquare,
} from 'lucide-react';
import {
  useWorkflows,
  useWorkflowBuilder,
  useWorkflowExecution,
  type Workflow,
  type WorkflowTrigger,
  type WorkflowCondition,
  type WorkflowAction,
  type ActionType,
  type ConditionField,
  type ConditionOperator,
} from '../../lib/workflow-engine';

// ============================================================================
// Types
// ============================================================================

interface DragItem {
  type: 'trigger' | 'condition' | 'action';
  id: string;
  data: TriggerOption | ConditionOption | ActionOption;
}

interface TriggerOption {
  type: WorkflowTrigger['type'];
  label: string;
  icon: React.ElementType;
  description: string;
}

interface ConditionOption {
  field: ConditionField;
  label: string;
  icon: React.ElementType;
  operators: ConditionOperator[];
}

interface ActionOption {
  type: ActionType;
  label: string;
  icon: React.ElementType;
  description: string;
  configFields: ConfigField[];
}

interface ConfigField {
  name: string;
  label: string;
  type: 'text' | 'email' | 'select' | 'number' | 'textarea' | 'folder' | 'label';
  required?: boolean;
  options?: { value: string; label: string }[];
  placeholder?: string;
}

// ============================================================================
// Options Data
// ============================================================================

const triggerOptions: TriggerOption[] = [
  { type: 'email_received', label: 'Email Received', icon: Mail, description: 'When a new email arrives' },
  { type: 'email_sent', label: 'Email Sent', icon: Forward, description: 'When you send an email' },
  { type: 'schedule', label: 'Scheduled', icon: Clock, description: 'Run on a schedule' },
  { type: 'trust_changed', label: 'Trust Changed', icon: Shield, description: 'When trust score changes' },
  { type: 'webhook', label: 'Webhook', icon: Webhook, description: 'External webhook trigger' },
  { type: 'manual', label: 'Manual', icon: Play, description: 'Run manually' },
];

const conditionOptions: ConditionOption[] = [
  { field: 'from', label: 'Sender', icon: Users, operators: ['equals', 'contains', 'not_contains', 'in_list'] },
  { field: 'to', label: 'Recipient', icon: Users, operators: ['equals', 'contains', 'in_list'] },
  { field: 'subject', label: 'Subject', icon: Mail, operators: ['contains', 'not_contains', 'starts_with', 'matches_regex'] },
  { field: 'body', label: 'Body', icon: MessageSquare, operators: ['contains', 'not_contains', 'matches_regex'] },
  { field: 'domain', label: 'Domain', icon: Mail, operators: ['equals', 'in_list', 'not_in_list'] },
  { field: 'trust_score', label: 'Trust Score', icon: Shield, operators: ['greater_than', 'less_than', 'equals'] },
  { field: 'has_attachment', label: 'Has Attachment', icon: Mail, operators: ['equals'] },
  { field: 'label', label: 'Label', icon: Tag, operators: ['equals', 'in_list'] },
  { field: 'folder', label: 'Folder', icon: Folder, operators: ['equals'] },
];

const actionOptions: ActionOption[] = [
  { type: 'move_to_folder', label: 'Move to Folder', icon: Folder, description: 'Move email to a folder', configFields: [{ name: 'folderId', label: 'Folder', type: 'folder', required: true }] },
  { type: 'add_label', label: 'Add Label', icon: Tag, description: 'Add a label', configFields: [{ name: 'labelId', label: 'Label', type: 'label', required: true }] },
  { type: 'remove_label', label: 'Remove Label', icon: Tag, description: 'Remove a label', configFields: [{ name: 'labelId', label: 'Label', type: 'label', required: true }] },
  { type: 'mark_read', label: 'Mark as Read', icon: CheckCircle, description: 'Mark email as read', configFields: [] },
  { type: 'mark_unread', label: 'Mark as Unread', icon: Mail, description: 'Mark email as unread', configFields: [] },
  { type: 'star', label: 'Star', icon: Star, description: 'Star the email', configFields: [] },
  { type: 'archive', label: 'Archive', icon: Archive, description: 'Archive the email', configFields: [] },
  { type: 'delete', label: 'Delete', icon: Trash2, description: 'Delete the email', configFields: [] },
  { type: 'forward', label: 'Forward', icon: Forward, description: 'Forward to another address', configFields: [{ name: 'forwardTo', label: 'Forward To', type: 'email', required: true }] },
  { type: 'auto_respond', label: 'Auto Reply', icon: Reply, description: 'Send automatic reply', configFields: [{ name: 'template', label: 'Message', type: 'textarea', required: true }] },
  { type: 'send_notification', label: 'Notification', icon: Bell, description: 'Send a notification', configFields: [{ name: 'message', label: 'Message', type: 'text', required: true }] },
  { type: 'webhook', label: 'Webhook', icon: Webhook, description: 'Call a webhook', configFields: [{ name: 'webhookUrl', label: 'URL', type: 'text', required: true }] },
  { type: 'slack', label: 'Slack', icon: MessageSquare, description: 'Send to Slack', configFields: [{ name: 'message', label: 'Message', type: 'text', required: true }, { name: 'channel', label: 'Channel', type: 'text', placeholder: '#general' }] },
  { type: 'set_trust', label: 'Set Trust', icon: Shield, description: 'Set trust score', configFields: [{ name: 'trustScore', label: 'Score', type: 'number', required: true }] },
];

// ============================================================================
// Components
// ============================================================================

// Palette Item (draggable)
function PaletteItem({
  item,
  type,
  onDragStart
}: {
  item: TriggerOption | ConditionOption | ActionOption;
  type: 'trigger' | 'condition' | 'action';
  onDragStart: (e: React.DragEvent, item: DragItem) => void;
}) {
  const Icon = item.icon;

  return (
    <div
      draggable
      onDragStart={(e) => onDragStart(e, { type, id: crypto.randomUUID(), data: item })}
      className="flex items-center gap-2 p-2 bg-white dark:bg-gray-800 border rounded-lg cursor-grab hover:shadow-md transition-shadow"
    >
      <Icon className="w-4 h-4 text-gray-500" />
      <span className="text-sm">{item.label}</span>
    </div>
  );
}

// Node Component
function WorkflowNode({
  node,
  type,
  isSelected,
  onSelect,
  onDelete,
  onConfigure,
}: {
  node: { id: string; data: TriggerOption | WorkflowCondition | WorkflowAction };
  type: 'trigger' | 'condition' | 'action';
  isSelected: boolean;
  onSelect: () => void;
  onDelete: () => void;
  onConfigure: () => void;
}) {
  const colors = {
    trigger: 'border-blue-500 bg-blue-50 dark:bg-blue-900/20',
    condition: 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20',
    action: 'border-green-500 bg-green-50 dark:bg-green-900/20',
  };

  const getIcon = () => {
    if ('icon' in node.data) return node.data.icon;
    if ('type' in node.data) {
      const opt = actionOptions.find(a => a.type === (node.data as WorkflowAction).type);
      return opt?.icon || Zap;
    }
    return Zap;
  };

  const getLabel = () => {
    if ('label' in node.data) return node.data.label;
    if ('type' in node.data) {
      const opt = actionOptions.find(a => a.type === (node.data as WorkflowAction).type);
      return opt?.label || 'Action';
    }
    return 'Node';
  };

  const Icon = getIcon();

  return (
    <div
      onClick={onSelect}
      className={`relative p-4 border-2 rounded-xl cursor-pointer transition-all ${colors[type]} ${
        isSelected ? 'ring-2 ring-offset-2 ring-blue-500' : ''
      }`}
    >
      <div className="flex items-center gap-3">
        <div className={`p-2 rounded-lg ${
          type === 'trigger' ? 'bg-blue-100 dark:bg-blue-800' :
          type === 'condition' ? 'bg-yellow-100 dark:bg-yellow-800' :
          'bg-green-100 dark:bg-green-800'
        }`}>
          <Icon className="w-5 h-5" />
        </div>
        <div>
          <p className="font-medium">{getLabel()}</p>
          <p className="text-xs text-gray-500 capitalize">{type}</p>
        </div>
      </div>

      {/* Actions */}
      <div className="absolute top-2 right-2 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
        <button
          onClick={(e) => { e.stopPropagation(); onConfigure(); }}
          className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
        >
          <Settings className="w-4 h-4" />
        </button>
        <button
          onClick={(e) => { e.stopPropagation(); onDelete(); }}
          className="p-1 hover:bg-red-100 dark:hover:bg-red-900 text-red-500 rounded"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>

      {/* Connection point */}
      {type !== 'action' && (
        <div className="absolute -bottom-3 left-1/2 -translate-x-1/2 w-6 h-6 bg-white dark:bg-gray-800 border-2 rounded-full flex items-center justify-center">
          <ArrowRight className="w-3 h-3" />
        </div>
      )}
    </div>
  );
}

// Condition Editor
function ConditionEditor({
  condition,
  onChange,
  onDelete,
}: {
  condition: WorkflowCondition;
  onChange: (updates: Partial<WorkflowCondition>) => void;
  onDelete: () => void;
}) {
  const fieldOption = conditionOptions.find(c => c.field === condition.field);

  return (
    <div className="p-4 border rounded-lg bg-yellow-50 dark:bg-yellow-900/20 space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">Condition</span>
        <button onClick={onDelete} className="p-1 hover:bg-red-100 rounded text-red-500">
          <X className="w-4 h-4" />
        </button>
      </div>

      <div className="grid grid-cols-3 gap-2">
        <select
          value={condition.field}
          onChange={(e) => onChange({ field: e.target.value as ConditionField })}
          className="px-2 py-1.5 border rounded text-sm bg-white dark:bg-gray-800"
        >
          {conditionOptions.map(opt => (
            <option key={opt.field} value={opt.field}>{opt.label}</option>
          ))}
        </select>

        <select
          value={condition.operator}
          onChange={(e) => onChange({ operator: e.target.value as ConditionOperator })}
          className="px-2 py-1.5 border rounded text-sm bg-white dark:bg-gray-800"
        >
          {fieldOption?.operators.map(op => (
            <option key={op} value={op}>{op.replace(/_/g, ' ')}</option>
          ))}
        </select>

        <input
          type="text"
          value={String(condition.value)}
          onChange={(e) => onChange({ value: e.target.value })}
          placeholder="Value"
          className="px-2 py-1.5 border rounded text-sm bg-white dark:bg-gray-800"
        />
      </div>

      <div className="flex items-center gap-2">
        <span className="text-xs text-gray-500">Logic:</span>
        <select
          value={condition.logic || 'and'}
          onChange={(e) => onChange({ logic: e.target.value as 'and' | 'or' })}
          className="px-2 py-1 border rounded text-xs bg-white dark:bg-gray-800"
        >
          <option value="and">AND</option>
          <option value="or">OR</option>
        </select>
      </div>
    </div>
  );
}

// Action Editor
function ActionEditor({
  action,
  actionOption,
  onChange,
  onDelete,
}: {
  action: WorkflowAction;
  actionOption: ActionOption;
  onChange: (updates: Partial<WorkflowAction>) => void;
  onDelete: () => void;
}) {
  const Icon = actionOption.icon;

  return (
    <div className="p-4 border rounded-lg bg-green-50 dark:bg-green-900/20 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Icon className="w-4 h-4" />
          <span className="text-sm font-medium">{actionOption.label}</span>
        </div>
        <button onClick={onDelete} className="p-1 hover:bg-red-100 rounded text-red-500">
          <X className="w-4 h-4" />
        </button>
      </div>

      {actionOption.configFields.map(field => (
        <div key={field.name}>
          <label className="block text-xs font-medium mb-1">{field.label}</label>
          {field.type === 'textarea' ? (
            <textarea
              value={(action.config as Record<string, string>)[field.name] || ''}
              onChange={(e) => onChange({ config: { ...action.config, [field.name]: e.target.value } })}
              placeholder={field.placeholder}
              rows={3}
              className="w-full px-2 py-1.5 border rounded text-sm bg-white dark:bg-gray-800"
            />
          ) : (
            <input
              type={field.type === 'number' ? 'number' : 'text'}
              value={(action.config as Record<string, string>)[field.name] || ''}
              onChange={(e) => onChange({ config: { ...action.config, [field.name]: e.target.value } })}
              placeholder={field.placeholder}
              className="w-full px-2 py-1.5 border rounded text-sm bg-white dark:bg-gray-800"
            />
          )}
        </div>
      ))}

      <div className="flex items-center gap-2">
        <span className="text-xs text-gray-500">Delay:</span>
        <input
          type="number"
          value={(action.delay || 0) / 1000}
          onChange={(e) => onChange({ delay: Number(e.target.value) * 1000 })}
          min={0}
          className="w-20 px-2 py-1 border rounded text-xs bg-white dark:bg-gray-800"
        />
        <span className="text-xs text-gray-500">seconds</span>
      </div>
    </div>
  );
}

// Main Workflow Builder
export function WorkflowBuilder({
  workflowId,
  onSave,
  onClose,
}: {
  workflowId?: string;
  onSave?: (workflow: Workflow) => void;
  onClose?: () => void;
}) {
  const { workflows, createWorkflow, updateWorkflow, templates } = useWorkflows();
  const { createCondition, createAction, validateWorkflow, testConditions } = useWorkflowBuilder();
  const { executeWorkflow } = useWorkflowExecution();

  // Load existing workflow or create new
  const existingWorkflow = workflowId ? workflows.find(w => w.id === workflowId) : null;

  const [name, setName] = useState(existingWorkflow?.name || 'New Workflow');
  const [description, setDescription] = useState(existingWorkflow?.description || '');
  const [isActive, setIsActive] = useState(existingWorkflow?.isActive ?? true);
  const [trigger, setTrigger] = useState<WorkflowTrigger>(
    existingWorkflow?.trigger || { type: 'email_received' }
  );
  const [conditions, setConditions] = useState<WorkflowCondition[]>(
    existingWorkflow?.conditions || []
  );
  const [actions, setActions] = useState<WorkflowAction[]>(
    existingWorkflow?.actions || []
  );

  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [showTemplates, setShowTemplates] = useState(false);
  const [testMode, setTestMode] = useState(false);
  const [testResult, setTestResult] = useState<'pass' | 'fail' | null>(null);

  const errors = useMemo(() => validateWorkflow({
    name,
    trigger,
    conditions,
    actions,
    isActive,
  }), [name, trigger, conditions, actions, isActive]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const data = e.dataTransfer.getData('application/json');
    if (!data) return;

    const item: DragItem = JSON.parse(data);

    if (item.type === 'trigger') {
      setTrigger({ type: (item.data as TriggerOption).type });
    } else if (item.type === 'condition') {
      const opt = item.data as ConditionOption;
      setConditions(prev => [...prev, createCondition(opt.field, opt.operators[0], '')]);
    } else if (item.type === 'action') {
      const opt = item.data as ActionOption;
      setActions(prev => [...prev, createAction(opt.type, {}, prev.length)]);
    }
  }, [createCondition, createAction]);

  const handleDragStart = (e: React.DragEvent, item: DragItem) => {
    e.dataTransfer.setData('application/json', JSON.stringify(item));
  };

  const handleSave = () => {
    if (errors.length > 0) return;

    const workflowData = {
      name,
      description,
      isActive,
      trigger,
      conditions,
      actions,
    };

    if (existingWorkflow) {
      updateWorkflow(existingWorkflow.id, workflowData);
      onSave?.(existingWorkflow);
    } else {
      const created = createWorkflow(workflowData);
      onSave?.(created);
    }
  };

  const handleTest = () => {
    const sampleContext = {
      email: {
        from: { email: 'test@example.com', name: 'Test User' },
        to: [{ email: 'me@mycelix.mail', name: 'Me' }],
        subject: 'Test email for workflow',
        body: 'This is a test email body with some content.',
        attachments: [],
        labels: [],
      },
      trustScore: 75,
    };

    const passed = testConditions(conditions, sampleContext);
    setTestResult(passed ? 'pass' : 'fail');
    setTestMode(true);
  };

  const loadTemplate = (templateId: string) => {
    const template = templates.find(t => t.id === templateId);
    if (!template) return;

    setName(template.workflow.name);
    setDescription(template.workflow.description || '');
    setTrigger(template.workflow.trigger);
    setConditions(template.workflow.conditions);
    setActions(template.workflow.actions);
    setShowTemplates(false);
  };

  return (
    <div className="flex h-full bg-gray-50 dark:bg-gray-900">
      {/* Palette Sidebar */}
      <div className="w-64 border-r bg-white dark:bg-gray-800 flex flex-col">
        <div className="p-4 border-b">
          <h3 className="font-semibold">Components</h3>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          {/* Triggers */}
          <div>
            <h4 className="text-xs font-semibold text-gray-500 uppercase mb-2">Triggers</h4>
            <div className="space-y-2">
              {triggerOptions.map(opt => (
                <PaletteItem key={opt.type} item={opt} type="trigger" onDragStart={handleDragStart} />
              ))}
            </div>
          </div>

          {/* Conditions */}
          <div>
            <h4 className="text-xs font-semibold text-gray-500 uppercase mb-2">Conditions</h4>
            <div className="space-y-2">
              {conditionOptions.slice(0, 6).map(opt => (
                <PaletteItem key={opt.field} item={opt} type="condition" onDragStart={handleDragStart} />
              ))}
            </div>
          </div>

          {/* Actions */}
          <div>
            <h4 className="text-xs font-semibold text-gray-500 uppercase mb-2">Actions</h4>
            <div className="space-y-2">
              {actionOptions.slice(0, 8).map(opt => (
                <PaletteItem key={opt.type} item={opt} type="action" onDragStart={handleDragStart} />
              ))}
            </div>
          </div>
        </div>

        <div className="p-4 border-t">
          <button
            onClick={() => setShowTemplates(true)}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 border rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700"
          >
            <Copy className="w-4 h-4" />
            Templates
          </button>
        </div>
      </div>

      {/* Canvas */}
      <div className="flex-1 flex flex-col">
        {/* Toolbar */}
        <div className="p-4 border-b bg-white dark:bg-gray-800 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="text-xl font-semibold bg-transparent border-none focus:outline-none focus:ring-2 focus:ring-blue-500 rounded px-2"
            />
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={isActive}
                onChange={(e) => setIsActive(e.target.checked)}
                className="rounded"
              />
              Active
            </label>
          </div>

          <div className="flex items-center gap-2">
            {errors.length > 0 && (
              <span className="flex items-center gap-1 text-red-500 text-sm">
                <AlertTriangle className="w-4 h-4" />
                {errors.length} error(s)
              </span>
            )}
            <button
              onClick={handleTest}
              className="flex items-center gap-2 px-4 py-2 border rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700"
            >
              <Play className="w-4 h-4" />
              Test
            </button>
            <button
              onClick={handleSave}
              disabled={errors.length > 0}
              className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
            >
              <Save className="w-4 h-4" />
              Save
            </button>
            {onClose && (
              <button onClick={onClose} className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded">
                <X className="w-5 h-5" />
              </button>
            )}
          </div>
        </div>

        {/* Drop Zone */}
        <div
          className="flex-1 p-8 overflow-y-auto"
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
        >
          <div className="max-w-2xl mx-auto space-y-6">
            {/* Trigger */}
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-gray-500 flex items-center gap-2">
                <Zap className="w-4 h-4" />
                When this happens...
              </h4>
              <WorkflowNode
                node={{ id: 'trigger', data: triggerOptions.find(t => t.type === trigger.type)! }}
                type="trigger"
                isSelected={selectedNode === 'trigger'}
                onSelect={() => setSelectedNode('trigger')}
                onDelete={() => {}}
                onConfigure={() => {}}
              />
            </div>

            {/* Conditions */}
            {conditions.length > 0 && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-gray-500 flex items-center gap-2">
                  <Filter className="w-4 h-4" />
                  If these conditions match...
                </h4>
                {conditions.map((cond, idx) => (
                  <ConditionEditor
                    key={cond.id}
                    condition={cond}
                    onChange={(updates) => {
                      setConditions(prev => prev.map((c, i) => i === idx ? { ...c, ...updates } : c));
                    }}
                    onDelete={() => {
                      setConditions(prev => prev.filter((_, i) => i !== idx));
                    }}
                  />
                ))}
              </div>
            )}

            {/* Add Condition Button */}
            <button
              onClick={() => setConditions(prev => [...prev, createCondition('from', 'contains', '')])}
              className="w-full p-3 border-2 border-dashed rounded-lg text-gray-500 hover:border-gray-400 hover:text-gray-600 flex items-center justify-center gap-2"
            >
              <Plus className="w-4 h-4" />
              Add Condition
            </button>

            {/* Actions */}
            {actions.length > 0 && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-gray-500 flex items-center gap-2">
                  <Play className="w-4 h-4" />
                  Then do this...
                </h4>
                {actions.map((action, idx) => {
                  const opt = actionOptions.find(a => a.type === action.type);
                  if (!opt) return null;
                  return (
                    <ActionEditor
                      key={action.id}
                      action={action}
                      actionOption={opt}
                      onChange={(updates) => {
                        setActions(prev => prev.map((a, i) => i === idx ? { ...a, ...updates } : a));
                      }}
                      onDelete={() => {
                        setActions(prev => prev.filter((_, i) => i !== idx));
                      }}
                    />
                  );
                })}
              </div>
            )}

            {/* Add Action Button */}
            <button
              onClick={() => setActions(prev => [...prev, createAction('mark_read', {}, prev.length)])}
              className="w-full p-3 border-2 border-dashed rounded-lg text-gray-500 hover:border-gray-400 hover:text-gray-600 flex items-center justify-center gap-2"
            >
              <Plus className="w-4 h-4" />
              Add Action
            </button>

            {/* Test Result */}
            {testMode && testResult && (
              <div className={`p-4 rounded-lg ${
                testResult === 'pass'
                  ? 'bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-200'
                  : 'bg-red-100 dark:bg-red-900/20 text-red-800 dark:text-red-200'
              }`}>
                <div className="flex items-center gap-2">
                  {testResult === 'pass' ? <CheckCircle className="w-5 h-5" /> : <XCircle className="w-5 h-5" />}
                  <span className="font-medium">
                    {testResult === 'pass' ? 'Test passed! Conditions matched.' : 'Test failed. Conditions did not match.'}
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Templates Modal */}
      {showTemplates && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl w-full max-w-2xl m-4 max-h-[80vh] flex flex-col">
            <div className="p-6 border-b flex items-center justify-between">
              <h2 className="text-xl font-semibold">Workflow Templates</h2>
              <button onClick={() => setShowTemplates(false)} className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded">
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-6">
              <div className="grid gap-4">
                {templates.map(template => (
                  <button
                    key={template.id}
                    onClick={() => loadTemplate(template.id)}
                    className="p-4 border rounded-lg text-left hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <h3 className="font-medium">{template.name}</h3>
                      <span className="text-xs bg-gray-100 dark:bg-gray-600 px-2 py-1 rounded">
                        {template.category}
                      </span>
                    </div>
                    <p className="text-sm text-gray-500 mt-1">{template.description}</p>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default WorkflowBuilder;
