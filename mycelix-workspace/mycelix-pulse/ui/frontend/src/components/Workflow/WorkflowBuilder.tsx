// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Track V: Workflow Automation Engine
 *
 * Visual workflow builder, conditional routing, scheduled actions,
 * email sequences, and cross-app integrations.
 */

import React, { useState, useCallback, useRef, useEffect, useMemo } from 'react';

// ============================================================================
// Types
// ============================================================================

interface Workflow {
  id: string;
  name: string;
  description?: string;
  trigger: WorkflowTrigger;
  nodes: WorkflowNode[];
  connections: NodeConnection[];
  enabled: boolean;
  runCount: number;
  lastRun?: string;
  createdAt: string;
  updatedAt: string;
}

interface WorkflowTrigger {
  type: string;
  filters?: TriggerFilter[];
  cron?: string;
  label?: string;
}

interface TriggerFilter {
  field: string;
  operator: string;
  value: string;
}

interface WorkflowNode {
  id: string;
  nodeType: NodeType;
  position: { x: number; y: number };
  config: Record<string, any>;
}

type NodeType =
  | { type: 'Condition'; conditions: TriggerFilter[]; logic: 'And' | 'Or' }
  | { type: 'MoveTo'; folder: string }
  | { type: 'AddLabel'; label: string }
  | { type: 'RemoveLabel'; label: string }
  | { type: 'MarkRead'; read: boolean }
  | { type: 'Star'; starred: boolean }
  | { type: 'Archive' }
  | { type: 'Delete'; permanent: boolean }
  | { type: 'Forward'; to: string[]; includeAttachments: boolean }
  | { type: 'Reply'; templateId?: string; body?: string; replyAll: boolean }
  | { type: 'Delay'; duration: { type: string; value: number } }
  | { type: 'Webhook'; url: string; method: string }
  | { type: 'SlackNotify'; channel: string; message: string }
  | { type: 'MatrixNotify'; roomId: string; message: string }
  | { type: 'CreateTask'; integration: string; title: string };

interface NodeConnection {
  fromNode: string;
  fromPort: string;
  toNode: string;
  toPort: string;
}

interface WorkflowExecution {
  id: string;
  workflowId: string;
  status: 'Running' | 'Completed' | 'Failed' | 'Waiting';
  currentNode?: string;
  logs: ExecutionLog[];
  startedAt: string;
  completedAt?: string;
  error?: string;
}

interface ExecutionLog {
  timestamp: string;
  nodeId: string;
  action: string;
  result: { type: 'Success' | 'Skipped' | 'Failed'; message?: string };
  durationMs: number;
}

// ============================================================================
// Node Definitions
// ============================================================================

const NODE_TYPES = {
  triggers: [
    { type: 'EmailReceived', label: 'Email Received', icon: '📧', color: '#3b82f6' },
    { type: 'EmailSent', label: 'Email Sent', icon: '📤', color: '#3b82f6' },
    { type: 'Schedule', label: 'Schedule', icon: '⏰', color: '#3b82f6' },
    { type: 'Webhook', label: 'Webhook', icon: '🔗', color: '#3b82f6' },
    { type: 'LabelAdded', label: 'Label Added', icon: '🏷️', color: '#3b82f6' },
  ],
  conditions: [
    { type: 'Condition', label: 'Condition', icon: '❓', color: '#eab308' },
    { type: 'Switch', label: 'Switch', icon: '🔀', color: '#eab308' },
    { type: 'Filter', label: 'Filter', icon: '🔍', color: '#eab308' },
  ],
  actions: [
    { type: 'MoveTo', label: 'Move to Folder', icon: '📁', color: '#22c55e' },
    { type: 'AddLabel', label: 'Add Label', icon: '🏷️', color: '#22c55e' },
    { type: 'RemoveLabel', label: 'Remove Label', icon: '🏷️', color: '#22c55e' },
    { type: 'MarkRead', label: 'Mark Read/Unread', icon: '👁️', color: '#22c55e' },
    { type: 'Star', label: 'Star/Unstar', icon: '⭐', color: '#22c55e' },
    { type: 'Archive', label: 'Archive', icon: '📦', color: '#22c55e' },
    { type: 'Delete', label: 'Delete', icon: '🗑️', color: '#ef4444' },
    { type: 'Forward', label: 'Forward', icon: '↪️', color: '#22c55e' },
    { type: 'Reply', label: 'Reply', icon: '↩️', color: '#22c55e' },
  ],
  timing: [
    { type: 'Delay', label: 'Delay', icon: '⏳', color: '#a855f7' },
    { type: 'ScheduleSend', label: 'Schedule Send', icon: '📅', color: '#a855f7' },
    { type: 'WaitFor', label: 'Wait For', icon: '⏸️', color: '#a855f7' },
  ],
  integrations: [
    { type: 'WebhookAction', label: 'Webhook', icon: '🔗', color: '#ec4899' },
    { type: 'SlackNotify', label: 'Slack', icon: '💬', color: '#ec4899' },
    { type: 'MatrixNotify', label: 'Matrix', icon: '💬', color: '#ec4899' },
    { type: 'CreateTask', label: 'Create Task', icon: '✅', color: '#ec4899' },
    { type: 'CreateEvent', label: 'Create Event', icon: '📅', color: '#ec4899' },
  ],
};

// ============================================================================
// Hooks
// ============================================================================

function useWorkflows() {
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/workflows')
      .then(res => res.json())
      .then(setWorkflows)
      .finally(() => setLoading(false));
  }, []);

  const createWorkflow = useCallback(async (workflow: Partial<Workflow>) => {
    const response = await fetch('/api/workflows', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(workflow),
    });
    const newWorkflow = await response.json();
    setWorkflows(prev => [...prev, newWorkflow]);
    return newWorkflow;
  }, []);

  const updateWorkflow = useCallback(async (id: string, updates: Partial<Workflow>) => {
    await fetch(`/api/workflows/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
    setWorkflows(prev => prev.map(w => w.id === id ? { ...w, ...updates } : w));
  }, []);

  const deleteWorkflow = useCallback(async (id: string) => {
    await fetch(`/api/workflows/${id}`, { method: 'DELETE' });
    setWorkflows(prev => prev.filter(w => w.id !== id));
  }, []);

  const toggleWorkflow = useCallback(async (id: string, enabled: boolean) => {
    await fetch(`/api/workflows/${id}/toggle`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    });
    setWorkflows(prev => prev.map(w => w.id === id ? { ...w, enabled } : w));
  }, []);

  return { workflows, loading, createWorkflow, updateWorkflow, deleteWorkflow, toggleWorkflow };
}

function useWorkflowEditor(initialWorkflow?: Workflow) {
  const [nodes, setNodes] = useState<WorkflowNode[]>(initialWorkflow?.nodes || []);
  const [connections, setConnections] = useState<NodeConnection[]>(initialWorkflow?.connections || []);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const addNode = useCallback((type: string, position: { x: number; y: number }) => {
    const id = `node_${Date.now()}`;
    const newNode: WorkflowNode = {
      id,
      nodeType: { type } as any,
      position,
      config: {},
    };
    setNodes(prev => [...prev, newNode]);
    setSelectedNode(id);
    return id;
  }, []);

  const updateNode = useCallback((id: string, updates: Partial<WorkflowNode>) => {
    setNodes(prev => prev.map(n => n.id === id ? { ...n, ...updates } : n));
  }, []);

  const deleteNode = useCallback((id: string) => {
    setNodes(prev => prev.filter(n => n.id !== id));
    setConnections(prev => prev.filter(c => c.fromNode !== id && c.toNode !== id));
    if (selectedNode === id) setSelectedNode(null);
  }, [selectedNode]);

  const addConnection = useCallback((from: { node: string; port: string }, to: { node: string; port: string }) => {
    const connection: NodeConnection = {
      fromNode: from.node,
      fromPort: from.port,
      toNode: to.node,
      toPort: to.port,
    };
    setConnections(prev => [...prev, connection]);
  }, []);

  const deleteConnection = useCallback((fromNode: string, toNode: string) => {
    setConnections(prev => prev.filter(c => !(c.fromNode === fromNode && c.toNode === toNode)));
  }, []);

  return {
    nodes,
    connections,
    selectedNode,
    isDragging,
    addNode,
    updateNode,
    deleteNode,
    addConnection,
    deleteConnection,
    setSelectedNode,
    setIsDragging,
  };
}

// ============================================================================
// Components
// ============================================================================

interface WorkflowCanvasProps {
  nodes: WorkflowNode[];
  connections: NodeConnection[];
  selectedNode: string | null;
  onNodeSelect: (id: string | null) => void;
  onNodeMove: (id: string, position: { x: number; y: number }) => void;
  onNodeDelete: (id: string) => void;
  onConnectionAdd: (from: { node: string; port: string }, to: { node: string; port: string }) => void;
  onDrop: (type: string, position: { x: number; y: number }) => void;
}

function WorkflowCanvas({
  nodes,
  connections,
  selectedNode,
  onNodeSelect,
  onNodeMove,
  onNodeDelete,
  onConnectionAdd,
  onDrop,
}: WorkflowCanvasProps) {
  const canvasRef = useRef<HTMLDivElement>(null);
  const [draggingNode, setDraggingNode] = useState<string | null>(null);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [connectingFrom, setConnectingFrom] = useState<{ node: string; port: string } | null>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const type = e.dataTransfer.getData('nodeType');
    if (!type || !canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const position = {
      x: e.clientX - rect.left - 75,
      y: e.clientY - rect.top - 25,
    };
    onDrop(type, position);
  };

  const handleNodeMouseDown = (e: React.MouseEvent, nodeId: string) => {
    if ((e.target as HTMLElement).classList.contains('port')) return;

    const node = nodes.find(n => n.id === nodeId);
    if (!node) return;

    setDraggingNode(nodeId);
    setDragOffset({
      x: e.clientX - node.position.x,
      y: e.clientY - node.position.y,
    });
    onNodeSelect(nodeId);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!draggingNode) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    onNodeMove(draggingNode, {
      x: e.clientX - rect.left - 75,
      y: e.clientY - rect.top - 25,
    });
  };

  const handleMouseUp = () => {
    setDraggingNode(null);
    setConnectingFrom(null);
  };

  const handlePortClick = (nodeId: string, portType: 'input' | 'output') => {
    if (portType === 'output') {
      setConnectingFrom({ node: nodeId, port: 'out' });
    } else if (connectingFrom && connectingFrom.node !== nodeId) {
      onConnectionAdd(connectingFrom, { node: nodeId, port: 'in' });
      setConnectingFrom(null);
    }
  };

  const getNodeDef = (type: string) => {
    for (const category of Object.values(NODE_TYPES)) {
      const def = category.find(n => n.type === type);
      if (def) return def;
    }
    return { icon: '❓', color: '#888', label: type };
  };

  return (
    <div
      ref={canvasRef}
      className="workflow-canvas"
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onClick={() => onNodeSelect(null)}
    >
      <svg className="connections-layer">
        {connections.map((conn, i) => {
          const fromNode = nodes.find(n => n.id === conn.fromNode);
          const toNode = nodes.find(n => n.id === conn.toNode);
          if (!fromNode || !toNode) return null;

          const x1 = fromNode.position.x + 150;
          const y1 = fromNode.position.y + 25;
          const x2 = toNode.position.x;
          const y2 = toNode.position.y + 25;
          const cx = (x1 + x2) / 2;

          return (
            <g key={i}>
              <path
                d={`M ${x1} ${y1} C ${cx} ${y1}, ${cx} ${y2}, ${x2} ${y2}`}
                fill="none"
                stroke="#666"
                strokeWidth={2}
              />
              <circle
                cx={cx}
                cy={(y1 + y2) / 2}
                r={8}
                fill="#333"
                className="connection-delete"
                onClick={() => onNodeSelect(null)}
              />
            </g>
          );
        })}
      </svg>

      {nodes.map(node => {
        const def = getNodeDef((node.nodeType as any).type);
        return (
          <div
            key={node.id}
            className={`workflow-node ${selectedNode === node.id ? 'selected' : ''}`}
            style={{
              left: node.position.x,
              top: node.position.y,
              borderColor: def.color,
            }}
            onMouseDown={e => handleNodeMouseDown(e, node.id)}
            onClick={e => e.stopPropagation()}
          >
            <div
              className="port input"
              onClick={() => handlePortClick(node.id, 'input')}
            />
            <div className="node-content">
              <span className="icon">{def.icon}</span>
              <span className="label">{def.label}</span>
            </div>
            <div
              className="port output"
              onClick={() => handlePortClick(node.id, 'output')}
            />
            <button
              className="delete-btn"
              onClick={(e) => {
                e.stopPropagation();
                onNodeDelete(node.id);
              }}
            >
              ×
            </button>
          </div>
        );
      })}

      {connectingFrom && (
        <div className="connecting-hint">Click on an input port to connect</div>
      )}
    </div>
  );
}

function NodePalette() {
  const handleDragStart = (e: React.DragEvent, type: string) => {
    e.dataTransfer.setData('nodeType', type);
  };

  return (
    <div className="node-palette">
      {Object.entries(NODE_TYPES).map(([category, nodeTypes]) => (
        <div key={category} className="palette-category">
          <h4>{category.charAt(0).toUpperCase() + category.slice(1)}</h4>
          <div className="node-list">
            {nodeTypes.map(nodeDef => (
              <div
                key={nodeDef.type}
                className="palette-node"
                draggable
                onDragStart={e => handleDragStart(e, nodeDef.type)}
                style={{ borderLeftColor: nodeDef.color }}
              >
                <span className="icon">{nodeDef.icon}</span>
                <span className="label">{nodeDef.label}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

interface NodeConfigPanelProps {
  node: WorkflowNode;
  onUpdate: (updates: Partial<WorkflowNode>) => void;
}

function NodeConfigPanel({ node, onUpdate }: NodeConfigPanelProps) {
  const nodeType = (node.nodeType as any).type;

  const renderConfig = () => {
    switch (nodeType) {
      case 'MoveTo':
        return (
          <div className="config-field">
            <label>Folder</label>
            <select
              value={node.config.folder || ''}
              onChange={e => onUpdate({ config: { ...node.config, folder: e.target.value } })}
            >
              <option value="">Select folder</option>
              <option value="Inbox">Inbox</option>
              <option value="Archive">Archive</option>
              <option value="Trash">Trash</option>
              <option value="Spam">Spam</option>
            </select>
          </div>
        );

      case 'AddLabel':
      case 'RemoveLabel':
        return (
          <div className="config-field">
            <label>Label</label>
            <input
              type="text"
              value={node.config.label || ''}
              onChange={e => onUpdate({ config: { ...node.config, label: e.target.value } })}
              placeholder="Enter label name"
            />
          </div>
        );

      case 'Condition':
        return (
          <ConditionConfig
            conditions={node.config.conditions || []}
            logic={node.config.logic || 'And'}
            onChange={(conditions, logic) => onUpdate({
              config: { ...node.config, conditions, logic }
            })}
          />
        );

      case 'Delay':
        return (
          <div className="config-fields">
            <div className="config-field">
              <label>Duration</label>
              <input
                type="number"
                value={node.config.value || 1}
                onChange={e => onUpdate({
                  config: { ...node.config, value: parseInt(e.target.value) }
                })}
              />
            </div>
            <div className="config-field">
              <label>Unit</label>
              <select
                value={node.config.unit || 'Hours'}
                onChange={e => onUpdate({ config: { ...node.config, unit: e.target.value } })}
              >
                <option value="Minutes">Minutes</option>
                <option value="Hours">Hours</option>
                <option value="Days">Days</option>
              </select>
            </div>
          </div>
        );

      case 'Forward':
        return (
          <div className="config-fields">
            <div className="config-field">
              <label>Forward To</label>
              <input
                type="email"
                value={node.config.to || ''}
                onChange={e => onUpdate({ config: { ...node.config, to: e.target.value } })}
                placeholder="email@example.com"
              />
            </div>
            <div className="config-field checkbox">
              <label>
                <input
                  type="checkbox"
                  checked={node.config.includeAttachments ?? true}
                  onChange={e => onUpdate({
                    config: { ...node.config, includeAttachments: e.target.checked }
                  })}
                />
                Include attachments
              </label>
            </div>
          </div>
        );

      case 'SlackNotify':
        return (
          <div className="config-fields">
            <div className="config-field">
              <label>Channel</label>
              <input
                type="text"
                value={node.config.channel || ''}
                onChange={e => onUpdate({ config: { ...node.config, channel: e.target.value } })}
                placeholder="#channel-name"
              />
            </div>
            <div className="config-field">
              <label>Message Template</label>
              <textarea
                value={node.config.message || ''}
                onChange={e => onUpdate({ config: { ...node.config, message: e.target.value } })}
                placeholder="New email from {{email.from}}: {{email.subject}}"
              />
            </div>
          </div>
        );

      case 'WebhookAction':
        return (
          <div className="config-fields">
            <div className="config-field">
              <label>URL</label>
              <input
                type="url"
                value={node.config.url || ''}
                onChange={e => onUpdate({ config: { ...node.config, url: e.target.value } })}
                placeholder="https://..."
              />
            </div>
            <div className="config-field">
              <label>Method</label>
              <select
                value={node.config.method || 'POST'}
                onChange={e => onUpdate({ config: { ...node.config, method: e.target.value } })}
              >
                <option value="GET">GET</option>
                <option value="POST">POST</option>
                <option value="PUT">PUT</option>
                <option value="PATCH">PATCH</option>
              </select>
            </div>
          </div>
        );

      default:
        return <p className="no-config">No configuration needed</p>;
    }
  };

  return (
    <div className="node-config-panel">
      <h3>Configure Node</h3>
      <div className="node-type">{nodeType}</div>
      {renderConfig()}
    </div>
  );
}

interface ConditionConfigProps {
  conditions: TriggerFilter[];
  logic: 'And' | 'Or';
  onChange: (conditions: TriggerFilter[], logic: 'And' | 'Or') => void;
}

function ConditionConfig({ conditions, logic, onChange }: ConditionConfigProps) {
  const addCondition = () => {
    onChange([...conditions, { field: 'From', operator: 'Contains', value: '' }], logic);
  };

  const updateCondition = (index: number, updates: Partial<TriggerFilter>) => {
    const newConditions = conditions.map((c, i) => i === index ? { ...c, ...updates } : c);
    onChange(newConditions, logic);
  };

  const removeCondition = (index: number) => {
    onChange(conditions.filter((_, i) => i !== index), logic);
  };

  return (
    <div className="condition-config">
      <div className="logic-toggle">
        <button
          className={logic === 'And' ? 'active' : ''}
          onClick={() => onChange(conditions, 'And')}
        >
          AND
        </button>
        <button
          className={logic === 'Or' ? 'active' : ''}
          onClick={() => onChange(conditions, 'Or')}
        >
          OR
        </button>
      </div>

      {conditions.map((condition, i) => (
        <div key={i} className="condition-row">
          <select
            value={condition.field}
            onChange={e => updateCondition(i, { field: e.target.value })}
          >
            <option value="From">From</option>
            <option value="To">To</option>
            <option value="Subject">Subject</option>
            <option value="Body">Body</option>
            <option value="HasAttachment">Has Attachment</option>
            <option value="Label">Label</option>
          </select>
          <select
            value={condition.operator}
            onChange={e => updateCondition(i, { operator: e.target.value })}
          >
            <option value="Contains">Contains</option>
            <option value="Equals">Equals</option>
            <option value="StartsWith">Starts With</option>
            <option value="EndsWith">Ends With</option>
            <option value="Matches">Matches (Regex)</option>
          </select>
          <input
            type="text"
            value={condition.value}
            onChange={e => updateCondition(i, { value: e.target.value })}
            placeholder="Value"
          />
          <button onClick={() => removeCondition(i)}>×</button>
        </div>
      ))}

      <button className="add-condition" onClick={addCondition}>+ Add Condition</button>
    </div>
  );
}

interface WorkflowListProps {
  workflows: Workflow[];
  onSelect: (id: string) => void;
  onToggle: (id: string, enabled: boolean) => void;
  onDelete: (id: string) => void;
  onCreate: () => void;
}

function WorkflowList({ workflows, onSelect, onToggle, onDelete, onCreate }: WorkflowListProps) {
  return (
    <div className="workflow-list">
      <header>
        <h2>Workflows</h2>
        <button onClick={onCreate}>+ New Workflow</button>
      </header>

      {workflows.length === 0 ? (
        <div className="empty-state">
          <p>No workflows yet. Create your first automation!</p>
          <button onClick={onCreate}>Create Workflow</button>
        </div>
      ) : (
        <div className="list">
          {workflows.map(workflow => (
            <div key={workflow.id} className="workflow-card" onClick={() => onSelect(workflow.id)}>
              <div className="header">
                <h3>{workflow.name}</h3>
                <label className="toggle" onClick={e => e.stopPropagation()}>
                  <input
                    type="checkbox"
                    checked={workflow.enabled}
                    onChange={e => onToggle(workflow.id, e.target.checked)}
                  />
                  <span className="slider" />
                </label>
              </div>
              {workflow.description && <p className="description">{workflow.description}</p>}
              <div className="meta">
                <span className="trigger">
                  Trigger: {workflow.trigger.type}
                </span>
                <span className="runs">{workflow.runCount} runs</span>
                {workflow.lastRun && (
                  <span className="last-run">
                    Last: {new Date(workflow.lastRun).toLocaleDateString()}
                  </span>
                )}
              </div>
              <button
                className="delete-btn"
                onClick={e => {
                  e.stopPropagation();
                  onDelete(workflow.id);
                }}
              >
                Delete
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function WorkflowBuilder() {
  const { workflows, loading, createWorkflow, updateWorkflow, deleteWorkflow, toggleWorkflow } = useWorkflows();
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<string | null>(null);
  const [isEditing, setIsEditing] = useState(false);

  const selectedWorkflow = workflows.find(w => w.id === selectedWorkflowId);

  const editor = useWorkflowEditor(selectedWorkflow);

  const handleSave = async () => {
    if (!selectedWorkflowId) {
      await createWorkflow({
        name: 'New Workflow',
        trigger: { type: 'EmailReceived' },
        nodes: editor.nodes,
        connections: editor.connections,
        enabled: false,
      });
    } else {
      await updateWorkflow(selectedWorkflowId, {
        nodes: editor.nodes,
        connections: editor.connections,
      });
    }
    setIsEditing(false);
  };

  const handleCreate = () => {
    setSelectedWorkflowId(null);
    setIsEditing(true);
  };

  if (loading) {
    return <div className="workflow-builder loading">Loading workflows...</div>;
  }

  if (!isEditing) {
    return (
      <WorkflowList
        workflows={workflows}
        onSelect={id => {
          setSelectedWorkflowId(id);
          setIsEditing(true);
        }}
        onToggle={toggleWorkflow}
        onDelete={deleteWorkflow}
        onCreate={handleCreate}
      />
    );
  }

  const selectedNode = editor.nodes.find(n => n.id === editor.selectedNode);

  return (
    <div className="workflow-builder editing">
      <header className="builder-header">
        <button onClick={() => setIsEditing(false)}>← Back</button>
        <input
          type="text"
          className="workflow-name"
          defaultValue={selectedWorkflow?.name || 'New Workflow'}
          placeholder="Workflow Name"
        />
        <button onClick={handleSave} className="save-btn">Save</button>
      </header>

      <div className="builder-layout">
        <NodePalette />

        <WorkflowCanvas
          nodes={editor.nodes}
          connections={editor.connections}
          selectedNode={editor.selectedNode}
          onNodeSelect={editor.setSelectedNode}
          onNodeMove={(id, pos) => editor.updateNode(id, { position: pos })}
          onNodeDelete={editor.deleteNode}
          onConnectionAdd={editor.addConnection}
          onDrop={editor.addNode}
        />

        {selectedNode && (
          <NodeConfigPanel
            node={selectedNode}
            onUpdate={updates => editor.updateNode(selectedNode.id, updates)}
          />
        )}
      </div>
    </div>
  );
}

export default WorkflowBuilder;
