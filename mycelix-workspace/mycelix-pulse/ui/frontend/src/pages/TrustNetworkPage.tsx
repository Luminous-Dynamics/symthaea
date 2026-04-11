// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Trust Network Page
 *
 * Visualizes the trust network with interactive graph
 */

import React, { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useI18n } from '../lib/i18n';
import { useAccessibility } from '../lib/a11y/AccessibilityProvider';
import { graphqlClient } from '../lib/api/graphql-client';
import { useTrustScore, useAttestations } from '../lib/api/holochain-client';

// Types
interface TrustNode {
  id: string;
  label: string;
  email: string;
  trustScore: number;
  x?: number;
  y?: number;
}

interface TrustEdge {
  from: string;
  to: string;
  weight: number;
  context: string;
}

interface TrustAttestation {
  id: string;
  fromAgentId: string;
  toAgentId: string;
  trustLevel: number;
  context: string;
  createdAt: string;
  expiresAt?: string;
}

// Trust Badge
function TrustBadge({ score }: { score: number }) {
  const getColor = (s: number) => {
    if (s >= 0.8) return '#4CAF50';
    if (s >= 0.5) return '#FFC107';
    if (s >= 0.2) return '#FF9800';
    return '#F44336';
  };

  const getLabel = (s: number) => {
    if (s >= 0.8) return 'Trusted';
    if (s >= 0.5) return 'Known';
    if (s >= 0.2) return 'Caution';
    return 'Unknown';
  };

  return (
    <div
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '8px',
        padding: '4px 12px',
        borderRadius: '16px',
        backgroundColor: getColor(score),
        color: 'white',
        fontWeight: 'bold',
        fontSize: '14px',
      }}
    >
      {score >= 0.8 ? '✓' : score >= 0.5 ? '~' : score >= 0.2 ? '!' : '?'}
      <span>{getLabel(score)} ({Math.round(score * 100)}%)</span>
    </div>
  );
}

// Simple Force-Directed Graph Component
function TrustGraph({
  nodes,
  edges,
  selectedNode,
  onNodeSelect,
}: {
  nodes: TrustNode[];
  edges: TrustEdge[];
  selectedNode: string | null;
  onNodeSelect: (id: string | null) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  // Simple force simulation
  const simulatedNodes = useMemo(() => {
    const centerX = dimensions.width / 2;
    const centerY = dimensions.height / 2;
    const radius = Math.min(dimensions.width, dimensions.height) * 0.35;

    return nodes.map((node, i) => {
      const angle = (2 * Math.PI * i) / nodes.length;
      return {
        ...node,
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle),
      };
    });
  }, [nodes, dimensions]);

  // Draw graph
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear
    ctx.clearRect(0, 0, dimensions.width, dimensions.height);

    // Draw edges
    edges.forEach((edge) => {
      const fromNode = simulatedNodes.find((n) => n.id === edge.from);
      const toNode = simulatedNodes.find((n) => n.id === edge.to);

      if (fromNode && toNode && fromNode.x && fromNode.y && toNode.x && toNode.y) {
        ctx.beginPath();
        ctx.moveTo(fromNode.x, fromNode.y);
        ctx.lineTo(toNode.x, toNode.y);
        ctx.strokeStyle = `rgba(100, 100, 100, ${edge.weight})`;
        ctx.lineWidth = 1 + edge.weight * 2;
        ctx.stroke();
      }
    });

    // Draw nodes
    simulatedNodes.forEach((node) => {
      if (!node.x || !node.y) return;

      const isSelected = node.id === selectedNode;
      const nodeRadius = isSelected ? 25 : 20;

      // Node circle
      ctx.beginPath();
      ctx.arc(node.x, node.y, nodeRadius, 0, 2 * Math.PI);

      // Color based on trust
      if (node.trustScore >= 0.8) ctx.fillStyle = '#4CAF50';
      else if (node.trustScore >= 0.5) ctx.fillStyle = '#FFC107';
      else if (node.trustScore >= 0.2) ctx.fillStyle = '#FF9800';
      else ctx.fillStyle = '#F44336';

      ctx.fill();

      if (isSelected) {
        ctx.strokeStyle = '#1976d2';
        ctx.lineWidth = 3;
        ctx.stroke();
      }

      // Label
      ctx.fillStyle = '#333';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(node.label, node.x, node.y + nodeRadius + 15);
    });
  }, [simulatedNodes, edges, selectedNode, dimensions]);

  // Handle click
  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      // Find clicked node
      const clickedNode = simulatedNodes.find((node) => {
        if (!node.x || !node.y) return false;
        const dx = node.x - x;
        const dy = node.y - y;
        return Math.sqrt(dx * dx + dy * dy) < 25;
      });

      onNodeSelect(clickedNode?.id || null);
    },
    [simulatedNodes, onNodeSelect]
  );

  // Resize handler
  useEffect(() => {
    const container = canvasRef.current?.parentElement;
    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setDimensions({ width: Math.floor(width), height: Math.floor(height) });
    });

    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  return (
    <canvas
      ref={canvasRef}
      width={dimensions.width}
      height={dimensions.height}
      onClick={handleClick}
      role="img"
      aria-label="Trust network graph visualization"
      style={{ width: '100%', height: '100%', cursor: 'pointer' }}
    />
  );
}

// Attestation Form
function CreateAttestationForm({
  onSubmit,
  onCancel,
}: {
  onSubmit: (data: { toAgentId: string; trustLevel: number; context: string }) => void;
  onCancel: () => void;
}) {
  const { t } = useI18n();
  const [toAgentId, setToAgentId] = useState('');
  const [trustLevel, setTrustLevel] = useState(0.5);
  const [context, setContext] = useState('general');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({ toAgentId, trustLevel, context });
  };

  return (
    <form onSubmit={handleSubmit} style={{ padding: '16px', backgroundColor: '#f5f5f5', borderRadius: '8px' }}>
      <h3 style={{ margin: '0 0 16px' }}>{t('trust.createAttestation')}</h3>

      <div style={{ marginBottom: '12px' }}>
        <label htmlFor="to-agent" style={{ display: 'block', marginBottom: '4px' }}>
          Agent/Email
        </label>
        <input
          id="to-agent"
          type="text"
          required
          value={toAgentId}
          onChange={(e) => setToAgentId(e.target.value)}
          placeholder="email@example.com or agent ID"
          style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }}
        />
      </div>

      <div style={{ marginBottom: '12px' }}>
        <label htmlFor="trust-level" style={{ display: 'block', marginBottom: '4px' }}>
          Trust Level: {Math.round(trustLevel * 100)}%
        </label>
        <input
          id="trust-level"
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={trustLevel}
          onChange={(e) => setTrustLevel(parseFloat(e.target.value))}
          style={{ width: '100%' }}
        />
      </div>

      <div style={{ marginBottom: '16px' }}>
        <label htmlFor="context" style={{ display: 'block', marginBottom: '4px' }}>
          Context
        </label>
        <select
          id="context"
          value={context}
          onChange={(e) => setContext(e.target.value)}
          style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }}
        >
          <option value="general">General</option>
          <option value="professional">Professional</option>
          <option value="personal">Personal</option>
          <option value="verified">Verified Identity</option>
        </select>
      </div>

      <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
        <button type="button" onClick={onCancel} style={{ padding: '8px 16px' }}>
          {t('common.cancel')}
        </button>
        <button
          type="submit"
          style={{
            padding: '8px 16px',
            backgroundColor: '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
          }}
        >
          Create Attestation
        </button>
      </div>
    </form>
  );
}

// Main Page
export default function TrustNetworkPage() {
  const { t } = useI18n();
  const { announce } = useAccessibility();
  const queryClient = useQueryClient();

  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [showCreateForm, setShowCreateForm] = useState(false);

  // Fetch trust network data
  const { data: networkData, isLoading } = useQuery({
    queryKey: ['trust-network'],
    queryFn: async () => {
      const response = await graphqlClient.query<{
        trustNetwork: { nodes: TrustNode[]; edges: TrustEdge[] };
      }>(
        `query GetTrustNetwork {
          trustNetwork {
            nodes { id label email trustScore }
            edges { from to weight context }
          }
        }`,
        {}
      );
      return response.trustNetwork;
    },
  });

  // Fetch attestations for selected node
  const { data: attestations } = useAttestations(selectedNode || undefined);

  // Create attestation mutation
  const createAttestationMutation = useMutation({
    mutationFn: async (data: { toAgentId: string; trustLevel: number; context: string }) => {
      return graphqlClient.mutate(
        `mutation CreateAttestation($input: CreateAttestationInput!) {
          createTrustAttestation(input: $input) { id }
        }`,
        { input: data }
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['trust-network'] });
      announce('Attestation created successfully');
      setShowCreateForm(false);
    },
  });

  // Revoke attestation
  const revokeAttestationMutation = useMutation({
    mutationFn: async (id: string) => {
      return graphqlClient.mutate(
        `mutation RevokeAttestation($id: ID!) { revokeTrustAttestation(id: $id) }`,
        { id }
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['trust-network'] });
      announce('Attestation revoked');
    },
  });

  const selectedNodeData = useMemo(
    () => networkData?.nodes.find((n) => n.id === selectedNode),
    [networkData, selectedNode]
  );

  if (isLoading) {
    return (
      <div role="status" style={{ padding: '2rem', textAlign: 'center' }}>
        {t('common.loading')}
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', height: 'calc(100vh - 64px)' }}>
      {/* Graph Panel */}
      <div style={{ flex: 1, position: 'relative', backgroundColor: '#fafafa' }}>
        <div style={{ position: 'absolute', top: 16, left: 16, zIndex: 10 }}>
          <h1 style={{ margin: 0, fontSize: '24px' }}>Trust Network</h1>
          <p style={{ margin: '8px 0 0', color: '#666' }}>
            {networkData?.nodes.length || 0} nodes, {networkData?.edges.length || 0} connections
          </p>
        </div>

        <TrustGraph
          nodes={networkData?.nodes || []}
          edges={networkData?.edges || []}
          selectedNode={selectedNode}
          onNodeSelect={setSelectedNode}
        />

        {/* Legend */}
        <div
          style={{
            position: 'absolute',
            bottom: 16,
            left: 16,
            backgroundColor: 'white',
            padding: '12px',
            borderRadius: '8px',
            boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          }}
        >
          <div style={{ fontWeight: 'bold', marginBottom: '8px' }}>Trust Levels</div>
          <div style={{ display: 'flex', gap: '12px', fontSize: '12px' }}>
            <span><span style={{ color: '#4CAF50' }}>●</span> Trusted (≥80%)</span>
            <span><span style={{ color: '#FFC107' }}>●</span> Known (50-79%)</span>
            <span><span style={{ color: '#FF9800' }}>●</span> Caution (20-49%)</span>
            <span><span style={{ color: '#F44336' }}>●</span> Unknown (&lt;20%)</span>
          </div>
        </div>
      </div>

      {/* Detail Panel */}
      <div
        style={{
          width: '360px',
          borderLeft: '1px solid #e0e0e0',
          padding: '24px',
          overflow: 'auto',
        }}
      >
        {selectedNodeData ? (
          <>
            <h2 style={{ margin: '0 0 8px' }}>{selectedNodeData.label}</h2>
            <p style={{ margin: '0 0 16px', color: '#666' }}>{selectedNodeData.email}</p>

            <TrustBadge score={selectedNodeData.trustScore} />

            <h3 style={{ marginTop: '24px' }}>{t('trust.attestations')}</h3>

            {attestations && attestations.length > 0 ? (
              <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                {attestations.map((att: TrustAttestation) => (
                  <li
                    key={att.id}
                    style={{
                      padding: '12px',
                      marginBottom: '8px',
                      backgroundColor: '#f5f5f5',
                      borderRadius: '4px',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>{Math.round(att.trustLevel * 100)}% - {att.context}</span>
                      <button
                        onClick={() => revokeAttestationMutation.mutate(att.id)}
                        style={{
                          background: 'none',
                          border: 'none',
                          color: '#d32f2f',
                          cursor: 'pointer',
                          fontSize: '12px',
                        }}
                      >
                        Revoke
                      </button>
                    </div>
                    <div style={{ fontSize: '12px', color: '#666', marginTop: '4px' }}>
                      {new Date(att.createdAt).toLocaleDateString()}
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <p style={{ color: '#666' }}>No attestations yet</p>
            )}

            <button
              onClick={() => setShowCreateForm(true)}
              style={{
                marginTop: '16px',
                padding: '10px 16px',
                backgroundColor: '#1976d2',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                width: '100%',
              }}
            >
              + {t('trust.createAttestation')}
            </button>
          </>
        ) : (
          <div style={{ color: '#666', textAlign: 'center', paddingTop: '2rem' }}>
            <p>Select a node to view details</p>
            <button
              onClick={() => setShowCreateForm(true)}
              style={{
                marginTop: '16px',
                padding: '10px 16px',
                backgroundColor: '#1976d2',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              + {t('trust.createAttestation')}
            </button>
          </div>
        )}

        {showCreateForm && (
          <div style={{ marginTop: '24px' }}>
            <CreateAttestationForm
              onSubmit={(data) => createAttestationMutation.mutate(data)}
              onCancel={() => setShowCreateForm(false)}
            />
          </div>
        )}
      </div>
    </div>
  );
}
