// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * BeliefGraph - Interactive Knowledge Graph Visualization
 *
 * Renders claims and their relationships as an interactive force-directed graph.
 * Supports zooming, panning, and click interactions.
 *
 * @example
 * ```tsx
 * import { BeliefGraph } from '@mycelix/knowledge-sdk/visualizations';
 *
 * <BeliefGraph
 *   claimId="uhCEk..."
 *   maxDepth={3}
 *   onNodeClick={(node) => console.log('Clicked:', node)}
 *   showCredibility={true}
 *   colorScheme="epistemic"
 * />
 * ```
 */

import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';

// ============================================================================
// Types
// ============================================================================

export interface GraphNode {
  id: string;
  label: string;
  type: 'claim' | 'author' | 'source' | 'market';
  credibility?: number;
  epistemicPosition?: {
    empirical: number;
    normative: number;
    mythic: number;
  };
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fixed?: boolean;
}

export interface GraphEdge {
  source: string;
  target: string;
  type: 'supports' | 'contradicts' | 'refines' | 'depends_on' | 'authored' | 'cites';
  weight: number;
  label?: string;
}

export interface BeliefGraphProps {
  /** Initial claim ID to center the graph on */
  claimId?: string;
  /** Pre-loaded nodes (alternative to claimId) */
  nodes?: GraphNode[];
  /** Pre-loaded edges (alternative to claimId) */
  edges?: GraphEdge[];
  /** Maximum depth to traverse from initial claim */
  maxDepth?: number;
  /** Width of the canvas (default: 100%) */
  width?: number | string;
  /** Height of the canvas (default: 600px) */
  height?: number | string;
  /** Show credibility scores on nodes */
  showCredibility?: boolean;
  /** Show edge labels */
  showEdgeLabels?: boolean;
  /** Color scheme: 'epistemic' | 'credibility' | 'type' | 'custom' */
  colorScheme?: 'epistemic' | 'credibility' | 'type' | 'custom';
  /** Custom color function when colorScheme is 'custom' */
  customColors?: (node: GraphNode) => string;
  /** Callback when a node is clicked */
  onNodeClick?: (node: GraphNode) => void;
  /** Callback when a node is hovered */
  onNodeHover?: (node: GraphNode | null) => void;
  /** Callback when an edge is clicked */
  onEdgeClick?: (edge: GraphEdge) => void;
  /** Enable physics simulation */
  enablePhysics?: boolean;
  /** Minimum zoom level */
  minZoom?: number;
  /** Maximum zoom level */
  maxZoom?: number;
  /** CSS class name */
  className?: string;
  /** Inline styles */
  style?: React.CSSProperties;
  /** Loading state */
  loading?: boolean;
  /** Error state */
  error?: Error | null;
}

interface SimulationState {
  nodes: GraphNode[];
  edges: GraphEdge[];
  transform: { x: number; y: number; scale: number };
  hoveredNode: GraphNode | null;
  selectedNode: GraphNode | null;
  isDragging: boolean;
  dragNode: GraphNode | null;
}

// ============================================================================
// Constants
// ============================================================================

const EDGE_COLORS: Record<GraphEdge['type'], string> = {
  supports: '#22c55e',      // green
  contradicts: '#ef4444',   // red
  refines: '#3b82f6',       // blue
  depends_on: '#f59e0b',    // amber
  authored: '#8b5cf6',      // purple
  cites: '#6b7280',         // gray
};

const NODE_TYPE_COLORS: Record<GraphNode['type'], string> = {
  claim: '#3b82f6',
  author: '#8b5cf6',
  source: '#f59e0b',
  market: '#22c55e',
};

const PHYSICS = {
  repulsion: 500,
  attraction: 0.01,
  damping: 0.9,
  centerForce: 0.01,
  minDistance: 50,
};

// ============================================================================
// Utility Functions
// ============================================================================

function getCredibilityColor(credibility: number): string {
  if (credibility >= 0.8) return '#22c55e';  // green
  if (credibility >= 0.6) return '#84cc16';  // lime
  if (credibility >= 0.4) return '#f59e0b';  // amber
  if (credibility >= 0.2) return '#f97316';  // orange
  return '#ef4444';                           // red
}

function getEpistemicColor(position: GraphNode['epistemicPosition']): string {
  if (!position) return '#6b7280';

  // Blend RGB based on E-N-M values
  const r = Math.round(255 * position.empirical);
  const g = Math.round(255 * position.normative);
  const b = Math.round(255 * position.mythic);

  return `rgb(${r}, ${g}, ${b})`;
}

function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength - 3) + '...';
}

// ============================================================================
// Physics Simulation
// ============================================================================

function applyForces(nodes: GraphNode[], edges: GraphEdge[], width: number, height: number): void {
  const centerX = width / 2;
  const centerY = height / 2;

  // Apply repulsion between all nodes
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const nodeA = nodes[i];
      const nodeB = nodes[j];

      const dx = (nodeB.x ?? 0) - (nodeA.x ?? 0);
      const dy = (nodeB.y ?? 0) - (nodeA.y ?? 0);
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;

      if (dist < PHYSICS.minDistance * 3) {
        const force = PHYSICS.repulsion / (dist * dist);
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;

        if (!nodeA.fixed) {
          nodeA.vx = (nodeA.vx ?? 0) - fx;
          nodeA.vy = (nodeA.vy ?? 0) - fy;
        }
        if (!nodeB.fixed) {
          nodeB.vx = (nodeB.vx ?? 0) + fx;
          nodeB.vy = (nodeB.vy ?? 0) + fy;
        }
      }
    }
  }

  // Apply attraction along edges
  for (const edge of edges) {
    const sourceNode = nodes.find(n => n.id === edge.source);
    const targetNode = nodes.find(n => n.id === edge.target);

    if (!sourceNode || !targetNode) continue;

    const dx = (targetNode.x ?? 0) - (sourceNode.x ?? 0);
    const dy = (targetNode.y ?? 0) - (sourceNode.y ?? 0);
    const dist = Math.sqrt(dx * dx + dy * dy) || 1;

    const force = dist * PHYSICS.attraction * edge.weight;
    const fx = (dx / dist) * force;
    const fy = (dy / dist) * force;

    if (!sourceNode.fixed) {
      sourceNode.vx = (sourceNode.vx ?? 0) + fx;
      sourceNode.vy = (sourceNode.vy ?? 0) + fy;
    }
    if (!targetNode.fixed) {
      targetNode.vx = (targetNode.vx ?? 0) - fx;
      targetNode.vy = (targetNode.vy ?? 0) - fy;
    }
  }

  // Apply center gravity and update positions
  for (const node of nodes) {
    if (node.fixed) continue;

    // Center force
    node.vx = (node.vx ?? 0) + (centerX - (node.x ?? 0)) * PHYSICS.centerForce;
    node.vy = (node.vy ?? 0) + (centerY - (node.y ?? 0)) * PHYSICS.centerForce;

    // Apply damping
    node.vx = (node.vx ?? 0) * PHYSICS.damping;
    node.vy = (node.vy ?? 0) * PHYSICS.damping;

    // Update position
    node.x = (node.x ?? centerX) + (node.vx ?? 0);
    node.y = (node.y ?? centerY) + (node.vy ?? 0);

    // Boundary constraints
    node.x = Math.max(50, Math.min(width - 50, node.x));
    node.y = Math.max(50, Math.min(height - 50, node.y));
  }
}

// ============================================================================
// Component
// ============================================================================

export function BeliefGraph({
  claimId,
  nodes: initialNodes = [],
  edges: initialEdges = [],
  maxDepth = 3,
  width = '100%',
  height = 600,
  showCredibility = true,
  showEdgeLabels = false,
  colorScheme = 'credibility',
  customColors,
  onNodeClick,
  onNodeHover,
  onEdgeClick,
  enablePhysics = true,
  minZoom = 0.1,
  maxZoom = 4,
  className,
  style,
  loading = false,
  error = null,
}: BeliefGraphProps): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animationRef = useRef<number>();

  const [state, setState] = useState<SimulationState>({
    nodes: initialNodes.map((n, i) => ({
      ...n,
      x: n.x ?? 300 + Math.cos(i * 0.5) * 200,
      y: n.y ?? 300 + Math.sin(i * 0.5) * 200,
      vx: 0,
      vy: 0,
    })),
    edges: initialEdges,
    transform: { x: 0, y: 0, scale: 1 },
    hoveredNode: null,
    selectedNode: null,
    isDragging: false,
    dragNode: null,
  });

  // Update nodes when props change
  useEffect(() => {
    if (initialNodes.length > 0) {
      setState(prev => ({
        ...prev,
        nodes: initialNodes.map((n, i) => ({
          ...n,
          x: n.x ?? 300 + Math.cos(i * 0.5) * 200,
          y: n.y ?? 300 + Math.sin(i * 0.5) * 200,
          vx: 0,
          vy: 0,
        })),
        edges: initialEdges,
      }));
    }
  }, [initialNodes, initialEdges]);

  // Get node color based on scheme
  const getNodeColor = useCallback((node: GraphNode): string => {
    switch (colorScheme) {
      case 'epistemic':
        return getEpistemicColor(node.epistemicPosition);
      case 'credibility':
        return getCredibilityColor(node.credibility ?? 0.5);
      case 'type':
        return NODE_TYPE_COLORS[node.type];
      case 'custom':
        return customColors?.(node) ?? '#6b7280';
      default:
        return '#6b7280';
    }
  }, [colorScheme, customColors]);

  // Get canvas dimensions
  const getDimensions = useCallback(() => {
    const container = containerRef.current;
    if (!container) return { width: 800, height: 600 };

    const w = typeof width === 'number' ? width : container.clientWidth;
    const h = typeof height === 'number' ? height : 600;

    return { width: w, height: h };
  }, [width, height]);

  // Render loop
  const render = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    const { width: w, height: h } = getDimensions();
    const { nodes, edges, transform, hoveredNode, selectedNode } = state;

    // Clear canvas
    ctx.clearRect(0, 0, w, h);

    // Apply transform
    ctx.save();
    ctx.translate(transform.x + w / 2, transform.y + h / 2);
    ctx.scale(transform.scale, transform.scale);
    ctx.translate(-w / 2, -h / 2);

    // Draw edges
    for (const edge of edges) {
      const sourceNode = nodes.find(n => n.id === edge.source);
      const targetNode = nodes.find(n => n.id === edge.target);

      if (!sourceNode || !targetNode) continue;

      const sx = sourceNode.x ?? 0;
      const sy = sourceNode.y ?? 0;
      const tx = targetNode.x ?? 0;
      const ty = targetNode.y ?? 0;

      // Draw line
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(tx, ty);
      ctx.strokeStyle = EDGE_COLORS[edge.type];
      ctx.lineWidth = Math.max(1, edge.weight * 3);
      ctx.globalAlpha = 0.6;
      ctx.stroke();
      ctx.globalAlpha = 1;

      // Draw arrow
      const angle = Math.atan2(ty - sy, tx - sx);
      const arrowSize = 10;
      const arrowX = tx - Math.cos(angle) * 25;
      const arrowY = ty - Math.sin(angle) * 25;

      ctx.beginPath();
      ctx.moveTo(arrowX, arrowY);
      ctx.lineTo(
        arrowX - arrowSize * Math.cos(angle - Math.PI / 6),
        arrowY - arrowSize * Math.sin(angle - Math.PI / 6)
      );
      ctx.lineTo(
        arrowX - arrowSize * Math.cos(angle + Math.PI / 6),
        arrowY - arrowSize * Math.sin(angle + Math.PI / 6)
      );
      ctx.closePath();
      ctx.fillStyle = EDGE_COLORS[edge.type];
      ctx.fill();

      // Draw edge label
      if (showEdgeLabels) {
        const midX = (sx + tx) / 2;
        const midY = (sy + ty) / 2;
        ctx.font = '10px sans-serif';
        ctx.fillStyle = '#6b7280';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(edge.label || edge.type, midX, midY - 10);
      }
    }

    // Draw nodes
    for (const node of nodes) {
      const x = node.x ?? 0;
      const y = node.y ?? 0;
      const radius = node.type === 'claim' ? 20 : 15;
      const isHovered = hoveredNode?.id === node.id;
      const isSelected = selectedNode?.id === node.id;

      // Draw node circle
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fillStyle = getNodeColor(node);
      ctx.fill();

      // Draw border
      if (isHovered || isSelected) {
        ctx.strokeStyle = isSelected ? '#3b82f6' : '#1f2937';
        ctx.lineWidth = 3;
        ctx.stroke();
      }

      // Draw credibility ring
      if (showCredibility && node.credibility !== undefined) {
        ctx.beginPath();
        ctx.arc(x, y, radius + 4, 0, Math.PI * 2 * node.credibility);
        ctx.strokeStyle = getCredibilityColor(node.credibility);
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Draw label
      ctx.font = isHovered ? 'bold 12px sans-serif' : '11px sans-serif';
      ctx.fillStyle = '#1f2937';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(truncateText(node.label, 20), x, y + radius + 5);

      // Draw type icon
      ctx.font = '10px sans-serif';
      ctx.fillStyle = '#ffffff';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const icon = node.type === 'claim' ? 'C' :
                   node.type === 'author' ? 'A' :
                   node.type === 'source' ? 'S' : 'M';
      ctx.fillText(icon, x, y);
    }

    ctx.restore();

    // Draw legend
    drawLegend(ctx, w);
  }, [state, getDimensions, getNodeColor, showCredibility, showEdgeLabels]);

  // Draw legend
  const drawLegend = (ctx: CanvasRenderingContext2D, width: number) => {
    const legendX = width - 150;
    const legendY = 20;

    ctx.font = 'bold 12px sans-serif';
    ctx.fillStyle = '#1f2937';
    ctx.textAlign = 'left';
    ctx.fillText('Relationships:', legendX, legendY);

    const types: GraphEdge['type'][] = ['supports', 'contradicts', 'refines', 'depends_on'];
    types.forEach((type, i) => {
      const y = legendY + 20 + i * 18;
      ctx.beginPath();
      ctx.moveTo(legendX, y);
      ctx.lineTo(legendX + 20, y);
      ctx.strokeStyle = EDGE_COLORS[type];
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.font = '11px sans-serif';
      ctx.fillStyle = '#6b7280';
      ctx.fillText(type.replace('_', ' '), legendX + 25, y + 4);
    });
  };

  // Physics simulation loop
  useEffect(() => {
    if (!enablePhysics || state.nodes.length === 0) return;

    const { width: w, height: h } = getDimensions();

    const tick = () => {
      setState(prev => {
        const newNodes = [...prev.nodes];
        applyForces(newNodes, prev.edges, w, h);
        return { ...prev, nodes: newNodes };
      });

      animationRef.current = requestAnimationFrame(tick);
    };

    animationRef.current = requestAnimationFrame(tick);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [enablePhysics, getDimensions, state.nodes.length]);

  // Render on state change
  useEffect(() => {
    render();
  }, [render]);

  // Mouse handlers
  const getMousePosition = useCallback((e: React.MouseEvent): { x: number; y: number } => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };

    const rect = canvas.getBoundingClientRect();
    const { width: w, height: h } = getDimensions();
    const { transform } = state;

    return {
      x: ((e.clientX - rect.left - transform.x - w / 2) / transform.scale) + w / 2,
      y: ((e.clientY - rect.top - transform.y - h / 2) / transform.scale) + h / 2,
    };
  }, [getDimensions, state]);

  const findNodeAtPosition = useCallback((x: number, y: number): GraphNode | null => {
    for (const node of state.nodes) {
      const nx = node.x ?? 0;
      const ny = node.y ?? 0;
      const radius = node.type === 'claim' ? 20 : 15;

      const dist = Math.sqrt((x - nx) ** 2 + (y - ny) ** 2);
      if (dist <= radius) return node;
    }
    return null;
  }, [state.nodes]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const pos = getMousePosition(e);

    if (state.isDragging && state.dragNode) {
      setState(prev => ({
        ...prev,
        nodes: prev.nodes.map(n =>
          n.id === prev.dragNode?.id
            ? { ...n, x: pos.x, y: pos.y, fixed: true }
            : n
        ),
      }));
      return;
    }

    const hovered = findNodeAtPosition(pos.x, pos.y);
    if (hovered !== state.hoveredNode) {
      setState(prev => ({ ...prev, hoveredNode: hovered }));
      onNodeHover?.(hovered);
    }
  }, [getMousePosition, findNodeAtPosition, state.isDragging, state.dragNode, state.hoveredNode, onNodeHover]);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    const pos = getMousePosition(e);
    const node = findNodeAtPosition(pos.x, pos.y);

    if (node) {
      setState(prev => ({
        ...prev,
        isDragging: true,
        dragNode: node,
        selectedNode: node,
      }));
    }
  }, [getMousePosition, findNodeAtPosition]);

  const handleMouseUp = useCallback(() => {
    if (state.dragNode && !state.isDragging) {
      onNodeClick?.(state.dragNode);
    }

    setState(prev => ({
      ...prev,
      isDragging: false,
      dragNode: null,
      nodes: prev.nodes.map(n => ({ ...n, fixed: false })),
    }));
  }, [state.dragNode, state.isDragging, onNodeClick]);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();

    const delta = e.deltaY > 0 ? 0.9 : 1.1;

    setState(prev => ({
      ...prev,
      transform: {
        ...prev.transform,
        scale: Math.max(minZoom, Math.min(maxZoom, prev.transform.scale * delta)),
      },
    }));
  }, [minZoom, maxZoom]);

  // Resize handler
  useEffect(() => {
    const handleResize = () => {
      const canvas = canvasRef.current;
      const container = containerRef.current;
      if (!canvas || !container) return;

      const { width: w, height: h } = getDimensions();
      canvas.width = w;
      canvas.height = h;
      render();
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [getDimensions, render]);

  // Loading and error states
  if (loading) {
    return (
      <div
        className={className}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          width,
          height,
          backgroundColor: '#f3f4f6',
          borderRadius: '8px',
          ...style,
        }}
      >
        <div style={{ textAlign: 'center', color: '#6b7280' }}>
          <div style={{
            width: '40px',
            height: '40px',
            border: '3px solid #e5e7eb',
            borderTopColor: '#3b82f6',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite',
            margin: '0 auto 16px',
          }} />
          Loading graph...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div
        className={className}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          width,
          height,
          backgroundColor: '#fef2f2',
          borderRadius: '8px',
          color: '#dc2626',
          ...style,
        }}
      >
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '24px', marginBottom: '8px' }}>Error</div>
          <div>{error.message}</div>
        </div>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={className}
      style={{
        width,
        height,
        position: 'relative',
        backgroundColor: '#ffffff',
        borderRadius: '8px',
        overflow: 'hidden',
        ...style,
      }}
    >
      <canvas
        ref={canvasRef}
        style={{ cursor: state.hoveredNode ? 'pointer' : 'default' }}
        onMouseMove={handleMouseMove}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      />

      {/* Tooltip */}
      {state.hoveredNode && (
        <div
          style={{
            position: 'absolute',
            left: (state.hoveredNode.x ?? 0) + 30,
            top: (state.hoveredNode.y ?? 0) - 20,
            backgroundColor: '#1f2937',
            color: '#ffffff',
            padding: '8px 12px',
            borderRadius: '6px',
            fontSize: '12px',
            pointerEvents: 'none',
            zIndex: 10,
            maxWidth: '200px',
          }}
        >
          <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>
            {state.hoveredNode.label}
          </div>
          <div style={{ color: '#9ca3af', textTransform: 'capitalize' }}>
            {state.hoveredNode.type}
          </div>
          {state.hoveredNode.credibility !== undefined && (
            <div style={{ marginTop: '4px' }}>
              Credibility: {(state.hoveredNode.credibility * 100).toFixed(1)}%
            </div>
          )}
          {state.hoveredNode.epistemicPosition && (
            <div style={{ marginTop: '4px', fontSize: '10px' }}>
              E: {(state.hoveredNode.epistemicPosition.empirical * 100).toFixed(0)}% |
              N: {(state.hoveredNode.epistemicPosition.normative * 100).toFixed(0)}% |
              M: {(state.hoveredNode.epistemicPosition.mythic * 100).toFixed(0)}%
            </div>
          )}
        </div>
      )}

      {/* Controls */}
      <div
        style={{
          position: 'absolute',
          bottom: '16px',
          left: '16px',
          display: 'flex',
          gap: '8px',
        }}
      >
        <button
          onClick={() => setState(prev => ({
            ...prev,
            transform: { ...prev.transform, scale: Math.min(maxZoom, prev.transform.scale * 1.2) },
          }))}
          style={{
            width: '32px',
            height: '32px',
            borderRadius: '6px',
            border: '1px solid #e5e7eb',
            backgroundColor: '#ffffff',
            cursor: 'pointer',
            fontSize: '16px',
          }}
        >
          +
        </button>
        <button
          onClick={() => setState(prev => ({
            ...prev,
            transform: { ...prev.transform, scale: Math.max(minZoom, prev.transform.scale / 1.2) },
          }))}
          style={{
            width: '32px',
            height: '32px',
            borderRadius: '6px',
            border: '1px solid #e5e7eb',
            backgroundColor: '#ffffff',
            cursor: 'pointer',
            fontSize: '16px',
          }}
        >
          -
        </button>
        <button
          onClick={() => setState(prev => ({
            ...prev,
            transform: { x: 0, y: 0, scale: 1 },
          }))}
          style={{
            height: '32px',
            padding: '0 12px',
            borderRadius: '6px',
            border: '1px solid #e5e7eb',
            backgroundColor: '#ffffff',
            cursor: 'pointer',
            fontSize: '12px',
          }}
        >
          Reset
        </button>
      </div>

      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

export default BeliefGraph;
