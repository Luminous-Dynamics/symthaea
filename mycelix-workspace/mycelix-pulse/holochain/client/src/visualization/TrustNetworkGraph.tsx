// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Trust Network Graph Visualization
 *
 * D3.js-powered interactive graph visualization of the trust network
 * with MATL algorithm insights.
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as d3 from 'd3';

export interface TrustNode {
  id: string;
  name?: string;
  email?: string;
  trustLevel: number;
  attestationCount: number;
  isMe?: boolean;
  group?: string;
}

export interface TrustEdge {
  source: string;
  target: string;
  trustLevel: number;
  createdAt: number;
  context?: string;
}

export interface TrustNetworkGraphProps {
  nodes: TrustNode[];
  edges: TrustEdge[];
  width?: number;
  height?: number;
  onNodeClick?: (node: TrustNode) => void;
  onEdgeClick?: (edge: TrustEdge) => void;
  selectedNodeId?: string;
  showLabels?: boolean;
  showLegend?: boolean;
  colorScheme?: 'trust' | 'group' | 'activity';
}

const TRUST_COLORS = [
  '#ef4444', // 0.0-0.2: Red (untrusted)
  '#f97316', // 0.2-0.4: Orange (low trust)
  '#eab308', // 0.4-0.6: Yellow (medium)
  '#22c55e', // 0.6-0.8: Green (trusted)
  '#3b82f6', // 0.8-1.0: Blue (highly trusted)
];

const GROUP_COLORS = d3.schemeCategory10;

export const TrustNetworkGraph: React.FC<TrustNetworkGraphProps> = ({
  nodes,
  edges,
  width = 800,
  height = 600,
  onNodeClick,
  onEdgeClick,
  selectedNodeId,
  showLabels = true,
  showLegend = true,
  colorScheme = 'trust',
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<{
    visible: boolean;
    x: number;
    y: number;
    content: React.ReactNode;
  }>({ visible: false, x: 0, y: 0, content: null });
  const [zoom, setZoom] = useState(1);

  const getNodeColor = useCallback(
    (node: TrustNode): string => {
      if (node.isMe) return '#8b5cf6'; // Purple for self

      switch (colorScheme) {
        case 'trust':
          const idx = Math.min(Math.floor(node.trustLevel * 5), 4);
          return TRUST_COLORS[idx];
        case 'group':
          const groupIndex = node.group
            ? Array.from(new Set(nodes.map((n) => n.group))).indexOf(node.group)
            : 0;
          return GROUP_COLORS[groupIndex % GROUP_COLORS.length];
        case 'activity':
          // Color based on attestation count
          const activity = Math.min(node.attestationCount / 10, 1);
          return d3.interpolateBlues(0.3 + activity * 0.7);
        default:
          return '#64748b';
      }
    },
    [colorScheme, nodes]
  );

  const getEdgeColor = useCallback((edge: TrustEdge): string => {
    const idx = Math.min(Math.floor(edge.trustLevel * 5), 4);
    return TRUST_COLORS[idx];
  }, []);

  const getNodeRadius = useCallback((node: TrustNode): number => {
    if (node.isMe) return 20;
    // Size based on attestation count
    return Math.max(8, Math.min(16, 8 + node.attestationCount));
  }, []);

  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Create container for zoom
    const container = svg.append('g').attr('class', 'container');

    // Setup zoom behavior
    const zoomBehavior = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        container.attr('transform', event.transform);
        setZoom(event.transform.k);
      });

    svg.call(zoomBehavior);

    // Create force simulation
    const simulation = d3
      .forceSimulation(nodes as d3.SimulationNodeDatum[])
      .force(
        'link',
        d3
          .forceLink(edges)
          .id((d: any) => d.id)
          .distance((d: any) => 100 - d.trustLevel * 50)
          .strength((d: any) => d.trustLevel * 0.5)
      )
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius((d: any) => getNodeRadius(d) + 5));

    // Create arrow marker for directed edges
    svg
      .append('defs')
      .append('marker')
      .attr('id', 'arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 20)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#94a3b8');

    // Create edges
    const link = container
      .append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(edges)
      .enter()
      .append('line')
      .attr('stroke', (d) => getEdgeColor(d))
      .attr('stroke-width', (d) => 1 + d.trustLevel * 2)
      .attr('stroke-opacity', 0.6)
      .attr('marker-end', 'url(#arrow)')
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        event.stopPropagation();
        onEdgeClick?.(d);
      })
      .on('mouseenter', (event, d) => {
        setTooltip({
          visible: true,
          x: event.pageX,
          y: event.pageY,
          content: (
            <div className="text-sm">
              <div className="font-medium">Trust: {(d.trustLevel * 100).toFixed(0)}%</div>
              {d.context && <div className="text-gray-500">{d.context}</div>}
              <div className="text-xs text-gray-400">
                {new Date(d.createdAt / 1000).toLocaleDateString()}
              </div>
            </div>
          ),
        });
      })
      .on('mouseleave', () => {
        setTooltip((prev) => ({ ...prev, visible: false }));
      });

    // Create nodes
    const node = container
      .append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', 'node')
      .style('cursor', 'pointer')
      .call(
        d3
          .drag<SVGGElement, TrustNode>()
          .on('start', (event, d: any) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on('drag', (event, d: any) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on('end', (event, d: any) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      )
      .on('click', (event, d) => {
        event.stopPropagation();
        onNodeClick?.(d);
      })
      .on('mouseenter', (event, d) => {
        setTooltip({
          visible: true,
          x: event.pageX,
          y: event.pageY,
          content: (
            <div className="text-sm">
              <div className="font-medium">{d.name || d.email || d.id.slice(0, 8)}</div>
              <div>Trust: {(d.trustLevel * 100).toFixed(0)}%</div>
              <div className="text-gray-500">{d.attestationCount} attestations</div>
              {d.group && <div className="text-xs text-gray-400">Group: {d.group}</div>}
            </div>
          ),
        });
      })
      .on('mouseleave', () => {
        setTooltip((prev) => ({ ...prev, visible: false }));
      });

    // Add circles to nodes
    node
      .append('circle')
      .attr('r', (d) => getNodeRadius(d))
      .attr('fill', (d) => getNodeColor(d))
      .attr('stroke', (d) => (d.id === selectedNodeId ? '#1e40af' : '#fff'))
      .attr('stroke-width', (d) => (d.id === selectedNodeId ? 3 : 2));

    // Add rings for selected node
    node
      .filter((d) => d.id === selectedNodeId)
      .append('circle')
      .attr('r', (d) => getNodeRadius(d) + 6)
      .attr('fill', 'none')
      .attr('stroke', '#3b82f6')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '4,4');

    // Add labels if enabled
    if (showLabels) {
      node
        .append('text')
        .attr('dy', (d) => getNodeRadius(d) + 14)
        .attr('text-anchor', 'middle')
        .attr('font-size', '11px')
        .attr('fill', '#374151')
        .text((d) => d.name || d.email?.split('@')[0] || d.id.slice(0, 6));
    }

    // Add icon for self
    node
      .filter((d) => d.isMe)
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', 5)
      .attr('font-size', '14px')
      .attr('fill', '#fff')
      .text('★');

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });

    // Cleanup
    return () => {
      simulation.stop();
    };
  }, [
    nodes,
    edges,
    width,
    height,
    selectedNodeId,
    showLabels,
    getNodeColor,
    getEdgeColor,
    getNodeRadius,
    onNodeClick,
    onEdgeClick,
  ]);

  return (
    <div className="relative">
      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="bg-gray-50 rounded-lg border"
      />

      {/* Tooltip */}
      {tooltip.visible && (
        <div
          className="absolute bg-white shadow-lg rounded-lg p-3 pointer-events-none z-50 border"
          style={{
            left: tooltip.x + 10,
            top: tooltip.y - 10,
            transform: 'translate(0, -100%)',
          }}
        >
          {tooltip.content}
        </div>
      )}

      {/* Controls */}
      <div className="absolute bottom-4 right-4 flex items-center gap-2">
        <button
          onClick={() => {
            const svg = d3.select(svgRef.current);
            svg.transition().call(
              d3.zoom<SVGSVGElement, unknown>().transform as any,
              d3.zoomIdentity
            );
          }}
          className="p-2 bg-white rounded shadow hover:bg-gray-100"
          title="Reset zoom"
        >
          <ResetIcon />
        </button>
        <div className="px-2 py-1 bg-white rounded shadow text-sm">
          {Math.round(zoom * 100)}%
        </div>
      </div>

      {/* Legend */}
      {showLegend && colorScheme === 'trust' && (
        <div className="absolute top-4 left-4 bg-white rounded-lg shadow p-3">
          <div className="text-sm font-medium mb-2">Trust Level</div>
          <div className="space-y-1">
            {['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'].map((label, i) => (
              <div key={label} className="flex items-center gap-2">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: TRUST_COLORS[i] }}
                />
                <span className="text-xs">{label}</span>
              </div>
            ))}
            <div className="flex items-center gap-2 mt-2 pt-2 border-t">
              <div className="w-3 h-3 rounded-full bg-purple-500" />
              <span className="text-xs">You</span>
            </div>
          </div>
        </div>
      )}

      {/* Stats */}
      <div className="absolute top-4 right-4 bg-white rounded-lg shadow p-3">
        <div className="text-sm">
          <div className="flex justify-between gap-4">
            <span className="text-gray-500">Nodes:</span>
            <span className="font-medium">{nodes.length}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-gray-500">Connections:</span>
            <span className="font-medium">{edges.length}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-gray-500">Avg Trust:</span>
            <span className="font-medium">
              {nodes.length > 0
                ? (nodes.reduce((sum, n) => sum + n.trustLevel, 0) / nodes.length * 100).toFixed(0)
                : 0}%
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

// Icon component
const ResetIcon: React.FC = () => (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
    <path d="M3 3v5h5" />
  </svg>
);

export default TrustNetworkGraph;
