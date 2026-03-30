// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Web of Trust Visualization
 *
 * Interactive force-directed graph showing trust relationships
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as d3 from 'd3';

interface TrustNode {
  id: string;
  name: string;
  email: string;
  trustScore: number;
  type: 'self' | 'direct' | 'indirect' | 'unknown';
  attestationCount: number;
}

interface TrustLink {
  source: string;
  target: string;
  strength: number;
  context: string;
  bidirectional: boolean;
}

interface TrustGraphData {
  nodes: TrustNode[];
  links: TrustLink[];
}

interface TrustGraphProps {
  data: TrustGraphData;
  width?: number;
  height?: number;
  onNodeClick?: (node: TrustNode) => void;
  onNodeHover?: (node: TrustNode | null) => void;
  selectedNodeId?: string;
  showLabels?: boolean;
  colorScheme?: 'trust' | 'type' | 'activity';
}

export function TrustGraph({
  data,
  width = 800,
  height = 600,
  onNodeClick,
  onNodeHover,
  selectedNodeId,
  showLabels = true,
  colorScheme = 'trust',
}: TrustGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoveredNode, setHoveredNode] = useState<TrustNode | null>(null);
  const [transform, setTransform] = useState(d3.zoomIdentity);

  // Color scales
  const trustColorScale = d3.scaleSequential(d3.interpolateRdYlGn).domain([0, 1]);

  const typeColorMap: Record<string, string> = {
    self: '#3b82f6',      // Blue
    direct: '#22c55e',    // Green
    indirect: '#eab308',  // Yellow
    unknown: '#6b7280',   // Gray
  };

  const getNodeColor = useCallback((node: TrustNode) => {
    switch (colorScheme) {
      case 'trust':
        return trustColorScale(node.trustScore);
      case 'type':
        return typeColorMap[node.type] || '#6b7280';
      case 'activity':
        const activity = Math.min(node.attestationCount / 10, 1);
        return d3.interpolateBlues(0.3 + activity * 0.7);
      default:
        return trustColorScale(node.trustScore);
    }
  }, [colorScheme]);

  const getNodeRadius = useCallback((node: TrustNode) => {
    if (node.type === 'self') return 20;
    const baseRadius = 8;
    const trustBonus = node.trustScore * 8;
    const attestationBonus = Math.min(node.attestationCount, 5) * 1.5;
    return baseRadius + trustBonus + attestationBonus;
  }, []);

  useEffect(() => {
    if (!svgRef.current || !data.nodes.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Create container for zoom
    const container = svg.append('g');

    // Set up zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.2, 4])
      .on('zoom', (event) => {
        container.attr('transform', event.transform);
        setTransform(event.transform);
      });

    svg.call(zoom);

    // Prepare data for D3
    const nodes = data.nodes.map(d => ({ ...d }));
    const links = data.links.map(d => ({ ...d }));

    // Create force simulation
    const simulation = d3.forceSimulation(nodes as any)
      .force('link', d3.forceLink(links as any)
        .id((d: any) => d.id)
        .distance(d => 100 - (d as any).strength * 50)
        .strength(d => (d as any).strength * 0.5))
      .force('charge', d3.forceManyBody()
        .strength(-200)
        .distanceMax(300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide()
        .radius((d: any) => getNodeRadius(d) + 5));

    // Create arrow marker for directed links
    svg.append('defs').selectAll('marker')
      .data(['arrow'])
      .join('marker')
      .attr('id', 'arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 20)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('fill', '#999')
      .attr('d', 'M0,-5L10,0L0,5');

    // Create links
    const link = container.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', d => d.bidirectional ? '#22c55e' : '#999')
      .attr('stroke-opacity', d => 0.3 + d.strength * 0.5)
      .attr('stroke-width', d => 1 + d.strength * 3)
      .attr('marker-end', d => d.bidirectional ? null : 'url(#arrow)');

    // Create node groups
    const node = container.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .attr('cursor', 'pointer')
      .call(d3.drag<SVGGElement, any>()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        }) as any);

    // Add circles to nodes
    node.append('circle')
      .attr('r', d => getNodeRadius(d))
      .attr('fill', d => getNodeColor(d))
      .attr('stroke', d => d.id === selectedNodeId ? '#000' : '#fff')
      .attr('stroke-width', d => d.id === selectedNodeId ? 3 : 2)
      .on('click', (event, d) => {
        event.stopPropagation();
        onNodeClick?.(d);
      })
      .on('mouseenter', (event, d) => {
        setHoveredNode(d);
        onNodeHover?.(d);
      })
      .on('mouseleave', () => {
        setHoveredNode(null);
        onNodeHover?.(null);
      });

    // Add labels
    if (showLabels) {
      node.append('text')
        .text(d => d.name || d.email.split('@')[0])
        .attr('x', d => getNodeRadius(d) + 5)
        .attr('y', 4)
        .attr('font-size', '12px')
        .attr('fill', '#374151')
        .attr('pointer-events', 'none');
    }

    // Add trust score indicator
    node.append('text')
      .text(d => `${Math.round(d.trustScore * 100)}%`)
      .attr('text-anchor', 'middle')
      .attr('y', d => getNodeRadius(d) + 15)
      .attr('font-size', '10px')
      .attr('fill', '#6b7280')
      .attr('pointer-events', 'none');

    // Update positions on tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => (d.source as any).x)
        .attr('y1', d => (d.source as any).y)
        .attr('x2', d => (d.target as any).x)
        .attr('y2', d => (d.target as any).y);

      node.attr('transform', d => `translate(${(d as any).x},${(d as any).y})`);
    });

    // Cleanup
    return () => {
      simulation.stop();
    };
  }, [data, width, height, selectedNodeId, showLabels, getNodeColor, getNodeRadius, onNodeClick, onNodeHover]);

  return (
    <div className="trust-graph relative">
      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="bg-gray-50 rounded-lg border border-gray-200"
      />

      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-white p-3 rounded-lg shadow-md text-sm">
        <div className="font-medium mb-2">Trust Levels</div>
        <div className="flex items-center gap-2 mb-1">
          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: trustColorScale(1) }} />
          <span>High Trust (80-100%)</span>
        </div>
        <div className="flex items-center gap-2 mb-1">
          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: trustColorScale(0.5) }} />
          <span>Medium Trust (40-80%)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: trustColorScale(0) }} />
          <span>Low Trust (0-40%)</span>
        </div>
      </div>

      {/* Hover tooltip */}
      {hoveredNode && (
        <div className="absolute top-4 right-4 bg-white p-4 rounded-lg shadow-lg max-w-xs">
          <div className="font-medium text-lg">{hoveredNode.name || 'Unknown'}</div>
          <div className="text-gray-500 text-sm mb-2">{hoveredNode.email}</div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <span className="text-gray-500">Trust Score:</span>
              <span className="ml-1 font-medium">
                {Math.round(hoveredNode.trustScore * 100)}%
              </span>
            </div>
            <div>
              <span className="text-gray-500">Attestations:</span>
              <span className="ml-1 font-medium">{hoveredNode.attestationCount}</span>
            </div>
            <div>
              <span className="text-gray-500">Connection:</span>
              <span className="ml-1 font-medium capitalize">{hoveredNode.type}</span>
            </div>
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="absolute top-4 left-4 flex gap-2">
        <button
          onClick={() => {
            const svg = d3.select(svgRef.current);
            svg.transition().duration(500).call(
              d3.zoom<SVGSVGElement, unknown>().transform as any,
              d3.zoomIdentity
            );
          }}
          className="px-3 py-1 bg-white rounded shadow hover:bg-gray-50 text-sm"
        >
          Reset View
        </button>
        <select
          className="px-3 py-1 bg-white rounded shadow text-sm"
          defaultValue={colorScheme}
        >
          <option value="trust">Color by Trust</option>
          <option value="type">Color by Type</option>
          <option value="activity">Color by Activity</option>
        </select>
      </div>
    </div>
  );
}

// Hook for fetching trust graph data
export function useTrustGraph(userId?: string) {
  const [data, setData] = useState<TrustGraphData>({ nodes: [], links: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    async function fetchGraph() {
      try {
        setLoading(true);
        const response = await fetch(`/api/v1/trust/graph${userId ? `?user_id=${userId}` : ''}`);
        if (!response.ok) throw new Error('Failed to fetch trust graph');
        const graphData = await response.json();
        setData(graphData);
      } catch (err) {
        setError(err as Error);
      } finally {
        setLoading(false);
      }
    }

    fetchGraph();
  }, [userId]);

  return { data, loading, error };
}

export default TrustGraph;
