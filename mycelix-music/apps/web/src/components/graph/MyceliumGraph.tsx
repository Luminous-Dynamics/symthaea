// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useRef, useEffect, useState, useCallback } from 'react';
import * as d3 from 'd3';
import {
  ZoomIn,
  ZoomOut,
  Maximize2,
  RefreshCw,
  Filter,
  Search,
  Info,
  Music2,
  Users,
  Heart,
} from 'lucide-react';

export type NodeType = 'artist' | 'listener' | 'song' | 'genre' | 'event';

export interface GraphNode {
  id: string;
  type: NodeType;
  name: string;
  avatar?: string;
  size: number; // Node importance/size
  connections: number;
  data?: Record<string, any>;
}

export interface GraphLink {
  source: string;
  target: string;
  type: 'listen' | 'collaborate' | 'patron' | 'influence' | 'similar';
  strength: number; // 0-1
}

interface MyceliumGraphProps {
  nodes: GraphNode[];
  links: GraphLink[];
  width?: number;
  height?: number;
  onNodeClick?: (node: GraphNode) => void;
  onLinkClick?: (link: GraphLink) => void;
  highlightedNodeId?: string;
}

const NODE_COLORS: Record<NodeType, string> = {
  artist: '#8B5CF6', // Purple
  listener: '#10B981', // Green
  song: '#F59E0B', // Amber
  genre: '#EC4899', // Pink
  event: '#3B82F6', // Blue
};

const LINK_COLORS: Record<GraphLink['type'], string> = {
  listen: '#10B981',
  collaborate: '#8B5CF6',
  patron: '#F59E0B',
  influence: '#EC4899',
  similar: '#6B7280',
};

export function MyceliumGraph({
  nodes,
  links,
  width = 800,
  height = 600,
  onNodeClick,
  onLinkClick,
  highlightedNodeId,
}: MyceliumGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [zoom, setZoom] = useState(1);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<NodeType | 'all'>('all');

  // D3 simulation
  const simulationRef = useRef<d3.Simulation<GraphNode, GraphLink> | null>(null);

  // Filter nodes
  const filteredNodes = nodes.filter((node) => {
    const matchesFilter = filterType === 'all' || node.type === filterType;
    const matchesSearch = !searchQuery ||
      node.name.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesFilter && matchesSearch;
  });

  const filteredNodeIds = new Set(filteredNodes.map((n) => n.id));
  const filteredLinks = links.filter(
    (link) => filteredNodeIds.has(link.source as string) && filteredNodeIds.has(link.target as string)
  );

  // Initialize D3 visualization
  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Create container for zoom
    const g = svg.append('g');

    // Zoom behavior
    const zoomBehavior = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
        setZoom(event.transform.k);
      });

    svg.call(zoomBehavior);

    // Create simulation
    const simulation = d3.forceSimulation<GraphNode>(filteredNodes)
      .force('link', d3.forceLink<GraphNode, GraphLink>(filteredLinks)
        .id((d) => d.id)
        .distance((d) => 100 / d.strength)
        .strength((d) => d.strength * 0.5))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius((d: any) => d.size * 3 + 10));

    simulationRef.current = simulation;

    // Create gradient definitions
    const defs = svg.append('defs');

    // Glow filter
    const filter = defs.append('filter')
      .attr('id', 'glow')
      .attr('width', '300%')
      .attr('height', '300%')
      .attr('x', '-100%')
      .attr('y', '-100%');

    filter.append('feGaussianBlur')
      .attr('stdDeviation', '3')
      .attr('result', 'coloredBlur');

    const feMerge = filter.append('feMerge');
    feMerge.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    // Create links
    const link = g.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(filteredLinks)
      .enter()
      .append('line')
      .attr('stroke', (d) => LINK_COLORS[d.type])
      .attr('stroke-width', (d) => d.strength * 3)
      .attr('stroke-opacity', 0.4)
      .attr('class', 'cursor-pointer')
      .on('click', (event, d) => {
        event.stopPropagation();
        onLinkClick?.(d);
      })
      .on('mouseenter', function() {
        d3.select(this).attr('stroke-opacity', 0.8);
      })
      .on('mouseleave', function() {
        d3.select(this).attr('stroke-opacity', 0.4);
      });

    // Create node groups
    const node = g.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(filteredNodes)
      .enter()
      .append('g')
      .attr('class', 'cursor-pointer')
      .call(d3.drag<SVGGElement, GraphNode>()
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
        }) as any)
      .on('click', (event, d) => {
        event.stopPropagation();
        setSelectedNode(d);
        onNodeClick?.(d);
      })
      .on('mouseenter', (event, d) => {
        setHoveredNode(d);

        // Highlight connected nodes and links
        const connectedNodeIds = new Set<string>();
        filteredLinks.forEach((l) => {
          const sourceId = typeof l.source === 'string' ? l.source : (l.source as any).id;
          const targetId = typeof l.target === 'string' ? l.target : (l.target as any).id;
          if (sourceId === d.id) connectedNodeIds.add(targetId);
          if (targetId === d.id) connectedNodeIds.add(sourceId);
        });

        node.style('opacity', (n) =>
          n.id === d.id || connectedNodeIds.has(n.id) ? 1 : 0.3
        );
        link.style('opacity', (l) => {
          const sourceId = typeof l.source === 'string' ? l.source : (l.source as any).id;
          const targetId = typeof l.target === 'string' ? l.target : (l.target as any).id;
          return sourceId === d.id || targetId === d.id ? 1 : 0.1;
        });
      })
      .on('mouseleave', () => {
        setHoveredNode(null);
        node.style('opacity', 1);
        link.style('opacity', 0.4);
      });

    // Node circles
    node.append('circle')
      .attr('r', (d) => d.size * 2 + 8)
      .attr('fill', (d) => NODE_COLORS[d.type])
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .attr('filter', (d) => d.id === highlightedNodeId ? 'url(#glow)' : null);

    // Node labels
    node.append('text')
      .text((d) => d.name.slice(0, 12) + (d.name.length > 12 ? '...' : ''))
      .attr('text-anchor', 'middle')
      .attr('dy', (d) => d.size * 2 + 24)
      .attr('fill', '#fff')
      .attr('font-size', '12px')
      .attr('font-weight', 'bold')
      .attr('pointer-events', 'none');

    // Type icon
    node.append('text')
      .text((d) => {
        switch (d.type) {
          case 'artist': return '🎤';
          case 'listener': return '👤';
          case 'song': return '🎵';
          case 'genre': return '🏷️';
          case 'event': return '🎪';
        }
      })
      .attr('text-anchor', 'middle')
      .attr('dy', 5)
      .attr('font-size', (d) => d.size + 12)
      .attr('pointer-events', 'none');

    // Update positions on tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });

    // Click background to deselect
    svg.on('click', () => {
      setSelectedNode(null);
    });

    return () => {
      simulation.stop();
    };
  }, [filteredNodes, filteredLinks, width, height, highlightedNodeId, onNodeClick, onLinkClick]);

  // Zoom controls
  const handleZoomIn = () => {
    if (svgRef.current) {
      const svg = d3.select(svgRef.current);
      svg.transition().call(
        d3.zoom<SVGSVGElement, unknown>().scaleBy as any,
        1.5
      );
    }
  };

  const handleZoomOut = () => {
    if (svgRef.current) {
      const svg = d3.select(svgRef.current);
      svg.transition().call(
        d3.zoom<SVGSVGElement, unknown>().scaleBy as any,
        0.67
      );
    }
  };

  const handleReset = () => {
    if (svgRef.current) {
      const svg = d3.select(svgRef.current);
      svg.transition().call(
        d3.zoom<SVGSVGElement, unknown>().transform as any,
        d3.zoomIdentity
      );
    }
  };

  return (
    <div ref={containerRef} className="relative bg-gray-900 rounded-2xl overflow-hidden">
      {/* Controls */}
      <div className="absolute top-4 left-4 z-10 flex flex-col gap-2">
        {/* Search */}
        <div className="flex items-center gap-2 px-3 py-2 bg-black/50 backdrop-blur-sm rounded-lg">
          <Search className="w-4 h-4 text-muted-foreground" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search nodes..."
            className="bg-transparent border-none outline-none text-sm w-32"
          />
        </div>

        {/* Filter */}
        <div className="flex items-center gap-2 px-3 py-2 bg-black/50 backdrop-blur-sm rounded-lg">
          <Filter className="w-4 h-4 text-muted-foreground" />
          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value as NodeType | 'all')}
            className="bg-transparent border-none outline-none text-sm"
          >
            <option value="all">All Types</option>
            <option value="artist">Artists</option>
            <option value="listener">Listeners</option>
            <option value="song">Songs</option>
            <option value="genre">Genres</option>
            <option value="event">Events</option>
          </select>
        </div>
      </div>

      {/* Zoom controls */}
      <div className="absolute top-4 right-4 z-10 flex flex-col gap-1 bg-black/50 backdrop-blur-sm rounded-lg p-1">
        <button
          onClick={handleZoomIn}
          className="p-2 hover:bg-white/10 rounded"
          title="Zoom In"
        >
          <ZoomIn className="w-4 h-4" />
        </button>
        <button
          onClick={handleZoomOut}
          className="p-2 hover:bg-white/10 rounded"
          title="Zoom Out"
        >
          <ZoomOut className="w-4 h-4" />
        </button>
        <button
          onClick={handleReset}
          className="p-2 hover:bg-white/10 rounded"
          title="Reset View"
        >
          <Maximize2 className="w-4 h-4" />
        </button>
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 z-10 flex flex-wrap gap-3 bg-black/50 backdrop-blur-sm rounded-lg p-3">
        {Object.entries(NODE_COLORS).map(([type, color]) => (
          <div key={type} className="flex items-center gap-2 text-xs">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: color }}
            />
            <span className="capitalize">{type}s</span>
          </div>
        ))}
      </div>

      {/* Stats */}
      <div className="absolute bottom-4 right-4 z-10 flex items-center gap-4 bg-black/50 backdrop-blur-sm rounded-lg px-4 py-2 text-xs text-muted-foreground">
        <span>{filteredNodes.length} nodes</span>
        <span>{filteredLinks.length} connections</span>
        <span>Zoom: {(zoom * 100).toFixed(0)}%</span>
      </div>

      {/* SVG Container */}
      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="cursor-grab active:cursor-grabbing"
      />

      {/* Node Detail Panel */}
      {selectedNode && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 bg-gray-800 rounded-xl p-4 shadow-xl border border-white/10 w-80">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-3">
              <div
                className="w-10 h-10 rounded-full flex items-center justify-center text-xl"
                style={{ backgroundColor: NODE_COLORS[selectedNode.type] }}
              >
                {selectedNode.type === 'artist' && '🎤'}
                {selectedNode.type === 'listener' && '👤'}
                {selectedNode.type === 'song' && '🎵'}
                {selectedNode.type === 'genre' && '🏷️'}
                {selectedNode.type === 'event' && '🎪'}
              </div>
              <div>
                <h4 className="font-semibold">{selectedNode.name}</h4>
                <p className="text-xs text-muted-foreground capitalize">{selectedNode.type}</p>
              </div>
            </div>
            <button
              onClick={() => setSelectedNode(null)}
              className="p-1 hover:bg-white/10 rounded"
            >
              ×
            </button>
          </div>

          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="bg-white/5 rounded-lg p-2">
              <p className="text-muted-foreground text-xs">Connections</p>
              <p className="font-semibold">{selectedNode.connections}</p>
            </div>
            <div className="bg-white/5 rounded-lg p-2">
              <p className="text-muted-foreground text-xs">Influence</p>
              <p className="font-semibold">{selectedNode.size}</p>
            </div>
          </div>

          <button className="w-full mt-3 py-2 bg-purple-500 rounded-lg text-sm font-medium hover:bg-purple-600 transition-colors">
            View Profile
          </button>
        </div>
      )}

      {/* Hover tooltip */}
      {hoveredNode && !selectedNode && (
        <div className="absolute pointer-events-none z-30 bg-gray-800 rounded-lg px-3 py-2 shadow-lg text-sm"
          style={{
            left: '50%',
            top: '10%',
            transform: 'translateX(-50%)',
          }}
        >
          <span className="font-medium">{hoveredNode.name}</span>
          <span className="text-muted-foreground ml-2 capitalize">({hoveredNode.type})</span>
        </div>
      )}
    </div>
  );
}
