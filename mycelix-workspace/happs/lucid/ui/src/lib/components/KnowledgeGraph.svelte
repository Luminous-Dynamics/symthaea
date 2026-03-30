<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import * as d3 from 'd3';
  import type { Thought } from '@mycelix/lucid-client';
  import { thoughts, selectedThought } from '../stores/thoughts';

  export let width = 800;
  export let height = 600;

  let container: HTMLDivElement;
  let svg: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  let simulation: d3.Simulation<GraphNode, GraphLink>;

  interface GraphNode extends d3.SimulationNodeDatum {
    id: string;
    thought: Thought;
    radius: number;
    color: string;
  }

  interface GraphLink extends d3.SimulationLinkDatum<GraphNode> {
    source: GraphNode | string;
    target: GraphNode | string;
    type: string;
  }

  // Color mapping for thought types
  const typeColors: Record<string, string> = {
    Claim: '#3b82f6',
    Question: '#8b5cf6',
    Observation: '#10b981',
    Belief: '#f59e0b',
    Hypothesis: '#ec4899',
    Definition: '#6366f1',
    Argument: '#ef4444',
    Evidence: '#14b8a6',
    Intuition: '#a855f7',
    Memory: '#f97316',
    Goal: '#22c55e',
    Plan: '#0ea5e9',
    Reflection: '#64748b',
    Quote: '#84cc16',
    Note: '#78716c',
  };

  function buildGraph(thoughtList: Thought[]): { nodes: GraphNode[]; links: GraphLink[] } {
    const nodes: GraphNode[] = thoughtList.map((t) => ({
      id: t.id,
      thought: t,
      radius: 8 + t.confidence * 12,
      color: typeColors[t.thought_type] || '#6b7280',
    }));

    const nodeIds = new Set(nodes.map((n) => n.id));
    const links: GraphLink[] = [];

    // Build links from related_thoughts
    thoughtList.forEach((t) => {
      t.related_thoughts?.forEach((relatedId) => {
        if (nodeIds.has(relatedId)) {
          links.push({
            source: t.id,
            target: relatedId,
            type: 'related',
          });
        }
      });
    });

    // Build links from shared tags (implicit relationships)
    const tagMap = new Map<string, string[]>();
    thoughtList.forEach((t) => {
      t.tags?.forEach((tag) => {
        if (!tagMap.has(tag)) tagMap.set(tag, []);
        tagMap.get(tag)!.push(t.id);
      });
    });

    // Use Set for O(1) link deduplication instead of O(n) array.some()
    const linkSet = new Set(links.map((l) => {
      const s = typeof l.source === 'string' ? l.source : l.source;
      const t = typeof l.target === 'string' ? l.target : l.target;
      return s < t ? `${s}:${t}` : `${t}:${s}`;
    }));

    tagMap.forEach((ids) => {
      // Limit tag-based connections to prevent graph explosion
      if (ids.length > 1 && ids.length <= 5) {
        for (let i = 0; i < ids.length; i++) {
          for (let j = i + 1; j < ids.length; j++) {
            // O(1) lookup instead of O(n)
            const linkKey = ids[i] < ids[j] ? `${ids[i]}:${ids[j]}` : `${ids[j]}:${ids[i]}`;
            if (!linkSet.has(linkKey)) {
              linkSet.add(linkKey);
              links.push({
                source: ids[i],
                target: ids[j],
                type: 'tag',
              });
            }
          }
        }
      }
    });

    return { nodes, links };
  }

  function initGraph() {
    if (!container) return;

    // Clear existing
    d3.select(container).selectAll('*').remove();

    const { nodes, links } = buildGraph($thoughts);

    if (nodes.length === 0) return;

    svg = d3
      .select(container)
      .append('svg')
      .attr('width', '100%')
      .attr('height', '100%')
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('role', 'img')
      .attr('aria-label', `Knowledge graph visualization with ${nodes.length} thoughts and ${links.length} connections`)
      .attr('tabindex', '0');

    // Add zoom behavior
    const g = svg.append('g');

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom);

    // Create arrow marker for directed links
    svg.append('defs').append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '-0 -5 10 10')
      .attr('refX', 20)
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .append('path')
      .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
      .attr('fill', '#4a4a6e');

    // Create simulation with performance optimizations
    simulation = d3.forceSimulation<GraphNode>(nodes)
      .alphaDecay(0.02) // Faster stabilization
      .alphaMin(0.001)  // Stop at low alpha
      .velocityDecay(0.3) // Faster damping
      .force('link', d3.forceLink<GraphNode, GraphLink>(links)
        .id((d) => d.id)
        .distance(100)
        .strength((d) => d.type === 'related' ? 0.5 : 0.1))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide<GraphNode>().radius((d) => d.radius + 5));

    // Draw links
    const link = g.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', (d) => d.type === 'related' ? '#7c3aed' : '#2a2a4e')
      .attr('stroke-opacity', (d) => d.type === 'related' ? 0.8 : 0.3)
      .attr('stroke-width', (d) => d.type === 'related' ? 2 : 1)
      .attr('marker-end', (d) => d.type === 'related' ? 'url(#arrowhead)' : null);

    // Draw nodes
    const node = g.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .attr('cursor', 'pointer')
      .call(drag(simulation) as any);

    // Node circles with accessibility
    node.append('circle')
      .attr('r', (d) => d.radius)
      .attr('fill', (d) => d.color)
      .attr('stroke', '#1a1a2e')
      .attr('stroke-width', 2)
      .attr('role', 'button')
      .attr('tabindex', '0')
      .attr('aria-label', (d) => `${d.thought.thought_type}: ${d.thought.content.slice(0, 50)}${d.thought.content.length > 50 ? '...' : ''}`)
      .on('click', (event, d) => {
        event.stopPropagation();
        selectedThought.set(d.thought);
      })
      .on('keydown', (event: KeyboardEvent, d) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          selectedThought.set(d.thought);
        }
      });

    // Node labels (shown on hover via CSS)
    node.append('text')
      .text((d) => d.thought.content.slice(0, 30) + (d.thought.content.length > 30 ? '...' : ''))
      .attr('x', (d) => d.radius + 5)
      .attr('y', 4)
      .attr('fill', '#888')
      .attr('font-size', '11px')
      .attr('class', 'node-label');

    // Highlight selected node
    node.select('circle')
      .attr('stroke', (d) => d.thought.id === $selectedThought?.id ? '#7c3aed' : '#1a1a2e')
      .attr('stroke-width', (d) => d.thought.id === $selectedThought?.id ? 4 : 2);

    // Tooltip
    node.append('title')
      .text((d) => `${d.thought.thought_type}: ${d.thought.content}\n\nTags: ${d.thought.tags?.join(', ') || 'none'}\nConfidence: ${Math.round(d.thought.confidence * 100)}%`);

    // Update positions on tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });
  }

  // Drag behavior
  function drag(simulation: d3.Simulation<GraphNode, GraphLink>) {
    function dragstarted(event: d3.D3DragEvent<SVGGElement, GraphNode, GraphNode>) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }

    function dragged(event: d3.D3DragEvent<SVGGElement, GraphNode, GraphNode>) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }

    function dragended(event: d3.D3DragEvent<SVGGElement, GraphNode, GraphNode>) {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }

    return d3.drag<SVGGElement, GraphNode>()
      .on('start', dragstarted)
      .on('drag', dragged)
      .on('end', dragended);
  }

  // Reactive: rebuild graph when thoughts change
  $: if (container && $thoughts) {
    initGraph();
  }

  onMount(() => {
    initGraph();
  });

  onDestroy(() => {
    if (simulation) simulation.stop();
  });
</script>

<div class="knowledge-graph" bind:this={container}></div>

<style>
  .knowledge-graph {
    width: 100%;
    height: 100%;
    min-height: 400px;
    background: #0f0f1a;
    border-radius: 12px;
    border: 1px solid #2a2a4e;
    overflow: hidden;
  }

  .knowledge-graph :global(.node-label) {
    opacity: 0;
    transition: opacity 0.2s;
    pointer-events: none;
  }

  .knowledge-graph :global(g:hover .node-label) {
    opacity: 1;
  }

  .knowledge-graph :global(circle) {
    transition: stroke-width 0.2s, stroke 0.2s;
  }

  .knowledge-graph :global(circle:hover) {
    stroke-width: 4px;
    stroke: #7c3aed;
  }
</style>
