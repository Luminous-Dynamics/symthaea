<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import * as d3 from 'd3';
  import { invoke } from '@tauri-apps/api/core';

  /**
   * TrustNetwork - Ego-centric network visualization of trust relationships
   *
   * Shows:
   * - Current user at center
   * - Connected agents with trust relationships
   * - Edge thickness = trust score
   * - Edge color = relationship stage
   * - Node size = interaction count
   */

  // ============================================================================
  // TYPES
  // ============================================================================

  interface AgentRelationship {
    other_agent: string;
    trust_score: number;
    interaction_count: number;
    last_interaction: number;
    relationship_stage: RelationshipStage;
    shared_domains: string[];
    agreement_ratio: number;
  }

  type RelationshipStage =
    | 'NoRelation'
    | 'Acquaintance'
    | 'Collaborator'
    | 'TrustedPeer'
    | 'PartnerInTruth';

  interface NetworkNode {
    id: string;
    label: string;
    isMe: boolean;
    trustScore: number;
    interactionCount: number;
    stage: RelationshipStage;
    domains: string[];
  }

  interface NetworkEdge {
    source: string;
    target: string;
    weight: number;
    stage: RelationshipStage;
  }

  // ============================================================================
  // PROPS
  // ============================================================================

  export let width = 600;
  export let height = 400;
  export let showLabels = true;

  // ============================================================================
  // STATE
  // ============================================================================

  let container: HTMLDivElement;
  let relationships: AgentRelationship[] = [];
  let nodes: NetworkNode[] = [];
  let edges: NetworkEdge[] = [];
  let isLoading = true;
  let selectedNode: NetworkNode | null = null;
  let simulation: d3.Simulation<any, any> | null = null;

  // ============================================================================
  // LIFECYCLE
  // ============================================================================

  onMount(async () => {
    await loadRelationships();
    if (relationships.length > 0) {
      buildGraph();
      renderNetwork();
    }
    isLoading = false;
  });

  onDestroy(() => {
    if (simulation) {
      simulation.stop();
    }
  });

  // ============================================================================
  // DATA LOADING
  // ============================================================================

  async function loadRelationships(): Promise<void> {
    try {
      relationships = await invoke<AgentRelationship[]>('get_my_relationships');
    } catch (error) {
      console.error('Failed to load relationships:', error);
      relationships = [];
    }
  }

  function buildGraph(): void {
    // Create "me" node at center
    nodes = [
      {
        id: 'me',
        label: 'You',
        isMe: true,
        trustScore: 1.0,
        interactionCount: 0,
        stage: 'PartnerInTruth' as RelationshipStage,
        domains: [],
      },
    ];

    edges = [];

    // Add nodes and edges for each relationship
    for (const rel of relationships) {
      const shortId = rel.other_agent.slice(0, 8) + '...';

      nodes.push({
        id: rel.other_agent,
        label: shortId,
        isMe: false,
        trustScore: rel.trust_score,
        interactionCount: rel.interaction_count,
        stage: rel.relationship_stage,
        domains: rel.shared_domains,
      });

      edges.push({
        source: 'me',
        target: rel.other_agent,
        weight: rel.trust_score,
        stage: rel.relationship_stage,
      });
    }
  }

  // ============================================================================
  // VISUALIZATION
  // ============================================================================

  function renderNetwork(): void {
    if (!container || nodes.length === 0) return;

    // Clear previous
    d3.select(container).selectAll('*').remove();

    const svg = d3
      .select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', `0 0 ${width} ${height}`);

    // Add zoom
    const g = svg.append('g');
    svg.call(
      d3.zoom<SVGSVGElement, unknown>().scaleExtent([0.5, 3]).on('zoom', (event) => {
        g.attr('transform', event.transform);
      }) as any
    );

    // Create force simulation
    simulation = d3
      .forceSimulation(nodes as any)
      .force(
        'link',
        d3
          .forceLink(edges as any)
          .id((d: any) => d.id)
          .distance((d: any) => 100 - d.weight * 50)
      )
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(30));

    // Draw edges
    const link = g
      .append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(edges)
      .enter()
      .append('line')
      .attr('stroke', (d) => getStageColor(d.stage))
      .attr('stroke-width', (d) => 1 + d.weight * 4)
      .attr('stroke-opacity', 0.6);

    // Draw nodes
    const node = g
      .append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', 'node')
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        selectedNode = d as NetworkNode;
      })
      .call(
        d3
          .drag<SVGGElement, NetworkNode>()
          .on('start', dragstarted)
          .on('drag', dragged)
          .on('end', dragended) as any
      );

    // Node circles
    node
      .append('circle')
      .attr('r', (d) => (d.isMe ? 25 : 10 + Math.min(d.interactionCount, 10) * 1.5))
      .attr('fill', (d) => (d.isMe ? 'var(--color-primary, #3b82f6)' : getStageColor(d.stage)))
      .attr('stroke', 'var(--bg-tertiary, #0f172a)')
      .attr('stroke-width', 2);

    // Node labels
    if (showLabels) {
      node
        .append('text')
        .text((d) => d.label)
        .attr('dy', (d) => (d.isMe ? 40 : 25))
        .attr('text-anchor', 'middle')
        .attr('fill', 'var(--text-primary, #f8fafc)')
        .attr('font-size', '10px');
    }

    // Trust score badge for non-me nodes
    node
      .filter((d) => !d.isMe)
      .append('text')
      .text((d) => `${Math.round(d.trustScore * 100)}%`)
      .attr('dy', -15)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--text-muted, #94a3b8)')
      .attr('font-size', '9px');

    // Simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });

    function dragstarted(event: any) {
      if (!event.active) simulation!.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }

    function dragged(event: any) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }

    function dragended(event: any) {
      if (!event.active) simulation!.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }
  }

  // ============================================================================
  // HELPERS
  // ============================================================================

  function getStageColor(stage: RelationshipStage): string {
    switch (stage) {
      case 'NoRelation':
        return '#6b7280';
      case 'Acquaintance':
        return '#94a3b8';
      case 'Collaborator':
        return '#3b82f6';
      case 'TrustedPeer':
        return '#8b5cf6';
      case 'PartnerInTruth':
        return '#22c55e';
      default:
        return '#6b7280';
    }
  }

  function getStageName(stage: RelationshipStage): string {
    switch (stage) {
      case 'NoRelation':
        return 'No Relation';
      case 'Acquaintance':
        return 'Acquaintance';
      case 'Collaborator':
        return 'Collaborator';
      case 'TrustedPeer':
        return 'Trusted Peer';
      case 'PartnerInTruth':
        return 'Partner in Truth';
      default:
        return 'Unknown';
    }
  }
</script>

<div class="trust-network">
  <div class="header">
    <h3>Trust Network</h3>
    <span class="count">{relationships.length} connections</span>
  </div>

  {#if isLoading}
    <div class="loading">
      <div class="spinner"></div>
      <span>Loading relationships...</span>
    </div>
  {:else if relationships.length === 0}
    <div class="empty">
      <span class="icon">🤝</span>
      <p>No relationships yet.</p>
      <p class="hint">Start validating beliefs to build your trust network.</p>
    </div>
  {:else}
    <div class="network-container" bind:this={container}></div>

    <!-- Legend -->
    <div class="legend">
      <span class="legend-title">Relationship Stages:</span>
      <div class="legend-items">
        <div class="legend-item">
          <span class="dot" style="background: #94a3b8"></span>
          <span>Acquaintance</span>
        </div>
        <div class="legend-item">
          <span class="dot" style="background: #3b82f6"></span>
          <span>Collaborator</span>
        </div>
        <div class="legend-item">
          <span class="dot" style="background: #8b5cf6"></span>
          <span>Trusted Peer</span>
        </div>
        <div class="legend-item">
          <span class="dot" style="background: #22c55e"></span>
          <span>Partner in Truth</span>
        </div>
      </div>
    </div>

    <!-- Selected node details -->
    {#if selectedNode && !selectedNode.isMe}
      <div class="node-details">
        <h4>Agent Details</h4>
        <div class="detail-row">
          <span class="label">Trust Score:</span>
          <span class="value">{Math.round(selectedNode.trustScore * 100)}%</span>
        </div>
        <div class="detail-row">
          <span class="label">Stage:</span>
          <span class="value stage-badge" style="background: {getStageColor(selectedNode.stage)}">
            {getStageName(selectedNode.stage)}
          </span>
        </div>
        <div class="detail-row">
          <span class="label">Interactions:</span>
          <span class="value">{selectedNode.interactionCount}</span>
        </div>
        {#if selectedNode.domains.length > 0}
          <div class="detail-row">
            <span class="label">Shared Domains:</span>
            <span class="value">{selectedNode.domains.join(', ')}</span>
          </div>
        {/if}
        <button class="close-btn" on:click={() => (selectedNode = null)}>Close</button>
      </div>
    {/if}
  {/if}
</div>

<style>
  .trust-network {
    background: var(--bg-secondary, #1e293b);
    border-radius: 12px;
    padding: 16px;
    color: var(--text-primary, #f8fafc);
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border, #334155);
  }

  .header h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
  }

  .count {
    font-size: 12px;
    color: var(--text-muted, #94a3b8);
  }

  .loading,
  .empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 200px;
    color: var(--text-muted, #94a3b8);
    gap: 12px;
  }

  .spinner {
    width: 24px;
    height: 24px;
    border: 2px solid var(--border, #334155);
    border-top-color: var(--color-primary, #3b82f6);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .empty .icon {
    font-size: 32px;
  }

  .empty p {
    margin: 0;
    text-align: center;
  }

  .hint {
    font-size: 12px;
  }

  .network-container {
    width: 100%;
    min-height: 300px;
    background: var(--bg-tertiary, #0f172a);
    border-radius: 8px;
    overflow: hidden;
  }

  .legend {
    margin-top: 12px;
    padding: 10px;
    background: var(--bg-tertiary, #0f172a);
    border-radius: 6px;
  }

  .legend-title {
    font-size: 11px;
    color: var(--text-muted, #94a3b8);
    display: block;
    margin-bottom: 8px;
  }

  .legend-items {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 10px;
  }

  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }

  .node-details {
    margin-top: 12px;
    padding: 12px;
    background: var(--bg-tertiary, #0f172a);
    border-radius: 8px;
    border-left: 3px solid var(--color-primary, #3b82f6);
  }

  .node-details h4 {
    margin: 0 0 12px 0;
    font-size: 12px;
    font-weight: 600;
  }

  .detail-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11px;
    margin-bottom: 8px;
  }

  .detail-row .label {
    color: var(--text-muted, #94a3b8);
  }

  .stage-badge {
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 10px;
    color: white;
  }

  .close-btn {
    width: 100%;
    margin-top: 8px;
    padding: 6px;
    background: var(--bg-secondary, #1e293b);
    border: 1px solid var(--border, #334155);
    border-radius: 4px;
    color: var(--text-primary, #f8fafc);
    font-size: 11px;
    cursor: pointer;
  }

  .close-btn:hover {
    background: var(--border, #334155);
  }
</style>
