<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher } from 'svelte';
  import * as d3 from 'd3';
  import type { AgentPubKey } from '@holochain/client';
  import {
    type AgentRelationship,
    RelationshipStage,
    getMyRelationships,
  } from '$services/collective-sensemaking';
  import { isConnected } from '$stores/holochain';
  import {
    getReputation,
    projectReputation,
    initializeReputation,
    type ReputationState,
  } from '$services/reputation-decay';

  /**
   * AgentRelationshipNetwork - Ego-centric network visualization
   *
   * Design:
   * - Current user in center
   * - Node size = interaction count
   * - Edge thickness = trust score
   * - Edge color = relationship stage
   *
   * Features:
   * - D3-based visualization with zoom/pan
   * - Detail tooltips on hover
   * - Click to view relationship details
   */

  // ============================================================================
  // PROPS
  // ============================================================================

  export let width = 500;
  export let height = 400;

  // ============================================================================
  // STATE
  // ============================================================================

  const dispatch = createEventDispatcher();

  let container: HTMLDivElement;
  let svg: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  let simulation: d3.Simulation<NetworkNode, NetworkLink>;

  let relationships: AgentRelationship[] = [];
  let isLoading = true;
  let selectedAgent: NetworkNode | null = null;

  // D3-compatible types
  interface NetworkNode extends d3.SimulationNodeDatum {
    id: string;
    label: string;
    isEgo: boolean;
    interactionCount: number;
    trustScore: number;
    stage: RelationshipStage;
    relationship?: AgentRelationship;
    // Reputation data
    reputation: number;
    reputationProjected7d: number;
    reputationTrend: 'rising' | 'stable' | 'declining';
  }

  interface NetworkLink extends d3.SimulationLinkDatum<NetworkNode> {
    source: NetworkNode | string;
    target: NetworkNode | string;
    trustScore: number;
    stage: RelationshipStage;
  }

  // ============================================================================
  // DATA LOADING
  // ============================================================================

  async function loadRelationships() {
    isLoading = true;

    try {
      if ($isConnected) {
        relationships = await getMyRelationships();
      } else {
        // Generate simulated relationships for demo
        relationships = generateSimulatedRelationships();
      }

      initGraph();
    } catch (e) {
      console.error('Failed to load relationships:', e);
      relationships = generateSimulatedRelationships();
      initGraph();
    } finally {
      isLoading = false;
    }
  }

  function generateSimulatedRelationships(): AgentRelationship[] {
    const stages = [
      RelationshipStage.NoRelation,
      RelationshipStage.Acquaintance,
      RelationshipStage.Collaborator,
      RelationshipStage.TrustedPeer,
      RelationshipStage.PartnerInTruth,
    ];

    const names = ['Alice', 'Bob', 'Carol', 'Dave', 'Eve', 'Frank', 'Grace'];

    return names.slice(0, 5 + Math.floor(Math.random() * 3)).map((name, i) => ({
      other_agent: `agent-${i}` as unknown as AgentPubKey,
      relationship_stage: stages[Math.min(i, stages.length - 1)],
      trust_score: 0.3 + Math.random() * 0.6,
      interaction_count: Math.floor(Math.random() * 50) + 5,
      last_interaction: (Date.now() * 1000 - Math.random() * 86400000000) as unknown as import('@holochain/client').Timestamp,
      shared_domains: [],
      agreement_ratio: 0.5 + Math.random() * 0.5,
    }));
  }

  // ============================================================================
  // GRAPH BUILDING
  // ============================================================================

  function getReputationData(agentId: string): { reputation: number; projected7d: number; trend: 'rising' | 'stable' | 'declining' } {
    // Initialize reputation if not exists
    let state = getReputation(agentId);
    if (!state) {
      // Simulate initial reputation based on trust score
      const baseRep = 50 + Math.random() * 50;
      initializeReputation(agentId, baseRep);
      state = getReputation(agentId);
    }

    const current = state?.currentReputation ?? 50;
    const projected = projectReputation(agentId, 7);

    let trend: 'rising' | 'stable' | 'declining' = 'stable';
    const diff = projected - current;
    if (diff > 2) trend = 'rising';
    else if (diff < -2) trend = 'declining';

    return { reputation: current, projected7d: projected, trend };
  }

  function buildNetwork(): { nodes: NetworkNode[]; links: NetworkLink[] } {
    const nodes: NetworkNode[] = [];
    const links: NetworkLink[] = [];

    // Get ego reputation
    const egoRep = getReputationData('ego');

    // Add ego node (current user)
    const egoNode: NetworkNode = {
      id: 'ego',
      label: 'You',
      isEgo: true,
      interactionCount: 0,
      trustScore: 1,
      stage: RelationshipStage.PartnerInTruth,
      reputation: egoRep.reputation,
      reputationProjected7d: egoRep.projected7d,
      reputationTrend: egoRep.trend,
    };
    nodes.push(egoNode);

    // Add relationship nodes
    relationships.forEach((rel, i) => {
      const agentId = typeof rel.other_agent === 'string'
        ? rel.other_agent
        : `agent-${i}`;

      const repData = getReputationData(agentId);

      const node: NetworkNode = {
        id: agentId,
        label: getAgentLabel(agentId, i),
        isEgo: false,
        interactionCount: rel.interaction_count,
        trustScore: rel.trust_score,
        stage: rel.relationship_stage,
        relationship: rel,
        reputation: repData.reputation,
        reputationProjected7d: repData.projected7d,
        reputationTrend: repData.trend,
      };
      nodes.push(node);

      // Link from ego to this agent
      links.push({
        source: 'ego',
        target: agentId,
        trustScore: rel.trust_score,
        stage: rel.relationship_stage,
      });
    });

    return { nodes, links };
  }

  function getAgentLabel(id: string, index: number): string {
    const names = ['Alice', 'Bob', 'Carol', 'Dave', 'Eve', 'Frank', 'Grace', 'Henry'];
    return names[index % names.length];
  }

  function getStageColor(stage: RelationshipStage): string {
    switch (stage) {
      case RelationshipStage.NoRelation: return '#6b7280';
      case RelationshipStage.Acquaintance: return '#94a3b8';
      case RelationshipStage.Collaborator: return '#3b82f6';
      case RelationshipStage.TrustedPeer: return '#8b5cf6';
      case RelationshipStage.PartnerInTruth: return '#22c55e';
      default: return '#6b7280';
    }
  }

  function getStageLabel(stage: RelationshipStage): string {
    switch (stage) {
      case RelationshipStage.NoRelation: return 'Stranger';
      case RelationshipStage.Acquaintance: return 'Acquaintance';
      case RelationshipStage.Collaborator: return 'Collaborator';
      case RelationshipStage.TrustedPeer: return 'Trusted Peer';
      case RelationshipStage.PartnerInTruth: return 'Partner in Truth';
      default: return 'Unknown';
    }
  }

  // ============================================================================
  // D3 VISUALIZATION
  // ============================================================================

  function initGraph() {
    if (!container) return;

    // Clear existing
    d3.select(container).selectAll('*').remove();

    if (relationships.length === 0) {
      d3.select(container)
        .append('div')
        .attr('class', 'empty-network')
        .html(`
          <span class="empty-icon">@</span>
          <p>No relationships yet</p>
          <p class="hint">Interact with the collective to build your network</p>
        `);
      return;
    }

    const { nodes, links } = buildNetwork();

    svg = d3
      .select(container)
      .append('svg')
      .attr('width', '100%')
      .attr('height', '100%')
      .attr('viewBox', `0 0 ${width} ${height}`);

    // Zoom behavior
    const g = svg.append('g');

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 3])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom);

    // Create force simulation (radial layout centered on ego)
    // Performance: alphaDecay controls how quickly the simulation stabilizes
    simulation = d3.forceSimulation<NetworkNode>(nodes)
      .alphaDecay(0.02) // Faster stabilization (default is 0.0228)
      .alphaMin(0.001)  // Stop simulation at this alpha
      .force('link', d3.forceLink<NetworkNode, NetworkLink>(links)
        .id((d) => d.id)
        .distance((d) => 80 + (1 - d.trustScore) * 60)
        .strength((d) => d.trustScore * 0.5))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('radial', d3.forceRadial<NetworkNode>(
        (d) => d.isEgo ? 0 : 100 + (1 - d.trustScore) * 50,
        width / 2,
        height / 2
      ).strength(0.5))
      .force('collision', d3.forceCollide<NetworkNode>().radius((d) => getNodeRadius(d) + 10));

    // Draw links
    const link = g.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', (d) => getStageColor(d.stage))
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', (d) => 1 + d.trustScore * 4);

    // Draw nodes
    const node = g.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .attr('cursor', 'pointer')
      .call(drag(simulation) as any);

    // Node circles
    node.append('circle')
      .attr('r', (d) => getNodeRadius(d))
      .attr('fill', (d) => d.isEgo ? '#7c3aed' : getStageColor(d.stage))
      .attr('stroke', '#1e293b')
      .attr('stroke-width', (d) => d.isEgo ? 3 : 2);

    // Ego indicator ring
    node.filter((d) => d.isEgo)
      .append('circle')
      .attr('r', (d) => getNodeRadius(d) + 5)
      .attr('fill', 'none')
      .attr('stroke', '#7c3aed')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '4,4');

    // Node labels
    node.append('text')
      .text((d) => d.label)
      .attr('x', 0)
      .attr('y', (d) => getNodeRadius(d) + 14)
      .attr('text-anchor', 'middle')
      .attr('fill', '#94a3b8')
      .attr('font-size', '11px')
      .attr('font-weight', (d) => d.isEgo ? '600' : '400');

    // Trust score badge
    node.filter((d) => !d.isEgo)
      .append('text')
      .text((d) => Math.round(d.trustScore * 100))
      .attr('x', 0)
      .attr('y', 4)
      .attr('text-anchor', 'middle')
      .attr('fill', 'white')
      .attr('font-size', '10px')
      .attr('font-weight', '600');

    // Reputation arc indicator (outer ring showing reputation level)
    const reputationArc = d3.arc<NetworkNode>()
      .innerRadius((d) => getNodeRadius(d) + 2)
      .outerRadius((d) => getNodeRadius(d) + 5)
      .startAngle(0)
      .endAngle((d) => (d.reputation / 100) * 2 * Math.PI);

    node.append('path')
      .attr('d', (d) => reputationArc(d) as string)
      .attr('fill', (d) => {
        if (d.reputation >= 80) return '#22c55e'; // green - high rep
        if (d.reputation >= 60) return '#3b82f6'; // blue - good rep
        if (d.reputation >= 40) return '#eab308'; // yellow - moderate rep
        return '#ef4444'; // red - low rep
      })
      .attr('opacity', 0.8);

    // Reputation trend indicator (small arrow)
    node.filter((d) => d.reputationTrend !== 'stable')
      .append('text')
      .text((d) => d.reputationTrend === 'rising' ? '↑' : '↓')
      .attr('x', (d) => getNodeRadius(d) + 8)
      .attr('y', (d) => -getNodeRadius(d) + 4)
      .attr('fill', (d) => d.reputationTrend === 'rising' ? '#22c55e' : '#ef4444')
      .attr('font-size', '10px')
      .attr('font-weight', '700');

    // Tooltips with reputation info
    node.append('title')
      .text((d) => {
        const trendIcon = d.reputationTrend === 'rising' ? '↑' : d.reputationTrend === 'declining' ? '↓' : '→';
        if (d.isEgo) {
          return `You (center of your network)\nReputation: ${Math.round(d.reputation)} ${trendIcon}\n7-day projection: ${Math.round(d.reputationProjected7d)}`;
        }
        return `${d.label}\nTrust: ${Math.round(d.trustScore * 100)}%\nStage: ${getStageLabel(d.stage)}\nInteractions: ${d.interactionCount}\n────────────\nReputation: ${Math.round(d.reputation)} ${trendIcon}\n7-day projection: ${Math.round(d.reputationProjected7d)}`;
      });

    // Event handlers
    node
      .on('click', (event, d) => {
        event.stopPropagation();
        if (!d.isEgo) {
          selectedAgent = d;
          dispatch('agentselect', d.relationship);
        }
      })
      .on('mouseenter', function(event, d) {
        if (!d.isEgo) {
          d3.select(this).select('circle')
            .attr('stroke', '#7c3aed')
            .attr('stroke-width', 3);
        }
      })
      .on('mouseleave', function(event, d) {
        if (!d.isEgo) {
          d3.select(this).select('circle')
            .attr('stroke', '#1e293b')
            .attr('stroke-width', 2);
        }
      });

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

  function getNodeRadius(d: NetworkNode): number {
    if (d.isEgo) return 25;
    // Size based on interaction count (min 12, max 22)
    return Math.min(22, Math.max(12, 12 + Math.log(d.interactionCount + 1) * 3));
  }

  // Drag behavior
  function drag(simulation: d3.Simulation<NetworkNode, NetworkLink>) {
    function dragstarted(event: d3.D3DragEvent<SVGGElement, NetworkNode, NetworkNode>) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }

    function dragged(event: d3.D3DragEvent<SVGGElement, NetworkNode, NetworkNode>) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }

    function dragended(event: d3.D3DragEvent<SVGGElement, NetworkNode, NetworkNode>) {
      if (!event.active) simulation.alphaTarget(0);
      // Keep ego fixed at center
      if (!event.subject.isEgo) {
        event.subject.fx = null;
        event.subject.fy = null;
      }
    }

    return d3.drag<SVGGElement, NetworkNode>()
      .on('start', dragstarted)
      .on('drag', dragged)
      .on('end', dragended);
  }

  // ============================================================================
  // LIFECYCLE
  // ============================================================================

  onMount(() => {
    loadRelationships();
  });

  onDestroy(() => {
    if (simulation) simulation.stop();
  });
</script>

<div class="relationship-network">
  {#if isLoading}
    <div class="loading-state">
      <div class="spinner"></div>
      <span>Loading network...</span>
    </div>
  {:else}
    <div class="network-graph" bind:this={container}></div>

    <!-- Legend -->
    <div class="network-legend">
      <div class="legend-title">Relationship Stages</div>
      <div class="legend-items">
        <div class="legend-item">
          <span class="legend-circle" style="background: {getStageColor(RelationshipStage.NoRelation)}"></span>
          <span>Stranger</span>
        </div>
        <div class="legend-item">
          <span class="legend-circle" style="background: {getStageColor(RelationshipStage.Acquaintance)}"></span>
          <span>Acquaintance</span>
        </div>
        <div class="legend-item">
          <span class="legend-circle" style="background: {getStageColor(RelationshipStage.Collaborator)}"></span>
          <span>Collaborator</span>
        </div>
        <div class="legend-item">
          <span class="legend-circle" style="background: {getStageColor(RelationshipStage.TrustedPeer)}"></span>
          <span>Trusted Peer</span>
        </div>
        <div class="legend-item">
          <span class="legend-circle" style="background: {getStageColor(RelationshipStage.PartnerInTruth)}"></span>
          <span>Partner in Truth</span>
        </div>
      </div>
    </div>

    <!-- Stats summary -->
    <div class="network-stats">
      <div class="stat">
        <span class="stat-value">{relationships.length}</span>
        <span class="stat-label">Connections</span>
      </div>
      <div class="stat">
        <span class="stat-value">
          {Math.round((relationships.reduce<number>((s, r) => s + r.trust_score, 0) / Math.max(1, relationships.length)) * 100)}%
        </span>
        <span class="stat-label">Avg Trust</span>
      </div>
      <div class="stat">
        <span class="stat-value">
          {relationships.reduce<number>((s, r) => s + r.interaction_count, 0)}
        </span>
        <span class="stat-label">Interactions</span>
      </div>
      {#if true}
        {@const egoRep = getReputation('ego')?.currentReputation ?? 50}
        <div class="stat">
          <span class="stat-value" style="color: {egoRep >= 70 ? '#22c55e' : egoRep >= 50 ? '#3b82f6' : '#eab308'}">
            {Math.round(egoRep)}
          </span>
          <span class="stat-label">Your Rep</span>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .relationship-network {
    display: flex;
    flex-direction: column;
    height: 100%;
    position: relative;
  }

  .loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    gap: 12px;
    color: var(--text-muted, #94a3b8);
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
    to { transform: rotate(360deg); }
  }

  .network-graph {
    flex: 1;
    min-height: 300px;
    background: #0f0f1a;
    border-radius: 8px;
    border: 1px solid var(--border, #334155);
    overflow: hidden;
  }

  .network-graph :global(.empty-network) {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--text-muted, #94a3b8);
    text-align: center;
  }

  .network-graph :global(.empty-icon) {
    font-size: 48px;
    opacity: 0.3;
    margin-bottom: 8px;
  }

  .network-graph :global(.hint) {
    font-size: 12px;
    opacity: 0.7;
  }

  /* Legend */
  .network-legend {
    position: absolute;
    bottom: 60px;
    left: 12px;
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid var(--border, #334155);
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 10px;
  }

  .legend-title {
    font-weight: 600;
    color: var(--text-muted, #94a3b8);
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .legend-items {
    display: flex;
    flex-direction: column;
    gap: 3px;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--text-muted, #94a3b8);
  }

  .legend-circle {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }

  /* Stats summary */
  .network-stats {
    display: flex;
    justify-content: space-around;
    padding: 12px;
    background: var(--bg-tertiary, #0f172a);
    border-radius: 8px;
    margin-top: 12px;
  }

  .stat {
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .stat-value {
    font-size: 18px;
    font-weight: 700;
    color: var(--color-primary, #3b82f6);
  }

  .stat-label {
    font-size: 10px;
    color: var(--text-muted, #94a3b8);
    text-transform: uppercase;
  }
</style>
