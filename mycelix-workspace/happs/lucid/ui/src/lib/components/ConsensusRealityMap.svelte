<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher } from 'svelte';
  import * as d3 from 'd3';
  import type {
    CollectiveView,
    RealityMapNode,
    RealityMapEdge,
    BeliefShare,
  } from '$services/collective-sensemaking';
  import { buildRealityMap, ConsensusType } from '$services/collective-sensemaking';
  import {
    kMeansClustering,
    type SemanticCluster,
  } from '$services/semantic-clustering';

  /**
   * ConsensusRealityMap - D3 force-directed graph of beliefs and consensus
   *
   * Visual Elements:
   * - Belief nodes (circles, sized by confidence)
   * - Pattern nodes (hexagons)
   * - Edges: dashed=related, solid=agreement, red=contradiction
   * - Color coding: green=strong consensus, yellow=moderate, red=contested
   *
   * Interactions:
   * - Click node → show detail panel
   * - Hover → highlight connections
   * - Zoom/pan support
   */

  // ============================================================================
  // PROPS
  // ============================================================================

  export let collectiveView: CollectiveView | null = null;
  export let width = 600;
  export let height = 400;
  export let showClusters = true;

  // ============================================================================
  // STATE
  // ============================================================================

  const dispatch = createEventDispatcher();

  let container: HTMLDivElement;
  let svg: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  let simulation: d3.Simulation<D3Node, D3Link>;
  let selectedNode: D3Node | null = null;
  let hoveredNode: D3Node | null = null;
  let clusters: SemanticCluster[] = [];
  let clusteringInProgress = false;

  // Performance optimization: throttle hull redraws
  let lastHullUpdate = 0;
  const HULL_UPDATE_INTERVAL = 50; // ms between hull redraws
  let hullsInitialized = false;

  // D3-compatible node/link types
  interface D3Node extends d3.SimulationNodeDatum {
    id: string;
    label: string;
    type: 'belief' | 'consensus' | 'pattern';
    size: number;
    color: string;
    data: RealityMapNode['data'];
    consensusScore?: number;
  }

  interface D3Link extends d3.SimulationLinkDatum<D3Node> {
    source: D3Node | string;
    target: D3Node | string;
    type: 'agreement' | 'disagreement' | 'related' | 'pattern';
    weight: number;
  }

  // Cluster colors (distinct, semi-transparent for hulls)
  const CLUSTER_COLORS = [
    'rgba(59, 130, 246, 0.15)', // blue
    'rgba(168, 85, 247, 0.15)', // purple
    'rgba(236, 72, 153, 0.15)', // pink
    'rgba(34, 197, 94, 0.15)',  // green
    'rgba(245, 158, 11, 0.15)', // amber
    'rgba(6, 182, 212, 0.15)',  // cyan
  ];

  const CLUSTER_BORDER_COLORS = [
    'rgba(59, 130, 246, 0.5)',
    'rgba(168, 85, 247, 0.5)',
    'rgba(236, 72, 153, 0.5)',
    'rgba(34, 197, 94, 0.5)',
    'rgba(245, 158, 11, 0.5)',
    'rgba(6, 182, 212, 0.5)',
  ];

  // ============================================================================
  // SEMANTIC CLUSTERING
  // ============================================================================

  async function computeClusters(beliefs: BeliefShare[]): Promise<void> {
    if (beliefs.length < 3) {
      clusters = [];
      return;
    }

    clusteringInProgress = true;
    try {
      // Determine optimal number of clusters (between 2 and min(6, beliefs/3))
      const k = Math.min(6, Math.max(2, Math.floor(beliefs.length / 3)));
      clusters = await kMeansClustering(beliefs, k);
    } catch (e) {
      console.warn('Clustering failed:', e);
      clusters = [];
    } finally {
      clusteringInProgress = false;
    }
  }

  function getClusterForNode(nodeId: string): number {
    for (let i = 0; i < clusters.length; i++) {
      if (clusters[i].members.some(m => m.beliefHash === nodeId)) {
        return i;
      }
    }
    return -1;
  }

  // ============================================================================
  // GRAPH BUILDING
  // ============================================================================

  function buildD3Graph(view: CollectiveView): { nodes: D3Node[]; links: D3Link[] } {
    const { nodes: mapNodes, edges: mapEdges } = buildRealityMap(view);

    const nodes: D3Node[] = mapNodes.map((n) => ({
      id: n.id,
      label: n.label,
      type: n.type,
      size: n.size,
      color: n.color,
      data: n.data,
      consensusScore: getConsensusScore(n, view),
    }));

    const nodeIds = new Set(nodes.map((n) => n.id));

    const links: D3Link[] = mapEdges
      .filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target))
      .map((e) => ({
        source: e.source,
        target: e.target,
        type: e.type,
        weight: e.weight,
      }));

    return { nodes, links };
  }

  function getConsensusScore(node: RealityMapNode, view: CollectiveView): number | undefined {
    if (node.type !== 'belief') return undefined;

    // Find consensus for this belief
    const consensus = view.consensusRecords.find(
      (c) => c.belief_share_hash.toString() === node.id
    );

    return consensus?.agreement_score;
  }

  function getNodeColor(node: D3Node): string {
    if (node.type === 'pattern') {
      return '#a855f7'; // Purple for patterns
    }

    if (node.consensusScore !== undefined) {
      if (node.consensusScore >= 0.8) return '#22c55e'; // Strong consensus - green
      if (node.consensusScore >= 0.6) return '#3b82f6'; // Moderate - blue
      if (node.consensusScore >= 0.4) return '#f59e0b'; // Weak - yellow
      return '#ef4444'; // Contested - red
    }

    return node.color;
  }

  // ============================================================================
  // D3 VISUALIZATION
  // ============================================================================

  function initGraph() {
    if (!container || !collectiveView) return;

    // Clear existing
    d3.select(container).selectAll('*').remove();

    const { nodes, links } = buildD3Graph(collectiveView);

    if (nodes.length === 0) {
      // Show empty state
      d3.select(container)
        .append('div')
        .attr('class', 'empty-graph')
        .html(`
          <span class="empty-icon">O</span>
          <p>No beliefs in the collective yet</p>
          <p class="hint">Share thoughts to build the consensus map</p>
        `);
      return;
    }

    svg = d3
      .select(container)
      .append('svg')
      .attr('width', '100%')
      .attr('height', '100%')
      .attr('viewBox', `0 0 ${width} ${height}`);

    // Add defs for markers and gradients
    const defs = svg.append('defs');

    // Arrow markers
    defs.append('marker')
      .attr('id', 'arrow-agreement')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 15)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#22c55e');

    defs.append('marker')
      .attr('id', 'arrow-disagreement')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 15)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#ef4444');

    // Zoom behavior
    const g = svg.append('g');

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.2, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom);

    // Compute clusters from belief nodes (combine my shares and public shares)
    const beliefNodes = [...collectiveView.myShares, ...collectiveView.publicShares];
    if (showClusters && beliefNodes.length >= 3) {
      computeClusters(beliefNodes).then(() => {
        // Redraw with clusters after they're computed
        drawClusterHulls(g, nodes);
      });
    }

    // Create force simulation with performance optimizations
    simulation = d3.forceSimulation<D3Node>(nodes)
      .alphaDecay(0.02) // Faster stabilization
      .alphaMin(0.001)  // Stop at low alpha
      .velocityDecay(0.3) // Slightly faster damping
      .force('link', d3.forceLink<D3Node, D3Link>(links)
        .id((d) => d.id)
        .distance((d) => d.type === 'agreement' ? 60 : 100)
        .strength((d) => d.weight))
      .force('charge', d3.forceManyBody().strength(-150))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide<D3Node>().radius((d) => d.size + 5));

    // Draw links
    const link = g.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', (d) => {
        switch (d.type) {
          case 'agreement': return '#22c55e';
          case 'disagreement': return '#ef4444';
          case 'pattern': return '#a855f7';
          default: return '#334155';
        }
      })
      .attr('stroke-opacity', (d) => d.type === 'related' ? 0.3 : 0.6)
      .attr('stroke-width', (d) => d.type === 'related' ? 1 : 2)
      .attr('stroke-dasharray', (d) => d.type === 'related' ? '4,4' : null)
      .attr('marker-end', (d) => {
        if (d.type === 'agreement') return 'url(#arrow-agreement)';
        if (d.type === 'disagreement') return 'url(#arrow-disagreement)';
        return null;
      });

    // Draw nodes
    const node = g.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .attr('cursor', 'pointer')
      .call(drag(simulation) as any);

    // Different shapes for different types
    node.each(function(d) {
      const el = d3.select(this);

      if (d.type === 'pattern') {
        // Hexagon for patterns
        const hexPoints = getHexagonPoints(d.size);
        el.append('polygon')
          .attr('points', hexPoints)
          .attr('fill', getNodeColor(d))
          .attr('stroke', '#1e293b')
          .attr('stroke-width', 2);
      } else {
        // Circle for beliefs
        el.append('circle')
          .attr('r', d.size)
          .attr('fill', getNodeColor(d))
          .attr('stroke', '#1e293b')
          .attr('stroke-width', 2);

        // Add consensus ring for beliefs with consensus
        if (d.consensusScore !== undefined) {
          el.append('circle')
            .attr('r', d.size + 3)
            .attr('fill', 'none')
            .attr('stroke', getNodeColor(d))
            .attr('stroke-width', 2)
            .attr('stroke-opacity', 0.3);
        }
      }
    });

    // Node labels (shown on hover)
    node.append('text')
      .text((d) => d.label)
      .attr('x', (d) => d.size + 5)
      .attr('y', 4)
      .attr('fill', '#94a3b8')
      .attr('font-size', '11px')
      .attr('class', 'node-label')
      .attr('pointer-events', 'none');

    // Tooltips
    node.append('title')
      .text((d) => {
        if (d.type === 'pattern') {
          return `Pattern: ${d.label}`;
        }
        const belief = d.data as BeliefShare;
        return `${belief.belief_type}: ${belief.content}\n\nTags: ${belief.tags.join(', ')}\nConfidence: ${Math.round(belief.confidence * 100)}%`;
      });

    // Event handlers
    node
      .on('click', (event, d) => {
        event.stopPropagation();
        selectedNode = d;
        highlightConnections(d, links, node, link);

        if (d.type === 'belief') {
          dispatch('nodeselect', d.data);
        }
      })
      .on('mouseenter', (event, d) => {
        hoveredNode = d;
        highlightConnections(d, links, node, link);
      })
      .on('mouseleave', () => {
        if (!selectedNode) {
          hoveredNode = null;
          resetHighlights(node, link);
        }
      });

    // Click on background to deselect
    svg.on('click', () => {
      selectedNode = null;
      hoveredNode = null;
      resetHighlights(node, link);
    });

    // Update positions on tick with performance optimizations
    simulation.on('tick', () => {
      // Update link positions
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      // Update node positions
      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);

      // Throttle cluster hull redraws for performance
      if (showClusters && clusters.length > 0) {
        const now = performance.now();
        if (now - lastHullUpdate > HULL_UPDATE_INTERVAL || !hullsInitialized) {
          lastHullUpdate = now;
          hullsInitialized = true;
          updateClusterHulls(g, nodes);
        }
      }
    });

    // Stop simulation after it stabilizes (prevent indefinite CPU usage)
    simulation.on('end', () => {
      // Final hull update when simulation ends
      if (showClusters && clusters.length > 0) {
        updateClusterHulls(g, nodes);
      }
    });
  }

  function getHexagonPoints(size: number): string {
    const points: [number, number][] = [];
    for (let i = 0; i < 6; i++) {
      const angle = (Math.PI / 3) * i - Math.PI / 6;
      points.push([
        Math.cos(angle) * size,
        Math.sin(angle) * size,
      ]);
    }
    return points.map((p) => p.join(',')).join(' ');
  }

  // Draw cluster hulls around semantically similar beliefs (initial creation)
  function drawClusterHulls(
    g: d3.Selection<SVGGElement, unknown, null, undefined>,
    nodes: D3Node[]
  ): void {
    // Remove existing hulls
    g.selectAll('.cluster-hulls').remove();

    if (!showClusters || clusters.length === 0) return;

    // Create hull group (behind other elements)
    const hullGroup = g.insert('g', ':first-child').attr('class', 'cluster-hulls');

    clusters.forEach((cluster, i) => {
      // Find nodes belonging to this cluster
      const clusterNodes = nodes.filter(n =>
        cluster.members.some(m => m.beliefHash === n.id)
      );

      if (clusterNodes.length < 3) return;

      // Get node positions (use current simulation positions)
      const points: [number, number][] = clusterNodes.map(n => [
        (n.x ?? 0),
        (n.y ?? 0)
      ]);

      // Add padding points for a larger hull
      const paddedPoints = expandHullPoints(points, 25);

      // Compute convex hull
      const hull = d3.polygonHull(paddedPoints);
      if (!hull) return;

      // Draw hull path
      hullGroup.append('path')
        .attr('class', 'cluster-hull')
        .attr('data-cluster', i)
        .attr('d', `M${hull.join('L')}Z`)
        .attr('fill', CLUSTER_COLORS[i % CLUSTER_COLORS.length])
        .attr('stroke', CLUSTER_BORDER_COLORS[i % CLUSTER_BORDER_COLORS.length])
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '5,5');

      // Add cluster label at centroid
      const centroid = d3.polygonCentroid(hull);
      hullGroup.append('text')
        .attr('class', 'cluster-label')
        .attr('data-cluster', i)
        .attr('x', centroid[0])
        .attr('y', centroid[1] - 20)
        .attr('text-anchor', 'middle')
        .attr('fill', CLUSTER_BORDER_COLORS[i % CLUSTER_BORDER_COLORS.length].replace('0.5', '0.9'))
        .attr('font-size', '10px')
        .attr('font-weight', '600')
        .text(cluster.label || `Cluster ${i + 1}`);
    });
  }

  // Performance-optimized hull update (only updates positions, doesn't recreate elements)
  function updateClusterHulls(
    g: d3.Selection<SVGGElement, unknown, null, undefined>,
    nodes: D3Node[]
  ): void {
    if (!showClusters || clusters.length === 0) return;

    const hullGroup = g.select('.cluster-hulls');
    if (hullGroup.empty()) {
      // If no hull group exists, create it
      drawClusterHulls(g, nodes);
      return;
    }

    clusters.forEach((cluster, i) => {
      const clusterNodes = nodes.filter(n =>
        cluster.members.some(m => m.beliefHash === n.id)
      );

      if (clusterNodes.length < 3) return;

      const points: [number, number][] = clusterNodes.map(n => [
        (n.x ?? 0),
        (n.y ?? 0)
      ]);

      const paddedPoints = expandHullPoints(points, 25);
      const hull = d3.polygonHull(paddedPoints);
      if (!hull) return;

      // Update existing hull path (no DOM element creation)
      hullGroup.select(`path[data-cluster="${i}"]`)
        .attr('d', `M${hull.join('L')}Z`);

      // Update label position
      const centroid = d3.polygonCentroid(hull);
      hullGroup.select(`text[data-cluster="${i}"]`)
        .attr('x', centroid[0])
        .attr('y', centroid[1] - 20);
    });
  }

  // Expand points outward for a larger hull
  function expandHullPoints(points: [number, number][], padding: number): [number, number][] {
    if (points.length === 0) return [];

    // Calculate centroid
    const cx = points.reduce((sum, p) => sum + p[0], 0) / points.length;
    const cy = points.reduce((sum, p) => sum + p[1], 0) / points.length;

    // Expand each point away from centroid
    return points.map(([x, y]) => {
      const dx = x - cx;
      const dy = y - cy;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      return [
        x + (dx / dist) * padding,
        y + (dy / dist) * padding
      ] as [number, number];
    });
  }

  function highlightConnections(
    d: D3Node,
    links: D3Link[],
    nodeSelection: d3.Selection<d3.BaseType | SVGGElement, D3Node, SVGGElement, unknown>,
    linkSelection: d3.Selection<d3.BaseType | SVGLineElement, D3Link, SVGGElement, unknown>
  ) {
    const connectedIds = new Set<string>();
    connectedIds.add(d.id);

    links.forEach((l) => {
      const sourceId = typeof l.source === 'string' ? l.source : (l.source as D3Node).id;
      const targetId = typeof l.target === 'string' ? l.target : (l.target as D3Node).id;
      if (sourceId === d.id) connectedIds.add(targetId);
      if (targetId === d.id) connectedIds.add(sourceId);
    });

    nodeSelection.attr('opacity', (n) => connectedIds.has(n.id) ? 1 : 0.2);
    linkSelection.attr('opacity', (l) => {
      const sourceId = typeof l.source === 'string' ? l.source : (l.source as D3Node).id;
      const targetId = typeof l.target === 'string' ? l.target : (l.target as D3Node).id;
      return sourceId === d.id || targetId === d.id ? 1 : 0.1;
    });
  }

  function resetHighlights(
    nodeSelection: d3.Selection<d3.BaseType | SVGGElement, D3Node, SVGGElement, unknown>,
    linkSelection: d3.Selection<d3.BaseType | SVGLineElement, D3Link, SVGGElement, unknown>
  ) {
    nodeSelection.attr('opacity', 1);
    linkSelection.attr('opacity', (d) => d.type === 'related' ? 0.3 : 0.6);
  }

  // Drag behavior
  function drag(simulation: d3.Simulation<D3Node, D3Link>) {
    function dragstarted(event: d3.D3DragEvent<SVGGElement, D3Node, D3Node>) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }

    function dragged(event: d3.D3DragEvent<SVGGElement, D3Node, D3Node>) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }

    function dragended(event: d3.D3DragEvent<SVGGElement, D3Node, D3Node>) {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }

    return d3.drag<SVGGElement, D3Node>()
      .on('start', dragstarted)
      .on('drag', dragged)
      .on('end', dragended);
  }

  // ============================================================================
  // LIFECYCLE
  // ============================================================================

  $: if (container && collectiveView) {
    initGraph();
  }

  onMount(() => {
    if (collectiveView) {
      initGraph();
    }
  });

  onDestroy(() => {
    if (simulation) simulation.stop();
  });
</script>

<div class="reality-map-container">
  <!-- Controls -->
  <div class="map-controls">
    <label class="cluster-toggle">
      <input type="checkbox" bind:checked={showClusters} />
      <span>Show Semantic Clusters</span>
      {#if clusteringInProgress}
        <span class="clustering-indicator">...</span>
      {:else if clusters.length > 0}
        <span class="cluster-count">({clusters.length})</span>
      {/if}
    </label>
  </div>

  <div class="reality-map" bind:this={container}></div>
</div>

<!-- Legend -->
<div class="map-legend">
  <div class="legend-title">Legend</div>
  <div class="legend-items">
    <div class="legend-item">
      <span class="legend-circle" style="background: #22c55e"></span>
      <span>Strong consensus</span>
    </div>
    <div class="legend-item">
      <span class="legend-circle" style="background: #3b82f6"></span>
      <span>Moderate</span>
    </div>
    <div class="legend-item">
      <span class="legend-circle" style="background: #f59e0b"></span>
      <span>Weak</span>
    </div>
    <div class="legend-item">
      <span class="legend-circle" style="background: #ef4444"></span>
      <span>Contested</span>
    </div>
    <div class="legend-item">
      <span class="legend-hexagon"></span>
      <span>Pattern</span>
    </div>
    {#if showClusters && clusters.length > 0}
      <div class="legend-divider"></div>
      <div class="legend-section-title">Semantic Clusters</div>
      {#each clusters.slice(0, 4) as cluster, i}
        <div class="legend-item">
          <span class="legend-cluster" style="background: {CLUSTER_COLORS[i]}; border-color: {CLUSTER_BORDER_COLORS[i]}"></span>
          <span class="cluster-name">{cluster.label || `Cluster ${i + 1}`}</span>
        </div>
      {/each}
      {#if clusters.length > 4}
        <div class="legend-item muted">
          <span>+{clusters.length - 4} more</span>
        </div>
      {/if}
    {/if}
  </div>
</div>

<style>
  .reality-map-container {
    display: flex;
    flex-direction: column;
    height: 100%;
    position: relative;
  }

  .map-controls {
    display: flex;
    justify-content: flex-end;
    padding: 8px 12px;
    background: var(--bg-secondary, #1e293b);
    border-radius: 8px 8px 0 0;
    border: 1px solid var(--border, #334155);
    border-bottom: none;
  }

  .cluster-toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--text-muted, #94a3b8);
    cursor: pointer;
  }

  .cluster-toggle input[type="checkbox"] {
    accent-color: var(--color-primary, #3b82f6);
  }

  .clustering-indicator {
    color: var(--color-primary, #3b82f6);
    animation: pulse 1s infinite;
  }

  .cluster-count {
    color: var(--text-muted, #64748b);
    font-size: 11px;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  .reality-map {
    width: 100%;
    flex: 1;
    min-height: 350px;
    background: #0f0f1a;
    border-radius: 0 0 8px 8px;
    border: 1px solid var(--border, #334155);
    overflow: hidden;
    position: relative;
  }

  .reality-map :global(.empty-graph) {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--text-muted, #94a3b8);
    text-align: center;
  }

  .reality-map :global(.empty-icon) {
    font-size: 48px;
    opacity: 0.3;
    margin-bottom: 8px;
  }

  .reality-map :global(.hint) {
    font-size: 12px;
    opacity: 0.7;
  }

  .reality-map :global(.node-label) {
    opacity: 0;
    transition: opacity 0.2s;
    pointer-events: none;
  }

  .reality-map :global(g:hover .node-label) {
    opacity: 1;
  }

  .reality-map :global(circle),
  .reality-map :global(polygon) {
    transition: stroke-width 0.2s, stroke 0.2s;
  }

  .reality-map :global(g:hover circle),
  .reality-map :global(g:hover polygon) {
    stroke-width: 3px;
    stroke: #7c3aed;
  }

  /* Legend */
  .map-legend {
    position: absolute;
    bottom: 12px;
    left: 12px;
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid var(--border, #334155);
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 11px;
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
    gap: 4px;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--text-muted, #94a3b8);
  }

  .legend-circle {
    width: 10px;
    height: 10px;
    border-radius: 50%;
  }

  .legend-hexagon {
    width: 10px;
    height: 10px;
    background: #a855f7;
    clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
  }

  .legend-divider {
    height: 1px;
    background: var(--border, #334155);
    margin: 6px 0;
  }

  .legend-section-title {
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted, #64748b);
    margin-bottom: 4px;
  }

  .legend-cluster {
    width: 12px;
    height: 12px;
    border-radius: 3px;
    border: 1px dashed;
  }

  .cluster-name {
    max-width: 100px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .legend-item.muted {
    font-size: 10px;
    color: var(--text-muted, #64748b);
    font-style: italic;
  }
</style>
