// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Belief Graph Service - Operations for belief propagation and dependency analysis
 *
 * Provides specialized operations for working with the belief graph,
 * including propagation, impact analysis, and visualization support.
 */

import { AppClient, ActionHash } from "@holochain/client";
import {
  KnowledgeClient,
  BeliefNode,
  PropagationResult,
  DependencyTree,
  DependencyTreeNode,
  CascadeImpact,
  InformationValue,
  ClaimDependency,
  DependencyType,
  InfluenceDirection,
} from "../index";

/**
 * Visualization-ready node for belief graph rendering
 */
export interface GraphNode {
  id: string;
  label: string;
  beliefStrength: number;
  confidence: number;
  x?: number;
  y?: number;
  size: number;
  color: string;
}

/**
 * Visualization-ready edge for belief graph rendering
 */
export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  weight: number;
  type: DependencyType;
  influence: InfluenceDirection;
  color: string;
}

/**
 * Complete graph structure for visualization
 */
export interface VisualizationGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: {
    totalNodes: number;
    totalEdges: number;
    averageBelief: number;
    maxDepth: number;
    strongestConnection: number;
  };
}

/**
 * Result of analyzing belief consistency
 */
export interface ConsistencyAnalysis {
  isConsistent: boolean;
  inconsistencies: Array<{
    claimId: string;
    expectedBelief: number;
    actualBelief: number;
    deviation: number;
    reason: string;
  }>;
  overallConsistencyScore: number;
  recommendations: string[];
}

/**
 * Path through the belief graph
 */
export interface BeliefPath {
  nodes: string[];
  totalWeight: number;
  averageInfluence: number;
  pathType: "support" | "contradiction" | "mixed";
}

/**
 * Specialized service for belief graph operations
 */
export class BeliefGraphService {
  private client: KnowledgeClient;

  constructor(appClient: AppClient, roleName: string = "knowledge") {
    this.client = new KnowledgeClient(appClient, roleName);
  }

  /**
   * Get visualization-ready graph centered on a claim
   */
  async getVisualizationGraph(
    claimId: string,
    maxDepth: number = 3,
    maxNodes: number = 50
  ): Promise<VisualizationGraph> {
    const tree = await this.client.graph.getDependencyTree(claimId, maxDepth);
    const nodes: GraphNode[] = [];
    const edges: GraphEdge[] = [];
    const visitedNodes = new Set<string>();

    // Process tree nodes
    for (const node of tree.nodes.slice(0, maxNodes)) {
      if (visitedNodes.has(node.claimId)) continue;
      visitedNodes.add(node.claimId);

      // Get belief node data if available
      let beliefNode: BeliefNode | null = null;
      try {
        beliefNode = await this.client.graph.getBeliefNode(node.claimId);
      } catch {
        // Node may not have belief data yet
      }

      nodes.push({
        id: node.claimId,
        label: node.claimId.substring(0, 8) + "...",
        beliefStrength: beliefNode?.beliefStrength ?? 0.5,
        confidence: beliefNode?.confidence ?? 0.5,
        size: Math.max(10, node.weight * 50),
        color: this.getNodeColor(beliefNode?.beliefStrength ?? 0.5),
      });

      // Add edges to children
      for (const childId of node.children) {
        edges.push({
          id: `${node.claimId}-${childId}`,
          source: node.claimId,
          target: childId,
          weight: node.weight,
          type: "Supports", // Default, would need to fetch actual type
          influence: "Positive",
          color: this.getEdgeColor("Positive", node.weight),
        });
      }
    }

    // Calculate stats
    const beliefs = nodes.map((n) => n.beliefStrength);
    const avgBelief =
      beliefs.length > 0
        ? beliefs.reduce((a, b) => a + b, 0) / beliefs.length
        : 0.5;

    return {
      nodes,
      edges,
      stats: {
        totalNodes: nodes.length,
        totalEdges: edges.length,
        averageBelief: avgBelief,
        maxDepth: tree.depth,
        strongestConnection: Math.max(...edges.map((e) => e.weight), 0),
      },
    };
  }

  /**
   * Propagate belief and return detailed analysis
   */
  async propagateWithAnalysis(claimId: string): Promise<{
    result: PropagationResult;
    beforeState: Map<string, number>;
    afterState: Map<string, number>;
    largestChanges: Array<{ claimId: string; change: number }>;
  }> {
    // Get current state
    const tree = await this.client.graph.getDependencyTree(claimId, 5);
    const beforeState = new Map<string, number>();

    for (const node of tree.nodes) {
      try {
        const belief = await this.client.graph.getBeliefNode(node.claimId);
        if (belief) {
          beforeState.set(node.claimId, belief.beliefStrength);
        }
      } catch {
        // Skip
      }
    }

    // Propagate
    const result = await this.client.graph.propagateBelief(claimId);

    // Get after state
    const afterState = new Map<string, number>();
    const changes: Array<{ claimId: string; change: number }> = [];

    for (const nodeId of result.updatedNodes) {
      try {
        const belief = await this.client.graph.getBeliefNode(nodeId);
        if (belief) {
          afterState.set(nodeId, belief.beliefStrength);
          const before = beforeState.get(nodeId) ?? 0.5;
          changes.push({
            claimId: nodeId,
            change: Math.abs(belief.beliefStrength - before),
          });
        }
      } catch {
        // Skip
      }
    }

    // Sort by largest changes
    changes.sort((a, b) => b.change - a.change);

    return {
      result,
      beforeState,
      afterState,
      largestChanges: changes.slice(0, 10),
    };
  }

  /**
   * Analyze consistency of beliefs in a subgraph
   */
  async analyzeConsistency(claimId: string): Promise<ConsistencyAnalysis> {
    const tree = await this.client.graph.getDependencyTree(claimId, 4);
    const inconsistencies: ConsistencyAnalysis["inconsistencies"] = [];
    const recommendations: string[] = [];

    // Check each node for consistency with its dependencies
    for (const node of tree.nodes) {
      if (node.isLeaf) continue;

      try {
        const beliefNode = await this.client.graph.getBeliefNode(node.claimId);
        if (!beliefNode) continue;

        // Calculate expected belief from influences
        let expectedBelief = beliefNode.priorBelief;
        for (const influence of beliefNode.influences) {
          if (influence.influenceType === "Support") {
            expectedBelief += influence.weight * influence.sourceBelief * 0.3;
          } else if (influence.influenceType === "Contradiction") {
            expectedBelief -= influence.weight * influence.sourceBelief * 0.3;
          }
        }
        expectedBelief = Math.max(0, Math.min(1, expectedBelief));

        const deviation = Math.abs(beliefNode.beliefStrength - expectedBelief);

        if (deviation > 0.2) {
          inconsistencies.push({
            claimId: node.claimId,
            expectedBelief,
            actualBelief: beliefNode.beliefStrength,
            deviation,
            reason: `Belief ${beliefNode.beliefStrength.toFixed(2)} deviates from expected ${expectedBelief.toFixed(2)} based on influences`,
          });
        }
      } catch {
        // Skip nodes without belief data
      }
    }

    // Generate recommendations
    if (inconsistencies.length > 0) {
      recommendations.push(
        `Found ${inconsistencies.length} inconsistencies. Consider re-propagating beliefs.`
      );
      if (inconsistencies.some((i) => i.deviation > 0.4)) {
        recommendations.push(
          "Some claims have severe inconsistencies. Review dependency weights."
        );
      }
    }

    // Check for circular dependencies
    const cycles = await this.client.graph.detectCircularDependencies(claimId);
    if (cycles.length > 0) {
      recommendations.push(
        `Detected ${cycles.length} circular dependency chains. Consider breaking cycles.`
      );
    }

    const overallConsistencyScore =
      inconsistencies.length === 0
        ? 1.0
        : 1.0 -
          inconsistencies.reduce((acc, i) => acc + i.deviation, 0) /
            (inconsistencies.length * 1.0);

    return {
      isConsistent: inconsistencies.length === 0,
      inconsistencies,
      overallConsistencyScore,
      recommendations,
    };
  }

  /**
   * Find all paths between two claims
   */
  async findPaths(
    sourceId: string,
    targetId: string,
    maxDepth: number = 5
  ): Promise<BeliefPath[]> {
    const paths: BeliefPath[] = [];
    const visited = new Set<string>();

    const dfs = async (
      currentId: string,
      path: string[],
      totalWeight: number,
      influences: InfluenceDirection[]
    ) => {
      if (path.length > maxDepth) return;
      if (currentId === targetId) {
        // Determine path type
        const hasPositive = influences.includes("Positive");
        const hasNegative = influences.includes("Negative");
        const pathType =
          hasPositive && hasNegative
            ? "mixed"
            : hasNegative
              ? "contradiction"
              : "support";

        paths.push({
          nodes: [...path, currentId],
          totalWeight,
          averageInfluence: totalWeight / path.length,
          pathType,
        });
        return;
      }

      if (visited.has(currentId)) return;
      visited.add(currentId);

      // Get outgoing relationships
      try {
        const relationships = await this.client.graph.getOutgoingRelationships(
          currentId
        );
        for (const rel of relationships) {
          const influence: InfluenceDirection =
            rel.relationshipType === "Contradicts" ? "Negative" : "Positive";
          await dfs(
            rel.target,
            [...path, currentId],
            totalWeight + rel.weight,
            [...influences, influence]
          );
        }
      } catch {
        // No outgoing relationships
      }

      visited.delete(currentId);
    };

    await dfs(sourceId, [], 0, []);

    // Sort by total weight (strongest paths first)
    paths.sort((a, b) => b.totalWeight - a.totalWeight);

    return paths.slice(0, 10); // Return top 10 paths
  }

  /**
   * Get claims with highest potential impact on the graph
   */
  async getHighImpactClaims(
    limit: number = 10
  ): Promise<
    Array<{
      claimId: string;
      impact: CascadeImpact;
      informationValue?: InformationValue;
    }>
  > {
    const ivRecords = await this.client.graph.rankByInformationValue(limit * 2);
    const results: Array<{
      claimId: string;
      impact: CascadeImpact;
      informationValue?: InformationValue;
    }> = [];

    for (const iv of ivRecords) {
      try {
        const impact = await this.client.graph.calculateCascadeImpact(iv.claimId);
        results.push({
          claimId: iv.claimId,
          impact,
          informationValue: iv,
        });
      } catch {
        // Skip claims without impact data
      }

      if (results.length >= limit) break;
    }

    // Sort by impact score
    results.sort((a, b) => b.impact.impactScore - a.impact.impactScore);

    return results;
  }

  /**
   * Simulate what would happen if a claim's belief changed
   */
  async simulateBeliefChange(
    claimId: string,
    newBelief: number
  ): Promise<{
    affectedClaims: number;
    estimatedChanges: Array<{ claimId: string; currentBelief: number; estimatedNewBelief: number }>;
    cascadeDepth: number;
    warning?: string;
  }> {
    const impact = await this.client.graph.calculateCascadeImpact(claimId);
    const tree = await this.client.graph.getDependencyTree(claimId, 4);

    // Get current belief
    const currentNode = await this.client.graph.getBeliefNode(claimId);
    const currentBelief = currentNode?.beliefStrength ?? 0.5;
    const beliefDelta = newBelief - currentBelief;

    // Estimate changes for dependent claims
    const estimatedChanges: Array<{
      claimId: string;
      currentBelief: number;
      estimatedNewBelief: number;
    }> = [];

    for (const node of tree.nodes.slice(0, 20)) {
      if (node.claimId === claimId) continue;

      try {
        const depNode = await this.client.graph.getBeliefNode(node.claimId);
        if (depNode) {
          // Simple estimation: change proportional to weight and depth
          const depthFactor = 1 / (node.depth + 1);
          const estimatedDelta = beliefDelta * node.weight * depthFactor * 0.5;
          const estimatedNew = Math.max(
            0,
            Math.min(1, depNode.beliefStrength + estimatedDelta)
          );

          estimatedChanges.push({
            claimId: node.claimId,
            currentBelief: depNode.beliefStrength,
            estimatedNewBelief: estimatedNew,
          });
        }
      } catch {
        // Skip
      }
    }

    // Generate warning if impact is high
    let warning: string | undefined;
    if (impact.totalAffected > 100) {
      warning = `This change would affect ${impact.totalAffected} claims. Proceed with caution.`;
    } else if (impact.highImpactClaims.length > 5) {
      warning = `This change would significantly impact ${impact.highImpactClaims.length} high-importance claims.`;
    }

    return {
      affectedClaims: impact.totalAffected,
      estimatedChanges,
      cascadeDepth: impact.maxDepth,
      warning,
    };
  }

  /**
   * Get color for a belief node based on belief strength
   */
  private getNodeColor(beliefStrength: number): string {
    // Green for high belief, red for low, yellow for uncertain
    if (beliefStrength > 0.7) {
      return `rgba(76, 175, 80, ${0.5 + beliefStrength * 0.5})`;
    } else if (beliefStrength < 0.3) {
      return `rgba(244, 67, 54, ${0.5 + (1 - beliefStrength) * 0.5})`;
    } else {
      return `rgba(255, 193, 7, 0.8)`;
    }
  }

  /**
   * Get color for an edge based on influence and weight
   */
  private getEdgeColor(influence: InfluenceDirection, weight: number): string {
    const opacity = 0.3 + weight * 0.7;
    switch (influence) {
      case "Positive":
        return `rgba(76, 175, 80, ${opacity})`;
      case "Negative":
        return `rgba(244, 67, 54, ${opacity})`;
      default:
        return `rgba(158, 158, 158, ${opacity})`;
    }
  }
}

export default BeliefGraphService;
