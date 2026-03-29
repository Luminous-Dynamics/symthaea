// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Knowledge SDK - Services
 *
 * High-level service abstractions that compose multiple client operations.
 */

export { KnowledgeService, default as KnowledgeServiceDefault } from "./knowledge-service";
export type {
  ClaimSubmissionResult,
  ComprehensiveFactCheckResult,
  EnrichedClaim,
} from "./knowledge-service";

export { BeliefGraphService, default as BeliefGraphServiceDefault } from "./belief-graph";
export type {
  GraphNode,
  GraphEdge,
  VisualizationGraph,
  ConsistencyAnalysis,
  BeliefPath,
} from "./belief-graph";
