// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Knowledge SDK - Visualization Components
 *
 * Rich visual components for displaying knowledge graph data,
 * epistemic classifications, and credibility information.
 *
 * @example
 * ```tsx
 * import {
 *   BeliefGraph,
 *   EpistemicCube,
 *   ClaimCard,
 *   CredibilityBadge,
 *   VerdictIndicator,
 *   EpistemicPositionBar,
 * } from '@mycelix/knowledge-sdk/visualizations';
 * ```
 */

// Graph Visualizations
export { BeliefGraph } from './BeliefGraph';
export type { BeliefGraphProps, GraphNode, GraphEdge } from './BeliefGraph';

export { EpistemicCube } from './EpistemicCube';
export type { EpistemicCubeProps, CubePoint } from './EpistemicCube';

// Indicator Components
export { CredibilityBadge } from './CredibilityBadge';
export type { CredibilityBadgeProps } from './CredibilityBadge';

export { VerdictIndicator } from './VerdictIndicator';
export type { VerdictIndicatorProps, Verdict } from './VerdictIndicator';

export { EpistemicPositionBar } from './EpistemicPositionBar';
export type { EpistemicPositionBarProps } from './EpistemicPositionBar';

// Card Components
export { ClaimCard } from './ClaimCard';
export type { ClaimCardProps, ClaimData, CredibilityData, RelationshipData } from './ClaimCard';
