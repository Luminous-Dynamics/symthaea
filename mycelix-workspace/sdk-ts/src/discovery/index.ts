// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Discovery Module
 *
 * Tools for finding unknown unknowns in the knowledge graph.
 *
 * Key principle: All gap detection comes with humility flags
 * acknowledging the limitations of our detection methods.
 *
 * @packageDocumentation
 * @module discovery
 */

export {
  type GapType,
  type HumilityFlags,
  type KnowledgeGap,
  type SuggestedAction,
  type GapResolution,
  type GapDetectorConfig,
  DEFAULT_GAP_DETECTOR_CONFIG,
  GapDetector,
  type GapDetectorStats,
  getGapDetector,
  resetGapDetector,
} from './gap-detector.js';
