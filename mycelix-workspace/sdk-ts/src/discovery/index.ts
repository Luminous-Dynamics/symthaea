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
