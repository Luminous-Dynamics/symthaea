// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Calibration Module
 *
 * The keystone of Self-Correcting Epistemic Infrastructure (SCEI).
 *
 * Calibration answers: "When we said 70%, were we right 70% of the time?"
 * Without calibration, the system becomes a confident bullshitter.
 *
 * @packageDocumentation
 * @module calibration
 */

// Types
export {
  type CalibrationBucket,
  type CalibrationConfig,
  type CalibrationReport,
  type CalibrationScope,
  type ResolutionRecord,
  type ResolutionEvidence,
  type AdjustedConfidence,
  type CredibilityCheck,
  DEFAULT_CALIBRATION_CONFIG,
  InsufficientCalibrationDataError,
} from './types.js';

// Engine
export {
  CalibrationEngine,
  type CalibrationStats,
  getCalibrationEngine,
  resetCalibrationEngine,
} from './engine.js';
