// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Epistemic Markets — stub exports for build compatibility.
 * TODO: Implement market prediction components.
 */

// Stub Svelte components as empty divs for now
export const MarketCard = null;
export const PredictionForm = null;
export const CalibrationProfile = null;
export const WisdomSeedDisplay = null;

export interface Market {
  id: string;
  question: string;
  probability: number;
  volume: number;
  status: 'open' | 'closed' | 'resolved';
  created_at: number;
}

export interface CalibrationProfile {
  agent_did: string;
  brier_score: number;
  total_predictions: number;
  calibration_curve: number[];
}
