// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * AI Components
 *
 * React components for AI-powered features:
 * - AudioAnalyzer - Analyze audio files with AI
 * - ModelTrainer - Train custom models
 */

export { AudioAnalyzer } from './AudioAnalyzer';
export { ModelTrainer } from './ModelTrainer';

export default {
  AudioAnalyzer: () => import('./AudioAnalyzer'),
  ModelTrainer: () => import('./ModelTrainer'),
};
