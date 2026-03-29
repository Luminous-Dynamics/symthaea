// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Audio Visualization Components
 *
 * Powered by mycelix-wasm for high-performance audio processing.
 */

export { WaveformViewer } from './WaveformViewer';
export type { WaveformViewerProps } from './WaveformViewer';

export { SpectrumAnalyzer } from './SpectrumAnalyzer';
export type { SpectrumAnalyzerProps } from './SpectrumAnalyzer';

// Re-export visualization types
export type { VisualizationType, WaveformData, AudioAnalysisResult } from '@/hooks/useWasmAudio';
