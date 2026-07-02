// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Lazy Component Loader
 *
 * Code splitting configuration for heavy components.
 * Reduces initial bundle size by loading components on demand.
 */

import dynamic from 'next/dynamic';
import { ComponentType } from 'react';

// Loading fallbacks
const LoadingSpinner = () => (
  <div className="flex items-center justify-center p-8">
    <div className="w-8 h-8 border-2 border-purple-500 border-t-transparent rounded-full animate-spin" />
  </div>
);

const LoadingPanel = () => (
  <div className="bg-white/5 rounded-xl p-8 animate-pulse">
    <div className="h-8 bg-white/10 rounded w-1/3 mb-4" />
    <div className="h-64 bg-white/10 rounded" />
  </div>
);

// ============================================================================
// Studio Components (Heavy - load on demand)
// ============================================================================

export const LazyWaveform = dynamic(
  () => import('@/components/player/Waveform').then(mod => mod.Waveform),
  { loading: LoadingSpinner, ssr: false }
);

export const LazyAudioVisualizer = dynamic(
  () => import('@/components/visualizer/AudioVisualizer').then(mod => mod.AudioVisualizer),
  { loading: LoadingSpinner, ssr: false }
);

export const LazyAudioVisualizer3D = dynamic(
  () => import('@/components/immersive/AudioVisualizer3D').then(mod => mod.AudioVisualizer3D),
  { loading: LoadingPanel, ssr: false }
);

export const LazySpatialVenue = dynamic(
  () => import('@/components/immersive/SpatialVenue').then(mod => mod.SpatialVenue),
  { loading: LoadingPanel, ssr: false }
);

export const LazyBeatGenerator = dynamic(
  () => import('@/components/studio/BeatGenerator').then(mod => mod.BeatGenerator),
  { loading: LoadingPanel, ssr: false }
);

export const LazyStyleTransfer = dynamic(
  () => import('@/components/studio/StyleTransfer').then(mod => mod.StyleTransfer),
  { loading: LoadingPanel, ssr: false }
);

export const LazyRemixStudio = dynamic(
  () => import('@/components/studio/RemixStudio').then(mod => mod.RemixStudio),
  { loading: LoadingPanel, ssr: false }
);

export const LazyMasteringPanel = dynamic(
  () => import('@/components/studio/MasteringPanel').then(mod => mod.MasteringPanel),
  { loading: LoadingPanel, ssr: false }
);

export const LazyTheoryAssistant = dynamic(
  () => import('@/components/studio/TheoryAssistant').then(mod => mod.TheoryAssistant),
  { loading: LoadingPanel, ssr: false }
);

// ============================================================================
// Dashboard Components
// ============================================================================

export const LazyAnalyticsChart = dynamic(
  () => import('@/components/dashboard/AnalyticsChart').then(mod => mod.AnalyticsChart),
  { loading: LoadingPanel, ssr: false }
);

export const LazyRevenueChart = dynamic(
  () => import('@/components/dashboard/RevenueChart').then(mod => mod.RevenueChart),
  { loading: LoadingPanel, ssr: false }
);

// ============================================================================
// Heavy Libraries (load on demand)
// ============================================================================

// Three.js - only load when needed for 3D visualizations
export const loadThree = () => import('three');

// Tone.js - only load for audio synthesis
export const loadTone = () => import('tone');

// ============================================================================
// Preload Utilities
// ============================================================================

/**
 * Preload a component when user hovers over a link
 */
export function preloadComponent(loader: () => Promise<any>) {
  // Start loading immediately
  loader();
}

/**
 * Preload studio components when user navigates to studio section
 */
export function preloadStudioComponents() {
  // Preload in order of likely usage
  import('@/components/studio/BeatGenerator');
  import('@/components/player/Waveform');
  import('@/components/visualizer/AudioVisualizer');
}

/**
 * Preload immersive components
 */
export function preloadImmersiveComponents() {
  import('@/components/immersive/AudioVisualizer3D');
  import('@/components/immersive/SpatialVenue');
}

// ============================================================================
// Route-based Preloading
// ============================================================================

const routePreloaders: Record<string, () => void> = {
  '/studio': preloadStudioComponents,
  '/immersive': preloadImmersiveComponents,
};

/**
 * Preload components for a route
 */
export function preloadRoute(route: string) {
  const preloader = Object.entries(routePreloaders).find(
    ([prefix]) => route.startsWith(prefix)
  );

  if (preloader) {
    preloader[1]();
  }
}

export default {
  LazyWaveform,
  LazyAudioVisualizer,
  LazyAudioVisualizer3D,
  LazySpatialVenue,
  LazyBeatGenerator,
  LazyStyleTransfer,
  LazyRemixStudio,
  LazyMasteringPanel,
  LazyTheoryAssistant,
  LazyAnalyticsChart,
  LazyRevenueChart,
  preloadComponent,
  preloadStudioComponents,
  preloadImmersiveComponents,
  preloadRoute,
};
