// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Lazy Component Loading
 * Optimized code splitting for heavy components
 */

import dynamic from 'next/dynamic';
import { ComponentType, ReactNode } from 'react';

// Loading component for suspense
const LoadingSpinner = () => (
  <div className="flex items-center justify-center p-8">
    <div className="animate-spin rounded-full h-8 w-8 border-2 border-primary-500 border-t-transparent" />
  </div>
);

const LoadingCard = () => (
  <div className="animate-pulse bg-gray-200 dark:bg-gray-800 rounded-lg h-48" />
);

const LoadingPage = () => (
  <div className="flex items-center justify-center min-h-screen">
    <div className="flex flex-col items-center gap-4">
      <div className="animate-spin rounded-full h-12 w-12 border-2 border-primary-500 border-t-transparent" />
      <p className="text-gray-500 dark:text-gray-400">Loading...</p>
    </div>
  </div>
);

// Studio Components (heavy, code-split)
export const DAWWorkspace = dynamic(
  () => import('@/components/studio/DAWWorkspace').then(mod => mod.DAWWorkspace),
  { loading: LoadingPage, ssr: false }
);

export const Mixer = dynamic(
  () => import('@/components/studio/Mixer').then(mod => mod.Mixer),
  { loading: LoadingSpinner, ssr: false }
);

export const WaveformEditor = dynamic(
  () => import('@/components/studio/WaveformEditor').then(mod => mod.default),
  { loading: LoadingSpinner, ssr: false }
);

export const MIDIEditor = dynamic(
  () => import('@/components/studio/MIDIEditor').then(mod => mod.default),
  { loading: LoadingSpinner, ssr: false }
);

export const EffectsRack = dynamic(
  () => import('@/components/studio/EffectsRack').then(mod => mod.default),
  { loading: LoadingSpinner, ssr: false }
);

// Player Components
export const EnhancedPlayer = dynamic(
  () => import('@/components/player/EnhancedPlayer').then(mod => mod.EnhancedPlayer),
  { loading: LoadingSpinner, ssr: false }
);

export const Visualizer = dynamic(
  () => import('@/components/player/Visualizer').then(mod => mod.default),
  { loading: LoadingSpinner, ssr: false }
);

export const QueuePanel = dynamic(
  () => import('@/components/player/QueuePanel').then(mod => mod.default),
  { loading: LoadingCard }
);

// Analytics Components
export const AnalyticsDashboard = dynamic(
  () => import('@/components/analytics/Dashboard').then(mod => mod.default),
  { loading: LoadingPage }
);

export const EarningsChart = dynamic(
  () => import('@/components/analytics/EarningsChart').then(mod => mod.default),
  { loading: LoadingCard }
);

export const ListenerMap = dynamic(
  () => import('@/components/analytics/ListenerMap').then(mod => mod.default),
  { loading: LoadingCard, ssr: false }
);

// AI/ML Components (very heavy)
export const AIGenerator = dynamic(
  () => import('@/components/ai/AIGenerator').then(mod => mod.default),
  { loading: LoadingPage, ssr: false }
);

export const StemSeparator = dynamic(
  () => import('@/components/ai/StemSeparator').then(mod => mod.default),
  { loading: LoadingSpinner, ssr: false }
);

export const AIMastering = dynamic(
  () => import('@/components/ai/AIMastering').then(mod => mod.default),
  { loading: LoadingSpinner, ssr: false }
);

// Immersive/WebXR Components
export const ImmersivePlayer = dynamic(
  () => import('@/components/immersive/ImmersivePlayer').then(mod => mod.default),
  { loading: LoadingPage, ssr: false }
);

export const SpatialAudioMixer = dynamic(
  () => import('@/components/immersive/SpatialAudioMixer').then(mod => mod.default),
  { loading: LoadingSpinner, ssr: false }
);

export const VRStudio = dynamic(
  () => import('@/components/immersive/VRStudio').then(mod => mod.default),
  { loading: LoadingPage, ssr: false }
);

// Economic Components
export const EconomicStrategyWizard = dynamic(
  () => import('@/components/economics/EconomicStrategyWizard').then(mod => mod.default),
  { loading: LoadingPage }
);

export const EconomicsLab = dynamic(
  () => import('@/components/economics/EconomicsLab').then(mod => mod.default),
  { loading: LoadingPage }
);

export const RoyaltySplitter = dynamic(
  () => import('@/components/economics/RoyaltySplitter').then(mod => mod.default),
  { loading: LoadingCard }
);

// Social Components
export const CommentSection = dynamic(
  () => import('@/components/social/CommentSection').then(mod => mod.default),
  { loading: LoadingCard }
);

export const ActivityFeed = dynamic(
  () => import('@/components/social/ActivityFeed').then(mod => mod.default),
  { loading: LoadingCard }
);

export const CollaborationRoom = dynamic(
  () => import('@/components/collaboration/CollaborationRoom').then(mod => mod.default),
  { loading: LoadingPage, ssr: false }
);

// Live Streaming Components
export const LiveStreamPlayer = dynamic(
  () => import('@/components/live/LiveStreamPlayer').then(mod => mod.default),
  { loading: LoadingPage, ssr: false }
);

export const LiveStreamControls = dynamic(
  () => import('@/components/live/LiveStreamControls').then(mod => mod.default),
  { loading: LoadingSpinner, ssr: false }
);

export const LiveChat = dynamic(
  () => import('@/components/live/LiveChat').then(mod => mod.default),
  { loading: LoadingCard }
);

// NFT/Marketplace Components
export const NFTGallery = dynamic(
  () => import('@/components/nft/NFTGallery').then(mod => mod.default),
  { loading: LoadingCard }
);

export const NFTMinter = dynamic(
  () => import('@/components/nft/NFTMinter').then(mod => mod.default),
  { loading: LoadingSpinner }
);

export const Marketplace = dynamic(
  () => import('@/components/marketplace/Marketplace').then(mod => mod.default),
  { loading: LoadingPage }
);

// Utility for creating lazy components with custom loading
export function createLazyComponent<T extends ComponentType<any>>(
  importFn: () => Promise<{ default: T }>,
  options: {
    loading?: () => ReactNode;
    ssr?: boolean;
  } = {}
) {
  return dynamic(importFn, {
    loading: options.loading || LoadingSpinner,
    ssr: options.ssr ?? true,
  });
}

// Preload function for critical components
export function preloadComponent(componentName: keyof typeof componentMap): void {
  const componentMap = {
    DAWWorkspace: () => import('@/components/studio/DAWWorkspace'),
    Mixer: () => import('@/components/studio/Mixer'),
    EnhancedPlayer: () => import('@/components/player/EnhancedPlayer'),
    AnalyticsDashboard: () => import('@/components/analytics/Dashboard'),
    AIGenerator: () => import('@/components/ai/AIGenerator'),
  };

  const loader = componentMap[componentName];
  if (loader) {
    loader();
  }
}

// Intersection Observer-based preloading
export function usePreloadOnVisible(
  componentName: keyof typeof componentMap,
  ref: React.RefObject<HTMLElement>
): void {
  if (typeof window === 'undefined') return;

  const componentMap = {
    DAWWorkspace: () => import('@/components/studio/DAWWorkspace'),
    Mixer: () => import('@/components/studio/Mixer'),
    EnhancedPlayer: () => import('@/components/player/EnhancedPlayer'),
    AnalyticsDashboard: () => import('@/components/analytics/Dashboard'),
    AIGenerator: () => import('@/components/ai/AIGenerator'),
  };

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const loader = componentMap[componentName];
          if (loader) {
            loader();
          }
          observer.disconnect();
        }
      });
    },
    { rootMargin: '100px' }
  );

  if (ref.current) {
    observer.observe(ref.current);
  }
}

export default {
  DAWWorkspace,
  Mixer,
  EnhancedPlayer,
  AnalyticsDashboard,
  AIGenerator,
  StemSeparator,
  ImmersivePlayer,
  EconomicStrategyWizard,
  CollaborationRoom,
  LiveStreamPlayer,
  NFTGallery,
  Marketplace,
};
