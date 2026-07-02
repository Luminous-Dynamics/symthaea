// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Components Index
 *
 * Central export for all components
 */

// Layout
export { Sidebar } from './layout/Sidebar';

// DJ & Studio
export { DJMixer } from './dj';
export { StemMixer } from './stems';
export { EffectsPanel } from './effects';
export { BroadcastStudio } from './broadcast';

// Discovery & Recommendations
export { DiscoveryPanel } from './discovery';

// Voice Control
export { VoiceControl } from './voice';

// Offline & PWA
export { OfflineStatus } from './offline';

// Analytics
export { AnalyticsDashboard } from './analytics';

// Listening Party
export { ListeningParty } from './listening-party';

// Notifications
export { NotificationCenter, NotificationBell } from './notifications';

// Audio Pipeline
export { AudioPipelineProvider, useAudioPipeline, AudioLevelMeter } from './audio';

// AI Music Generation
export { BeatGenerator, StyleTransfer } from './generation';

// Collaborative Remix Studio
export { RemixStudio } from './remix';

// Immersive Audio/Visuals
export { SpatialVenue, AudioVisualizer3D } from './immersive';

// Music Intelligence
export { MasteringPanel, TheoryAssistant } from './intelligence';
