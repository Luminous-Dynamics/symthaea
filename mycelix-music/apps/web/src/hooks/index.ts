// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Hooks Index
 *
 * Re-exports all custom hooks for convenient importing.
 */

// Core hooks
export { useSongs } from './useSongs';
export { useArtistData } from './useArtistData';
export { useWallet } from './useWallet';
export { useAuth } from './useAuth';

// Collaboration hooks
export { useCollaboration } from './useCollaboration';
export { useListeningParty } from './useListeningParty';
export type {
  Participant,
  PartySettings,
  Reaction,
  ListeningPartyState,
} from './useListeningParty';

// Audio & Analysis hooks
export { useAudioAnalysis } from './useAudioAnalysis';
export { useWasmAudio, useWasmVisualizer } from './useWasmAudio';
export type { WasmAudioState, WaveformData } from './useWasmAudio';

// P2P & Networking hooks
export { useWebRTC, getMicrophoneStream, isWebRTCSupported, getConnectionStats } from './useWebRTC';
export type {
  PeerConnection,
  IceCandidate,
  SignalingMessage,
  WebRTCState,
  UseWebRTCReturn,
} from './useWebRTC';

// Context hooks
export { useContextDetection } from './useContextDetection';
export { useMycelium } from './useMycelium';

// Rust integration
export { useRustSession } from './useRustSession';

// DJ & Audio Production hooks
export { useDJMixer } from './useDJMixer';
export { useAudioEffects } from './useAudioEffects';
export { useStemSeparation } from './useStemSeparation';
export type { StemType, Stem, SeparationProgress, StemControls } from './useStemSeparation';

// Discovery & Recommendations
export { useDiscovery } from './useDiscovery';

// Voice Control
export { useVoiceCommands } from './useVoiceCommands';
export type { VoiceCommand, CommandParams, VoiceCommandResult } from './useVoiceCommands';

// Broadcasting
export { useBroadcast, useBroadcastListener } from './useBroadcast';
export type { BroadcastConfig, Listener, ChatMessage, BroadcastStats, BroadcastState } from './useBroadcast';

// Analytics
export { useArtistAnalytics, TIME_RANGES } from './useArtistAnalytics';
export type { StreamData, TrackAnalytics, ListenerDemographics, RevenueData, ArtistInsight } from './useArtistAnalytics';

// Offline & PWA
export { useOfflineStorage } from './useOfflineStorage';

// Notifications & Social
export { useNotifications } from './useNotifications';
export type { Notification, NotificationType, NotificationPreferences } from './useNotifications';

// AI Music Generation
export { useAIGeneration } from './useAIGeneration';
export type {
  GenerationMode,
  MusicStyle,
  BeatPattern,
  DrumTrack,
  MelodySequence,
  NoteEvent,
  StyleTransferParams,
  GenerationParams,
  GenerationResult,
  AIGenerationState,
} from './useAIGeneration';

// Collaborative Remix Studio
export { useRemixStudio } from './useRemixStudio';
export type {
  Collaborator,
  CollaboratorPermissions,
  RemixProject,
  RemixTrack,
  Clip,
  Effect,
  Marker,
  ProjectVersion,
  Operation,
  OperationType,
  Comment,
  MarketplaceStem,
  RemixStudioState,
} from './useRemixStudio';

// Spatial Audio
export { useSpatialAudio } from './useSpatialAudio';
export type {
  Position3D,
  Orientation3D,
  SpatialSource,
  SpatialListener,
  RoomPreset,
  RoomAcoustics,
  VirtualVenue,
  VenueHotspot,
  SpatialAudioState,
} from './useSpatialAudio';

// Immersive Visuals
export { useImmersiveVisuals } from './useImmersiveVisuals';
export type {
  VisualizationMode,
  ColorScheme,
  VisualSettings,
  AudioAnalysis,
  ImmersiveVisualsState,
} from './useImmersiveVisuals';

// AI Mastering
export { useAIMastering } from './useAIMastering';
export type {
  MasteringPreset,
  TargetPlatform,
  MasteringChain,
  EQSettings,
  EQBand,
  CompressorSettings,
  MultibandSettings,
  SaturationSettings,
  StereoSettings,
  LimiterSettings,
  LoudnessSettings,
  AudioAnalysisResult,
  MasteringIssue,
  ReferenceComparison,
  AIMasteringState,
} from './useAIMastering';

// Music Theory
export { useMusicTheory } from './useMusicTheory';
export type {
  NoteName,
  KeyMode,
  MusicalKey,
  ChordQuality,
  Chord,
  ScaleInfo,
  ChordProgression,
  HarmonicAnalysis,
  Section,
  TheorySuggestion,
  MusicTheoryState,
} from './useMusicTheory';

// Audio Worker (Web Worker for heavy processing)
export { useAudioWorker } from './useAudioWorker';

// Keyboard Shortcuts
export { useKeyboardShortcuts, getShortcutDisplay, getDefaultShortcuts } from './useKeyboardShortcuts';
export type { Shortcut, ShortcutCategory } from './useKeyboardShortcuts';

// Undo/Redo
export { useUndoRedo, useMultiUndoRedo, useCommandHistory } from './useUndoRedo';
export type { UndoRedoState, UndoRedoActions, Command } from './useUndoRedo';

// AI Vocals
export { useAIVocals } from './useAIVocals';
export type {
  VoiceProfile,
  SynthesisParams,
  NoteEvent,
  ExpressionParams,
  TransformParams,
  HarmonyParams,
  EnhancementOptions,
  AIVocalsState,
} from './useAIVocals';

// Smart Recommendations
export { useSmartRecommendations, MOOD_PRESETS } from './useSmartRecommendations';
export type {
  Track,
  Stem,
  Sample,
  RecommendationContext,
  RecommendationResult,
  MoodDescriptor,
  SmartRecommendationsState,
} from './useSmartRecommendations';

// Accessibility
export {
  useAccessibility,
  useFocusTrap,
  useSkipLink,
  useRovingTabIndex,
  LiveRegion,
  AccessibilityContext,
} from './useAccessibility';
export type {
  AccessibilityPreferences,
  FocusTrapOptions,
  AnnouncementPriority,
  Announcement,
} from './useAccessibility';

// MIDI Controllers
export { useMIDI, MIDI_ACTIONS } from './useMIDI';
export type {
  MIDIMessageType,
  MIDIMessage,
  MIDIDevice,
  MIDIMapping,
  MIDIControllerProfile,
  MIDIState,
} from './useMIDI';

// Social Features
export { useSocialFeatures } from './useSocialFeatures';
export type {
  UserProfile,
  ArtistSkill,
  CollaborationMatch,
  CollaborationRequest,
  Challenge,
  Prize,
  ChallengeSubmission,
  Tip,
  SocialInteraction,
  Comment,
  SocialFeaturesState,
} from './useSocialFeatures';

// Mobile Gestures
export {
  useMobileGestures,
  useSwipeToDismiss,
  usePullToRefresh,
} from './useMobileGestures';
export type {
  GestureType,
  Point,
  GestureEvent,
  GestureConfig,
  GestureHandlers,
} from './useMobileGestures';

// Advanced Audio
export { useAdvancedAudio } from './useAdvancedAudio';
export type {
  SpectralRegion,
  SpectralData,
  LoudnessData,
  MultibandConfig,
  BandConfig,
  SpatialFormat,
  SpatialConfig,
  AdvancedAudioState,
} from './useAdvancedAudio';

// Monetization
export { useMonetization, SUBSCRIPTION_TIERS, LICENSE_TEMPLATES } from './useMonetization';
export type {
  StemListing,
  SellerProfile,
  LicenseOption,
  LicenseRights,
  ListingStats,
  Purchase,
  Subscription,
  SubscriptionTier,
  PaymentMethod,
  RevenueData,
  PayoutInfo,
  PayoutHistory,
  MonetizationState,
} from './useMonetization';

// ==================== Frontier Features ====================

// Real-time AI Processing
export { useRealtimeAI } from './useRealtimeAI';

// Web3 & Blockchain
export { useWeb3Music } from './useWeb3Music';

// Advanced Visualization
export { useAdvancedVisuals } from './useAdvancedVisuals';

// Music Education
export { useMusicEducation } from './useMusicEducation';

// Live Performance
export { useLivePerformance } from './useLivePerformance';

// Content Intelligence
export { useContentIntelligence } from './useContentIntelligence';

// Professional Integration
export { useProfessionalIntegration } from './useProfessionalIntegration';
