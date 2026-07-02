// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Real-time Collaboration System
 *
 * Complete collaboration infrastructure:
 * - CRDT for conflict-free sync
 * - WebRTC for P2P connections
 * - Presence awareness
 * - Session management
 */

// CRDT
export {
  VectorClockImpl,
  LWWRegister,
  GCounter,
  PNCounter,
  ORSet,
  LWWMap,
  RGA,
  DocumentCRDT,
  type NodeId,
  type Timestamp,
  type VectorClock,
  type CRDTMetadata,
  type DocumentOperation,
} from './crdt';

// WebRTC
export {
  SignalingClient,
  PeerConnection,
  WebRTCManager,
  type ConnectionState,
  type PeerInfo,
  type RTCConfig,
  type DataMessage,
  type AudioStreamConfig,
  type SignalingMessage,
} from './webrtc';

// Presence
export {
  PresenceManager,
  CursorRenderer,
  SelectionRenderer,
  TypingIndicator,
  type CursorPosition,
  type Selection,
  type UserStatus,
  type UserPresence,
  type PresenceUpdate,
  type CursorStyle,
} from './presence';

// Session
export {
  CollaborationSession,
  useCollaborationSession,
  type SessionConfig,
  type SessionState,
  type SessionCallbacks,
  type ChatMessage,
} from './session';

export default {
  // Re-export main classes
  DocumentCRDT: () => import('./crdt').then(m => m.DocumentCRDT),
  WebRTCManager: () => import('./webrtc').then(m => m.WebRTCManager),
  PresenceManager: () => import('./presence').then(m => m.PresenceManager),
  CollaborationSession: () => import('./session').then(m => m.CollaborationSession),
};
