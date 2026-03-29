// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * WebRTC Module
 *
 * P2P real-time messaging via WebRTC with Holochain signaling
 */

export {
  SignalingService,
  type PeerConnection,
  type SignalingMessage,
  type WebRTCConfig,
} from './SignalingService';

export default SignalingService;
