// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix RTC - Real-time Collaboration
 *
 * WebRTC-based peer-to-peer communication for the Mycelix ecosystem.
 * Provides signaling, peer connections, and collaboration rooms.
 *
 * @example Basic usage
 * ```typescript
 * import { joinRoom } from '@mycelix/sdk/rtc';
 *
 * const room = await joinRoom({
 *   signalingUrl: 'wss://signal.mycelix.net',
 *   roomId: 'my-document-123',
 *   agentId: myAgentId,
 *   displayName: 'Alice',
 * });
 *
 * // Listen for messages
 * room.on('message', (event) => {
 *   console.log(`Message from ${event.peerId}:`, event.data);
 * });
 *
 * // Send to all peers
 * room.broadcast('cursor-move', { x: 100, y: 200 });
 *
 * // Update presence
 * room.updatePresence({ status: 'away', cursor: { x: 100, y: 200 } });
 *
 * // Leave when done
 * room.leave();
 * ```
 *
 * @example Low-level API
 * ```typescript
 * import { createSignalingClient, createPeerManager } from '@mycelix/sdk/rtc';
 *
 * const signaling = createSignalingClient({
 *   serverUrl: 'wss://signal.mycelix.net',
 *   roomId: 'room-1',
 *   agentId: myAgentId,
 * });
 *
 * await signaling.connect();
 *
 * const peers = createPeerManager(myAgentId, signaling);
 *
 * // Handle incoming data
 * peers.onChannelMessage('mycelix-data', (message, channel) => {
 *   console.log('Received:', message);
 * });
 *
 * // Connect to a specific peer
 * await peers.connectToPeer(otherAgentId);
 *
 * // Send data
 * peers.sendToPeer(otherAgentId, 'mycelix-data', 'hello', { text: 'Hi!' });
 * ```
 *
 * @module rtc
 */

// Signaling
export {
  SignalingClient,
  createSignalingClient,
  type SignalingConfig,
  type SignalingMessage,
  type SignalType,
  type PeerInfo,
  type SignalingEventHandler,
  type ConnectionState,
} from './signaling.js';

// Peer connections
export {
  PeerManager,
  createPeerManager,
  type PeerConnectionConfig,
  type DataChannelConfig,
  type PeerConnection,
  type PeerConnectionState,
  type DataChannelMessage,
  type DataChannelHandler,
} from './peer.js';

// Collaboration rooms
export {
  CollaborationRoom,
  createRoom,
  joinRoom,
  type RoomConfig,
  type RoomState,
  type RoomPeer,
  type PresenceData,
  type RoomEvent,
  type RoomEventType,
  type RoomEventHandler,
} from './room.js';
