// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Music Web App Library
 * Re-exports SDK and app-specific utilities
 */

// Re-export SDK types and utilities
export {
  // Economic strategy types
  PaymentModel,
  PaymentType,
  type Split,
  type EconomicConfig,
  type PaymentReceipt,
  PRESET_STRATEGIES,
  EconomicStrategySDK,
  // Network utilities
  getNetwork,
  getRouterAddress,
  getStrategyAddress,
  isNetworkSupported,
  getSupportedChainIds,
  type NetworkConfig,
  GNOSIS_MAINNET,
  CHIADO_TESTNET,
  LOCALHOST,
  // EIP-712 signing
  buildDomain,
  signTypedData,
  signSongPayload,
  signPlayPayload,
  signClaimPayload,
} from '@mycelix/sdk';

// Export CDN helpers for audio delivery
export * from './cdn';

// Export Rust backend API client
export {
  rustApi,
  RustApiClient,
  type TrackInfo,
  type AnalysisResult,
  type SimilarTrack,
  type SearchResult,
  type SearchQuery,
  type SessionInfo,
  type SessionDetail,
  type FormatInfo,
  type FormatsResponse,
  type QualityPreset,
} from './rustApi';

// Export WebSocket client for real-time communication
export {
  MycelixWebSocket,
  getWebSocket,
  connectWebSocket,
  disconnectWebSocket,
  type SessionState,
  type WsMessage,
  type WsCommand,
  type WsEventHandlers,
  type WsOptions,
} from './websocket';
