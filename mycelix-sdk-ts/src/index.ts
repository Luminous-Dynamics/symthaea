// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @mycelix/sdk - TypeScript SDK for building Mycelix extensions.
 *
 * Provides typed access to Holochain conductors running Mycelix
 * cluster DNAs, along with consciousness gating and data sovereignty
 * types that match the Rust structs in bridge-common and
 * cluster-manifest.
 *
 * @example
 * ```ts
 * import { MycelixClient, deriveTier } from "@mycelix/sdk";
 *
 * const client = new MycelixClient({
 *   conductor: { url: "ws://localhost:8300", installedAppId: "mycelix" },
 * });
 * await client.connect();
 *
 * const profile = await client.getConsciousnessProfile();
 * if (profile) {
 *   console.log("Trust tier:", deriveTier(profile));
 * }
 * ```
 *
 * @packageDocumentation
 */

// Client
export { MycelixClient } from "./client";
export type { MycelixClientOptions, ConnectionState } from "./client";

// Consciousness
export {
  combinedScore,
  deriveTier,
  meetsTier,
  TIER_RANK,
} from "./consciousness";
export type { ConsciousnessProfile, TrustTier } from "./consciousness";

// Sovereignty
export { computeDataFlows } from "./sovereignty";
export type {
  DataFlow,
  ClusterManifest,
  ExternalFrontendManifest,
  ClusterCatalog,
} from "./sovereignty";

// Common types
export type {
  BridgeDirection,
  DataSensitivity,
  ZomeCallResult,
  ConductorConfig,
  ClusterDependency,
  BridgeDeclaration,
  EntryTypeDeclaration,
} from "./types";
