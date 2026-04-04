// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * Common types shared across the Mycelix TypeScript SDK.
 *
 * These mirror the Rust structs in mycelix-bridge-common and
 * mycelix-cluster-manifest to ensure type safety across the
 * Holochain bridge boundary.
 */

/** Direction of a bridge connection between clusters. */
export type BridgeDirection = "Outbound" | "Inbound" | "Bidirectional";

/** Sensitivity classification for data entries. */
export type DataSensitivity = "Public" | "Internal" | "Sensitive" | "Restricted";

/** Result of a zome call, wrapping either success or error. */
export interface ZomeCallResult<T> {
  ok: boolean;
  data?: T;
  error?: string;
}

/** Configuration for connecting to a Holochain conductor. */
export interface ConductorConfig {
  /** WebSocket URL for the app port (e.g., "ws://localhost:8300"). */
  url: string;
  /** Installed app ID to connect to. */
  installedAppId: string;
  /** Optional authentication token. */
  token?: string;
  /** Connection timeout in milliseconds. Defaults to 15000. */
  timeoutMs?: number;
}

/** A dependency on another cluster. */
export interface ClusterDependency {
  /** Domain ID of the dependency (e.g., "identity"). */
  clusterId: string;
  /** Whether this is a hard requirement or optional enhancement. */
  required: boolean;
  /** Human-readable reason. */
  reason: string;
}

/** A bridge connection declaration between clusters. */
export interface BridgeDeclaration {
  /** Target cluster ID. */
  targetCluster: string;
  /** Direction of the bridge. */
  direction: BridgeDirection;
  /** Zome names allowed across this bridge. */
  allowedZomes: string[];
  /** Human-readable purpose. */
  purpose: string;
}

/** An entry type declaration for the data sovereignty dashboard. */
export interface EntryTypeDeclaration {
  /** Zome this entry type belongs to. */
  zome: string;
  /** Entry type identifier. */
  entryType: string;
  /** Human-readable label. */
  label: string;
  /** Data sensitivity classification. */
  sensitivity: DataSensitivity;
  /** Short description. */
  description: string;
}
