// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * Data sovereignty types for the Mycelix dashboard.
 *
 * Maps to the Rust structs in `crates/mycelix-cluster-manifest/src/lib.rs`.
 * These types power the data sovereignty dashboard, letting users see
 * and control how data flows between clusters.
 */

import type {
  BridgeDeclaration,
  BridgeDirection,
  ClusterDependency,
  DataSensitivity,
  EntryTypeDeclaration,
} from "./types";
import type { TrustTier } from "./consciousness";

/**
 * A data flow between two clusters.
 *
 * Computed from bridge declarations for installed clusters.
 * Matches `DataFlow` in mycelix-cluster-manifest.
 */
export interface DataFlow {
  /** Source cluster ID. */
  sourceId: string;
  /** Target cluster ID. */
  targetId: string;
  /** Direction of the flow. */
  direction: BridgeDirection;
  /** Number of zomes involved in this flow. */
  zomeCount: number;
  /** Human-readable purpose of this data flow. */
  purpose: string;
}

/**
 * Complete manifest for a Mycelix cluster.
 *
 * Describes everything the portal needs to display the cluster in
 * the catalog, build the dependency graph, and populate the data
 * sovereignty dashboard.
 *
 * Matches `ClusterManifest` in mycelix-cluster-manifest.
 */
export interface ClusterManifest {
  /** Unique cluster identifier (e.g., "health", "governance"). */
  id: string;
  /** Human-readable name. */
  name: string;
  /** Biological metaphor name (e.g., "Homeostasis"). */
  bioName: string;
  /** Version string. */
  version: string;
  /** Short description. */
  description: string;
  /** Minimum consciousness tier to access. */
  minTier: TrustTier;
  /** Holochain hApp role name. */
  happRole: string;
  /** Zome names in this cluster. */
  zomes: string[];
  /** Dependencies on other clusters. */
  dependencies: ClusterDependency[];
  /** Bridge connections to other clusters. */
  bridges: BridgeDeclaration[];
  /** Entry types managed by this cluster. */
  entryTypes: EntryTypeDeclaration[];
  /** Primary CSS color. */
  colorPrimary: string;
  /** Glow/accent CSS color. */
  colorGlow: string;
  /** Optional external frontend URL (for community extensions). */
  frontendUrl?: string;
}

/**
 * External frontend manifest for community extensions.
 *
 * Loaded from `~/.mycelix/extensions.json` or a registry DNA.
 * Matches `ExternalFrontendManifest` in mycelix-cluster-manifest.
 */
export interface ExternalFrontendManifest {
  /** Unique extension identifier. */
  id: string;
  /** Human-readable name. */
  name: string;
  /** Version string. */
  version: string;
  /** Author or organization. */
  author: string;
  /** Biological metaphor name. */
  bioName: string;
  /** Primary CSS color. */
  colorPrimary: string;
  /** Glow/accent CSS color. */
  colorGlow: string;
  /** Minimum consciousness tier. */
  minTier: TrustTier;
  /** URL where the frontend is hosted. */
  frontendUrl: string;
  /** Clusters this extension requires. */
  requiredClusters: string[];
  /** Clusters this extension can optionally use. */
  optionalClusters: string[];
  /** Short description. */
  description: string;
}

/**
 * A catalog of all known clusters and external extensions.
 */
export interface ClusterCatalog {
  /** Built-in cluster manifests (compiled from feature flags). */
  clusters: ClusterManifest[];
  /** External community extensions (loaded at runtime). */
  extensions: ExternalFrontendManifest[];
}

/**
 * Compute data flows between installed clusters from their bridge declarations.
 *
 * @param clusters - All cluster manifests
 * @param installedIds - IDs of clusters the user has installed
 * @returns Array of data flows between installed clusters
 */
export function computeDataFlows(
  clusters: ClusterManifest[],
  installedIds: string[],
): DataFlow[] {
  const flows: DataFlow[] = [];
  for (const cluster of clusters) {
    if (!installedIds.includes(cluster.id)) continue;
    for (const bridge of cluster.bridges) {
      if (!installedIds.includes(bridge.targetCluster)) continue;
      flows.push({
        sourceId: cluster.id,
        targetId: bridge.targetCluster,
        direction: bridge.direction,
        zomeCount: bridge.allowedZomes.length,
        purpose: bridge.purpose,
      });
    }
  }
  return flows;
}
