// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Attribution Registry client for the Observatory dashboard.
 *
 * Wraps callZome from $lib/conductor with mock fallback for demo mode.
 */

import { isConnected, callZome } from './conductor';

// ── Types ──────────────────────────────────────────────────────────

export interface AttributionStats {
  totalDependencies: number;
  totalEcosystems: number;
  verifiedCount: number;
  totalUsage: number;
  activeAttestations: number;
  totalPledges: number;
}

export interface LeaderboardEntry {
  dependency_id: string;
  usage_count: number;
  pledge_count: number;
  weighted_score: number;
}

export interface TopDependency {
  dependency_id: string;
  usage_count: number;
}

export interface EcosystemStat {
  ecosystem: string;
  count: number;
}

export interface EcosystemStatistics {
  total_dependencies: number;
  ecosystems: EcosystemStat[];
  verified_count: number;
}

// ── Mock Data ──────────────────────────────────────────────────────

const MOCK_STATS: AttributionStats = {
  totalDependencies: 847,
  totalEcosystems: 6,
  verifiedCount: 312,
  totalUsage: 4_231,
  activeAttestations: 89,
  totalPledges: 156,
};

const MOCK_LEADERBOARD: LeaderboardEntry[] = [
  { dependency_id: 'crate:serde:1.0.219', usage_count: 423, pledge_count: 18, weighted_score: 47.2 },
  { dependency_id: 'npm:react:18.2.0', usage_count: 381, pledge_count: 24, weighted_score: 42.8 },
  { dependency_id: 'crate:tokio:1.36', usage_count: 367, pledge_count: 12, weighted_score: 38.1 },
  { dependency_id: 'pip:numpy:1.26', usage_count: 298, pledge_count: 15, weighted_score: 34.5 },
  { dependency_id: 'crate:hdk:0.6.0', usage_count: 245, pledge_count: 8, weighted_score: 28.7 },
  { dependency_id: 'npm:svelte:4.2', usage_count: 198, pledge_count: 11, weighted_score: 24.3 },
  { dependency_id: 'go:golang.org/x/net:v0.17', usage_count: 176, pledge_count: 5, weighted_score: 18.9 },
  { dependency_id: 'gem:rails:7.1', usage_count: 154, pledge_count: 9, weighted_score: 16.4 },
];

const MOCK_TOP_DEPS: TopDependency[] = [
  { dependency_id: 'crate:serde:1.0.219', usage_count: 423 },
  { dependency_id: 'npm:react:18.2.0', usage_count: 381 },
  { dependency_id: 'crate:tokio:1.36', usage_count: 367 },
  { dependency_id: 'pip:numpy:1.26', usage_count: 298 },
  { dependency_id: 'crate:hdk:0.6.0', usage_count: 245 },
];

const MOCK_UNDER_SUPPORTED: LeaderboardEntry[] = [
  { dependency_id: 'crate:libc:0.2', usage_count: 892, pledge_count: 1, weighted_score: 0.8 },
  { dependency_id: 'npm:glob:10.3', usage_count: 456, pledge_count: 0, weighted_score: 0.0 },
  { dependency_id: 'pip:setuptools:69', usage_count: 387, pledge_count: 0, weighted_score: 0.0 },
  { dependency_id: 'crate:cc:1.0', usage_count: 341, pledge_count: 1, weighted_score: 0.4 },
];

const MOCK_ECOSYSTEMS: EcosystemStat[] = [
  { ecosystem: 'RustCrate', count: 312 },
  { ecosystem: 'NpmPackage', count: 234 },
  { ecosystem: 'PythonPackage', count: 145 },
  { ecosystem: 'GoModule', count: 78 },
  { ecosystem: 'NixFlake', count: 42 },
  { ecosystem: 'RubyGem', count: 24 },
  { ecosystem: 'MavenPackage', count: 12 },
];

// ── Client ─────────────────────────────────────────────────────────

let isLive = false;

async function tryCallZome<T>(zome: string, fn_name: string, payload: unknown): Promise<T | null> {
  // Skip the attempt entirely when we know the conductor is down
  if (!isConnected()) {
    isLive = false;
    return null;
  }
  try {
    const result = await callZome<T>({
      role_name: 'attribution',
      zome_name: zome,
      fn_name,
      payload,
    });
    isLive = true;
    return result;
  } catch {
    isLive = false;
    return null;
  }
}

export function getIsLive(): boolean {
  return isLive;
}

export async function fetchStats(): Promise<AttributionStats> {
  const eco = await tryCallZome<EcosystemStatistics>('registry', 'get_ecosystem_statistics', null);
  if (eco) {
    const usageCount = await tryCallZome<TopDependency[]>('usage', 'get_top_dependencies', 1000) ?? [];
    const totalUsage = usageCount.reduce((sum, d) => sum + d.usage_count, 0);
    const activeAttestations = await tryCallZome<number>('usage', 'get_attestation_count', null) ?? 0;
    const totalPledges = await tryCallZome<number>('reciprocity', 'get_pledge_count', null) ?? 0;
    return {
      totalDependencies: eco.total_dependencies,
      totalEcosystems: eco.ecosystems.length,
      verifiedCount: eco.verified_count,
      totalUsage,
      activeAttestations,
      totalPledges,
    };
  }
  return { ...MOCK_STATS };
}

export async function fetchLeaderboard(limit: number): Promise<LeaderboardEntry[]> {
  const result = await tryCallZome<LeaderboardEntry[]>('reciprocity', 'get_stewardship_leaderboard', limit);
  return result ?? MOCK_LEADERBOARD.slice(0, limit);
}

export async function fetchTopDependencies(limit: number): Promise<TopDependency[]> {
  const result = await tryCallZome<TopDependency[]>('usage', 'get_top_dependencies', limit);
  return result ?? MOCK_TOP_DEPS.slice(0, limit);
}

export async function fetchUnderSupported(limit: number): Promise<LeaderboardEntry[]> {
  const result = await tryCallZome<LeaderboardEntry[]>('reciprocity', 'get_under_supported_dependencies', limit);
  return result ?? MOCK_UNDER_SUPPORTED.slice(0, limit);
}

export async function fetchEcosystems(): Promise<EcosystemStat[]> {
  const result = await tryCallZome<EcosystemStatistics>('registry', 'get_ecosystem_statistics', null);
  return result?.ecosystems ?? MOCK_ECOSYSTEMS;
}
