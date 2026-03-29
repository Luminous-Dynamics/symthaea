// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Jobs Index
 *
 * Central export for job scheduler and all job definitions.
 */

export { JobScheduler, getScheduler } from './scheduler';
export type { JobDefinition, JobResult, JobStatus, SchedulerEvents } from './scheduler';

export {
  createAnalyticsRefreshJob,
  createTopSongsCacheJob,
  createPendingPlaysCleanupJob,
  createDailyStatsJob,
  createVacuumJob,
  createCacheCleanupJob,
  registerAnalyticsJobs,
} from './analytics.job';
