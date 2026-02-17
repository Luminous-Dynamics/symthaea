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
