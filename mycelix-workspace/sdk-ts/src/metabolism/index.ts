/**
 * @mycelix/sdk Knowledge Metabolism Module
 *
 * Manages the lifecycle of claims - from birth through death to decomposition.
 * Key principle: Claims are living things that must be actively maintained.
 *
 * Dead claims leave tombstones that preserve lessons learned.
 *
 * @packageDocumentation
 * @module metabolism
 */

// Types
export {
  type LifecyclePhase,
  type DeathReason,
  type ClaimHealth,
  type ClaimActivity,
  type Tombstone,
  type TombstoneSummary,
  type LifecycleTransition,
  type TransitionRule,
  type MetabolizingClaim,
  type DecompositionResult,
  type MetabolismConfig,
  type MetabolismEvent,
  DEFAULT_METABOLISM_CONFIG,
} from './types.js';

// Engine
export {
  MetabolismEngine,
  type MetabolismStats,
  getMetabolismEngine,
  resetMetabolismEngine,
} from './engine.js';

// Cross-Domain Dashboard
export {
  CrossDomainDashboard,
  getDashboard,
  resetDashboard,
  formatHealthScore,
  statusEmoji,
  calculateResilience,
  DEFAULT_DASHBOARD_CONFIG,
  type ResourceDomain,
  type DomainHealthStatus,
  type TrendDirection,
  type BaseDomainMetrics,
  type FoodMetrics,
  type WaterMetrics,
  type EnergyMetrics,
  type ShelterMetrics,
  type MedicineMetrics,
  type DomainMetrics,
  type DomainDependency,
  type CrossDomainAlert,
  type CrossDomainRecommendation,
  type CascadeRisk,
  type DashboardState,
  type DashboardSummary,
  type DashboardConfig,
} from './cross-domain-dashboard.js';
