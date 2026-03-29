// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Epistemic Components Index
 *
 * Central export for all epistemic UI components:
 * - Verifiable claims and badges
 * - Trust graph visualization
 * - AI-powered insights
 * - Contact profiles with trust history
 * - Compose with claims attachment
 * - Epistemic search and filtering
 * - Thread summaries
 * - Trust network dashboard
 * - Enhanced email view with sidebar
 * - Onboarding wizard
 * - Trust notifications
 * - Epistemic settings panel
 * - Keyboard shortcuts help
 */

// Core epistemic components
export { default as EpistemicInsightsPanel } from './EpistemicInsightsPanel';
export { default as EpistemicEmailRow, EpistemicIndicatorsBar } from './EpistemicEmailRow';

// Trust visualization and management
export { default as TrustGraphVisualizer, TrustPathInline } from '../trust/TrustGraphVisualizer';
export {
  default as TrustDashboard,
  StatCard,
  TrustScoreGauge,
  AttestationItem,
  CreateAttestationForm,
  PendingIntroductions,
} from '../trust/TrustDashboard';

// AI-powered insights
export { default as AIInsightsPanel, IntentBadge, PriorityIndicator } from '../ai/AIInsightsPanel';

// Verifiable claims
export { default as ClaimBadge, AssuranceLevelBadge, EmailClaimsList } from '../claims/ClaimBadge';

// Consent and introductions
export { default as IntroductionManager } from '../consent/IntroductionManager';

// Contact profiles
export { default as ContactProfile } from '../contacts/ContactProfile';

// Compose with claims
export {
  default as ComposeWithClaims,
  CredentialCard,
  QuickAttachBadges,
  AttachedClaimsSummary,
} from '../compose/ComposeWithClaims';
export type { AttachedClaim } from '../compose/ComposeWithClaims';

// Epistemic search and filtering
export {
  default as EpistemicSearch,
  FilterChip,
  TrustScoreSlider,
  PathLengthSelector,
  ActiveFiltersBar,
} from '../search/EpistemicSearch';
export type { EpistemicFilters } from '../search/EpistemicSearch';

// Thread analysis
export {
  default as ThreadSummaryPanel,
  SentimentIndicator,
  UrgencyBadge,
  ParticipantCard,
  ActionItemCard,
  DecisionTimeline,
  EpistemicQualityMeter,
} from '../thread/ThreadSummaryPanel';

// Enhanced email view with epistemic sidebar
export { default as EpistemicEmailView } from '../email/EpistemicEmailView';

// Onboarding wizard
export { default as EpistemicOnboarding } from '../onboarding/EpistemicOnboarding';

// Trust notifications
export {
  default as TrustNotifications,
  NotificationBell,
  useTrustNotificationStore,
  useTrustNotifications,
  useSimulateTrustNotifications,
} from '../notifications/TrustNotifications';
export type {
  TrustNotification,
  TrustNotificationType,
} from '../notifications/TrustNotifications';

// Settings panel
export {
  default as EpistemicSettings,
  useEpistemicSettings,
  SettingToggle,
  SettingSlider,
  SettingSelect,
} from '../settings/EpistemicSettings';
export type {
  EpistemicSettingsState,
  TrustSettings,
  QuarantineSettings,
  AISettings,
  NotificationSettings,
  PrivacySettings,
  DisplaySettings,
} from '../settings/EpistemicSettings';

// Help modal
export {
  default as KeyboardShortcuts,
  useKeyboardShortcuts,
} from '../help/KeyboardShortcuts';

// UI components (skeletons, error boundaries, animations)
export {
  Skeleton,
  SkeletonCircle,
  EmailListSkeleton,
  ContactProfileSkeleton,
  InsightsPanelSkeleton,
  TrustGraphSkeleton,
  DashboardSkeleton,
  ErrorBoundary,
  TrustErrorBoundary,
  AIErrorBoundary,
  NetworkErrorBoundary,
  Fade,
  Slide,
  Scale,
  Collapse,
  StaggeredList,
  Counter,
  ProgressBar,
  usePrefersReducedMotion,
} from '../ui';

// Batch operations
export {
  BatchAttestationCreator,
  TrustNetworkExportImport,
  BatchRevocation,
} from '../trust/BatchOperations';

// External identity providers
export {
  default as ExternalIdentityProviders,
  ENSVerification,
  GitHubVerification,
  DomainVerification,
} from '../identity/ExternalProviders';

// App integration
export {
  default as EpistemicApp,
  EpistemicAppShell,
  EpistemicProviders,
} from '../app/EpistemicApp';

// Advanced search with filters
export {
  AdvancedSearch,
  AdvancedSearchModal,
  SearchHistoryDropdown,
  SavedSearches,
  useAdvancedSearch,
  useSearchHistory,
} from '../search/AdvancedSearch';
export type { SearchFilters, SavedSearch } from '../search/AdvancedSearch';

// Email templates
export {
  EmailTemplateSelector,
  TemplatePreview,
  CustomTemplateEditor,
  useEmailTemplates,
  builtInTemplates,
} from '../compose/EmailTemplates';
export type { EmailTemplate, TemplateVariable } from '../compose/EmailTemplates';

// Trust analytics
export {
  TrustAnalyticsDashboard,
  LineChart,
  BarChart,
  DonutChart,
  MetricCard,
  TimeRangeSelector,
  useTrustAnalytics,
} from '../analytics/TrustAnalytics';
export type {
  TrustAnalyticsData,
  TimeSeriesPoint,
  TierDistribution,
  TimeRange,
} from '../analytics/TrustAnalytics';

// Performance (virtual list)
export {
  VirtualList,
  VirtualEmailList,
  useWindowedData,
} from '../performance/VirtualList';
export type {
  VirtualListProps,
  VirtualListHandle,
  UseWindowedDataOptions,
} from '../performance/VirtualList';
