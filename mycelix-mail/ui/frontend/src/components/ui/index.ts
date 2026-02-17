/**
 * UI Components Index
 *
 * Core UI building blocks:
 * - Skeletons for loading states
 * - Error boundaries for graceful error handling
 * - Animations for smooth transitions
 */

// Loading Skeletons
export {
  Skeleton,
  SkeletonCircle,
  SkeletonText,
  EmailRowSkeleton,
  EmailListSkeleton,
  TrustBadgeSkeleton,
  ContactProfileSkeleton,
  InsightsPanelSkeleton,
  TrustGraphSkeleton,
  ThreadSummarySkeleton,
  DashboardSkeleton,
  SettingsSkeleton,
} from './Skeleton';

// Error Boundaries
export {
  ErrorBoundary,
  DefaultErrorFallback,
  TrustErrorBoundary,
  AIErrorBoundary,
  NetworkErrorBoundary,
  QueryError,
  AsyncBoundary,
  useErrorHandler,
} from './ErrorBoundary';

// Animations
export {
  Fade,
  Slide,
  Scale,
  Collapse,
  StaggeredList,
  AnimatePresence,
  Pulse,
  Shimmer,
  Counter,
  ProgressBar,
  usePrefersReducedMotion,
  animationKeyframes,
} from './Animations';
