// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Core
 *
 * Platform infrastructure and orchestration
 */

// Event Bus
export {
  EventBus,
  getEventBus,
  createLoggingMiddleware,
  createAnalyticsMiddleware,
  createPersistenceMiddleware,
  createRateLimitMiddleware,
} from './event-bus';
export type {
  PlatformEvents,
  EventName,
  EventPayload,
  EventHandler,
  EventMeta,
  EventRecord,
  EventMiddleware,
} from './event-bus';

// Platform Manager
export {
  PlatformManager,
  getPlatformManager,
  initializePlatform,
} from './platform';
export type {
  PlatformConfig,
  FeatureFlags,
  ServiceStatus,
  ServiceName,
  PlatformState,
  UserState,
  UserPreferences,
  PlaybackState,
  NetworkState,
} from './platform';

// Configuration
export {
  ConfigManager,
  getConfigManager,
  createConfigHook,
  createFeatureFlagHook,
  createExperimentHook,
} from './config';
export type {
  AppConfig,
  AppSettings,
  Environment,
  LogLevel,
  APISettings,
  LimitSettings,
  UISettings,
  AudioSettings,
  AnalyticsSettings,
  ExperimentSettings,
  Experiment,
  ExperimentVariant,
} from './config';

// Observability
export {
  ObservabilityManager,
  getObservabilityManager,
  ErrorTracker,
  PerformanceMonitor,
  Analytics,
} from './observability';
export type {
  ErrorReport,
  ErrorType,
  ErrorSeverity,
  ErrorContext,
  Breadcrumb,
  PerformanceMetric,
  WebVitals,
  AnalyticsEvent,
  Span,
  UserSession,
  DeviceInfo,
} from './observability';

// Security
export {
  SecurityManager,
  getSecurityManager,
  CSRFManager,
  RequestSigner,
  Validator,
  Sanitizer,
  RateLimiter,
  RateLimitError,
  SecureStorage,
  CSPBuilder,
} from './security';
export type {
  SecurityConfig,
  ValidationRule,
  ValidationSchema,
  ValidationResult,
  ValidationError,
  RateLimitState,
} from './security';

// SDK
export {
  createSDK,
  MycelixSDK,
  OAuthClient,
  APIClient,
  APIError,
  WSClient,
  EmbedBuilder,
  WebhookManager,
} from './sdk';
export type {
  SDKConfig,
  AuthToken,
  User,
  Track,
  Album,
  Playlist,
  EmbedOptions,
  WebhookEvent,
  WebhookEventType,
  WebhookConfig,
} from './sdk';

// Re-export for convenience
import { getPlatformManager, initializePlatform } from './platform';
import { getEventBus } from './event-bus';
import { getConfigManager } from './config';
import { getObservabilityManager } from './observability';
import { getSecurityManager } from './security';
import { createSDK } from './sdk';

export default {
  // Managers
  getPlatformManager,
  initializePlatform,
  getEventBus,
  getConfigManager,
  getObservabilityManager,
  getSecurityManager,
  createSDK,
};
