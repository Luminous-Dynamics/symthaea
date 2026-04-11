// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Mail - Library Utilities Index
 *
 * Core utilities and services for the revolutionary epistemic mail system.
 */

// ============================================================================
// Original Services
// ============================================================================

// Holochain integration (legacy)
export * from './holochain';

// Internationalization
export {
  I18nProvider,
  useI18n,
  useTranslation,
  Trans,
  LocaleSelector,
  supportedLocales,
  getLocaleDirection,
  SUPPORTED_LOCALES,
  DEFAULT_LOCALE,
} from './i18n.js';
export type { Locale, TranslationKey } from './i18n.js';

// Service Worker / PWA
export {
  register as registerServiceWorker,
  unregister as unregisterServiceWorker,
  requestNotificationPermission,
  subscribeToPush,
  showNotification,
  addToBackgroundSync,
  useOfflineStatus,
  SERVICE_WORKER_SCRIPT,
} from './serviceWorker';
export type { ServiceWorkerConfig } from './serviceWorker';

// ============================================================================
// Enhanced Holochain Trust Network
// ============================================================================

export * from './holochain-client';

// ============================================================================
// AI Services
// ============================================================================

export * from './ai-service';

// ============================================================================
// Encryption & Security
// ============================================================================

export * from './encryption';

// ============================================================================
// Real-time & Offline
// ============================================================================

export * from './realtime-sync';
export * from './offline-storage';

// ============================================================================
// Semantic Search & Intelligence
// ============================================================================

export * from './semantic-search';

// ============================================================================
// Decentralized Identity
// ============================================================================

export * from './decentralized-identity';

// ============================================================================
// Workflow Automation
// ============================================================================

export * from './workflow-engine';

// ============================================================================
// Voice & Accessibility
// ============================================================================

export * from './voice-accessibility';

// ============================================================================
// Calendar & Focus Modes
// ============================================================================

export * from './calendar-focus';
