/**
 * Application Environment Configuration
 *
 * Centralized configuration for environment variables.
 * All environment variables should be accessed through this module.
 */

export const config = {
  // API Configuration
  apiUrl: import.meta.env.VITE_API_URL || 'http://localhost:3000',
  wsUrl: import.meta.env.VITE_WS_URL || 'ws://localhost:3000',

  // Application Info
  appName: import.meta.env.VITE_APP_NAME || 'Mycelix Mail',
  appVersion: import.meta.env.VITE_APP_VERSION || '1.0.0',

  // Feature Flags
  enableDebug: import.meta.env.VITE_ENABLE_DEBUG === 'true',
  enableDraftAutosave: import.meta.env.VITE_ENABLE_DRAFT_AUTOSAVE !== 'false', // Default true

  // Performance Settings
  queryStaleTime: parseInt(import.meta.env.VITE_QUERY_STALE_TIME || '30000', 10),
  queryCacheTime: parseInt(import.meta.env.VITE_QUERY_CACHE_TIME || '300000', 10),

  // Environment
  isDevelopment: import.meta.env.DEV,
  isProduction: import.meta.env.PROD,
  mode: import.meta.env.MODE,
} as const;

// Type-safe config access
export type Config = typeof config;

// Log configuration in development
if (config.isDevelopment && config.enableDebug) {
  console.log('[Config] Application configuration:', config);
}
