/**
 * Security Package
 *
 * Comprehensive security middleware, authentication, and utilities for Mycelix services.
 */

// Rate limiting
export {
  createRateLimiter,
  createTieredRateLimiter,
  createSlidingWindowLimiter,
  RateLimiterOptions,
  RateLimitInfo,
  TieredRateLimiterOptions,
} from './rate-limiter';

// Security headers
export {
  securityHeaders,
  corsMiddleware,
  apiSecurityHeaders,
  CorsOptions,
} from './headers';

// Basic authentication
export { authMiddleware, AuthOptions, AuthenticatedRequest } from './auth';

// Enhanced authentication (EIP-191, sessions)
export {
  enhancedAuth,
  optionalAuth,
  requirePermissions,
  requireAdmin,
  verifyEthereumSignature,
  createAuthMessage,
  createAuthChallenge,
  generateNonce,
  createSessionToken,
  verifySessionToken,
  EnhancedAuthOptions,
  AuthPayload,
  SessionData,
  NonceStore,
  MemoryNonceStore,
} from './auth-enhanced';

// CSRF protection
export {
  csrfProtection,
  generateCsrfToken,
  csrfTokenHandler,
  doubleSubmitCookie,
  csrfHelpers,
  CsrfOptions,
} from './csrf';

// Validation
export {
  validateRequest,
  commonSchemas,
  ValidationSchema,
} from './validation';

// Sanitization
export {
  sanitizeInput,
  escapeHtml,
  sanitizeFilename,
  sanitizeRedirectUrl,
  stripDangerousTags,
  sanitizeMarkdown,
  sanitizeEthereumAddress,
  sanitizeSqlInput,
} from './sanitize';

// IP filtering
export { ipBlocklist, IpBlocklistOptions } from './ip-blocklist';
