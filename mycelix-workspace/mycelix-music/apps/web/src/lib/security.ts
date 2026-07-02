// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Security Utilities
 *
 * Input validation, sanitization, and security helpers.
 */

import DOMPurify from 'isomorphic-dompurify';

// ============================================================================
// Input Validation
// ============================================================================

/**
 * Validate email format
 */
export function isValidEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

/**
 * Validate Ethereum address
 */
export function isValidAddress(address: string): boolean {
  return /^0x[a-fA-F0-9]{40}$/.test(address);
}

/**
 * Validate URL
 */
export function isValidUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    return ['http:', 'https:'].includes(parsed.protocol);
  } catch {
    return false;
  }
}

/**
 * Validate track ID format
 */
export function isValidTrackId(id: string): boolean {
  // UUID format or CID format
  const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  const cidRegex = /^(Qm[1-9A-HJ-NP-Za-km-z]{44}|b[A-Za-z2-7]{58})$/;
  return uuidRegex.test(id) || cidRegex.test(id);
}

/**
 * Validate username
 */
export function isValidUsername(username: string): boolean {
  // 3-30 characters, alphanumeric and underscores
  return /^[a-zA-Z0-9_]{3,30}$/.test(username);
}

/**
 * Validate playlist name
 */
export function isValidPlaylistName(name: string): boolean {
  // 1-100 characters, no special control characters
  const sanitized = name.trim();
  return sanitized.length >= 1 && sanitized.length <= 100 && !/[\x00-\x1F]/.test(sanitized);
}

// ============================================================================
// Sanitization
// ============================================================================

/**
 * Sanitize HTML content
 */
export function sanitizeHtml(html: string): string {
  return DOMPurify.sanitize(html, {
    ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'a', 'p', 'br'],
    ALLOWED_ATTR: ['href', 'target', 'rel'],
    ADD_ATTR: ['rel'],
    FORBID_TAGS: ['script', 'style', 'iframe', 'form', 'input'],
    FORBID_ATTR: ['onerror', 'onload', 'onclick'],
  });
}

/**
 * Sanitize user input for display
 */
export function sanitizeText(text: string): string {
  return DOMPurify.sanitize(text, {
    ALLOWED_TAGS: [],
    ALLOWED_ATTR: [],
  });
}

/**
 * Sanitize filename
 */
export function sanitizeFilename(filename: string): string {
  // Remove path traversal and dangerous characters
  return filename
    .replace(/[/\\:*?"<>|]/g, '_')
    .replace(/\.\./g, '_')
    .replace(/^\./, '_')
    .slice(0, 255);
}

/**
 * Sanitize search query
 */
export function sanitizeSearchQuery(query: string): string {
  return query
    .trim()
    .replace(/[<>]/g, '')
    .slice(0, 200);
}

// ============================================================================
// CSRF Protection
// ============================================================================

const CSRF_TOKEN_KEY = 'mycelix_csrf_token';

/**
 * Generate CSRF token
 */
export function generateCsrfToken(): string {
  const array = new Uint8Array(32);
  crypto.getRandomValues(array);
  const token = Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('');

  if (typeof sessionStorage !== 'undefined') {
    sessionStorage.setItem(CSRF_TOKEN_KEY, token);
  }

  return token;
}

/**
 * Get current CSRF token
 */
export function getCsrfToken(): string | null {
  if (typeof sessionStorage === 'undefined') return null;
  return sessionStorage.getItem(CSRF_TOKEN_KEY);
}

/**
 * Validate CSRF token
 */
export function validateCsrfToken(token: string): boolean {
  const storedToken = getCsrfToken();
  if (!storedToken || !token) return false;

  // Timing-safe comparison
  if (token.length !== storedToken.length) return false;

  let result = 0;
  for (let i = 0; i < token.length; i++) {
    result |= token.charCodeAt(i) ^ storedToken.charCodeAt(i);
  }

  return result === 0;
}

// ============================================================================
// Rate Limiting (Client-side)
// ============================================================================

interface RateLimitEntry {
  count: number;
  resetTime: number;
}

const rateLimitMap = new Map<string, RateLimitEntry>();

/**
 * Check if action is rate limited
 */
export function isRateLimited(
  key: string,
  maxRequests: number = 10,
  windowMs: number = 60000
): boolean {
  const now = Date.now();
  const entry = rateLimitMap.get(key);

  if (!entry || now > entry.resetTime) {
    rateLimitMap.set(key, { count: 1, resetTime: now + windowMs });
    return false;
  }

  if (entry.count >= maxRequests) {
    return true;
  }

  entry.count++;
  return false;
}

/**
 * Get remaining rate limit allowance
 */
export function getRateLimitRemaining(key: string, maxRequests: number = 10): number {
  const entry = rateLimitMap.get(key);
  if (!entry || Date.now() > entry.resetTime) {
    return maxRequests;
  }
  return Math.max(0, maxRequests - entry.count);
}

// ============================================================================
// Content Security
// ============================================================================

/**
 * Check if file type is allowed for upload
 */
export function isAllowedAudioType(mimeType: string): boolean {
  const allowedTypes = [
    'audio/mpeg',      // MP3
    'audio/mp3',
    'audio/wav',
    'audio/wave',
    'audio/x-wav',
    'audio/flac',
    'audio/x-flac',
    'audio/ogg',
    'audio/x-ogg',
    'audio/aac',
    'audio/m4a',
    'audio/x-m4a',
    'audio/mp4',
  ];

  return allowedTypes.includes(mimeType.toLowerCase());
}

/**
 * Check if file type is allowed for images
 */
export function isAllowedImageType(mimeType: string): boolean {
  const allowedTypes = [
    'image/jpeg',
    'image/png',
    'image/gif',
    'image/webp',
  ];

  return allowedTypes.includes(mimeType.toLowerCase());
}

/**
 * Validate file size
 */
export function isValidFileSize(sizeBytes: number, maxMB: number): boolean {
  return sizeBytes <= maxMB * 1024 * 1024;
}

// ============================================================================
// Session Security
// ============================================================================

/**
 * Check for session anomalies
 */
export function detectSessionAnomaly(): boolean {
  if (typeof window === 'undefined') return false;

  // Check for embedded context (potential clickjacking)
  if (window.top !== window.self) {
    console.warn('Page is embedded in iframe');
    return true;
  }

  // Check for devtools (optional, can be removed if too aggressive)
  // const devtoolsOpen = window.outerHeight - window.innerHeight > 200;

  return false;
}

/**
 * Generate secure random ID
 */
export function generateSecureId(length: number = 16): string {
  const array = new Uint8Array(length);
  crypto.getRandomValues(array);
  return Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('');
}

// ============================================================================
// Export
// ============================================================================

export default {
  // Validation
  isValidEmail,
  isValidAddress,
  isValidUrl,
  isValidTrackId,
  isValidUsername,
  isValidPlaylistName,

  // Sanitization
  sanitizeHtml,
  sanitizeText,
  sanitizeFilename,
  sanitizeSearchQuery,

  // CSRF
  generateCsrfToken,
  getCsrfToken,
  validateCsrfToken,

  // Rate limiting
  isRateLimited,
  getRateLimitRemaining,

  // Content security
  isAllowedAudioType,
  isAllowedImageType,
  isValidFileSize,

  // Session
  detectSessionAnomaly,
  generateSecureId,
};
