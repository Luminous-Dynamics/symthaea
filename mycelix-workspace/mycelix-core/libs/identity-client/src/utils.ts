// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Identity Client - Utility Functions
 */

import type { AssuranceLevel } from './types';
import { ASSURANCE_LEVEL_VALUE } from './types';

/**
 * Compare two assurance levels
 *
 * @param a - First level
 * @param b - Second level
 * @returns -1 if a < b, 0 if equal, 1 if a > b
 */
export function compareAssuranceLevels(a: AssuranceLevel, b: AssuranceLevel): number {
  const aValue = ASSURANCE_LEVEL_VALUE[a];
  const bValue = ASSURANCE_LEVEL_VALUE[b];

  if (aValue < bValue) return -1;
  if (aValue > bValue) return 1;
  return 0;
}

/**
 * Check if a string is a valid Mycelix DID
 *
 * @param did - String to validate
 * @returns Whether the string is a valid DID
 */
export function isValidDID(did: string): boolean {
  if (!did || typeof did !== 'string') {
    return false;
  }

  // Must start with did:mycelix:
  if (!did.startsWith('did:mycelix:')) {
    return false;
  }

  // Must have an identifier after the method
  const identifier = did.slice('did:mycelix:'.length);
  if (!identifier || identifier.length < 1) {
    return false;
  }

  // Basic character validation (alphanumeric, some special chars)
  const validPattern = /^[a-zA-Z0-9_\-+=]+$/;
  return validPattern.test(identifier);
}

/**
 * Parse a DID into its components
 *
 * @param did - DID to parse
 * @returns Parsed components or null if invalid
 */
export function parseDID(did: string): { method: string; identifier: string } | null {
  if (!isValidDID(did)) {
    return null;
  }

  const parts = did.split(':');
  if (parts.length < 3) {
    return null;
  }

  return {
    method: parts[1], // 'mycelix'
    identifier: parts.slice(2).join(':'), // Handle identifiers with colons
  };
}

/**
 * Format assurance level for display
 *
 * @param level - Assurance level
 * @returns Human-readable description
 */
export function formatAssuranceLevel(level: AssuranceLevel): string {
  const descriptions: Record<AssuranceLevel, string> = {
    E0: 'Unverified',
    E1: 'Email Verified',
    E2: 'Phone + Recovery',
    E3: 'Government ID',
    E4: 'Biometric + MFA',
  };

  return descriptions[level] || 'Unknown';
}

/**
 * Get minimum assurance level for a transaction value
 *
 * @param value - Transaction value
 * @param currency - Currency code
 * @returns Required assurance level
 */
export function getRequiredAssuranceLevel(value: number, currency: string = 'USD'): AssuranceLevel {
  // Thresholds (in USD equivalent)
  const thresholds: [number, AssuranceLevel][] = [
    [10000, 'E4'],
    [1000, 'E3'],
    [100, 'E2'],
    [10, 'E1'],
    [0, 'E0'],
  ];

  // Simple currency conversion (production would use real rates)
  const usdValue = convertToUSD(value, currency);

  for (const [threshold, level] of thresholds) {
    if (usdValue >= threshold) {
      return level;
    }
  }

  return 'E0';
}

/**
 * Simple currency conversion to USD (placeholder)
 */
function convertToUSD(value: number, currency: string): number {
  const rates: Record<string, number> = {
    USD: 1,
    EUR: 1.1,
    GBP: 1.27,
    ETH: 2000,
    BTC: 40000,
  };

  return value * (rates[currency.toUpperCase()] || 1);
}

/**
 * Generate a unique verification request ID
 */
export function generateVerificationId(): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).slice(2, 10);
  return `verify_${timestamp}_${random}`;
}

/**
 * Check if an assurance level meets a minimum requirement
 */
export function meetsMinimumLevel(current: AssuranceLevel, minimum: AssuranceLevel): boolean {
  return ASSURANCE_LEVEL_VALUE[current] >= ASSURANCE_LEVEL_VALUE[minimum];
}

/**
 * Get a color code for assurance level (for UI)
 */
export function getAssuranceLevelColor(level: AssuranceLevel): string {
  const colors: Record<AssuranceLevel, string> = {
    E0: '#9CA3AF', // Gray
    E1: '#60A5FA', // Blue
    E2: '#34D399', // Green
    E3: '#FBBF24', // Yellow
    E4: '#A78BFA', // Purple
  };

  return colors[level] || '#9CA3AF';
}

/**
 * Get an icon/emoji for assurance level (for UI)
 */
export function getAssuranceLevelIcon(level: AssuranceLevel): string {
  const icons: Record<AssuranceLevel, string> = {
    E0: 'user', // Unverified
    E1: 'mail', // Email
    E2: 'phone', // Phone
    E3: 'shield', // ID
    E4: 'shield-check', // Full verification
  };

  return icons[level] || 'user';
}
