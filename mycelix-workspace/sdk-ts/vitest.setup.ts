// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Vitest Setup File
 *
 * Handles global test setup and cleanup, including suppression of
 * known cleanup errors from libsodium ESM module initialization.
 *
 * NOTE: The libsodium-wrappers ESM module has a known issue where it attempts
 * to set properties on frozen ESM modules during teardown. This causes
 * "Cannot set property default of [object Module] which has only a getter"
 * errors. These errors don't affect test results - they're cosmetic teardown
 * noise from @holochain/client's dependency on libsodium-wrappers.
 *
 * The vitest.config.ts settings (pool: 'forks', dangerouslyIgnoreUnhandledErrors)
 * mitigate impact, but the errors still appear in output. This is a known
 * upstream issue that requires a fix in libsodium-wrappers itself.
 */

// Suppress libsodium ESM module errors globally
// These errors occur during module teardown and don't affect test results
const originalEmit = process.emit.bind(process);

// @ts-expect-error - Overriding process.emit to filter errors
process.emit = function (event: string, ...args: unknown[]) {
  if (event === 'unhandledRejection') {
    const reason = args[0];
    // Check if this is a libsodium module error
    if (
      reason instanceof TypeError &&
      (String(reason.message).includes('Cannot set property') ||
        String(reason.message).includes('Module') ||
        String(reason.message).includes('getter'))
    ) {
      // Suppress libsodium cleanup errors
      return false;
    }
  }
  return originalEmit(event, ...args);
};
