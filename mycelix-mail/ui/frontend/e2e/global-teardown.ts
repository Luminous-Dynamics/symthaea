// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Playwright Global Teardown
 *
 * Runs once after all tests complete
 */

import { FullConfig } from '@playwright/test';

async function globalTeardown(config: FullConfig) {
  console.log('Running global teardown...');

  // Clean up test data if needed
  if (process.env.CLEANUP_TEST_DB) {
    console.log('Cleaning up test database...');
    // Could truncate tables, remove test users, etc.
  }

  // Generate coverage report if collected
  if (process.env.COLLECT_COVERAGE) {
    console.log('Generating coverage report...');
  }

  console.log('Global teardown complete');
}

export default globalTeardown;
