// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Playwright Global Setup
 *
 * Runs once before all tests
 */

import { chromium, FullConfig } from '@playwright/test';

async function globalSetup(config: FullConfig) {
  console.log('Running global setup...');

  // Verify the app is running
  const browser = await chromium.launch();
  const page = await browser.newPage();

  try {
    const baseURL = config.projects[0].use?.baseURL || 'http://localhost:5173';
    await page.goto(baseURL, { timeout: 30000 });
    console.log('App is running at', baseURL);
  } catch (error) {
    console.error('Failed to connect to app. Make sure it is running.');
    throw error;
  } finally {
    await browser.close();
  }

  // Set up test database state if needed
  if (process.env.SETUP_TEST_DB) {
    console.log('Setting up test database...');
    // Could run migrations, seed data, etc.
  }

  console.log('Global setup complete');
}

export default globalSetup;
