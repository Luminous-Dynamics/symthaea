// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();

  console.log('Navigating to production...');
  await page.goto('https://atlas.luminousdynamics.io', {
    waitUntil: 'domcontentloaded',
    timeout: 30000
  });

  console.log('Waiting 5 seconds for initial render...');
  await page.waitForTimeout(5000);

  console.log('Taking screenshot...');
  await page.screenshot({
    path: 'screenshots/quick-check.png',
    fullPage: false,
    timeout: 15000
  });

  console.log('✅ Screenshot saved to screenshots/quick-check.png');
  await browser.close();
})();