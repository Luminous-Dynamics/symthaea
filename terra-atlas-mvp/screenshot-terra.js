// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
const playwright = require('playwright');

(async () => {
  const browser = await playwright.chromium.launch();
  const page = await browser.newPage();

  console.log('Navigating to http://localhost:3001...');
  await page.goto('http://localhost:3001', { waitUntil: 'networkidle' });

  // Wait for globe to load
  console.log('Waiting for globe to load...');
  await page.waitForTimeout(8000);

  console.log('Taking screenshot...');
  await page.screenshot({ path: '/tmp/terra-atlas-homepage.png', fullPage: false });

  console.log('✅ Screenshot saved to /tmp/terra-atlas-homepage.png');
  await browser.close();
})();
