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