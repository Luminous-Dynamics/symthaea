const { chromium } = require('playwright');

(async () => {
  console.log('Launching browser...');
  const browser = await chromium.launch({
    headless: true,
    executablePath: '/home/tstoltz/.nix-profile/bin/chromium',
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-gpu'
    ]
  });

  try {
    const context = await browser.newContext({
      viewport: { width: 1920, height: 1080 },
      deviceScaleFactor: 2, // Retina quality
    });

    const page = await context.newPage();

    console.log('Navigating to https://atlas.luminousdynamics.io...');
    await page.goto('https://atlas.luminousdynamics.io', {
      waitUntil: 'domcontentloaded',
      timeout: 60000
    });

    // Wait for the globe to render
    console.log('Waiting for page to fully load...');
    await page.waitForTimeout(8000);

    // Take full page screenshot
    console.log('Taking screenshot...');
    await page.screenshot({
      path: 'screenshots/homepage-full.png',
      fullPage: true,
      timeout: 60000
    });

    // Take viewport screenshot (above the fold)
    await page.screenshot({
      path: 'screenshots/homepage-hero.png',
      fullPage: false
    });

    console.log('✅ Screenshots saved to screenshots/ directory');

    // Get page title to verify
    const title = await page.title();
    console.log(`Page title: ${title}`);

  } catch (error) {
    console.error('❌ Error:', error.message);
  } finally {
    await browser.close();
  }
})();