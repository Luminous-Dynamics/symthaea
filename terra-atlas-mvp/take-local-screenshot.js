const playwright = require('playwright');

(async () => {
  const browser = await playwright.chromium.launch();
  const page = await browser.newPage();
  
  console.log('Opening local dev server...');
  await page.goto('http://localhost:3001', { waitUntil: 'domcontentloaded', timeout: 30000 });
  
  console.log('Waiting for page to render...');
  await page.waitForTimeout(5000);
  
  console.log('Taking screenshot...');
  await page.screenshot({ path: 'screenshots/local-dev.png', fullPage: false });
  
  console.log('✅ Local screenshot saved!');
  await browser.close();
})();
