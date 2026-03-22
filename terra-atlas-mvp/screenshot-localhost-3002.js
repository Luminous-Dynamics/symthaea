const { firefox } = require('playwright');

(async () => {
  const browser = await firefox.launch({ headless: true });
  const page = await browser.newPage();

  console.log('📡 http://localhost:3002...');
  await page.goto('http://localhost:3002', {
    waitUntil: 'domcontentloaded',
    timeout: 30000
  });

  await page.waitForTimeout(8000);

  await page.screenshot({
    path: 'screenshots/localhost-3002.png',
    fullPage: false
  });

  console.log('✅ Screenshot: screenshots/localhost-3002.png');

  const html = await page.content();
  if (html.includes('BRAND NEW EARTH GLOBE')) {
    console.log('🎉 RED MARKER FOUND!');
  } else if (html.includes('Application error')) {
    console.log('❌ Application error detected');
  } else {
    console.log('⚠️  No markers detected');
  }

  await browser.close();
})();