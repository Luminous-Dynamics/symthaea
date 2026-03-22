const { firefox } = require('playwright');

(async () => {
  console.log('🦊 Launching Firefox for localhost...');
  const browser = await firefox.launch({ headless: true });
  const page = await browser.newPage();

  console.log('📡 Navigating to localhost:3001...');
  await page.goto('http://localhost:3001', {
    waitUntil: 'domcontentloaded',
    timeout: 30000
  });

  console.log('⏳ Waiting 8 seconds for globe to render...');
  await page.waitForTimeout(8000);

  console.log('📸 Taking screenshot...');
  await page.screenshot({
    path: 'screenshots/localhost-firefox.png',
    fullPage: false
  });

  console.log('✅ Screenshot saved!');

  // Check for verification markers
  const html = await page.content();
  if (html.includes('BRAND NEW EARTH GLOBE')) {
    console.log('🎉 VERIFICATION MARKER FOUND - New component loaded!');
  } else {
    console.log('⚠️  No verification markers detected');
  }

  await browser.close();
})().catch(err => {
  console.error('❌ Error:', err.message);
  process.exit(1);
});