#!/usr/bin/env node

import { chromium, firefox } from '../../../holochain/client/node_modules/@playwright/test/index.mjs';

const rootUrl = process.argv[2] ?? 'http://127.0.0.1:8117';
const chromiumPath = process.env.CHROMIUM_PATH ?? '/home/tstoltz/.nix-profile/bin/chromium';
const firefoxPath = process.env.FIREFOX_PATH ?? '/home/tstoltz/.nix-profile/bin/firefox';

function fail(message) {
  console.error(`FAIL: ${message}`);
  process.exitCode = 1;
}

function pass(message) {
  console.log(`OK: ${message}`);
}

async function verifyWith(browserType, executablePath, browserName) {
  const browser = await browserType.launch({
    headless: true,
    executablePath,
    args: browserName === 'chromium'
      ? ['--no-sandbox', '--disable-dev-shm-usage']
      : [],
  });

  const page = await browser.newPage();
  try {
    await page.goto(`${rootUrl}/settings`, {
      waitUntil: 'domcontentloaded',
      timeout: 30000,
    });
    await page.waitForLoadState('networkidle', { timeout: 15000 }).catch(() => {});

    const title = await page.title();
    if (!title.includes('Mycelix Pulse')) {
      throw new Error(`unexpected page title: ${title}`);
    }
    pass(`settings route loaded with Mycelix Pulse title in ${browserName}`);

    await page.waitForSelector('[data-testid="settings-page"]', { timeout: 15000 });
    await page.waitForSelector('[data-testid="live-slice-verify-button"]', { timeout: 15000 });
    pass(`settings page rendered the live-slice verification UI in ${browserName}`);

    const bridgeState = await page.evaluate(async () => {
      const configResponse = await fetch('/conductor-config.json').catch(() => null);
      const config = configResponse && configResponse.ok
        ? await configResponse.json().catch(() => null)
        : null;

      return {
        hasTokenPromise: typeof window.__HC_TOKEN_PROMISE?.then === 'function',
        hasCallZome: typeof window.__HC_CALL_ZOME === 'function',
        hasAppInfo: !!window.__HC_APP_INFO,
        status: window.__HC_STATUS ?? null,
        config,
      };
    });

    if (!bridgeState.hasTokenPromise) {
      throw new Error('window.__HC_TOKEN_PROMISE is missing');
    }
    pass(`browser runtime exposed the Holochain token promise in ${browserName}`);

    if (!bridgeState.config || bridgeState.config.admin_port !== 33800 || bridgeState.config.app_port !== 8888) {
      throw new Error(`unexpected conductor config payload: ${JSON.stringify(bridgeState.config)}`);
    }
    pass(`browser runtime fetched the expected conductor config in ${browserName}`);

    if (!bridgeState.hasCallZome) {
      throw new Error(`window.__HC_CALL_ZOME is unavailable (status: ${bridgeState.status ?? 'unknown'})`);
    }
    pass(`browser runtime exposed the zome bridge with status ${bridgeState.status ?? 'unknown'} in ${browserName}`);
  } finally {
    await browser.close();
  }
}

let lastError = null;

for (const [browserType, executablePath, browserName] of [
  [chromium, chromiumPath, 'chromium'],
  [firefox, firefoxPath, 'firefox'],
]) {
  try {
    await verifyWith(browserType, executablePath, browserName);
    process.exit(0);
  } catch (error) {
    lastError = error;
    console.warn(`WARN: ${browserName} verification failed: ${error instanceof Error ? error.message : String(error)}`);
  }
}

try {
  if (lastError) {
    throw lastError;
  }
} catch (error) {
  fail(error instanceof Error ? error.message : String(error));
}

if (process.exitCode && process.exitCode !== 0) {
  process.exit(process.exitCode);
}
