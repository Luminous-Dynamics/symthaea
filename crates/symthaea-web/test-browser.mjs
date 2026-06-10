#!/usr/bin/env node
/**
 * Headless browser test for the Symthaea Leptos portal.
 * Takes screenshots of each tab and captures console errors.
 *
 * Usage: node test-browser.mjs [url]
 * Default URL: http://localhost:8093 (local dist serve)
 */

import puppeteer from 'puppeteer-core';
import { execSync, spawn } from 'child_process';
import { existsSync, mkdirSync } from 'fs';
import path from 'path';

const URL = process.argv[2] || 'http://localhost:8093';
const SCREENSHOT_DIR = path.join(import.meta.dirname, 'screenshots');
const CHROMIUM = execSync('which chromium').toString().trim();

if (!existsSync(SCREENSHOT_DIR)) mkdirSync(SCREENSHOT_DIR);

// Start local server if testing locally
let server = null;
if (URL.includes('localhost')) {
  const distDir = path.join(import.meta.dirname, 'dist');
  if (!existsSync(distDir)) {
    console.error('dist/ not found — run trunk build first');
    process.exit(1);
  }
  server = spawn('python3', ['-m', 'http.server', '8093', '--bind', '127.0.0.1', '--directory', distDir], {
    stdio: 'ignore',
    detached: true,
  });
  await new Promise(r => setTimeout(r, 1500));
  console.log('Local server started on :8093');
}

const consoleErrors = [];
const consoleWarnings = [];
const consoleLogs = [];

try {
  const browser = await puppeteer.launch({
    executablePath: CHROMIUM,
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu'],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 900 });

  // Capture console output
  page.on('console', msg => {
    const type = msg.type();
    const text = msg.text();
    if (type === 'error') consoleErrors.push(text);
    else if (type === 'warning') consoleWarnings.push(text);
    else consoleLogs.push(text);
  });

  page.on('pageerror', err => consoleErrors.push(`PAGE ERROR: ${err.message}`));

  // Load the page
  console.log(`\nLoading ${URL}...`);
  await page.goto(URL, { waitUntil: 'networkidle2', timeout: 30000 });
  await new Promise(r => setTimeout(r, 3000)); // Wait for WASM init

  // Screenshot the initial view
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, '01-initial.png'), fullPage: false });
  console.log('  📸 01-initial.png');

  // Check page title
  const title = await page.title();
  console.log(`  Title: "${title}"`);

  // Check if Leptos rendered (look for any content)
  const bodyText = await page.evaluate(() => document.body?.innerText?.substring(0, 200) || 'EMPTY');
  console.log(`  Body text: "${bodyText.replace(/\n/g, ' ').substring(0, 100)}..."`);

  // Check if tabs exist
  const tabs = await page.evaluate(() => {
    const buttons = document.querySelectorAll('.tab, button[data-tab], nav button');
    return Array.from(buttons).map(b => b.textContent?.trim());
  });
  console.log(`  Tabs found: ${JSON.stringify(tabs)}`);

  // Click each tab and screenshot
  const tabNames = ['Chat', 'Topology', 'Experiments', 'Dreams', 'Inoculate'];
  for (let i = 0; i < tabNames.length; i++) {
    const tabName = tabNames[i];
    try {
      // Try clicking by text content
      const clicked = await page.evaluate((name) => {
        const buttons = document.querySelectorAll('button');
        for (const btn of buttons) {
          if (btn.textContent?.trim() === name) {
            btn.click();
            return true;
          }
        }
        return false;
      }, tabName);

      if (clicked) {
        await new Promise(r => setTimeout(r, 1000));
        await page.screenshot({
          path: path.join(SCREENSHOT_DIR, `0${i + 2}-${tabName.toLowerCase()}.png`),
          fullPage: false
        });
        console.log(`  📸 0${i + 2}-${tabName.toLowerCase()}.png`);
      } else {
        console.log(`  ⚠️  Tab "${tabName}" not found`);
      }
    } catch (e) {
      console.log(`  ❌ Tab "${tabName}" error: ${e.message}`);
    }
  }

  // Check WASM loading
  const wasmLoaded = await page.evaluate(() => {
    return typeof window.wasmBindings !== 'undefined' ||
           document.querySelector('script[type="module"]') !== null;
  });
  console.log(`\n  WASM module: ${wasmLoaded ? '✓ loaded' : '✗ not found'}`);

  // Check worker
  const workerStatus = await page.evaluate(() => {
    // Check if any worker messages were logged
    return document.body?.innerHTML?.includes('consciousness') ||
           document.body?.innerHTML?.includes('Phi') ||
           document.body?.innerHTML?.includes('Symthaea');
  });
  console.log(`  Content rendered: ${workerStatus ? '✓ yes' : '✗ no'}`);

  await browser.close();

  // Report
  console.log('\n' + '='.repeat(50));
  console.log('CONSOLE ERRORS:', consoleErrors.length);
  consoleErrors.forEach(e => console.log('  ❌', e.substring(0, 200)));
  console.log('CONSOLE WARNINGS:', consoleWarnings.length);
  consoleWarnings.slice(0, 5).forEach(w => console.log('  ⚠️ ', w.substring(0, 200)));
  console.log('CONSOLE LOGS:', consoleLogs.length);
  consoleLogs.slice(0, 5).forEach(l => console.log('  📝', l.substring(0, 200)));

  console.log('\n' + '='.repeat(50));
  if (consoleErrors.length === 0) {
    console.log('✅ NO ERRORS — portal looks healthy');
  } else {
    console.log(`❌ ${consoleErrors.length} ERRORS — needs fixing`);
  }

  console.log(`\nScreenshots saved to: ${SCREENSHOT_DIR}/`);

} catch (e) {
  console.error('Browser test failed:', e.message);
} finally {
  if (server) {
    process.kill(-server.pid);
  }
}
