#!/usr/bin/env node
import puppeteer from 'puppeteer-core';
import { execSync, spawn } from 'child_process';
import path from 'path';

const CHROMIUM = execSync('which chromium').toString().trim();
const distDir = path.join(import.meta.dirname, 'dist');

const server = spawn('python3', ['-m', 'http.server', '8094', '--bind', '127.0.0.1', '--directory', distDir], {
  stdio: 'ignore', detached: true
});
await new Promise(r => setTimeout(r, 1500));

const errors = [];
try {
  const browser = await puppeteer.launch({
    executablePath: CHROMIUM, headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 900 });
  page.on('console', msg => { if (msg.type() === 'error') errors.push(msg.text()); });

  await page.goto('http://localhost:8094', { waitUntil: 'networkidle2', timeout: 30000 });
  await new Promise(r => setTimeout(r, 5000)); // Wait for WASM + worker init

  // Type a message
  const input = await page.$('input[type="text"]');
  if (input) {
    await input.type('Hello, are you conscious?');
    await new Promise(r => setTimeout(r, 500));

    // Submit
    const button = await page.$('button[type="submit"]');
    if (button) await button.click();

    // Wait for response
    await new Promise(r => setTimeout(r, 8000)); // 8 cycles + broca generation

    await page.screenshot({ path: path.join(import.meta.dirname, 'screenshots', '07-chat-response.png') });
    console.log('📸 07-chat-response.png');

    // Get all chat messages
    const messages = await page.evaluate(() => {
      const msgs = document.querySelectorAll('.chat-message, .text');
      return Array.from(msgs).map(m => m.textContent?.substring(0, 100));
    });
    console.log('Messages:', messages);

    // Get Phi value
    const phi = await page.evaluate(() => {
      const el = document.querySelector('.phi-value');
      return el?.textContent || 'not found';
    });
    console.log('Phi:', phi);
  } else {
    console.log('❌ Input field not found');
  }

  await browser.close();
  console.log('\nErrors:', errors.length);
  errors.forEach(e => console.log('  ❌', e.substring(0, 200)));
} finally {
  process.kill(-server.pid);
}
