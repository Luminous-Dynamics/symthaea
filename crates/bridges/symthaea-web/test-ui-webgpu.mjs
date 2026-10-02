#!/usr/bin/env node
/**
 * Deterministic browser qualification for the symthaea-ui WebGPU path.
 *
 * This is intentionally separate from test-browser.mjs: the legacy smoke
 * harness disables GPU acceleration. This lane exercises the WebGPU path with
 * an explicit browser configuration, then runs a forced-fallback pass.
 *
 * Usage:
 *   node test-ui-webgpu.mjs [url]
 *
 * Environment:
 *   WEBGPU_MODE=swiftshader|hardware  (default: swiftshader)
 *   SYMTHAEA_UI_DIST=../symthaea-ui/dist
 *   EXPECTED_COGNITIVE_SHA256=<optional expected screenshot hash>
 *   EXPECTED_MOVIE_SHA256=<optional expected screenshot hash>
 */

import puppeteer from 'puppeteer-core';
import { execSync, spawn } from 'child_process';
import { createHash } from 'crypto';
import { existsSync, readFileSync, mkdirSync, writeFileSync } from 'fs';
import path from 'path';

const ROOT = import.meta.dirname;
const DIST_DIR = process.env.SYMTHAEA_UI_DIST
  ? path.resolve(ROOT, process.env.SYMTHAEA_UI_DIST)
  : path.resolve(ROOT, '../symthaea-ui/dist');
const TARGET_URL = process.argv[2] || 'http://localhost:8401/?symthaea_webgpu_fixture=1';
const ARTIFACT_DIR = path.join(ROOT, 'webgpu-qualification');
const MODE = process.env.WEBGPU_MODE || 'swiftshader';

function findExecutable() {
  if (process.env.CHROMIUM) return process.env.CHROMIUM;
  for (const candidate of ['chromium', 'chromium-browser', 'google-chrome', 'google-chrome-stable']) {
    try {
      const value = execSync(`which ${candidate}`, { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim();
      if (value) return value;
    } catch {}
  }
  throw new Error('No Chromium-compatible executable found');
}

function sha256(file) {
  return createHash('sha256').update(readFileSync(file)).digest('hex');
}

function visible(page, selector) {
  return page.$eval(selector, el => {
    const style = getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
  });
}

async function waitVisible(page, selector, timeout = 15000) {
  await page.waitForFunction(
    sel => {
      const el = document.querySelector(sel);
      if (!el) return false;
      const style = getComputedStyle(el);
      const rect = el.getBoundingClientRect();
      return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
    },
    { timeout },
    selector,
  );
}

async function runBrowser({ name, args, requireWebGpu, requireFallback }) {
  const browser = await puppeteer.launch({
    executablePath: findExecutable(),
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', ...args],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 1 });

  const errors = [];
  const warnings = [];
  page.on('console', msg => {
    if (msg.type() === 'error') errors.push(msg.text());
    else if (msg.type() === 'warning') warnings.push(msg.text());
  });
  page.on('pageerror', err => errors.push(`PAGE ERROR: ${err.message}`));

  const result = {
    name,
    mode: MODE,
    url: TARGET_URL,
    errors,
    warnings,
  };

  try {
    await page.goto(TARGET_URL, { waitUntil: 'networkidle2', timeout: 30000 });
    await new Promise(resolve => setTimeout(resolve, 2000));

    result.capability = await page.evaluate(async () => {
      const gpu = navigator.gpu;
      if (!gpu) return { navigator_gpu: false, adapter: false, device: false };
      const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) return { navigator_gpu: true, adapter: false, device: false };
      try {
        const device = await adapter.requestDevice();
        return {
          navigator_gpu: true,
          adapter: true,
          device: true,
          feature_count: adapter.features.size,
          max_texture_dimension_2d: device.limits.maxTextureDimension2D,
        };
      } catch (error) {
        return {
          navigator_gpu: true,
          adapter: true,
          device: false,
          error: String(error),
        };
      }
    });

    result.canvas_contract = await page.evaluate(() => ({
      cognitive: Boolean(document.querySelector('#webgpu-cognitive-canvas')),
      movie_webgpu: Boolean(document.querySelector('#webgpu-movie-canvas')),
      movie_fallback: Boolean(document.querySelector('#canvas2d-movie-fallback')),
    }));

    if (requireWebGpu) {
      if (!result.capability.navigator_gpu || !result.capability.adapter || !result.capability.device) {
        throw new Error(`WebGPU capability preflight failed: ${JSON.stringify(result.capability)}`);
      }
      await waitVisible(page, '#webgpu-cognitive-canvas');
      await waitVisible(page, '#webgpu-movie-canvas');
      if (await visible(page, '#canvas2d-movie-fallback')) {
        throw new Error('Canvas2D movie fallback is visible during WebGPU qualification');
      }

      const cognitivePath = path.join(ARTIFACT_DIR, `${name}-cognitive.png`);
      const moviePath = path.join(ARTIFACT_DIR, `${name}-movie.png`);
      await (await page.$('#webgpu-cognitive-canvas')).screenshot({ path: cognitivePath });
      await (await page.$('#webgpu-movie-canvas')).screenshot({ path: moviePath });
      result.cognitive_sha256 = sha256(cognitivePath);
      result.movie_sha256 = sha256(moviePath);
      result.fixture_canvas_sizes = await page.evaluate(() => ({
        cognitive: [document.querySelector('#webgpu-cognitive-canvas').width, document.querySelector('#webgpu-cognitive-canvas').height],
        movie: [document.querySelector('#webgpu-movie-canvas').width, document.querySelector('#webgpu-movie-canvas').height],
      }));
      result.expected_cognitive_match =
        !process.env.EXPECTED_COGNITIVE_SHA256 ||
        process.env.EXPECTED_COGNITIVE_SHA256 === result.cognitive_sha256;
      result.expected_movie_match =
        !process.env.EXPECTED_MOVIE_SHA256 ||
        process.env.EXPECTED_MOVIE_SHA256 === result.movie_sha256;
      if (!result.expected_cognitive_match || !result.expected_movie_match) {
        throw new Error('Deterministic screenshot hash mismatch');
      }
    }

    if (requireFallback) {
      await page.waitForFunction(
        () => {
          const canvas = document.querySelector('#webgpu-cognitive-canvas');
          return canvas && getComputedStyle(canvas).display === 'none';
        },
        { timeout: 15000 },
      );
      await waitVisible(page, '#canvas2d-movie-fallback');
      await waitVisible(page, '#svg-cognitive-fallback');
      if (await visible(page, '#webgpu-movie-canvas')) {
        throw new Error('WebGPU movie canvas is visible during forced fallback');
      }
      result.fallback = await page.evaluate(() => ({
        cognitive_webgpu_hidden: getComputedStyle(document.querySelector('#webgpu-cognitive-canvas')).display === 'none',
        cognitive_svg_visible: getComputedStyle(document.querySelector('#svg-cognitive-fallback')).display !== 'none',
        movie_webgpu_hidden: getComputedStyle(document.querySelector('#webgpu-movie-canvas')).display === 'none',
        movie_canvas2d_visible: getComputedStyle(document.querySelector('#canvas2d-movie-fallback')).display !== 'none',
      }));
    }

    result.ok = errors.length === 0;
  } finally {
    await browser.close();
  }

  if (!result.ok) {
    throw new Error(`${name} observed console errors: ${JSON.stringify(errors)}`);
  }
  return result;
}

mkdirSync(ARTIFACT_DIR, { recursive: true });

let server = null;
if (TARGET_URL.startsWith('http://localhost:') || TARGET_URL.startsWith('https://localhost:')) {
  if (!existsSync(DIST_DIR)) {
    throw new Error(`symthaea-ui dist not found at ${DIST_DIR}; build the WASM UI with the browser-qualification feature first`);
  }
  const port = new globalThis.URL(TARGET_URL).port || '8401';
  server = spawn('python3', ['-m', 'http.server', port, '--bind', '127.0.0.1', '--directory', DIST_DIR], {
    stdio: 'ignore',
    detached: true,
  });
  await new Promise(resolve => setTimeout(resolve, 1200));
}

try {
  const gpuArgs = MODE === 'hardware'
    ? ['--enable-gpu', '--enable-unsafe-webgpu']
    : ['--enable-gpu', '--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--use-gl=angle', '--use-angle=swiftshader'];

  const qualification = await runBrowser({
    name: MODE === 'hardware' ? 'hardware-webgpu' : 'swiftshader-webgpu',
    args: gpuArgs,
    requireWebGpu: true,
    requireFallback: false,
  });

  const fallback = await runBrowser({
    name: 'forced-fallback',
    args: ['--disable-gpu'],
    requireWebGpu: false,
    requireFallback: true,
  });

  const manifest = {
    contract: 'symthaea-ui-webgpu-qualification-v1',
    generated_at: new Date().toISOString(),
    qualification,
    fallback,
  };
  const manifestPath = path.join(ARTIFACT_DIR, 'manifest.json');
  writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + '\n');
  console.log(JSON.stringify(manifest, null, 2));
  console.log(`Artifacts: ${ARTIFACT_DIR}`);
} finally {
  if (server) {
    process.kill(-server.pid);
  }
}
