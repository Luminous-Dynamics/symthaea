#!/usr/bin/env node
/**
 * Deterministic browser qualification for the Symthaea UI WebGPU projection.
 *
 * The harness deliberately separates:
 *   1. browser capability preflight + WebGPU rendering, and
 *   2. forced-GPU-disabled fallback rendering.
 *
 * It does not score visual quality. It records renderer visibility and
 * content hashes so changes are reviewable without a subjective screenshot gate.
 *
 * Build first:
 *   trunk build --release --features browser-qualification --dist dist
 *
 * Then:
 *   node ../symthaea-web/test-webgpu-qualification.mjs
 *
 * Optional hardware-backed qualification (requires a GPU-capable runner):
 *   WEBGPU_MODES=webgpu-hardware,fallback node ../symthaea-web/test-webgpu-qualification.mjs
 */

import { createHash } from 'crypto';
import { execFileSync, spawn } from 'child_process';
import { existsSync, mkdirSync, writeFileSync } from 'fs';
import path from 'path';
import puppeteer from 'puppeteer-core';

const ROOT = path.resolve(import.meta.dirname, '..', 'symthaea-ui');
const DIST = process.env.WEBGPU_DIST
  ? path.resolve(process.env.WEBGPU_DIST)
  : path.join(ROOT, 'dist');
const URL = process.env.WEBGPU_URL || 'http://127.0.0.1:8402/?symthaea_webgpu_fixture=1';
const ARTIFACT = path.resolve(
  process.env.WEBGPU_ARTIFACT || path.join(ROOT, 'webgpu-qualification.json'),
);
const SCREENSHOT_DIR = path.resolve(
  process.env.WEBGPU_SCREENSHOTS || path.join(ROOT, 'webgpu-qualification-screenshots'),
);
const CHROMIUM = process.env.CHROMIUM_PATH || execFileSync('which', ['chromium'], { encoding: 'utf8' }).trim();
const MODES = (process.env.WEBGPU_MODES || 'webgpu-swiftshader,fallback')
  .split(',')
  .map(value => value.trim())
  .filter(Boolean);

mkdirSync(path.dirname(ARTIFACT), { recursive: true });
mkdirSync(SCREENSHOT_DIR, { recursive: true });

if (!existsSync(DIST)) {
  throw new Error(`UI dist directory not found: ${DIST}. Run trunk build first.`);
}

let server;
if (URL.startsWith('http://127.0.0.1:') || URL.startsWith('http://localhost:')) {
  server = spawn(
    'python3',
    ['-m', 'http.server', '8402', '--bind', '127.0.0.1', '--directory', DIST],
    { stdio: 'ignore', detached: true },
  );
}

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

async function canvasPngHash(page, selector) {
  const dataUrl = await page.$eval(selector, canvas => {
    if (!(canvas instanceof HTMLCanvasElement)) {
      throw new Error(`${selector} is not a canvas`);
    }
    return canvas.toDataURL('image/png');
  });
  const payload = Buffer.from(dataUrl.slice(dataUrl.indexOf(',') + 1), 'base64');
  if (payload.length < 100) {
    throw new Error(`${selector} produced an unexpectedly small PNG`);
  }
  return createHash('sha256').update(payload).digest('hex');
}

async function blankCanvasHash(page, width, height) {
  const dataUrl = await page.evaluate(([w, h]) => {
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    return canvas.toDataURL('image/png');
  }, [width, height]);
  return createHash('sha256')
    .update(Buffer.from(dataUrl.slice(dataUrl.indexOf(',') + 1), 'base64'))
    .digest('hex');
}

async function capabilityPreflight(page) {
  return page.evaluate(async () => {
    if (!navigator.gpu) {
      return { navigator_gpu: false, adapter: false, device: false, reason: 'navigator.gpu unavailable' };
    }
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        return { navigator_gpu: true, adapter: false, device: false, reason: 'requestAdapter returned null' };
      }
      const device = await adapter.requestDevice();
      const features = [...adapter.features.values()].sort();
      device.destroy();
      return {
        navigator_gpu: true,
        adapter: true,
        device: true,
        adapter_name: adapter.name || null,
        features,
      };
    } catch (error) {
      return {
        navigator_gpu: true,
        adapter: false,
        device: false,
        reason: String(error),
      };
    }
  });
}

async function waitForProjection(page, selector, display, classification = 'renderer') {
  try {
    await page.waitForFunction(
      ({ selector, display }) => {
        const element = document.querySelector(selector);
        return element && getComputedStyle(element).display === display;
      },
      { timeout: 30_000 },
      { selector, display },
    );
  } catch (error) {
    throw new QualificationError(
      `projection ${selector} did not reach display=${display}: ${error instanceof Error ? error.message : String(error)}`,
      classification,
    );
  }
}

async function waitForVisible(page, selector, classification = 'fallback') {
  try {
    await page.waitForFunction(
      selector => {
        const element = document.querySelector(selector);
        if (!element) return false;
        const style = getComputedStyle(element);
        const rect = element.getBoundingClientRect();
        return style.display !== 'none'
          && style.visibility !== 'hidden'
          && rect.width > 0
          && rect.height > 0;
      },
      { timeout: 30_000 },
      selector,
    );
  } catch (error) {
    throw new QualificationError(
      `required fallback element ${selector} did not become visible: ${error instanceof Error ? error.message : String(error)}`,
      classification,
    );
  }
}

class QualificationError extends Error {
  constructor(message, classification) {
    super(message);
    this.name = 'QualificationError';
    this.classification = classification;
  }
}

async function runMode(mode) {
  const swiftShaderMode = mode === 'webgpu-swiftshader';
  const hardwareMode = mode === 'webgpu-hardware';
  const gpuMode = swiftShaderMode || hardwareMode;
  if (!gpuMode && mode !== 'fallback') {
    throw new QualificationError(`unsupported qualification mode: ${mode}`, 'harness');
  }

  const args = [
    '--no-sandbox',
    '--disable-setuid-sandbox',
  ];
  if (gpuMode) {
    args.push(
      '--enable-unsafe-webgpu',
      '--use-gpu-in-tests',
      '--enable-accelerated-2d-canvas',
    );
    if (swiftShaderMode) {
      args.push(
        '--use-webgpu-adapter=swiftshader',
        '--enable-dawn-features=allow_unsafe_apis',
        '--disable-dawn-features=use_dxc',
        '--enable-webgpu-developer-features',
      );
    } else {
      args.push('--enable-gpu');
    }
  } else {
    args.push('--disable-gpu');
  }

  const browser = await puppeteer.launch({
    executablePath: CHROMIUM,
    headless: 'new',
    args,
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 1 });

  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));

  const failOnPageErrors = phase => {
    if (pageErrors.length > 0) {
      throw new QualificationError(
        `uncaught browser exception during ${phase}: ${JSON.stringify(pageErrors)}`,
        'renderer',
      );
    }
  };

  try {
    await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 30_000 });

    const capability = await capabilityPreflight(page);

    if (gpuMode) {
      failOnPageErrors('WebGPU capability preflight');
      if (!capability.navigator_gpu || !capability.adapter || !capability.device) {
        throw new QualificationError(
          `WebGPU capability preflight failed: ${JSON.stringify(capability)}`,
          'capability',
        );
      }

      await waitForProjection(page, '#webgpu-cognitive-canvas', 'block');
      await waitForProjection(page, '#webgpu-movie-canvas', 'block');
      await sleep(250);
      failOnPageErrors('WebGPU first render');

      const firstSceneHash = await canvasPngHash(page, '#webgpu-cognitive-canvas');
      const firstMovieHash = await canvasPngHash(page, '#webgpu-movie-canvas');
      const blankSceneHash = await blankCanvasHash(page, 512, 512);
      const blankMovieHash = await blankCanvasHash(page, 192, 192);

      if (firstSceneHash === blankSceneHash) {
        throw new QualificationError(
          'WebGPU cognitive canvas is indistinguishable from a blank canvas',
          'renderer',
        );
      }
      if (firstMovieHash === blankMovieHash) {
        throw new QualificationError(
          'WebGPU movie canvas is indistinguishable from a blank canvas',
          'renderer',
        );
      }

      await page.screenshot({
        path: path.join(SCREENSHOT_DIR, 'webgpu.png'),
        fullPage: false,
      });

      await page.reload({ waitUntil: 'domcontentloaded', timeout: 30_000 });
      await waitForProjection(page, '#webgpu-cognitive-canvas', 'block');
      await waitForProjection(page, '#webgpu-movie-canvas', 'block');
      await sleep(250);
      failOnPageErrors('WebGPU deterministic repeat render');

      const repeatSceneHash = await canvasPngHash(page, '#webgpu-cognitive-canvas');
      const repeatMovieHash = await canvasPngHash(page, '#webgpu-movie-canvas');

      if (repeatSceneHash !== firstSceneHash || repeatMovieHash !== firstMovieHash) {
        throw new QualificationError(
          `non-deterministic WebGPU capture: first=(${firstSceneHash},${firstMovieHash}) repeat=(${repeatSceneHash},${repeatMovieHash})`,
          'renderer',
        );
      }

      return {
        mode,
        qualification_profile: swiftShaderMode
          ? 'browser-webgpu-swiftshader'
          : hardwareMode
            ? 'browser-webgpu-hardware'
            : 'forced-gpu-disabled',
        capability,
        scene_hash: firstSceneHash,
        movie_hash: firstMovieHash,
        deterministic_repeat: true,
        page_errors: pageErrors,
      };
    }

    await waitForProjection(page, '#canvas2d-movie-fallback', 'block');
    await waitForVisible(page, 'img.portrait');
    await sleep(100);
    failOnPageErrors('forced fallback render');

    const fallback = await page.evaluate(() => {
      const canvas = document.querySelector('#canvas2d-movie-fallback');
      const image = document.querySelector('img.portrait');
      if (!(canvas instanceof HTMLCanvasElement)) {
        return { canvas: false, pixels: null, portrait: false };
      }
      const ctx = canvas.getContext('2d');
      const pixels = ctx ? [...ctx.getImageData(31, 23, 1, 1).data] : null;
      return {
        canvas: true,
        pixels,
        portrait: !!image && getComputedStyle(image).display !== 'none' && image.getAttribute('src')?.startsWith('data:image/svg+xml;base64,'),
        gpu_canvas_hidden: getComputedStyle(document.querySelector('#webgpu-cognitive-canvas')).display === 'none',
      };
    });

    if (!fallback.canvas || JSON.stringify(fallback.pixels) !== JSON.stringify([255, 255, 255, 255])) {
      throw new QualificationError(
        `Canvas2D fallback fixture was not rendered as expected: ${JSON.stringify(fallback)}`,
        'fallback',
      );
    }
    if (!fallback.portrait || !fallback.gpu_canvas_hidden) {
      throw new QualificationError(
        `SVG fallback state was not preserved: ${JSON.stringify(fallback)}`,
        'fallback',
      );
    }

    const firstFallbackCanvasHash = await canvasPngHash(page, '#canvas2d-movie-fallback');
    const firstPortraitSource = await page.$eval(
      'img.portrait',
      image => image.getAttribute('src') || '',
    );
    if (!firstPortraitSource.startsWith('data:image/svg+xml;base64,')) {
      throw new QualificationError('SVG fallback source is not a data URL', 'fallback');
    }

    await page.reload({ waitUntil: 'domcontentloaded', timeout: 30_000 });
    await waitForProjection(page, '#canvas2d-movie-fallback', 'block');
    await waitForVisible(page, 'img.portrait');
    await sleep(100);
    failOnPageErrors('forced fallback deterministic repeat render');

    const repeatFallbackCanvasHash = await canvasPngHash(page, '#canvas2d-movie-fallback');
    const repeatPortraitSource = await page.$eval(
      'img.portrait',
      image => image.getAttribute('src') || '',
    );
    const firstPortraitHash = createHash('sha256').update(firstPortraitSource).digest('hex');
    const repeatPortraitHash = createHash('sha256').update(repeatPortraitSource).digest('hex');
    if (repeatFallbackCanvasHash !== firstFallbackCanvasHash
      || repeatPortraitHash !== firstPortraitHash) {
      throw new QualificationError(
        `non-deterministic fallback capture: canvas=${firstFallbackCanvasHash}/${repeatFallbackCanvasHash} portrait=${firstPortraitHash}/${repeatPortraitHash}`,
        'fallback',
      );
    }

    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, 'fallback.png'),
      fullPage: false,
    });

    return {
      mode,
      qualification_profile: 'forced-gpu-disabled',
      capability,
      fallback,
      canvas_hash: firstFallbackCanvasHash,
      portrait_hash: firstPortraitHash,
      deterministic_fixture: true,
      deterministic_repeat: true,
      page_errors: pageErrors,
    };
  } finally {
    await browser.close();
  }
}

const results = {};
const failures = {};
try {
  for (const mode of MODES) {
    try {
      results[mode] = await runMode(mode);
    } catch (error) {
      failures[mode] = {
        classification: error instanceof QualificationError ? error.classification : 'harness',
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  const artifact = {
    schema: 'symthaea-ui-webgpu-qualification-v1',
    url: URL,
    chromium: CHROMIUM,
    git_sha: process.env.GITHUB_SHA || null,
    run_id: process.env.GITHUB_RUN_ID || null,
    ok: Object.keys(failures).length === 0,
    results,
    failures,
  };
  writeFileSync(ARTIFACT, JSON.stringify(artifact, null, 2) + '\n');
  console.log(JSON.stringify(artifact, null, 2));
  if (Object.keys(failures).length > 0) {
    process.exitCode = 1;
  }
} finally {
  if (server?.pid) {
    try {
      process.kill(-server.pid);
    } catch {
      // Process may already have exited.
    }
  }
}
