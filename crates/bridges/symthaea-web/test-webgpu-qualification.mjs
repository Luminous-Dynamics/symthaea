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
const HEADLESS = process.env.WEBGPU_HEADLESS === 'false' ? false : 'new';
const CHECKED_OUT_SHA = (() => {
  try {
    return execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
  } catch {
    return null;
  }
})();
const EXPECTED_CHECKED_OUT_SHA = process.env.EXPECTED_CHECKED_OUT_SHA || null;

function commandVersion(command, args) {
  try {
    return execFileSync(command, args, { encoding: 'utf8' }).trim();
  } catch {
    return null;
  }
}

const QUALIFICATION_ENVIRONMENT = {
  node: process.version,
  chromium: commandVersion(CHROMIUM, ['--version']),
  rustc: commandVersion('rustc', ['--version']),
  runner_os: process.env.RUNNER_OS || null,
  runner_arch: process.env.RUNNER_ARCH || null,
  runner_name: process.env.RUNNER_NAME || null,
};
if (process.env.GITHUB_EVENT_NAME === 'pull_request' && !EXPECTED_CHECKED_OUT_SHA) {
  throw new Error('qualification missing EXPECTED_CHECKED_OUT_SHA for pull_request run');
}
if (EXPECTED_CHECKED_OUT_SHA && CHECKED_OUT_SHA !== EXPECTED_CHECKED_OUT_SHA) {
  throw new Error(
    `qualification checkout identity mismatch: expected ${EXPECTED_CHECKED_OUT_SHA}, got ${CHECKED_OUT_SHA}`,
  );
}

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

async function canvasPngHash(page, selector) {
  const dataUrl = await page.$eval(selector, canvas => {
    if (!(canvas instanceof HTMLCanvasElement)) {
      throw new Error(`selected element is not a canvas`);
    }
    return canvas.toDataURL('image/png');
  });
  const payload = Buffer.from(dataUrl.slice(dataUrl.indexOf(',') + 1), 'base64');
  if (payload.length < 100) {
    throw new Error(`${selector} produced an unexpectedly small PNG`);
  }
  return createHash('sha256').update(payload).digest('hex');
}

async function canvasPixelSamples(page, selector, points) {
  return page.$eval(
    selector,
    async (canvas, points) => {
      if (!(canvas instanceof HTMLCanvasElement)) {
        throw new Error(`selected element is not a canvas`);
      }
      const dataUrl = canvas.toDataURL('image/png');
      const image = new Image();
      image.src = dataUrl;
      await image.decode();
      const probe = document.createElement('canvas');
      probe.width = canvas.width;
      probe.height = canvas.height;
      const context = probe.getContext('2d');
      if (!context) {
        throw new Error(`could not create probe context for selected canvas`);
      }
      context.drawImage(image, 0, 0);
      const rgba = context.getImageData(0, 0, canvas.width, canvas.height).data;
      return points.map(({ name, x, y }) => {
        if (!Number.isInteger(x) || !Number.isInteger(y)
          || x < 0 || y < 0 || x >= canvas.width || y >= canvas.height) {
          throw new Error(`invalid semantic probe ${name}: (${x},${y})`);
        }
        const offset = (y * canvas.width + x) * 4;
        return {
          name,
          x,
          y,
          rgba: [...rgba.slice(offset, offset + 4)],
        };
      });
    },
    points,
  );
}

function assertSemanticSceneSamples(samples) {
  const byName = new Map(samples.map(sample => [sample.name, sample.rgba]));
  const background = byName.get('background');
  const polygon = byName.get('polygon');
  const transformedLine = byName.get('transformed-line');
  const circle = byName.get('circle');

  if (!background || !polygon || !transformedLine || !circle) {
    throw new QualificationError(
      `WebGPU semantic scene probes missing: ${JSON.stringify(samples)}`,
      'renderer',
    );
  }

  if (!(background[2] > background[0])) {
    throw new QualificationError(
      `WebGPU background probe has unexpected channel ordering: ${JSON.stringify(background)}`,
      'renderer',
    );
  }
  if (!(polygon[0] > polygon[2] + 40 && polygon[1] > polygon[2])) {
    throw new QualificationError(
      `WebGPU polygon probe does not preserve the fixture's warm fill: ${JSON.stringify(polygon)}`,
      'renderer',
    );
  }
  if (!(
    circle[2] > circle[0] + 40
    && circle[1] > circle[0]
  )) {
    throw new QualificationError(
      `WebGPU circle probe does not preserve the fixture's cool fill: ${JSON.stringify(circle)}`,
      'renderer',
    );
  }

  const lineBrightness = (transformedLine[0] + transformedLine[1] + transformedLine[2]) / 3;
  if (lineBrightness < 120 || transformedLine[3] === 0) {
    throw new QualificationError(
      `WebGPU transformed-line probe is unexpectedly dark: ${JSON.stringify(transformedLine)}`,
      'renderer',
    );
  }
}

function assertSemanticMovieSamples(samples, label = 'WebGPU', classification = 'renderer') {
  const byName = new Map(samples.map(sample => [sample.name, sample.rgba]));
  const left = byName.get('left');
  const right = byName.get('right');
  const top = byName.get('top');
  const bottom = byName.get('bottom');

  if (!left || !right || !top || !bottom) {
    throw new QualificationError(
      `${label} semantic movie probes missing: ${JSON.stringify(samples)}`,
      classification,
    );
  }
  if (!(left[0] + 20 < right[0])) {
    throw new QualificationError(
      `${label} movie horizontal gradient is not increasing red left-to-right: ${JSON.stringify(samples)}`,
      classification,
    );
  }
  if (!(top[1] + 20 < bottom[1])) {
    throw new QualificationError(
      `${label} movie vertical gradient is not increasing green top-to-bottom: ${JSON.stringify(samples)}`,
      classification,
    );
  }
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

async function waitForQualificationReady(page, selector) {
  try {
    await page.waitForFunction(
      selector => document.querySelector(selector)?.getAttribute('data-qualification-ready') === 'true',
      { timeout: 30_000 },
      selector,
    );
    await page.evaluate(() => new Promise(requestAnimationFrame));
    await page.evaluate(() => new Promise(requestAnimationFrame));
  } catch (error) {
    throw new QualificationError(
      `WebGPU projection ${selector} did not report render completion: ${error instanceof Error ? error.message : String(error)}`,
      'renderer',
    );
  }
}

async function waitForFallbackPaint(page) {
  try {
    await page.waitForFunction(
      () => {
        const canvas = document.querySelector('#canvas2d-movie-fallback');
        const image = document.querySelector('img.portrait');
        if (!(canvas instanceof HTMLCanvasElement) || !image) return false;
        if (getComputedStyle(canvas).display === 'none' || getComputedStyle(image).display === 'none') {
          return false;
        }
        const ctx = canvas.getContext('2d');
        if (!ctx || canvas.width < 32 || canvas.height < 24) return false;
        const pixels = ctx.getImageData(31, 23, 1, 1).data;
        return pixels[0] === 255
          && pixels[1] === 255
          && pixels[2] === 255
          && pixels[3] === 255
          && image.getAttribute('src')?.startsWith('data:image/svg+xml;base64,');
      },
      { timeout: 30_000 },
    );
  } catch (error) {
    throw new QualificationError(
      `Canvas2D/SVG fallback did not report its expected painted state: ${error instanceof Error ? error.message : String(error)}`,
      'fallback',
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
  const swiftShaderMode = mode === 'webgpu' || mode === 'webgpu-swiftshader';
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
      '--disable-dev-shm-usage',
      '--enable-features=Vulkan',
      '--use-angle=vulkan',
      '--disable-vulkan-surface',
    );
    if (swiftShaderMode) {
      args.push(
        '--use-vulkan=swiftshader',
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
    headless: HEADLESS,
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
      await waitForQualificationReady(page, '#webgpu-cognitive-canvas');
      await waitForQualificationReady(page, '#webgpu-movie-canvas');
      failOnPageErrors('WebGPU first render');

      const firstSceneHash = await canvasPngHash(page, '#webgpu-cognitive-canvas');
      const firstMovieHash = await canvasPngHash(page, '#webgpu-movie-canvas');
      const blankSceneHash = await blankCanvasHash(page, 512, 512);
      const blankMovieHash = await blankCanvasHash(page, 192, 192);

      const semanticSceneSamples = await canvasPixelSamples(page, '#webgpu-cognitive-canvas', [
        { name: 'background', x: 10, y: 10 },
        { name: 'polygon', x: 100, y: 100 },
        { name: 'transformed-line', x: 43, y: 371 },
        { name: 'circle', x: 360, y: 350 },
      ]);
      assertSemanticSceneSamples(semanticSceneSamples);

      const semanticMovieSamples = await canvasPixelSamples(page, '#webgpu-movie-canvas', [
        { name: 'left', x: 48, y: 96 },
        { name: 'right', x: 144, y: 96 },
        { name: 'top', x: 96, y: 48 },
        { name: 'bottom', x: 96, y: 144 },
      ]);
      assertSemanticMovieSamples(semanticMovieSamples);

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
        path: path.join(SCREENSHOT_DIR, `${mode}.png`),
        fullPage: false,
      });

      await page.reload({ waitUntil: 'domcontentloaded', timeout: 30_000 });
      await waitForProjection(page, '#webgpu-cognitive-canvas', 'block');
      await waitForProjection(page, '#webgpu-movie-canvas', 'block');
      await waitForQualificationReady(page, '#webgpu-cognitive-canvas');
      await waitForQualificationReady(page, '#webgpu-movie-canvas');
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
        semantic_scene_samples: semanticSceneSamples,
        semantic_movie_samples: semanticMovieSamples,
        deterministic_repeat: true,
        page_errors: pageErrors,
      };
    }

    await waitForProjection(page, '#canvas2d-movie-fallback', 'block');
    await waitForVisible(page, 'img.portrait');
    await waitForFallbackPaint(page);
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

    const semanticFallbackMovieSamples = await canvasPixelSamples(
      page,
      '#canvas2d-movie-fallback',
      [
        { name: 'left', x: 4, y: 12 },
        { name: 'right', x: 28, y: 12 },
        { name: 'top', x: 16, y: 2 },
        { name: 'bottom', x: 16, y: 22 },
      ],
    );
    assertSemanticMovieSamples(semanticFallbackMovieSamples, 'Canvas2D fallback', 'fallback');

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
    await waitForFallbackPaint(page);
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
      path: path.join(SCREENSHOT_DIR, `${mode}.png`),
      fullPage: false,
    });

    return {
      mode,
      qualification_profile: 'forced-gpu-disabled',
      capability,
      fallback,
      canvas_hash: firstFallbackCanvasHash,
      portrait_hash: firstPortraitHash,
      semantic_movie_samples: semanticFallbackMovieSamples,
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
    schema: 'symthaea-ui-webgpu-qualification-v2',
    url: URL,
    chromium: CHROMIUM,
    headed_under_xvfb: HEADLESS === false,
    modes: MODES,
    workflow_sha: process.env.GITHUB_SHA || null,
    checked_out_sha: CHECKED_OUT_SHA,
    expected_pr_head_sha: EXPECTED_CHECKED_OUT_SHA,
    qualification_environment: QUALIFICATION_ENVIRONMENT,
    run_id: process.env.GITHUB_RUN_ID || null,
    workflow_ref: process.env.GITHUB_WORKFLOW_REF || null,
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
