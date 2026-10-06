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
const URL = process.env.WEBGPU_URL || 'http://127.0.0.1:8402/?symthaea_webgpu_fixture=1&symthaea_webgpu_recovery=1';
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

function sha256File(filePath) {
  const output = commandVersion('sha256sum', [filePath]);
  return output?.split(/\s+/)[0] || null;
}

const REPO_ROOT = path.resolve(import.meta.dirname, '..', '..', '..');
const QUALIFICATION_INPUT_DIGESTS = {
  harness: sha256File(import.meta.filename),
  package_json: sha256File(path.join(import.meta.dirname, 'package.json')),
  package_lock: sha256File(path.join(import.meta.dirname, 'package-lock.json')),
  checked_out_workflow: sha256File(
    path.join(REPO_ROOT, '.github', 'workflows', 'symthaea-ui-webgpu.yml'),
  ),
};

const QUALIFICATION_ENVIRONMENT = {
  node: process.version,
  chromium: commandVersion(CHROMIUM, ['--version']),
  chromium_binary_sha256: sha256File(CHROMIUM),
  chromium_package: commandVersion('dpkg-query', ['-W', 'chromium']),
  rustc: commandVersion('rustc', ['--version']),
  cargo: commandVersion('cargo', ['--version']),
  trunk: commandVersion('trunk', ['--version']),
  wasm_bindgen: commandVersion('wasm-bindgen', ['--version']),
  npm: commandVersion('npm', ['--version']),
  runner_os: process.env.RUNNER_OS || null,
  runner_arch: process.env.RUNNER_ARCH || null,
  runner_name: process.env.RUNNER_NAME || null,
  chromium_ozone_platform: 'x11',
};
const missingInputDigests = Object.entries(QUALIFICATION_INPUT_DIGESTS)
  .filter(([, digest]) => !/^[a-f0-9]{64}$/.test(digest || ''))
  .map(([name]) => name);
if (missingInputDigests.length > 0) {
  throw new Error('qualification input digest collection failed: ' + missingInputDigests.join(', '));
}

const requiredEnvironmentFields = [
  'node',
  'chromium',
  'chromium_binary_sha256',
  'chromium_package',
  'rustc',
  'cargo',
  'trunk',
  'wasm_bindgen',
  'npm',
];
const missingEnvironmentFields = requiredEnvironmentFields.filter(
  field => !QUALIFICATION_ENVIRONMENT[field],
);
if (missingEnvironmentFields.length > 0) {
  throw new Error('qualification environment provenance incomplete: ' + missingEnvironmentFields.join(', '));
}
if (process.env.GITHUB_EVENT_NAME === 'pull_request' && !EXPECTED_CHECKED_OUT_SHA) {
  throw new Error('qualification missing EXPECTED_CHECKED_OUT_SHA for pull_request run');
}
if (EXPECTED_CHECKED_OUT_SHA && CHECKED_OUT_SHA !== EXPECTED_CHECKED_OUT_SHA) {
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


async function canvasPixelStatistics(page, selector) {
  return page.$eval(selector, async canvas => {
    if (!(canvas instanceof HTMLCanvasElement)) {
      throw new Error('selected element is not a canvas');
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
      throw new Error('could not create statistics probe context');
    }
    context.drawImage(image, 0, 0);
    const rgba = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let opaqueBlack = 0;
    let nonOpaque = 0;
    let nonBlack = 0;
    let minX = canvas.width;
    let minY = canvas.height;
    let maxX = -1;
    let maxY = -1;
    const samples = [];
    const stepX = Math.max(1, Math.floor(canvas.width / 16));
    const stepY = Math.max(1, Math.floor(canvas.height / 16));
    for (let y = 0; y < canvas.height; y += stepY) {
      for (let x = 0; x < canvas.width; x += stepX) {
        const offset = (y * canvas.width + x) * 4;
        samples.push({ x, y, rgba: [...rgba.slice(offset, offset + 4)] });
      }
    }
    for (let y = 0; y < canvas.height; y++) {
      for (let x = 0; x < canvas.width; x++) {
        const offset = (y * canvas.width + x) * 4;
        const r = rgba[offset];
        const g = rgba[offset + 1];
        const b = rgba[offset + 2];
        const a = rgba[offset + 3];
        if (r === 0 && g === 0 && b === 0 && a === 255) {
          opaqueBlack++;
        }
        if (a !== 255) {
          nonOpaque++;
        }
        if (r !== 0 || g !== 0 || b !== 0) {
          nonBlack++;
          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          maxX = Math.max(maxX, x);
          maxY = Math.max(maxY, y);
        }
      }
    }
    return {
      width: canvas.width,
      height: canvas.height,
      opaque_black_pixels: opaqueBlack,
      non_opaque_pixels: nonOpaque,
      non_black_pixels: nonBlack,
      non_black_fraction: (canvas.width * canvas.height) > 0
        ? nonBlack / (canvas.width * canvas.height)
        : 0,
      non_black_bounds: maxX >= 0 ? { min_x: minX, min_y: minY, max_x: maxX, max_y: maxY } : null,
      coarse_samples: samples,
    };
  });
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

async function screenshotCanvasPixelStatistics(page, selector) {
  const geometry = await page.$eval(selector, canvas => {
    if (!(canvas instanceof HTMLCanvasElement)) {
      throw new Error('selected element is not a canvas');
    }
    const rect = canvas.getBoundingClientRect();
    return {
      rect: {
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
      },
      width: canvas.width,
      height: canvas.height,
    };
  });
  const screenshot = await page.screenshot({
    clip: geometry.rect,
    type: 'png',
  });
  const dataUrl = screenshot.toString('base64');
  return page.evaluate(async ({ dataUrl, width, height }) => {
    const image = new Image();
    image.src = 'data:image/png;base64,' + dataUrl;
    await image.decode();
    const probe = document.createElement('canvas');
    probe.width = image.naturalWidth;
    probe.height = image.naturalHeight;
    const context = probe.getContext('2d');
    if (!context) {
      throw new Error('could not create screenshot compositor statistics context');
    }
    context.drawImage(image, 0, 0);
    const rgba = context.getImageData(0, 0, probe.width, probe.height).data;
    let opaqueBlack = 0;
    let nonOpaque = 0;
    let nonBlack = 0;
    let minX = probe.width;
    let minY = probe.height;
    let maxX = -1;
    let maxY = -1;
    for (let y = 0; y < probe.height; y++) {
      for (let x = 0; x < probe.width; x++) {
        const offset = (y * probe.width + x) * 4;
        const r = rgba[offset];
        const g = rgba[offset + 1];
        const b = rgba[offset + 2];
        const a = rgba[offset + 3];
        if (r === 0 && g === 0 && b === 0 && a === 255) {
          opaqueBlack++;
        }
        if (a !== 255) {
          nonOpaque++;
        }
        if (r !== 0 || g !== 0 || b !== 0) {
          nonBlack++;
          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          maxX = Math.max(maxX, x);
          maxY = Math.max(maxY, y);
        }
      }
    }
    return {
      width: probe.width,
      height: probe.height,
      source_width: width,
      source_height: height,
      opaque_black_pixels: opaqueBlack,
      non_opaque_pixels: nonOpaque,
      non_black_pixels: nonBlack,
      non_black_fraction: (probe.width * probe.height) > 0
        ? nonBlack / (probe.width * probe.height)
        : 0,
      non_black_bounds: maxX >= 0
        ? { min_x: minX, min_y: minY, max_x: maxX, max_y: maxY }
        : null,
      screenshot_bytes: null,
    };
  }, {
    dataUrl,
    width: geometry.width,
    height: geometry.height,
  }).then(stats => ({
    ...stats,
    screenshot_bytes: screenshot.length,
  }));
}

async function screenshotCanvasPixelSamples(page, selector, points) {
  const geometry = await page.$eval(selector, canvas => {
    if (!(canvas instanceof HTMLCanvasElement)) {
      throw new Error('selected element is not a canvas');
    }
    const rect = canvas.getBoundingClientRect();
    return {
      rect: {
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
      },
      width: canvas.width,
      height: canvas.height,
    };
  });
  const screenshot = await page.screenshot({
    clip: geometry.rect,
    type: 'png',
  });
  const dataUrl = screenshot.toString('base64');
  const samples = await page.evaluate(async ({ dataUrl, points, width, height }) => {
    const image = new Image();
    image.src = 'data:image/png;base64,' + dataUrl;
    await image.decode();
    const probe = document.createElement('canvas');
    probe.width = image.naturalWidth;
    probe.height = image.naturalHeight;
    const context = probe.getContext('2d');
    if (!context) {
      throw new Error('could not create screenshot compositor probe context');
    }
    context.drawImage(image, 0, 0);
    const rgba = context.getImageData(0, 0, probe.width, probe.height).data;
    return {
      width: probe.width,
      height: probe.height,
      samples: points.map(({ name, x, y }) => {
        if (!Number.isInteger(x) || !Number.isInteger(y)
          || x < 0 || y < 0 || x >= width || y >= height) {
          throw new Error(`invalid compositor semantic probe ${name}: (${x},${y})`);
        }
        const sampleX = Math.min(
          probe.width - 1,
          Math.max(0, Math.floor((x + 0.5) * probe.width / width)),
        );
        const sampleY = Math.min(
          probe.height - 1,
          Math.max(0, Math.floor((y + 0.5) * probe.height / height)),
        );
        const offset = (sampleY * probe.width + sampleX) * 4;
        return {
          name,
          source_x: x,
          source_y: y,
          screenshot_x: sampleX,
          screenshot_y: sampleY,
          rgba: [...rgba.slice(offset, offset + 4)],
        };
      }),
    };
  }, { dataUrl, points, width: geometry.width, height: geometry.height });
  return {
    selector,
    canvas_width: geometry.width,
    canvas_height: geometry.height,
    screenshot_width: samples.width,
    screenshot_height: samples.height,
    screenshot_samples: samples.samples,
  };
}

function assertSemanticSceneSamples(
  samples,
  label = 'WebGPU',
  classification = 'renderer',
) {
  const byName = new Map(samples.map(sample => [sample.name, sample.rgba]));
  const background = byName.get('background');
  const polygon = byName.get('polygon');
  const transformedLine = byName.get('transformed-line');
  const circle = byName.get('circle');

  if (!background || !polygon || !transformedLine || !circle) {
    throw new QualificationError(
      `${label} semantic scene probes missing: ${JSON.stringify(samples)}`,
      classification,
    );
  }

  if (!(background[2] > background[0])) {
    throw new QualificationError(
      `${label} background probe has unexpected channel ordering: ${JSON.stringify(background)}`,
      classification,
    );
  }
  if (!(polygon[0] > polygon[2] + 40 && polygon[1] > polygon[2])) {
    throw new QualificationError(
      `${label} polygon probe does not preserve the fixture's warm fill: ${JSON.stringify(polygon)}`,
      classification,
    );
  }
  if (!(
    circle[2] > circle[0] + 40
    && circle[1] > circle[0]
  )) {
    throw new QualificationError(
      `${label} circle probe does not preserve the fixture's cool fill: ${JSON.stringify(circle)}`,
      classification,
    );
  }

  const lineBrightness = (transformedLine[0] + transformedLine[1] + transformedLine[2]) / 3;
  if (lineBrightness < 120 || transformedLine[3] === 0) {
    throw new QualificationError(
      `${label} transformed-line probe is unexpectedly dark: ${JSON.stringify(transformedLine)}`,
      classification,
    );
  }
}

function expectedWgpuSurfaceFormat(preferredFormat) {
  switch (preferredFormat) {
    case 'rgba8unorm':
      return 'Rgba8Unorm';
    case 'bgra8unorm':
      return 'Bgra8Unorm';
    default:
      throw new QualificationError(
        `unsupported browser preferred canvas format: ${preferredFormat}`,
        'capability',
      );
  }
}

function assertCanvasStatistics(statistics, width, height, label = 'WebGPU', classification = 'renderer') {
  const expectedPixels = width * height;
  if (!statistics
    || statistics.width !== width
    || statistics.height !== height
    || statistics.non_opaque_pixels !== 0
    || statistics.opaque_black_pixels > expectedPixels * 0.05
    || statistics.non_black_pixels < expectedPixels * 0.95
    || !statistics.non_black_bounds
    || statistics.non_black_bounds.min_x !== 0
    || statistics.non_black_bounds.min_y !== 0
    || statistics.non_black_bounds.max_x !== width - 1
    || statistics.non_black_bounds.max_y !== height - 1) {
    throw new QualificationError(
      `${label} full-frame paint statistics failed: ${JSON.stringify(statistics)}`,
      classification,
    );
  }
}

function assertCompositorMatchesCanvasBitmap(bitmapSamples, compositorSamples, label = 'WebGPU compositor', classification = 'renderer') {
  const bitmapByName = new Map(bitmapSamples.map(sample => [sample.name, sample.rgba]));
  const compositorByName = new Map(compositorSamples.map(sample => [sample.name, sample.rgba]));
  const mismatches = [];
  for (const [name, bitmap] of bitmapByName) {
    const compositor = compositorByName.get(name);
    if (!compositor || bitmap.length !== compositor.length) {
      mismatches.push({ name, bitmap, compositor, reason: 'missing-or-invalid-sample' });
      continue;
    }
    const maxChannelDelta = Math.max(
      ...bitmap.map((value, index) => Math.abs(value - compositor[index])),
    );
    if (maxChannelDelta > 12) {
      mismatches.push({ name, bitmap, compositor, max_channel_delta: maxChannelDelta });
    }
  }
  if (mismatches.length > 0) {
    throw new QualificationError(
      `${label} pixels diverge from the canvas bitmap: ${JSON.stringify(mismatches)}`,
      classification,
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


async function rawWebGpuExecutionCanary(page) {
  return page.evaluate(async () => {
    if (!navigator.gpu) {
      return { supported: false, reason: 'navigator.gpu unavailable' };
    }

    let device = null;
    let texture = null;
    let readback = null;
    const uncapturedErrors = [];
    let deviceLost = null;

    try {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
        forceFallbackAdapter: false,
      });
      if (!adapter) {
        return { supported: false, reason: 'raw execution canary requestAdapter returned null' };
      }

      device = await adapter.requestDevice();
      device.addEventListener('uncapturederror', event => {
        const error = event.error;
        uncapturedErrors.push({
          name: error?.name || 'UnknownGPUError',
          message: error?.message || String(error),
        });
      });
      void device.lost.then(info => {
        deviceLost = {
          reason: info?.reason || null,
          message: info?.message || null,
        };
        globalThis[statusKey] = deviceLost;
      });

      const width = 4;
      const height = 4;
      const bytesPerRow = 256;

      texture = device.createTexture({
        size: { width, height, depthOrArrayLayers: 1 },
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      readback = device.createBuffer({
        size: bytesPerRow * height,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

      device.pushErrorScope('validation');
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: texture.createView(),
          clearValue: { r: 1, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      pass.end();
      encoder.copyTextureToBuffer(
        { texture },
        { buffer: readback, bytesPerRow, rowsPerImage: height },
        { width, height, depthOrArrayLayers: 1 },
      );

      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      const validationError = await device.popErrorScope();
      await readback.mapAsync(GPUMapMode.READ);

      const bytes = new Uint8Array(readback.getMappedRange());
      const pixels = [];
      let nonRedPixels = 0;
      let minRed = 255;
      let maxGreen = 0;
      let maxBlue = 0;
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const offset = y * bytesPerRow + x * 4;
          const pixel = [...bytes.slice(offset, offset + 4)];
          pixels.push(pixel);
          minRed = Math.min(minRed, pixel[0]);
          maxGreen = Math.max(maxGreen, pixel[1]);
          maxBlue = Math.max(maxBlue, pixel[2]);
          if (!(pixel[0] > 240
            && pixel[1] < 16
            && pixel[2] < 16
            && pixel[3] === 255)) {
            nonRedPixels++;
          }
        }
      }
      const executedRed = nonRedPixels === 0;

      readback.unmap();

      return {
        supported: true,
        adapter_name: adapter.name || null,
        adapter_info: (() => {
          const info = adapter.info || device.adapterInfo || null;
          return info ? {
            vendor: info.vendor || null,
            architecture: info.architecture || null,
            device: info.device || null,
            description: info.description || null,
            is_fallback_adapter: info.isFallbackAdapter === true,
            subgroup_min_size: Number.isInteger(info.subgroupMinSize) ? info.subgroupMinSize : null,
            subgroup_max_size: Number.isInteger(info.subgroupMaxSize) ? info.subgroupMaxSize : null,
          } : null;
        })(),
        format: 'rgba8unorm',
        width,
        height,
        pixel_count: pixels.length,
        representative_pixel: pixels[Math.floor(pixels.length / 2)] || null,
        non_red_pixels: nonRedPixels,
        min_red: minRed,
        max_green: maxGreen,
        max_blue: maxBlue,
        executed_red: executedRed,
        validation_error: validationError ? {
          name: validationError.name || null,
          message: validationError.message || null,
        } : null,
        uncaptured_errors: uncapturedErrors,
        device_lost: deviceLost,
      };
    } catch (error) {
      return {
        supported: true,
        format: 'rgba8unorm',
        pixel: null,
        executed_red: false,
        uncaptured_errors: uncapturedErrors,
        device_lost: deviceLost,
        exception: error instanceof Error ? error.message : String(error),
      };
    } finally {
      readback?.destroy();
      texture?.destroy();
      device?.destroy();
    }
  });
}

async function screenshotRedStatistics(page, screenshotBuffer) {
  const base64 = screenshotBuffer.toString('base64');
  return page.evaluate(async dataUrl => {
    const image = new Image();
    image.src = 'data:image/png;base64,' + dataUrl;
    await image.decode();
    const canvas = document.createElement('canvas');
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext('2d');
    if (!context) {
      throw new Error('could not create screenshot probe context');
    }
    context.drawImage(image, 0, 0);
    const rgba = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let nonRedPixels = 0;
    let minRed = 255;
    let maxGreen = 0;
    let maxBlue = 0;
    let minAlpha = 255;
    for (let offset = 0; offset < rgba.length; offset += 4) {
      const r = rgba[offset];
      const g = rgba[offset + 1];
      const b = rgba[offset + 2];
      const a = rgba[offset + 3];
      minRed = Math.min(minRed, r);
      maxGreen = Math.max(maxGreen, g);
      maxBlue = Math.max(maxBlue, b);
      minAlpha = Math.min(minAlpha, a);
      if (!(r >= 245 && g <= 10 && b <= 10 && a >= 250)) {
        nonRedPixels++;
      }
    }
    return {
      width: canvas.width,
      height: canvas.height,
      pixel_count: canvas.width * canvas.height,
      non_red_pixels: nonRedPixels,
      min_red: minRed,
      max_green: maxGreen,
      max_blue: maxBlue,
      min_alpha: minAlpha,
    };
  }, base64);
}

async function screenshotPixelDelta(page, leftBuffer, rightBuffer) {
  const leftBase64 = leftBuffer.toString('base64');
  const rightBase64 = rightBuffer.toString('base64');
  return page.evaluate(async ({ leftData, rightData }) => {
    async function decode(data) {
      const image = new Image();
      image.src = 'data:image/png;base64,' + data;
      await image.decode();
      const canvas = document.createElement('canvas');
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const context = canvas.getContext('2d');
      if (!context) throw new Error('could not create screenshot comparison context');
      context.drawImage(image, 0, 0);
      return {
        width: canvas.width,
        height: canvas.height,
        pixels: context.getImageData(0, 0, canvas.width, canvas.height).data,
      };
    }

    const [left, right] = await Promise.all([decode(leftData), decode(rightData)]);
    if (left.width !== right.width || left.height !== right.height) {
      return {
        comparable: false,
        reason: 'screenshot dimensions differ',
        width: left.width,
        height: left.height,
        right_width: right.width,
        right_height: right.height,
      };
    }

    let differingPixels = 0;
    let maxChannelDelta = 0;
    for (let offset = 0; offset < left.pixels.length; offset += 4) {
      const delta = Math.max(
        Math.abs(left.pixels[offset] - right.pixels[offset]),
        Math.abs(left.pixels[offset + 1] - right.pixels[offset + 1]),
        Math.abs(left.pixels[offset + 2] - right.pixels[offset + 2]),
        Math.abs(left.pixels[offset + 3] - right.pixels[offset + 3]),
      );
      maxChannelDelta = Math.max(maxChannelDelta, delta);
      if (delta > 2) {
        differingPixels++;
      }
    }
    return {
      comparable: true,
      width: left.width,
      height: left.height,
      pixel_count: left.width * left.height,
      differing_pixels: differingPixels,
      max_channel_delta: maxChannelDelta,
    };
  }, { leftData: leftBase64, rightData: rightBase64 });
}

async function rawWebGpuCompositorCanary(page) {
  return page.evaluate(async () => {
    if (!navigator.gpu) {
      return { supported: false, reason: 'navigator.gpu unavailable' };
    }

    const webgpuCanvas = document.createElement('canvas');
    const controlCanvas = document.createElement('canvas');
    const cleanupKey = '__symthaeaWebGpuCompositorCanaryCleanup';
    const statusKey = '__symthaeaWebGpuCompositorCanaryStatus';
    let handoff = false;
    for (const canvas of [webgpuCanvas, controlCanvas]) {
      canvas.width = 64;
      canvas.height = 64;
      canvas.style.cssText = [
        'position:fixed',
        'top:0',
        'width:64px',
        'height:64px',
        'opacity:1',
        'pointer-events:none',
        'z-index:2147483647',
      ].join(';');
      document.body.appendChild(canvas);
    }
    webgpuCanvas.id = 'symthaea-webgpu-compositor-canary';
    controlCanvas.id = 'symthaea-webgpu-compositor-control';
    webgpuCanvas.style.left = '0';
    controlCanvas.style.left = '72px';

    let device = null;
    const uncapturedErrors = [];
    let deviceLost = null;
    try {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
        forceFallbackAdapter: false,
      });
      if (!adapter) {
        return { supported: false, reason: 'compositor canary requestAdapter returned null' };
      }
      device = await adapter.requestDevice();
      device.addEventListener('uncapturederror', event => {
        const error = event.error;
        uncapturedErrors.push({
          name: error?.name || 'UnknownGPUError',
          message: error?.message || String(error),
        });
      });
      void device.lost.then(info => {
        deviceLost = {
          reason: info?.reason || null,
          message: info?.message || null,
        };
      });

      const format = navigator.gpu.getPreferredCanvasFormat();
      const context = webgpuCanvas.getContext('webgpu');
      const controlContext = controlCanvas.getContext('2d');
      if (!context || !controlContext) {
        return {
          supported: false,
          reason: 'compositor canary context unavailable',
          format,
        };
      }

      controlContext.fillStyle = 'rgb(255, 0, 0)';
      controlContext.fillRect(0, 0, 64, 64);

      device.pushErrorScope('validation');
      context.configure({
        device,
        format,
        alphaMode: 'opaque',
      });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 1, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      pass.end();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      const validationError = await device.popErrorScope();
      await new Promise(requestAnimationFrame);
      await new Promise(requestAnimationFrame);

      globalThis[statusKey] = null;
      globalThis[cleanupKey] = () => {
        try {
          device?.destroy();
        } finally {
          webgpuCanvas.remove();
          controlCanvas.remove();
          delete globalThis[cleanupKey];
        }
      };
      handoff = true;

      return {
        supported: true,
        format,
        webgpu_rect: (() => {
          const rect = webgpuCanvas.getBoundingClientRect();
          return { x: rect.x, y: rect.y, width: rect.width, height: rect.height };
        })(),
        control_rect: (() => {
          const rect = controlCanvas.getBoundingClientRect();
          return { x: rect.x, y: rect.y, width: rect.width, height: rect.height };
        })(),
        validation_error: validationError ? {
          name: validationError.name || null,
          message: validationError.message || null,
        } : null,
        uncaptured_errors: uncapturedErrors,
        device_lost: deviceLost,
        cleanup_key: cleanupKey,
        status_key: statusKey,
      };
    } catch (error) {
      return {
        supported: true,
        format: null,
        webgpu_rect: null,
        control_rect: null,
        validation_error: null,
        uncaptured_errors: uncapturedErrors,
        device_lost: deviceLost,
        exception: error instanceof Error ? error.message : String(error),
      };
    } finally {
      if (!handoff) {
        device?.destroy();
        webgpuCanvas.remove();
        controlCanvas.remove();
      }
    }
  });
}
async function rawWebGpuCanvasCanary(page) {
  return page.evaluate(async () => {
    if (!navigator.gpu) {
      return { supported: false, reason: 'navigator.gpu unavailable' };
    }
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    canvas.style.cssText = [
      'position:fixed',
      'left:0',
      'top:0',
      'width:64px',
      'height:64px',
      'opacity:0.01',
      'pointer-events:none',
      'z-index:2147483647',
    ].join(';');
    document.body.appendChild(canvas);

    let device = null;
    const uncapturedErrors = [];
    try {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
        forceFallbackAdapter: false,
      });
      if (!adapter) {
        return { supported: false, reason: 'raw canary requestAdapter returned null' };
      }
      device = await adapter.requestDevice();
      device.addEventListener('uncapturederror', event => {
        const error = event.error;
        uncapturedErrors.push({
          name: error?.name || 'UnknownGPUError',
          message: error?.message || String(error),
        });
      });

      const format = navigator.gpu.getPreferredCanvasFormat();
      const context = canvas.getContext('webgpu');
      if (!context) {
        return {
          supported: false,
          reason: 'raw canary WebGPU context unavailable',
          format,
        };
      }

      context.configure({
        device,
        format,
        alphaMode: 'opaque',
      });

      device.pushErrorScope('validation');
      const encoder = device.createCommandEncoder();
      const view = context.getCurrentTexture().createView();
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view,
          clearValue: { r: 1, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      pass.end();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      const validationError = await device.popErrorScope();
      await new Promise(requestAnimationFrame);
      await new Promise(requestAnimationFrame);

      const dataUrl = canvas.toDataURL('image/png');
      const image = new Image();
      image.src = dataUrl;
      await image.decode();
      const probe = document.createElement('canvas');
      probe.width = canvas.width;
      probe.height = canvas.height;
      const probeContext = probe.getContext('2d');
      if (!probeContext) {
        return {
          supported: true,
          format,
          pixel: null,
          painted_red: false,
          uncaptured_errors: uncapturedErrors,
          reason: 'raw canary 2D probe context unavailable',
        };
      }
      probeContext.drawImage(image, 0, 0);
      const imageData = probeContext.getImageData(0, 0, canvas.width, canvas.height).data;
      const centerOffset = ((32 * canvas.width) + 32) * 4;
      const pixel = [...imageData.slice(centerOffset, centerOffset + 4)];
      let nonRedPixels = 0;
      let minRed = 255;
      let maxGreen = 0;
      let maxBlue = 0;
      for (let offset = 0; offset < imageData.length; offset += 4) {
        const r = imageData[offset];
        const g = imageData[offset + 1];
        const b = imageData[offset + 2];
        const a = imageData[offset + 3];
        minRed = Math.min(minRed, r);
        maxGreen = Math.max(maxGreen, g);
        maxBlue = Math.max(maxBlue, b);
        if (!(r > 200 && g < 40 && b < 40 && a === 255)) {
          nonRedPixels++;
        }
      }

      const configuration = context.getConfiguration();
      return {
        supported: true,
        format,
        configuration: {
          format: configuration?.format || null,
          usage: configuration?.usage || null,
          alphaMode: configuration?.alphaMode || null,
          colorSpace: configuration?.colorSpace || null,
          toneMapping: configuration?.toneMapping || null,
          viewFormats: configuration?.viewFormats ? [...configuration.viewFormats] : [],
          desiredMaximumFrameLatency: configuration?.desiredMaximumFrameLatency || null,
        },
        width: canvas.width,
        height: canvas.height,
        pixel_count: canvas.width * canvas.height,
        configuration_matches_preferred_format: configuration?.format === format,
        configuration_has_opaque_alpha: configuration?.alphaMode === 'opaque',
        pixel,
        non_red_pixels: nonRedPixels,
        min_red: minRed,
        max_green: maxGreen,
        max_blue: maxBlue,
        painted_red: nonRedPixels === 0,
        validation_error: validationError ? {
          name: validationError.name || null,
          message: validationError.message || null,
        } : null,
        uncaptured_errors: uncapturedErrors,
      };
    } catch (error) {
      return {
        supported: true,
        format: null,
        pixel: null,
        painted_red: false,
        uncaptured_errors: uncapturedErrors,
        exception: error instanceof Error ? error.message : String(error),
      };
    } finally {
      device?.destroy();
      canvas.remove();
    }
  });
}

async function capabilityPreflight(page) {
  return page.evaluate(async () => {
    if (!navigator.gpu) {
      return { navigator_gpu: false, adapter: false, device: false, reason: 'navigator.gpu unavailable' };
    }
    try {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
        forceFallbackAdapter: false,
      });
      if (!adapter) {
        return { navigator_gpu: true, adapter: false, device: false, reason: 'requestAdapter returned null' };
      }
      const device = await adapter.requestDevice();
      const features = [...adapter.features.values()].sort();
      const info = adapter.info || device.adapterInfo || null;
      const adapter_info = info ? {
        vendor: info.vendor || null,
        architecture: info.architecture || null,
        device: info.device || null,
        description: info.description || null,
        is_fallback_adapter: info.isFallbackAdapter === true,
        subgroup_min_size: Number.isInteger(info.subgroupMinSize) ? info.subgroupMinSize : null,
        subgroup_max_size: Number.isInteger(info.subgroupMaxSize) ? info.subgroupMaxSize : null,
      } : null;
      device.destroy();
      return {
        navigator_gpu: true,
        adapter: true,
        device: true,
        adapter_name: adapter.name || null,
        adapter_info,
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

async function waitForQualificationRecovery(page, selector) {
  try {
    await page.waitForFunction(
      selector => {
        const canvas = document.querySelector(selector);
        if (!canvas) return false;
        const initCount = Number.parseInt(
          canvas.getAttribute('data-qualification-init-count') || '',
          10,
        );
        return canvas.getAttribute('data-qualification-recovery-requested') === 'true'
          && canvas.getAttribute('data-qualification-loss-observed') === 'true'
          && canvas.getAttribute('data-qualification-loss-reason') === 'destroyed'
          && canvas.getAttribute('data-qualification-recovered') === 'true'
          && Number.isInteger(initCount)
          && initCount >= 2;
      },
      { timeout: 30_000 },
      selector,
    );
  } catch (error) {
    throw new QualificationError(
      `WebGPU renderer ${selector} did not complete the deterministic recovery proof (requested destroy + observed destroyed loss + second initialization): ${error instanceof Error ? error.message : String(error)}`,
      'renderer',
    );
  }
}

async function waitForQualificationReady(page, selector, probe) {
  const blankDataUrl = await page.evaluate(({ selector }) => {
    const canvas = document.querySelector(selector);
    if (!(canvas instanceof HTMLCanvasElement)) return null;
    const blank = document.createElement('canvas');
    blank.width = canvas.width;
    blank.height = canvas.height;
    const context = blank.getContext('2d');
    if (!context) return null;
    context.fillStyle = '#000';
    context.fillRect(0, 0, blank.width, blank.height);
    return blank.toDataURL('image/png');
  }, { selector });
  if (!blankDataUrl) {
    throw new QualificationError(
      'WebGPU projection ' + selector + ' canvas was not available for paint qualification',
      'renderer',
    );
  }
  try {
    await page.waitForFunction(
      ({ selector, blankDataUrl, probe }) => {
        const canvas = document.querySelector(selector);
        if (!(canvas instanceof HTMLCanvasElement)
          || canvas.getAttribute('data-qualification-ready') !== 'true') {
          return false;
        }
        if (probe.x < 0 || probe.y < 0 || probe.x >= canvas.width || probe.y >= canvas.height) {
          return false;
        }
        return canvas.toDataURL('image/png') !== blankDataUrl;
      },
      { timeout: 30_000 },
      { selector, blankDataUrl, probe },
    );
    await page.evaluate(() => new Promise(requestAnimationFrame));
    await page.evaluate(() => new Promise(requestAnimationFrame));
  } catch (error) {
    throw new QualificationError(
      'WebGPU projection ' + selector + ' did not reach an observable painted state at '
        + '(' + probe.x + ',' + probe.y + '): '
        + (error instanceof Error ? error.message : String(error)),
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

function runHarnessSelfTests() {
  const healthy = {
    width: 4,
    height: 4,
    opaque_black_pixels: 0,
    non_opaque_pixels: 0,
    non_black_pixels: 16,
    non_black_bounds: { min_x: 0, min_y: 0, max_x: 3, max_y: 3 },
  };
  assertCanvasStatistics(healthy, 4, 4, 'harness self-test', 'harness');

  const black = {
    width: 4,
    height: 4,
    opaque_black_pixels: 16,
    non_opaque_pixels: 0,
    non_black_pixels: 0,
    non_black_bounds: null,
  };
  let rejectedBlack = false;
  try {
    assertCanvasStatistics(black, 4, 4, 'harness self-test black', 'harness');
  } catch (error) {
    rejectedBlack = error instanceof QualificationError;
  }
  if (!rejectedBlack) {
    throw new Error('qualification harness self-test failed: black frame was accepted');
  }

  if (expectedWgpuSurfaceFormat('rgba8unorm') !== 'Rgba8Unorm'
    || expectedWgpuSurfaceFormat('bgra8unorm') !== 'Bgra8Unorm') {
    throw new Error('qualification harness self-test failed: browser/wgpu format mapping drift');
  }
}

runHarnessSelfTests();

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
    '--ozone-platform=x11',
  ];
  if (gpuMode) {
    args.push(
      '--enable-unsafe-webgpu',
      '--use-gpu-in-tests',
      '--enable-accelerated-2d-canvas',
      '--disable-dev-shm-usage',
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
    headless: HEADLESS,
    args,
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 1 });

  const pageConsoleMessages = [];
  page.on('console', message => {
    const type = message.type();
    if (type === 'error' || type === 'warning') {
      pageConsoleMessages.push({ type, text: message.text() });
    }
  });
  const diagnostics = {};
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
      diagnostics.raw_webgpu_execution_canary = await rawWebGpuExecutionCanary(page);
      diagnostics.raw_webgpu_canary = await rawWebGpuCanvasCanary(page);
      diagnostics.raw_webgpu_compositor_canary = await rawWebGpuCompositorCanary(page);
      if (diagnostics.raw_webgpu_compositor_canary?.supported
        && diagnostics.raw_webgpu_compositor_canary?.webgpu_rect
        && diagnostics.raw_webgpu_compositor_canary?.control_rect) {
        const { webgpu_rect: webgpuRect, control_rect: controlRect } =
          diagnostics.raw_webgpu_compositor_canary;
        const cleanupKey = diagnostics.raw_webgpu_compositor_canary.cleanup_key;
        try {
          const webgpuShot = await page.screenshot({
            clip: webgpuRect,
            type: 'png',
          });
          const controlShot = await page.screenshot({
            clip: controlRect,
            type: 'png',
          });
          diagnostics.raw_webgpu_compositor_canary.screenshot_bytes = webgpuShot.length;
          diagnostics.raw_webgpu_compositor_canary.control_screenshot_bytes = controlShot.length;
          diagnostics.raw_webgpu_compositor_canary.screenshot_hash =
            createHash('sha256').update(webgpuShot).digest('hex');
          diagnostics.raw_webgpu_compositor_canary.control_screenshot_hash =
            createHash('sha256').update(controlShot).digest('hex');
          diagnostics.raw_webgpu_compositor_canary.webgpu_screenshot_red =
            await screenshotRedStatistics(page, webgpuShot);
          diagnostics.raw_webgpu_compositor_canary.control_screenshot_red =
            await screenshotRedStatistics(page, controlShot);
          diagnostics.raw_webgpu_compositor_canary.screenshot_pixel_delta =
            await screenshotPixelDelta(page, webgpuShot, controlShot);
          diagnostics.raw_webgpu_compositor_canary.screenshot_matches_2d_control =
            diagnostics.raw_webgpu_compositor_canary.screenshot_pixel_delta.comparable === true
            && diagnostics.raw_webgpu_compositor_canary.screenshot_pixel_delta.differing_pixels === 0;
        } finally {
          if (cleanupKey) {
            const compositorLossAfterCapture = await page.evaluate(
              ({ cleanupKey, statusKey }) => {
                return {
                  loss: globalThis[statusKey] || null,
                  cleanup_present: typeof globalThis[cleanupKey] === 'function',
                };
              },
              { cleanupKey, statusKey },
            );
            if (compositorLossAfterCapture.loss?.reason) {
              throw new QualificationError(
                `Synthetic compositor canary device lost during evidence capture: ${JSON.stringify(compositorLossAfterCapture.loss)}`,
                'renderer',
              );
            }
            await page.evaluate(key => {
              globalThis[key]?.();
              delete globalThis[key];
            }, cleanupKey);
            await page.evaluate(statusKey => {
              delete globalThis[statusKey];
            }, statusKey);
          }
        }
      }
      const rawExecutionOk = diagnostics.raw_webgpu_execution_canary?.executed_red === true;
      const rawPresentationOk = diagnostics.raw_webgpu_canary?.painted_red === true;
      diagnostics.raw_webgpu_boundary = {
        execution_canary_passed: rawExecutionOk,
        presentation_canary_passed: rawPresentationOk,
        compositor_canary_passed:
          diagnostics.raw_webgpu_compositor_canary?.screenshot_matches_2d_control === true,
        implicated_boundary: !rawExecutionOk
          ? 'webgpu-device-execution-or-driver'
          : !rawPresentationOk
            ? 'canvas-presentation-or-browser-platform'
            : 'application-renderer-or-wgpu-path',
        interpretation_scope: 'diagnostic-only; qualification remains fail-closed',
      };
      failOnPageErrors('WebGPU capability preflight');
      if (!capability.navigator_gpu || !capability.adapter || !capability.device) {
        throw new QualificationError(
          `WebGPU capability preflight failed: ${JSON.stringify(capability)}`,
          'capability',
        );
      }
      if (hardwareMode && capability.adapter_info?.is_fallback_adapter === true) {
        throw new QualificationError(
          `WebGPU hardware qualification selected a fallback adapter: ${JSON.stringify(capability.adapter_info)}`,
          'capability',
        );
      }
      if (hardwareMode) {
        for (const [canvas, configuration] of Object.entries(diagnostics.surface_configuration)) {
          if (configuration.adapter_device_type === 'Cpu') {
            throw new QualificationError(
              `WebGPU hardware qualification renderer selected a software adapter for ${canvas}: ${JSON.stringify(configuration)}`,
              'capability',
            );
          }
        }
      }
      if (!diagnostics.raw_webgpu_execution_canary?.executed_red) {
        throw new QualificationError(
          `Raw WebGPU execution canary failed: ${JSON.stringify(diagnostics.raw_webgpu_execution_canary)}`,
          'capability',
        );
      }
      if (diagnostics.raw_webgpu_execution_canary?.validation_error
        || diagnostics.raw_webgpu_canary?.validation_error
        || diagnostics.raw_webgpu_compositor_canary?.validation_error
        || diagnostics.raw_webgpu_execution_canary?.uncaptured_errors?.length > 0
        || diagnostics.raw_webgpu_canary?.uncaptured_errors?.length > 0
        || diagnostics.raw_webgpu_compositor_canary?.uncaptured_errors?.length > 0) {
        throw new QualificationError(
          `Raw WebGPU validation error observed: ${JSON.stringify({
            execution_validation: diagnostics.raw_webgpu_execution_canary?.validation_error || null,
            presentation_validation: diagnostics.raw_webgpu_canary?.validation_error || null,
            compositor_validation: diagnostics.raw_webgpu_compositor_canary?.validation_error || null,
            execution_uncaptured: diagnostics.raw_webgpu_execution_canary?.uncaptured_errors || [],
            presentation_uncaptured: diagnostics.raw_webgpu_canary?.uncaptured_errors || [],
            compositor_uncaptured: diagnostics.raw_webgpu_compositor_canary?.uncaptured_errors || [],
          })}`,
          'capability',
        );
      }
      if (!diagnostics.raw_webgpu_canary?.configuration_matches_preferred_format
        || !diagnostics.raw_webgpu_canary?.configuration_has_opaque_alpha) {
        throw new QualificationError(
          `Raw WebGPU canvas configuration is not self-consistent: ${JSON.stringify(diagnostics.raw_webgpu_canary)}`,
          'capability',
        );
      }
      if (!diagnostics.raw_webgpu_canary?.painted_red) {
        throw new QualificationError(
          `Raw WebGPU canvas presentation canary failed: ${JSON.stringify(diagnostics.raw_webgpu_canary)}`,
          'renderer',
        );
      }
      const compositor = diagnostics.raw_webgpu_compositor_canary;
      const webgpuShotRed = compositor?.webgpu_screenshot_red;
      const controlShotRed = compositor?.control_screenshot_red;
      const screenshotDelta = compositor?.screenshot_pixel_delta;
      if (compositor?.screenshot_bytes <= 0
        || compositor?.control_screenshot_bytes <= 0
        || !webgpuShotRed
        || !controlShotRed
        || compositor?.device_lost?.reason
        || compositor?.validation_error
        || compositor?.uncaptured_errors?.length > 0
        || webgpuShotRed.non_red_pixels !== 0
        || controlShotRed.non_red_pixels !== 0
        || screenshotDelta?.comparable !== true
        || screenshotDelta?.differing_pixels !== 0
        || screenshotDelta?.max_channel_delta > 2) {
        throw new QualificationError(
          `Raw WebGPU compositor output does not match the 2D red control: ${JSON.stringify(diagnostics.raw_webgpu_compositor_canary)}`,
          'renderer',
        );
      }

      await waitForProjection(page, '#webgpu-cognitive-canvas', 'block');
      await waitForProjection(page, '#webgpu-movie-canvas', 'block');
      await waitForQualificationReady(page, '#webgpu-cognitive-canvas', { x: 10, y: 10 });
      await waitForQualificationReady(page, '#webgpu-movie-canvas', { x: 48, y: 96 });
      await waitForQualificationRecovery(page, '#webgpu-cognitive-canvas');
      await waitForQualificationRecovery(page, '#webgpu-movie-canvas');
      await waitForQualificationReady(page, '#webgpu-cognitive-canvas', { x: 10, y: 10 });
      await waitForQualificationReady(page, '#webgpu-movie-canvas', { x: 48, y: 96 });
      failOnPageErrors('WebGPU first render after recovery');

      const firstSceneHash = await canvasPngHash(page, '#webgpu-cognitive-canvas');
      const firstMovieHash = await canvasPngHash(page, '#webgpu-movie-canvas');
      const blankSceneHash = await blankCanvasHash(page, 512, 512);
      const blankMovieHash = await blankCanvasHash(page, 192, 192);
      diagnostics.scene = {
        scene_hash: firstSceneHash,
        blank_hash: blankSceneHash,
        pixel_statistics: await canvasPixelStatistics(page, '#webgpu-cognitive-canvas'),
      };
      diagnostics.movie = {
        movie_hash: firstMovieHash,
        blank_hash: blankMovieHash,
        pixel_statistics: await canvasPixelStatistics(page, '#webgpu-movie-canvas'),
      };
      assertCanvasStatistics(
        diagnostics.scene.pixel_statistics,
        512,
        512,
        'WebGPU cognitive scene',
        'renderer',
      );
      assertCanvasStatistics(
        diagnostics.movie.pixel_statistics,
        192,
        192,
        'WebGPU movie',
        'renderer',
      );
      diagnostics.browser_preferred_canvas_format =
        diagnostics.raw_webgpu_canary?.format || null;

      diagnostics.surface_configuration = await page.evaluate(() => {
        const read = selector => {
          const canvas = document.querySelector(selector);
          const format = canvas?.getAttribute('data-qualification-surface-format') || null;
          const formats = (canvas?.getAttribute('data-qualification-surface-formats') || '')
            .split(',')
            .map(value => value.trim())
            .filter(Boolean);
          const alphaMode = canvas?.getAttribute('data-qualification-alpha-mode') || null;
          const presentMode = canvas?.getAttribute('data-qualification-present-mode') || null;
          const adapterName = canvas?.getAttribute('data-qualification-adapter-name') || null;
          const adapterDeviceType =
            canvas?.getAttribute('data-qualification-adapter-device-type') || null;
          const adapterBackend = canvas?.getAttribute('data-qualification-adapter-backend') || null;
          const adapterVendor = canvas?.getAttribute('data-qualification-adapter-vendor') || null;
          const adapterDevice = canvas?.getAttribute('data-qualification-adapter-device') || null;
          const context = canvas instanceof HTMLCanvasElement
            ? canvas.getContext('webgpu')
            : null;
          const browserConfiguration =
            context && typeof context.getConfiguration === 'function'
              ? context.getConfiguration()
              : null;
          return {
            format,
            formats,
            alpha_mode: alphaMode,
            present_mode: presentMode,
            selected_format_advertised: !!format && formats.includes(format),
            selected_format_preferred: !!format && formats[0] === format,
            selected_alpha_advertised: !!alphaMode,
            selected_present_mode_advertised: !!presentMode,
            adapter_name: adapterName,
            adapter_device_type: adapterDeviceType,
            adapter_backend: adapterBackend,
            adapter_vendor: adapterVendor,
            adapter_device: adapterDevice,
            browser_configuration: browserConfiguration ? {
              format: browserConfiguration.format || null,
              usage: browserConfiguration.usage || null,
              alpha_mode: browserConfiguration.alphaMode || null,
              color_space: browserConfiguration.colorSpace || null,
              tone_mapping: browserConfiguration.toneMapping || null,
              view_formats: browserConfiguration.viewFormats
                ? [...browserConfiguration.viewFormats]
                : [],
              desired_maximum_frame_latency:
                browserConfiguration.desiredMaximumFrameLatency || null,
            } : null,
          };
        };
        return {
          cognitive: read('#webgpu-cognitive-canvas'),
          movie: read('#webgpu-movie-canvas'),
        };
      });
      const expectedRendererFormat =
        expectedWgpuSurfaceFormat(diagnostics.browser_preferred_canvas_format);
      diagnostics.surface_configuration_expected_format = expectedRendererFormat;
      for (const [canvas, configuration] of Object.entries(diagnostics.surface_configuration)) {
        if (!configuration.selected_format_advertised
          || !configuration.selected_format_preferred
          || !configuration.selected_alpha_advertised
          || !configuration.selected_present_mode_advertised
          || configuration.format !== expectedRendererFormat
          || configuration.browser_configuration?.format !== diagnostics.browser_preferred_canvas_format
          || configuration.browser_configuration?.alpha_mode !== 'opaque'
          || !configuration.adapter_name
          || !configuration.adapter_device_type) {
          throw new QualificationError(
            `WebGPU ${canvas} surface configuration is inconsistent with the browser-preferred format/capabilities: ${JSON.stringify({
              configuration,
              browser_preferred_canvas_format: diagnostics.browser_preferred_canvas_format,
              expected_renderer_format: expectedRendererFormat,
            })}`,
            'capability',
          );
        }
      }

      await page.screenshot({
        path: path.join(SCREENSHOT_DIR, `${mode}-preassert.png`),
        fullPage: false,
      });

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

      const compositorSceneStatistics = await screenshotCanvasPixelStatistics(
        page,
        '#webgpu-cognitive-canvas',
      );
      assertCanvasStatistics(
        compositorSceneStatistics,
        220,
        220,
        'WebGPU compositor scene',
        'renderer',
      );
      const compositorSceneSamples = await screenshotCanvasPixelSamples(
        page,
        '#webgpu-cognitive-canvas',
        [
          { name: 'background', x: 32, y: 32 },
          { name: 'polygon', x: 100, y: 100 },
          { name: 'transformed-line', x: 43, y: 371 },
          { name: 'circle', x: 360, y: 350 },
        ],
      );
      assertSemanticSceneSamples(
        compositorSceneSamples.screenshot_samples,
        'WebGPU compositor scene',
        'renderer',
      );
      const sceneBitmapSamples = await canvasPixelSamples(page, '#webgpu-cognitive-canvas', [
        { name: 'background', x: 32, y: 32 },
        { name: 'polygon', x: 100, y: 100 },
        { name: 'transformed-line', x: 43, y: 371 },
        { name: 'circle', x: 360, y: 350 },
      ]);
      assertCompositorMatchesCanvasBitmap(
        sceneBitmapSamples,
        compositorSceneSamples.screenshot_samples,
        'WebGPU compositor scene',
        'renderer',
      );

      const compositorMovieStatistics = await screenshotCanvasPixelStatistics(
        page,
        '#webgpu-movie-canvas',
      );
      assertCanvasStatistics(
        compositorMovieStatistics,
        192,
        192,
        'WebGPU compositor movie',
        'renderer',
      );
      const compositorMovieSamples = await screenshotCanvasPixelSamples(
        page,
        '#webgpu-movie-canvas',
        [
          { name: 'left', x: 48, y: 96 },
          { name: 'right', x: 144, y: 96 },
          { name: 'top', x: 96, y: 48 },
          { name: 'bottom', x: 96, y: 144 },
        ],
      );
      assertSemanticMovieSamples(
        compositorMovieSamples.screenshot_samples,
        'WebGPU compositor movie',
        'renderer',
      );
      const movieBitmapSamples = await canvasPixelSamples(page, '#webgpu-movie-canvas', [
        { name: 'left', x: 48, y: 96 },
        { name: 'right', x: 144, y: 96 },
        { name: 'top', x: 96, y: 48 },
        { name: 'bottom', x: 96, y: 144 },
      ]);
      assertCompositorMatchesCanvasBitmap(
        movieBitmapSamples,
        compositorMovieSamples.screenshot_samples,
        'WebGPU compositor movie',
        'renderer',
      );

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
      await waitForQualificationRecovery(page, '#webgpu-cognitive-canvas');
      await waitForQualificationRecovery(page, '#webgpu-movie-canvas');
      await waitForQualificationReady(page, '#webgpu-cognitive-canvas', { x: 10, y: 10 });
      await waitForQualificationReady(page, '#webgpu-movie-canvas', { x: 48, y: 96 });
      failOnPageErrors('WebGPU deterministic repeat render after recovery');

      const repeatSceneHash = await canvasPngHash(page, '#webgpu-cognitive-canvas');
      const repeatMovieHash = await canvasPngHash(page, '#webgpu-movie-canvas');
      const repeatCompositorSceneStatistics = await screenshotCanvasPixelStatistics(
        page,
        '#webgpu-cognitive-canvas',
      );
      assertCanvasStatistics(
        repeatCompositorSceneStatistics,
        220,
        220,
        'WebGPU compositor repeat scene',
        'renderer',
      );
      const repeatCompositorMovieStatistics = await screenshotCanvasPixelStatistics(
        page,
        '#webgpu-movie-canvas',
      );
      assertCanvasStatistics(
        repeatCompositorMovieStatistics,
        192,
        192,
        'WebGPU compositor repeat movie',
        'renderer',
      );
      const repeatCompositorSceneSamples = await screenshotCanvasPixelSamples(
        page,
        '#webgpu-cognitive-canvas',
        [
          { name: 'background', x: 32, y: 32 },
          { name: 'polygon', x: 100, y: 100 },
          { name: 'transformed-line', x: 43, y: 371 },
          { name: 'circle', x: 360, y: 350 },
        ],
      );
      const repeatSceneBitmapSamples = await canvasPixelSamples(page, '#webgpu-cognitive-canvas', [
        { name: 'background', x: 32, y: 32 },
        { name: 'polygon', x: 100, y: 100 },
        { name: 'transformed-line', x: 43, y: 371 },
        { name: 'circle', x: 360, y: 350 },
      ]);
      assertCompositorMatchesCanvasBitmap(
        repeatSceneBitmapSamples,
        repeatCompositorSceneSamples.screenshot_samples,
        'WebGPU compositor repeat scene',
        'renderer',
      );
      const repeatCompositorMovieSamples = await screenshotCanvasPixelSamples(
        page,
        '#webgpu-movie-canvas',
        [
          { name: 'left', x: 48, y: 96 },
          { name: 'right', x: 144, y: 96 },
          { name: 'top', x: 96, y: 48 },
          { name: 'bottom', x: 96, y: 144 },
        ],
      );
      const repeatMovieBitmapSamples = await canvasPixelSamples(page, '#webgpu-movie-canvas', [
        { name: 'left', x: 48, y: 96 },
        { name: 'right', x: 144, y: 96 },
        { name: 'top', x: 96, y: 48 },
        { name: 'bottom', x: 96, y: 144 },
      ]);
      assertCompositorMatchesCanvasBitmap(
        repeatMovieBitmapSamples,
        repeatCompositorMovieSamples.screenshot_samples,
        'WebGPU compositor repeat movie',
        'renderer',
      );
      assertSemanticSceneSamples(
        repeatCompositorSceneSamples.screenshot_samples,
        'WebGPU compositor repeat scene',
        'renderer',
      );
      assertSemanticMovieSamples(
        repeatCompositorMovieSamples.screenshot_samples,
        'WebGPU compositor repeat movie',
        'renderer',
      );

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
        raw_webgpu_execution_canary: diagnostics.raw_webgpu_execution_canary || null,
        raw_webgpu_canary: diagnostics.raw_webgpu_canary || null,
        raw_webgpu_compositor_canary: diagnostics.raw_webgpu_compositor_canary || null,
        raw_webgpu_boundary: diagnostics.raw_webgpu_boundary || null,
        scene_hash: firstSceneHash,
        movie_hash: firstMovieHash,
        semantic_scene_samples: semanticSceneSamples,
        semantic_movie_samples: semanticMovieSamples,
        compositor_scene_statistics: compositorSceneStatistics,
        compositor_scene_bitmap_samples: sceneBitmapSamples,
        compositor_scene_samples: compositorSceneSamples,
        compositor_movie_statistics: compositorMovieStatistics,
        compositor_movie_bitmap_samples: movieBitmapSamples,
        compositor_movie_samples: compositorMovieSamples,
        deterministic_repeat: true,
        page_errors: pageErrors,
        device_recovery: await page.evaluate(() => {
          const cognitive = document.querySelector('#webgpu-cognitive-canvas');
          const movie = document.querySelector('#webgpu-movie-canvas');
          return {
            cognitive_init_count: cognitive?.getAttribute('data-qualification-init-count') || null,
            movie_init_count: movie?.getAttribute('data-qualification-init-count') || null,
            cognitive_recovery_requested: cognitive?.getAttribute('data-qualification-recovery-requested') === 'true',
            movie_recovery_requested: movie?.getAttribute('data-qualification-recovery-requested') === 'true',
            cognitive_loss_observed: cognitive?.getAttribute('data-qualification-loss-observed') === 'true',
            movie_loss_observed: movie?.getAttribute('data-qualification-loss-observed') === 'true',
            cognitive_loss_reason: cognitive?.getAttribute('data-qualification-loss-reason') || null,
            movie_loss_reason: movie?.getAttribute('data-qualification-loss-reason') || null,
            cognitive_recovered: cognitive?.getAttribute('data-qualification-recovered') === 'true',
            movie_recovered: movie?.getAttribute('data-qualification-recovered') === 'true',
          };
        }),
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
  } catch (error) {
    if (error && typeof error === 'object') {
      error.page_console_messages = pageConsoleMessages;
      error.diagnostics = diagnostics;
    }
    throw error;
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
        page_console_messages: error?.page_console_messages || [],
        diagnostics: error?.diagnostics || {},
      };
    }
  }

  const artifact = {
    schema: 'symthaea-ui-webgpu-qualification-v12',
    harness_self_tests_passed: true,
    url: URL,
    chromium: CHROMIUM,
    headed_under_xvfb: HEADLESS === false,
    modes: MODES,
    workflow_sha: process.env.GITHUB_SHA || null,
    checked_out_sha: CHECKED_OUT_SHA,
    expected_pr_head_sha: EXPECTED_CHECKED_OUT_SHA,
    qualification_environment: QUALIFICATION_ENVIRONMENT,
    qualification_input_digests: QUALIFICATION_INPUT_DIGESTS,
    browser_execution_contract: {
      ozone_platform: 'x11',
      headed_under_xvfb: HEADLESS === false,
      viewport: { width: 1280, height: 900, device_scale_factor: 1 },
      adapter_request: {
        power_preference: 'high-performance',
        force_fallback_adapter: false,
        hardware_mode_rejects_fallback_adapter: true,
      },
    },
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
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
        forceFallbackAdapter: false,
      });
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
      await waitForQualificationReady(page, '#webgpu-cognitive-canvas', { x: 10, y: 10 });
      await waitForQualificationReady(page, '#webgpu-movie-canvas', { x: 48, y: 96 });
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
      await waitForQualificationReady(page, '#webgpu-cognitive-canvas', { x: 10, y: 10 });
      await waitForQualificationReady(page, '#webgpu-movie-canvas', { x: 48, y: 96 });
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
