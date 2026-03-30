#!/usr/bin/env node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Pre-deploy portal validation
// Catches: missing DOM IDs, undefined worker actions, broken tab references,
// missing function definitions, CSS class mismatches

const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, 'portal.html'), 'utf8');
const workerJs = fs.readFileSync(path.join(__dirname, 'spore-worker.js'), 'utf8');

let errors = 0;
let warnings = 0;

function error(msg) { console.error('ERROR:', msg); errors++; }
function warn(msg) { console.warn('WARN:', msg); warnings++; }
function pass(msg) { console.log('PASS:', msg); }

// 1. Extract all getElementById() calls from JS in portal.html
// Check each ID exists in the HTML
const getByIdCalls = [...html.matchAll(/getElementById\(['"]([^'"]+)['"]\)/g)];
const htmlIds = [...html.matchAll(/id=["']([^"']+)["']/g)].map(m => m[1]);
const idSet = new Set(htmlIds);

// IDs created dynamically at runtime (not in static HTML)
const dynamicIds = new Set([
  'flake-download-panel', 'ssh-deploy-panel', 'disk-selector',
  'download-panel', 'music-toggle', 'voice-toggle',
  'app-compat-panel', 'layout-select', 'install-hostname',
  'secure-boot-toggle', 'tpm2-toggle',
  'cfg-timezone', 'cfg-keyboard', 'cfg-desktop', 'cfg-gpu',
  'cfg-wifi-ssid', 'cfg-wifi-pass', 'cfg-hostname',
  'constellation-canvas',
]);

let missingIds = 0;
for (const [, id] of getByIdCalls) {
  if (!idSet.has(id) && !dynamicIds.has(id)) {
    error(`getElementById('${id}') references non-existent ID`);
    missingIds++;
  }
}
if (missingIds === 0) pass(`All ${getByIdCalls.length} getElementById() calls reference valid IDs`);

// 2. Extract all send() action names from portal.html
// Check each exists as a case in spore-worker.js
const sendCalls = [...html.matchAll(/send\(['"](\w+)['"]/g)].map(m => m[1]);
const workerCases = [...workerJs.matchAll(/case\s+['"](\w+)['"]/g)].map(m => m[1]);
const workerSet = new Set(workerCases);

let missingActions = 0;
for (const action of new Set(sendCalls)) {
  if (!workerSet.has(action)) {
    error(`send('${action}') has no handler in spore-worker.js`);
    missingActions++;
  }
}
if (missingActions === 0) pass(`All ${new Set(sendCalls).size} send() actions have worker handlers`);

// 3. Check tab buttons match tab-content panels
const tabButtons = [...html.matchAll(/data-tab=["'](\w+)["']/g)].map(m => m[1]);
const tabPanels = [...html.matchAll(/id=["']tab-(\w+)["']/g)].map(m => m[1]);
const panelSet = new Set(tabPanels);

let missingPanels = 0;
for (const tab of tabButtons) {
  if (!panelSet.has(tab)) {
    error(`Tab button data-tab="${tab}" has no matching panel id="tab-${tab}"`);
    missingPanels++;
  }
}
if (missingPanels === 0) pass(`All ${tabButtons.length} tab buttons have matching panels`);

// 4. Check for common JS issues in inline script
// Extract all JS from <script> tags
const scriptBlocks = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const allJs = scriptBlocks.join('\n');

// Check for undefined function calls (rough heuristic)
const funcDefs = [...allJs.matchAll(/function\s+(\w+)\s*\(/g)].map(m => m[1]);
const funcDefSet = new Set(funcDefs);

// Check addEventListener callbacks reference defined functions
const callbackRefs = [...allJs.matchAll(/addEventListener\([^,]+,\s*(\w+)\)/g)].map(m => m[1]);
for (const ref of callbackRefs) {
  if (!funcDefSet.has(ref) && ref !== 'function') {
    warn(`addEventListener references '${ref}' which may not be defined`);
  }
}

// 5. Check that all CSS classes used in JS exist in CSS
const jsClassRefs = [...allJs.matchAll(/classList\.(add|remove|toggle|contains)\(['"]([^'"]+)['"]\)/g)].map(m => m[2]);
const cssClasses = [...html.matchAll(/\.([a-zA-Z][\w-]*)\s*[{,]/g)].map(m => m[1]);
const cssClassSet = new Set(cssClasses);

let missingClasses = 0;
for (const cls of new Set(jsClassRefs)) {
  if (!cssClassSet.has(cls) && !['active', 'visible', 'open', 'running'].includes(cls)) {
    warn(`JS uses class '${cls}' not found in CSS (may be inherited or dynamic)`);
    missingClasses++;
  }
}

// 6. Check WASM export references match what pkg/symthaea_spore.js exports
const wasmJs = path.join(__dirname, 'pkg', 'symthaea_spore.js');
if (fs.existsSync(wasmJs)) {
  const wasmSource = fs.readFileSync(wasmJs, 'utf8');
  const workerMethodCalls = [...workerJs.matchAll(/engine\.(\w+)\(/g)].map(m => m[1]);

  let missingExports = 0;
  for (const method of new Set(workerMethodCalls)) {
    if (!wasmSource.includes(method)) {
      warn(`Worker calls engine.${method}() but it's not exported from WASM (may need feature rebuild)`);
      missingExports++;
    }
  }
  if (missingExports === 0) pass(`All ${new Set(workerMethodCalls).size} engine method calls exist in WASM exports`);
  else warn(`${missingExports} WASM methods missing — rebuild with all features if needed`);
} else {
  warn('pkg/symthaea_spore.js not found — skipping WASM export check (run build-wasm.sh first)');
}

// 7. Check HTML structure
const openDivs = (html.match(/<div/g) || []).length;
const closeDivs = (html.match(/<\/div>/g) || []).length;
if (openDivs !== closeDivs) {
  error(`Unbalanced divs: ${openDivs} opening vs ${closeDivs} closing`);
} else {
  pass(`HTML structure balanced (${openDivs} divs)`);
}

// Summary
console.log('\n' + '='.repeat(50));
console.log(`Results: ${errors} errors, ${warnings} warnings`);
if (errors > 0) {
  console.log('DEPLOY BLOCKED — fix errors first');
  process.exit(1);
} else {
  console.log('DEPLOY OK');
  process.exit(0);
}
