#!/usr/bin/env node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

const fs = require('fs-extra');
const path = require('path');
const archiver = require('archiver');

const ROOT = path.join(__dirname, '..');
const DIST = path.join(ROOT, 'dist');

const COMMON_FILES = [
  'background.js',
  'content.js',
  'content.css',
  'popup.html',
  'popup.js',
  'options.html',
  'icons',
  'PRIVACY.md',
];

async function build(target) {
  const targetDir = path.join(DIST, target);

  console.log(`Building for ${target}...`);

  // Clean and create target directory
  await fs.remove(targetDir);
  await fs.ensureDir(targetDir);

  // Copy common files
  for (const file of COMMON_FILES) {
    const src = path.join(ROOT, file);
    const dest = path.join(targetDir, file);

    if (await fs.pathExists(src)) {
      await fs.copy(src, dest);
      console.log(`  Copied: ${file}`);
    } else {
      console.log(`  Warning: ${file} not found`);
    }
  }

  // Copy appropriate manifest
  const manifestSrc = target === 'firefox'
    ? path.join(ROOT, 'manifest.firefox.json')
    : path.join(ROOT, 'manifest.json');

  await fs.copy(manifestSrc, path.join(targetDir, 'manifest.json'));
  console.log(`  Copied: manifest.json (${target})`);

  // Create README
  const readme = `# Mycelix Knowledge - Fact Checker

This is the ${target === 'firefox' ? 'Firefox' : 'Chrome'} extension build.

For more information, see https://knowledge.mycelix.net/integrations/browser
`;
  await fs.writeFile(path.join(targetDir, 'README.txt'), readme);

  console.log(`Build complete: ${targetDir}`);
  return targetDir;
}

async function createZip(sourceDir, target) {
  const zipPath = path.join(DIST, `mycelix-knowledge-${target}.zip`);

  console.log(`Creating ${target} zip...`);

  return new Promise((resolve, reject) => {
    const output = fs.createWriteStream(zipPath);
    const archive = archiver('zip', { zlib: { level: 9 } });

    output.on('close', () => {
      console.log(`  Created: ${zipPath} (${archive.pointer()} bytes)`);
      resolve(zipPath);
    });

    archive.on('error', reject);
    archive.pipe(output);
    archive.directory(sourceDir, false);
    archive.finalize();
  });
}

async function main() {
  const target = process.argv[2];

  if (!target || !['chrome', 'firefox', 'all'].includes(target)) {
    console.log('Usage: node build.js <chrome|firefox|all>');
    process.exit(1);
  }

  await fs.ensureDir(DIST);

  const targets = target === 'all' ? ['chrome', 'firefox'] : [target];

  for (const t of targets) {
    const dir = await build(t);
    await createZip(dir, t);
  }

  console.log('\nAll builds complete!');
  console.log(`Output: ${DIST}/`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
