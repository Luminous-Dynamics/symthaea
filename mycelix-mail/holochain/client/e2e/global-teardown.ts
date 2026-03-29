// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Global Teardown - Cleans up Holochain conductor and test environment
 */

import { readFileSync, rmSync, existsSync } from 'fs';
import path from 'path';
import { execSync } from 'child_process';

async function globalTeardown() {
  console.log('🧹 Cleaning up test environment...');

  const testDataDir = path.join(__dirname, '.test-data');
  const setupInfoPath = path.join(testDataDir, 'setup-info.json');

  if (existsSync(setupInfoPath)) {
    const setupInfo = JSON.parse(readFileSync(setupInfoPath, 'utf-8'));

    // Kill conductor process if we started it
    if (!setupInfo.useExistingConductor && setupInfo.conductorPid) {
      console.log(`📡 Stopping Holochain conductor (PID: ${setupInfo.conductorPid})...`);

      try {
        // Kill the conductor process and all child processes
        if (process.platform === 'win32') {
          execSync(`taskkill /PID ${setupInfo.conductorPid} /T /F`, { stdio: 'ignore' });
        } else {
          execSync(`kill -TERM -${setupInfo.conductorPid} 2>/dev/null || kill -TERM ${setupInfo.conductorPid}`, { stdio: 'ignore' });
        }
        console.log('✅ Conductor stopped');
      } catch (error) {
        // Process may have already exited
        console.log('⚠️ Conductor process already terminated');
      }
    }

    // Clean up Holochain sandbox data
    try {
      console.log('🗑️ Cleaning up sandbox data...');
      execSync('hc sandbox clean', {
        cwd: path.join(__dirname, '..', '..'),
        stdio: 'ignore',
      });
    } catch (error) {
      // Ignore cleanup errors
    }
  }

  // Remove test data directory
  if (existsSync(testDataDir)) {
    rmSync(testDataDir, { recursive: true, force: true });
  }

  console.log('✅ Cleanup complete');
}

export default globalTeardown;
