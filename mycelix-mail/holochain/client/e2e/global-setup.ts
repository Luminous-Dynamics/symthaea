// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Global Setup - Starts Holochain conductor and initializes test environment
 */

import { spawn, ChildProcess } from 'child_process';
import { writeFileSync, mkdirSync, existsSync } from 'fs';
import path from 'path';

let conductorProcess: ChildProcess | null = null;

async function globalSetup() {
  console.log('🔧 Setting up Holochain test environment...');

  // Create test data directory
  const testDataDir = path.join(__dirname, '.test-data');
  if (!existsSync(testDataDir)) {
    mkdirSync(testDataDir, { recursive: true });
  }

  // Check if we should use an existing conductor or start a new one
  const useExistingConductor = process.env.USE_EXISTING_CONDUCTOR === 'true';

  if (!useExistingConductor) {
    // Start Holochain conductor in sandbox mode
    console.log('📡 Starting Holochain conductor...');

    conductorProcess = spawn('hc', [
      'sandbox',
      'generate',
      '--num-sandboxes', '2',
      '--run', '8888',
      '--app-port', '8889',
    ], {
      cwd: path.join(__dirname, '..', '..'),
      env: {
        ...process.env,
        RUST_LOG: 'warn',
      },
      stdio: 'pipe',
    });

    // Wait for conductor to be ready
    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Conductor startup timeout'));
      }, 60000);

      conductorProcess!.stdout?.on('data', (data) => {
        const output = data.toString();
        console.log('[Conductor]', output);

        if (output.includes('Conductor ready') || output.includes('listening')) {
          clearTimeout(timeout);
          resolve();
        }
      });

      conductorProcess!.stderr?.on('data', (data) => {
        console.error('[Conductor Error]', data.toString());
      });

      conductorProcess!.on('error', (error) => {
        clearTimeout(timeout);
        reject(error);
      });
    });

    console.log('✅ Holochain conductor started');
  }

  // Store conductor process info for teardown
  const setupInfo = {
    conductorPid: conductorProcess?.pid,
    useExistingConductor,
    timestamp: Date.now(),
  };

  writeFileSync(
    path.join(testDataDir, 'setup-info.json'),
    JSON.stringify(setupInfo, null, 2)
  );

  console.log('✅ Test environment ready');
}

export default globalSetup;
