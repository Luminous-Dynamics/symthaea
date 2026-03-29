#!/usr/bin/env tsx

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Generate TypeScript API Client from OpenAPI Spec
 *
 * Usage:
 *   npx tsx scripts/generate-api-client.ts
 *   npx tsx scripts/generate-api-client.ts --url http://localhost:3100/openapi.json
 *
 * Output: packages/api-client/src/generated/
 */

import { execSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

const API_URL = process.argv.includes('--url')
  ? process.argv[process.argv.indexOf('--url') + 1]
  : 'http://localhost:3100/openapi.json';

const OUTPUT_DIR = path.join(__dirname, '..', 'packages', 'api-client', 'src', 'generated');

async function main() {
  console.log(`Fetching OpenAPI spec from: ${API_URL}`);

  // Ensure output directory exists
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });

  // Fetch the OpenAPI spec
  const response = await fetch(API_URL);
  if (!response.ok) {
    throw new Error(`Failed to fetch OpenAPI spec: ${response.status} ${response.statusText}`);
  }

  const spec = await response.json();

  // Save the spec locally
  const specPath = path.join(OUTPUT_DIR, 'openapi.json');
  fs.writeFileSync(specPath, JSON.stringify(spec, null, 2));
  console.log(`Saved OpenAPI spec to: ${specPath}`);

  // Generate TypeScript client using openapi-typescript-codegen
  console.log('Generating TypeScript client...');

  try {
    execSync(
      `npx openapi-typescript-codegen --input ${specPath} --output ${OUTPUT_DIR} --client fetch --useOptions --useUnionTypes`,
      { stdio: 'inherit' }
    );
  } catch (error) {
    console.error('Failed to generate client. Make sure openapi-typescript-codegen is installed:');
    console.error('  npm install -D openapi-typescript-codegen');
    process.exit(1);
  }

  // Generate index.ts that re-exports everything
  const indexPath = path.join(OUTPUT_DIR, '..', 'index.ts');
  const indexContent = `// Auto-generated API client
// Generated at: ${new Date().toISOString()}

export * from './generated';
export { OpenAPI } from './generated/core/OpenAPI';

// Configure the API base URL
import { OpenAPI } from './generated/core/OpenAPI';

export function configureApi(baseUrl: string, token?: string) {
  OpenAPI.BASE = baseUrl;
  if (token) {
    OpenAPI.TOKEN = token;
  }
}

// Export a pre-configured client for common use cases
export function createApiClient(baseUrl: string) {
  configureApi(baseUrl);
  return {
    // Re-export service methods here after generation
  };
}
`;

  fs.writeFileSync(indexPath, indexContent);
  console.log(`Generated index file: ${indexPath}`);

  console.log('\nAPI client generated successfully!');
  console.log(`Output directory: ${OUTPUT_DIR}`);
  console.log('\nUsage in your code:');
  console.log("  import { SongsService, configureApi } from '@mycelix/api-client';");
  console.log("  configureApi('https://api.mycelix.io');");
  console.log('  const songs = await SongsService.getSongs();');
}

main().catch((error) => {
  console.error('Error:', error);
  process.exit(1);
});
