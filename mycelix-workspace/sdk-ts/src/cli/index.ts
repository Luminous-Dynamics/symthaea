#!/usr/bin/env node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk CLI
 *
 * Developer tools for building Mycelix hApps.
 *
 * Usage:
 *   npx @mycelix/sdk init <name>       - Initialize a new Mycelix hApp
 *   npx @mycelix/sdk generate <type>   - Generate code (bridge, zome, integration)
 *   npx @mycelix/sdk validate          - Validate project structure
 *   npx @mycelix/sdk test              - Run hApp tests with Tryorama
 *   npx @mycelix/sdk status            - Check conductor status
 *
 * @packageDocumentation
 * @module cli
 */

import * as fs from 'fs';
import * as path from 'path';

// ============================================================================
// Types
// ============================================================================

interface CliCommand {
  name: string;
  description: string;
  usage: string;
  options?: Array<{ flag: string; description: string }>;
  action: (args: string[], options: Record<string, string | boolean>) => Promise<void>;
}

interface ProjectConfig {
  name: string;
  version: string;
  happ: {
    name: string;
    roles: Array<{
      name: string;
      dna: string;
    }>;
  };
  integrations: string[];
}

// ============================================================================
// ANSI Color helpers
// ============================================================================

const colors = {
  reset: '\x1b[0m',
  bold: '\x1b[1m',
  dim: '\x1b[2m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
};

function colorize(text: string, color: keyof typeof colors): string {
  return `${colors[color]}${text}${colors.reset}`;
}

function success(msg: string): void {
  console.log(`${colorize('✓', 'green')} ${msg}`);
}

function info(msg: string): void {
  console.log(`${colorize('ℹ', 'blue')} ${msg}`);
}

function _warn(msg: string): void {
  console.log(`${colorize('⚠', 'yellow')} ${msg}`);
}

// Exported for use by consumers of the CLI module
export { _warn as warn };

function error(msg: string): void {
  console.error(`${colorize('✗', 'red')} ${msg}`);
}

// ============================================================================
// Templates
// ============================================================================

const HAPP_YAML_TEMPLATE = (name: string) => `manifest_version: "1"
name: ${name}
description: A Mycelix-powered Holochain application
roles:
  - name: ${name}
    provisioning:
      strategy: create
      deferred: false
    dna:
      bundled: "./dna/${name}.dna"
`;

const DNA_YAML_TEMPLATE = (name: string) => `manifest_version: "1"
name: ${name}
integrity:
  origin_time: 2024-01-01T00:00:00.000000Z
  network_seed: ~
  properties: ~
  zomes:
    - name: ${name}_integrity
      bundled: "./target/wasm32-unknown-unknown/release/${name}_integrity.wasm"
coordinator:
  zomes:
    - name: ${name}_coordinator
      bundled: "./target/wasm32-unknown-unknown/release/${name}_coordinator.wasm"
      dependencies:
        - name: ${name}_integrity
    - name: bridge
      bundled: "./target/wasm32-unknown-unknown/release/bridge.wasm"
      dependencies:
        - name: ${name}_integrity
`;

const CARGO_TOML_TEMPLATE = (name: string) => `[package]
name = "${name}"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
hdi = "0.5"
hdk = "0.4"
serde = { version = "1", features = ["derive"] }
serde_json = "1"

[profile.release]
opt-level = "z"
lto = true
`;

const INTEGRITY_LIB_TEMPLATE = (name: string) => `use hdi::prelude::*;

/// Entry types for ${name}
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    // Add your entry types here
    // Example:
    // #[entry_type]
    // MyEntry(MyEntry),
}

/// Link types for ${name}
#[hdk_link_types]
pub enum LinkTypes {
    // Add your link types here
}
`;

const COORDINATOR_LIB_TEMPLATE = (name: string) => `use hdk::prelude::*;
use ${name}_integrity::*;

/// Get all entries of a specific type
#[hdk_extern]
pub fn init(_: ()) -> ExternResult<InitCallbackResult> {
    Ok(InitCallbackResult::Pass)
}

// Add your zome functions here
// Example:
// #[hdk_extern]
// pub fn create_entry(input: CreateInput) -> ExternResult<ActionHash> {
//     create_entry(EntryTypes::MyEntry(input.into()))
// }
`;

const BRIDGE_TEMPLATE = `use hdk::prelude::*;

/// Register this hApp with the Mycelix bridge network
#[hdk_extern]
pub fn register_with_bridge(happ_name: String) -> ExternResult<()> {
    // Bridge registration logic
    Ok(())
}

/// Query cross-hApp reputation
#[hdk_extern]
pub fn get_cross_happ_reputation(agent: AgentPubKey) -> ExternResult<f64> {
    // Reputation query logic
    Ok(0.5)
}
`;

const PACKAGE_JSON_TEMPLATE = (name: string) => `{
  "name": "${name}",
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "start": "hc sandbox run",
    "test": "vitest run",
    "test:e2e": "npx @mycelix/sdk test",
    "build": "cargo build --release --target wasm32-unknown-unknown",
    "package": "hc app pack ./workdir"
  },
  "dependencies": {
    "@holochain/client": "^0.20.0",
    "@mycelix/sdk": "^0.6.0"
  },
  "devDependencies": {
    "@holochain/tryorama": "^0.18.0",
    "typescript": "^5.3.0",
    "vitest": "^1.2.0"
  }
}
`;

const TSCONFIG_TEMPLATE = `{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "declaration": true,
    "outDir": "./dist"
  },
  "include": ["src/**/*", "tests/**/*"]
}
`;

const README_TEMPLATE = (name: string) => `# ${name}

A Mycelix-powered Holochain application.

## Quick Start

\`\`\`bash
# Install dependencies
npm install

# Build the hApp
npm run build

# Run tests
npm test

# Start development sandbox
npm start
\`\`\`

## Project Structure

\`\`\`
${name}/
├── dna/                  # Holochain DNA definitions
│   └── ${name}.yaml      # DNA manifest
├── zomes/                # Zome source code
│   ├── integrity/        # Integrity zome
│   └── coordinator/      # Coordinator zome
├── ui/                   # Frontend application
├── tests/                # E2E tests with Tryorama
└── package.json
\`\`\`

## Mycelix Integration

This hApp integrates with the Mycelix ecosystem via:

- **MATL**: Mycelix Adaptive Trust Layer for reputation
- **Bridge**: Cross-hApp communication protocol
- **Epistemic**: Truth classification for claims

See [@mycelix/sdk documentation](https://github.com/Luminous-Dynamics/mycelix) for more.
`;

const TEST_TEMPLATE = (name: string) => `import { describe, it, expect } from 'vitest';
import { createClient, MockMycelixClient } from '@mycelix/sdk';

describe('${name} hApp', () => {
  it('should connect to conductor', async () => {
    const client = new MockMycelixClient();
    await client.connect();
    expect(client.isConnected()).toBe(true);
  });

  it('should register with bridge', async () => {
    const client = new MockMycelixClient();
    await client.connect();

    const result = await client.registerHapp({
      happ_id: '${name}',
      happ_name: '${name}',
      capabilities: ['identity_query', 'reputation_report'],
    });

    expect(result).toBeDefined();
  });
});
`;

const GITIGNORE_TEMPLATE = `# Rust
target/
*.wasm

# Node
node_modules/
dist/

# Holochain
.hc
*.happ
*.dna

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
`;

const CLIENT_TEMPLATE = (name: string, integrations: string[]) => `/**
 * ${name} Client
 *
 * TypeScript client for the ${name} hApp.
 */

import {
  createClient,
  createSignalManager,
  type MycelixClient,
  type BridgeSignalManager,
  ${integrations.map((i) => `${i}SignalHandler,`).join('\n  ')}
} from '@mycelix/sdk';

export class ${pascalCase(name)}Client {
  private client: MycelixClient;
  private signals: BridgeSignalManager;

  constructor() {
    this.client = createClient({ installedAppId: '${name}' });
    this.signals = createSignalManager();
  }

  async connect(): Promise<void> {
    await this.client.connect();
    this.signals.connect(this.client);
  }

  async disconnect(): Promise<void> {
    this.signals.disconnect();
    await this.client.disconnect();
  }

  // Add your client methods here
}

function pascalCase(str: string): string {
  return str
    .split('-')
    .map((s) => s.charAt(0).toUpperCase() + s.slice(1))
    .join('');
}
`;

// ============================================================================
// Utility Functions
// ============================================================================

function pascalCase(str: string): string {
  return str
    .split('-')
    .map((s) => s.charAt(0).toUpperCase() + s.slice(1))
    .join('');
}

function kebabCase(str: string): string {
  return str
    .replace(/([a-z])([A-Z])/g, '$1-$2')
    .replace(/[\s_]+/g, '-')
    .toLowerCase();
}

function ensureDir(dir: string): void {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

function writeFile(filePath: string, content: string): void {
  ensureDir(path.dirname(filePath));
  fs.writeFileSync(filePath, content.trim() + '\n');
}

// ============================================================================
// Codegen Types and Functions
// ============================================================================

interface ZomeFunction {
  name: string;
  inputType: string;
  outputType: string;
  docs: string[];
  zomeName: string;
}

interface ZomeType {
  name: string;
  kind: 'struct' | 'enum';
  fields: Array<{ name: string; type: string }>;
  variants?: string[];
  docs: string[];
}

interface ZomeDefinitions {
  functions: ZomeFunction[];
  types: ZomeType[];
}

/**
 * Find all Rust source files in a directory
 */
function findRustFiles(dir: string): string[] {
  const results: string[] = [];

  function walk(currentDir: string): void {
    const entries = fs.readdirSync(currentDir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(currentDir, entry.name);
      if (entry.isDirectory() && entry.name !== 'target' && entry.name !== '.git') {
        walk(fullPath);
      } else if (entry.isFile() && entry.name.endsWith('.rs')) {
        results.push(fullPath);
      }
    }
  }

  walk(dir);
  return results;
}

/**
 * Parse Rust files for zome definitions
 */
function parseZomeDefinitions(files: string[]): ZomeDefinitions {
  const functions: ZomeFunction[] = [];
  const types: ZomeType[] = [];

  for (const file of files) {
    const content = fs.readFileSync(file, 'utf8');
    const zomeName = extractZomeName(file);

    // Extract #[hdk_extern] functions
    // eslint-disable-next-line security/detect-unsafe-regex -- Parses trusted Rust source files, not user input
    const externRegex = /(?:\/\/\/[^\n]*\n)*\s*#\[hdk_extern\]\s*pub\s+(?:async\s+)?fn\s+(\w+)\s*\(([^)]*)\)\s*->\s*ExternResult<([^>]+)>/g;
    let match;

    while ((match = externRegex.exec(content)) !== null) {
      const [fullMatch, name, params, returnType] = match;

      // Extract docs
      // eslint-disable-next-line security/detect-unsafe-regex -- Parses trusted Rust source files, not user input
      const docsMatch = fullMatch.match(/(?:\/\/\/[^\n]*\n)+/);
      const docs = docsMatch
        ? docsMatch[0]
            .split('\n')
            .filter((l) => l.trim().startsWith('///'))
            .map((l) => l.replace(/^\s*\/\/\/\s?/, '').trim())
        : [];

      // Parse input type
      const inputType = params.trim()
        ? params.split(':').pop()?.trim().replace(/\)$/, '') || 'void'
        : 'void';

      functions.push({
        name,
        inputType: rustTypeToTs(inputType),
        outputType: rustTypeToTs(returnType.trim()),
        docs,
        zomeName,
      });
    }

    // Extract structs with #[hdk_entry_helper] or #[derive(Serialize)]
    // eslint-disable-next-line security/detect-unsafe-regex -- Parses trusted Rust source files, not user input
    const structRegex = /(?:\/\/\/[^\n]*\n)*\s*(?:#\[(?:hdk_entry_helper|derive\([^)]*Serialize[^)]*\))\]\s*)+pub\s+struct\s+(\w+)\s*\{([^}]+)\}/g;

    while ((match = structRegex.exec(content)) !== null) {
      const [fullMatch, name, fieldsStr] = match;

      // Extract docs
      // eslint-disable-next-line security/detect-unsafe-regex -- Parses trusted Rust source files, not user input
      const docsMatch = fullMatch.match(/(?:\/\/\/[^\n]*\n)+/);
      const docs = docsMatch
        ? docsMatch[0]
            .split('\n')
            .filter((l) => l.trim().startsWith('///'))
            .map((l) => l.replace(/^\s*\/\/\/\s?/, '').trim())
        : [];

      // Parse fields
      const fields = fieldsStr
        .split(',')
        .map((f) => f.trim())
        .filter((f) => f && !f.startsWith('//'))
        .map((f) => {
          const fieldMatch = f.match(/pub\s+(\w+)\s*:\s*(.+)/);
          if (fieldMatch) {
            return {
              name: fieldMatch[1],
              type: rustTypeToTs(fieldMatch[2].trim()),
            };
          }
          return null;
        })
        .filter((f): f is { name: string; type: string } => f !== null);

      types.push({
        name,
        kind: 'struct',
        fields,
        docs,
      });
    }

    // Extract enums with #[derive(Serialize)]
    // eslint-disable-next-line security/detect-unsafe-regex -- Parses trusted Rust source files, not user input
    const enumRegex = /(?:\/\/\/[^\n]*\n)*\s*#\[derive\([^)]*Serialize[^)]*\)\]\s*pub\s+enum\s+(\w+)\s*\{([^}]+)\}/g;

    while ((match = enumRegex.exec(content)) !== null) {
      const [fullMatch, name, variantsStr] = match;

      // Extract docs
      // eslint-disable-next-line security/detect-unsafe-regex -- Parses trusted Rust source files, not user input
      const docsMatch = fullMatch.match(/(?:\/\/\/[^\n]*\n)+/);
      const docs = docsMatch
        ? docsMatch[0]
            .split('\n')
            .filter((l) => l.trim().startsWith('///'))
            .map((l) => l.replace(/^\s*\/\/\/\s?/, '').trim())
        : [];

      // Parse variants
      const variants = variantsStr
        .split(',')
        .map((v) => v.trim())
        .filter((v) => v && !v.startsWith('//'))
        .map((v) => v.split('(')[0].split('{')[0].trim());

      types.push({
        name,
        kind: 'enum',
        fields: [],
        variants,
        docs,
      });
    }
  }

  return { functions, types };
}

/**
 * Extract zome name from file path
 */
function extractZomeName(filePath: string): string {
  const parts = filePath.split(path.sep);
  const zomesIndex = parts.findIndex((p) => p === 'zomes');
  if (zomesIndex >= 0 && parts[zomesIndex + 1]) {
    return parts[zomesIndex + 1];
  }
  return 'unknown';
}

/**
 * Convert Rust type to TypeScript type
 */
function rustTypeToTs(rustType: string): string {
  const typeMap: Record<string, string> = {
    'String': 'string',
    '&str': 'string',
    'str': 'string',
    'bool': 'boolean',
    'u8': 'number',
    'u16': 'number',
    'u32': 'number',
    'u64': 'number',
    'u128': 'bigint',
    'i8': 'number',
    'i16': 'number',
    'i32': 'number',
    'i64': 'number',
    'i128': 'bigint',
    'f32': 'number',
    'f64': 'number',
    'usize': 'number',
    'isize': 'number',
    '()': 'void',
    'AgentPubKey': 'AgentPubKey',
    'ActionHash': 'ActionHash',
    'EntryHash': 'EntryHash',
    'DnaHash': 'DnaHash',
    'Timestamp': 'number',
  };

  // Direct mapping
  if (typeMap[rustType]) {
    return typeMap[rustType];
  }

  // Option<T> -> T | null
  const optionMatch = rustType.match(/Option<(.+)>/);
  if (optionMatch) {
    return `${rustTypeToTs(optionMatch[1])} | null`;
  }

  // Vec<T> -> T[]
  const vecMatch = rustType.match(/Vec<(.+)>/);
  if (vecMatch) {
    return `${rustTypeToTs(vecMatch[1])}[]`;
  }

  // HashMap<K, V> -> Record<K, V>
  const mapMatch = rustType.match(/HashMap<([^,]+),\s*(.+)>/);
  if (mapMatch) {
    return `Record<${rustTypeToTs(mapMatch[1])}, ${rustTypeToTs(mapMatch[2])}>`;
  }

  // (T1, T2) -> [T1, T2]
  const tupleMatch = rustType.match(/\(([^)]+)\)/);
  if (tupleMatch) {
    const parts = tupleMatch[1].split(',').map((t) => rustTypeToTs(t.trim()));
    return `[${parts.join(', ')}]`;
  }

  // Result<T, E> -> T (we handle errors separately)
  const resultMatch = rustType.match(/Result<([^,]+)/);
  if (resultMatch) {
    return rustTypeToTs(resultMatch[1]);
  }

  // Keep custom types as-is
  return rustType;
}

/**
 * Generate TypeScript code from zome definitions
 */
function generateTypeScript(defs: ZomeDefinitions): string {
  const lines: string[] = [];

  // Header
  lines.push(`/**`);
  lines.push(` * Auto-generated TypeScript bindings for Mycelix zomes`);
  lines.push(` * Generated: ${new Date().toISOString()}`);
  lines.push(` * `);
  lines.push(` * DO NOT EDIT - regenerate with: npx @mycelix/sdk codegen`);
  lines.push(` * `);
  lines.push(` * @packageDocumentation`);
  lines.push(` * @module generated/zome-bindings`);
  lines.push(` */`);
  lines.push('');

  // Imports
  lines.push(`import type { ActionHash, AgentPubKey, EntryHash, DnaHash, Record as HolochainRecord } from '@holochain/client';`);
  lines.push(`import type { MycelixClient } from '@mycelix/sdk';`);
  lines.push('');

  // Generate types
  if (defs.types.length > 0) {
    lines.push('// ============================================================================');
    lines.push('// Types');
    lines.push('// ============================================================================');
    lines.push('');

    for (const type of defs.types) {
      // Add docs
      if (type.docs.length > 0) {
        lines.push('/**');
        type.docs.forEach((doc) => lines.push(` * ${doc}`));
        lines.push(' */');
      }

      if (type.kind === 'struct') {
        lines.push(`export interface ${type.name} {`);
        for (const field of type.fields) {
          lines.push(`  ${field.name}: ${field.type};`);
        }
        lines.push('}');
      } else if (type.kind === 'enum' && type.variants) {
        lines.push(`export type ${type.name} =`);
        lines.push(`  | ${type.variants.map((v) => `'${v}'`).join('\n  | ')};`);
      }
      lines.push('');
    }
  }

  // Generate zome client
  if (defs.functions.length > 0) {
    lines.push('// ============================================================================');
    lines.push('// Zome Client');
    lines.push('// ============================================================================');
    lines.push('');

    // Group functions by zome
    const functionsByZome = new Map<string, ZomeFunction[]>();
    for (const fn of defs.functions) {
      const existing = functionsByZome.get(fn.zomeName) || [];
      existing.push(fn);
      functionsByZome.set(fn.zomeName, existing);
    }

    // Generate client class for each zome
    for (const [zomeName, functions] of functionsByZome) {
      const className = `${pascalCase(zomeName)}ZomeClient`;

      lines.push(`/**`);
      lines.push(` * Type-safe client for the ${zomeName} zome`);
      lines.push(` */`);
      lines.push(`export class ${className} {`);
      lines.push(`  constructor(private client: MycelixClient, private zomeName: string = '${zomeName}') {}`);
      lines.push('');

      for (const fn of functions) {
        // Add docs
        if (fn.docs.length > 0) {
          lines.push('  /**');
          fn.docs.forEach((doc) => lines.push(`   * ${doc}`));
          lines.push('   */');
        }

        const inputParam = fn.inputType !== 'void' ? `input: ${fn.inputType}` : '';
        const payload = fn.inputType !== 'void' ? 'input' : 'null';

        lines.push(`  async ${fn.name}(${inputParam}): Promise<${fn.outputType}> {`);
        lines.push(`    return this.client.callZome({`);
        lines.push(`      zome_name: this.zomeName,`);
        lines.push(`      fn_name: '${fn.name}',`);
        lines.push(`      payload: ${payload},`);
        lines.push(`    });`);
        lines.push(`  }`);
        lines.push('');
      }

      lines.push('}');
      lines.push('');
    }

    // Generate factory function
    lines.push('/**');
    lines.push(' * Create zome clients from a Mycelix client');
    lines.push(' */');
    lines.push('export function createZomeClients(client: MycelixClient): {');

    for (const [zomeName] of functionsByZome) {
      const varName = kebabCase(zomeName).replace(/-/g, '_');
      lines.push(`  ${varName}: ${pascalCase(zomeName)}ZomeClient;`);
    }

    lines.push('} {');
    lines.push('  return {');

    for (const [zomeName] of functionsByZome) {
      const varName = kebabCase(zomeName).replace(/-/g, '_');
      lines.push(`    ${varName}: new ${pascalCase(zomeName)}ZomeClient(client),`);
    }

    lines.push('  };');
    lines.push('}');
    lines.push('');

    // Export function names for convenience
    lines.push('/**');
    lines.push(' * All available zome function names');
    lines.push(' */');
    lines.push('export const ZOME_FUNCTIONS = {');
    for (const [zomeName, functions] of functionsByZome) {
      lines.push(`  ${kebabCase(zomeName).replace(/-/g, '_')}: [`);
      functions.forEach((fn) => lines.push(`    '${fn.name}',`));
      lines.push('  ] as const,');
    }
    lines.push('} as const;');
  }

  return lines.join('\n');
}

function parseArgs(argv: string[]): {
  command: string;
  args: string[];
  options: Record<string, string | boolean>;
} {
  const options: Record<string, string | boolean> = {};
  const args: string[] = [];
  let command = '';

  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];

    if (arg.startsWith('--')) {
      const [key, value] = arg.slice(2).split('=');
      options[key] = value ?? true;
    } else if (arg.startsWith('-')) {
      options[arg.slice(1)] = true;
    } else if (!command) {
      command = arg;
    } else {
      args.push(arg);
    }
  }

  return { command, args, options };
}

// ============================================================================
// Commands
// ============================================================================

const commands: Record<string, CliCommand> = {
  init: {
    name: 'init',
    description: 'Initialize a new Mycelix hApp project',
    usage: 'npx @mycelix/sdk init <name> [--with-ui] [--integrations=identity,finance]',
    options: [
      { flag: '--with-ui', description: 'Include frontend scaffolding' },
      {
        flag: '--integrations',
        description: 'Comma-separated list of integrations (identity,finance,energy,etc)',
      },
    ],
    action: async (args, options) => {
      const name = args[0];
      if (!name) {
        error('Project name is required');
        console.log('Usage: npx @mycelix/sdk init <name>');
        process.exit(1);
      }

      const projectDir = path.resolve(process.cwd(), kebabCase(name));

      if (fs.existsSync(projectDir)) {
        error(`Directory ${projectDir} already exists`);
        process.exit(1);
      }

      info(`Creating new Mycelix hApp: ${name}`);

      // Parse integrations
      const integrations = options.integrations
        ? String(options.integrations)
            .split(',')
            .map((s) => s.trim())
        : ['identity'];

      // Create project structure
      ensureDir(projectDir);
      ensureDir(path.join(projectDir, 'dna'));
      ensureDir(path.join(projectDir, 'zomes', 'integrity', 'src'));
      ensureDir(path.join(projectDir, 'zomes', 'coordinator', 'src'));
      ensureDir(path.join(projectDir, 'zomes', 'bridge', 'src'));
      ensureDir(path.join(projectDir, 'tests'));
      ensureDir(path.join(projectDir, 'src'));

      const snakeName = name.replace(/-/g, '_');

      // Write files
      writeFile(path.join(projectDir, 'happ.yaml'), HAPP_YAML_TEMPLATE(name));
      writeFile(path.join(projectDir, 'dna', `${name}.yaml`), DNA_YAML_TEMPLATE(snakeName));
      writeFile(
        path.join(projectDir, 'zomes', 'integrity', 'Cargo.toml'),
        CARGO_TOML_TEMPLATE(`${snakeName}_integrity`)
      );
      writeFile(
        path.join(projectDir, 'zomes', 'integrity', 'src', 'lib.rs'),
        INTEGRITY_LIB_TEMPLATE(snakeName)
      );
      writeFile(
        path.join(projectDir, 'zomes', 'coordinator', 'Cargo.toml'),
        CARGO_TOML_TEMPLATE(`${snakeName}_coordinator`)
      );
      writeFile(
        path.join(projectDir, 'zomes', 'coordinator', 'src', 'lib.rs'),
        COORDINATOR_LIB_TEMPLATE(snakeName)
      );
      writeFile(
        path.join(projectDir, 'zomes', 'bridge', 'Cargo.toml'),
        CARGO_TOML_TEMPLATE('bridge')
      );
      writeFile(path.join(projectDir, 'zomes', 'bridge', 'src', 'lib.rs'), BRIDGE_TEMPLATE);
      writeFile(path.join(projectDir, 'package.json'), PACKAGE_JSON_TEMPLATE(name));
      writeFile(path.join(projectDir, 'tsconfig.json'), TSCONFIG_TEMPLATE);
      writeFile(path.join(projectDir, 'README.md'), README_TEMPLATE(name));
      writeFile(path.join(projectDir, 'tests', `${name}.test.ts`), TEST_TEMPLATE(name));
      writeFile(path.join(projectDir, 'src', 'client.ts'), CLIENT_TEMPLATE(name, integrations));
      writeFile(path.join(projectDir, '.gitignore'), GITIGNORE_TEMPLATE);

      // Create mycelix.json config
      const config: ProjectConfig = {
        name,
        version: '0.1.0',
        happ: {
          name,
          roles: [{ name, dna: `./dna/${name}.dna` }],
        },
        integrations,
      };
      writeFile(path.join(projectDir, 'mycelix.json'), JSON.stringify(config, null, 2));

      success(`Created ${name} at ${projectDir}`);
      console.log('');
      info('Next steps:');
      console.log(`  cd ${kebabCase(name)}`);
      console.log('  npm install');
      console.log('  npm run build');
      console.log('  npm test');
    },
  },

  generate: {
    name: 'generate',
    description: 'Generate code (zome, bridge, integration)',
    usage: 'npx @mycelix/sdk generate <type> <name>',
    options: [{ flag: '--dry-run', description: 'Preview without creating files' }],
    action: async (args, options) => {
      const [type, name] = args;

      if (!type || !name) {
        error('Type and name are required');
        console.log('Usage: npx @mycelix/sdk generate <type> <name>');
        console.log('Types: zome, bridge, integration, signal');
        process.exit(1);
      }

      info(`Generating ${type}: ${name}`);

      switch (type) {
        case 'zome':
          if (!options['dry-run']) {
            ensureDir(path.join(process.cwd(), 'zomes', name, 'src'));
            writeFile(
              path.join(process.cwd(), 'zomes', name, 'Cargo.toml'),
              CARGO_TOML_TEMPLATE(name)
            );
            writeFile(
              path.join(process.cwd(), 'zomes', name, 'src', 'lib.rs'),
              COORDINATOR_LIB_TEMPLATE(name)
            );
          }
          success(`Generated zome: ${name}`);
          break;

        case 'bridge':
          if (!options['dry-run']) {
            ensureDir(path.join(process.cwd(), 'zomes', 'bridge', 'src'));
            writeFile(
              path.join(process.cwd(), 'zomes', 'bridge', 'Cargo.toml'),
              CARGO_TOML_TEMPLATE('bridge')
            );
            writeFile(
              path.join(process.cwd(), 'zomes', 'bridge', 'src', 'lib.rs'),
              BRIDGE_TEMPLATE
            );
          }
          success('Generated bridge zome');
          break;

        default:
          error(`Unknown type: ${type}`);
          console.log('Valid types: zome, bridge, integration, signal');
          process.exit(1);
      }
    },
  },

  validate: {
    name: 'validate',
    description: 'Validate Mycelix project structure',
    usage: 'npx @mycelix/sdk validate',
    action: async () => {
      info('Validating project structure...');

      const issues: string[] = [];

      // Check for mycelix.json
      if (!fs.existsSync('mycelix.json')) {
        issues.push('Missing mycelix.json configuration file');
      }

      // Check for happ.yaml
      if (!fs.existsSync('happ.yaml')) {
        issues.push('Missing happ.yaml manifest');
      }

      // Check for package.json
      if (!fs.existsSync('package.json')) {
        issues.push('Missing package.json');
      }

      // Check for zomes directory
      if (!fs.existsSync('zomes')) {
        issues.push('Missing zomes directory');
      }

      if (issues.length > 0) {
        error('Validation failed:');
        issues.forEach((issue) => console.log(`  - ${issue}`));
        process.exit(1);
      }

      success('Project structure is valid');
    },
  },

  status: {
    name: 'status',
    description: 'Check Holochain conductor status',
    usage: 'npx @mycelix/sdk status [--url=ws://localhost:8888]',
    options: [{ flag: '--url', description: 'Conductor WebSocket URL' }],
    action: async (_args, options) => {
      const url = (options.url as string) || 'ws://localhost:8888';

      info(`Checking conductor at ${url}...`);

      try {
        // Dynamic import to avoid issues when holochain isn't available
        const { AppWebsocket } = await import('@holochain/client');
        const client = await AppWebsocket.connect({ url: new URL(url) });

        success('Conductor is running');

        const appInfo = await client.appInfo();
        if (appInfo) {
          console.log(`  App ID: ${appInfo.installed_app_id}`);
          console.log(`  Status: ${JSON.stringify(appInfo.status)}`);
        }

        await client.client.close();
      } catch (e) {
        error(`Failed to connect to conductor at ${url}`);
        console.log('  Make sure the Holochain conductor is running');
        process.exit(1);
      }
    },
  },

  doctor: {
    name: 'doctor',
    description: 'Diagnose common issues with your Mycelix setup',
    usage: 'npx @mycelix/sdk doctor',
    action: async () => {
      console.log('');
      console.log(colorize('  Mycelix Doctor', 'bold'));
      console.log(colorize('  ══════════════', 'dim'));
      console.log('');

      const checks: Array<{ name: string; check: () => Promise<boolean>; fix?: string }> = [
        {
          name: 'Node.js version',
          check: async () => {
            const version = process.version;
            const major = parseInt(version.slice(1).split('.')[0], 10);
            if (major >= 20) {
              console.log(`  ${colorize('✓', 'green')} Node.js ${version}`);
              return true;
            }
            console.log(`  ${colorize('✗', 'red')} Node.js ${version} (requires v20+)`);
            return false;
          },
          fix: 'Install Node.js v20 or later: https://nodejs.org/',
        },
        {
          name: 'Project configuration',
          check: async () => {
            if (fs.existsSync('mycelix.json')) {
              console.log(`  ${colorize('✓', 'green')} mycelix.json found`);
              return true;
            }
            console.log(`  ${colorize('✗', 'red')} mycelix.json not found`);
            return false;
          },
          fix: 'Run "npx @mycelix/sdk init <name>" to create a project',
        },
        {
          name: 'Package dependencies',
          check: async () => {
            if (fs.existsSync('node_modules/@mycelix/sdk')) {
              console.log(`  ${colorize('✓', 'green')} @mycelix/sdk installed`);
              return true;
            }
            if (fs.existsSync('node_modules')) {
              console.log(`  ${colorize('⚠', 'yellow')} @mycelix/sdk not in node_modules`);
              return true; // Not critical
            }
            console.log(`  ${colorize('✗', 'red')} No node_modules found`);
            return false;
          },
          fix: 'Run "npm install" to install dependencies',
        },
        {
          name: 'Holochain client',
          check: async () => {
            try {
              await import('@holochain/client');
              console.log(`  ${colorize('✓', 'green')} @holochain/client available`);
              return true;
            } catch {
              console.log(`  ${colorize('⚠', 'yellow')} @holochain/client not available`);
              return true; // Not critical for all use cases
            }
          },
          fix: 'Run "npm install @holochain/client" if you need conductor access',
        },
        {
          name: 'Conductor connection',
          check: async () => {
            try {
              const { AppWebsocket } = await import('@holochain/client');
              const client = await AppWebsocket.connect({
                url: new URL('ws://localhost:8888'),
              });
              await client.client.close();
              console.log(`  ${colorize('✓', 'green')} Conductor running at localhost:8888`);
              return true;
            } catch {
              console.log(`  ${colorize('○', 'dim')} Conductor not running (optional)`);
              return true; // Not required
            }
          },
        },
        {
          name: 'Rust toolchain',
          check: async () => {
            try {
              const { execSync } = await import('child_process');
              const version = execSync('rustc --version 2>/dev/null', { encoding: 'utf8' }).trim();
              console.log(`  ${colorize('✓', 'green')} ${version}`);
              return true;
            } catch {
              console.log(
                `  ${colorize('○', 'dim')} Rust not installed (needed for zome development)`
              );
              return true; // Not required for SDK-only use
            }
          },
          fix: 'Install Rust: https://rustup.rs/',
        },
      ];

      let allPassed = true;
      const fixes: string[] = [];

      for (const { name, check, fix } of checks) {
        try {
          const passed = await check();
          if (!passed) {
            allPassed = false;
            if (fix) fixes.push(fix);
          }
        } catch (e) {
          console.log(`  ${colorize('?', 'yellow')} ${name}: check failed`);
        }
      }

      console.log('');

      if (allPassed) {
        success('All checks passed! Your environment is ready.');
      } else {
        _warn('Some checks failed. Suggested fixes:');
        fixes.forEach((fix, i) => console.log(`  ${i + 1}. ${fix}`));
      }

      console.log('');
    },
  },

  info: {
    name: 'info',
    description: 'Show information about current Mycelix project',
    usage: 'npx @mycelix/sdk info',
    action: async () => {
      console.log('');
      console.log(colorize('  Project Information', 'bold'));
      console.log(colorize('  ═══════════════════', 'dim'));
      console.log('');

      // Read mycelix.json
      if (fs.existsSync('mycelix.json')) {
        try {
          const config = JSON.parse(fs.readFileSync('mycelix.json', 'utf8'));
          console.log(`  ${colorize('Name:', 'cyan')}          ${config.name}`);
          console.log(`  ${colorize('Version:', 'cyan')}       ${config.version}`);
          console.log(`  ${colorize('hApp:', 'cyan')}          ${config.happ?.name || 'N/A'}`);
          console.log(
            `  ${colorize('Integrations:', 'cyan')}  ${config.integrations?.join(', ') || 'none'}`
          );
        } catch {
          error('Failed to parse mycelix.json');
        }
      } else {
        info('No mycelix.json found. Run "npx @mycelix/sdk init" to create a project.');
        return;
      }

      // Read package.json
      if (fs.existsSync('package.json')) {
        try {
          const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
          console.log('');
          console.log(colorize('  Dependencies', 'bold'));
          console.log(colorize('  ────────────', 'dim'));

          const deps = { ...pkg.dependencies, ...pkg.devDependencies };
          const mycelixDeps = Object.entries(deps).filter(
            ([name]) => name.includes('mycelix') || name.includes('holochain')
          );

          if (mycelixDeps.length > 0) {
            mycelixDeps.forEach(([name, version]) => {
              console.log(`  ${name}: ${String(version)}`);
            });
          } else {
            console.log('  No Mycelix/Holochain dependencies found');
          }
        } catch {
          // Ignore
        }
      }

      // Check zomes
      if (fs.existsSync('zomes')) {
        const zomes = fs
          .readdirSync('zomes')
          .filter((f) => fs.statSync(path.join('zomes', f)).isDirectory());
        console.log('');
        console.log(colorize('  Zomes', 'bold'));
        console.log(colorize('  ─────', 'dim'));
        zomes.forEach((zome) => {
          const hasCargoToml = fs.existsSync(path.join('zomes', zome, 'Cargo.toml'));
          const icon = hasCargoToml ? colorize('●', 'green') : colorize('○', 'dim');
          console.log(`  ${icon} ${zome}`);
        });
      }

      console.log('');
    },
  },

  test: {
    name: 'test',
    description: 'Run hApp tests with Tryorama',
    usage: 'npx @mycelix/sdk test [--watch] [--coverage]',
    options: [
      { flag: '--watch', description: 'Run in watch mode' },
      { flag: '--coverage', description: 'Generate coverage report' },
    ],
    action: async (_args, options) => {
      info('Running Mycelix hApp tests...');

      const { spawn } = await import('child_process');

      const vitestArgs = ['run'];
      if (options.watch) vitestArgs[0] = 'watch';
      if (options.coverage) vitestArgs.push('--coverage');

      const child = spawn('npx', ['vitest', ...vitestArgs], {
        stdio: 'inherit',
        shell: true,
      });

      child.on('exit', (code) => {
        process.exit(code ?? 0);
      });
    },
  },

  upgrade: {
    name: 'upgrade',
    description: 'Upgrade @mycelix/sdk to latest version',
    usage: 'npx @mycelix/sdk upgrade',
    action: async () => {
      info('Checking for updates...');

      try {
        const { execSync } = await import('child_process');

        // Get current version
        const currentPkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
        const currentVersion =
          currentPkg.dependencies?.['@mycelix/sdk'] ||
          currentPkg.devDependencies?.['@mycelix/sdk'] ||
          'unknown';

        // Get latest version
        const latestVersion = execSync('npm view @mycelix/sdk version 2>/dev/null', {
          encoding: 'utf8',
        }).trim();

        console.log(`  Current: ${currentVersion}`);
        console.log(`  Latest:  ${latestVersion}`);

        if (currentVersion.includes(latestVersion)) {
          success('Already on latest version!');
          return;
        }

        info('Upgrading...');
        execSync('npm install @mycelix/sdk@latest', { stdio: 'inherit' });
        success(`Upgraded to @mycelix/sdk@${latestVersion}`);
      } catch (e) {
        error('Failed to upgrade. Try running: npm install @mycelix/sdk@latest');
      }
    },
  },

  codegen: {
    name: 'codegen',
    description: 'Generate TypeScript bindings from Rust zomes',
    usage: 'npx @mycelix/sdk codegen [--input=zomes] [--output=src/generated] [--format=ts|json|dts]',
    options: [
      { flag: '--input', description: 'Path to zomes directory (default: zomes)' },
      { flag: '--output', description: 'Output path for generated files (default: src/generated/)' },
      { flag: '--format', description: 'Output format: ts, json, or dts (default: ts)' },
      { flag: '--dry-run', description: 'Preview without writing files' },
      { flag: '--verbose', description: 'Show detailed output including parsed definitions' },
    ],
    action: async (_args, options) => {
      const inputDir = (options.input as string) || 'zomes';
      const outputPath = (options.output as string) || 'src/generated';
      const format = (options.format as string) || 'ts';
      const dryRun = !!options['dry-run'];
      const verbose = !!options.verbose;

      // Validate format
      const validFormats = ['ts', 'json', 'dts'];
      if (!validFormats.includes(format)) {
        error(`Invalid format: ${format}. Valid formats: ${validFormats.join(', ')}`);
        process.exit(1);
      }

      info(`Scanning Rust zomes in ${inputDir}...`);

      if (!fs.existsSync(inputDir)) {
        error(`Zomes directory not found: ${inputDir}`);
        console.log('');
        console.log('  Suggestions:');
        console.log('    - Make sure you are in a Mycelix project directory');
        console.log('    - Use --input=<path> to specify a different zomes directory');
        console.log('    - Run "npx @mycelix/sdk init <name>" to create a new project');
        process.exit(1);
      }

      // Find all Rust source files
      const rustFiles = findRustFiles(inputDir);
      if (rustFiles.length === 0) {
        error('No Rust files found in zomes directory');
        console.log('');
        console.log('  Suggestions:');
        console.log('    - Check that your zomes have .rs source files');
        console.log('    - Ensure zomes are not in the "target" directory');
        process.exit(1);
      }

      info(`Found ${rustFiles.length} Rust file(s)`);

      if (verbose) {
        rustFiles.forEach((f) => console.log(`    - ${f}`));
      }

      // Parse zome definitions
      const zomeDefinitions = parseZomeDefinitions(rustFiles);

      if (zomeDefinitions.functions.length === 0 && zomeDefinitions.types.length === 0) {
        _warn('No #[hdk_extern] functions or serializable types found');
        console.log('');
        console.log('  To generate bindings, ensure your Rust code has:');
        console.log('    - #[hdk_extern] pub fn function_name(...) -> ExternResult<T>');
        console.log('    - #[hdk_entry_helper] pub struct MyEntry { ... }');
        console.log('    - #[derive(Serialize)] pub struct MyType { ... }');
        process.exit(0);
      }

      info(`Found ${zomeDefinitions.functions.length} extern function(s) and ${zomeDefinitions.types.length} type(s)`);

      if (verbose) {
        console.log('');
        console.log(colorize('  Functions:', 'cyan'));
        zomeDefinitions.functions.forEach((fn) => {
          console.log(`    - ${fn.zomeName}::${fn.name}(${fn.inputType}) -> ${fn.outputType}`);
        });
        console.log('');
        console.log(colorize('  Types:', 'cyan'));
        zomeDefinitions.types.forEach((t) => {
          if (t.kind === 'struct') {
            console.log(`    - ${t.name} { ${t.fields.map((f) => f.name).join(', ')} }`);
          } else {
            console.log(`    - ${t.name} = ${t.variants?.join(' | ')}`);
          }
        });
      }

      // Generate output based on format
      let generatedCode: string;
      let outputFileName: string;

      switch (format) {
        case 'json':
          generatedCode = JSON.stringify(
            {
              generated: new Date().toISOString(),
              generator: '@mycelix/sdk codegen',
              functions: zomeDefinitions.functions,
              types: zomeDefinitions.types,
            },
            null,
            2
          );
          outputFileName = 'zome-bindings.json';
          break;

        case 'dts':
          generatedCode = generateTypeScript(zomeDefinitions)
            .replace(/export class (\w+) \{[\s\S]*?\n\}/g, '')  // Remove class implementations
            .replace(/export function createZomeClients[\s\S]*?\n\}/g, '')  // Remove factory function
            .replace(/export const ZOME_FUNCTIONS[\s\S]*?as const;/g, '');  // Remove constants
          outputFileName = 'zome-bindings.d.ts';
          break;

        default: // ts
          generatedCode = generateTypeScript(zomeDefinitions);
          outputFileName = 'zome-bindings.ts';
      }

      if (dryRun) {
        console.log('');
        console.log(colorize(`Generated Code (dry-run, format=${format}):`, 'cyan'));
        console.log('');
        console.log(generatedCode);
        return;
      }

      // Determine if output is a file or directory
      const isOutputFile = outputPath.endsWith('.ts') || outputPath.endsWith('.json') || outputPath.endsWith('.d.ts');
      let finalOutputPath: string;

      if (isOutputFile) {
        ensureDir(path.dirname(outputPath));
        finalOutputPath = outputPath;
      } else {
        ensureDir(outputPath);
        finalOutputPath = path.join(outputPath, outputFileName);
      }

      writeFile(finalOutputPath, generatedCode);

      // Generate index.ts for TypeScript output (unless output is a single file)
      if (format === 'ts' && !isOutputFile) {
        const indexContent = `// Auto-generated by @mycelix/sdk codegen
// DO NOT EDIT - regenerate with: npx @mycelix/sdk codegen

export * from './zome-bindings.js';
`;
        writeFile(path.join(outputPath, 'index.ts'), indexContent);
      }

      // Summary
      success(`Generated bindings at ${finalOutputPath}`);
      console.log('');
      console.log(colorize('  Summary:', 'bold'));
      console.log(`    Format:     ${format}`);
      console.log(`    Functions:  ${zomeDefinitions.functions.length}`);
      console.log(`    Types:      ${zomeDefinitions.types.length}`);

      // Group by zome for better summary
      const zomeGroups = new Map<string, number>();
      zomeDefinitions.functions.forEach((fn) => {
        zomeGroups.set(fn.zomeName, (zomeGroups.get(fn.zomeName) || 0) + 1);
      });

      if (zomeGroups.size > 0) {
        console.log('    By Zome:');
        for (const [zome, count] of zomeGroups) {
          console.log(`      - ${zome}: ${count} functions`);
        }
      }

      console.log('');
      if (format === 'ts' && !isOutputFile) {
        info('Import in your code:');
        console.log(`    import { createZomeClients } from '${outputPath.replace(/^src\//, './')}/index.js';`);
      }
    },
  },

  help: {
    name: 'help',
    description: 'Show help information',
    usage: 'npx @mycelix/sdk help [command]',
    action: async (args) => {
      const cmdName = args[0];

      if (cmdName && commands[cmdName]) {
        const cmd = commands[cmdName];
        console.log('');
        console.log(colorize(`  ${cmd.name}`, 'bold'), '-', cmd.description);
        console.log('');
        console.log(`  Usage: ${cmd.usage}`);

        if (cmd.options && cmd.options.length > 0) {
          console.log('');
          console.log('  Options:');
          cmd.options.forEach((opt) => {
            console.log(`    ${opt.flag.padEnd(20)} ${opt.description}`);
          });
        }
        console.log('');
        return;
      }

      console.log('');
      console.log(colorize('  Mycelix SDK CLI', 'bold'));
      console.log('');
      console.log('  Developer tools for building Mycelix hApps');
      console.log('');
      console.log('  Commands:');
      console.log('');

      Object.values(commands).forEach((cmd) => {
        console.log(`    ${colorize(cmd.name.padEnd(12), 'cyan')} ${cmd.description}`);
      });

      console.log('');
      console.log('  Run "npx @mycelix/sdk help <command>" for detailed usage');
      console.log('');
    },
  },
};

// ============================================================================
// Main
// ============================================================================

async function main(): Promise<void> {
  const { command, args, options } = parseArgs(process.argv);

  // Show banner
  if (!options.quiet) {
    console.log('');
    console.log(colorize('  ╔═══════════════════════════════════╗', 'magenta'));
    console.log(
      colorize('  ║', 'magenta'),
      colorize('  Mycelix SDK v0.6.0', 'bold'),
      colorize('           ║', 'magenta')
    );
    console.log(
      colorize('  ║', 'magenta'),
      '  Build trusted hApps          ',
      colorize('║', 'magenta')
    );
    console.log(colorize('  ╚═══════════════════════════════════╝', 'magenta'));
    console.log('');
  }

  // Handle version flag
  if (options.version || options.v) {
    console.log('0.6.0');
    return;
  }

  // Default to help
  const cmdName = command || 'help';
  const cmd = commands[cmdName];

  if (!cmd) {
    error(`Unknown command: ${cmdName}`);
    console.log('Run "npx @mycelix/sdk help" to see available commands');
    process.exit(1);
  }

  try {
    await cmd.action(args, options);
  } catch (e) {
    error(e instanceof Error ? e.message : String(e));
    process.exit(1);
  }
}

main().catch((e) => {
  error(e instanceof Error ? e.message : String(e));
  process.exit(1);
});
