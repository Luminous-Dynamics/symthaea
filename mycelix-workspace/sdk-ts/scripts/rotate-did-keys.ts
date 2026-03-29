// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { readFileSync } from 'fs';
import { AppWebsocket } from '@holochain/client';
import { DidClient } from '../src/clients/identity/did.js';
import type { VerificationMethod } from '../src/clients/identity/types.js';

type Args = {
  conductorUrl: string;
  roleName: string;
  oldKeyId: string;
  newAuthKey?: string;
  newKemKey?: string;
  dryRun: boolean;
};

function parseArgs(argv: string[]): Args {
  const args: Args = {
    conductorUrl: 'ws://localhost:8888',
    roleName: 'mycelix_identity',
    oldKeyId: '',
    dryRun: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = argv[i + 1];
    switch (arg) {
      case '--conductor':
        args.conductorUrl = next;
        i += 1;
        break;
      case '--role':
        args.roleName = next;
        i += 1;
        break;
      case '--old-key-id':
        args.oldKeyId = next;
        i += 1;
        break;
      case '--new-auth-key':
        args.newAuthKey = next;
        i += 1;
        break;
      case '--new-kem-key':
        args.newKemKey = next;
        i += 1;
        break;
      case '--dry-run':
        args.dryRun = true;
        break;
      case '--help':
        printUsage();
        process.exit(0);
      default:
        break;
    }
  }

  return args;
}

function printUsage(): void {
  console.log(`Usage:
  ts-node --esm scripts/rotate-did-keys.ts \\
    --old-key-id "#keys-1" \\
    --new-auth-key path/to/new-auth-key.json \\
    --new-kem-key path/to/new-kem-key.json \\
    [--conductor ws://localhost:8888] \\
    [--role mycelix_identity] \\
    [--dry-run]

Notes:
- new-auth-key/new-kem-key JSON files must match VerificationMethod shape.
- You can supply only one of --new-auth-key or --new-kem-key.
`);
}

function loadMethod(path: string): VerificationMethod {
  const raw = readFileSync(path, 'utf-8');
  return JSON.parse(raw) as VerificationMethod;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (!args.oldKeyId || (!args.newAuthKey && !args.newKemKey)) {
    printUsage();
    process.exit(1);
  }

  if (args.dryRun) {
    console.log('Dry run: would rotate keys with args:', args);
    process.exit(0);
  }

  const client = await AppWebsocket.connect({ url: args.conductorUrl });
  const did = new DidClient(client, { roleName: args.roleName });

  if (args.newAuthKey) {
    const method = loadMethod(args.newAuthKey);
    await did.rotateKey({
      old_key_id: args.oldKeyId,
      new_method: method,
    });
    console.log(`✅ Rotated signing key: ${args.oldKeyId} -> ${method.id}`);
  }

  if (args.newKemKey) {
    const method = loadMethod(args.newKemKey);
    await did.rotateKeyAgreement({
      old_key_id: args.oldKeyId,
      new_method: method,
    });
    console.log(`✅ Rotated KEM key: ${args.oldKeyId} -> ${method.id}`);
  }

  await client.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
