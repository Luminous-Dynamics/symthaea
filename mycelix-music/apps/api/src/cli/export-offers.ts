// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import * as fs from 'fs';
import * as path from 'path';
import * as dotenv from 'dotenv';
import fetch from 'node-fetch';
import crypto from 'crypto';
import { ethers } from 'ethers';

dotenv.config();

type Split = { role: string; pct: number };
type OfferRule = { title: string; condition: string; action: string };

type ExportPayload = {
  modules: string[];
  pricing: {
    baseAmount: number;
    loyaltyMultiplier: number;
  };
  programmableOffers: OfferRule[];
  splits: Split[];
};

function validatePayload(payload: ExportPayload) {
  if (!Array.isArray(payload.modules)) throw new Error('modules must be an array');
  if (typeof payload.pricing?.baseAmount !== 'number') throw new Error('pricing.baseAmount required');
  if (typeof payload.pricing?.loyaltyMultiplier !== 'number') throw new Error('pricing.loyaltyMultiplier required');
  if (!Array.isArray(payload.programmableOffers)) throw new Error('programmableOffers must be an array');
  payload.programmableOffers.forEach((r, i) => {
    if (!r.title || !r.condition || !r.action) throw new Error(`rule ${i} missing fields`);
  });
  if (!Array.isArray(payload.splits)) throw new Error('splits must be an array');
}

function sha256(data: string) {
  return crypto.createHash('sha256').update(data).digest('hex');
}

async function pushConfig(output: any) {
  const api = process.env.API_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3100';
  const key = process.env.API_ADMIN_KEY;
  if (!key) {
    throw new Error('API_ADMIN_KEY is required to push');
  }
  const resp = await fetch(`${api}/api/strategy-configs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'x-api-key': key },
    body: JSON.stringify({
      name: output.modules.join('+'),
      payload: output,
      published: false,
      hash: output.hash,
      adminSignature: output.admin_signature || null,
    }),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Push failed: ${resp.status} ${text}`);
  }
}

async function main() {
  const input = process.argv[2];
  const push = process.argv.includes('--push');
  if (!input) {
    console.error('Usage: tsx export-offers.ts <path-to-json> [--push]');
    process.exit(1);
  }
  const filePath = path.resolve(process.cwd(), input);
  const raw = fs.readFileSync(filePath, 'utf8');
  const parsed = JSON.parse(raw) as ExportPayload;
  validatePayload(parsed);
  const output: {
    kind: string;
    version: number;
    modules: string[];
    pricing: { baseAmount: number; loyaltyMultiplier: number };
    offers: OfferRule[];
    splits: Split[];
    generatedAt: string;
    hash?: string;
    admin_signature?: string;
  } = {
    kind: 'mycelix.strategy.config',
    version: 1,
    modules: parsed.modules,
    pricing: parsed.pricing,
    offers: parsed.programmableOffers,
    splits: parsed.splits,
    generatedAt: new Date().toISOString(),
  };
  output.hash = sha256(JSON.stringify(output));
  const adminKey = process.env.ADMIN_SIGNER_KEY;
  if (adminKey) {
    const wallet = new ethers.Wallet(adminKey);
    output.admin_signature = await wallet.signMessage(output.hash);
  }

  if (push) {
    await pushConfig(output);
    console.log('Pushed strategy config to API');
  } else {
    process.stdout.write(JSON.stringify(output, null, 2));
  }
}

main().catch((e) => {
  console.error(e.message || e);
  process.exit(1);
});
