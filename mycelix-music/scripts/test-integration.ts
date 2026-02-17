#!/usr/bin/env tsx
/**
 * Basic integration checks for local dev
 * - Verifies API health endpoints
 * - Verifies Next.js rewrites work (when web running)
 * - Optionally checks a minimal songs query
 */

import * as dotenv from 'dotenv';
dotenv.config();

const apiBase = (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3100').replace(/\/$/, '');
const webBase = (process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000').replace(/\/$/, '');
const checkWeb = String(process.env.CHECK_WEB || 'false').toLowerCase() === 'true';

async function expectOk(name: string, url: string) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${name} failed: ${res.status}`);
  return res.json().catch(() => null);
}

async function main() {
  let failures = 0;
  try {
    console.log(`API health: ${apiBase}/health`);
    const h = await expectOk('API /health', `${apiBase}/health`);
    console.log(' ✓', h);
  } catch (e: any) {
    failures++; console.error(' ✗', e.message);
  }

  try {
    console.log(`API detailed health: ${apiBase}/health/details`);
    const d = await expectOk('API /health/details', `${apiBase}/health/details`);
    console.log(' ✓', d);
  } catch (e: any) {
    failures++; console.error(' ✗', e.message);
  }

  try {
    console.log(`API songs: ${apiBase}/api/songs?limit=1`);
    const s = await expectOk('API /api/songs', `${apiBase}/api/songs?limit=1`);
    console.log(' ✓ rows:', Array.isArray(s) ? s.length : 0);
  } catch (e: any) {
    failures++; console.error(' ✗', e.message);
  }

  if (checkWeb) {
    try {
      console.log(`Web rewrite health: ${webBase}/health/details`);
      const w = await expectOk('Web /health/details', `${webBase}/health/details`);
      console.log(' ✓', w);
    } catch (e: any) {
      failures++; console.error(' ✗', e.message);
    }
  } else {
    console.log('Skipping web health check (set CHECK_WEB=true to enable).');
  }

  if (failures > 0) {
    console.error(`\n❌ Integration checks completed with ${failures} failure(s)`);
    process.exit(1);
  } else {
    console.log('\n✅ Integration checks passed');
  }
}

main();
