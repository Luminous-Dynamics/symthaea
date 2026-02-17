import * as dotenv from 'dotenv';
import { ethers } from 'ethers';
import { pool, initDatabase } from './index';
import { createClient } from 'redis';

dotenv.config();

async function main() {
  const results: Record<string, string> = {};
  const errors: string[] = [];

  // Env presence
  ['DATABASE_URL', 'RPC_URL', 'API_CHAIN_RPC_URL', 'ROUTER_ADDRESS'].forEach((key) => {
    if (!process.env[key]) {
      errors.push(`Missing ${key}`);
    }
  });
  if (!process.env.API_ADMIN_KEY) {
    results['api_admin_key'] = 'warning: not set';
  }

  // DB
  try {
    await initDatabase();
    await pool.query('SELECT 1');
    results.db = 'ok';
  } catch (e: any) {
    errors.push(`DB error: ${e?.message || e}`);
  }

  // Redis (optional)
  if (process.env.REDIS_URL) {
    const redis = createClient({ url: process.env.REDIS_URL });
    try {
      await redis.connect();
      const pong = await redis.ping();
      results.redis = pong === 'PONG' ? 'ok' : pong;
      await redis.disconnect();
    } catch (e: any) {
      errors.push(`Redis error: ${e?.message || e}`);
    }
  } else {
    results.redis = 'skipped (REDIS_URL not set)';
  }

  // RPC + Router
  try {
    const rpcUrl = process.env.API_CHAIN_RPC_URL || process.env.RPC_URL;
    const provider = new ethers.JsonRpcProvider(rpcUrl);
    const head = await provider.getBlockNumber();
    results.rpc = `ok (head ${head})`;

    const router = process.env.ROUTER_ADDRESS || '';
    const code = await provider.getCode(router);
    if (!code || code === '0x') {
      errors.push('Router address has no code on-chain');
    } else {
      results.router = 'ok (code present)';
    }
  } catch (e: any) {
    errors.push(`RPC error: ${e?.message || e}`);
  }

  // Output
  Object.entries(results).forEach(([k, v]) => console.log(`${k}: ${v}`));
  if (errors.length) {
    console.error('Errors:', errors.join(' | '));
    process.exitCode = 1;
  } else {
    console.log('Doctor: all checks passed');
  }
}

main().catch((e) => {
  console.error('Doctor failed', e);
  process.exit(1);
});
