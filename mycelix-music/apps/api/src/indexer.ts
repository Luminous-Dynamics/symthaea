// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import * as dotenv from 'dotenv';
import { ethers } from 'ethers';
import { createServer } from 'http';
import client from 'prom-client';
import { pool, initDatabase } from './index';
import { recordPoison, setRetryQueueLength } from './metrics';

dotenv.config();

const RPC_URL = process.env.API_CHAIN_RPC_URL || process.env.RPC_URL || 'http://localhost:8545';
const ROUTER_ADDRESS = process.env.ROUTER_ADDRESS || process.env.NEXT_PUBLIC_ROUTER_ADDRESS || '';
const CHUNK_SIZE = parseInt(String(process.env.INDEXER_CHUNK_SIZE || '2000'), 10);
const START_BLOCK_ENV = process.env.INDEXER_START_BLOCK ? parseInt(String(process.env.INDEXER_START_BLOCK), 10) : null;
const METRICS_PORT = parseInt(String(process.env.INDEXER_METRICS_PORT || '9400'), 10);
const RETRY_LIMIT = parseInt(String(process.env.INDEXER_RETRY_LIMIT || '3'), 10);

if (!ROUTER_ADDRESS) {
  console.error('ROUTER_ADDRESS (or NEXT_PUBLIC_ROUTER_ADDRESS) is required for the indexer');
  process.exit(1);
}

if (!process.env.DATABASE_URL) {
  console.error('DATABASE_URL is required to run the indexer');
  process.exit(1);
}

const provider = new ethers.JsonRpcProvider(RPC_URL);

const routerInterface = new ethers.Interface([
  'event PaymentRecorded(bytes32 indexed songId, address indexed listener, uint256 grossAmount, uint256 protocolFee, uint256 netAmount, uint8 paymentType)',
]);

type PaymentEvt = {
  songId: string;
  listener: string;
  grossAmount: string;
  protocolFee: string;
  netAmount: string;
  paymentType: number;
  txHash: string;
  logIndex: number;
  blockNumber: number;
};

const paymentTypeMap: Record<number, string> = {
  0: 'stream',
  1: 'download',
  2: 'tip',
  3: 'patronage',
  4: 'nft_access',
};

// Metrics
const register = new client.Registry();
client.collectDefaultMetrics({ register });
const eventsCounter = new client.Counter({
  name: 'indexer_events_total',
  help: 'Total PaymentRecorded events processed',
});
const lagGauge = new client.Gauge({
  name: 'indexer_lag_blocks',
  help: 'Current lag in blocks vs chain head',
});
register.registerMetric(eventsCounter);
register.registerMetric(lagGauge);

// Retry queue
type RetryItem = PaymentEvt & { reason: string; attempts: number };
const retryQueue: RetryItem[] = [];
const poison: RetryItem[] = [];

async function enqueueRetry(evt: PaymentEvt, reason: string) {
  const existing = retryQueue.find((r) => r.txHash === evt.txHash && r.logIndex === evt.logIndex);
  if (existing) {
    existing.attempts += 1;
    existing.reason = reason;
  } else {
    retryQueue.push({ ...evt, reason, attempts: 1 });
  }
}

async function processRetries() {
  for (let i = retryQueue.length - 1; i >= 0; i--) {
    const item = retryQueue[i];
    try {
      await upsertEvent(item);
      retryQueue.splice(i, 1);
    } catch (e: any) {
      item.attempts += 1;
      item.reason = e?.message || 'unknown';
      if (item.attempts > RETRY_LIMIT) {
        await storePoison(item);
        retryQueue.splice(i, 1);
        recordPoison();
      }
    }
  }
  setRetryQueueLength(retryQueue.length);
}

async function storePoison(item: RetryItem) {
  try {
    await pool.query(
      `INSERT INTO indexer_poison (tx_hash, log_index, song_hash, reason, attempts, block_number)
       VALUES ($1, $2, $3, $4, $5, $6)
       ON CONFLICT (tx_hash, log_index) DO UPDATE SET reason = EXCLUDED.reason, attempts = EXCLUDED.attempts, block_number = EXCLUDED.block_number`,
      [item.txHash, item.logIndex, item.songId, item.reason, item.attempts, item.blockNumber],
    );
  } catch (e) {
    console.error('[indexer] failed to store poison item', e);
  }
}

async function ensureDeps() {
  await initDatabase();
}

async function getStartBlock(): Promise<number> {
  const fromState = await loadCheckpoint();
  if (fromState !== null) return fromState + 1;
  if (START_BLOCK_ENV !== null && !Number.isNaN(START_BLOCK_ENV)) return START_BLOCK_ENV;
  return provider.getBlockNumber();
}

async function loadCheckpoint(): Promise<number | null> {
  try {
    const rows = await pool.query('SELECT last_block FROM indexer_state WHERE name = $1 LIMIT 1', ['router']);
    if (rows.rows.length) {
      return Number(rows.rows[0].last_block);
    }
  } catch (e) {
    console.warn('[indexer] failed to load checkpoint from db', e);
  }
  return null;
}

async function storeCheckpoint(blockNumber: number) {
  try {
    await pool.query(
      `INSERT INTO indexer_state (name, last_block, updated_at)
       VALUES ($1, $2, NOW())
       ON CONFLICT (name) DO UPDATE SET last_block = EXCLUDED.last_block, updated_at = NOW()`,
      ['router', blockNumber],
    );
  } catch (e) {
    console.warn('[indexer] failed to persist checkpoint', e);
  }
}

async function fetchEvents(from: number, to: number): Promise<PaymentEvt[]> {
  // ethers v6: use getEvent().topicHash instead of getEventTopic()
  const eventFragment = routerInterface.getEvent('PaymentRecorded');
  const topic = eventFragment?.topicHash;
  if (!topic) {
    console.warn('[indexer] PaymentRecorded event not found in ABI');
    return [];
  }
  const logs = await provider.getLogs({
    address: ROUTER_ADDRESS,
    fromBlock: from,
    toBlock: to,
    topics: [topic],
  });
  return logs.map((log) => {
    const parsed = routerInterface.parseLog(log);
    if (!parsed) {
      console.warn('[indexer] Failed to parse log:', log.transactionHash);
      return null;
    }
    const { songId, listener, grossAmount, protocolFee, netAmount, paymentType } = parsed.args as any;
    return {
      songId: String(songId).toLowerCase(),
      listener: String(listener),
      grossAmount: ethers.formatEther(grossAmount),
      protocolFee: ethers.formatEther(protocolFee),
      netAmount: ethers.formatEther(netAmount),
      paymentType: Number(paymentType),
      txHash: log.transactionHash,
      logIndex: log.index,
      blockNumber: log.blockNumber,
    };
  }).filter((evt): evt is PaymentEvt => evt !== null);
}

async function upsertEvent(evt: PaymentEvt) {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    const songRow = await client.query('SELECT id FROM songs WHERE song_hash = $1 LIMIT 1', [evt.songId]);
    const songId = songRow.rows[0]?.id || null;
    const paymentType = paymentTypeMap[evt.paymentType] || 'stream';
    const insert = await client.query(
      `INSERT INTO plays (song_id, song_hash, listener_address, amount, payment_type, tx_hash, log_index, block_number, protocol_fee, net_amount, timestamp)
       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10, NOW())
       ON CONFLICT (tx_hash, log_index) DO NOTHING
       RETURNING id`,
      [
        songId,
        evt.songId,
        evt.listener,
        evt.grossAmount,
        paymentType,
        evt.txHash,
        evt.logIndex,
        evt.blockNumber,
        evt.protocolFee,
        evt.netAmount,
      ],
    );

    if ((insert.rowCount ?? 0) > 0 && songId) {
      await client.query(
        'UPDATE songs SET plays = plays + 1, earnings = earnings + $2 WHERE id = $1',
        [songId, evt.netAmount],
      );
    }

    await client.query('COMMIT');
  } catch (e) {
    await client.query('ROLLBACK');
    throw e;
  } finally {
    client.release();
  }
}

async function run() {
  await ensureDeps();
  let from = await getStartBlock();
  console.log(`[indexer] starting from block ${from}, router ${ROUTER_ADDRESS}`);

  while (true) {
    try {
      const head = await provider.getBlockNumber();
      if (from > head) {
        await new Promise((r) => setTimeout(r, 1000));
        continue;
      }
      const to = Math.min(head, from + CHUNK_SIZE);
      const events = await fetchEvents(from, to);
      for (const evt of events) {
        try {
          await upsertEvent(evt);
          eventsCounter.inc();
        } catch (e: any) {
          await enqueueRetry(evt, e?.message || 'unknown');
        }
      }
      await processRetries();
      await storeCheckpoint(to);
      lagGauge.set(Math.max(0, head - to));
      from = to + 1;
    } catch (e: any) {
      console.error('[indexer] error', e?.message || e);
      await new Promise((r) => setTimeout(r, 3000));
    }
  }
}

function startMetricsServer() {
  const server = createServer(async (req, res) => {
    if (req.url === '/metrics') {
      res.setHeader('Content-Type', register.contentType);
      res.end(await register.metrics());
      return;
    }
    res.statusCode = 404;
    res.end();
  });
  server.listen(METRICS_PORT, () => {
    console.log(`[indexer] metrics listening on :${METRICS_PORT}/metrics`);
  });
}

startMetricsServer();

run().catch((e) => {
  console.error('[indexer] fatal', e);
  process.exit(1);
});
