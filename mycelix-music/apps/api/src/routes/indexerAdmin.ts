// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { Router } from 'express';
import { JsonRpcProvider, ethers } from 'ethers';
import { requireAdminKey } from '../security';
import { pool } from '../index';

type ReplayRequest = {
  fromBlock: number;
  toBlock: number;
};

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

export function registerIndexerAdminRoutes(app: any, provider: JsonRpcProvider, routerAddress: string) {
  const router = Router();
  const iface = new ethers.Interface([
    'event PaymentRecorded(bytes32 indexed songId, address indexed listener, uint256 grossAmount, uint256 protocolFee, uint256 netAmount, uint8 paymentType)',
  ]);

  router.post('/replay', requireAdminKey, async (req, res) => {
    const { fromBlock, toBlock } = req.body as ReplayRequest;
    if (Number.isNaN(fromBlock) || Number.isNaN(toBlock) || fromBlock > toBlock) {
      return res.status(400).json({ error: 'invalid_block_range' });
    }
    try {
      const logs = await provider.getLogs({
        address: routerAddress,
        fromBlock,
        toBlock,
        topics: [iface.getEvent('PaymentRecorded')?.topicHash ?? ''],
      });
      let success = 0;
      for (const log of logs) {
        const evt = parseLog(log, iface);
        try {
          await upsertEvent(evt);
          success += 1;
        } catch (e: any) {
          await storePoison(evt, e?.message || 'unknown');
        }
      }
      return res.json({ ok: true, scanned: logs.length, success });
    } catch (e: any) {
      return res.status(500).json({ error: 'replay_failed', message: e?.message || 'unknown' });
    }
  });

  router.get('/poison', requireAdminKey, async (_req, res) => {
    try {
      const rows = await pool.query('SELECT tx_hash, log_index, song_hash, reason, attempts, block_number, created_at FROM indexer_poison ORDER BY created_at DESC LIMIT 100');
      res.json(rows.rows);
    } catch (e: any) {
      res.status(500).json({ error: 'poison_fetch_failed', message: e?.message || 'unknown' });
    }
  });

  router.post('/poison/retry', requireAdminKey, async (_req, res) => {
    try {
      const rows = await pool.query('SELECT tx_hash, log_index, song_hash, block_number FROM indexer_poison ORDER BY created_at ASC LIMIT 50');
      let retried = 0;
      for (const row of rows.rows) {
        try {
          const receipt = await provider.getTransactionReceipt(row.tx_hash);
          if (!receipt) continue;
          const log = receipt.logs.find((l: any) => l.logIndex === Number(row.log_index));
          if (!log) continue;
          const evt = parseLog(log, iface);
          await upsertEvent(evt);
          await pool.query('DELETE FROM indexer_poison WHERE tx_hash = $1 AND log_index = $2', [row.tx_hash, row.log_index]);
          retried += 1;
        } catch (e) {
          // leave in poison table
        }
      }
      return res.json({ ok: true, retried });
    } catch (e: any) {
      return res.status(500).json({ error: 'poison_retry_failed', message: e?.message || 'unknown' });
    }
  });

  app.use('/api/indexer', router);
}

function parseLog(log: any, iface: ethers.Interface): PaymentEvt {
  const parsed = iface.parseLog(log);
  if (!parsed) {
    throw new Error('Failed to parse log');
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
    if (insert.rowCount && insert.rowCount > 0 && songId) {
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

async function storePoison(evt: PaymentEvt, reason: string) {
  try {
    await pool.query(
      `INSERT INTO indexer_poison (tx_hash, log_index, song_hash, reason, attempts)
       VALUES ($1, $2, $3, $4, $5)
       ON CONFLICT (tx_hash, log_index) DO UPDATE SET reason = EXCLUDED.reason, attempts = indexer_poison.attempts + 1`,
      [evt.txHash, evt.logIndex, evt.songId, reason, 1],
    );
  } catch (e) {
    console.error('[indexer-admin] failed to store poison', e);
  }
}
