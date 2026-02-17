import { Pool } from 'pg';
import { requireAdminKey, isAdmin } from '../security';
import { ethers } from 'ethers';

export function registerStrategyConfigRoutes(app: any, pool: Pool, adminKeyLimiter: any) {
  app.post('/api/strategy-configs', adminKeyLimiter, requireAdminKey, async (req: any, res: any) => {
    try {
      const { name, payload, published, hash, adminSignature } = req.body || {};
      if (!name || !payload || !hash) {
        return res.status(400).json({ error: 'name, payload, and hash required' });
      }
      await pool.query(
        'INSERT INTO strategy_configs (name, payload, published, published_at, hash, admin_signature) VALUES ($1, $2, $3, $4, $5, $6)',
        [name, payload, Boolean(published), published ? new Date().toISOString() : null, hash, adminSignature || null],
      );
      return res.status(201).json({ ok: true });
    } catch (e: any) {
      console.error('Failed to save strategy config', e);
      return res.status(500).json({ error: 'Failed to save strategy config', message: e?.message || 'unknown' });
    }
  });

  app.get('/api/strategy-configs/latest', async (req: any, res: any) => {
    try {
      const includeUnpublished = String(req.query.admin || 'false').toLowerCase() === 'true';
      if (includeUnpublished && !isAdmin(req)) {
        return res.status(403).json({ error: 'forbidden' });
      }
      const row = await pool.query(
        includeUnpublished
          ? 'SELECT id, name, payload, published, published_at, created_at, hash, admin_signature FROM strategy_configs ORDER BY created_at DESC LIMIT 1'
          : 'SELECT id, name, payload, published, published_at, created_at, hash, admin_signature FROM strategy_configs WHERE published = TRUE ORDER BY created_at DESC LIMIT 1'
      );
      if (!row.rows.length) {
        return res.status(404).json({ error: 'no_configs' });
      }
      return res.json(withSignatureStatus(row.rows[0]));
    } catch (e: any) {
      console.error('Failed to fetch latest strategy config', e);
      return res.status(500).json({ error: 'Failed to fetch strategy config', message: e?.message || 'unknown' });
    }
  });

  app.get('/api/strategy-configs', adminKeyLimiter, requireAdminKey, async (_req: any, res: any) => {
    try {
      const rows = await pool.query('SELECT id, name, published, published_at, created_at, hash, admin_signature FROM strategy_configs ORDER BY created_at DESC LIMIT 50');
      return res.json(rows.rows.map(withSignatureStatus));
    } catch (e: any) {
      console.error('Failed to list strategy configs', e);
      return res.status(500).json({ error: 'Failed to list strategy configs', message: e?.message || 'unknown' });
    }
  });

  app.get('/api/strategy-configs/:id', async (req: any, res: any) => {
    try {
      const { id } = req.params;
      const includeUnpublished = String(req.query.admin || 'false').toLowerCase() === 'true';
      if (includeUnpublished && !isAdmin(req)) {
        return res.status(403).json({ error: 'forbidden' });
      }
      const row = includeUnpublished
        ? await pool.query('SELECT id, name, payload, published, published_at, created_at, hash, admin_signature FROM strategy_configs WHERE id = $1', [id])
        : await pool.query('SELECT id, name, payload, published, published_at, created_at, hash, admin_signature FROM strategy_configs WHERE id = $1 AND published = TRUE', [id]);
      if (!row.rows.length) {
        return res.status(404).json({ error: 'not_found' });
      }
      return res.json(withSignatureStatus(row.rows[0]));
    } catch (e: any) {
      console.error('Failed to fetch strategy config', e);
      return res.status(500).json({ error: 'Failed to fetch strategy config', message: e?.message || 'unknown' });
    }
  });

  app.post('/api/strategy-configs/:id/publish', adminKeyLimiter, requireAdminKey, async (req: any, res: any) => {
    try {
      const { id } = req.params;
      const { expectedHash, adminSignature } = req.body || {};
      const check = await pool.query('SELECT hash FROM strategy_configs WHERE id = $1', [id]);
      if (!check.rows.length) return res.status(404).json({ error: 'not_found' });
      if (!expectedHash || check.rows[0].hash !== expectedHash) {
        return res.status(409).json({ error: 'hash_mismatch' });
      }
      const pub = process.env.ADMIN_SIGNER_PUBLIC_KEY;
      if (pub) {
        const sig = adminSignature || null;
        if (!sig) return res.status(400).json({ error: 'admin_signature_required' });
        const signer = ethers.verifyMessage(expectedHash, sig);
        const expectedAddr = ethers.getAddress(pub);
        if (ethers.getAddress(signer) !== expectedAddr) {
          return res.status(403).json({ error: 'invalid_signature' });
        }
      }
      const result = await pool.query(
        'UPDATE strategy_configs SET published = TRUE, published_at = NOW(), admin_signature = $2 WHERE id = $1 RETURNING id',
        [id, adminSignature || null],
      );
      if (!result.rowCount) {
        return res.status(404).json({ error: 'not_found' });
      }
      return res.json({ ok: true, id });
    } catch (e: any) {
      console.error('Failed to publish strategy config', e);
      return res.status(500).json({ error: 'Failed to publish strategy config', message: e?.message || 'unknown' });
    }
  });
}

function withSignatureStatus(row: any) {
  const pub = process.env.ADMIN_SIGNER_PUBLIC_KEY;
  let signature_valid = false;
  if (pub && row?.hash && row?.admin_signature) {
    try {
      const signer = ethers.verifyMessage(row.hash, row.admin_signature);
      signature_valid = ethers.getAddress(signer) === ethers.getAddress(pub);
    } catch {}
  }
  return { ...row, signature_valid };
}
