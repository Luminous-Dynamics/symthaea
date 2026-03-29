// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import express from 'express';
import cors from 'cors';
import { Pool } from 'pg';
import { createClient } from 'redis';
import * as dotenv from 'dotenv';
import multer from 'multer';
import FormData from 'form-data';
import { CeramicClient } from '@ceramicnetwork/http-client';
import { TileDocument } from '@ceramicnetwork/stream-tile';
import { DID } from 'dids';
import { Ed25519Provider } from 'key-did-provider-ed25519';
import { getResolver as KeyDidResolver } from 'key-did-resolver';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import { openapi } from './openapi';
import { ethers } from 'ethers';
import { messageForClaim, messageForPlay, messageForSong } from './types/auth';
import { metricsHandler, recordRequest, recordUploadBlock, setIndexerLag } from './metrics';
import { requireAdminKey } from './security';
import { registerStrategyConfigRoutes } from './routes/strategyConfigs';
import { registerPreviewRoute } from './routes/preview';
import { registerIndexerAdminRoutes } from './routes/indexerAdmin';
import { fileTypeFromBuffer } from 'file-type';
import { validateBody, validateQuery, songSchema, playSchema, claimSchema, analyticsQuerySchema, songsQuerySchema, artistSongsQuerySchema, topSongsQuerySchema, playsQuerySchema, healthDetailsQuerySchema } from './validators';
import { validateUploadHeaders } from './validators/uploads';
import { TypedDataDomain, TypedDataField } from 'ethers';
import { buildTypedPayload } from './typedPayloads';
import { loadConfig } from './config';

dotenv.config();
const config = loadConfig();

const app = express();
const PORT = config.port || 3100;
const SIGNATURE_TTL_MS = parseInt(String(process.env.SIGNATURE_TTL_MS || '300000'), 10); // 5 min default
const RPC_URL = config.rpcUrl;
const ROUTER_ADDRESS = config.routerAddress;
const provider = new ethers.JsonRpcProvider(RPC_URL);

// Log environment warnings
function logEnvWarnings() {
  const warns: string[] = [];
  if (!process.env.DATABASE_URL) warns.push('DATABASE_URL not set');
  if (!process.env.REDIS_URL) warns.push('REDIS_URL not set');
  if (!process.env.IPFS_API_URL) warns.push('IPFS_API_URL not set (using default http://localhost:5001)');
  if (!process.env.CERAMIC_URL) warns.push('CERAMIC_URL not set (claims will be recorded but not verifiable)');
  if (!process.env.API_ADMIN_KEY) warns.push('API_ADMIN_KEY not set (signature auth only)');
  if (!ROUTER_ADDRESS) warns.push('ROUTER_ADDRESS not set (indexer admin routes disabled)');
  if (warns.length) {
    console.warn('[env] Warnings:', warns.join('; '));
  }
}

function ensureRequiredEnv() {
  const missing = config.missingCritical;
  if (missing.length && !config.isTest) {
    console.error(`[env] Missing required env vars: ${missing.join(', ')}; refusing to start`);
    process.exit(1);
  }
}

function normalizeAddress(addr: string | undefined) {
  return addr ? ethers.getAddress(addr) : '';
}

function verifySignature(expectedSigner: string, messageParts: string[], signature: string): boolean {
  try {
    const signer = ethers.verifyMessage(messageParts.join('|'), signature);
    return normalizeAddress(signer) === normalizeAddress(expectedSigner);
  } catch {
    return false;
  }
}

function verifyTypedData(
  expectedSigner: string,
  domain: TypedDataDomain,
  types: Record<string, Array<TypedDataField>>,
  value: Record<string, any>,
  signature: string,
): boolean {
  try {
    const signer = ethers.verifyTypedData(domain, types, value, signature);
    return normalizeAddress(signer) === normalizeAddress(expectedSigner);
  } catch {
    return false;
  }
}

async function isReplay(message: string): Promise<boolean> {
  if (!redis.isOpen) return false;
  try {
    const key = `auth-replay:${message}`;
    const exists = await redis.exists(key);
    if (exists) return true;
    const ttlSeconds = Math.max(1, Math.floor(SIGNATURE_TTL_MS / 1000));
    await redis.setEx(key, ttlSeconds, '1');
  } catch {}
  return false;
}

type AuthFailureReason = 'missing_signature' | 'expired' | 'invalid_signature' | 'missing_admin_key';

const NONCE_TTL_SECONDS = parseInt(String(process.env.AUTH_NONCE_TTL_SECONDS || '86400'), 10);
const ENABLE_EIP712 = String(process.env.ENABLE_EIP712 || 'true').toLowerCase() === 'true';
const EIP712_CHAIN_ID = parseInt(String(process.env.EIP712_CHAIN_ID || '31337'), 10);
const EIP712_VERIFIER = process.env.EIP712_VERIFIER || ROUTER_ADDRESS;

async function isNonceReplay(signer: string, nonce?: string): Promise<boolean> {
  if (!nonce || !redis.isOpen) return false;
  try {
    const key = `auth-nonce:${normalizeAddress(signer)}:${nonce}`;
    const exists = await redis.exists(key);
    if (exists) return true;
    await redis.setEx(key, NONCE_TTL_SECONDS, '1');
  } catch {}
  return false;
}

function buildAuthGuard(messageBuilder: (req: any) => string[], context: string) {
  return (req: any, res: any, next: any) => {
    const adminKey = process.env.API_ADMIN_KEY;
    const provided = req.headers['x-api-key'];
    if (adminKey && provided && String(provided) === adminKey) {
      console.log(JSON.stringify({
        ts: new Date().toISOString(),
        level: 'info',
        msg: 'auth bypass via api key',
        context,
        requestId: req?.requestId || 'unknown',
      }));
      return next();
    }

    const { signature, signer, timestamp, method } = req.body || {};
    const nonce = req.body?.nonce as string | undefined;
    const ts = Number(timestamp);
    if (!signature || !signer || Number.isNaN(ts)) {
      return respondAuthError(res, 'missing_signature');
    }
    const now = Date.now();
    const skew = Math.abs(now - ts);
    if (skew > SIGNATURE_TTL_MS) {
      return respondAuthError(res, 'expired', {
        ts_diff_ms: skew,
        server_time: new Date(now).toISOString(),
        client_ts: ts,
      });
    }

    const messageParts = messageBuilder(req);
    let ok = false;
    if (method === 'eip712' && ENABLE_EIP712 && EIP712_VERIFIER) {
      const domain = { name: 'MycelixMusic', version: '1', chainId: EIP712_CHAIN_ID, verifyingContract: EIP712_VERIFIER };
      const typed = buildTypedPayload(context, req, ts, nonce);
      if (typed) {
        ok = verifyTypedData(String(signer), domain, typed.types, typed.value, String(signature));
      }
    }
    if (!ok) {
      ok = verifySignature(String(signer), messageParts, String(signature));
    }
    if (!ok) {
      logAuthFailure(context, req, messageParts);
      return respondAuthError(res, 'invalid_signature');
    }
    const message = messageParts.join('|');
    void (async () => {
      if (await isNonceReplay(String(signer), nonce)) {
        logAuthFailure(context, req, messageParts);
        return res.status(403).json({ error: 'auth_failed', reason: 'replay' });
      }
      const seen = await isReplay(message);
      if (seen) {
        logAuthFailure(context, req, messageParts);
        return res.status(403).json({ error: 'auth_failed', reason: 'replay' });
      }
      return next();
    })();
  };
}

function respondAuthError(res: any, reason: AuthFailureReason, meta?: Record<string, unknown>) {
  const status = reason === 'expired' ? 400 : 403;
  return res.status(status).json({ error: 'auth_failed', reason, ...(meta || {}) });
}

function logAuthFailure(context: string, req: any, messageParts: string[]) {
  const rid = req?.requestId || 'unknown';
  console.warn(JSON.stringify({
    ts: new Date().toISOString(),
    level: 'warn',
    msg: 'auth signature validation failed',
    context,
    requestId: rid,
    message: messageParts.join('|'),
  }));
}

async function getIndexerState(): Promise<{ lastBlock: number | null; source: 'db' | 'redis' | null; error?: string }> {
  try {
    const dbState = await pool.query('SELECT last_block FROM indexer_state WHERE name = $1 LIMIT 1', ['router']);
    if (dbState.rows.length) {
      return { lastBlock: Number(dbState.rows[0].last_block), source: 'db' };
    }
  } catch (e: any) {
    return { lastBlock: null, source: null, error: e?.message || 'db_error' };
  }
  try {
    const last = await redis.get(INDEXER_LAST_BLOCK_KEY);
    if (last) return { lastBlock: Number(last), source: 'redis' };
  } catch (e: any) {
    return { lastBlock: null, source: null, error: e?.message || 'redis_error' };
  }
  return { lastBlock: null, source: null };
}

// Middleware
// CORS: restrict origins unless ENABLE_CORS=true
const enableCors = String(process.env.ENABLE_CORS || 'true').toLowerCase() === 'true';
const allowedOriginsEnv = process.env.ALLOWED_ORIGINS || process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000';
const allowedOrigins = allowedOriginsEnv.split(',').map((s) => s.trim()).filter(Boolean);
app.use(cors({
  origin: (origin: any, callback: any) => {
    if (enableCors) return callback(null, true);
    if (!origin) return callback(null, true); // SSR / server-to-server
    if (allowedOrigins.includes(origin)) return callback(null, true);
    return callback(new Error('CORS blocked'), false);
  },
  credentials: true,
}));
app.disable('x-powered-by');
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'", "'unsafe-inline'", "https://unpkg.com"], // For Swagger UI
      styleSrc: ["'self'", "'unsafe-inline'", "https://unpkg.com"],
      imgSrc: ["'self'", "data:", "https:"],
      connectSrc: ["'self'"],
      frameSrc: ["'none'"],
      objectSrc: ["'none'"],
    },
  },
  crossOriginEmbedderPolicy: false, // Required for external resources
  crossOriginResourcePolicy: { policy: 'cross-origin' }, // Allow CORS for API
  hsts: {
    maxAge: 31536000, // 1 year
    includeSubDomains: true,
    preload: true,
  },
  noSniff: true,
  xssFilter: true,
  referrerPolicy: { policy: 'strict-origin-when-cross-origin' },
}));
app.set('trust proxy', 1);
// Attach request id for traceability
app.use((req: any, res: any, next: any) => {
  const rid = req.headers['x-request-id'] || `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  (req as any).requestId = rid;
  res.setHeader('X-Request-Id', String(rid));
  next();
});
// Structured request logging (toggle via LOG_REQUESTS)
const enableRequestLogs = String(process.env.LOG_REQUESTS || 'true').toLowerCase() === 'true';
if (enableRequestLogs) {
  app.use((req: any, res: any, next: any) => {
    const start = Date.now();
    const { method, url } = req;
    const rid = (req as any).requestId;
    const ua = req.headers['user-agent'] || '';
    res.on('finish', () => {
      const duration = Date.now() - start;
      const log = {
        ts: new Date().toISOString(),
        level: 'info',
        requestId: rid,
        method,
        path: url,
        status: res.statusCode,
        dur_ms: duration,
        ua,
      };
      console.log(JSON.stringify(log));
      recordRequest({ method, path: url, status: res.statusCode }, duration);
    });
    next();
  });
}
const GLOBAL_RATE_WINDOW_MS = parseInt(String(process.env.RATE_LIMIT_WINDOW_MS || '60000'), 10);
const GLOBAL_RATE_MAX = parseInt(String(process.env.RATE_LIMIT_MAX || '120'), 10);
const STRICT_RATE_WINDOW_MS = parseInt(String(process.env.STRICT_RATE_LIMIT_WINDOW_MS || '60000'), 10);
const STRICT_RATE_MAX = parseInt(String(process.env.STRICT_RATE_LIMIT_MAX || '30'), 10);

const globalLimiter = rateLimit({
  windowMs: GLOBAL_RATE_WINDOW_MS,
  max: GLOBAL_RATE_MAX,
  standardHeaders: true,
  legacyHeaders: false,
  // Narrowly type to avoid implicit any complaint in strict mode
  skip: (req: any) => req.path.startsWith('/health'),
});
app.use(globalLimiter);
// JSON body parsing with size limit to prevent DoS
app.use(express.json({ limit: '1mb' }));
// URL-encoded body parsing with size limit
app.use(express.urlencoded({ extended: true, limit: '1mb' }));
// File uploads (in-memory)
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 100 * 1024 * 1024 } });
// Stricter limiter for heavy routes
const strictLimiter = rateLimit({ windowMs: STRICT_RATE_WINDOW_MS, max: STRICT_RATE_MAX, standardHeaders: true, legacyHeaders: false });
const ADMIN_KEY_RATE_WINDOW_MS = parseInt(String(process.env.ADMIN_KEY_RATE_WINDOW_MS || '60000'), 10);
const ADMIN_KEY_RATE_MAX = parseInt(String(process.env.ADMIN_KEY_RATE_MAX || '20'), 10);
const ENABLE_MANUAL_PLAY = String(process.env.ENABLE_MANUAL_PLAY || 'false').toLowerCase() === 'true';
const MANUAL_PLAY_ALLOWED = ENABLE_MANUAL_PLAY && process.env.NODE_ENV === 'test';
const ENABLE_UPLOADS = String(process.env.ENABLE_UPLOADS || 'true').toLowerCase() === 'true';
const UPLOAD_QUOTA_WINDOW_SECONDS = parseInt(String(process.env.UPLOAD_QUOTA_WINDOW_SECONDS || '600'), 10);
const UPLOAD_QUOTA_MAX = parseInt(String(process.env.UPLOAD_QUOTA_MAX || '20'), 10);
const UPLOAD_QUOTA_WALLET_MAX = parseInt(String(process.env.UPLOAD_QUOTA_WALLET_MAX || '10'), 10);
const INDEXER_LAST_BLOCK_KEY = process.env.INDEXER_LAST_BLOCK_KEY || 'idx:last_block';
const REQUIRE_MIGRATIONS = String(process.env.REQUIRE_MIGRATIONS || 'false').toLowerCase() === 'true';
const ENABLE_UPLOAD_VALIDATION = String(process.env.ENABLE_UPLOAD_VALIDATION || 'true').toLowerCase() === 'true';
const adminKeyLimiter = rateLimit({
  windowMs: ADMIN_KEY_RATE_WINDOW_MS,
  max: ADMIN_KEY_RATE_MAX,
  standardHeaders: true,
  legacyHeaders: false,
  keyGenerator: (req: any) => req.headers['x-api-key'] || req.ip,
  skip: (req: any) => !req.headers['x-api-key'],
});

const requireAdminKeyIfConfigured = (req: any, res: any, next: any) => {
  if (config.uploadAuthMode === 'open' && !process.env.API_ADMIN_KEY) {
    return next();
  }
  return requireAdminKey(req as any, res as any, next as any);
};

// Database connection
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  max: parseInt(String(process.env.DB_POOL_MAX || '10'), 10),
  idleTimeoutMillis: parseInt(String(process.env.DB_POOL_IDLE || '10000'), 10),
  connectionTimeoutMillis: parseInt(String(process.env.DB_POOL_TIMEOUT || '0'), 10),
});

// Redis connection
const redis = createClient({
  url: process.env.REDIS_URL,
});

redis.on('error', (err) => console.log('Redis Client Error', err));

// Initialize database
async function initDatabase() {
  const client = await pool.connect();
  try {
    if (REQUIRE_MIGRATIONS) {
      const missing: string[] = [];
      const tables = await client.query("SELECT to_regclass('public.songs') AS songs, to_regclass('public.plays') AS plays");
      if (!tables.rows[0]?.songs) missing.push('songs');
      if (!tables.rows[0]?.plays) missing.push('plays');
      if (missing.length) {
        throw new Error(`Required tables missing (${missing.join(', ')}); run migrations before starting or unset REQUIRE_MIGRATIONS`);
      }
      return;
    }

    await client.query(`
      CREATE TABLE IF NOT EXISTS songs (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        artist TEXT NOT NULL,
        artist_address TEXT NOT NULL,
        genre TEXT NOT NULL,
        description TEXT,
        ipfs_hash TEXT NOT NULL,
        payment_model TEXT NOT NULL,
        cover_art TEXT,
        audio_url TEXT,
        claim_stream_id TEXT,
        song_hash TEXT,
        created_at TIMESTAMP DEFAULT NOW(),
        plays INTEGER DEFAULT 0,
        earnings NUMERIC DEFAULT 0
      )
    `);

    await client.query(`
      CREATE TABLE IF NOT EXISTS strategy_configs (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        payload JSONB NOT NULL,
        published BOOLEAN DEFAULT FALSE,
        published_at TIMESTAMP,
        hash TEXT,
        admin_signature TEXT,
        created_at TIMESTAMP DEFAULT NOW()
      )
    `);
    await client.query(`CREATE UNIQUE INDEX IF NOT EXISTS idx_strategy_configs_hash ON strategy_configs (hash)`);

    // Backfill columns if schema existed before
    await client.query(`ALTER TABLE songs ADD COLUMN IF NOT EXISTS cover_art TEXT`);
    await client.query(`ALTER TABLE songs ADD COLUMN IF NOT EXISTS audio_url TEXT`);
    await client.query(`ALTER TABLE songs ADD COLUMN IF NOT EXISTS claim_stream_id TEXT`);
    await client.query(`ALTER TABLE songs ADD COLUMN IF NOT EXISTS song_hash TEXT`);
    await client.query(`CREATE UNIQUE INDEX IF NOT EXISTS idx_songs_song_hash ON songs (song_hash)`);

    await client.query(`
      CREATE TABLE IF NOT EXISTS plays (
        id SERIAL PRIMARY KEY,
        song_id TEXT REFERENCES songs(id),
        listener_address TEXT NOT NULL,
        amount NUMERIC DEFAULT 0,
        payment_type TEXT NOT NULL,
        song_hash TEXT,
        tx_hash TEXT,
        log_index INTEGER,
        block_number INTEGER,
        protocol_fee NUMERIC DEFAULT 0,
        net_amount NUMERIC DEFAULT 0,
        timestamp TIMESTAMP DEFAULT NOW()
      )
    `);
    await client.query(`ALTER TABLE plays ADD COLUMN IF NOT EXISTS song_hash TEXT`);
    await client.query(`ALTER TABLE plays ADD COLUMN IF NOT EXISTS tx_hash TEXT`);
    await client.query(`ALTER TABLE plays ADD COLUMN IF NOT EXISTS log_index INTEGER`);
    await client.query(`ALTER TABLE plays ADD COLUMN IF NOT EXISTS block_number INTEGER`);
    await client.query(`ALTER TABLE plays ADD COLUMN IF NOT EXISTS protocol_fee NUMERIC DEFAULT 0`);
    await client.query(`ALTER TABLE plays ADD COLUMN IF NOT EXISTS net_amount NUMERIC DEFAULT 0`);
    await client.query(`CREATE UNIQUE INDEX IF NOT EXISTS idx_plays_tx_log ON plays (tx_hash, log_index)`);

    // Useful indexes for filtering
    await client.query(`CREATE INDEX IF NOT EXISTS idx_songs_created_at ON songs (created_at DESC)`);
    await client.query(`CREATE INDEX IF NOT EXISTS idx_songs_genre ON songs (genre)`);
    await client.query(`CREATE INDEX IF NOT EXISTS idx_songs_payment_model ON songs (payment_model)`);
    await client.query(`CREATE INDEX IF NOT EXISTS idx_songs_title_lower ON songs ((LOWER(title)))`);
    await client.query(`CREATE INDEX IF NOT EXISTS idx_songs_artist_lower ON songs ((LOWER(artist)))`);
    await client.query(`CREATE INDEX IF NOT EXISTS idx_songs_plays ON songs (plays)`);
    await client.query(`CREATE INDEX IF NOT EXISTS idx_songs_earnings ON songs (earnings)`);
    await client.query(`CREATE INDEX IF NOT EXISTS idx_plays_timestamp ON plays (timestamp)`);

    await client.query(`
      CREATE TABLE IF NOT EXISTS indexer_state (
        name TEXT PRIMARY KEY,
        last_block INTEGER NOT NULL,
        updated_at TIMESTAMP DEFAULT NOW()
      )
    `);
    await client.query(`
      CREATE TABLE IF NOT EXISTS indexer_poison (
        id SERIAL PRIMARY KEY,
        tx_hash TEXT NOT NULL,
        log_index INTEGER NOT NULL,
        song_hash TEXT,
        reason TEXT,
        attempts INTEGER DEFAULT 0,
        block_number INTEGER,
        created_at TIMESTAMP DEFAULT NOW(),
        UNIQUE (tx_hash, log_index)
      )
    `);

    // Backfill song_hash where missing
    const songsNeedingHash = await client.query('SELECT id FROM songs WHERE song_hash IS NULL OR song_hash = \'\'');
    for (const row of songsNeedingHash.rows) {
      try {
        const hash = ethers.id(row.id);
        await client.query('UPDATE songs SET song_hash = $1 WHERE id = $2', [hash, row.id]);
      } catch (e) {
        console.warn('Failed to backfill song_hash for', row.id, e);
      }
    }

    console.log('✓ Database tables initialized');
  } finally {
    client.release();
  }
}

// ============================================================
// API Routes
// ============================================================

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Detailed health (DB/Redis/IPFS)
app.get('/health/details', validateQuery(healthDetailsQuerySchema), async (req, res) => {
  const ipfsApi = process.env.IPFS_API_URL || 'http://localhost:5001';
  const now = Date.now();
  const details: any = {
    time: new Date(now).toISOString(),
    signature_ttl_ms: SIGNATURE_TTL_MS,
    server_unix_ms: now,
  };
  const ceramicUrl = process.env.CERAMIC_URL;
  const clientTsRaw = req.query?.client_ts as number | undefined;
  const clientTs = typeof clientTsRaw === 'number' ? clientTsRaw : undefined;
  if (typeof clientTs === 'number') {
    details.clock_skew_ms = now - clientTs;
  }
  const idxState = await getIndexerState();
  if (idxState.error) {
    details.indexer_last_block = `error: ${idxState.error}`;
  } else {
    details.indexer_last_block = idxState.lastBlock;
    details.indexer_state_source = idxState.source;
    if (typeof idxState.lastBlock === 'number') {
      try {
        const chainBlock = await provider.getBlockNumber();
        details.indexer_lag_blocks = Math.max(0, chainBlock - idxState.lastBlock);
      } catch (e: any) {
        details.indexer_lag_blocks = `error: ${e?.message || 'unknown'}`;
      }
    }
  }
  try {
    await pool.query('SELECT 1');
    details.db = 'ok';
  } catch (e: any) {
    details.db = `error: ${e?.message || 'unknown'}`;
  }

  try {
    const ping = await redis.ping();
    details.redis = ping === 'PONG' ? 'ok' : `error: ${ping}`;
  } catch (e: any) {
    details.redis = `error: ${e?.message || 'unknown'}`;
  }

  try {
    const poison = await pool.query('SELECT COUNT(*)::int AS c FROM indexer_poison');
    details.indexer_poison = Number(poison.rows[0]?.c ?? 0);
  } catch (e: any) {
    details.indexer_poison = `error: ${e?.message || 'unknown'}`;
  }

  try {
    const resp = await fetch(`${ipfsApi}/api/v0/version`);
    details.ipfs = resp.ok ? 'ok' : `error: ${resp.status}`;
  } catch (e: any) {
    details.ipfs = `error: ${e?.message || 'unknown'}`;
  }

  // Ceramic status is based on configuration (no network check required)
  details.ceramic = ceramicUrl ? 'configured' : 'not_configured';

  res.json(details);
});

// Indexer-specific health/lag
app.get('/health/indexer', async (_req, res) => {
  try {
    const state = await getIndexerState();
    if (state.error) {
      return res.json({ ok: false, last_block: null, lag_blocks: null, message: state.error });
    }
    if (state.lastBlock === null) {
      return res.json({ ok: false, last_block: null, lag_blocks: null, message: 'no indexer state' });
    }
    const chainBlock = await provider.getBlockNumber();
    const lag = Math.max(0, chainBlock - state.lastBlock);
    setIndexerLag(lag);
    return res.json({ ok: true, last_block: state.lastBlock, lag_blocks: lag, chain_head: chainBlock, source: state.source });
  } catch (e: any) {
    return res.status(500).json({ ok: false, error: e?.message || 'unknown' });
  }
});

// Readiness probe: 200 only if core dependencies ready
app.get('/health/ready', async (req, res) => {
  const result: any = { time: new Date().toISOString(), db: 'unknown', redis: 'unknown' };
  let ok = true;
  try { await pool.query('SELECT 1'); result.db = 'ok'; } catch (e: any) { result.db = `error: ${e?.message || 'unknown'}`; ok = false; }
  try { const ping = await redis.ping(); result.redis = ping === 'PONG' ? 'ok' : `error: ${ping}`; if (ping !== 'PONG') ok = false; } catch (e: any) { result.redis = `error: ${e?.message || 'unknown'}`; ok = false; }
  res.status(ok ? 200 : 503).json(result);
});

// Liveness probe: 200 if the process is running (for Kubernetes)
app.get('/health/live', (_req, res) => {
  res.status(200).json({
    alive: true,
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
  });
});

// Startup probe: 200 once the server is initialized (for Kubernetes slow-starting containers)
let serverStarted = false;
app.get('/health/startup', (_req, res) => {
  if (serverStarted) {
    res.status(200).json({
      started: true,
      timestamp: new Date().toISOString(),
    });
  } else {
    res.status(503).json({
      started: false,
      message: 'Server is still initializing',
    });
  }
});

// Toggle API docs exposure
const enableApiDocs = String(process.env.ENABLE_API_DOCS || 'true').toLowerCase() === 'true';

// OpenAPI spec (static)
app.get('/openapi.json', (req, res) => {
  if (!enableApiDocs) return res.status(404).json({ error: 'Disabled' });
  res.json(openapi);
});

// Minimal HTML docs page referencing OpenAPI and common endpoints
app.get('/docs', (_req, res) => {
  if (!enableApiDocs) return res.status(404).send('Docs disabled');
  const html = `<!doctype html>
  <html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Mycelix Music API Docs</title>
    <style>
      body{font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem;}
      code{background:#f2f2f2; padding:2px 4px; border-radius:4px}
      a{color:#6b46c1; text-decoration:none}
      a:hover{text-decoration:underline}
    </style>
  </head>
  <body>
    <h1>Mycelix Music API</h1>
    <p>Explore the REST API. OpenAPI spec: <a href="/openapi.json">/openapi.json</a></p>
    <h2>Health</h2>
    <ul>
      <li><a href="/health">/health</a></li>
      <li><a href="/health/details">/health/details</a></li>
      <li><a href="/health/ready">/health/ready</a></li>
    </ul>
    <h2>Songs</h2>
    <ul>
      <li><a href="/api/songs?limit=5">/api/songs?limit=5</a></li>
      <li><a href="/api/songs/export?limit=100">/api/songs/export</a></li>
    </ul>
    <h2>Analytics</h2>
    <ul>
      <li><a href="/api/analytics/top-songs">/api/analytics/top-songs</a></li>
    </ul>
  </body>
  </html>`;
  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  res.send(html);
});

// Swagger UI (served from CDN; enable with ENABLE_SWAGGER_UI=true)
app.get('/swagger', (_req, res) => {
  const enabled = String(process.env.ENABLE_SWAGGER_UI || 'true').toLowerCase() === 'true';
  if (!enabled || !enableApiDocs) return res.status(404).send('Swagger UI disabled');
  const html = `<!doctype html>
  <html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Mycelix API – Swagger UI</title>
    <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
    <style>body{margin:0}</style>
  </head>
  <body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js" crossorigin></script>
    <script>
      window.onload = () => {
        window.ui = SwaggerUIBundle({
          url: '/openapi.json',
          dom_id: '#swagger-ui',
          presets: [SwaggerUIBundle.presets.apis],
          layout: 'BaseLayout',
        });
      };
    </script>
  </body>
  </html>`;
  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  res.send(html);
});

// Prometheus metrics
app.get('/metrics', metricsHandler);

// Get songs with optional filtering & pagination
app.get('/api/songs', validateQuery(songsQuerySchema), async (req, res) => {
  try {
    const { q, genre, model, limit = '50', offset = '0', sort = 'created_at', order = 'desc', cursor, format } = req.query as Record<string, string>;

    const where: string[] = [];
    const params: any[] = [];

    if (q && q.trim().length > 0) {
      params.push(`%${q.trim().toLowerCase()}%`);
      where.push('(LOWER(title) LIKE $' + params.length + ' OR LOWER(artist) LIKE $' + params.length + ' OR LOWER(genre) LIKE $' + params.length + ')');
    }
    if (genre && genre !== 'all') {
      params.push(genre);
      where.push('genre = $' + params.length);
    }
    if (model && model !== 'all') {
      params.push(model);
      where.push('payment_model = $' + params.length);
    }

    // Validate limit/offset
    const lim = Math.max(1, Math.min(100, parseInt(limit as string, 10) || 50));
    const off = Math.max(0, parseInt(offset as string, 10) || 0);

    // Whitelist sorting
    const sortColumn = ((): string => {
      const s = (sort || '').toLowerCase();
      if (s === 'plays') return 'plays';
      if (s === 'earnings') return 'earnings';
      return 'created_at';
    })();
    const sortOrder = (order || '').toLowerCase() === 'asc' ? 'ASC' : 'DESC';

    let whereSql = where.length ? 'WHERE ' + where.join(' AND ') : '';
    let listSql: string;
    const orderBy = `ORDER BY ${sortColumn} ${sortOrder}, id ${sortOrder}`;

    // Cursor pagination for created_at sort only
    if (sortColumn === 'created_at' && cursor) {
      try {
        const decoded = Buffer.from(cursor, 'base64').toString('utf8');
        const [iso, lastId] = decoded.split('|');
        const cmp = sortOrder === 'DESC' ? '<' : '>';
        const paramTimeIndex = params.push(iso) as unknown as number; // push returns new length
        const paramIdIndex = params.push(lastId) as unknown as number;
        where.push(`(created_at, id) ${cmp} ($${paramTimeIndex}, $${paramIdIndex})`);
        whereSql = 'WHERE ' + where.join(' AND ');
      } catch {}
    }

    listSql = `SELECT * FROM songs ${whereSql} ${orderBy} LIMIT ${lim} OFFSET ${off}`;
    const countSql = `SELECT COUNT(*) FROM songs ${whereSql}`;
    const [result, total] = await Promise.all([
      pool.query(listSql, params),
      pool.query(countSql, params),
    ]);
    res.setHeader('X-Total-Count', total.rows[0].count || '0');
    // Next-cursor header for client (only for created_at sort)
    if (sortColumn === 'created_at' && result.rows.length === lim) {
      const last = result.rows[result.rows.length - 1];
      if (last?.created_at && last?.id) {
        const nextCursor = Buffer.from(`${new Date(last.created_at).toISOString()}|${last.id}`).toString('base64');
        res.setHeader('X-Next-Cursor', nextCursor);
      }
    }

    if ((format || '').toLowerCase() === 'csv') {
      res.setHeader('Content-Type', 'text/csv');
      res.setHeader('Content-Disposition', `attachment; filename="songs_export.csv"`);
      const header = 'id,title,artist,genre,payment_model,plays,earnings,created_at\n';
      const body = result.rows.map((r:any) => [
        r.id,
        quoteCSV(r.title),
        quoteCSV(r.artist),
        quoteCSV(r.genre),
        r.payment_model,
        r.plays ?? 0,
        r.earnings ?? 0,
        r.created_at ? new Date(r.created_at).toISOString() : ''
      ].join(',')).join('\n');
      return res.send(header + body);
    }
    res.json(result.rows);
  } catch (error) {
    console.error('Failed to fetch songs:', error);
    res.status(500).json({ error: 'Failed to fetch songs' });
  }
});

// Export all songs matching filters in CSV (no pagination)
app.get('/api/songs/export', validateQuery(songsQuerySchema), async (req, res) => {
  try {
    const { q, genre, model, sort = 'created_at', order = 'desc' } = req.query as Record<string, string>;

    const where: string[] = [];
    const params: any[] = [];

    if (q && q.trim().length > 0) {
      params.push(`%${q.trim().toLowerCase()}%`);
      where.push('(LOWER(title) LIKE $' + params.length + ' OR LOWER(artist) LIKE $' + params.length + ' OR LOWER(genre) LIKE $' + params.length + ')');
    }
    if (genre && genre !== 'all') {
      params.push(genre);
      where.push('genre = $' + params.length);
    }
    if (model && model !== 'all') {
      params.push(model);
      where.push('payment_model = $' + params.length);
    }

    const sortColumn = ((): string => {
      const s = (sort || '').toLowerCase();
      if (s === 'plays') return 'plays';
      if (s === 'earnings') return 'earnings';
      return 'created_at';
    })();
    const sortOrder = (order || '').toLowerCase() === 'asc' ? 'ASC' : 'DESC';

    const whereSql = where.length ? 'WHERE ' + where.join(' AND ') : '';
    // Add a safety limit for memory protection (configurable via env)
    const exportLimit = parseInt(process.env.CSV_EXPORT_LIMIT || '50000', 10);
    const sql = `SELECT id, title, artist, genre, payment_model, plays, earnings, created_at FROM songs ${whereSql} ORDER BY ${sortColumn} ${sortOrder}, id ${sortOrder} LIMIT ${exportLimit}`;
    const rows = await pool.query(sql, params);

    res.setHeader('Content-Type', 'text/csv; charset=utf-8');
    res.setHeader('Content-Disposition', 'attachment; filename="songs_export_all.csv"');
    res.setHeader('Transfer-Encoding', 'chunked');
    res.setHeader('X-Export-Limit', String(exportLimit));
    res.setHeader('X-Rows-Returned', String(rows.rows.length));

    // Stream header first
    res.write('id,title,artist,genre,payment_model,plays,earnings,created_at\n');

    // Stream rows in chunks to reduce memory pressure
    const chunkSize = 1000;
    for (let i = 0; i < rows.rows.length; i += chunkSize) {
      const chunk = rows.rows.slice(i, i + chunkSize);
      const csvChunk = chunk.map((r: any) => [
        r.id,
        quoteCSV(r.title),
        quoteCSV(r.artist),
        quoteCSV(r.genre),
        r.payment_model,
        r.plays ?? 0,
        r.earnings ?? 0,
        r.created_at ? new Date(r.created_at).toISOString() : ''
      ].join(',')).join('\n');
      res.write(csvChunk + '\n');
    }
    return res.end();
  } catch (error) {
    console.error('Failed to export songs CSV:', error);
    res.status(500).json({ error: 'Failed to export songs CSV' });
  }
});

function quoteCSV(s: string) {
  if (!s) return '';
  const str = String(s);
  if (str.includes(',') || str.includes('"') || str.includes('\n')) {
    return '"' + str.replace(/"/g, '""') + '"';
  }
  return str;
}

// Get a song's Ceramic claim (if configured)
app.get('/api/songs/:id/claim', async (req, res) => {
  try {
    const { id } = req.params;
    const ceramicUrl = process.env.CERAMIC_URL;
    // Look up the stream id from DB
    const row = await pool.query('SELECT claim_stream_id FROM songs WHERE id = $1', [id]);
    if (row.rows.length === 0 || !row.rows[0].claim_stream_id) {
      return res.status(404).json({ error: 'No claim found for song' });
    }
    const streamId = row.rows[0].claim_stream_id as string;
    if (!ceramicUrl) {
      // Return stub without content if Ceramic not configured
      return res.json({ streamId, available: false });
    }
    const ceramic = new CeramicClient(ceramicUrl);
    const doc = await TileDocument.load(ceramic, streamId);
    return res.json({ streamId, available: true, content: doc.content, metadata: doc.metadata });
  } catch (error: any) {
    console.error('Failed to read DKG claim:', error);
    res.status(500).json({ error: 'Failed to read DKG claim', errorId: Date.now().toString(36) });
  }
});

// Read a claim directly by stream id
app.get('/api/claims/:streamId', async (req, res) => {
  try {
    const ceramicUrl = process.env.CERAMIC_URL;
    const { streamId } = req.params;
    if (!ceramicUrl) return res.json({ streamId, available: false });
    const ceramic = new CeramicClient(ceramicUrl);
    const doc = await TileDocument.load(ceramic, streamId);
    return res.json({ streamId, available: true, content: doc.content, metadata: doc.metadata });
  } catch (error: any) {
    console.error('Failed to read claim:', error);
    res.status(500).json({ error: 'Failed to read claim', errorId: Date.now().toString(36) });
  }
});

// Get single song
app.get('/api/songs/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const result = await pool.query('SELECT * FROM songs WHERE id = $1', [id]);

    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'Song not found' });
    }

    res.json(result.rows[0]);
  } catch (error) {
    console.error('Failed to fetch song:', error);
    res.status(500).json({ error: 'Failed to fetch song' });
  }
});

// Get songs for an artist (with optional CSV export)
app.get('/api/artists/:address/songs', validateQuery(artistSongsQuerySchema), async (req, res) => {
  try {
    const { address } = req.params;
    const limit = Math.min(200, Math.max(1, parseInt(String(req.query.limit || '50'), 10)));
    const offset = Math.max(0, parseInt(String(req.query.offset || '0'), 10));
    const order = String(req.query.order || 'desc').toLowerCase() === 'asc' ? 'ASC' : 'DESC';
    const format = String(req.query.format || 'json').toLowerCase();
    const all = String(req.query.all || 'false').toLowerCase() === 'true';

    const listSql = all
      ? `SELECT * FROM songs WHERE artist_address = $1 ORDER BY created_at ${order}`
      : `SELECT * FROM songs WHERE artist_address = $1 ORDER BY created_at ${order} LIMIT ${limit} OFFSET ${offset}`;
    const countSql = `SELECT COUNT(*) FROM songs WHERE artist_address = $1`;
    const [rows, total] = await Promise.all([
      pool.query(listSql, [address]),
      pool.query(countSql, [address])
    ]);

    res.setHeader('X-Total-Count', total.rows[0].count || '0');
    if (format === 'csv') {
      res.setHeader('Content-Type', 'text/csv');
      res.setHeader('Content-Disposition', `attachment; filename="songs_${address}.csv"`);
      const header = 'id,title,artist,genre,payment_model,plays,earnings,created_at\n';
      const body = rows.rows.map((r:any) => [
        r.id,
        quoteCSV(r.title),
        quoteCSV(r.artist),
        quoteCSV(r.genre),
        r.payment_model,
        r.plays ?? 0,
        r.earnings ?? 0,
        r.created_at ? new Date(r.created_at).toISOString() : ''
      ].join(',')).join('\n');
      return res.send(header + body);
    }
    res.json(rows.rows);
  } catch (error) {
    console.error('Failed to fetch artist songs:', error);
    res.status(500).json({ error: 'Failed to fetch artist songs' });
  }
});

const songWriteGuard = buildAuthGuard((req) => messageForSong({
  id: String(req.body?.id || ''),
  artistAddress: String(req.body?.artistAddress || ''),
  ipfsHash: String(req.body?.ipfsHash || ''),
  paymentModel: String(req.body?.paymentModel || ''),
  timestamp: req.body?.timestamp,
}), 'song');

// Register a new song
app.post('/api/songs', adminKeyLimiter, validateBody(songSchema), songWriteGuard, async (req, res) => {
  try {
    const { id, title, artist, artistAddress, genre, description, ipfsHash, paymentModel, coverArt, audioUrl, claimStreamId } = req.body;

    // Basic validation
    if (!id || !title || !artist || !artistAddress || !genre || !ipfsHash || !paymentModel) {
      return res.status(400).json({ error: 'Missing required fields' });
    }
    const allowedModels = new Set([
      'pay_per_stream',
      'gift_economy',
      'pay_per_download',
      'subscription',
      'nft_gated',
      'staking_gated',
      'token_tip',
      'time_barter',
      'patronage',
      'freemium',
      'pay_what_you_want',
      'auction',
    ]);
    if (!allowedModels.has(String(paymentModel))) {
      return res.status(400).json({ error: 'Invalid paymentModel' });
    }
    if (normalizeAddress(artistAddress) !== normalizeAddress(req.body.signer)) {
      return res.status(403).json({ error: 'Signer must match artistAddress' });
    }
    const songHash = ethers.id(String(id));

    const result = await pool.query(
      `INSERT INTO songs (id, title, artist, artist_address, genre, description, ipfs_hash, payment_model, cover_art, audio_url, claim_stream_id, song_hash)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
       ON CONFLICT (id) DO UPDATE SET
         title = EXCLUDED.title,
         artist = EXCLUDED.artist,
         artist_address = EXCLUDED.artist_address,
         genre = EXCLUDED.genre,
         description = EXCLUDED.description,
         ipfs_hash = EXCLUDED.ipfs_hash,
         payment_model = EXCLUDED.payment_model,
         cover_art = EXCLUDED.cover_art,
         audio_url = EXCLUDED.audio_url,
         claim_stream_id = EXCLUDED.claim_stream_id,
         song_hash = EXCLUDED.song_hash
       RETURNING *`,
      [id, title, artist, artistAddress, genre, description, ipfsHash, paymentModel, coverArt || null, audioUrl || null, claimStreamId || null, songHash]
    );

    res.status(201).json(result.rows[0]);
  } catch (error) {
    console.error('Failed to register song:', error);
    res.status(500).json({ error: 'Failed to register song' });
  }
});

const playWriteGuard = buildAuthGuard((req) => messageForPlay({
  songId: String(req.params.id || ''),
  listener: String(req.body?.listenerAddress || 'anonymous'),
  amount: req.body?.amount ?? 0,
  paymentType: String(req.body?.paymentType || 'stream'),
  timestamp: req.body?.timestamp,
}), 'play');

// Record a play
app.post('/api/songs/:id/play', adminKeyLimiter, validateBody(playSchema), playWriteGuard, async (req, res) => {
  if (!MANUAL_PLAY_ALLOWED) {
    return res.status(410).json({ error: 'manual_play_disabled' });
  }
  try {
    const { id } = req.params;
    const { listenerAddress, amount, paymentType } = req.body || {};

    const amt = Number(amount);
    if (Number.isNaN(amt) || amt < 0) {
      return res.status(400).json({ error: 'Invalid amount' });
    }
    const typeStr = String(paymentType || 'stream');
    const allowedTypes = new Set(['stream', 'download', 'tip', 'patronage', 'nft_access']);
    if (!allowedTypes.has(typeStr)) {
      return res.status(400).json({ error: 'Invalid paymentType' });
    }
    if (normalizeAddress(listenerAddress) !== normalizeAddress(req.body.signer)) {
      return res.status(403).json({ error: 'Signer must match listenerAddress' });
    }

    // Insert play record
    await pool.query(
      `INSERT INTO plays (song_id, listener_address, amount, payment_type)
       VALUES ($1, $2, $3, $4)`,
      [id, listenerAddress || 'anonymous', amt, typeStr]
    );

    // Update song stats
    await pool.query(
      `UPDATE songs
       SET plays = plays + 1,
           earnings = earnings + $2
       WHERE id = $1`,
      [id, amt]
    );

    res.json({ success: true });
  } catch (error) {
    console.error('Failed to record play:', error);
    res.status(500).json({ error: 'Failed to record play' });
  }
});

// Get play history for a song
app.get('/api/songs/:id/plays', validateQuery(playsQuerySchema), async (req, res) => {
  try {
    const { id } = req.params;
    const limit = Math.min(1000, Math.max(1, parseInt(String(req.query.limit || '100'), 10)));
    const format = String(req.query.format || 'json').toLowerCase();
    const result = await pool.query(
      'SELECT * FROM plays WHERE song_id = $1 ORDER BY timestamp DESC LIMIT $2',
      [id, limit]
    );

    if (format === 'csv') {
      res.setHeader('Content-Type', 'text/csv');
      res.setHeader('Content-Disposition', `attachment; filename="plays_${id}.csv"`);
      const header = 'timestamp,listener,amount,payment_type\n';
      const body = result.rows.map((r:any) => [
        r.timestamp ? new Date(r.timestamp).toISOString() : '',
        r.listener_address || 'anonymous',
        r.amount ?? 0,
        r.payment_type
      ].join(',')).join('\n');
      return res.send(header + body);
    }

    res.json(result.rows);
  } catch (error) {
    console.error('Failed to fetch plays:', error);
    res.status(500).json({ error: 'Failed to fetch plays' });
  }
});

// Song analytics (timeseries)
app.get('/api/analytics/song/:id', strictLimiter, validateQuery(analyticsQuerySchema), async (req, res) => {
  try {
    const { id } = req.params;
    const days = Math.min(90, Math.max(1, parseInt(String(req.query.days || '30'), 10)));
    const format = String(req.query.format || 'json').toLowerCase();
    const timeseries = await pool.query(
      `SELECT date_trunc('day', timestamp) AS day,
              COUNT(*)::int AS plays,
              COALESCE(SUM(amount),0)::float AS earnings
       FROM plays
       WHERE song_id = $1 AND timestamp >= NOW() - ($2::text || ' days')::interval
       GROUP BY day
       ORDER BY day ASC`,
      [id, String(days)]
    );
    if (format === 'csv') {
      res.setHeader('Content-Type', 'text/csv');
      res.setHeader('Content-Disposition', `attachment; filename="song_${id}_${days}d.csv"`);
      const header = 'day,plays,earnings\n';
      const body = timeseries.rows.map((r:any)=> `${new Date(r.day).toISOString()},${r.plays},${r.earnings}`).join('\n');
      return res.send(header + body);
    }
    res.json({ days, timeseries: timeseries.rows });
  } catch (error) {
    console.error('Failed to fetch song analytics:', error);
    res.status(500).json({ error: 'Failed to fetch song analytics' });
  }
});

// Upload to IPFS (via Web3.Storage)
app.post('/api/upload-to-ipfs', adminKeyLimiter, requireAdminKeyIfConfigured, strictLimiter, validateUploadHeaders, upload.single('file'), async (req, res) => {
  if (!ENABLE_UPLOADS) {
    return res.status(503).json({ error: 'uploads_disabled' });
  }
  try {
    const allowedMime = ['audio/mpeg', 'audio/wav', 'audio/flac', 'image/png', 'image/jpeg'];
    const maxBytes = 100 * 1024 * 1024;
    const ip = req.ip || req.headers['x-forwarded-for'] || 'unknown';
    const wallet = (req.headers['x-wallet'] as string | undefined) || '';
    if (redis.isOpen) {
      try {
        const key = `upload:quota:${ip}`;
        const current = await redis.incr(key);
        if (current === 1) {
          await redis.expire(key, UPLOAD_QUOTA_WINDOW_SECONDS);
        }
        if (current > UPLOAD_QUOTA_MAX) {
          recordUploadBlock('ip');
          return res.status(429).json({ error: 'upload_quota_exceeded', window_seconds: UPLOAD_QUOTA_WINDOW_SECONDS });
        }
        if (wallet) {
          const wKey = `upload:quota:wallet:${wallet.toLowerCase()}`;
          const wCur = await redis.incr(wKey);
          if (wCur === 1) await redis.expire(wKey, UPLOAD_QUOTA_WINDOW_SECONDS);
          if (wCur > UPLOAD_QUOTA_WALLET_MAX) {
            recordUploadBlock('wallet');
            return res.status(429).json({ error: 'upload_wallet_quota_exceeded', window_seconds: UPLOAD_QUOTA_WINDOW_SECONDS });
          }
        }
      } catch (e) {
        console.warn('Upload quota check failed', e);
      }
    }
    // Prefer local IPFS API if configured and service is up
    const ipfsApi = process.env.IPFS_API_URL || 'http://localhost:5001';
    const gateway = process.env.IPFS_GATEWAY || 'https://w3s.link/ipfs';

    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }
    if (req.file.size > maxBytes) {
      return res.status(400).json({ error: 'File too large' });
    }
    if (ENABLE_UPLOAD_VALIDATION) {
      const sniffed = await fileTypeFromBuffer(req.file.buffer);
      const mimeToCheck = sniffed?.mime || req.file.mimetype;
      if (mimeToCheck && !allowedMime.includes(mimeToCheck)) {
        return res.status(400).json({ error: 'Unsupported file type' });
      }
    }

    // Submit to IPFS HTTP API /api/v0/add
    const form = new FormData();
    form.append('file', req.file.buffer, { filename: req.file.originalname, contentType: req.file.mimetype });

    const addResp = await fetch(`${ipfsApi}/api/v0/add?pin=true`, {
      method: 'POST',
      body: form as any,
      headers: (form as any).getHeaders ? (form as any).getHeaders() : undefined,
    });

    if (!addResp.ok) {
      const text = await addResp.text();
      throw new Error(`IPFS add failed: ${addResp.status} ${text}`);
    }

    const bodyText = await addResp.text();
    // The IPFS add API can return NDJSON; take the last JSON line
    const lines = bodyText.trim().split('\n');
    const last = JSON.parse(lines[lines.length - 1]);
    const ipfsHash = last.Hash || last.cid || last.Cid || last.Hashes?.[0];

    if (!ipfsHash) throw new Error('No IPFS hash in response');

    res.json({
      ipfsHash,
      gateway: `${gateway}/${ipfsHash}`,
      name: req.file.originalname,
      size: req.file.size,
    });
  } catch (error: any) {
    console.error('Failed to upload to IPFS:', error);
    res.status(500).json({ error: 'Failed to upload to IPFS', errorId: Date.now().toString(36) });
  }
});

// Create DKG claim (via Ceramic)
const claimWriteGuard = buildAuthGuard((req) => messageForClaim({
  songId: String(req.body?.songId || ''),
  artistAddress: String(req.body?.artistAddress || ''),
  ipfsHash: String(req.body?.ipfsHash || ''),
  title: String(req.body?.title || ''),
  timestamp: req.body?.timestamp,
}), 'claim');

// Create DKG claim (via Ceramic)
app.post('/api/create-dkg-claim', strictLimiter, adminKeyLimiter, validateBody(claimSchema), claimWriteGuard, async (req, res) => {
  const ceramicUrl = process.env.CERAMIC_URL;
  const adminSeedHex = process.env.CERAMIC_ADMIN_SEED;
  try {
    const {
      songId,
      title,
      artist,
      ipfsHash,
      artistAddress,
      epistemicTier,
      networkTier,
      memoryTier,
    } = req.body;

    if (!ceramicUrl || !adminSeedHex) {
      const mockStreamId = `kjzl6cwe1jw14${Math.random().toString(36).substring(2, 15)}`;
      return res.json({
        streamId: mockStreamId,
        songId,
        title,
        artist,
        ipfsHash,
        artistAddress,
        epistemicTier,
        networkTier,
        memoryTier,
        timestamp: new Date().toISOString(),
        mock: true,
      });
    }

    const ceramic = new CeramicClient(ceramicUrl);
    const seedBuf = Buffer.from(adminSeedHex.replace(/^0x/, ''), 'hex');
    if (seedBuf.length !== 32) {
      throw new Error('CERAMIC_ADMIN_SEED must be 32-byte hex');
    }
    const provider = new Ed25519Provider(seedBuf);
    const did = new DID({ provider, resolver: KeyDidResolver() });
    await did.authenticate();
    ceramic.did = did;

    const content = {
      kind: 'mycelix.song.claim',
      songId,
      title,
      artist,
      ipfsHash,
      artistAddress,
      tiers: { epistemicTier, networkTier, memoryTier },
      createdAt: new Date().toISOString(),
      version: 1,
    };

    const doc = await TileDocument.create(ceramic, content, {
      controllers: [did.id],
      family: 'mycelix-music',
      tags: ['song', 'claim', 'mycelix'],
    });

    return res.json({
      streamId: doc.id.toString(),
      did: did.id,
      songId,
      createdAt: content.createdAt,
    });
  } catch (error: any) {
    console.error('Failed to create DKG claim:', error);
    res.status(500).json({ error: 'Failed to create DKG claim', errorId: Date.now().toString(36) });
  }
});

// Strategy configs storage (admin only)
registerStrategyConfigRoutes(app, pool, adminKeyLimiter);
registerPreviewRoute(app, pool);
if (ROUTER_ADDRESS) {
  registerIndexerAdminRoutes(app, provider as any, ROUTER_ADDRESS);
} else {
  console.warn('[init] ROUTER_ADDRESS missing; /api/indexer admin routes not registered');
}

// Get artist stats
app.get('/api/artists/:address/stats', async (req, res) => {
  try {
    const { address } = req.params;

    const songsResult = await pool.query(
      `SELECT 
         COUNT(*)::int AS total_songs,
         COALESCE(SUM(plays), 0)::int AS total_plays,
         COALESCE(SUM(earnings), 0)::float AS total_earnings
       FROM songs 
       WHERE artist_address = $1`,
      [address]
    );

    const stats = songsResult.rows[0] || { total_songs: 0, total_plays: 0, total_earnings: 0 };

    res.json({
      artistAddress: address,
      totalSongs: Number(stats.total_songs) || 0,
      totalPlays: Number(stats.total_plays) || 0,
      totalEarnings: Number(stats.total_earnings) || 0,
    });
  } catch (error) {
    console.error('Failed to fetch artist stats:', error);
    res.status(500).json({ error: 'Failed to fetch artist stats' });
  }
});

// Analytics: per-artist timeseries and top songs
app.get('/api/analytics/artist/:address', strictLimiter, validateQuery(analyticsQuerySchema), async (req, res) => {
  try {
    const { address } = req.params;
    const days = Math.min(90, Math.max(1, parseInt(String(req.query.days || '30'), 10)));
    const format = String(req.query.format || 'json').toLowerCase();

    const cacheKey = `analytics:artist:${address}:${days}`;
    try {
      const cached = await redis.get(cacheKey);
      if (cached && format === 'json') {
        res.setHeader('X-Cache', 'HIT');
        return res.json(JSON.parse(cached));
      }
    } catch {}

    const timeseries = await pool.query(
      `SELECT date_trunc('day', p.timestamp) AS day,
              COUNT(*)::int AS plays,
              COALESCE(SUM(p.amount),0)::float AS earnings
       FROM plays p
       JOIN songs s ON s.id = p.song_id
       WHERE s.artist_address = $1 AND p.timestamp >= NOW() - ($2::text || ' days')::interval
       GROUP BY day
       ORDER BY day ASC`,
      [address, String(days)]
    );

    const topSongs = await pool.query(
      `SELECT s.id, s.title, s.plays::int, s.earnings::float
       FROM songs s
       WHERE s.artist_address = $1
       ORDER BY s.plays DESC
       LIMIT 10`,
      [address]
    );

    const payload = { days, timeseries: timeseries.rows, topSongs: topSongs.rows };

    if (format === 'csv') {
      res.setHeader('Content-Type', 'text/csv');
      res.setHeader('Content-Disposition', `attachment; filename="analytics_${address}_${days}d.csv"`);
      const header = 'day,plays,earnings\n';
      const body = timeseries.rows.map((r:any) => `${new Date(r.day).toISOString()},${r.plays},${r.earnings}`).join('\n');
      return res.send(header + body);
    }

    const ttl = parseInt(String(process.env.ANALYTICS_CACHE_TTL || '60'), 10);
    try { await redis.setEx(cacheKey, Math.max(0, ttl), JSON.stringify(payload)); } catch {}
    res.json(payload);
  } catch (error) {
    console.error('Failed to fetch analytics:', error);
    res.status(500).json({ error: 'Failed to fetch analytics' });
  }
});

app.get('/api/analytics/top-songs', strictLimiter, validateQuery(topSongsQuerySchema), async (req, res) => {
  try {
    const limit = Math.min(50, Math.max(1, parseInt(String(req.query.limit || '10'), 10)));
    const format = String(req.query.format || 'json').toLowerCase();
    const rows = await pool.query(
      `SELECT id, title, artist, plays::int, earnings::float
       FROM songs
       ORDER BY plays DESC
       LIMIT $1`,
      [limit]
    );
    if (format === 'csv') {
      res.setHeader('Content-Type', 'text/csv');
      res.setHeader('Content-Disposition', `attachment; filename="top_songs_${limit}.csv"`);
      const header = 'id,title,artist,plays,earnings\n';
      const body = rows.rows.map((r:any) => [r.id, quoteCSV(r.title), quoteCSV(r.artist), r.plays ?? 0, r.earnings ?? 0].join(',')).join('\n');
      return res.send(header + body);
    }
    res.json(rows.rows);
  } catch (error) {
    console.error('Failed to fetch top songs:', error);
    res.status(500).json({ error: 'Failed to fetch top songs' });
  }
});

// ============================================================
// Start Server
// ============================================================

async function startServer() {
  try {
    ensureRequiredEnv();
    logEnvWarnings();
    // Connect to Redis
    await redis.connect();
    console.log('✓ Connected to Redis');

    // Initialize database
    await initDatabase();

    // Start Express server
    const server = app.listen(PORT, () => {
      serverStarted = true;
      console.log(`✓ API server running on port ${PORT}`);
      console.log(`   Health check: http://localhost:${PORT}/health`);
    });

    // Graceful shutdown
    const shutdown = async (signal: string) => {
      try {
        console.log(`\n[shutdown] Received ${signal}, closing server...`);
        await new Promise<void>((resolve) => server.close(() => resolve()));
      } catch (e) {
        console.error('[shutdown] Error closing server', e);
      }
      try {
        await redis.disconnect();
      } catch (e) {}
      try {
        await pool.end();
      } catch (e) {}
      process.exit(0);
    };
    process.on('SIGINT', () => shutdown('SIGINT'));
    process.on('SIGTERM', () => shutdown('SIGTERM'));
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

if (require.main === module) {
  void startServer();
} else if (process.env.NODE_ENV === 'test') {
  // Tests need the app without opening a network port, but only wire deps if envs are provided
  void (async () => {
    try {
      if (process.env.REDIS_URL) {
        await redis.connect();
      }
    } catch (e) {
      console.error('Redis connect (test) failed', e);
    }
    try {
      if (process.env.DATABASE_URL) {
        await initDatabase();
      }
    } catch (e) {
      console.error('DB init (test) failed', e);
    }
  })();
}

// Centralized error handler
// eslint-disable-next-line @typescript-eslint/no-unused-vars
app.use((err: any, req: any, res: any, _next: any) => {
  const rid = req?.requestId;
  console.error('Unhandled error', { requestId: rid, err });
  res.status(500).json({ error: 'Internal Server Error', requestId: rid });
});

export {
  app,
  pool,
  redis,
  SIGNATURE_TTL_MS,
  startServer,
  initDatabase,
};
