// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { z } from 'zod';

// Ethereum address validation (0x + 40 hex characters)
const ethereumAddress = z.string().regex(
  /^0x[a-fA-F0-9]{40}$/,
  'Invalid Ethereum address format'
);

// IPFS CID validation (CIDv0 starts with Qm, CIDv1 starts with b)
const ipfsHash = z.string().regex(
  /^(Qm[1-9A-HJ-NP-Za-km-z]{44}|b[a-z2-7]{58,})$/,
  'Invalid IPFS CID format'
);

// Sanitize string to prevent XSS - strips HTML tags and control characters
const sanitizedString = (maxLen: number) => z.string()
  .max(maxLen)
  .transform((s) => s.replace(/<[^>]*>/g, '').replace(/[\x00-\x08\x0B\x0C\x0E-\x1F]/g, ''));

const timestamp = z.preprocess((v) => {
  if (typeof v === 'string' || typeof v === 'number') return Number(v);
  return v;
}, z.number().int().nonnegative().optional());

// Allowed payment models - strict enum validation
const paymentModelEnum = z.enum([
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

// Allowed genres for filtering
const genreEnum = z.enum([
  'Electronic',
  'Rock',
  'Classical',
  'Hip-Hop',
  'Jazz',
  'Ambient',
  'Pop',
  'R&B',
  'Country',
  'Folk',
  'Metal',
  'Punk',
  'Reggae',
  'Blues',
  'Soul',
  'World',
  'Other',
]).or(z.string().min(1).max(64)); // Allow custom genres too

export const songSchema = z.object({
  id: z.string().min(1).max(256).regex(/^[a-zA-Z0-9_-]+$/, 'ID must be alphanumeric with dashes/underscores'),
  title: sanitizedString(256).pipe(z.string().min(1)),
  artist: sanitizedString(256).pipe(z.string().min(1)),
  artistAddress: ethereumAddress,
  genre: sanitizedString(128).pipe(z.string().min(1)),
  description: sanitizedString(2000).optional(),
  ipfsHash: ipfsHash,
  paymentModel: paymentModelEnum,
  coverArt: z.string().url().max(2048).optional().nullable(),
  audioUrl: z.string().url().max(2048).optional().nullable(),
  claimStreamId: z.string().max(512).optional().nullable(),
  nonce: z.string().max(128).regex(/^[a-zA-Z0-9_-]*$/).optional(),
  timestamp,
  signer: ethereumAddress.optional(),
  signature: z.string().regex(/^0x[a-fA-F0-9]+$/, 'Invalid signature format').optional(),
});

// Payment types enum
const paymentTypeEnum = z.enum(['stream', 'download', 'tip', 'patronage', 'nft_access']);

export const playSchema = z.object({
  listenerAddress: ethereumAddress.or(z.literal('anonymous')),
  amount: z.preprocess(
    (v) => (typeof v === 'string' || typeof v === 'number' ? Number(v) : v),
    z.number().nonnegative().max(1000000) // Max 1M to prevent overflow
  ),
  paymentType: paymentTypeEnum,
  nonce: z.string().max(128).regex(/^[a-zA-Z0-9_-]*$/).optional(),
  signer: ethereumAddress.optional(),
  timestamp,
  signature: z.string().regex(/^0x[a-fA-F0-9]+$/, 'Invalid signature format').optional(),
});

export const claimSchema = z.object({
  songId: z.string().min(1).max(256).regex(/^[a-zA-Z0-9_-]+$/),
  title: sanitizedString(256).pipe(z.string().min(1)),
  artist: sanitizedString(256).pipe(z.string().min(1)),
  ipfsHash: ipfsHash,
  artistAddress: ethereumAddress,
  epistemicTier: z.number().int().min(0).max(10).optional(),
  networkTier: z.number().int().min(0).max(10).optional(),
  memoryTier: z.number().int().min(0).max(10).optional(),
  nonce: z.string().max(128).regex(/^[a-zA-Z0-9_-]*$/).optional(),
  signer: ethereumAddress.optional(),
  signature: z.string().regex(/^0x[a-fA-F0-9]+$/, 'Invalid signature format').optional(),
  timestamp,
});

export const analyticsQuerySchema = z.object({
  days: z.preprocess((v) => (v === undefined ? 30 : Number(v)), z.number().int().min(1).max(90)).optional(),
  format: z.enum(['json', 'csv']).optional(),
});

// Search query sanitization - removes special SQL/regex characters
const searchQuery = z.string()
  .max(256)
  .transform((s) => s.replace(/[%_'"\\;]/g, '').trim())
  .optional();

// Allowed sort columns - whitelist
const sortColumn = z.enum(['created_at', 'plays', 'earnings', 'title', 'artist']).optional();

export const songsQuerySchema = z.object({
  q: searchQuery,
  genre: z.string().max(64).optional(),
  model: paymentModelEnum.optional().or(z.literal('all')),
  limit: z.preprocess((v) => (v === undefined ? 50 : Number(v)), z.number().int().min(1).max(100)).optional(),
  offset: z.preprocess((v) => (v === undefined ? 0 : Number(v)), z.number().int().min(0).max(10000)).optional(),
  sort: sortColumn,
  order: z.enum(['asc', 'desc']).optional(),
  cursor: z.string().max(256).regex(/^[a-zA-Z0-9+/=]*$/).optional(), // Base64 format
  format: z.enum(['json', 'csv']).optional(),
});

export const artistSongsQuerySchema = z.object({
  limit: z.preprocess((v) => (v === undefined ? 50 : Number(v)), z.number().int().min(1).max(200)).optional(),
  offset: z.preprocess((v) => (v === undefined ? 0 : Number(v)), z.number().int().min(0)).optional(),
  order: z.enum(['asc', 'desc']).optional(),
  format: z.enum(['json', 'csv']).optional(),
  all: z.preprocess((v) => String(v ?? 'false').toLowerCase() === 'true', z.boolean()).optional(),
});

export const topSongsQuerySchema = z.object({
  limit: z.preprocess((v) => (v === undefined ? 10 : Number(v)), z.number().int().min(1).max(50)).optional(),
  format: z.enum(['json', 'csv']).optional(),
});

export const playsQuerySchema = z.object({
  limit: z.preprocess((v) => (v === undefined ? 100 : Number(v)), z.number().int().min(1).max(1000)).optional(),
  format: z.enum(['json', 'csv']).optional(),
});

export const healthDetailsQuerySchema = z.object({
  client_ts: z.preprocess((v) => (v === undefined ? undefined : Number(v)), z.number().int().optional()).optional(),
});

function toIssues(issues: z.ZodIssue[]) {
  return issues.map((i) => ({
    path: i.path.join('.'),
    message: i.message,
  }));
}

export function validateBody(schema: z.ZodTypeAny) {
  return (req: any, res: any, next: any) => {
    const result = schema.safeParse(req.body);
    if (!result.success) {
      return res.status(400).json({ error: 'invalid_request', issues: toIssues(result.error.issues) });
    }
    req.body = result.data;
    return next();
  };
}

export function validateQuery(schema: z.ZodTypeAny) {
  return (req: any, res: any, next: any) => {
    const result = schema.safeParse(req.query);
    if (!result.success) {
      return res.status(400).json({ error: 'invalid_request', issues: toIssues(result.error.issues) });
    }
    req.query = result.data;
    return next();
  };
}
