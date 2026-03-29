// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Common Validation Schemas
 *
 * Reusable Zod schemas for common types across the API.
 */

import { z } from 'zod';

/**
 * Ethereum wallet address validation
 */
export const walletAddressSchema = z
  .string()
  .regex(/^0x[a-fA-F0-9]{40}$/, 'Invalid Ethereum address format')
  .transform((val) => val.toLowerCase() as `0x${string}`);

/**
 * UUID validation
 */
export const uuidSchema = z
  .string()
  .uuid('Invalid UUID format');

/**
 * IPFS hash validation (CIDv0 or CIDv1)
 */
export const ipfsHashSchema = z
  .string()
  .regex(/^(Qm[1-9A-HJ-NP-Za-km-z]{44}|b[A-Za-z2-7]{58})$/, 'Invalid IPFS hash format');

/**
 * Pagination query parameters
 */
export const paginationSchema = z.object({
  limit: z
    .string()
    .optional()
    .transform((val) => (val ? parseInt(val, 10) : 20))
    .pipe(z.number().min(1).max(100)),
  offset: z
    .string()
    .optional()
    .transform((val) => (val ? parseInt(val, 10) : 0))
    .pipe(z.number().min(0)),
  cursor: z.string().optional(),
});

/**
 * Sort query parameters
 */
export const sortSchema = z.object({
  sort: z
    .string()
    .optional()
    .default('created_at'),
  order: z
    .enum(['asc', 'desc'])
    .optional()
    .default('desc'),
});

/**
 * Search query parameter
 */
export const searchSchema = z.object({
  q: z.string().min(1).max(200).optional(),
});

/**
 * Date range parameters
 */
export const dateRangeSchema = z.object({
  start_date: z
    .string()
    .datetime()
    .optional()
    .transform((val) => (val ? new Date(val) : undefined)),
  end_date: z
    .string()
    .datetime()
    .optional()
    .transform((val) => (val ? new Date(val) : undefined)),
}).refine(
  (data) => {
    if (data.start_date && data.end_date) {
      return data.start_date <= data.end_date;
    }
    return true;
  },
  { message: 'start_date must be before or equal to end_date' }
);

/**
 * Numeric string (for blockchain amounts)
 */
export const numericStringSchema = z
  .string()
  .regex(/^\d+(\.\d+)?$/, 'Must be a valid numeric string')
  .refine((val) => parseFloat(val) >= 0, 'Must be non-negative');

/**
 * Payment model enum
 */
export const paymentModelSchema = z.enum(['per_play', 'subscription', 'tip']);

/**
 * Transaction status enum
 */
export const transactionStatusSchema = z.enum(['pending', 'confirmed', 'failed']);

/**
 * Genre validation (alphanumeric with spaces/hyphens)
 */
export const genreSchema = z
  .string()
  .min(2)
  .max(50)
  .regex(/^[a-zA-Z0-9\s\-]+$/, 'Genre must contain only letters, numbers, spaces, and hyphens')
  .transform((val) => val.toLowerCase());

/**
 * Safe string (no special characters that could cause issues)
 */
export const safeStringSchema = z
  .string()
  .max(1000)
  .transform((val) => val.trim());

/**
 * Title validation
 */
export const titleSchema = z
  .string()
  .min(1, 'Title is required')
  .max(200, 'Title must be 200 characters or less')
  .transform((val) => val.trim());

/**
 * Description validation
 */
export const descriptionSchema = z
  .string()
  .max(5000, 'Description must be 5000 characters or less')
  .optional()
  .transform((val) => val?.trim());

/**
 * URL validation
 */
export const urlSchema = z
  .string()
  .url('Invalid URL format')
  .optional();

/**
 * ID parameter (from URL)
 */
export const idParamSchema = z.object({
  id: uuidSchema,
});

/**
 * Address parameter (from URL)
 */
export const addressParamSchema = z.object({
  address: walletAddressSchema,
});

// Type exports
export type WalletAddress = z.infer<typeof walletAddressSchema>;
export type PaginationParams = z.infer<typeof paginationSchema>;
export type SortParams = z.infer<typeof sortSchema>;
export type PaymentModel = z.infer<typeof paymentModelSchema>;
export type TransactionStatus = z.infer<typeof transactionStatusSchema>;
