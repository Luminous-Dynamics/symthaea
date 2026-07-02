// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix ERP SDK
 *
 * TypeScript client for interacting with the Mycelix ERP API.
 *
 * Modules included:
 * - SCM (Supply Chain Management) - Verifiable provenance tracking
 * - FIN (Finance & Accounting) - Double-entry bookkeeping
 * - Auth (Authentication) - JWT-based authentication
 *
 * @example
 * ```typescript
 * import { MycelixClient } from '@mycelix/erp-sdk';
 *
 * const client = new MycelixClient({
 *   baseUrl: 'https://api.mycelix.net',
 *   apiKey: 'your-api-key', // Optional
 * });
 *
 * // Health check
 * const health = await client.health();
 *
 * // Supply Chain: Track products
 * const event = client.scm.createProducedEvent({
 *   issuer: 'did:mycelix:org:my-company',
 *   productId: 'SKU-123',
 *   batchId: 'BATCH-001',
 *   quantity: 1000,
 *   unit: 'kg',
 *   facility: { id: 'FAC-001', name: 'My Factory' },
 * });
 * const result = await client.scm.ingestEvent(event);
 *
 * // Finance: Create and send invoice
 * const invoice = await client.fin.createInvoice({
 *   customer_id: 'CUST-001',
 *   due_date: '2025-02-15',
 *   lines: [
 *     { description: 'Product A', quantity: '10', unit_price: '99.00' },
 *   ],
 * });
 * await client.fin.sendInvoice(invoice.id);
 *
 * // Auth: Login
 * await client.auth.login({
 *   email: 'user@example.com',
 *   password: 'password',
 * });
 * ```
 *
 * @packageDocumentation
 */

// Main client
export { MycelixClient, SupplyChainClient, SupplyChainModule } from './client';
export type { ClientConfig, DetailedHealthResponse, ComponentHealth } from './client';

// Supply Chain types
export * from './types';

// Finance module
export { FinanceClient } from './finance-client';
export * from './finance-types';

// Auth module
export { AuthClient } from './auth';
export type {
  LoginRequest,
  LoginResponse,
  RefreshRequest,
  User,
  UserRole,
  Tenant,
  TenantSettings,
  RegisterRequest,
  RegisterResponse,
  PasswordResetRequest,
  PasswordResetConfirm,
  ChangePasswordRequest,
  ApiKeyResponse,
  CreateApiKeyRequest,
} from './auth';
