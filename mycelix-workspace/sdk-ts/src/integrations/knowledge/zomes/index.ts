// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Knowledge Zome Clients
 *
 * Individual clients for each knowledge domain.
 *
 * @module @mycelix/sdk/integrations/knowledge/zomes
 */

export { ClaimsClient } from './claims';
export type { ClaimsClientConfig } from './claims';

export { GraphClient } from './graph';
export type { GraphClientConfig } from './graph';

export { FactCheckClient } from './factcheck';
export type { FactCheckClientConfig } from './factcheck';
