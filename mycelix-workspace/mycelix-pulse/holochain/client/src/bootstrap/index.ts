// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Mail - Bootstrap Module
 *
 * Unified client initialization and service container.
 */

export {
  MycelixClient,
  createMycelixClient,
  getMycelixClient,
  useMycelixClient,
} from './MycelixClient';

export type {
  MycelixClientConfig,
  ServiceContainer,
} from './MycelixClient';

// Re-export commonly used types
export type { SignalHub } from '../SignalHub';
