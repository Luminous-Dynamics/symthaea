// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Events Index
 *
 * Central export for the event system.
 */

export {
  TypedEventEmitter,
  getEventEmitter,
  resetEventEmitter,
} from './emitter';

export type { AppEvents, EventHandler } from './emitter';

export {
  registerCacheHandlers,
  registerLoggingHandlers,
  registerMilestoneHandlers,
  registerWebSocketHandlers,
  registerAllHandlers,
} from './handlers';
