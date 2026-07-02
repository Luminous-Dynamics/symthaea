// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Event Handlers
 *
 * Registers handlers for application events.
 * Keeps event handling logic centralized.
 */

import { TypedEventEmitter, AppEvents } from './emitter';
import { CacheService, CacheTags } from '../services/cache.service';

/**
 * Register cache invalidation handlers
 */
export function registerCacheHandlers(
  events: TypedEventEmitter,
  cache: CacheService
): void {
  // Invalidate song cache on changes
  events.on('song:created', async ({ songId }) => {
    await cache.invalidateTag(CacheTags.SONGS);
    console.log(`[Events] Cache invalidated for new song: ${songId}`);
  });

  events.on('song:updated', async ({ songId }) => {
    await cache.delete(`song:${songId}`);
    await cache.invalidateTag(CacheTags.SONGS);
    console.log(`[Events] Cache invalidated for updated song: ${songId}`);
  });

  events.on('song:deleted', async ({ songId }) => {
    await cache.delete(`song:${songId}`);
    await cache.invalidateTag(CacheTags.SONGS);
    console.log(`[Events] Cache invalidated for deleted song: ${songId}`);
  });

  // Invalidate analytics cache on plays
  events.on('play:confirmed', async ({ songId }) => {
    await cache.delete(`song:${songId}`);
    await cache.invalidateTag(CacheTags.ANALYTICS);
    await cache.invalidateTag(CacheTags.PLAYS);
  });

  console.log('[Events] Cache handlers registered');
}

/**
 * Register logging handlers
 */
export function registerLoggingHandlers(events: TypedEventEmitter): void {
  // Log important events
  events.on('song:created', ({ songId, title, artistAddress }) => {
    console.log(`[Events] Song created: ${title} (${songId}) by ${artistAddress}`);
  });

  events.on('play:recorded', ({ playId, songId, amount }) => {
    console.log(`[Events] Play recorded: ${playId} for song ${songId}, amount: ${amount}`);
  });

  events.on('play:confirmed', ({ playId, transactionHash }) => {
    console.log(`[Events] Play confirmed: ${playId}, tx: ${transactionHash}`);
  });

  events.on('play:failed', ({ playId, reason }) => {
    console.warn(`[Events] Play failed: ${playId}, reason: ${reason}`);
  });

  events.on('system:error', ({ error, context }) => {
    console.error('[Events] System error:', error.message, context);
  });

  console.log('[Events] Logging handlers registered');
}

/**
 * Register milestone handlers
 */
export function registerMilestoneHandlers(events: TypedEventEmitter): void {
  const playMilestones = [100, 1000, 10000, 100000, 1000000];
  const earningsMilestones = ['1', '10', '100', '1000', '10000'];

  // Check for play milestones
  events.on('play:confirmed', async ({ songId }) => {
    // This would normally query the database
    // For now, just demonstrate the pattern
  });

  console.log('[Events] Milestone handlers registered');
}

/**
 * Register WebSocket broadcast handlers
 */
export function registerWebSocketHandlers(
  events: TypedEventEmitter,
  broadcast: (event: string, data: unknown) => void
): void {
  events.on('song:created', (payload) => {
    broadcast('song:new', payload);
  });

  events.on('play:recorded', (payload) => {
    broadcast('play:new', {
      songId: payload.songId,
      amount: payload.amount,
    });
  });

  events.on('play:confirmed', (payload) => {
    broadcast('play:confirmed', {
      songId: payload.songId,
      playId: payload.playId,
    });
  });

  console.log('[Events] WebSocket handlers registered');
}

/**
 * Register all event handlers
 */
export function registerAllHandlers(
  events: TypedEventEmitter,
  options: {
    cache?: CacheService;
    broadcast?: (event: string, data: unknown) => void;
  } = {}
): void {
  // Always register logging
  registerLoggingHandlers(events);

  // Register cache handlers if cache provided
  if (options.cache) {
    registerCacheHandlers(events, options.cache);
  }

  // Register milestone handlers
  registerMilestoneHandlers(events);

  // Register WebSocket handlers if broadcast function provided
  if (options.broadcast) {
    registerWebSocketHandlers(events, options.broadcast);
  }

  console.log('[Events] All handlers registered');
}
