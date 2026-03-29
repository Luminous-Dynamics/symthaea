// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * WebSocket Support for Real-time Updates
 *
 * Events emitted:
 *   - play:new       - When a new play is recorded
 *   - song:new       - When a new song is registered
 *   - song:update    - When song stats change
 *   - earnings:update - Artist earnings update
 *
 * Client subscription:
 *   socket.emit('subscribe', { channel: 'artist', address: '0x...' })
 *   socket.emit('subscribe', { channel: 'song', songId: 'xxx' })
 *   socket.emit('unsubscribe', { channel: 'artist', address: '0x...' })
 */

import { Server as HttpServer } from 'http';
import { Server, Socket } from 'socket.io';
import { createAdapter } from '@socket.io/redis-adapter';
import { createClient } from 'redis';

let io: Server | null = null;

interface PlayEvent {
  songId: string;
  songTitle: string;
  artistAddress: string;
  listenerAddress: string;
  amount: number;
  paymentType: string;
  timestamp: string;
}

interface SongEvent {
  id: string;
  title: string;
  artist: string;
  artistAddress: string;
  genre: string;
  paymentModel: string;
}

interface EarningsEvent {
  artistAddress: string;
  songId: string;
  amount: number;
  totalEarnings: number;
  timestamp: string;
}

export async function initWebSocket(
  httpServer: HttpServer,
  redisUrl?: string
): Promise<Server> {
  io = new Server(httpServer, {
    cors: {
      origin: process.env.ALLOWED_ORIGINS?.split(',') || '*',
      methods: ['GET', 'POST'],
      credentials: true,
    },
    path: '/socket.io',
    transports: ['websocket', 'polling'],
    pingTimeout: 60000,
    pingInterval: 25000,
  });

  // Use Redis adapter for horizontal scaling (multiple API instances)
  if (redisUrl) {
    try {
      const pubClient = createClient({ url: redisUrl });
      const subClient = pubClient.duplicate();

      await Promise.all([pubClient.connect(), subClient.connect()]);

      io.adapter(createAdapter(pubClient, subClient));
      console.log('[websocket] Redis adapter connected for horizontal scaling');
    } catch (error) {
      console.warn('[websocket] Redis adapter failed, using in-memory adapter:', error);
    }
  }

  io.on('connection', (socket: Socket) => {
    console.log(`[websocket] Client connected: ${socket.id}`);

    // Handle subscriptions
    socket.on('subscribe', (data: { channel: string; address?: string; songId?: string }) => {
      const { channel, address, songId } = data;

      switch (channel) {
        case 'artist':
          if (address) {
            const room = `artist:${address.toLowerCase()}`;
            socket.join(room);
            console.log(`[websocket] ${socket.id} subscribed to ${room}`);
            socket.emit('subscribed', { channel, address });
          }
          break;

        case 'song':
          if (songId) {
            const room = `song:${songId}`;
            socket.join(room);
            console.log(`[websocket] ${socket.id} subscribed to ${room}`);
            socket.emit('subscribed', { channel, songId });
          }
          break;

        case 'global':
          socket.join('global');
          console.log(`[websocket] ${socket.id} subscribed to global updates`);
          socket.emit('subscribed', { channel: 'global' });
          break;

        default:
          socket.emit('error', { message: `Unknown channel: ${channel}` });
      }
    });

    socket.on('unsubscribe', (data: { channel: string; address?: string; songId?: string }) => {
      const { channel, address, songId } = data;

      switch (channel) {
        case 'artist':
          if (address) {
            socket.leave(`artist:${address.toLowerCase()}`);
            socket.emit('unsubscribed', { channel, address });
          }
          break;

        case 'song':
          if (songId) {
            socket.leave(`song:${songId}`);
            socket.emit('unsubscribed', { channel, songId });
          }
          break;

        case 'global':
          socket.leave('global');
          socket.emit('unsubscribed', { channel: 'global' });
          break;
      }
    });

    socket.on('disconnect', (reason) => {
      console.log(`[websocket] Client disconnected: ${socket.id} (${reason})`);
    });

    socket.on('error', (error) => {
      console.error(`[websocket] Socket error for ${socket.id}:`, error);
    });
  });

  console.log('[websocket] WebSocket server initialized');
  return io;
}

/**
 * Emit a new play event
 */
export function emitPlay(play: PlayEvent): void {
  if (!io) return;

  // Emit to global subscribers
  io.to('global').emit('play:new', play);

  // Emit to artist subscribers
  io.to(`artist:${play.artistAddress.toLowerCase()}`).emit('play:new', play);

  // Emit to song subscribers
  io.to(`song:${play.songId}`).emit('play:new', play);
}

/**
 * Emit a new song event
 */
export function emitNewSong(song: SongEvent): void {
  if (!io) return;

  // Emit to global subscribers
  io.to('global').emit('song:new', song);

  // Emit to artist subscribers
  io.to(`artist:${song.artistAddress.toLowerCase()}`).emit('song:new', song);
}

/**
 * Emit song stats update
 */
export function emitSongUpdate(songId: string, stats: { plays: number; earnings: number }): void {
  if (!io) return;

  io.to(`song:${songId}`).emit('song:update', { songId, ...stats });
  io.to('global').emit('song:update', { songId, ...stats });
}

/**
 * Emit earnings update for an artist
 */
export function emitEarningsUpdate(event: EarningsEvent): void {
  if (!io) return;

  io.to(`artist:${event.artistAddress.toLowerCase()}`).emit('earnings:update', event);
}

/**
 * Get WebSocket server instance
 */
export function getIO(): Server | null {
  return io;
}

/**
 * Get connected client count
 */
export async function getConnectedCount(): Promise<number> {
  if (!io) return 0;
  const sockets = await io.fetchSockets();
  return sockets.length;
}

/**
 * Broadcast to all connected clients
 */
export function broadcast(event: string, data: unknown): void {
  if (!io) return;
  io.emit(event, data);
}

export { io };
