// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Graceful Shutdown Handler
 *
 * Handles SIGTERM/SIGINT signals for clean shutdown:
 * - Stops accepting new connections
 * - Waits for in-flight requests to complete
 * - Closes database connections
 * - Closes Redis connections
 * - Flushes logs and metrics
 */

import { Server } from 'http';
import { Pool } from 'pg';
import Redis from 'ioredis';

/**
 * Shutdown configuration
 */
export interface ShutdownConfig {
  /** Timeout before force shutdown (ms) */
  timeout: number;
  /** Signals to handle */
  signals: NodeJS.Signals[];
  /** Callback before shutdown starts */
  onShutdownStart?: () => void;
  /** Callback after shutdown completes */
  onShutdownComplete?: () => void;
}

const DEFAULT_CONFIG: ShutdownConfig = {
  timeout: 30000, // 30 seconds
  signals: ['SIGTERM', 'SIGINT'],
};

/**
 * Shutdown handler class
 */
export class GracefulShutdown {
  private config: ShutdownConfig;
  private isShuttingDown = false;
  private connections: Set<any> = new Set();

  // Resources to clean up
  private server: Server | null = null;
  private pool: Pool | null = null;
  private redis: Redis | null = null;
  private cleanupCallbacks: (() => Promise<void>)[] = [];

  constructor(config: Partial<ShutdownConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Set HTTP server to track connections
   */
  setServer(server: Server): void {
    this.server = server;

    // Track connections
    server.on('connection', (socket) => {
      this.connections.add(socket);
      socket.on('close', () => {
        this.connections.delete(socket);
      });
    });
  }

  /**
   * Set database pool
   */
  setPool(pool: Pool): void {
    this.pool = pool;
  }

  /**
   * Set Redis client
   */
  setRedis(redis: Redis): void {
    this.redis = redis;
  }

  /**
   * Register cleanup callback
   */
  onCleanup(callback: () => Promise<void>): void {
    this.cleanupCallbacks.push(callback);
  }

  /**
   * Start listening for shutdown signals
   */
  listen(): void {
    for (const signal of this.config.signals) {
      process.on(signal, () => this.shutdown(signal));
    }

    // Handle uncaught exceptions
    process.on('uncaughtException', (error) => {
      console.error('Uncaught exception:', error);
      this.shutdown('uncaughtException');
    });

    // Handle unhandled rejections
    process.on('unhandledRejection', (reason) => {
      console.error('Unhandled rejection:', reason);
      // Don't shutdown on unhandled rejection, just log
    });
  }

  /**
   * Initiate shutdown
   */
  async shutdown(signal: string): Promise<void> {
    if (this.isShuttingDown) {
      console.log('Shutdown already in progress...');
      return;
    }

    this.isShuttingDown = true;
    console.log(`\nReceived ${signal}, starting graceful shutdown...`);

    this.config.onShutdownStart?.();

    // Set force shutdown timeout
    const forceShutdownTimeout = setTimeout(() => {
      console.error('Shutdown timeout exceeded, forcing exit');
      process.exit(1);
    }, this.config.timeout);

    try {
      // Phase 1: Stop accepting new connections
      if (this.server) {
        console.log('Stopping HTTP server...');
        await this.closeServer();
      }

      // Phase 2: Run custom cleanup callbacks
      console.log('Running cleanup callbacks...');
      for (const callback of this.cleanupCallbacks) {
        try {
          await callback();
        } catch (error) {
          console.error('Cleanup callback error:', error);
        }
      }

      // Phase 3: Close database connections
      if (this.pool) {
        console.log('Closing database connections...');
        await this.pool.end();
      }

      // Phase 4: Close Redis connections
      if (this.redis) {
        console.log('Closing Redis connection...');
        await this.redis.quit();
      }

      // Clear the timeout
      clearTimeout(forceShutdownTimeout);

      console.log('Graceful shutdown complete');
      this.config.onShutdownComplete?.();

      process.exit(0);
    } catch (error) {
      console.error('Error during shutdown:', error);
      clearTimeout(forceShutdownTimeout);
      process.exit(1);
    }
  }

  /**
   * Close HTTP server and wait for connections to drain
   */
  private closeServer(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.server) {
        resolve();
        return;
      }

      // Stop accepting new connections
      this.server.close((err) => {
        if (err) {
          reject(err);
          return;
        }
        resolve();
      });

      // Close idle connections immediately
      for (const socket of this.connections) {
        // Destroy idle sockets
        if (!(socket as any).server) {
          socket.destroy();
          this.connections.delete(socket);
        }
      }

      // Set deadline for active connections
      const connectionDeadline = setTimeout(() => {
        console.log(`Forcing ${this.connections.size} connections to close`);
        for (const socket of this.connections) {
          socket.destroy();
        }
      }, 10000); // 10 second deadline for connections

      // If all connections close naturally, clear the deadline
      const checkConnections = setInterval(() => {
        if (this.connections.size === 0) {
          clearInterval(checkConnections);
          clearTimeout(connectionDeadline);
        }
      }, 100);
    });
  }

  /**
   * Check if shutdown is in progress
   */
  get shuttingDown(): boolean {
    return this.isShuttingDown;
  }
}

// Singleton instance
let _shutdown: GracefulShutdown | null = null;

/**
 * Get or create shutdown handler
 */
export function getShutdownHandler(config?: Partial<ShutdownConfig>): GracefulShutdown {
  if (!_shutdown) {
    _shutdown = new GracefulShutdown(config);
  }
  return _shutdown;
}

/**
 * Express middleware to reject requests during shutdown
 */
export function shutdownMiddleware(handler: GracefulShutdown) {
  return (req: any, res: any, next: any) => {
    if (handler.shuttingDown) {
      res.setHeader('Connection', 'close');
      res.status(503).json({
        success: false,
        error: {
          code: 'SERVICE_UNAVAILABLE',
          message: 'Server is shutting down',
        },
      });
      return;
    }
    next();
  };
}

export default GracefulShutdown;
