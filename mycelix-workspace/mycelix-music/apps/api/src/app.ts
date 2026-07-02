// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Application Bootstrap
 *
 * Clean startup sequence that wires all components together:
 * - Configuration validation
 * - Database connection
 * - Redis connection
 * - Container initialization
 * - Event handlers
 * - Background jobs
 * - HTTP server
 * - Graceful shutdown
 */

import express, { Express } from 'express';
import { Server } from 'http';
import { Pool } from 'pg';
import Redis from 'ioredis';

import { getConfig, Config } from './config';
import { Container, createContainerFromEnv } from './container';
import { HealthChecker, createHealthRouter, getHealthChecker } from './health';
import { GracefulShutdown, getShutdownHandler, shutdownMiddleware } from './shutdown';
import { JobScheduler, getScheduler, registerAnalyticsJobs } from './jobs';
import { getEventEmitter, registerAllHandlers } from './events';
import { setupMiddlewarePipeline, setupErrorHandling } from './middleware';
import { requestLogger } from './middleware/request-logger';
import { versionMiddleware } from './versioning';

// Routes
import v2Routes from './routes/v2';

/**
 * Application instance
 */
export interface Application {
  app: Express;
  server: Server;
  container: Container;
  config: Config;
  shutdown: () => Promise<void>;
}

/**
 * Bootstrap options
 */
export interface BootstrapOptions {
  /** Skip background jobs */
  skipJobs?: boolean;
  /** Skip event handlers */
  skipEvents?: boolean;
  /** Custom port override */
  port?: number;
}

/**
 * Create and configure Express app
 */
function createExpressApp(config: Config, container: Container): Express {
  const app = express();

  // Get shutdown handler for middleware
  const shutdown = getShutdownHandler();

  // Shutdown check middleware (reject requests during shutdown)
  app.use(shutdownMiddleware(shutdown));

  // Setup standard middleware pipeline
  setupMiddlewarePipeline(app, {
    cors: {
      origin: config.security.corsOrigin,
      credentials: true,
    },
    bodyLimit: '10mb',
    trustProxy: true,
    enableAuditLog: config.features.auditLog,
    enableHelmet: true,
  });

  // Request logging
  app.use(requestLogger({
    logBody: config.isDev,
    logResponseBody: false,
    format: config.logging.format,
    skipRoutes: ['/health', '/metrics', '/favicon.ico'],
  }));

  // API versioning
  app.use(versionMiddleware);

  // Health check routes
  const healthChecker = getHealthChecker(process.env.npm_package_version || '1.0.0');
  app.use('/health', createHealthRouter(healthChecker));

  // API v2 routes
  app.use('/api/v2', v2Routes);

  // OpenAPI spec endpoint
  if (config.features.swagger) {
    app.get('/api/openapi.json', async (req, res) => {
      const { buildOpenAPISpec } = await import('./openapi');
      res.json(buildOpenAPISpec());
    });
  }

  // Root endpoint
  app.get('/', (req, res) => {
    res.json({
      name: 'Mycelix Music API',
      version: process.env.npm_package_version || '1.0.0',
      documentation: '/api/openapi.json',
      health: '/health',
    });
  });

  // Setup error handling (must be last)
  setupErrorHandling(app);

  return app;
}

/**
 * Bootstrap the application
 */
export async function bootstrap(options: BootstrapOptions = {}): Promise<Application> {
  console.log('\n🎵 Starting Mycelix Music API...\n');

  // Step 1: Load and validate configuration
  console.log('📋 Loading configuration...');
  const config = getConfig();
  console.log(`   Environment: ${config.env}`);
  console.log(`   Port: ${options.port || config.server.port}`);

  // Step 2: Initialize container (database, redis, services)
  console.log('🔌 Initializing services...');
  const container = createContainerFromEnv();
  await container.initialize();
  console.log('   ✓ Database connected');
  if (config.redis.enabled) {
    console.log('   ✓ Redis connected');
  }

  // Step 3: Setup health checker
  console.log('💓 Setting up health checks...');
  const healthChecker = getHealthChecker();
  healthChecker.setPool(container.pool);
  if (container.redis) {
    healthChecker.setRedis(container.redis);
  }

  // Step 4: Setup event handlers
  if (!options.skipEvents) {
    console.log('📡 Registering event handlers...');
    const events = getEventEmitter();
    registerAllHandlers(events, {
      cache: container.cache,
    });
  }

  // Step 5: Setup background jobs
  let scheduler: JobScheduler | null = null;
  if (!options.skipJobs && config.env !== 'test') {
    console.log('⏰ Starting background jobs...');
    scheduler = getScheduler();
    registerAnalyticsJobs(scheduler, container.pool);
    scheduler.start();
  }

  // Step 6: Create Express app
  console.log('🌐 Creating HTTP server...');
  const app = createExpressApp(config, container);

  // Step 7: Start HTTP server
  const port = options.port || config.server.port;
  const server = app.listen(port, config.server.host, () => {
    console.log(`\n✨ Server running at http://${config.server.host}:${port}`);
    console.log(`   Health: http://${config.server.host}:${port}/health`);
    console.log(`   API: http://${config.server.host}:${port}/api/v2`);
    if (config.features.swagger) {
      console.log(`   OpenAPI: http://${config.server.host}:${port}/api/openapi.json`);
    }
    console.log('');
  });

  // Step 8: Setup graceful shutdown
  console.log('🛡️  Setting up graceful shutdown...');
  const shutdown = getShutdownHandler({
    timeout: 30000,
    onShutdownStart: () => {
      console.log('\n🛑 Shutdown initiated...');
      // Emit shutdown event
      getEventEmitter().emit('system:shutdown', {
        timestamp: new Date(),
        reason: 'SIGTERM/SIGINT received',
      });
    },
    onShutdownComplete: () => {
      console.log('👋 Goodbye!\n');
    },
  });

  shutdown.setServer(server);
  shutdown.setPool(container.pool);
  if (container.redis) {
    shutdown.setRedis(container.redis);
  }

  // Register cleanup callbacks
  shutdown.onCleanup(async () => {
    if (scheduler) {
      scheduler.stop();
    }
  });

  shutdown.listen();

  // Mark startup complete
  healthChecker.setStartupComplete();

  // Emit startup event
  getEventEmitter().emit('system:startup', { timestamp: new Date() });

  // Return application instance
  return {
    app,
    server,
    container,
    config,
    shutdown: async () => {
      await shutdown.shutdown('manual');
    },
  };
}

/**
 * Quick start for development
 */
export async function start(): Promise<void> {
  try {
    await bootstrap();
  } catch (error) {
    console.error('❌ Failed to start application:', error);
    process.exit(1);
  }
}

// Start if run directly
if (require.main === module) {
  start();
}

export default bootstrap;
