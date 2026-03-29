// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Knowledge - GraphQL Server
 *
 * Apollo Server with Holochain integration and real-time subscriptions
 */

import { ApolloServer } from '@apollo/server';
import { expressMiddleware } from '@apollo/server/express4';
import { ApolloServerPluginDrainHttpServer } from '@apollo/server/plugin/drainHttpServer';
import { makeExecutableSchema } from '@graphql-tools/schema';
import { WebSocketServer } from 'ws';
import { useServer } from 'graphql-ws/lib/use/ws';
import express from 'express';
import { createServer } from 'http';
import cors from 'cors';
import { readFileSync } from 'fs';
import { join } from 'path';
import { PubSub } from 'graphql-subscriptions';
import { KnowledgeClient } from '../client/src';
import { resolvers } from './resolvers';

// ============================================================================
// Configuration
// ============================================================================

const PORT = process.env.PORT || 4000;
const HOLOCHAIN_URL = process.env.HOLOCHAIN_URL || 'ws://localhost:8888';
const HOLOCHAIN_APP_ID = process.env.HOLOCHAIN_APP_ID || 'mycelix-knowledge';

// ============================================================================
// Schema Loading
// ============================================================================

const typeDefs = readFileSync(join(__dirname, 'schema.graphql'), 'utf-8');

// ============================================================================
// Context Factory
// ============================================================================

interface ServerContext {
  client: KnowledgeClient;
  pubsub: PubSub;
}

const pubsub = new PubSub();

async function createContext(): Promise<ServerContext> {
  // Initialize Holochain client
  const client = new KnowledgeClient({
    url: HOLOCHAIN_URL,
    appId: HOLOCHAIN_APP_ID,
  });

  await client.connect();

  return {
    client,
    pubsub,
  };
}

// ============================================================================
// Server Setup
// ============================================================================

async function startServer() {
  // Create Express app and HTTP server
  const app = express();
  const httpServer = createServer(app);

  // Create executable schema
  const schema = makeExecutableSchema({ typeDefs, resolvers });

  // Create WebSocket server for subscriptions
  const wsServer = new WebSocketServer({
    server: httpServer,
    path: '/graphql',
  });

  // Setup WebSocket subscription handler
  const serverCleanup = useServer(
    {
      schema,
      context: async () => createContext(),
      onConnect: async () => {
        console.log('Client connected to subscriptions');
      },
      onDisconnect: async () => {
        console.log('Client disconnected from subscriptions');
      },
    },
    wsServer
  );

  // Create Apollo Server
  const server = new ApolloServer<ServerContext>({
    schema,
    plugins: [
      // Proper shutdown for HTTP server
      ApolloServerPluginDrainHttpServer({ httpServer }),
      // Proper shutdown for WebSocket server
      {
        async serverWillStart() {
          return {
            async drainServer() {
              await serverCleanup.dispose();
            },
          };
        },
      },
    ],
  });

  // Start Apollo Server
  await server.start();

  // Apply middleware
  app.use(
    '/graphql',
    cors<cors.CorsRequest>({
      origin: process.env.CORS_ORIGIN || '*',
    }),
    express.json(),
    expressMiddleware(server, {
      context: createContext,
    })
  );

  // Health check endpoint
  app.get('/health', (_, res) => {
    res.json({ status: 'healthy', timestamp: new Date().toISOString() });
  });

  // GraphQL Playground redirect
  app.get('/', (_, res) => {
    res.redirect('/graphql');
  });

  // Start listening
  await new Promise<void>((resolve) => httpServer.listen({ port: PORT }, resolve));

  console.log(`
╔═══════════════════════════════════════════════════════════════╗
║           Mycelix Knowledge GraphQL Server                     ║
╠═══════════════════════════════════════════════════════════════╣
║  GraphQL Endpoint:    http://localhost:${PORT}/graphql            ║
║  Subscriptions:       ws://localhost:${PORT}/graphql              ║
║  Health Check:        http://localhost:${PORT}/health             ║
║  Holochain:           ${HOLOCHAIN_URL.padEnd(35)}║
╚═══════════════════════════════════════════════════════════════╝
  `);
}

// ============================================================================
// Startup
// ============================================================================

startServer().catch((error) => {
  console.error('Failed to start GraphQL server:', error);
  process.exit(1);
});

export { startServer, pubsub };
