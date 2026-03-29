// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import { config } from './config';
import axios from 'axios';
import { errorHandler } from './middleware/errorHandler';
import { authRoutes } from './routes/auth.routes';
import { accountRoutes } from './routes/account.routes';
import { emailRoutes } from './routes/email.routes';
import { folderRoutes } from './routes/folder.routes';
import { trustRoutes } from './routes/trust.routes';
import { setupWebSocket } from './websocket';
import { rateLimiter } from './middleware/rateLimiter';
import { trustService } from './services/trust.service';

const app = express();

// Security middleware
app.use(helmet());
app.use(cors({
  origin: config.corsOrigin,
  credentials: true,
}));

// Body parsing middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Rate limiting
app.use(rateLimiter);

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// API routes
app.use('/api/auth', authRoutes);
app.use('/api/accounts', accountRoutes);
app.use('/api/emails', emailRoutes);
app.use('/api/folders', folderRoutes);
app.use('/api/trust', trustRoutes);

// Error handling
app.use(errorHandler);

// Start server
const server = app.listen(config.port, () => {
  console.log(`🚀 Server running on port ${config.port}`);
console.log(`📧 Mycelix-Mail API ready`);
console.log(`🌍 Environment: ${config.nodeEnv}`);
console.log(`🛡️ Trust cache TTL: ${config.trustCacheTtlMs / 60000} minutes`);
});

// Optional external trust provider (e.g., MATL/Holochain bridge)
if (config.trustProviderUrl) {
  trustService.registerProvider(async (sender: string) => {
    const resp = await axios.get(config.trustProviderUrl as string, {
      params: { sender },
      headers: config.trustProviderApiKey
        ? { 'x-api-key': config.trustProviderApiKey }
        : undefined,
    });
    const summary = (resp.data?.data?.summary || resp.data?.summary) as any;
    if (!summary) return null;
    return {
      score: summary.score,
      tier: summary.tier,
      reasons: summary.reasons || [],
      pathLength: summary.pathLength,
      decayAt: summary.decayAt,
      attestations: summary.attestations || [],
      quarantined: summary.quarantined,
      fetchedAt: summary.fetchedAt || new Date().toISOString(),
    };
  });
  console.log(`🔗 Trust provider registered: ${config.trustProviderUrl}`);
}

// Setup WebSocket
setupWebSocket(server);

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully...');
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});

export default app;
