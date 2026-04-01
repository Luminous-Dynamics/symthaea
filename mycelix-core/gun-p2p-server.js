#!/usr/bin/env node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

/**
 * 🌐 Gun.js P2P Server - Real Decentralized Database
 * Combines our WebRTC signaling with Gun.js for actual P2P functionality
 */

const express = require('express');
const Gun = require('gun');
const http = require('http');
const socketIO = require('socket.io');
const path = require('path');

// Security defaults: keep this service local-only unless explicitly configured.
const HOST = process.env.GUN_HOST || '127.0.0.1';
const PORT = process.env.GUN_PORT || 9003;

// Explicit origin allowlist for browser access (comma-separated).
// Example:
//   GUN_ALLOWED_ORIGINS="http://localhost:7777,http://127.0.0.1:7777"
const ALLOWED_ORIGINS = (process.env.GUN_ALLOWED_ORIGINS ||
  `http://localhost:${PORT},http://127.0.0.1:${PORT}`
).split(',').map(s => s.trim()).filter(Boolean);

function isOriginAllowed(origin) {
  // Non-browser clients often have no Origin header; allow them when bound to localhost.
  if (!origin) return true;
  return ALLOWED_ORIGINS.includes(origin);
}

// Create Express app
const app = express();
const server = http.createServer(app);
const io = socketIO(server, {
  cors: {
    origin: (origin, callback) => {
      if (isOriginAllowed(origin)) return callback(null, true);
      return callback(new Error('Origin not allowed'), false);
    },
    methods: ['GET', 'POST', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    credentials: false,
  }
});

// Basic CORS gate for HTTP endpoints (no wildcard).
app.use((req, res, next) => {
  const origin = req.headers.origin;
  if (origin && isOriginAllowed(origin)) {
    res.setHeader('Access-Control-Allow-Origin', origin);
    res.setHeader('Vary', 'Origin');
    res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type,Authorization');
  }

  if (req.method === 'OPTIONS') {
    if (origin && !isOriginAllowed(origin)) return res.status(403).end('Forbidden');
    return res.status(204).end();
  }

  next();
});

// Static files
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());

// Initialize Gun with WebRTC and WebSocket support
const gun = Gun({
    web: server,
    peers: [],
    localStorage: false,
    radisk: true,
    file: 'gundata'
});

// Mycelix Consciousness Network namespace
const mycelix = gun.get('mycelix');
const agents = mycelix.get('agents');
const messages = mycelix.get('messages');
const reputation = mycelix.get('reputation');

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ 
        status: 'healthy',
        service: 'Gun.js P2P Server',
        timestamp: new Date().toISOString(),
        peers: Object.keys(gun._.opt.peers || {}).length
    });
});

// API: Register new agent
app.post('/api/agent/register', async (req, res) => {
    const { agentId, name, publicKey } = req.body;
    
    const agent = {
        id: agentId || Gun.node.Soul.uuid(),
        name: name || `Agent-${Date.now()}`,
        publicKey: publicKey || 'simulated-key',
        registered: Date.now(),
        reputation: 100
    };
    
    // Store in Gun's distributed database
    agents.get(agent.id).put(agent);
    
    res.json({ 
        success: true, 
        agent,
        message: 'Agent registered in decentralized network'
    });
});

// API: Send message through P2P network
app.post('/api/message/send', async (req, res) => {
    const { from, to, content, type = 'direct' } = req.body;
    
    const message = {
        id: Gun.node.Soul.uuid(),
        from,
        to,
        content,
        type,
        timestamp: Date.now()
    };
    
    // Store message in Gun
    messages.get(message.id).put(message);
    
    // If broadcast, also emit through WebSocket
    if (type === 'broadcast') {
        io.emit('broadcast-message', message);
    }
    
    res.json({ 
        success: true, 
        messageId: message.id,
        stored: 'decentralized'
    });
});

// API: Get agent's reputation
app.get('/api/reputation/:agentId', (req, res) => {
    const { agentId } = req.params;
    
    reputation.get(agentId).once((data) => {
        res.json({
            agentId,
            reputation: data || 100,
            source: 'decentralized'
        });
    });
});

// API: Update reputation
app.post('/api/reputation/update', (req, res) => {
    const { agentId, change, reason } = req.body;
    
    reputation.get(agentId).once((current) => {
        const newRep = (current || 100) + change;
        reputation.get(agentId).put(newRep);
        
        // Log reputation change
        mycelix.get('reputation-log').set({
            agentId,
            change,
            reason,
            newValue: newRep,
            timestamp: Date.now()
        });
        
        res.json({
            success: true,
            agentId,
            oldReputation: current || 100,
            newReputation: newRep,
            change
        });
    });
});

// API: Query decentralized data
app.post('/api/query', (req, res) => {
    const { path, filter } = req.body;
    const results = [];
    
    gun.get(path).map().once((data, key) => {
        if (!filter || Object.keys(filter).every(k => data[k] === filter[k])) {
            results.push({ ...data, _key: key });
        }
    });
    
    // Give Gun time to collect results
    setTimeout(() => {
        res.json({
            path,
            filter,
            results,
            count: results.length
        });
    }, 500);
});

// WebSocket handling for real-time P2P
io.on('connection', (socket) => {
    const peerId = `gun-peer-${Math.random().toString(36).substr(2, 9)}`;
    
    console.log(`🟢 Gun.js peer connected: ${peerId}`);
    
    // Subscribe to Gun changes
    const agentSub = agents.map().on((data, key) => {
        socket.emit('agent-update', { key, data });
    });
    
    const messageSub = messages.map().on((data, key) => {
        socket.emit('message-update', { key, data });
    });
    
    // Handle peer messages
    socket.on('gun-data', (data) => {
        gun.get(data.path).put(data.value);
    });
    
    socket.on('disconnect', () => {
        console.log(`🔴 Gun.js peer disconnected: ${peerId}`);
        agentSub.off();
        messageSub.off();
    });
});

// Start server
server.listen(PORT, HOST, () => {
    console.log(`
╔════════════════════════════════════════════════╗
║        🌐 GUN.JS P2P SERVER ACTIVE             ║
╠════════════════════════════════════════════════╣
║  Real Decentralized Database Network          ║
║                                                ║
║  📡 API:       http://${HOST}:${PORT}          ║
║  🔫 Gun.js:    ws://${HOST}:${PORT}/gun        ║
║  📊 Health:    http://${HOST}:${PORT}/health   ║
║                                                ║
║  Status: READY FOR DECENTRALIZED DATA         ║
╚════════════════════════════════════════════════╝
    `);

    console.log('🔒 Security defaults:');
    console.log(`  • Bound to: ${HOST} (set GUN_HOST=0.0.0.0 to expose intentionally)`);
    console.log('  • CORS allowlist:');
    ALLOWED_ORIGINS.forEach(o => console.log(`    - ${o}`));
    
    console.log('✨ Gun.js Features:');
    console.log('  • Real P2P data synchronization');
    console.log('  • Decentralized graph database');
    console.log('  • Conflict-free replicated data types (CRDTs)');
    console.log('  • Works in browser and Node.js');
    console.log('  • No blockchain required');
});

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\n🛑 Shutting down Gun.js server...');
    server.close(() => {
        console.log('✅ Server closed');
        process.exit(0);
    });
});
