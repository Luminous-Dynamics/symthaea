#!/usr/bin/env node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * WebRTC Signaling Server for Holochain P2P Network
 * Enables real peer-to-peer connections between Holochain nodes
 */

const WebSocket = require('ws');
const http = require('http');
const crypto = require('crypto');

const DEFAULT_ALLOWED_ORIGIN_HOSTS = new Set(['localhost', '127.0.0.1', '::1']);

function isLoopbackHost(host) {
    return host === '127.0.0.1' || host === '::1' || host === 'localhost';
}

function isOriginAllowed(origin, allowedOrigins) {
    if (!origin) return true; // Non-browser clients
    if (origin === 'null') return false; // Avoid file:// or sandbox bypass origins by default

    if (allowedOrigins.length > 0) {
        return allowedOrigins.includes(origin);
    }

    try {
        const url = new URL(origin);
        return DEFAULT_ALLOWED_ORIGIN_HOSTS.has(url.hostname);
    } catch {
        return false;
    }
}

function getTokenFromReq(req) {
    try {
        const url = new URL(req.url, 'http://localhost');
        return url.searchParams.get('token') || '';
    } catch {
        return '';
    }
}

const HOST = process.env.SIGNAL_HOST || '127.0.0.1';
const PORT = Number(process.env.SIGNAL_PORT) || 9002;
const REQUIRE_TOKEN = process.env.SIGNAL_REQUIRE_TOKEN === '1' || !isLoopbackHost(HOST);
const AUTH_TOKEN = process.env.SIGNAL_AUTH_TOKEN || crypto.randomBytes(32).toString('hex');
const ALLOWED_ORIGINS = (process.env.SIGNAL_ALLOWED_ORIGINS || '')
    .split(',')
    .map(s => s.trim())
    .filter(Boolean);

// Create HTTP server for health checks
const server = http.createServer((req, res) => {
    if (req.url === '/health') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ 
            status: 'healthy',
            connections: wss.clients.size,
            timestamp: new Date().toISOString()
        }));
    } else {
        res.writeHead(404);
        res.end('Not Found');
    }
});

// Create WebSocket server for signaling
const wss = new WebSocket.Server({ server });

// Store peer connections
const peers = new Map();

wss.on('connection', (ws, req) => {
    const origin = req.headers.origin;
    const remoteAddress = req.socket.remoteAddress;

    if (!isOriginAllowed(origin, ALLOWED_ORIGINS)) {
        console.warn(`🛑 Rejected peer (disallowed Origin) from ${remoteAddress}: ${origin}`);
        ws.close(1008, 'Origin not allowed');
        return;
    }

    if (REQUIRE_TOKEN) {
        const token = getTokenFromReq(req);
        if (token !== AUTH_TOKEN) {
            console.warn(`🛑 Rejected peer (invalid token) from ${remoteAddress}`);
            ws.close(1008, 'Unauthorized');
            return;
        }
    }

    const peerId = generatePeerId();
    
    console.log(`🟢 New peer connected: ${peerId}`);
    console.log(`   From: ${remoteAddress}`);
    
    // Store peer
    peers.set(peerId, {
        ws,
        info: {
            id: peerId,
            connected: new Date().toISOString(),
            address: req.connection.remoteAddress
        }
    });
    
    // Send welcome message
    ws.send(JSON.stringify({
        type: 'welcome',
        peerId,
        peers: Array.from(peers.keys()).filter(id => id !== peerId)
    }));
    
    // Notify other peers of new connection
    broadcast({
        type: 'peer-joined',
        peerId,
        totalPeers: peers.size
    }, peerId);
    
    // Handle messages
    ws.on('message', (data) => {
        try {
            const msg = JSON.parse(data.toString());
            handleMessage(peerId, msg);
        } catch (err) {
            console.error(`❌ Invalid message from ${peerId}:`, err);
        }
    });
    
    // Handle disconnection
    ws.on('close', () => {
        console.log(`🔴 Peer disconnected: ${peerId}`);
        peers.delete(peerId);
        
        // Notify others
        broadcast({
            type: 'peer-left',
            peerId,
            totalPeers: peers.size
        });
    });
    
    // Handle errors
    ws.on('error', (err) => {
        console.error(`❌ WebSocket error for ${peerId}:`, err);
    });
});

/**
 * Handle signaling messages
 */
function handleMessage(fromPeerId, msg) {
    console.log(`📨 Message from ${fromPeerId}:`, msg.type);
    
    switch (msg.type) {
        case 'offer':
        case 'answer':
        case 'ice-candidate':
            // Forward WebRTC signaling to target peer
            if (msg.targetPeerId && peers.has(msg.targetPeerId)) {
                const targetPeer = peers.get(msg.targetPeerId);
                targetPeer.ws.send(JSON.stringify({
                    ...msg,
                    fromPeerId
                }));
                console.log(`   ➡️ Forwarded to ${msg.targetPeerId}`);
            } else {
                console.log(`   ⚠️ Target peer not found: ${msg.targetPeerId}`);
            }
            break;
            
        case 'broadcast':
            // Broadcast message to all peers
            broadcast({
                type: 'broadcast',
                fromPeerId,
                data: msg.data,
                timestamp: new Date().toISOString()
            }, fromPeerId);
            console.log(`   📢 Broadcasted to ${peers.size - 1} peers`);
            break;
            
        case 'ping':
            // Respond to ping
            const pingPeer = peers.get(fromPeerId);
            if (pingPeer) {
                pingPeer.ws.send(JSON.stringify({
                    type: 'pong',
                    timestamp: new Date().toISOString()
                }));
            }
            break;
            
        case 'get-peers':
            // Send list of connected peers
            const getPeer = peers.get(fromPeerId);
            if (getPeer) {
                getPeer.ws.send(JSON.stringify({
                    type: 'peers-list',
                    peers: Array.from(peers.entries()).map(([id, p]) => ({
                        id,
                        ...p.info
                    })).filter(p => p.id !== fromPeerId)
                }));
            }
            break;
            
        default:
            console.log(`   ❓ Unknown message type: ${msg.type}`);
    }
}

/**
 * Broadcast message to all peers except sender
 */
function broadcast(msg, excludePeerId = null) {
    const message = JSON.stringify(msg);
    
    peers.forEach((peer, peerId) => {
        if (peerId !== excludePeerId && peer.ws.readyState === WebSocket.OPEN) {
            peer.ws.send(message);
        }
    });
}

/**
 * Generate unique peer ID
 */
function generatePeerId() {
    return 'peer-' + Math.random().toString(36).substr(2, 9);
}

// Start server
server.listen(PORT, HOST, () => {
    console.log(`
╔════════════════════════════════════════════════╗
║       🧬 MYCELIX SIGNALING SERVER              ║
╠════════════════════════════════════════════════╣
║  WebRTC Signaling for Holochain P2P Network   ║
║                                                ║
║  🌐 WebSocket: ws://${HOST}:${PORT}             ║
║  📊 Health:    http://${HOST}:${PORT}/health    ║
║                                                ║
║  Status: READY FOR P2P CONNECTIONS            ║
╚════════════════════════════════════════════════╝
    `);

    if (REQUIRE_TOKEN) {
        console.log(`🔐 Auth required: ws://${HOST}:${PORT}?token=${AUTH_TOKEN}`);
    } else {
        console.log('🔐 Loopback bind: Origin checks enabled; token not required.');
    }
    
    // Log stats every 30 seconds
    setInterval(() => {
        if (peers.size > 0) {
            console.log(`\n📊 Network Stats:`);
            console.log(`   • Connected peers: ${peers.size}`);
            console.log(`   • Memory usage: ${(process.memoryUsage().heapUsed / 1024 / 1024).toFixed(2)} MB`);
            console.log(`   • Uptime: ${(process.uptime() / 60).toFixed(1)} minutes`);
        }
    }, 30000);
});

// Handle graceful shutdown
process.on('SIGINT', () => {
    console.log('\n🛑 Shutting down signaling server...');
    
    // Notify all peers
    broadcast({
        type: 'server-shutdown',
        message: 'Signaling server is shutting down'
    });
    
    // Close all connections
    peers.forEach((peer) => {
        peer.ws.close();
    });
    
    // Close server
    server.close(() => {
        console.log('✅ Server shut down gracefully');
        process.exit(0);
    });
});
