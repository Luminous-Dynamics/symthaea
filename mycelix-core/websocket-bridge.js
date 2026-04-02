#!/usr/bin/env node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * WebSocket Bridge for Mycelix Network
 * Connects the web UI to the Holochain hApp
 */

const WebSocket = require('ws');
const http = require('http');
const express = require('express');
const path = require('path');
const crypto = require('crypto');

const app = express();
const server = http.createServer(app);

// Configuration
const DEFAULT_ALLOWED_ORIGIN_HOSTS = new Set(['localhost', '127.0.0.1', '::1']);

function isLoopbackHost(host) {
    return host === '127.0.0.1' || host === '::1' || host === 'localhost';
}

function isOriginAllowed(origin, allowedOrigins) {
    if (!origin) return true; // Non-browser clients
    if (origin === 'null') return false;

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

const HOLOCHAIN_WS_URL = process.env.HOLOCHAIN_WS_URL || 'ws://127.0.0.1:8888'; // Holochain app interface
const BRIDGE_HOST = process.env.BRIDGE_HOST || '127.0.0.1';
const BRIDGE_PORT = Number(process.env.BRIDGE_PORT) || 8889;
const REQUIRE_TOKEN = process.env.BRIDGE_REQUIRE_TOKEN === '1' || !isLoopbackHost(BRIDGE_HOST);
const AUTH_TOKEN = process.env.BRIDGE_AUTH_TOKEN || crypto.randomBytes(32).toString('hex');
const ALLOWED_ORIGINS = (process.env.BRIDGE_ALLOWED_ORIGINS || '')
    .split(',')
    .map(s => s.trim())
    .filter(Boolean);

// Serve static files from UI directory
app.use(express.static(path.join(__dirname, 'ui')));

// WebSocket server for UI clients
const wss = new WebSocket.Server({ server });

// Connection to Holochain conductor (when available)
let holochainConnection = null;
let isHolochainConnected = false;

// Active UI clients
const clients = new Set();

// Simulated agent data (fallback when Holochain not connected)
const simulatedAgents = [
    { id: '1', name: 'You', status: 'online', coherence: 0.72 },
    { id: '2', name: 'Alice', status: 'online', coherence: 0.85 },
    { id: '3', name: 'Bob', status: 'away', coherence: 0.64 },
    { id: '4', name: 'Eve', status: 'online', coherence: 0.91 },
    { id: '5', name: 'Charlie', status: 'meditating', coherence: 0.95 }
];

// Simulated messages
const messageHistory = [
    { author: 'System', text: 'Welcome to Mycelix Network!', timestamp: Date.now() - 3600000 },
    { author: 'Alice', text: 'The field feels particularly strong today', timestamp: Date.now() - 1800000 },
    { author: 'Eve', text: 'I agree, the coherence is remarkable', timestamp: Date.now() - 900000 }
];

// Try to connect to Holochain
function connectToHolochain() {
    if (holochainConnection && holochainConnection.readyState === WebSocket.OPEN) {
        return; // Already connected
    }

    console.log('🔗 Attempting to connect to Holochain at', HOLOCHAIN_WS_URL);
    
    holochainConnection = new WebSocket(HOLOCHAIN_WS_URL);
    
    holochainConnection.on('open', () => {
        console.log('✅ Connected to Holochain conductor');
        isHolochainConnected = true;
        broadcastToClients({
            type: 'holochain_status',
            connected: true,
            message: 'Connected to Holochain network'
        });
    });
    
    holochainConnection.on('message', (data) => {
        // Forward Holochain messages to UI clients
        const message = JSON.parse(data);
        broadcastToClients({
            type: 'holochain_message',
            data: message
        });
    });
    
    holochainConnection.on('close', () => {
        console.log('⚠️  Holochain connection closed');
        isHolochainConnected = false;
        holochainConnection = null;
        broadcastToClients({
            type: 'holochain_status',
            connected: false,
            message: 'Disconnected from Holochain - using simulation mode'
        });
        
        // Retry connection after 5 seconds
        setTimeout(connectToHolochain, 5000);
    });
    
    holochainConnection.on('error', (err) => {
        console.log('⚠️  Holochain connection error:', err.message);
        isHolochainConnected = false;
    });
}

// Broadcast message to all connected UI clients
function broadcastToClients(message) {
    const data = JSON.stringify(message);
    clients.forEach(client => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(data);
        }
    });
}

// Handle UI client connections
wss.on('connection', (ws, req) => {
    const origin = req.headers.origin;
    const remoteAddress = req.socket.remoteAddress;

    if (!isOriginAllowed(origin, ALLOWED_ORIGINS)) {
        console.warn(`🛑 Rejected UI client (disallowed Origin) from ${remoteAddress}: ${origin}`);
        ws.close(1008, 'Origin not allowed');
        return;
    }

    if (REQUIRE_TOKEN) {
        const token = getTokenFromReq(req);
        if (token !== AUTH_TOKEN) {
            console.warn(`🛑 Rejected UI client (invalid token) from ${remoteAddress}`);
            ws.close(1008, 'Unauthorized');
            return;
        }
    }

    console.log('📱 New UI client connected from', remoteAddress);
    clients.add(ws);
    
    // Send initial state
    ws.send(JSON.stringify({
        type: 'initial_state',
        holochain_connected: isHolochainConnected,
        agents: simulatedAgents,
        messages: messageHistory,
        field_metrics: {
            coherence: 0.72,
            resonance: 0.85,
            amplitude: 0.64,
            frequency: 432
        }
    }));
    
    // Handle messages from UI
    ws.on('message', (message) => {
        try {
            const data = JSON.parse(message);
            
            switch(data.type) {
                case 'send_message':
                    handleSendMessage(data);
                    break;
                    
                case 'update_status':
                    handleUpdateStatus(data);
                    break;
                    
                case 'get_agents':
                    ws.send(JSON.stringify({
                        type: 'agents_list',
                        agents: simulatedAgents
                    }));
                    break;
                    
                case 'zome_call':
                    if (isHolochainConnected && holochainConnection) {
                        // Forward to Holochain
                        holochainConnection.send(JSON.stringify(data));
                    } else {
                        // Simulate response
                        ws.send(JSON.stringify({
                            type: 'zome_response',
                            success: false,
                            message: 'Holochain not connected - simulation mode active'
                        }));
                    }
                    break;
                    
                default:
                    console.log('Unknown message type:', data.type);
            }
        } catch (err) {
            console.error('Error handling message:', err);
        }
    });
    
    ws.on('close', () => {
        console.log('📱 UI client disconnected');
        clients.delete(ws);
    });
    
    ws.on('error', (err) => {
        console.error('Client error:', err);
        clients.delete(ws);
    });
});

// Handle sending messages
function handleSendMessage(data) {
    const message = {
        author: data.author || 'Anonymous',
        text: data.text,
        timestamp: Date.now()
    };
    
    messageHistory.push(message);
    
    // Broadcast to all clients
    broadcastToClients({
        type: 'new_message',
        message: message
    });
    
    // Simulate response after delay
    setTimeout(() => {
        const responses = [
            'I resonate with that perspective.',
            'The field is strengthening.',
            'Beautiful coherence emerging.',
            'Yes, I feel the connection too.',
            'This aligns with my experience.',
            '✨ Sacred geometry in motion'
        ];
        
        const responder = simulatedAgents[Math.floor(Math.random() * simulatedAgents.length)];
        if (responder.name !== message.author) {
            const response = {
                author: responder.name,
                text: responses[Math.floor(Math.random() * responses.length)],
                timestamp: Date.now()
            };
            
            messageHistory.push(response);
            broadcastToClients({
                type: 'new_message',
                message: response
            });
        }
    }, 1000 + Math.random() * 2000);
}

// Handle status updates
function handleUpdateStatus(data) {
    const agent = simulatedAgents.find(a => a.id === data.agentId);
    if (agent) {
        agent.status = data.status;
        broadcastToClients({
            type: 'agent_status_update',
            agentId: data.agentId,
            status: data.status
        });
    }
}

// Update field metrics periodically
setInterval(() => {
    const metrics = {
        coherence: (0.6 + Math.random() * 0.35).toFixed(2),
        resonance: (0.7 + Math.random() * 0.25).toFixed(2),
        amplitude: (0.5 + Math.random() * 0.4).toFixed(2),
        frequency: (420 + Math.random() * 24).toFixed(0)
    };
    
    broadcastToClients({
        type: 'field_update',
        metrics: metrics
    });
}, 5000);

// Start the server
server.listen(BRIDGE_PORT, BRIDGE_HOST, () => {
    console.log('🌊 Mycelix WebSocket Bridge');
    console.log('================================');
    console.log(`📡 Bridge bind: ${BRIDGE_HOST}:${BRIDGE_PORT}`);
    console.log(`🌐 Web UI: http://${BRIDGE_HOST}:${BRIDGE_PORT}`);
    console.log('');
    console.log('Features:');
    console.log('• Real-time P2P messaging');
    console.log('• Consciousness field visualization');
    console.log('• Agent presence tracking');
    console.log('• Automatic Holochain connection (when available)');
    console.log('• Simulation mode fallback');
    console.log('');

    if (REQUIRE_TOKEN) {
        console.log(`🔐 Auth required: ws://${BRIDGE_HOST}:${BRIDGE_PORT}?token=${AUTH_TOKEN}`);
    } else {
        console.log('🔐 Loopback bind: Origin checks enabled; token not required.');
    }
    
    // Try to connect to Holochain
    connectToHolochain();
});

// Graceful shutdown
process.on('SIGTERM', () => {
    console.log('Shutting down gracefully...');
    wss.clients.forEach(client => client.close());
    server.close(() => {
        process.exit(0);
    });
});
