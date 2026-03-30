#!/usr/bin/env node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Install DNA using Holochain JavaScript client

const WebSocket = require('ws');
const fs = require('fs');

const ADMIN_PORT = 46063;
const DNA_FILE = 'h-fl.dna';

async function installDNA() {
    console.log('Connecting to Holochain admin interface...');
    const ws = new WebSocket(`ws://localhost:${ADMIN_PORT}`);
    
    return new Promise((resolve, reject) => {
        ws.on('open', async () => {
            console.log('Connected!');
            
            // Read DNA file
            const dnaBytes = fs.readFileSync(DNA_FILE);
            const dnaBase64 = dnaBytes.toString('base64');
            
            // Generate agent key
            const generateKeyRequest = {
                id: 1,
                type: 'Request',
                data: {
                    method: 'GenerateAgentPubKey',
                    params: {}
                }
            };
            
            ws.send(JSON.stringify(generateKeyRequest));
        });
        
        ws.on('message', (data) => {
            const response = JSON.parse(data);
            console.log('Response:', response);
            
            if (response.id === 1 && response.type === 'Response') {
                const agentKey = response.data;
                console.log('Agent key generated:', agentKey);
                
                // Now register DNA...
                // Continue with registration flow
            }
        });
        
        ws.on('error', (err) => {
            console.error('WebSocket error:', err);
            reject(err);
        });
    });
}

// Check if we have the ws module
try {
    require('ws');
    installDNA().catch(console.error);
} catch (e) {
    console.log('Please install ws module: npm install ws');
}