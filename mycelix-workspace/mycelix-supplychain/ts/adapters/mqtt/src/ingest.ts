#!/usr/bin/env node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * MQTT Adapter for Mycelix Supply Chain
 *
 * Subscribes to MQTT topics and ingests IoT sensor events
 */

import { SupplyChainClient } from '@mycelix/supplychain-sdk';
import * as mqtt from 'mqtt';
import { program } from 'commander';

program
  .name('mqtt-adapter')
  .description('Ingest supply chain events from MQTT broker')
  .option('-b, --broker <url>', 'MQTT broker URL', 'mqtt://localhost:1883')
  .option('-t, --topic <topic>', 'MQTT topic to subscribe', 'supplychain/events/#')
  .option('-u, --url <url>', 'API base URL', 'http://localhost:8080')
  .parse();

const options = program.opts();

async function startMqttAdapter(brokerUrl: string, topic: string, apiUrl: string) {
  const client = new SupplyChainClient({ baseUrl: apiUrl });

  // Test API connection
  try {
    const health = await client.health();
    console.log(`Connected to API (version ${health.version})`);
  } catch (error) {
    console.error('Failed to connect to API:', error);
    process.exit(1);
  }

  // Connect to MQTT broker
  console.log(`Connecting to MQTT broker: ${brokerUrl}`);
  const mqttClient = mqtt.connect(brokerUrl);

  mqttClient.on('connect', () => {
    console.log(`Connected to MQTT broker`);
    console.log(`Subscribing to topic: ${topic}`);
    mqttClient.subscribe(topic, (err) => {
      if (err) {
        console.error('Subscription error:', err);
        process.exit(1);
      }
      console.log('Subscription successful. Waiting for messages...');
    });
  });

  mqttClient.on('message', async (receivedTopic, message) => {
    try {
      const payload = JSON.parse(message.toString());
      console.log(`Received message on ${receivedTopic}:`, payload);

      // Convert MQTT payload to SupplyEventVC
      // Assumes payload has the right structure
      const result = await client.ingestEvent(payload);
      console.log(`✓ Ingested event → claim ${result.claim_id}`);
    } catch (error) {
      console.error(`✗ Failed to process message:`, error);
    }
  });

  mqttClient.on('error', (error) => {
    console.error('MQTT error:', error);
  });

  // Graceful shutdown
  process.on('SIGINT', () => {
    console.log('\nShutting down...');
    mqttClient.end();
    process.exit(0);
  });
}

startMqttAdapter(options.broker, options.topic, options.url).catch(console.error);
