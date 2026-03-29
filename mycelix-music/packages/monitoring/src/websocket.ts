// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * WebSocket Metrics
 *
 * Metrics for tracking WebSocket connections and messages.
 */

import { Registry, Counter, Gauge, Histogram } from 'prom-client';
import { createCounter, createGauge, createHistogram } from './registry';

export interface WebSocketMetrics {
  connectionsTotal: Counter;
  connectionsCurrent: Gauge;
  messagesReceived: Counter;
  messagesSent: Counter;
  messageSize: Histogram;
  connectionDuration: Histogram;
  roomsActive: Gauge;
  usersOnline: Gauge;
}

/**
 * Create WebSocket metrics
 */
export function createWebSocketMetrics(registry: Registry): WebSocketMetrics {
  return {
    connectionsTotal: createCounter(
      registry,
      'websocket_connections_total',
      'Total number of WebSocket connections',
      ['status'] // 'opened', 'closed', 'error'
    ),
    connectionsCurrent: createGauge(
      registry,
      'websocket_connections_current',
      'Current number of active WebSocket connections'
    ),
    messagesReceived: createCounter(
      registry,
      'websocket_messages_received_total',
      'Total number of WebSocket messages received',
      ['type']
    ),
    messagesSent: createCounter(
      registry,
      'websocket_messages_sent_total',
      'Total number of WebSocket messages sent',
      ['type']
    ),
    messageSize: createHistogram(
      registry,
      'websocket_message_size_bytes',
      'WebSocket message size in bytes',
      ['direction'], // 'inbound', 'outbound'
      [64, 256, 1024, 4096, 16384, 65536]
    ),
    connectionDuration: createHistogram(
      registry,
      'websocket_connection_duration_seconds',
      'WebSocket connection duration in seconds',
      [],
      [1, 5, 15, 30, 60, 300, 600, 1800, 3600]
    ),
    roomsActive: createGauge(
      registry,
      'websocket_rooms_active',
      'Number of active rooms/channels'
    ),
    usersOnline: createGauge(
      registry,
      'websocket_users_online',
      'Number of authenticated online users'
    ),
  };
}

/**
 * Helper class to track WebSocket metrics
 */
export class WebSocketMetricsTracker {
  private connectionStartTimes: Map<string, number> = new Map();

  constructor(private metrics: WebSocketMetrics) {}

  /**
   * Record a new connection
   */
  onConnect(connectionId: string): void {
    this.connectionStartTimes.set(connectionId, Date.now());
    this.metrics.connectionsTotal.labels('opened').inc();
    this.metrics.connectionsCurrent.inc();
  }

  /**
   * Record a connection close
   */
  onDisconnect(connectionId: string): void {
    const startTime = this.connectionStartTimes.get(connectionId);
    if (startTime) {
      const duration = (Date.now() - startTime) / 1000;
      this.metrics.connectionDuration.observe(duration);
      this.connectionStartTimes.delete(connectionId);
    }
    this.metrics.connectionsTotal.labels('closed').inc();
    this.metrics.connectionsCurrent.dec();
  }

  /**
   * Record a connection error
   */
  onError(connectionId: string): void {
    this.metrics.connectionsTotal.labels('error').inc();
  }

  /**
   * Record a received message
   */
  onMessageReceived(type: string, sizeBytes: number): void {
    this.metrics.messagesReceived.labels(type).inc();
    this.metrics.messageSize.labels('inbound').observe(sizeBytes);
  }

  /**
   * Record a sent message
   */
  onMessageSent(type: string, sizeBytes: number): void {
    this.metrics.messagesSent.labels(type).inc();
    this.metrics.messageSize.labels('outbound').observe(sizeBytes);
  }

  /**
   * Update active rooms count
   */
  setActiveRooms(count: number): void {
    this.metrics.roomsActive.set(count);
  }

  /**
   * Update online users count
   */
  setOnlineUsers(count: number): void {
    this.metrics.usersOnline.set(count);
  }
}
