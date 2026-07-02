// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Webhook System
 *
 * Allow external services to subscribe to events.
 * Supports signature verification, retries, and delivery tracking.
 */

import { Pool } from 'pg';
import { createHmac, randomUUID } from 'crypto';
import { getEventEmitter, AppEvents } from '../events';
import { getLogger } from '../logging';
import { getCircuitBreaker } from '../resilience/circuit-breaker';

/**
 * Webhook event types
 */
export type WebhookEvent =
  | 'song.created'
  | 'song.updated'
  | 'song.deleted'
  | 'play.recorded'
  | 'milestone.reached'
  | 'payment.received'
  | 'artist.verified';

/**
 * Webhook subscription
 */
export interface WebhookSubscription {
  id: string;
  walletAddress: string;
  url: string;
  events: WebhookEvent[];
  secret: string;
  active: boolean;
  createdAt: Date;
  updatedAt: Date;
  metadata?: Record<string, unknown>;
}

/**
 * Webhook delivery record
 */
export interface WebhookDelivery {
  id: string;
  subscriptionId: string;
  event: WebhookEvent;
  payload: unknown;
  status: 'pending' | 'success' | 'failed';
  attempts: number;
  lastAttemptAt?: Date;
  responseStatus?: number;
  responseBody?: string;
  error?: string;
  createdAt: Date;
}

/**
 * Webhook payload structure
 */
export interface WebhookPayload<T = unknown> {
  id: string;
  event: WebhookEvent;
  timestamp: string;
  data: T;
}

/**
 * Webhook manager configuration
 */
export interface WebhookConfig {
  maxRetries: number;
  retryDelays: number[]; // Delays in ms between retries
  timeout: number;
  signatureHeader: string;
  maxConcurrent: number;
}

/**
 * Default configuration
 */
const defaultConfig: WebhookConfig = {
  maxRetries: 5,
  retryDelays: [1000, 5000, 30000, 120000, 600000], // 1s, 5s, 30s, 2m, 10m
  timeout: 30000,
  signatureHeader: 'X-Webhook-Signature',
  maxConcurrent: 10,
};

/**
 * Webhook Manager
 */
export class WebhookManager {
  private config: WebhookConfig;
  private subscriptions: Map<string, WebhookSubscription> = new Map();
  private pendingDeliveries: Map<string, WebhookDelivery> = new Map();
  private eventSubscriptions: Map<WebhookEvent, Set<string>> = new Map();
  private logger = getLogger();
  private processing = false;

  constructor(
    private pool: Pool,
    config: Partial<WebhookConfig> = {}
  ) {
    this.config = { ...defaultConfig, ...config };
    this.initializeEventHandlers();
  }

  /**
   * Initialize event system handlers
   */
  private initializeEventHandlers(): void {
    const events = getEventEmitter();

    // Map internal events to webhook events
    events.on('song:created', (data) => {
      this.dispatch('song.created', data);
    });

    events.on('song:updated', (data) => {
      this.dispatch('song.updated', data);
    });

    events.on('play:recorded', (data) => {
      this.dispatch('play.recorded', data);
    });

    events.on('milestone:reached', (data) => {
      this.dispatch('milestone.reached', data);
    });
  }

  /**
   * Load subscriptions from database
   */
  async loadSubscriptions(): Promise<void> {
    const result = await this.pool.query(`
      SELECT * FROM webhook_subscriptions WHERE active = true
    `);

    for (const row of result.rows) {
      const subscription: WebhookSubscription = {
        id: row.id,
        walletAddress: row.wallet_address,
        url: row.url,
        events: row.events,
        secret: row.secret,
        active: row.active,
        createdAt: row.created_at,
        updatedAt: row.updated_at,
        metadata: row.metadata,
      };

      this.subscriptions.set(subscription.id, subscription);

      // Index by event type
      for (const event of subscription.events) {
        if (!this.eventSubscriptions.has(event)) {
          this.eventSubscriptions.set(event, new Set());
        }
        this.eventSubscriptions.get(event)!.add(subscription.id);
      }
    }

    this.logger.info(`Loaded ${this.subscriptions.size} webhook subscriptions`);
  }

  /**
   * Create a new subscription
   */
  async createSubscription(params: {
    walletAddress: string;
    url: string;
    events: WebhookEvent[];
    metadata?: Record<string, unknown>;
  }): Promise<WebhookSubscription> {
    // Validate URL
    try {
      new URL(params.url);
    } catch {
      throw new Error('Invalid webhook URL');
    }

    // Validate URL is HTTPS in production
    if (process.env.NODE_ENV === 'production' && !params.url.startsWith('https://')) {
      throw new Error('Webhook URL must use HTTPS');
    }

    const id = randomUUID();
    const secret = this.generateSecret();

    const result = await this.pool.query(`
      INSERT INTO webhook_subscriptions
        (id, wallet_address, url, events, secret, active, metadata)
      VALUES ($1, $2, $3, $4, $5, true, $6)
      RETURNING *
    `, [id, params.walletAddress, params.url, params.events, secret, params.metadata || {}]);

    const subscription: WebhookSubscription = {
      id: result.rows[0].id,
      walletAddress: result.rows[0].wallet_address,
      url: result.rows[0].url,
      events: result.rows[0].events,
      secret: result.rows[0].secret,
      active: result.rows[0].active,
      createdAt: result.rows[0].created_at,
      updatedAt: result.rows[0].updated_at,
      metadata: result.rows[0].metadata,
    };

    this.subscriptions.set(subscription.id, subscription);

    for (const event of subscription.events) {
      if (!this.eventSubscriptions.has(event)) {
        this.eventSubscriptions.set(event, new Set());
      }
      this.eventSubscriptions.get(event)!.add(subscription.id);
    }

    this.logger.info('Created webhook subscription', {
      subscriptionId: subscription.id,
      events: subscription.events,
    });

    return subscription;
  }

  /**
   * Update a subscription
   */
  async updateSubscription(
    id: string,
    updates: Partial<Pick<WebhookSubscription, 'url' | 'events' | 'active' | 'metadata'>>
  ): Promise<WebhookSubscription | null> {
    const existing = this.subscriptions.get(id);
    if (!existing) return null;

    const result = await this.pool.query(`
      UPDATE webhook_subscriptions
      SET
        url = COALESCE($2, url),
        events = COALESCE($3, events),
        active = COALESCE($4, active),
        metadata = COALESCE($5, metadata),
        updated_at = NOW()
      WHERE id = $1
      RETURNING *
    `, [id, updates.url, updates.events, updates.active, updates.metadata]);

    if (result.rows.length === 0) return null;

    const subscription: WebhookSubscription = {
      ...existing,
      url: result.rows[0].url,
      events: result.rows[0].events,
      active: result.rows[0].active,
      metadata: result.rows[0].metadata,
      updatedAt: result.rows[0].updated_at,
    };

    this.subscriptions.set(id, subscription);

    // Rebuild event index
    for (const [event, subs] of this.eventSubscriptions) {
      subs.delete(id);
    }
    for (const event of subscription.events) {
      if (!this.eventSubscriptions.has(event)) {
        this.eventSubscriptions.set(event, new Set());
      }
      this.eventSubscriptions.get(event)!.add(id);
    }

    return subscription;
  }

  /**
   * Delete a subscription
   */
  async deleteSubscription(id: string): Promise<boolean> {
    await this.pool.query('DELETE FROM webhook_subscriptions WHERE id = $1', [id]);

    this.subscriptions.delete(id);
    for (const subs of this.eventSubscriptions.values()) {
      subs.delete(id);
    }

    return true;
  }

  /**
   * Get subscriptions for a wallet
   */
  getSubscriptionsForWallet(walletAddress: string): WebhookSubscription[] {
    return Array.from(this.subscriptions.values())
      .filter(s => s.walletAddress.toLowerCase() === walletAddress.toLowerCase());
  }

  /**
   * Dispatch event to subscribers
   */
  async dispatch(event: WebhookEvent, data: unknown): Promise<void> {
    const subscriberIds = this.eventSubscriptions.get(event);
    if (!subscriberIds || subscriberIds.size === 0) return;

    const payload: WebhookPayload = {
      id: randomUUID(),
      event,
      timestamp: new Date().toISOString(),
      data,
    };

    for (const subscriptionId of subscriberIds) {
      const subscription = this.subscriptions.get(subscriptionId);
      if (!subscription || !subscription.active) continue;

      const delivery: WebhookDelivery = {
        id: randomUUID(),
        subscriptionId,
        event,
        payload,
        status: 'pending',
        attempts: 0,
        createdAt: new Date(),
      };

      this.pendingDeliveries.set(delivery.id, delivery);

      // Store in database
      await this.storeDelivery(delivery);
    }

    // Process deliveries
    this.processDeliveries();
  }

  /**
   * Store delivery record
   */
  private async storeDelivery(delivery: WebhookDelivery): Promise<void> {
    await this.pool.query(`
      INSERT INTO webhook_deliveries
        (id, subscription_id, event, payload, status, attempts, created_at)
      VALUES ($1, $2, $3, $4, $5, $6, $7)
    `, [
      delivery.id,
      delivery.subscriptionId,
      delivery.event,
      JSON.stringify(delivery.payload),
      delivery.status,
      delivery.attempts,
      delivery.createdAt,
    ]);
  }

  /**
   * Update delivery record
   */
  private async updateDelivery(delivery: WebhookDelivery): Promise<void> {
    await this.pool.query(`
      UPDATE webhook_deliveries
      SET
        status = $2,
        attempts = $3,
        last_attempt_at = $4,
        response_status = $5,
        response_body = $6,
        error = $7
      WHERE id = $1
    `, [
      delivery.id,
      delivery.status,
      delivery.attempts,
      delivery.lastAttemptAt,
      delivery.responseStatus,
      delivery.responseBody?.slice(0, 1000),
      delivery.error,
    ]);
  }

  /**
   * Process pending deliveries
   */
  private async processDeliveries(): Promise<void> {
    if (this.processing) return;
    this.processing = true;

    try {
      const pending = Array.from(this.pendingDeliveries.values())
        .filter(d => d.status === 'pending')
        .slice(0, this.config.maxConcurrent);

      await Promise.all(pending.map(d => this.deliver(d)));
    } finally {
      this.processing = false;
    }
  }

  /**
   * Deliver a webhook
   */
  private async deliver(delivery: WebhookDelivery): Promise<void> {
    const subscription = this.subscriptions.get(delivery.subscriptionId);
    if (!subscription) {
      delivery.status = 'failed';
      delivery.error = 'Subscription not found';
      await this.updateDelivery(delivery);
      this.pendingDeliveries.delete(delivery.id);
      return;
    }

    delivery.attempts++;
    delivery.lastAttemptAt = new Date();

    const circuitBreaker = getCircuitBreaker(`webhook:${subscription.id}`, {
      failureThreshold: 10,
      timeout: 300000, // 5 minutes
    });

    try {
      await circuitBreaker.execute(async () => {
        const body = JSON.stringify(delivery.payload);
        const signature = this.sign(body, subscription.secret);

        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), this.config.timeout);

        try {
          const response = await fetch(subscription.url, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              [this.config.signatureHeader]: signature,
              'X-Webhook-ID': delivery.id,
              'X-Webhook-Timestamp': delivery.payload.timestamp,
            },
            body,
            signal: controller.signal,
          });

          clearTimeout(timeout);

          delivery.responseStatus = response.status;
          delivery.responseBody = await response.text().catch(() => '');

          if (response.ok) {
            delivery.status = 'success';
            this.logger.debug('Webhook delivered successfully', {
              deliveryId: delivery.id,
              subscriptionId: subscription.id,
            });
          } else {
            throw new Error(`HTTP ${response.status}`);
          }
        } finally {
          clearTimeout(timeout);
        }
      });
    } catch (error) {
      delivery.error = (error as Error).message;

      if (delivery.attempts >= this.config.maxRetries) {
        delivery.status = 'failed';
        this.logger.warn('Webhook delivery failed permanently', {
          deliveryId: delivery.id,
          subscriptionId: subscription.id,
          attempts: delivery.attempts,
          error: delivery.error,
        });
      } else {
        // Schedule retry
        const delay = this.config.retryDelays[delivery.attempts - 1] || 600000;
        setTimeout(() => this.deliver(delivery), delay);
        this.logger.debug('Scheduling webhook retry', {
          deliveryId: delivery.id,
          attempt: delivery.attempts,
          nextRetryIn: delay,
        });
      }
    }

    await this.updateDelivery(delivery);

    if (delivery.status !== 'pending') {
      this.pendingDeliveries.delete(delivery.id);
    }
  }

  /**
   * Generate HMAC signature
   */
  private sign(payload: string, secret: string): string {
    const hmac = createHmac('sha256', secret);
    hmac.update(payload);
    return `sha256=${hmac.digest('hex')}`;
  }

  /**
   * Verify webhook signature (for external webhooks)
   */
  static verifySignature(payload: string, signature: string, secret: string): boolean {
    const expected = createHmac('sha256', secret).update(payload).digest('hex');
    return signature === `sha256=${expected}`;
  }

  /**
   * Generate a secret key
   */
  private generateSecret(): string {
    return `whsec_${randomUUID().replace(/-/g, '')}`;
  }

  /**
   * Get delivery history for a subscription
   */
  async getDeliveryHistory(
    subscriptionId: string,
    options: { limit?: number; offset?: number } = {}
  ): Promise<WebhookDelivery[]> {
    const { limit = 50, offset = 0 } = options;

    const result = await this.pool.query(`
      SELECT * FROM webhook_deliveries
      WHERE subscription_id = $1
      ORDER BY created_at DESC
      LIMIT $2 OFFSET $3
    `, [subscriptionId, limit, offset]);

    return result.rows.map(row => ({
      id: row.id,
      subscriptionId: row.subscription_id,
      event: row.event,
      payload: row.payload,
      status: row.status,
      attempts: row.attempts,
      lastAttemptAt: row.last_attempt_at,
      responseStatus: row.response_status,
      responseBody: row.response_body,
      error: row.error,
      createdAt: row.created_at,
    }));
  }

  /**
   * Retry a failed delivery
   */
  async retryDelivery(deliveryId: string): Promise<boolean> {
    const result = await this.pool.query(`
      SELECT * FROM webhook_deliveries WHERE id = $1
    `, [deliveryId]);

    if (result.rows.length === 0) return false;

    const row = result.rows[0];
    const delivery: WebhookDelivery = {
      id: row.id,
      subscriptionId: row.subscription_id,
      event: row.event,
      payload: row.payload,
      status: 'pending',
      attempts: 0,
      createdAt: row.created_at,
    };

    this.pendingDeliveries.set(delivery.id, delivery);
    this.processDeliveries();

    return true;
  }

  /**
   * Test a subscription (send test event)
   */
  async testSubscription(subscriptionId: string): Promise<{
    success: boolean;
    statusCode?: number;
    error?: string;
  }> {
    const subscription = this.subscriptions.get(subscriptionId);
    if (!subscription) {
      return { success: false, error: 'Subscription not found' };
    }

    const testPayload: WebhookPayload = {
      id: randomUUID(),
      event: 'song.created',
      timestamp: new Date().toISOString(),
      data: { test: true, message: 'This is a test webhook' },
    };

    try {
      const body = JSON.stringify(testPayload);
      const signature = this.sign(body, subscription.secret);

      const response = await fetch(subscription.url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          [this.config.signatureHeader]: signature,
          'X-Webhook-ID': testPayload.id,
          'X-Webhook-Test': 'true',
        },
        body,
      });

      return {
        success: response.ok,
        statusCode: response.status,
        error: response.ok ? undefined : `HTTP ${response.status}`,
      };
    } catch (error) {
      return {
        success: false,
        error: (error as Error).message,
      };
    }
  }
}

/**
 * Database migration for webhooks
 */
export const WEBHOOK_MIGRATION_SQL = `
CREATE TABLE IF NOT EXISTS webhook_subscriptions (
  id UUID PRIMARY KEY,
  wallet_address VARCHAR(42) NOT NULL,
  url TEXT NOT NULL,
  events TEXT[] NOT NULL,
  secret VARCHAR(64) NOT NULL,
  active BOOLEAN DEFAULT true,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_webhook_subscriptions_wallet ON webhook_subscriptions(wallet_address);
CREATE INDEX idx_webhook_subscriptions_active ON webhook_subscriptions(active) WHERE active = true;

CREATE TABLE IF NOT EXISTS webhook_deliveries (
  id UUID PRIMARY KEY,
  subscription_id UUID REFERENCES webhook_subscriptions(id) ON DELETE CASCADE,
  event VARCHAR(50) NOT NULL,
  payload JSONB NOT NULL,
  status VARCHAR(20) NOT NULL,
  attempts INTEGER DEFAULT 0,
  last_attempt_at TIMESTAMPTZ,
  response_status INTEGER,
  response_body TEXT,
  error TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_webhook_deliveries_subscription ON webhook_deliveries(subscription_id);
CREATE INDEX idx_webhook_deliveries_status ON webhook_deliveries(status) WHERE status = 'pending';
CREATE INDEX idx_webhook_deliveries_created ON webhook_deliveries(created_at);
`;

export default WebhookManager;
