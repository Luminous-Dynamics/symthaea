/**
 * Bridge Protocol Validator
 *
 * Validates that all hApps in the Mycelix ecosystem can properly
 * send and receive bridge protocol messages.
 *
 * This is critical for production readiness as it tests real
 * cross-hApp communication patterns.
 */

import {
  CONDUCTOR_ENABLED,
  setupTestContext,
  teardownTestContext,
  waitForSync,
  generateTestAgentId,
  type TestContext,
} from './conductor-harness';

/**
 * Bridge message types as defined in the bridge protocol
 */
export enum BridgeMessageType {
  // Reputation queries
  REPUTATION_QUERY = 'reputation_query',
  REPUTATION_RESPONSE = 'reputation_response',

  // Credential verification
  CREDENTIAL_VERIFY = 'credential_verify',
  CREDENTIAL_RESULT = 'credential_result',

  // Cross-hApp events
  EVENT_BROADCAST = 'event_broadcast',
  EVENT_ACK = 'event_ack',

  // Registration
  HAPP_REGISTER = 'happ_register',
  HAPP_REGISTERED = 'happ_registered',
}

/**
 * Standard bridge message structure
 */
export interface BridgeMessage {
  id: string;
  type: BridgeMessageType;
  source_happ: string;
  target_happ: string;
  payload: unknown;
  timestamp: number;
  signature?: string;
}

/**
 * Bridge validation result
 */
export interface ValidationResult {
  happ: string;
  canSend: boolean;
  canReceive: boolean;
  errors: string[];
  latencyMs?: number;
}

/**
 * All hApps that should support bridge protocol
 */
export const BRIDGE_ENABLED_HAPPS = [
  'core',
  'identity',
  'governance',
  'knowledge',
  'epistemic-markets',
  'fabrication',
  'marketplace',
  'supplychain',
  'mail',
  'edunet',
  'justice',
  'finance',
  'property',
  'energy',
  'media',
  'desci',
  'music',
  'consensus',
] as const;

export type BridgeHappId = (typeof BRIDGE_ENABLED_HAPPS)[number];

/**
 * Bridge Protocol Validator class
 */
export class BridgeProtocolValidator {
  private ctx: TestContext | null = null;
  private results: Map<string, ValidationResult> = new Map();

  /**
   * Initialize validator with conductor context
   */
  async initialize(): Promise<void> {
    if (!CONDUCTOR_ENABLED) {
      throw new Error('Conductor not available. Set CONDUCTOR_AVAILABLE=true');
    }
    this.ctx = await setupTestContext();
  }

  /**
   * Cleanup resources
   */
  async cleanup(): Promise<void> {
    if (this.ctx) {
      await teardownTestContext(this.ctx);
      this.ctx = null;
    }
  }

  /**
   * Validate a single hApp's bridge capability
   */
  async validateHapp(happId: BridgeHappId): Promise<ValidationResult> {
    if (!this.ctx) {
      throw new Error('Validator not initialized');
    }

    const result: ValidationResult = {
      happ: happId,
      canSend: false,
      canReceive: false,
      errors: [],
    };

    const startTime = Date.now();

    try {
      // Test sending capability - register with bridge
      const registerMsg: BridgeMessage = {
        id: generateTestAgentId(),
        type: BridgeMessageType.HAPP_REGISTER,
        source_happ: happId,
        target_happ: 'core',
        payload: {
          happ_id: happId,
          version: '1.0.0',
          capabilities: ['reputation', 'events'],
        },
        timestamp: Date.now(),
      };

      const sendResult = await this.ctx.appClient.callZome({
        cap_secret: null,
        role_name: happId,
        zome_name: 'bridge',
        fn_name: 'send_bridge_message',
        payload: registerMsg,
      });

      result.canSend = sendResult !== null;

      // Test receiving capability - query for messages
      const queryResult = await this.ctx.appClient.callZome({
        cap_secret: null,
        role_name: happId,
        zome_name: 'bridge',
        fn_name: 'get_pending_messages',
        payload: { limit: 10 },
      });

      result.canReceive = Array.isArray(queryResult);

      result.latencyMs = Date.now() - startTime;
    } catch (error) {
      result.errors.push(`${happId}: ${(error as Error).message}`);
    }

    this.results.set(happId, result);
    return result;
  }

  /**
   * Validate all hApps
   */
  async validateAll(): Promise<Map<string, ValidationResult>> {
    for (const happId of BRIDGE_ENABLED_HAPPS) {
      try {
        await this.validateHapp(happId);
      } catch (error) {
        this.results.set(happId, {
          happ: happId,
          canSend: false,
          canReceive: false,
          errors: [(error as Error).message],
        });
      }
    }
    return this.results;
  }

  /**
   * Test cross-hApp message passing between two specific hApps
   */
  async testCrossHappMessage(
    sourceHapp: BridgeHappId,
    targetHapp: BridgeHappId,
    messageType: BridgeMessageType = BridgeMessageType.REPUTATION_QUERY
  ): Promise<{ success: boolean; latencyMs: number; error?: string }> {
    if (!this.ctx) {
      throw new Error('Validator not initialized');
    }

    const startTime = Date.now();

    try {
      // Send message from source
      const message: BridgeMessage = {
        id: generateTestAgentId(),
        type: messageType,
        source_happ: sourceHapp,
        target_happ: targetHapp,
        payload: {
          query_type: 'aggregate_reputation',
          agent_id: generateTestAgentId(),
        },
        timestamp: Date.now(),
      };

      await this.ctx.appClient.callZome({
        cap_secret: null,
        role_name: sourceHapp,
        zome_name: 'bridge',
        fn_name: 'send_bridge_message',
        payload: message,
      });

      // Wait for DHT propagation
      await waitForSync(500);

      // Check if target received the message
      const received = await this.ctx.appClient.callZome({
        cap_secret: null,
        role_name: targetHapp,
        zome_name: 'bridge',
        fn_name: 'get_messages_from',
        payload: { source_happ: sourceHapp, limit: 10 },
      });

      const found = Array.isArray(received) && received.some((m: BridgeMessage) => m.id === message.id);

      return {
        success: found,
        latencyMs: Date.now() - startTime,
      };
    } catch (error) {
      return {
        success: false,
        latencyMs: Date.now() - startTime,
        error: (error as Error).message,
      };
    }
  }

  /**
   * Test the full reputation aggregation flow across all hApps
   */
  async testReputationAggregation(): Promise<{
    success: boolean;
    participatingHapps: string[];
    aggregateScore: number;
    errors: string[];
  }> {
    if (!this.ctx) {
      throw new Error('Validator not initialized');
    }

    const errors: string[] = [];
    const participatingHapps: string[] = [];

    try {
      // Query aggregate reputation from core
      const result = await this.ctx.appClient.callZome({
        cap_secret: null,
        role_name: 'core',
        zome_name: 'bridge',
        fn_name: 'query_aggregate_reputation',
        payload: { agent_id: generateTestAgentId() },
      });

      if (result && typeof result.aggregate_score === 'number') {
        return {
          success: true,
          participatingHapps: result.contributing_happs || [],
          aggregateScore: result.aggregate_score,
          errors: [],
        };
      }

      errors.push('Invalid response from aggregate reputation query');
    } catch (error) {
      errors.push((error as Error).message);
    }

    return {
      success: false,
      participatingHapps,
      aggregateScore: 0,
      errors,
    };
  }

  /**
   * Generate validation report
   */
  generateReport(): string {
    const lines: string[] = [
      '═══════════════════════════════════════════════════════════════',
      '                 BRIDGE PROTOCOL VALIDATION REPORT              ',
      '═══════════════════════════════════════════════════════════════',
      '',
    ];

    let passing = 0;
    let failing = 0;

    for (const [happId, result] of this.results) {
      const status = result.canSend && result.canReceive ? '✅' : '❌';
      if (result.canSend && result.canReceive) passing++;
      else failing++;

      lines.push(`${status} ${happId.padEnd(20)} Send: ${result.canSend ? '✓' : '✗'}  Recv: ${result.canReceive ? '✓' : '✗'}  ${result.latencyMs ? `${result.latencyMs}ms` : ''}`);

      if (result.errors.length > 0) {
        result.errors.forEach((err) => lines.push(`   └─ Error: ${err}`));
      }
    }

    lines.push('');
    lines.push('───────────────────────────────────────────────────────────────');
    lines.push(`Summary: ${passing} passing, ${failing} failing out of ${this.results.size} hApps`);
    lines.push('═══════════════════════════════════════════════════════════════');

    return lines.join('\n');
  }
}

/**
 * Create a pre-configured validator instance
 */
export async function createValidator(): Promise<BridgeProtocolValidator> {
  const validator = new BridgeProtocolValidator();
  await validator.initialize();
  return validator;
}
