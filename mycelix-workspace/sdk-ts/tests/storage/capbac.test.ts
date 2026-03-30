// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Capability-Based Access Control (CapBAC) Test Suite
 *
 * Comprehensive tests for the UESS CapBAC module.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  CapBACManager,
  AccessRight,
  createCapBACManager,
  createCapBACManagerWithKey,
  createReadCapability,
  createFullAccessCapability,
  type CapabilityToken,
  type ValidationContext,
  type CapabilityConstraints,
} from '../../src/storage/capbac.js';
import { secureRandomBytes } from '../../src/security/index.js';

// =============================================================================
// Test Helpers
// =============================================================================

async function createTestManager(): Promise<CapBACManager> {
  const signingKey = await secureRandomBytes(32);
  return new CapBACManager({
    signingKey,
    maxDelegationDepth: 3,
    enableRevocation: true,
  });
}

// =============================================================================
// CapBACManager Tests
// =============================================================================

describe('CapBACManager', () => {
  let manager: CapBACManager;

  beforeEach(async () => {
    manager = await createTestManager();
  });

  describe('createCapability', () => {
    it('should create a valid capability token', async () => {
      const capability = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ],
        issuer: 'alice',
        holder: 'bob',
      });

      expect(capability.id).toBeDefined();
      expect(capability.resource).toBe('doc:123');
      expect(capability.rights).toContain(AccessRight.READ);
      expect(capability.issuer).toBe('alice');
      expect(capability.holder).toBe('bob');
      expect(capability.signature).toBeDefined();
      expect(capability.issuedAt).toBeLessThanOrEqual(Date.now());
    });

    it('should create capability with multiple rights', async () => {
      const capability = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ, AccessRight.WRITE, AccessRight.DELETE],
        issuer: 'alice',
        holder: 'bob',
      });

      expect(capability.rights).toContain(AccessRight.READ);
      expect(capability.rights).toContain(AccessRight.WRITE);
      expect(capability.rights).toContain(AccessRight.DELETE);
    });

    it('should set expiration time', async () => {
      const expiresAt = Date.now() + 3600000;
      const capability = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ],
        issuer: 'alice',
        holder: 'bob',
        expiresAt,
      });

      expect(capability.expiresAt).toBe(expiresAt);
    });

    it('should use default TTL when configured', async () => {
      const signingKey = await secureRandomBytes(32);
      const managerWithTtl = new CapBACManager({
        signingKey,
        maxDelegationDepth: 3,
        defaultTtlMs: 3600000,
        enableRevocation: true,
      });

      const now = Date.now();
      const capability = await managerWithTtl.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ],
        issuer: 'alice',
        holder: 'bob',
      });

      expect(capability.expiresAt).toBeGreaterThan(now);
      expect(capability.expiresAt).toBeLessThanOrEqual(now + 3600000 + 100);
    });

    it('should set delegation depth', async () => {
      const capability = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ, AccessRight.DELEGATE],
        issuer: 'alice',
        holder: 'bob',
        delegationDepth: 2,
      });

      expect(capability.delegationDepth).toBe(2);
    });

    it('should default delegation depth to 0', async () => {
      const capability = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ],
        issuer: 'alice',
        holder: 'bob',
      });

      expect(capability.delegationDepth).toBe(0);
    });

    it('should set pattern flag', async () => {
      const capability = await manager.createCapability({
        resource: 'doc:*',
        isPattern: true,
        rights: [AccessRight.READ],
        issuer: 'alice',
        holder: 'bob',
      });

      expect(capability.isPattern).toBe(true);
    });

    it('should default isPattern to false', async () => {
      const capability = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ],
        issuer: 'alice',
        holder: 'bob',
      });

      expect(capability.isPattern).toBe(false);
    });

    it('should initialize delegation chain with issuer', async () => {
      const capability = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ],
        issuer: 'alice',
        holder: 'bob',
      });

      expect(capability.delegationChain).toEqual(['alice']);
    });

    it('should include constraints', async () => {
      const constraints: CapabilityConstraints = {
        maxUses: 5,
        allowedIPs: ['192.168.1.1'],
      };

      const capability = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ],
        issuer: 'alice',
        holder: 'bob',
        constraints,
      });

      expect(capability.constraints).toEqual(constraints);
    });
  });

  describe('validateCapability', () => {
    it('should validate a valid capability', async () => {
      const capability = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ],
        issuer: 'alice',
        holder: 'bob',
      });

      const result = await manager.validateCapability(capability, {
        agentId: 'bob',
        resource: 'doc:123',
        right: AccessRight.READ,
        timestamp: Date.now(),
      });

      expect(result.valid).toBe(true);
      expect(result.capability).toEqual(capability);
    });

    it('should reject invalid signature', async () => {
      const capability = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ],
        issuer: 'alice',
        holder: 'bob',
      });

      // Tamper with the token
      capability.resource = 'doc:456';

      const result = await manager.validateCapability(capability, {
        agentId: 'bob',
        resource: 'doc:456',
        right: AccessRight.READ,
        timestamp: Date.now(),
      });

      expect(result.valid).toBe(false);
      expect(result.error).toBe('Invalid signature');
    });

    it('should reject mismatched holder', async () => {
      const capability = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ],
        issuer: 'alice',
        holder: 'bob',
      });

      const result = await manager.validateCapability(capability, {
        agentId: 'charlie',
        resource: 'doc:123',
        right: AccessRight.READ,
        timestamp: Date.now(),
      });

      expect(result.valid).toBe(false);
      expect(result.error).toBe('Token holder does not match agent');
    });

    it('should reject expired token', async () => {
      const capability = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ],
        issuer: 'alice',
        holder: 'bob',
        expiresAt: Date.now() - 1000,
      });

      const result = await manager.validateCapability(capability, {
        agentId: 'bob',
        resource: 'doc:123',
        right: AccessRight.READ,
        timestamp: Date.now(),
      });

      expect(result.valid).toBe(false);
      expect(result.error).toBe('Token has expired');
    });

    it('should reject mismatched resource', async () => {
      const capability = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ],
        issuer: 'alice',
        holder: 'bob',
      });

      const result = await manager.validateCapability(capability, {
        agentId: 'bob',
        resource: 'doc:456',
        right: AccessRight.READ,
        timestamp: Date.now(),
      });

      expect(result.valid).toBe(false);
      expect(result.error).toBe('Resource does not match capability');
    });

    it('should reject missing right', async () => {
      const capability = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ],
        issuer: 'alice',
        holder: 'bob',
      });

      const result = await manager.validateCapability(capability, {
        agentId: 'bob',
        resource: 'doc:123',
        right: AccessRight.WRITE,
        timestamp: Date.now(),
      });

      expect(result.valid).toBe(false);
      expect(result.error).toContain('not granted');
    });

    it('should allow ADMIN right to access any right', async () => {
      const capability = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.ADMIN],
        issuer: 'alice',
        holder: 'bob',
      });

      const result = await manager.validateCapability(capability, {
        agentId: 'bob',
        resource: 'doc:123',
        right: AccessRight.DELETE,
        timestamp: Date.now(),
      });

      expect(result.valid).toBe(true);
    });

    it('should match pattern resources', async () => {
      const capability = await manager.createCapability({
        resource: 'doc:*',
        isPattern: true,
        rights: [AccessRight.READ],
        issuer: 'alice',
        holder: 'bob',
      });

      const result = await manager.validateCapability(capability, {
        agentId: 'bob',
        resource: 'doc:123',
        right: AccessRight.READ,
        timestamp: Date.now(),
      });

      expect(result.valid).toBe(true);
    });

    it('should match complex patterns', async () => {
      const capability = await manager.createCapability({
        resource: 'user:alice:doc:*',
        isPattern: true,
        rights: [AccessRight.READ],
        issuer: 'alice',
        holder: 'bob',
      });

      const result = await manager.validateCapability(capability, {
        agentId: 'bob',
        resource: 'user:alice:doc:123',
        right: AccessRight.READ,
        timestamp: Date.now(),
      });

      expect(result.valid).toBe(true);
    });

    it('should reject revoked token', async () => {
      const capability = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ],
        issuer: 'alice',
        holder: 'bob',
      });

      manager.revoke(capability.id);

      const result = await manager.validateCapability(capability, {
        agentId: 'bob',
        resource: 'doc:123',
        right: AccessRight.READ,
        timestamp: Date.now(),
      });

      expect(result.valid).toBe(false);
      expect(result.error).toBe('Token has been revoked');
    });
  });

  describe('constraint validation', () => {
    it('should enforce maxUses constraint', async () => {
      const capability = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ],
        issuer: 'alice',
        holder: 'bob',
        constraints: { maxUses: 2 },
      });

      const context: ValidationContext = {
        agentId: 'bob',
        resource: 'doc:123',
        right: AccessRight.READ,
        timestamp: Date.now(),
      };

      // First use
      let result = await manager.validateCapability(capability, context);
      expect(result.valid).toBe(true);

      // Second use
      result = await manager.validateCapability(capability, context);
      expect(result.valid).toBe(true);

      // Third use - should fail
      result = await manager.validateCapability(capability, context);
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Maximum uses exceeded');
    });

    it('should enforce IP restrictions', async () => {
      const capability = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ],
        issuer: 'alice',
        holder: 'bob',
        constraints: { allowedIPs: ['192.168.1.1', '192.168.1.2'] },
      });

      // Allowed IP
      let result = await manager.validateCapability(capability, {
        agentId: 'bob',
        resource: 'doc:123',
        right: AccessRight.READ,
        timestamp: Date.now(),
        clientIP: '192.168.1.1',
      });
      expect(result.valid).toBe(true);

      // Disallowed IP
      result = await manager.validateCapability(capability, {
        agentId: 'bob',
        resource: 'doc:123',
        right: AccessRight.READ,
        timestamp: Date.now(),
        clientIP: '10.0.0.1',
      });
      expect(result.valid).toBe(false);
      expect(result.error).toBe('IP address not allowed');
    });

    it('should handle CIDR notation in IP restrictions', async () => {
      const capability = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ],
        issuer: 'alice',
        holder: 'bob',
        constraints: { allowedIPs: ['192.168.1.0/24'] },
      });

      const result = await manager.validateCapability(capability, {
        agentId: 'bob',
        resource: 'doc:123',
        right: AccessRight.READ,
        timestamp: Date.now(),
        clientIP: '192.168.1.50',
      });
      expect(result.valid).toBe(true);
    });

    it('should enforce time windows', async () => {
      const currentHour = new Date().getHours();

      const capability = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ],
        issuer: 'alice',
        holder: 'bob',
        constraints: {
          timeWindows: [{ start: currentHour, end: currentHour + 1 }],
        },
      });

      const result = await manager.validateCapability(capability, {
        agentId: 'bob',
        resource: 'doc:123',
        right: AccessRight.READ,
        timestamp: Date.now(),
      });
      expect(result.valid).toBe(true);
    });

    it('should reject outside time window', async () => {
      const capability = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ],
        issuer: 'alice',
        holder: 'bob',
        constraints: {
          timeWindows: [{ start: 2, end: 3 }], // 2am-3am
        },
      });

      // Use a timestamp outside the window
      const outsideWindow = new Date();
      outsideWindow.setHours(12, 0, 0, 0); // Noon

      const result = await manager.validateCapability(capability, {
        agentId: 'bob',
        resource: 'doc:123',
        right: AccessRight.READ,
        timestamp: outsideWindow.getTime(),
      });
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Access outside allowed time window');
    });

    it('should enforce day constraints in time windows', async () => {
      const now = new Date();
      const currentDay = now.getDay();
      const currentHour = now.getHours();

      const capability = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ],
        issuer: 'alice',
        holder: 'bob',
        constraints: {
          timeWindows: [{ start: currentHour, end: currentHour + 1, days: [currentDay] }],
        },
      });

      const result = await manager.validateCapability(capability, {
        agentId: 'bob',
        resource: 'doc:123',
        right: AccessRight.READ,
        timestamp: Date.now(),
      });
      expect(result.valid).toBe(true);
    });

    it('should enforce required context', async () => {
      const capability = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ],
        issuer: 'alice',
        holder: 'bob',
        constraints: {
          requiredContext: { department: 'engineering' },
        },
      });

      // Matching context
      let result = await manager.validateCapability(capability, {
        agentId: 'bob',
        resource: 'doc:123',
        right: AccessRight.READ,
        timestamp: Date.now(),
        contextData: { department: 'engineering' },
      });
      expect(result.valid).toBe(true);

      // Non-matching context
      result = await manager.validateCapability(capability, {
        agentId: 'bob',
        resource: 'doc:123',
        right: AccessRight.READ,
        timestamp: Date.now(),
        contextData: { department: 'sales' },
      });
      expect(result.valid).toBe(false);
      expect(result.error).toContain('Required context');
    });
  });

  describe('delegateCapability', () => {
    it('should delegate capability to new holder', async () => {
      const original = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ, AccessRight.DELEGATE],
        issuer: 'alice',
        holder: 'bob',
        delegationDepth: 2,
      });

      const delegated = await manager.delegateCapability(original, 'charlie');

      expect(delegated.holder).toBe('charlie');
      expect(delegated.issuer).toBe('bob');
      expect(delegated.parentId).toBe(original.id);
      expect(delegated.delegationChain).toEqual(['alice', 'bob']);
      expect(delegated.delegationDepth).toBe(1);
    });

    it('should restrict rights on delegation', async () => {
      const original = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ, AccessRight.WRITE, AccessRight.DELEGATE],
        issuer: 'alice',
        holder: 'bob',
        delegationDepth: 2,
      });

      const delegated = await manager.delegateCapability(original, 'charlie', {
        rights: [AccessRight.READ],
      });

      expect(delegated.rights).toEqual([AccessRight.READ]);
      expect(delegated.rights).not.toContain(AccessRight.WRITE);
    });

    it('should not allow expanding rights on delegation', async () => {
      const original = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ, AccessRight.DELEGATE],
        issuer: 'alice',
        holder: 'bob',
        delegationDepth: 2,
      });

      await expect(
        manager.delegateCapability(original, 'charlie', {
          rights: [AccessRight.READ, AccessRight.WRITE],
        })
      ).rejects.toThrow("Cannot grant right 'write' not present in parent capability");
    });

    it('should fail when delegation depth exhausted', async () => {
      const original = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ, AccessRight.DELEGATE],
        issuer: 'alice',
        holder: 'bob',
        delegationDepth: 0,
      });

      await expect(manager.delegateCapability(original, 'charlie')).rejects.toThrow(
        'delegation depth exhausted'
      );
    });

    it('should shorten expiration on delegation', async () => {
      const originalExpires = Date.now() + 7200000; // 2 hours
      const original = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ, AccessRight.DELEGATE],
        issuer: 'alice',
        holder: 'bob',
        delegationDepth: 2,
        expiresAt: originalExpires,
      });

      // Try to extend expiration - should be capped
      const delegated = await manager.delegateCapability(original, 'charlie', {
        expiresAt: Date.now() + 14400000, // 4 hours
      });

      expect(delegated.expiresAt).toBe(originalExpires);
    });

    it('should allow shortening expiration on delegation', async () => {
      const originalExpires = Date.now() + 7200000; // 2 hours
      const shorterExpires = Date.now() + 3600000; // 1 hour

      const original = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ, AccessRight.DELEGATE],
        issuer: 'alice',
        holder: 'bob',
        delegationDepth: 2,
        expiresAt: originalExpires,
      });

      const delegated = await manager.delegateCapability(original, 'charlie', {
        expiresAt: shorterExpires,
      });

      expect(delegated.expiresAt).toBe(shorterExpires);
    });

    it('should merge constraints on delegation', async () => {
      const original = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ, AccessRight.DELEGATE],
        issuer: 'alice',
        holder: 'bob',
        delegationDepth: 2,
        constraints: {
          maxUses: 10,
          allowedIPs: ['192.168.1.1', '192.168.1.2', '192.168.1.3'],
        },
      });

      const delegated = await manager.delegateCapability(original, 'charlie', {
        constraints: {
          maxUses: 5,
          allowedIPs: ['192.168.1.1', '192.168.1.2'],
        },
      });

      // Should take more restrictive maxUses
      expect(delegated.constraints?.maxUses).toBe(5);
      // Should intersect IPs
      expect(delegated.constraints?.allowedIPs).toEqual(['192.168.1.1', '192.168.1.2']);
    });

    it('should fail if parent capability is invalid', async () => {
      const original = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ, AccessRight.DELEGATE],
        issuer: 'alice',
        holder: 'bob',
        delegationDepth: 2,
      });

      manager.revoke(original.id);

      await expect(manager.delegateCapability(original, 'charlie')).rejects.toThrow(
        'Cannot delegate'
      );
    });
  });

  describe('hasAccess', () => {
    it('should return true for valid access', async () => {
      const capability = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ],
        issuer: 'alice',
        holder: 'bob',
      });

      const hasAccess = await manager.hasAccess(capability, 'doc:123', AccessRight.READ, 'bob');
      expect(hasAccess).toBe(true);
    });

    it('should return false for invalid access', async () => {
      const capability = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ],
        issuer: 'alice',
        holder: 'bob',
      });

      const hasAccess = await manager.hasAccess(capability, 'doc:123', AccessRight.WRITE, 'bob');
      expect(hasAccess).toBe(false);
    });
  });

  describe('revocation', () => {
    it('should revoke a token', () => {
      manager.revoke('token-123');
      expect(manager.isRevoked('token-123')).toBe(true);
    });

    it('should return false for non-revoked token', () => {
      expect(manager.isRevoked('token-456')).toBe(false);
    });

    it('should revoke chain', () => {
      manager.revokeChain('root-token');
      expect(manager.isRevoked('root-token')).toBe(true);
    });

    it('should clear revocations', () => {
      manager.revoke('token-123');
      expect(manager.isRevoked('token-123')).toBe(true);

      manager.clearRevocations();
      expect(manager.isRevoked('token-123')).toBe(false);
    });
  });

  describe('serialization', () => {
    it('should serialize and deserialize capability', async () => {
      const capability = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ, AccessRight.WRITE],
        issuer: 'alice',
        holder: 'bob',
        expiresAt: Date.now() + 3600000,
        constraints: { maxUses: 5 },
      });

      const serialized = manager.serializeCapability(capability);
      expect(typeof serialized).toBe('string');

      const deserialized = manager.deserializeCapability(serialized);
      expect(deserialized).toEqual(capability);
    });

    it('should produce URL-safe serialized output', async () => {
      const capability = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ],
        issuer: 'alice',
        holder: 'bob',
      });

      const serialized = manager.serializeCapability(capability);
      // base64url should not contain +, /, or =
      expect(serialized).not.toMatch(/[+/=]/);
    });
  });
});

// =============================================================================
// Factory Function Tests
// =============================================================================

describe('Factory Functions', () => {
  describe('createCapBACManager', () => {
    it('should create manager with random signing key', async () => {
      const manager = await createCapBACManager();
      expect(manager).toBeInstanceOf(CapBACManager);

      // Should be able to create and validate tokens
      const capability = await manager.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ],
        issuer: 'alice',
        holder: 'bob',
      });
      expect(capability.signature).toBeDefined();
    });

    it('should accept custom options', async () => {
      const manager = await createCapBACManager({
        maxDelegationDepth: 5,
        defaultTtlMs: 7200000,
        enableRevocation: false,
      });
      expect(manager).toBeInstanceOf(CapBACManager);
    });
  });

  describe('createCapBACManagerWithKey', () => {
    it('should create manager with specific signing key', async () => {
      const signingKey = await secureRandomBytes(32);
      const manager = createCapBACManagerWithKey(signingKey);
      expect(manager).toBeInstanceOf(CapBACManager);
    });

    it('should produce consistent signatures with same key', async () => {
      const signingKey = await secureRandomBytes(32);
      const manager1 = createCapBACManagerWithKey(signingKey);
      const manager2 = createCapBACManagerWithKey(signingKey);

      const cap1 = await manager1.createCapability({
        resource: 'doc:123',
        rights: [AccessRight.READ],
        issuer: 'alice',
        holder: 'bob',
      });

      // Validate with second manager using same key
      const result = await manager2.validateCapability(cap1, {
        agentId: 'bob',
        resource: 'doc:123',
        right: AccessRight.READ,
        timestamp: Date.now(),
      });

      expect(result.valid).toBe(true);
    });
  });

  describe('createReadCapability', () => {
    it('should create read-only capability', async () => {
      const manager = await createCapBACManager();
      const capability = await createReadCapability(
        manager,
        'doc:123',
        'alice',
        'bob'
      );

      expect(capability.rights).toEqual([AccessRight.READ]);
      expect(capability.resource).toBe('doc:123');
      expect(capability.issuer).toBe('alice');
      expect(capability.holder).toBe('bob');
    });

    it('should accept TTL parameter', async () => {
      const manager = await createCapBACManager();
      const now = Date.now();
      const capability = await createReadCapability(
        manager,
        'doc:123',
        'alice',
        'bob',
        3600000
      );

      expect(capability.expiresAt).toBeGreaterThan(now);
      expect(capability.expiresAt).toBeLessThanOrEqual(now + 3600000 + 100);
    });
  });

  describe('createFullAccessCapability', () => {
    it('should create full access capability', async () => {
      const manager = await createCapBACManager();
      const capability = await createFullAccessCapability(
        manager,
        'doc:123',
        'alice',
        'bob'
      );

      expect(capability.rights).toContain(AccessRight.READ);
      expect(capability.rights).toContain(AccessRight.WRITE);
      expect(capability.rights).toContain(AccessRight.DELETE);
      expect(capability.rights).toContain(AccessRight.DELEGATE);
      expect(capability.delegationDepth).toBe(2);
    });

    it('should accept TTL parameter', async () => {
      const manager = await createCapBACManager();
      const now = Date.now();
      const capability = await createFullAccessCapability(
        manager,
        'doc:123',
        'alice',
        'bob',
        7200000
      );

      expect(capability.expiresAt).toBeGreaterThan(now);
    });
  });
});

// =============================================================================
// Edge Cases
// =============================================================================

describe('Edge Cases', () => {
  it('should handle special characters in resource names', async () => {
    const manager = await createTestManager();
    const capability = await manager.createCapability({
      resource: 'user:alice@example.com:doc:123',
      rights: [AccessRight.READ],
      issuer: 'alice',
      holder: 'bob',
    });

    const result = await manager.validateCapability(capability, {
      agentId: 'bob',
      resource: 'user:alice@example.com:doc:123',
      right: AccessRight.READ,
      timestamp: Date.now(),
    });

    expect(result.valid).toBe(true);
  });

  it('should handle question mark in pattern', async () => {
    const manager = await createTestManager();
    const capability = await manager.createCapability({
      resource: 'doc:?',
      isPattern: true,
      rights: [AccessRight.READ],
      issuer: 'alice',
      holder: 'bob',
    });

    // Should match single character
    const result = await manager.validateCapability(capability, {
      agentId: 'bob',
      resource: 'doc:1',
      right: AccessRight.READ,
      timestamp: Date.now(),
    });

    expect(result.valid).toBe(true);
  });

  it('should handle patterns with special regex characters', async () => {
    const manager = await createTestManager();
    const capability = await manager.createCapability({
      resource: 'doc.(v1)*',
      isPattern: true,
      rights: [AccessRight.READ],
      issuer: 'alice',
      holder: 'bob',
    });

    // The dots should be escaped
    const result = await manager.validateCapability(capability, {
      agentId: 'bob',
      resource: 'doc.(v1)test',
      right: AccessRight.READ,
      timestamp: Date.now(),
    });

    expect(result.valid).toBe(true);
  });

  it('should validate capabilities with parentId', async () => {
    const manager = await createTestManager();
    const parent = await manager.createCapability({
      resource: 'doc:123',
      rights: [AccessRight.READ, AccessRight.DELEGATE],
      issuer: 'alice',
      holder: 'bob',
      delegationDepth: 2,
    });

    const child = await manager.delegateCapability(parent, 'charlie');

    const result = await manager.validateCapability(child, {
      agentId: 'charlie',
      resource: 'doc:123',
      right: AccessRight.READ,
      timestamp: Date.now(),
    });

    expect(result.valid).toBe(true);
    expect(child.parentId).toBe(parent.id);
  });

  it('should handle constraint merging with only parent constraints', async () => {
    const manager = await createTestManager();
    const parent = await manager.createCapability({
      resource: 'doc:123',
      rights: [AccessRight.READ, AccessRight.DELEGATE],
      issuer: 'alice',
      holder: 'bob',
      delegationDepth: 2,
      constraints: { maxUses: 10 },
    });

    const child = await manager.delegateCapability(parent, 'charlie');

    expect(child.constraints?.maxUses).toBe(10);
  });

  it('should handle constraint merging with only child constraints', async () => {
    const manager = await createTestManager();
    const parent = await manager.createCapability({
      resource: 'doc:123',
      rights: [AccessRight.READ, AccessRight.DELEGATE],
      issuer: 'alice',
      holder: 'bob',
      delegationDepth: 2,
    });

    const child = await manager.delegateCapability(parent, 'charlie', {
      constraints: { maxUses: 5 },
    });

    expect(child.constraints?.maxUses).toBe(5);
  });

  it('should handle validation without revocation enabled', async () => {
    const signingKey = await secureRandomBytes(32);
    const manager = new CapBACManager({
      signingKey,
      maxDelegationDepth: 3,
      enableRevocation: false,
    });

    const capability = await manager.createCapability({
      resource: 'doc:123',
      rights: [AccessRight.READ],
      issuer: 'alice',
      holder: 'bob',
    });

    manager.revoke(capability.id);

    // Should still be valid because revocation is disabled
    const result = await manager.validateCapability(capability, {
      agentId: 'bob',
      resource: 'doc:123',
      right: AccessRight.READ,
      timestamp: Date.now(),
    });

    expect(result.valid).toBe(true);
  });
});
