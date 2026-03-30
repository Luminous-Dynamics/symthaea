// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Capability-Based Access Control Integration Tests
 *
 * Tests the complete capability system including:
 * - Capability creation and management
 * - Delegation with attenuation
 * - Constraint enforcement
 * - Revocation propagation
 * - Access control patterns
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  CapabilityManager,
  CapabilityGuardedBridge,
  CapabilityTemplates,
  getCapabilityManager,
  createCapabilityGuardedBridge,
  resetCapabilityManager,
  type Capability,
  type CapabilityOperation,
  type ResourceType,
  type CapabilityInvocation,
  type CapabilityConstraints,
} from '../../src/bridge/index.js';

describe('Capability-Based Access Control', () => {
  let manager: CapabilityManager;

  beforeEach(() => {
    manager = new CapabilityManager();
  });

  describe('CapabilityManager', () => {
    describe('createCapability', () => {
      it('should create a basic capability', () => {
        const capability = manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          operations: ['read', 'write'],
        });

        expect(capability.id).toMatch(/^cap-/);
        expect(capability.issuerDid).toBe('did:mycelix:alice');
        expect(capability.granteeDid).toBe('did:mycelix:bob');
        expect(capability.resource).toBe('knowledge');
        expect(capability.operations).toEqual(['read', 'write']);
        expect(capability.delegationChain).toEqual(['did:mycelix:alice']);
        expect(capability.revoked).toBe(false);
        expect(capability.useCount).toBe(0);
      });

      it('should create a capability with resource ID', () => {
        const capability = manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          resourceId: 'claim-123',
          operations: ['read'],
        });

        expect(capability.resourceId).toBe('claim-123');
      });

      it('should create a delegatable capability', () => {
        const capability = manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'governance',
          operations: ['read', 'write', 'delegate'],
          delegatable: true,
        });

        expect(capability.delegatable).toBe(true);
      });

      it('should create a capability with constraints', () => {
        const constraints: CapabilityConstraints = {
          maxUses: 10,
          expiresAt: Date.now() + 3600000,
          rateLimit: {
            maxRequests: 5,
            windowMs: 60000,
          },
        };

        const capability = manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'finance',
          operations: ['read'],
          constraints,
        });

        expect(capability.constraints).toEqual(constraints);
      });

      it('should generate unique capability IDs', () => {
        const cap1 = manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          operations: ['read'],
        });

        const cap2 = manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          operations: ['read'],
        });

        expect(cap1.id).not.toBe(cap2.id);
      });

      it('should generate cryptographic proof', () => {
        const capability = manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'identity',
          operations: ['read'],
        });

        expect(capability.proof).toBeDefined();
        expect(capability.proof.length).toBeGreaterThan(0);
      });
    });

    describe('delegateCapability', () => {
      it('should delegate a capability to another entity', () => {
        const parent = manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          operations: ['read', 'write', 'execute'],
          delegatable: true,
        });

        const delegated = manager.delegateCapability({
          parentCapabilityId: parent.id,
          delegateeDid: 'did:mycelix:carol',
          operations: ['read', 'write'],
          delegatable: true,
        });

        expect(delegated).not.toBeNull();
        expect(delegated!.issuerDid).toBe('did:mycelix:bob');
        expect(delegated!.granteeDid).toBe('did:mycelix:carol');
        expect(delegated!.operations).toEqual(['read', 'write']);
        expect(delegated!.parentCapabilityId).toBe(parent.id);
        expect(delegated!.delegationChain).toEqual([
          'did:mycelix:alice',
          'did:mycelix:bob',
        ]);
      });

      it('should fail to delegate non-delegatable capability', () => {
        const parent = manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          operations: ['read'],
          delegatable: false,
        });

        const delegated = manager.delegateCapability({
          parentCapabilityId: parent.id,
          delegateeDid: 'did:mycelix:carol',
          operations: ['read'],
          delegatable: false,
        });

        expect(delegated).toBeNull();
      });

      it('should fail to delegate with more operations than parent', () => {
        const parent = manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          operations: ['read'],
          delegatable: true,
        });

        const delegated = manager.delegateCapability({
          parentCapabilityId: parent.id,
          delegateeDid: 'did:mycelix:carol',
          operations: ['read', 'write'], // write not in parent
          delegatable: false,
        });

        expect(delegated).toBeNull();
      });

      it('should merge constraints during delegation', () => {
        const parent = manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          operations: ['read', 'write'],
          constraints: {
            maxUses: 100,
            expiresAt: Date.now() + 7200000, // 2 hours
          },
          delegatable: true,
        });

        const delegated = manager.delegateCapability({
          parentCapabilityId: parent.id,
          delegateeDid: 'did:mycelix:carol',
          operations: ['read'],
          additionalConstraints: {
            maxUses: 10, // More restrictive
            expiresAt: Date.now() + 3600000, // 1 hour - more restrictive
          },
          delegatable: false,
        });

        expect(delegated).not.toBeNull();
        expect(delegated!.constraints!.maxUses).toBe(10);
        expect(delegated!.constraints!.expiresAt).toBeLessThanOrEqual(
          Date.now() + 3600000
        );
      });

      it('should fail to delegate revoked capability', () => {
        const parent = manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          operations: ['read'],
          delegatable: true,
        });

        manager.revokeCapability(parent.id, 'did:mycelix:alice');

        const delegated = manager.delegateCapability({
          parentCapabilityId: parent.id,
          delegateeDid: 'did:mycelix:carol',
          operations: ['read'],
          delegatable: false,
        });

        expect(delegated).toBeNull();
      });

      it('should fail to delegate non-existent capability', () => {
        const delegated = manager.delegateCapability({
          parentCapabilityId: 'non-existent-cap',
          delegateeDid: 'did:mycelix:carol',
          operations: ['read'],
          delegatable: false,
        });

        expect(delegated).toBeNull();
      });
    });

    describe('verifyCapability', () => {
      it('should verify a valid capability invocation', () => {
        const capability = manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          operations: ['read', 'write'],
        });

        const invocation: CapabilityInvocation = {
          capabilityId: capability.id,
          operation: 'read',
          resource: 'knowledge',
          invokerDid: 'did:mycelix:bob',
          timestamp: Date.now(),
        };

        const result = manager.verifyCapability(invocation);

        expect(result.valid).toBe(true);
        expect(result.capability).toBeDefined();
        expect(result.capability!.id).toBe(capability.id);
      });

      it('should reject non-existent capability', () => {
        const invocation: CapabilityInvocation = {
          capabilityId: 'non-existent',
          operation: 'read',
          resource: 'knowledge',
          invokerDid: 'did:mycelix:bob',
          timestamp: Date.now(),
        };

        const result = manager.verifyCapability(invocation);

        expect(result.valid).toBe(false);
        expect(result.reason).toBe('Capability not found');
      });

      it('should reject revoked capability', () => {
        const capability = manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          operations: ['read'],
        });

        manager.revokeCapability(capability.id, 'did:mycelix:alice');

        const invocation: CapabilityInvocation = {
          capabilityId: capability.id,
          operation: 'read',
          resource: 'knowledge',
          invokerDid: 'did:mycelix:bob',
          timestamp: Date.now(),
        };

        const result = manager.verifyCapability(invocation);

        expect(result.valid).toBe(false);
        expect(result.reason).toBe('Capability has been revoked');
      });

      it('should reject wrong invoker', () => {
        const capability = manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          operations: ['read'],
        });

        const invocation: CapabilityInvocation = {
          capabilityId: capability.id,
          operation: 'read',
          resource: 'knowledge',
          invokerDid: 'did:mycelix:eve', // Wrong invoker
          timestamp: Date.now(),
        };

        const result = manager.verifyCapability(invocation);

        expect(result.valid).toBe(false);
        expect(result.reason).toBe('Invoker does not match capability grantee');
      });

      it('should reject unauthorized operation', () => {
        const capability = manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          operations: ['read'], // Only read
        });

        const invocation: CapabilityInvocation = {
          capabilityId: capability.id,
          operation: 'write', // Not allowed
          resource: 'knowledge',
          invokerDid: 'did:mycelix:bob',
          timestamp: Date.now(),
        };

        const result = manager.verifyCapability(invocation);

        expect(result.valid).toBe(false);
        expect(result.reason).toContain("Operation 'write' not allowed");
      });

      it('should reject wrong resource type', () => {
        const capability = manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          operations: ['read'],
        });

        const invocation: CapabilityInvocation = {
          capabilityId: capability.id,
          operation: 'read',
          resource: 'finance', // Wrong resource
          invokerDid: 'did:mycelix:bob',
          timestamp: Date.now(),
        };

        const result = manager.verifyCapability(invocation);

        expect(result.valid).toBe(false);
        expect(result.reason).toContain("Resource 'finance' not covered");
      });

      it('should allow wildcard resource access', () => {
        const capability = manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: '*',
          operations: ['read'],
        });

        const invocation: CapabilityInvocation = {
          capabilityId: capability.id,
          operation: 'read',
          resource: 'finance',
          invokerDid: 'did:mycelix:bob',
          timestamp: Date.now(),
        };

        const result = manager.verifyCapability(invocation);

        expect(result.valid).toBe(true);
      });

      it('should reject wrong resource ID', () => {
        const capability = manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          resourceId: 'claim-123',
          operations: ['read'],
        });

        const invocation: CapabilityInvocation = {
          capabilityId: capability.id,
          operation: 'read',
          resource: 'knowledge',
          resourceId: 'claim-456', // Wrong ID
          invokerDid: 'did:mycelix:bob',
          timestamp: Date.now(),
        };

        const result = manager.verifyCapability(invocation);

        expect(result.valid).toBe(false);
        expect(result.reason).toBe('Resource ID does not match capability');
      });

      it('should enforce max uses constraint', () => {
        const capability = manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          operations: ['read'],
          constraints: {
            maxUses: 2,
          },
        });

        const invocation: CapabilityInvocation = {
          capabilityId: capability.id,
          operation: 'read',
          resource: 'knowledge',
          invokerDid: 'did:mycelix:bob',
          timestamp: Date.now(),
        };

        // First use
        let result = manager.verifyCapability(invocation);
        expect(result.valid).toBe(true);
        expect(result.remainingUses).toBe(1);

        // Second use
        result = manager.verifyCapability(invocation);
        expect(result.valid).toBe(true);
        expect(result.remainingUses).toBe(0);

        // Third use - should fail
        result = manager.verifyCapability(invocation);
        expect(result.valid).toBe(false);
        expect(result.reason).toBe('Capability has exceeded maximum uses');
      });

      it('should enforce expiration constraint', () => {
        const capability = manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          operations: ['read'],
          constraints: {
            expiresAt: Date.now() - 1000, // Already expired
          },
        });

        const invocation: CapabilityInvocation = {
          capabilityId: capability.id,
          operation: 'read',
          resource: 'knowledge',
          invokerDid: 'did:mycelix:bob',
          timestamp: Date.now(),
        };

        const result = manager.verifyCapability(invocation);

        expect(result.valid).toBe(false);
        expect(result.reason).toBe('Capability has expired');
      });

      it('should return time until expiration', () => {
        const expiresAt = Date.now() + 60000; // 1 minute from now
        const capability = manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          operations: ['read'],
          constraints: {
            expiresAt,
          },
        });

        const invocation: CapabilityInvocation = {
          capabilityId: capability.id,
          operation: 'read',
          resource: 'knowledge',
          invokerDid: 'did:mycelix:bob',
          timestamp: Date.now(),
        };

        const result = manager.verifyCapability(invocation);

        expect(result.valid).toBe(true);
        expect(result.expiresIn).toBeDefined();
        expect(result.expiresIn).toBeLessThanOrEqual(60000);
        expect(result.expiresIn).toBeGreaterThan(0);
      });

      it('should track use count', () => {
        const capability = manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          operations: ['read'],
        });

        const invocation: CapabilityInvocation = {
          capabilityId: capability.id,
          operation: 'read',
          resource: 'knowledge',
          invokerDid: 'did:mycelix:bob',
          timestamp: Date.now(),
        };

        manager.verifyCapability(invocation);
        manager.verifyCapability(invocation);
        manager.verifyCapability(invocation);

        const updatedCap = manager.getCapability(capability.id);
        expect(updatedCap!.useCount).toBe(3);
      });
    });

    describe('revokeCapability', () => {
      it('should revoke a capability', () => {
        const capability = manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          operations: ['read'],
        });

        const result = manager.revokeCapability(
          capability.id,
          'did:mycelix:alice'
        );

        expect(result).toBe(true);

        const updatedCap = manager.getCapability(capability.id);
        expect(updatedCap!.revoked).toBe(true);
      });

      it('should fail to revoke with wrong revoker', () => {
        const capability = manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          operations: ['read'],
        });

        const result = manager.revokeCapability(
          capability.id,
          'did:mycelix:eve' // Not the issuer
        );

        expect(result).toBe(false);
      });

      it('should cascade revocation to children', () => {
        const parent = manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          operations: ['read', 'write'],
          delegatable: true,
        });

        const child = manager.delegateCapability({
          parentCapabilityId: parent.id,
          delegateeDid: 'did:mycelix:carol',
          operations: ['read'],
          delegatable: true,
        })!;

        const grandchild = manager.delegateCapability({
          parentCapabilityId: child.id,
          delegateeDid: 'did:mycelix:dave',
          operations: ['read'],
          delegatable: false,
        })!;

        // Revoke parent
        manager.revokeCapability(parent.id, 'did:mycelix:alice');

        // Check all are revoked
        const parentCap = manager.getCapability(parent.id);
        const childCap = manager.getCapability(child.id);
        const grandchildCap = manager.getCapability(grandchild.id);

        expect(parentCap!.revoked).toBe(true);
        expect(childCap!.revoked).toBe(true);
        expect(grandchildCap!.revoked).toBe(true);
      });

      it('should reject invocations after revocation', () => {
        const parent = manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          operations: ['read'],
          delegatable: true,
        });

        const child = manager.delegateCapability({
          parentCapabilityId: parent.id,
          delegateeDid: 'did:mycelix:carol',
          operations: ['read'],
          delegatable: false,
        })!;

        // Revoke parent
        manager.revokeCapability(parent.id, 'did:mycelix:alice');

        // Try to use child
        const invocation: CapabilityInvocation = {
          capabilityId: child.id,
          operation: 'read',
          resource: 'knowledge',
          invokerDid: 'did:mycelix:carol',
          timestamp: Date.now(),
        };

        const result = manager.verifyCapability(invocation);
        expect(result.valid).toBe(false);
      });

      it('should allow grantee to revoke delegated capabilities', () => {
        const parent = manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          operations: ['read'],
          delegatable: true,
        });

        const child = manager.delegateCapability({
          parentCapabilityId: parent.id,
          delegateeDid: 'did:mycelix:carol',
          operations: ['read'],
          delegatable: false,
        })!;

        // Bob (grantee of parent) should be able to revoke child
        const result = manager.revokeCapability(child.id, 'did:mycelix:bob');

        expect(result).toBe(true);
      });
    });

    describe('getCapabilitiesForGrantee', () => {
      it('should return all capabilities for a grantee', () => {
        manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          operations: ['read'],
        });

        manager.createCapability({
          issuerDid: 'did:mycelix:carol',
          granteeDid: 'did:mycelix:bob',
          resource: 'finance',
          operations: ['read', 'write'],
        });

        manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:dave',
          resource: 'governance',
          operations: ['read'],
        });

        const bobCaps = manager.getCapabilitiesForGrantee('did:mycelix:bob');

        expect(bobCaps.length).toBe(2);
        expect(bobCaps.some((c) => c.resource === 'knowledge')).toBe(true);
        expect(bobCaps.some((c) => c.resource === 'finance')).toBe(true);
      });

      it('should exclude revoked capabilities', () => {
        const cap1 = manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          operations: ['read'],
        });

        manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'finance',
          operations: ['read'],
        });

        manager.revokeCapability(cap1.id, 'did:mycelix:alice');

        const bobCaps = manager.getCapabilitiesForGrantee('did:mycelix:bob');

        expect(bobCaps.length).toBe(1);
        expect(bobCaps[0].resource).toBe('finance');
      });
    });

    describe('hasCapability', () => {
      it('should return true when capability exists', () => {
        manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          operations: ['read', 'write'],
        });

        expect(
          manager.hasCapability('did:mycelix:bob', 'knowledge', 'read')
        ).toBe(true);
        expect(
          manager.hasCapability('did:mycelix:bob', 'knowledge', 'write')
        ).toBe(true);
      });

      it('should return false when capability does not exist', () => {
        manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          operations: ['read'],
        });

        expect(
          manager.hasCapability('did:mycelix:bob', 'knowledge', 'write')
        ).toBe(false);
        expect(
          manager.hasCapability('did:mycelix:bob', 'finance', 'read')
        ).toBe(false);
        expect(
          manager.hasCapability('did:mycelix:carol', 'knowledge', 'read')
        ).toBe(false);
      });

      it('should respect resource ID restrictions', () => {
        manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          resourceId: 'claim-123',
          operations: ['read'],
        });

        expect(
          manager.hasCapability(
            'did:mycelix:bob',
            'knowledge',
            'read',
            'claim-123'
          )
        ).toBe(true);
        expect(
          manager.hasCapability(
            'did:mycelix:bob',
            'knowledge',
            'read',
            'claim-456'
          )
        ).toBe(false);
      });
    });

    describe('clear', () => {
      it('should remove all capabilities', () => {
        manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          operations: ['read'],
        });

        manager.createCapability({
          issuerDid: 'did:mycelix:alice',
          granteeDid: 'did:mycelix:carol',
          resource: 'finance',
          operations: ['read'],
        });

        manager.clear();

        expect(
          manager.getCapabilitiesForGrantee('did:mycelix:bob').length
        ).toBe(0);
        expect(
          manager.getCapabilitiesForGrantee('did:mycelix:carol').length
        ).toBe(0);
      });
    });
  });

  describe('CapabilityGuardedBridge', () => {
    let bridge: CapabilityGuardedBridge;

    beforeEach(() => {
      bridge = new CapabilityGuardedBridge(manager, 'did:mycelix:alice');
    });

    describe('grantCapability', () => {
      it('should grant a capability with bridge owner as issuer', () => {
        const capability = bridge.grantCapability({
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          operations: ['read'],
        });

        expect(capability.issuerDid).toBe('did:mycelix:alice');
        expect(capability.granteeDid).toBe('did:mycelix:bob');
      });
    });

    describe('executeWithCapability', () => {
      it('should execute operation with valid capability', async () => {
        const capability = manager.createCapability({
          issuerDid: 'did:mycelix:other',
          granteeDid: 'did:mycelix:alice',
          resource: 'knowledge',
          operations: ['read'],
        });

        const invocation: CapabilityInvocation = {
          capabilityId: capability.id,
          operation: 'read',
          resource: 'knowledge',
          invokerDid: 'did:mycelix:alice',
          timestamp: Date.now(),
        };

        const result = await bridge.executeWithCapability(invocation, async () => {
          return 'success';
        });

        expect(result.success).toBe(true);
        expect(result.result).toBe('success');
      });

      it('should reject operation with invalid capability', async () => {
        const invocation: CapabilityInvocation = {
          capabilityId: 'non-existent',
          operation: 'read',
          resource: 'knowledge',
          invokerDid: 'did:mycelix:alice',
          timestamp: Date.now(),
        };

        const result = await bridge.executeWithCapability(invocation, async () => {
          return 'should not execute';
        });

        expect(result.success).toBe(false);
        expect(result.error).toContain('Access denied');
      });

      it('should handle operation errors', async () => {
        const capability = manager.createCapability({
          issuerDid: 'did:mycelix:other',
          granteeDid: 'did:mycelix:alice',
          resource: 'knowledge',
          operations: ['read'],
        });

        const invocation: CapabilityInvocation = {
          capabilityId: capability.id,
          operation: 'read',
          resource: 'knowledge',
          invokerDid: 'did:mycelix:alice',
          timestamp: Date.now(),
        };

        const result = await bridge.executeWithCapability(invocation, async () => {
          throw new Error('Operation failed');
        });

        expect(result.success).toBe(false);
        expect(result.error).toBe('Operation failed');
      });
    });

    describe('revokeCapability', () => {
      it('should revoke a capability granted by the bridge', () => {
        const capability = bridge.grantCapability({
          granteeDid: 'did:mycelix:bob',
          resource: 'knowledge',
          operations: ['read'],
        });

        const result = bridge.revokeCapability(capability.id);

        expect(result).toBe(true);
      });
    });

    describe('canAccess', () => {
      it('should check if bridge owner has capability', () => {
        manager.createCapability({
          issuerDid: 'did:mycelix:other',
          granteeDid: 'did:mycelix:alice',
          resource: 'knowledge',
          operations: ['read', 'write'],
        });

        expect(bridge.canAccess('knowledge', 'read')).toBe(true);
        expect(bridge.canAccess('knowledge', 'write')).toBe(true);
        expect(bridge.canAccess('knowledge', 'execute')).toBe(false);
        expect(bridge.canAccess('finance', 'read')).toBe(false);
      });
    });

    describe('getMyCapabilities', () => {
      it('should return capabilities for the bridge owner', () => {
        manager.createCapability({
          issuerDid: 'did:mycelix:other',
          granteeDid: 'did:mycelix:alice',
          resource: 'knowledge',
          operations: ['read'],
        });

        manager.createCapability({
          issuerDid: 'did:mycelix:other',
          granteeDid: 'did:mycelix:alice',
          resource: 'finance',
          operations: ['read', 'write'],
        });

        const caps = bridge.getMyCapabilities();

        expect(caps.length).toBe(2);
      });
    });
  });

  describe('CapabilityTemplates', () => {
    it('should provide read-only operations', () => {
      const ops = CapabilityTemplates.readOnly();
      expect(ops).toEqual(['read']);
    });

    it('should provide read-write operations', () => {
      const ops = CapabilityTemplates.readWrite();
      expect(ops).toEqual(['read', 'write']);
    });

    it('should provide full access operations', () => {
      const ops = CapabilityTemplates.fullAccess();
      expect(ops).toEqual(['read', 'write', 'execute']);
    });

    it('should provide admin operations', () => {
      const ops = CapabilityTemplates.admin();
      expect(ops).toEqual(['read', 'write', 'execute', 'delegate', 'admin']);
    });

    it('should create one-hour constraint', () => {
      const constraint = CapabilityTemplates.oneHourConstraint();
      expect(constraint.expiresAt).toBeDefined();
      expect(constraint.expiresAt! - Date.now()).toBeLessThanOrEqual(3600000);
      expect(constraint.expiresAt! - Date.now()).toBeGreaterThan(3500000);
    });

    it('should create one-day constraint', () => {
      const constraint = CapabilityTemplates.oneDayConstraint();
      expect(constraint.expiresAt).toBeDefined();
      expect(constraint.expiresAt! - Date.now()).toBeLessThanOrEqual(86400000);
    });

    it('should create single-use constraint', () => {
      const constraint = CapabilityTemplates.singleUse();
      expect(constraint.maxUses).toBe(1);
    });

    it('should create business hours constraint', () => {
      const constraint = CapabilityTemplates.businessHoursOnly();
      expect(constraint.timeWindow).toBeDefined();
      expect(constraint.timeWindow!.startHour).toBe(9);
      expect(constraint.timeWindow!.endHour).toBe(17);
    });

    it('should create rate-limited constraint', () => {
      const constraint = CapabilityTemplates.rateLimited();
      expect(constraint.rateLimit).toBeDefined();
      expect(constraint.rateLimit!.maxRequests).toBe(10);
      expect(constraint.rateLimit!.windowMs).toBe(60000);
    });
  });

  describe('Singleton Factory Functions', () => {
    beforeEach(() => {
      resetCapabilityManager();
    });

    it('should return singleton CapabilityManager', () => {
      const manager1 = getCapabilityManager();
      const manager2 = getCapabilityManager();

      expect(manager1).toBe(manager2);
    });

    it('should create CapabilityGuardedBridge with singleton manager', () => {
      const bridge1 = createCapabilityGuardedBridge('did:mycelix:alice');
      const bridge2 = createCapabilityGuardedBridge('did:mycelix:bob');

      // Grant capability from bridge1
      const cap = bridge1.grantCapability({
        granteeDid: 'did:mycelix:bob',
        resource: 'knowledge',
        operations: ['read'],
      });

      // Bridge2 should see the capability
      const bobCaps = bridge2.getMyCapabilities();
      expect(bobCaps.some((c) => c.id === cap.id)).toBe(true);
    });

    it('should reset singleton on resetCapabilityManager', () => {
      const manager1 = getCapabilityManager();
      manager1.createCapability({
        issuerDid: 'did:mycelix:alice',
        granteeDid: 'did:mycelix:bob',
        resource: 'knowledge',
        operations: ['read'],
      });

      resetCapabilityManager();

      const manager2 = getCapabilityManager();
      expect(
        manager2.getCapabilitiesForGrantee('did:mycelix:bob').length
      ).toBe(0);
    });
  });

  describe('Complex Scenarios', () => {
    it('should handle multi-level delegation chain', () => {
      // Alice grants to Bob
      const capAliceBob = manager.createCapability({
        issuerDid: 'did:mycelix:alice',
        granteeDid: 'did:mycelix:bob',
        resource: 'knowledge',
        operations: ['read', 'write', 'execute', 'delegate'],
        delegatable: true,
      });

      // Bob delegates to Carol
      const capBobCarol = manager.delegateCapability({
        parentCapabilityId: capAliceBob.id,
        delegateeDid: 'did:mycelix:carol',
        operations: ['read', 'write', 'delegate'],
        delegatable: true,
      })!;

      // Carol delegates to Dave
      const capCarolDave = manager.delegateCapability({
        parentCapabilityId: capBobCarol.id,
        delegateeDid: 'did:mycelix:dave',
        operations: ['read'],
        delegatable: false,
      })!;

      // Verify Dave's capability
      const invocation: CapabilityInvocation = {
        capabilityId: capCarolDave.id,
        operation: 'read',
        resource: 'knowledge',
        invokerDid: 'did:mycelix:dave',
        timestamp: Date.now(),
      };

      const result = manager.verifyCapability(invocation);
      expect(result.valid).toBe(true);

      // Check delegation chain
      expect(capCarolDave.delegationChain).toEqual([
        'did:mycelix:alice',
        'did:mycelix:bob',
        'did:mycelix:carol',
      ]);
    });

    it('should handle cross-hApp capability scenario', () => {
      // Identity hApp grants Knowledge hApp read access to DIDs
      const identityToKnowledge = manager.createCapability({
        issuerDid: 'did:mycelix:identity-happ',
        granteeDid: 'did:mycelix:knowledge-happ',
        resource: 'identity',
        operations: ['read'],
      });

      // Knowledge hApp grants Justice hApp read access
      const knowledgeToJustice = manager.createCapability({
        issuerDid: 'did:mycelix:knowledge-happ',
        granteeDid: 'did:mycelix:justice-happ',
        resource: 'knowledge',
        operations: ['read'],
      });

      // Justice hApp grants Finance hApp penalty execution
      const justiceToFinance = manager.createCapability({
        issuerDid: 'did:mycelix:justice-happ',
        granteeDid: 'did:mycelix:finance-happ',
        resource: 'justice',
        operations: ['execute'],
      });

      // All capabilities should be valid
      expect(
        manager.hasCapability('did:mycelix:knowledge-happ', 'identity', 'read')
      ).toBe(true);
      expect(
        manager.hasCapability('did:mycelix:justice-happ', 'knowledge', 'read')
      ).toBe(true);
      expect(
        manager.hasCapability('did:mycelix:finance-happ', 'justice', 'execute')
      ).toBe(true);
    });

    it('should handle rate limiting correctly', async () => {
      const capability = manager.createCapability({
        issuerDid: 'did:mycelix:alice',
        granteeDid: 'did:mycelix:bob',
        resource: 'knowledge',
        operations: ['read'],
        constraints: {
          rateLimit: {
            maxRequests: 3,
            windowMs: 10000, // 10 seconds
          },
        },
      });

      const invocation: CapabilityInvocation = {
        capabilityId: capability.id,
        operation: 'read',
        resource: 'knowledge',
        invokerDid: 'did:mycelix:bob',
        timestamp: Date.now(),
      };

      // First 3 requests should succeed
      expect(manager.verifyCapability(invocation).valid).toBe(true);
      expect(manager.verifyCapability(invocation).valid).toBe(true);
      expect(manager.verifyCapability(invocation).valid).toBe(true);

      // 4th request should fail due to rate limit
      const result = manager.verifyCapability(invocation);
      expect(result.valid).toBe(false);
      expect(result.reason).toBe('Rate limit exceeded');
    });

    it('should handle time window constraint', () => {
      const currentHour = new Date().getHours();

      // Create constraint that excludes current hour
      const capability = manager.createCapability({
        issuerDid: 'did:mycelix:alice',
        granteeDid: 'did:mycelix:bob',
        resource: 'knowledge',
        operations: ['read'],
        constraints: {
          timeWindow: {
            startHour: (currentHour + 2) % 24, // 2 hours from now
            endHour: (currentHour + 4) % 24, // 4 hours from now
          },
        },
      });

      const invocation: CapabilityInvocation = {
        capabilityId: capability.id,
        operation: 'read',
        resource: 'knowledge',
        invokerDid: 'did:mycelix:bob',
        timestamp: Date.now(),
      };

      const result = manager.verifyCapability(invocation);
      expect(result.valid).toBe(false);
      expect(result.reason).toBe(
        'Capability not valid during current time window'
      );
    });
  });
});
