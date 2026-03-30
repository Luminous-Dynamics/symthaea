// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * DAO Delegation Module Tests
 *
 * Tests for DelegationClient - liquid democracy and vote delegation
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  DelegationClient,
  MAX_DELEGATION_DEPTH,
  MAX_POWER_CONCENTRATION,
} from '../../src/dao/delegation';
import type { DAOClient } from '../../src/dao/client';
import type { Delegation, DelegationDomain, DelegatedPowerResult } from '../../src/dao/types';

// Mock DAOClient
const createMockClient = () => {
  const callZome = vi.fn();
  const getZomeName = vi.fn().mockReturnValue('governance');
  const getMyPubKey = vi.fn().mockResolvedValue(new Uint8Array(32));

  return {
    callZome,
    getZomeName,
    getMyPubKey,
  } as unknown as DAOClient;
};

// Mock delegation data
const mockDelegatorKey = new Uint8Array(32);
const mockDelegateKey = new Uint8Array(Array(32).fill(1));

const mockDelegation: Delegation = {
  delegator: mockDelegatorKey,
  delegate: mockDelegateKey,
  domain: 'All',
  created_at: Date.now() * 1000,
  revoked_at: null,
};

const mockRecord = {
  signed_action: { hashed: { hash: new Uint8Array(32) } },
  entry: { Present: { entry: mockDelegation } },
};

const mockPowerResult: DelegatedPowerResult = {
  direct_power: 100,
  transitive_power: 50,
  total_power: 150,
  delegator_count: 2,
};

describe('DelegationClient', () => {
  let client: DAOClient;
  let delegationClient: DelegationClient;

  beforeEach(() => {
    client = createMockClient();
    delegationClient = new DelegationClient(client);
  });

  describe('constants', () => {
    it('should define maximum delegation depth', () => {
      expect(MAX_DELEGATION_DEPTH).toBe(5);
    });

    it('should define maximum power concentration', () => {
      expect(MAX_POWER_CONCENTRATION).toBe(0.15);
    });
  });

  describe('delegate', () => {
    beforeEach(() => {
      // Mock chain check (no cycle)
      (client.callZome as any).mockImplementation((zome: string, fn: string) => {
        if (fn === 'get_delegation_chain') return Promise.resolve([]);
        if (fn === 'get_delegated_power') return Promise.resolve(mockPowerResult);
        if (fn === 'get_power_distribution') return Promise.resolve({ total_power: 10000 });
        if (fn === 'create_delegation') return Promise.resolve(mockRecord);
        return Promise.resolve(null);
      });
    });

    it('should create a delegation to all domains', async () => {
      const result = await delegationClient.delegate(mockDelegateKey, 'All');

      expect(result.domain).toBe('All');
      expect(client.callZome).toHaveBeenCalledWith(
        'governance',
        'create_delegation',
        expect.objectContaining({ delegate: mockDelegateKey, domain: 'All' })
      );
    });

    it('should create a domain-specific delegation', async () => {
      const domains: DelegationDomain[] = [
        'Governance',
        'Technical',
        'Economic',
        'Social',
        'Cultural',
      ];

      for (const domain of domains) {
        (client.callZome as any).mockImplementation((zome: string, fn: string) => {
          if (fn === 'get_delegation_chain') return Promise.resolve([]);
          if (fn === 'get_delegated_power') return Promise.resolve(mockPowerResult);
          if (fn === 'get_power_distribution') return Promise.resolve({ total_power: 10000 });
          if (fn === 'create_delegation')
            return Promise.resolve({
              ...mockRecord,
              entry: { Present: { entry: { ...mockDelegation, domain } } },
            });
          return Promise.resolve(null);
        });

        const result = await delegationClient.delegate(mockDelegateKey, domain);
        expect(result.domain).toBe(domain);
      }
    });

    it('should prevent delegation that would create cycle', async () => {
      // Mock cycle detection - delegate already delegates back to us
      (client.callZome as any).mockImplementation((zome: string, fn: string) => {
        if (fn === 'get_delegation_chain') return Promise.resolve([mockDelegatorKey]); // Cycle!
        return Promise.resolve(null);
      });

      await expect(delegationClient.delegate(mockDelegateKey)).rejects.toThrow(
        'would create a cycle'
      );
    });

    it('should prevent delegation that exceeds power concentration', async () => {
      // Mock high power concentration
      (client.callZome as any).mockImplementation((zome: string, fn: string) => {
        if (fn === 'get_delegation_chain') return Promise.resolve([]);
        if (fn === 'get_delegated_power')
          return Promise.resolve({ ...mockPowerResult, total_power: 5000 });
        if (fn === 'get_power_distribution') return Promise.resolve({ total_power: 10000 }); // Would be 50%
        return Promise.resolve(null);
      });

      await expect(delegationClient.delegate(mockDelegateKey)).rejects.toThrow(
        'power concentration limit'
      );
    });
  });

  describe('revoke', () => {
    it('should revoke a delegation', async () => {
      (client.callZome as any).mockResolvedValue(undefined);

      await delegationClient.revoke(mockDelegateKey, 'All');

      expect(client.callZome).toHaveBeenCalledWith('governance', 'revoke_delegation', {
        delegate: mockDelegateKey,
        domain: 'All',
      });
    });

    it('should revoke without domain', async () => {
      (client.callZome as any).mockResolvedValue(undefined);

      await delegationClient.revoke(mockDelegateKey);

      expect(client.callZome).toHaveBeenCalledWith('governance', 'revoke_delegation', {
        delegate: mockDelegateKey,
        domain: undefined,
      });
    });
  });

  describe('revokeAll', () => {
    it('should revoke all delegations', async () => {
      (client.callZome as any).mockResolvedValue(undefined);

      await delegationClient.revokeAll();

      expect(client.callZome).toHaveBeenCalledWith('governance', 'revoke_all_delegations', null);
    });
  });

  describe('getMyDelegations', () => {
    it('should retrieve delegations from current user', async () => {
      (client.callZome as any).mockResolvedValue([mockRecord, mockRecord]);

      const result = await delegationClient.getMyDelegations();

      expect(result).toHaveLength(2);
    });

    it('should return empty array when no delegations exist', async () => {
      (client.callZome as any).mockResolvedValue([]);

      const result = await delegationClient.getMyDelegations();

      expect(result).toEqual([]);
    });
  });

  describe('getDelegationsTo', () => {
    it('should retrieve delegations to an agent', async () => {
      (client.callZome as any).mockResolvedValue([mockRecord]);

      const result = await delegationClient.getDelegationsTo(mockDelegateKey);

      expect(result).toHaveLength(1);
    });
  });

  describe('getVotingPower', () => {
    it('should get voting power for current user', async () => {
      (client.callZome as any).mockResolvedValue(mockPowerResult);

      const result = await delegationClient.getVotingPower();

      expect(result.direct_power).toBe(100);
      expect(result.transitive_power).toBe(50);
      expect(result.total_power).toBe(150);
      expect(result.delegator_count).toBe(2);
    });

    it('should get voting power for specific agent', async () => {
      (client.callZome as any).mockResolvedValue(mockPowerResult);

      const result = await delegationClient.getVotingPower(mockDelegateKey, 'Technical');

      expect(client.callZome).toHaveBeenCalledWith('governance', 'get_delegated_power', {
        agent: mockDelegateKey,
        domain: 'Technical',
      });
    });
  });

  describe('getDelegationChain', () => {
    it('should get delegation chain', async () => {
      const chain = [mockDelegateKey, new Uint8Array(Array(32).fill(2))];
      (client.callZome as any).mockResolvedValue(chain);

      const result = await delegationClient.getDelegationChain(mockDelegatorKey);

      expect(result).toHaveLength(2);
    });
  });

  describe('domain helpers', () => {
    it('should return all domains', () => {
      const domains = delegationClient.getDomains();
      expect(domains).toContain('All');
      expect(domains).toContain('Technical');
      expect(domains).toContain('Economic');
    });

    it('should check domain subset correctly', () => {
      expect(delegationClient.isDomainSubset('Technical', 'All')).toBe(true);
      expect(delegationClient.isDomainSubset('Technical', 'Technical')).toBe(true);
      expect(delegationClient.isDomainSubset('Technical', 'Economic')).toBe(false);
    });

    it('should describe domains', () => {
      expect(delegationClient.describeDomain('All')).toContain('All');
      expect(delegationClient.describeDomain('Technical')).toContain('Protocol');
    });
  });

  describe('error handling', () => {
    it('should propagate network errors', async () => {
      (client.callZome as any).mockRejectedValue(new Error('Connection lost'));

      await expect(delegationClient.getMyDelegations()).rejects.toThrow('Connection lost');
    });
  });
});
