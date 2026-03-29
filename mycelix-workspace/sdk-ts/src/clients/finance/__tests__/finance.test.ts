// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Finance Client Tests
 *
 * Verifies zome call arguments, response mapping, and operations for
 * CgcClient, TendClient, TreasuryClient, and PaymentsClient.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { CgcClient } from '../cgc';
import { TendClient } from '../tend';
import { TreasuryClient } from '../treasury';
import { PaymentsClient } from '../payments';
import type { AppClient } from '@holochain/client';

// ============================================================================
// MOCK HELPERS
// ============================================================================

function createMockClient(): AppClient {
  return {
    callZome: vi.fn(),
  } as unknown as AppClient;
}

function mockRecord(entry: Record<string, unknown>) {
  return {
    entry: { Present: entry },
    signed_action: { hashed: { hash: new Uint8Array(32) } },
  };
}

// ============================================================================
// MOCK ENTRIES (snake_case, matching Rust serde output)
// ============================================================================

const ALLOCATION_ENTRY = {
  member_did: 'did:mycelix:alice',
  cycle_id: 'cycle-2026-02',
  total_allocated: 10,
  remaining: 7,
  gifted: 3,
  expires_at: 1709510400,
};

const GIFT_ENTRY = {
  id: 'gift-001',
  giver_did: 'did:mycelix:alice',
  receiver_did: 'did:mycelix:bob',
  amount: 3,
  gratitude_message: 'Thanks for the documentation work!',
  contribution_type: 'Documentation',
  timestamp: 1708200000,
};

const EXCHANGE_ENTRY = {
  id: 'exchange-001',
  provider_did: 'did:mycelix:alice',
  receiver_did: 'did:mycelix:bob',
  hours: 2,
  service_description: 'Web development consultation',
  service_category: 'Technology',
  status: 'Proposed',
  timestamp: 1708200000,
};

const BALANCE_ENTRY = {
  member_did: 'did:mycelix:alice',
  dao_did: 'did:mycelix:my-dao',
  balance: 5.0,
  can_provide: true,
  can_receive: true,
  total_provided: 12.0,
  total_received: 7.0,
  exchange_count: 8,
};

const LISTING_ENTRY = {
  id: 'listing-001',
  provider_did: 'did:mycelix:alice',
  dao_did: 'did:mycelix:my-dao',
  title: 'Web Development',
  description: 'Full-stack web dev consulting',
  category: 'Technology',
  estimated_hours: 4,
  availability: 'Weekday evenings',
  active: true,
  created: 1708200000,
};

const TREASURY_ENTRY = {
  id: 'treasury-001',
  name: 'Community Fund',
  description: 'Main community treasury',
  currency: 'MYC',
  balance: 10000,
  reserve_ratio: 0.2,
  managers: ['did:mycelix:alice', 'did:mycelix:bob'],
  created: 1708200000,
  last_updated: 1708200000,
};

const CONTRIBUTION_ENTRY = {
  id: 'contrib-001',
  treasury_id: 'treasury-001',
  contributor_did: 'did:mycelix:carol',
  amount: 1000,
  currency: 'MYC',
  contribution_type: 'Donation',
  timestamp: 1708200000,
};

const PAYMENT_ENTRY = {
  id: 'pay-001',
  from_did: 'did:mycelix:alice',
  to_did: 'did:mycelix:bob',
  amount: 100,
  currency: 'MYC',
  payment_type: 'Direct',
  status: 'Completed',
  memo: 'Rent payment',
  created: 1708200000,
  completed: 1708200100,
};

const CHANNEL_ENTRY = {
  id: 'chan-001',
  party_a: 'did:mycelix:alice',
  party_b: 'did:mycelix:bob',
  currency: 'MYC',
  balance_a: 500,
  balance_b: 500,
  opened: 1708200000,
  last_updated: 1708200000,
};

// ============================================================================
// CGC CLIENT TESTS
// ============================================================================

describe('CgcClient', () => {
  let mockClient: AppClient;
  let cgc: CgcClient;

  beforeEach(() => {
    mockClient = createMockClient();
    cgc = new CgcClient(mockClient);
  });

  it('getOrCreateAllocation calls correct zome', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(ALLOCATION_ENTRY);

    const result = await cgc.getOrCreateAllocation('did:mycelix:alice');

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'finance',
      zome_name: 'cgc',
      fn_name: 'get_or_create_allocation',
      payload: 'did:mycelix:alice',
    });
    expect(result.remaining).toBe(7);
    expect(result.total_allocated).toBe(10);
  });

  it('giftCgc sends gift and returns record', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(GIFT_ENTRY);

    const input = {
      receiver_did: 'did:mycelix:bob',
      amount: 3,
      gratitude_message: 'Thanks for the documentation work!',
      contribution_type: 'Documentation' as const,
    };
    const result = await cgc.giftCgc(input);

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'finance',
      zome_name: 'cgc',
      fn_name: 'gift_cgc',
      payload: input,
    });
    expect(result.id).toBe('gift-001');
    expect(result.giver_did).toBe('did:mycelix:alice');
    expect(result.amount).toBe(3);
  });

  it('getGiftsGiven returns gift list', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([GIFT_ENTRY]);

    const result = await cgc.getGiftsGiven('did:mycelix:alice');

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'finance',
      zome_name: 'cgc',
      fn_name: 'get_gifts_given',
      payload: 'did:mycelix:alice',
    });
    expect(result).toHaveLength(1);
    expect(result[0].receiver_did).toBe('did:mycelix:bob');
  });
});

// ============================================================================
// TEND CLIENT TESTS
// ============================================================================

describe('TendClient', () => {
  let mockClient: AppClient;
  let tend: TendClient;

  beforeEach(() => {
    mockClient = createMockClient();
    tend = new TendClient(mockClient);
  });

  it('recordExchange calls correct zome', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(EXCHANGE_ENTRY);

    const input = {
      receiver_did: 'did:mycelix:bob',
      hours: 2,
      service_description: 'Web development consultation',
      service_category: 'Technology' as const,
      dao_did: 'did:mycelix:my-dao',
    };
    const result = await tend.recordExchange(input);

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'finance',
      zome_name: 'tend',
      fn_name: 'record_exchange',
      payload: input,
    });
    expect(result.id).toBe('exchange-001');
    expect(result.hours).toBe(2);
    expect(result.status).toBe('Proposed');
  });

  it('getBalance returns balance info', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(BALANCE_ENTRY);

    const input = { member_did: 'did:mycelix:alice', dao_did: 'did:mycelix:my-dao' };
    const result = await tend.getBalance(input);

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'finance',
      zome_name: 'tend',
      fn_name: 'get_balance',
      payload: input,
    });
    expect(result.balance).toBe(5.0);
    expect(result.can_provide).toBe(true);
    expect(result.exchange_count).toBe(8);
  });

  it('createListing creates service listing', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(LISTING_ENTRY);

    const input = {
      dao_did: 'did:mycelix:my-dao',
      title: 'Web Development',
      description: 'Full-stack web dev consulting',
      category: 'Technology' as const,
      estimated_hours: 4,
      availability: 'Weekday evenings',
    };
    const result = await tend.createListing(input);

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'finance',
      zome_name: 'tend',
      fn_name: 'create_listing',
      payload: input,
    });
    expect(result.id).toBe('listing-001');
    expect(result.active).toBe(true);
  });
});

// ============================================================================
// TREASURY CLIENT TESTS
// ============================================================================

describe('TreasuryClient', () => {
  let mockClient: AppClient;
  let treasury: TreasuryClient;

  beforeEach(() => {
    mockClient = createMockClient();
    treasury = new TreasuryClient(mockClient);
  });

  it('createTreasury extracts entry from record', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockRecord(TREASURY_ENTRY)
    );

    const input = {
      name: 'Community Fund',
      description: 'Main community treasury',
      currency: 'MYC',
      reserve_ratio: 0.2,
      managers: ['did:mycelix:alice', 'did:mycelix:bob'],
    };
    const result = await treasury.createTreasury(input);

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'finance',
      zome_name: 'treasury',
      fn_name: 'create_treasury',
      payload: input,
    });
    expect(result.id).toBe('treasury-001');
    expect(result.balance).toBe(10000);
    expect(result.managers).toHaveLength(2);
  });

  it('getManagerTreasuries maps records to entries', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
      mockRecord(TREASURY_ENTRY),
    ]);

    const result = await treasury.getManagerTreasuries('did:mycelix:alice');

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'finance',
      zome_name: 'treasury',
      fn_name: 'get_manager_treasuries',
      payload: 'did:mycelix:alice',
    });
    expect(result).toHaveLength(1);
    expect(result[0].name).toBe('Community Fund');
  });

  it('contribute creates contribution record', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockRecord(CONTRIBUTION_ENTRY)
    );

    const input = {
      treasury_id: 'treasury-001',
      contributor_did: 'did:mycelix:carol',
      amount: 1000,
      currency: 'MYC',
      contribution_type: 'Donation' as const,
    };
    const result = await treasury.contribute(input);

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'finance',
      zome_name: 'treasury',
      fn_name: 'contribute',
      payload: input,
    });
    expect(result.id).toBe('contrib-001');
    expect(result.amount).toBe(1000);
  });
});

// ============================================================================
// PAYMENTS CLIENT TESTS
// ============================================================================

describe('PaymentsClient', () => {
  let mockClient: AppClient;
  let payments: PaymentsClient;

  beforeEach(() => {
    mockClient = createMockClient();
    payments = new PaymentsClient(mockClient);
  });

  it('sendPayment extracts entry from record', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockRecord(PAYMENT_ENTRY)
    );

    const input = {
      from_did: 'did:mycelix:alice',
      to_did: 'did:mycelix:bob',
      amount: 100,
      currency: 'MYC',
      payment_type: 'Direct' as const,
      memo: 'Rent payment',
    };
    const result = await payments.sendPayment(input);

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'finance',
      zome_name: 'payments',
      fn_name: 'send_payment',
      payload: input,
    });
    expect(result.id).toBe('pay-001');
    expect(result.amount).toBe(100);
    expect(result.status).toBe('Completed');
  });

  it('getPaymentHistory maps records', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
      mockRecord(PAYMENT_ENTRY),
    ]);

    const result = await payments.getPaymentHistory('did:mycelix:alice');

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'finance',
      zome_name: 'payments',
      fn_name: 'get_payment_history',
      payload: 'did:mycelix:alice',
    });
    expect(result).toHaveLength(1);
    expect(result[0].from_did).toBe('did:mycelix:alice');
  });

  it('openPaymentChannel creates channel', async () => {
    (mockClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockRecord(CHANNEL_ENTRY)
    );

    const input = {
      party_a: 'did:mycelix:alice',
      party_b: 'did:mycelix:bob',
      currency: 'MYC',
      initial_deposit_a: 500,
      initial_deposit_b: 500,
    };
    const result = await payments.openPaymentChannel(input);

    expect(mockClient.callZome).toHaveBeenCalledWith({
      role_name: 'finance',
      zome_name: 'payments',
      fn_name: 'open_payment_channel',
      payload: input,
    });
    expect(result.id).toBe('chan-001');
    expect(result.balance_a).toBe(500);
    expect(result.balance_b).toBe(500);
  });
});
