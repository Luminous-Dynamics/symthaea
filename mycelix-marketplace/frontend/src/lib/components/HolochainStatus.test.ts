// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// @vitest-environment jsdom

import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';

vi.mock('$lib/holochain', () => ({
  initHolochainClient: vi.fn().mockResolvedValue({}),
}));

import { render, screen, fireEvent, cleanup } from '@testing-library/svelte';
import { tick } from 'svelte';
import HolochainStatus from './HolochainStatus.svelte';
import { holochain } from '$lib/stores';
import { initHolochainClient } from '$lib/holochain';

afterEach(() => {
  cleanup();
});

describe('HolochainStatus', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    holochain.reset();
  });

  it('shows a banner when not connected', async () => {
    render(HolochainStatus);
    await tick();

    expect(screen.getByText(/Not connected to Holochain/i)).toBeTruthy();
    expect(screen.getByText(/Retry connection/i)).toBeTruthy();
  });

  it('shows error details when present', async () => {
    render(HolochainStatus);
    holochain.setError('bad socket');
    holochain.setStatus('error');
    await tick();

    expect(screen.getByText(/Holochain connection failed/i)).toBeTruthy();
    expect(screen.getByText(/bad socket/i)).toBeTruthy();
  });

  it('invokes reconnect when retry is clicked', async () => {
    render(HolochainStatus);
    await tick();

    const button = screen.getByRole('button', { name: /retry connection/i });
    await fireEvent.click(button);

    expect(initHolochainClient).toHaveBeenCalledTimes(1);
  });
});
