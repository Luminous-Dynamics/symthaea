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
import ConnectionNotice from './ConnectionNotice.svelte';
import { holochain } from '$lib/stores';
import { initHolochainClient } from '$lib/holochain';

afterEach(() => {
  cleanup();
});

describe('ConnectionNotice', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    holochain.reset();
  });

  it('shows disconnected copy by default', async () => {
    render(ConnectionNotice);
    await tick();

    expect(screen.getByText(/Not connected to Holochain/i)).toBeTruthy();
    expect(screen.getByRole('button', { name: /retry/i })).toBeTruthy();
  });

  it('shows connecting state and disables retry', async () => {
    render(ConnectionNotice, { showWhenConnecting: true });
    holochain.setStatus('connecting');
    await tick();

    const button = screen.getByRole('button', { name: /Connecting…/i });
    expect(button).toBeTruthy();
    expect((button as HTMLButtonElement).disabled).toBe(true);
  });

  it('invokes initHolochainClient on retry', async () => {
    holochain.setUrl('ws://test-url');
    render(ConnectionNotice);
    await tick();

    const button = screen.getByRole('button', { name: /retry/i });
    await fireEvent.click(button);

    expect(initHolochainClient).toHaveBeenCalledTimes(1);
    expect(initHolochainClient).toHaveBeenCalledWith('ws://test-url');
  });
});
