// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { renderHook, act } from '@testing-library/react';

// Mock the Privy hooks before importing useWallet
jest.mock('@privy-io/react-auth', () => ({
  usePrivy: jest.fn(),
  useWallets: jest.fn(),
}));

jest.mock('ethers', () => ({
  ethers: {
    BrowserProvider: jest.fn(() => ({
      getSigner: jest.fn().mockResolvedValue({ address: '0x123' }),
      getNetwork: jest.fn().mockResolvedValue({ chainId: 100n }),
    })),
  },
}));

import { useWallet, useChainName, useIsSupported } from '../useWallet';
import { usePrivy, useWallets } from '@privy-io/react-auth';

describe('useWallet', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('returns disconnected state when not authenticated', () => {
    (usePrivy as jest.Mock).mockReturnValue({
      ready: true,
      authenticated: false,
      login: jest.fn(),
      logout: jest.fn(),
      user: null,
    });
    (useWallets as jest.Mock).mockReturnValue({ wallets: [] });

    const { result } = renderHook(() => useWallet());

    expect(result.current.connected).toBe(false);
    expect(result.current.address).toBeNull();
    expect(result.current.connecting).toBe(false);
  });

  it('returns connected state with wallet when authenticated', async () => {
    const mockWallet = {
      address: '0x1234567890abcdef1234567890abcdef12345678',
      walletClientType: 'privy',
      getEthereumProvider: jest.fn().mockResolvedValue({}),
      switchChain: jest.fn(),
    };

    (usePrivy as jest.Mock).mockReturnValue({
      ready: true,
      authenticated: true,
      login: jest.fn(),
      logout: jest.fn(),
      user: { wallet: { address: mockWallet.address } },
    });
    (useWallets as jest.Mock).mockReturnValue({ wallets: [mockWallet] });

    const { result } = renderHook(() => useWallet());

    // Wait for useEffect to complete
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(result.current.address).toBe(mockWallet.address);
  });

  it('calls login when connect is invoked', async () => {
    const mockLogin = jest.fn();

    (usePrivy as jest.Mock).mockReturnValue({
      ready: true,
      authenticated: false,
      login: mockLogin,
      logout: jest.fn(),
      user: null,
    });
    (useWallets as jest.Mock).mockReturnValue({ wallets: [] });

    const { result } = renderHook(() => useWallet());

    await act(async () => {
      await result.current.connect();
    });

    expect(mockLogin).toHaveBeenCalled();
  });

  it('calls logout when disconnect is invoked', async () => {
    const mockLogout = jest.fn();

    (usePrivy as jest.Mock).mockReturnValue({
      ready: true,
      authenticated: true,
      login: jest.fn(),
      logout: mockLogout,
      user: { wallet: { address: '0x123' } },
    });
    (useWallets as jest.Mock).mockReturnValue({ wallets: [] });

    const { result } = renderHook(() => useWallet());

    await act(async () => {
      await result.current.disconnect();
    });

    expect(mockLogout).toHaveBeenCalled();
  });
});

describe('useChainName', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('returns null when not connected', () => {
    (usePrivy as jest.Mock).mockReturnValue({
      ready: true,
      authenticated: false,
      login: jest.fn(),
      logout: jest.fn(),
      user: null,
    });
    (useWallets as jest.Mock).mockReturnValue({ wallets: [] });

    const { result } = renderHook(() => useChainName());

    expect(result.current).toBeNull();
  });
});

describe('useIsSupported', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('returns false when not connected', () => {
    (usePrivy as jest.Mock).mockReturnValue({
      ready: true,
      authenticated: false,
      login: jest.fn(),
      logout: jest.fn(),
      user: null,
    });
    (useWallets as jest.Mock).mockReturnValue({ wallets: [] });

    const { result } = renderHook(() => useIsSupported());

    expect(result.current).toBe(false);
  });
});
