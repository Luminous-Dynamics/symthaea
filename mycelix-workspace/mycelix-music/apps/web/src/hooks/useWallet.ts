// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { useState, useEffect, useCallback } from 'react';
import { usePrivy, useWallets } from '@privy-io/react-auth';
import { ethers } from 'ethers';

/**
 * Wallet hook that integrates Privy authentication with ethers.js
 * Provides wallet connection state and ethers signer for transactions
 */

interface WalletState {
  address: string | null;
  connected: boolean;
  connecting: boolean;
  chainId: number | null;
  signer: ethers.Signer | null;
  provider: ethers.BrowserProvider | null;
}

interface UseWalletReturn extends WalletState {
  connect: () => Promise<void>;
  disconnect: () => Promise<void>;
  switchChain: (chainId: number) => Promise<void>;
  getSigner: () => Promise<ethers.Signer | null>;
}

// Supported chains
const SUPPORTED_CHAINS: Record<number, string> = {
  1: 'Ethereum Mainnet',
  100: 'Gnosis Chain',
  31337: 'Localhost',
};

export function useWallet(): UseWalletReturn {
  const { ready, authenticated, login, logout, user } = usePrivy();
  const { wallets } = useWallets();

  const [state, setState] = useState<WalletState>({
    address: null,
    connected: false,
    connecting: false,
    chainId: null,
    signer: null,
    provider: null,
  });

  // Get the embedded wallet or first connected wallet
  const activeWallet = wallets.find((w) => w.walletClientType === 'privy') || wallets[0];

  // Initialize provider and signer when wallet connects
  useEffect(() => {
    async function initializeWallet() {
      if (!ready || !authenticated || !activeWallet) {
        setState({
          address: null,
          connected: false,
          connecting: false,
          chainId: null,
          signer: null,
          provider: null,
        });
        return;
      }

      try {
        // Get the ethereum provider from the wallet
        const ethereumProvider = await activeWallet.getEthereumProvider();
        const provider = new ethers.BrowserProvider(ethereumProvider);
        const signer = await provider.getSigner();
        const network = await provider.getNetwork();

        setState({
          address: activeWallet.address,
          connected: true,
          connecting: false,
          chainId: Number(network.chainId),
          signer,
          provider,
        });
      } catch (error) {
        console.error('Failed to initialize wallet:', error);
        setState({
          address: activeWallet.address,
          connected: true,
          connecting: false,
          chainId: null,
          signer: null,
          provider: null,
        });
      }
    }

    initializeWallet();
  }, [ready, authenticated, activeWallet]);

  const connect = useCallback(async () => {
    if (authenticated) return;

    setState((prev) => ({ ...prev, connecting: true }));

    try {
      await login();
    } catch (error) {
      console.error('Failed to connect wallet:', error);
      setState((prev) => ({ ...prev, connecting: false }));
    }
  }, [authenticated, login]);

  const disconnect = useCallback(async () => {
    try {
      await logout();
      setState({
        address: null,
        connected: false,
        connecting: false,
        chainId: null,
        signer: null,
        provider: null,
      });
    } catch (error) {
      console.error('Failed to disconnect wallet:', error);
    }
  }, [logout]);

  const switchChain = useCallback(async (chainId: number) => {
    if (!activeWallet) {
      throw new Error('No wallet connected');
    }

    if (!SUPPORTED_CHAINS[chainId]) {
      throw new Error(`Chain ${chainId} is not supported`);
    }

    try {
      await activeWallet.switchChain(chainId);

      // Reinitialize provider after chain switch
      const ethereumProvider = await activeWallet.getEthereumProvider();
      const provider = new ethers.BrowserProvider(ethereumProvider);
      const signer = await provider.getSigner();

      setState((prev) => ({
        ...prev,
        chainId,
        provider,
        signer,
      }));
    } catch (error) {
      console.error('Failed to switch chain:', error);
      throw error;
    }
  }, [activeWallet]);

  const getSigner = useCallback(async (): Promise<ethers.Signer | null> => {
    if (!activeWallet) return null;

    try {
      const ethereumProvider = await activeWallet.getEthereumProvider();
      const provider = new ethers.BrowserProvider(ethereumProvider);
      return await provider.getSigner();
    } catch (error) {
      console.error('Failed to get signer:', error);
      return null;
    }
  }, [activeWallet]);

  return {
    address: state.address,
    connected: state.connected,
    connecting: state.connecting,
    chainId: state.chainId,
    signer: state.signer,
    provider: state.provider,
    connect,
    disconnect,
    switchChain,
    getSigner,
  };
}

/**
 * Hook to get the current chain name
 */
export function useChainName(): string | null {
  const { chainId } = useWallet();
  return chainId ? SUPPORTED_CHAINS[chainId] || `Unknown Chain (${chainId})` : null;
}

/**
 * Hook to check if on a supported chain
 */
export function useIsSupported(): boolean {
  const { chainId } = useWallet();
  return chainId !== null && chainId in SUPPORTED_CHAINS;
}
