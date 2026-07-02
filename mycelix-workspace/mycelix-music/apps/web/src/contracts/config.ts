// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Web3 Configuration for Mycelix
 *
 * Configures wagmi chains, transports, and contract addresses.
 */

import { http, createConfig } from 'wagmi';
import { mainnet, sepolia, base, baseSepolia, optimism, optimismSepolia } from 'wagmi/chains';
import { coinbaseWallet, injected, walletConnect } from 'wagmi/connectors';

// ============================================================================
// Contract Addresses by Chain
// ============================================================================

export interface ContractAddresses {
  SoulRegistry: `0x${string}`;
  ProofOfPresence: `0x${string}`;
  PatronageRegistry?: `0x${string}`;
  SporeToken?: `0x${string}`;
  NutrientFlow?: `0x${string}`;
}

// Addresses will be populated after deployment
export const CONTRACT_ADDRESSES: Record<number, ContractAddresses> = {
  // Sepolia (Ethereum testnet)
  [sepolia.id]: {
    SoulRegistry: '0x0000000000000000000000000000000000000000',
    ProofOfPresence: '0x0000000000000000000000000000000000000000',
  },

  // Base Sepolia (testnet)
  [baseSepolia.id]: {
    SoulRegistry: '0x0000000000000000000000000000000000000000',
    ProofOfPresence: '0x0000000000000000000000000000000000000000',
  },

  // Optimism Sepolia (testnet)
  [optimismSepolia.id]: {
    SoulRegistry: '0x0000000000000000000000000000000000000000',
    ProofOfPresence: '0x0000000000000000000000000000000000000000',
  },

  // Mainnet (future)
  [mainnet.id]: {
    SoulRegistry: '0x0000000000000000000000000000000000000000',
    ProofOfPresence: '0x0000000000000000000000000000000000000000',
  },

  // Base (future)
  [base.id]: {
    SoulRegistry: '0x0000000000000000000000000000000000000000',
    ProofOfPresence: '0x0000000000000000000000000000000000000000',
  },

  // Optimism (future)
  [optimism.id]: {
    SoulRegistry: '0x0000000000000000000000000000000000000000',
    ProofOfPresence: '0x0000000000000000000000000000000000000000',
  },
};

// ============================================================================
// Wagmi Configuration
// ============================================================================

const projectId = process.env.NEXT_PUBLIC_WALLETCONNECT_PROJECT_ID || '';

export const wagmiConfig = createConfig({
  chains: [
    // Testnets first for development
    baseSepolia,
    optimismSepolia,
    sepolia,
    // Mainnets
    base,
    optimism,
    mainnet,
  ],

  connectors: [
    injected(),
    coinbaseWallet({
      appName: 'Mycelix',
      preference: 'smartWalletOnly', // Enable smart wallet for gasless
    }),
    walletConnect({ projectId }),
  ],

  transports: {
    [mainnet.id]: http(),
    [sepolia.id]: http(),
    [base.id]: http(),
    [baseSepolia.id]: http(),
    [optimism.id]: http(),
    [optimismSepolia.id]: http(),
  },

  ssr: true,
});

// ============================================================================
// Helper Functions
// ============================================================================

export function getContractAddress(
  chainId: number,
  contract: keyof ContractAddresses
): `0x${string}` | undefined {
  return CONTRACT_ADDRESSES[chainId]?.[contract];
}

export function getSupportedChainIds(): number[] {
  return Object.keys(CONTRACT_ADDRESSES).map(Number);
}

export function isChainSupported(chainId: number): boolean {
  return chainId in CONTRACT_ADDRESSES;
}

// Default to Base Sepolia for development
export const DEFAULT_CHAIN_ID = baseSepolia.id;
