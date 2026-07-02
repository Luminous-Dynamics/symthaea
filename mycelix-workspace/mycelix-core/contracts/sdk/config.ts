// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Contract Configuration
 *
 * This file contains contract addresses and configuration for the Mycelix SDK.
 * Update addresses after deployment to each network.
 */

export interface NetworkConfig {
  chainId: number;
  rpcUrl: string;
  contracts: {
    registry: string;
    reputation: string;
    payment: string;
  };
  blockExplorer?: string;
}

export const networks: Record<string, NetworkConfig> = {
  // Local development (Anvil)
  local: {
    chainId: 31337,
    rpcUrl: "http://127.0.0.1:8545",
    contracts: {
      registry: "0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0",
      reputation: "0x5FbDB2315678afecb367f032d93F642f64180aa3",
      payment: "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512",
    },
  },

  // Ethereum Sepolia Testnet - LIVE (Deployed 2026-01-08)
  sepolia: {
    chainId: 11155111,
    rpcUrl: "https://ethereum-sepolia-rpc.publicnode.com",
    contracts: {
      registry: "0x556b810371e3d8D9E5753117514F03cC6C93b835",
      reputation: "0xf3B343888a9b82274cEfaa15921252DB6c5f48C9",
      payment: "0x94417A3645824CeBDC0872c358cf810e4Ce4D1cB",
    },
    blockExplorer: "https://sepolia.etherscan.io",
  },

  // Polygon Amoy Testnet
  "polygon-amoy": {
    chainId: 80002,
    rpcUrl: "https://rpc-amoy.polygon.technology",
    contracts: {
      registry: "0x0000000000000000000000000000000000000000",
      reputation: "0x0000000000000000000000000000000000000000",
      payment: "0x0000000000000000000000000000000000000000",
    },
    blockExplorer: "https://amoy.polygonscan.com",
  },

  // Polygon Mainnet
  polygon: {
    chainId: 137,
    rpcUrl: "https://polygon-rpc.com",
    contracts: {
      registry: "0x0000000000000000000000000000000000000000",
      reputation: "0x0000000000000000000000000000000000000000",
      payment: "0x0000000000000000000000000000000000000000",
    },
    blockExplorer: "https://polygonscan.com",
  },
};

export const defaultNetwork = "sepolia";

export function getNetworkConfig(network: string = defaultNetwork): NetworkConfig {
  const config = networks[network];
  if (!config) {
    throw new Error(`Unknown network: ${network}. Available: ${Object.keys(networks).join(", ")}`);
  }
  return config;
}

export function getContractAddress(
  contract: keyof NetworkConfig["contracts"],
  network: string = defaultNetwork
): string {
  const config = getNetworkConfig(network);
  return config.contracts[contract];
}
