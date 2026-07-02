// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Music Network Configuration
 *
 * Contains deployed contract addresses for each supported network
 */

export interface NetworkConfig {
  chainId: number;
  name: string;
  routerAddress: string;
  flowTokenAddress: string;
  cgcRegistryAddress: string;
  strategies: {
    payPerStream: string;
    giftEconomy: string;
    freemium: string;
    patronage: string;
    nftGated: string;
    subscription: string;
    payPerDownload: string;
    stakingGated: string;
    tokenTip: string;
    timeBarter: string;
    auction: string;
    payWhatYouWant: string;
  };
  rpcUrl: string;
  blockExplorer?: string;
}

// Gnosis Chain (Production)
export const GNOSIS_MAINNET: NetworkConfig = {
  chainId: 100,
  name: 'Gnosis Chain',
  routerAddress: process.env.NEXT_PUBLIC_ROUTER_ADDRESS || '',
  flowTokenAddress: process.env.FLOW_TOKEN_ADDRESS || '',
  cgcRegistryAddress: process.env.CGC_REGISTRY_ADDRESS || '',
  strategies: {
    payPerStream: process.env.PAY_PER_STREAM_ADDRESS || '',
    giftEconomy: process.env.GIFT_ECONOMY_ADDRESS || '',
    freemium: process.env.FREEMIUM_ADDRESS || '',
    patronage: process.env.PATRONAGE_ADDRESS || '',
    nftGated: process.env.NFT_GATED_ADDRESS || '',
    subscription: process.env.SUBSCRIPTION_ADDRESS || '',
    payPerDownload: process.env.PAY_PER_DOWNLOAD_ADDRESS || '',
    stakingGated: process.env.STAKING_GATED_ADDRESS || '',
    tokenTip: process.env.TOKEN_TIP_ADDRESS || '',
    timeBarter: process.env.TIME_BARTER_ADDRESS || '',
    auction: process.env.AUCTION_ADDRESS || '',
    payWhatYouWant: process.env.PAY_WHAT_YOU_WANT_ADDRESS || '',
  },
  rpcUrl: 'https://rpc.gnosischain.com',
  blockExplorer: 'https://gnosisscan.io',
};

// Chiado Testnet
export const CHIADO_TESTNET: NetworkConfig = {
  chainId: 10200,
  name: 'Chiado Testnet',
  routerAddress: process.env.TESTNET_ROUTER_ADDRESS || '',
  flowTokenAddress: process.env.TESTNET_FLOW_TOKEN_ADDRESS || '',
  cgcRegistryAddress: process.env.TESTNET_CGC_REGISTRY_ADDRESS || '',
  strategies: {
    payPerStream: '',
    giftEconomy: '',
    freemium: '',
    patronage: '',
    nftGated: '',
    subscription: '',
    payPerDownload: '',
    stakingGated: '',
    tokenTip: '',
    timeBarter: '',
    auction: '',
    payWhatYouWant: '',
  },
  rpcUrl: 'https://rpc.chiado.gnosis.gateway.fm',
  blockExplorer: 'https://gnosis-chiado.blockscout.com',
};

// Localhost (Development)
export const LOCALHOST: NetworkConfig = {
  chainId: 31337,
  name: 'Localhost',
  routerAddress: '0x5FbDB2315678afecb367f032d93F642f64180aa3', // Default Anvil deployment
  flowTokenAddress: '0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512',
  cgcRegistryAddress: '0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0',
  strategies: {
    payPerStream: '0xCf7Ed3AccA5a467e9e704C703E8D87F634fB0Fc9',
    giftEconomy: '0xDc64a140Aa3E981100a9becA4E685f962f0cF6C9',
    freemium: '0x5FC8d32690cc91D4c39d9d3abcBD16989F875707',
    patronage: '0x0165878A594ca255338adfa4d48449f69242Eb8F',
    nftGated: '0xa513E6E4b8f2a923D98304ec87F64353C4D5C853',
    subscription: '0x2279B7A0a67DB372996a5FaB50D91eAA73d2eBe6',
    payPerDownload: '0x8A791620dd6260079BF849Dc5567aDC3F2FdC318',
    stakingGated: '0x610178dA211FEF7D417bC0e6FeD39F05609AD788',
    tokenTip: '0xB7f8BC63BbcaD18155201308C8f3540b07f84F5e',
    timeBarter: '0xA51c1fc2f0D1a1b8494Ed1FE312d7C3a78Ed91C0',
    auction: '0x0DCd1Bf9A1b36cE34237eEaFef220932846BCD82',
    payWhatYouWant: '0x9A676e781A523b5d0C0e43731313A708CB607508',
  },
  rpcUrl: 'http://localhost:8545',
};

// Ethereum Mainnet (Future)
export const ETHEREUM_MAINNET: NetworkConfig = {
  chainId: 1,
  name: 'Ethereum Mainnet',
  routerAddress: '', // Not deployed yet
  flowTokenAddress: '',
  cgcRegistryAddress: '',
  strategies: {
    payPerStream: '',
    giftEconomy: '',
    freemium: '',
    patronage: '',
    nftGated: '',
    subscription: '',
    payPerDownload: '',
    stakingGated: '',
    tokenTip: '',
    timeBarter: '',
    auction: '',
    payWhatYouWant: '',
  },
  rpcUrl: 'https://eth.llamarpc.com',
  blockExplorer: 'https://etherscan.io',
};

// Network registry by chain ID
export const NETWORKS: Record<number, NetworkConfig> = {
  1: ETHEREUM_MAINNET,
  100: GNOSIS_MAINNET,
  10200: CHIADO_TESTNET,
  31337: LOCALHOST,
};

/**
 * Get network configuration by chain ID
 */
export function getNetwork(chainId: number): NetworkConfig | undefined {
  return NETWORKS[chainId];
}

/**
 * Get router address for a chain
 */
export function getRouterAddress(chainId: number): string {
  const network = NETWORKS[chainId];
  if (!network) {
    throw new Error(`Unsupported chain ID: ${chainId}`);
  }
  if (!network.routerAddress) {
    throw new Error(`Router not deployed on ${network.name}`);
  }
  return network.routerAddress;
}

/**
 * Get strategy address for a chain and strategy type
 */
export function getStrategyAddress(
  chainId: number,
  strategy: keyof NetworkConfig['strategies']
): string {
  const network = NETWORKS[chainId];
  if (!network) {
    throw new Error(`Unsupported chain ID: ${chainId}`);
  }
  const address = network.strategies[strategy];
  if (!address) {
    throw new Error(`Strategy ${strategy} not deployed on ${network.name}`);
  }
  return address;
}

/**
 * Check if a network is supported
 */
export function isNetworkSupported(chainId: number): boolean {
  return chainId in NETWORKS;
}

/**
 * Get all supported chain IDs
 */
export function getSupportedChainIds(): number[] {
  return Object.keys(NETWORKS).map(Number);
}
