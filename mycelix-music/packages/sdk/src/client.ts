// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Music SDK Client
 *
 * High-level client for interacting with Mycelix Music smart contracts
 */

import { ethers } from 'ethers';
import {
  NetworkConfig,
  getNetwork,
  getRouterAddress,
  getStrategyAddress,
  isNetworkSupported,
} from './networks';
import {
  PaymentModel,
  PRESET_STRATEGIES,
  EconomicConfig,
} from './economic-strategies';
import { buildDomain, signTypedData } from './eip712';

// Router ABI (minimal for core functions)
const ROUTER_ABI = [
  'function registerArtist(address artistAddress, string calldata metadataUri) external',
  'function registerTrack(bytes32 trackId, address artist, string calldata metadataUri, address strategy) external',
  'function processPayment(bytes32 trackId, address listener, uint256 amount) external payable',
  'function getTrackStrategy(bytes32 trackId) external view returns (address)',
  'function getArtistMetadata(address artist) external view returns (string memory)',
  'function isRegisteredArtist(address artist) external view returns (bool)',
  'event ArtistRegistered(address indexed artist, string metadataUri)',
  'event TrackRegistered(bytes32 indexed trackId, address indexed artist, address strategy)',
  'event PaymentProcessed(bytes32 indexed trackId, address indexed listener, uint256 amount)',
];

// CGC Registry ABI
const CGC_REGISTRY_ABI = [
  'function registerCGC(bytes32 cgcId, string calldata metadataUri, address[] calldata members) external',
  'function getCGCMembers(bytes32 cgcId) external view returns (address[] memory)',
  'function isCGCMember(bytes32 cgcId, address member) external view returns (bool)',
  'event CGCRegistered(bytes32 indexed cgcId, string metadataUri)',
];

// Flow Token ABI
const FLOW_TOKEN_ABI = [
  'function balanceOf(address account) external view returns (uint256)',
  'function transfer(address to, uint256 amount) external returns (bool)',
  'function approve(address spender, uint256 amount) external returns (bool)',
  'function allowance(address owner, address spender) external view returns (uint256)',
  'function mint(address to, uint256 amount) external',
  'function burn(uint256 amount) external',
  'event Transfer(address indexed from, address indexed to, uint256 value)',
];

export interface ClientConfig {
  chainId: number;
  signer?: ethers.Signer;
  provider?: ethers.Provider;
}

export interface TrackRegistration {
  trackId: string;
  metadataUri: string;
  paymentModel: PaymentModel;
}

export interface ArtistRegistration {
  metadataUri: string;
}

export interface CGCRegistration {
  cgcId: string;
  metadataUri: string;
  members: string[];
}

export class MycelixClient {
  private readonly config: ClientConfig;
  private readonly network: NetworkConfig;
  private readonly provider: ethers.Provider;
  private signer?: ethers.Signer;

  private routerContract?: ethers.Contract;
  private cgcRegistryContract?: ethers.Contract;
  private flowTokenContract?: ethers.Contract;

  constructor(config: ClientConfig) {
    if (!isNetworkSupported(config.chainId)) {
      throw new Error(`Unsupported chain ID: ${config.chainId}`);
    }

    this.config = config;
    const network = getNetwork(config.chainId);
    if (!network) {
      throw new Error(`Network configuration not found for chain ${config.chainId}`);
    }
    this.network = network;

    // Set up provider
    if (config.provider) {
      this.provider = config.provider;
    } else if (config.signer?.provider) {
      this.provider = config.signer.provider;
    } else {
      this.provider = new ethers.JsonRpcProvider(network.rpcUrl);
    }

    this.signer = config.signer;
    this.initializeContracts();
  }

  private initializeContracts(): void {
    const signerOrProvider = this.signer || this.provider;

    if (this.network.routerAddress) {
      this.routerContract = new ethers.Contract(
        this.network.routerAddress,
        ROUTER_ABI,
        signerOrProvider
      );
    }

    if (this.network.cgcRegistryAddress) {
      this.cgcRegistryContract = new ethers.Contract(
        this.network.cgcRegistryAddress,
        CGC_REGISTRY_ABI,
        signerOrProvider
      );
    }

    if (this.network.flowTokenAddress) {
      this.flowTokenContract = new ethers.Contract(
        this.network.flowTokenAddress,
        FLOW_TOKEN_ABI,
        signerOrProvider
      );
    }
  }

  /**
   * Connect a signer to enable write operations
   */
  connect(signer: ethers.Signer): MycelixClient {
    return new MycelixClient({
      ...this.config,
      signer,
      provider: this.provider,
    });
  }

  /**
   * Get network information
   */
  getNetwork(): NetworkConfig {
    return this.network;
  }

  /**
   * Get the router contract address
   */
  getRouterAddress(): string {
    return getRouterAddress(this.config.chainId);
  }

  /**
   * Get strategy address by payment model
   */
  getStrategyAddressForModel(paymentModel: PaymentModel): string {
    const keyMap: Record<PaymentModel, keyof NetworkConfig['strategies']> = {
      [PaymentModel.PAY_PER_STREAM]: 'payPerStream',
      [PaymentModel.GIFT_ECONOMY]: 'giftEconomy',
      [PaymentModel.FREEMIUM]: 'freemium',
      [PaymentModel.PATRONAGE]: 'patronage',
      [PaymentModel.NFT_GATED]: 'nftGated',
      [PaymentModel.SUBSCRIPTION]: 'subscription',
      [PaymentModel.PAY_PER_DOWNLOAD]: 'payPerDownload',
      [PaymentModel.STAKING_GATED]: 'stakingGated',
      [PaymentModel.TOKEN_TIP]: 'tokenTip',
      [PaymentModel.TIME_BARTER]: 'timeBarter',
      [PaymentModel.AUCTION]: 'auction',
      [PaymentModel.PAY_WHAT_YOU_WANT]: 'payWhatYouWant',
    };
    return getStrategyAddress(this.config.chainId, keyMap[paymentModel]);
  }

  /**
   * Get preset strategy configurations
   */
  getPresetStrategies(): typeof PRESET_STRATEGIES {
    return PRESET_STRATEGIES;
  }

  // ============ Artist Operations ============

  /**
   * Register as an artist
   */
  async registerArtist(registration: ArtistRegistration): Promise<ethers.ContractTransactionReceipt> {
    if (!this.signer) {
      throw new Error('Signer required for write operations');
    }
    if (!this.routerContract) {
      throw new Error('Router contract not available on this network');
    }

    const address = await this.signer.getAddress();
    const tx = await this.routerContract.registerArtist(address, registration.metadataUri);
    return tx.wait();
  }

  /**
   * Check if an address is a registered artist
   */
  async isRegisteredArtist(address: string): Promise<boolean> {
    if (!this.routerContract) {
      throw new Error('Router contract not available on this network');
    }
    return this.routerContract.isRegisteredArtist(address);
  }

  /**
   * Get artist metadata URI
   */
  async getArtistMetadata(address: string): Promise<string> {
    if (!this.routerContract) {
      throw new Error('Router contract not available on this network');
    }
    return this.routerContract.getArtistMetadata(address);
  }

  // ============ Track Operations ============

  /**
   * Register a new track
   */
  async registerTrack(registration: TrackRegistration): Promise<ethers.ContractTransactionReceipt> {
    if (!this.signer) {
      throw new Error('Signer required for write operations');
    }
    if (!this.routerContract) {
      throw new Error('Router contract not available on this network');
    }

    const artistAddress = await this.signer.getAddress();
    const strategyAddress = this.getStrategyAddressForModel(registration.paymentModel);
    const trackIdBytes = ethers.id(registration.trackId);

    const tx = await this.routerContract.registerTrack(
      trackIdBytes,
      artistAddress,
      registration.metadataUri,
      strategyAddress
    );
    return tx.wait();
  }

  /**
   * Get the strategy address for a track
   */
  async getTrackStrategy(trackId: string): Promise<string> {
    if (!this.routerContract) {
      throw new Error('Router contract not available on this network');
    }
    const trackIdBytes = ethers.id(trackId);
    return this.routerContract.getTrackStrategy(trackIdBytes);
  }

  /**
   * Process payment for a track (streaming payment)
   */
  async processPayment(
    trackId: string,
    amount: bigint
  ): Promise<ethers.ContractTransactionReceipt> {
    if (!this.signer) {
      throw new Error('Signer required for write operations');
    }
    if (!this.routerContract) {
      throw new Error('Router contract not available on this network');
    }

    const listenerAddress = await this.signer.getAddress();
    const trackIdBytes = ethers.id(trackId);

    const tx = await this.routerContract.processPayment(
      trackIdBytes,
      listenerAddress,
      amount,
      { value: amount }
    );
    return tx.wait();
  }

  // ============ CGC Operations ============

  /**
   * Register a Creative Gratitude Circle
   */
  async registerCGC(registration: CGCRegistration): Promise<ethers.ContractTransactionReceipt> {
    if (!this.signer) {
      throw new Error('Signer required for write operations');
    }
    if (!this.cgcRegistryContract) {
      throw new Error('CGC Registry contract not available on this network');
    }

    const cgcIdBytes = ethers.id(registration.cgcId);
    const tx = await this.cgcRegistryContract.registerCGC(
      cgcIdBytes,
      registration.metadataUri,
      registration.members
    );
    return tx.wait();
  }

  /**
   * Get members of a CGC
   */
  async getCGCMembers(cgcId: string): Promise<string[]> {
    if (!this.cgcRegistryContract) {
      throw new Error('CGC Registry contract not available on this network');
    }
    const cgcIdBytes = ethers.id(cgcId);
    return this.cgcRegistryContract.getCGCMembers(cgcIdBytes);
  }

  /**
   * Check if an address is a member of a CGC
   */
  async isCGCMember(cgcId: string, memberAddress: string): Promise<boolean> {
    if (!this.cgcRegistryContract) {
      throw new Error('CGC Registry contract not available on this network');
    }
    const cgcIdBytes = ethers.id(cgcId);
    return this.cgcRegistryContract.isCGCMember(cgcIdBytes, memberAddress);
  }

  // ============ Token Operations ============

  /**
   * Get FLOW token balance
   */
  async getFlowBalance(address: string): Promise<bigint> {
    if (!this.flowTokenContract) {
      throw new Error('FLOW token contract not available on this network');
    }
    return this.flowTokenContract.balanceOf(address);
  }

  /**
   * Transfer FLOW tokens
   */
  async transferFlow(
    to: string,
    amount: bigint
  ): Promise<ethers.ContractTransactionReceipt> {
    if (!this.signer) {
      throw new Error('Signer required for write operations');
    }
    if (!this.flowTokenContract) {
      throw new Error('FLOW token contract not available on this network');
    }

    const tx = await this.flowTokenContract.transfer(to, amount);
    return tx.wait();
  }

  /**
   * Approve FLOW token spending
   */
  async approveFlow(
    spender: string,
    amount: bigint
  ): Promise<ethers.ContractTransactionReceipt> {
    if (!this.signer) {
      throw new Error('Signer required for write operations');
    }
    if (!this.flowTokenContract) {
      throw new Error('FLOW token contract not available on this network');
    }

    const tx = await this.flowTokenContract.approve(spender, amount);
    return tx.wait();
  }

  /**
   * Get FLOW token allowance
   */
  async getFlowAllowance(owner: string, spender: string): Promise<bigint> {
    if (!this.flowTokenContract) {
      throw new Error('FLOW token contract not available on this network');
    }
    return this.flowTokenContract.allowance(owner, spender);
  }

  // ============ Utility Methods ============

  /**
   * Generate EIP-712 domain for signing
   */
  getEIP712Domain(): ReturnType<typeof buildDomain> {
    return buildDomain(this.config.chainId, this.network.routerAddress);
  }

  /**
   * Sign typed data with EIP-712
   */
  async signTyped(
    types: Record<string, Array<{ name: string; type: string }>>,
    value: Record<string, unknown>
  ): Promise<string> {
    if (!this.signer) {
      throw new Error('Signer required for signing');
    }
    return signTypedData(this.signer, this.getEIP712Domain(), types, value);
  }

  /**
   * Convert a human-readable amount to wei
   */
  static parseEther(amount: string): bigint {
    return ethers.parseEther(amount);
  }

  /**
   * Convert wei to human-readable format
   */
  static formatEther(amount: bigint): string {
    return ethers.formatEther(amount);
  }

  /**
   * Generate a track ID from artist address and track name
   */
  static generateTrackId(artistAddress: string, trackName: string): string {
    return `${artistAddress.toLowerCase()}-${trackName.toLowerCase().replace(/\s+/g, '-')}`;
  }
}

/**
 * Create a Mycelix client for a specific chain
 */
export function createClient(config: ClientConfig): MycelixClient {
  return new MycelixClient(config);
}

/**
 * Create a read-only client (no signer required)
 */
export function createReadOnlyClient(chainId: number): MycelixClient {
  return new MycelixClient({ chainId });
}
