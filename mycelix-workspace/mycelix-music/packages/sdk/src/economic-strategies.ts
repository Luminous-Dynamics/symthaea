// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Music Economic Strategies SDK
 *
 * High-level TypeScript API for interacting with economic strategy contracts
 */

import { ethers } from 'ethers';

// ========== TYPES ==========

export enum PaymentModel {
  PAY_PER_STREAM = 'pay_per_stream',
  SUBSCRIPTION = 'subscription',
  PAY_PER_DOWNLOAD = 'pay_per_download',
  NFT_GATED = 'nft_gated',
  STAKING_GATED = 'staking_gated',
  TOKEN_TIP = 'token_tip',
  GIFT_ECONOMY = 'gift_economy',
  TIME_BARTER = 'time_barter',
  PATRONAGE = 'patronage',
  FREEMIUM = 'freemium',
  PAY_WHAT_YOU_WANT = 'pay_what_you_want',
  AUCTION = 'auction',
}

export enum PaymentType {
  STREAM = 0,
  DOWNLOAD = 1,
  TIP = 2,
  PATRONAGE = 3,
  NFT_ACCESS = 4,
}

export interface Split {
  recipient: string;  // Address or DID
  basisPoints: number;  // 10000 = 100%
  role: string;
}

export interface EconomicConfig {
  strategyId: string;
  paymentModel: PaymentModel;
  distributionSplits: Split[];
  minimumPayment?: number;
  acceptsGifts?: boolean;
}

export interface PaymentReceipt {
  songId: string;
  listener: string;
  amount: string;
  paymentType: PaymentType;
  timestamp: number;
  transactionHash: string;
  splits: Split[];
}

// ========== SDK CLASS ==========

export class EconomicStrategySDK {
  private provider: ethers.Provider;
  private signer?: ethers.Signer;
  private routerContract: ethers.Contract;
  private strategyContracts: Map<string, ethers.Contract>;

  constructor(
    provider: ethers.Provider,
    routerAddress: string,
    signer?: ethers.Signer
  ) {
    this.provider = provider;
    this.signer = signer;
    this.strategyContracts = new Map();

    // Initialize router contract
    this.routerContract = new ethers.Contract(
      routerAddress,
      ROUTER_ABI,
      signer || provider
    );
  }

  // ========== ARTIST FUNCTIONS ==========

  /**
   * Register a new song with an economic strategy
   */
  async registerSong(
    songId: string,
    config: EconomicConfig
  ): Promise<ethers.TransactionReceipt> {
    if (!this.signer) throw new Error('Signer required');

    // Register song with router
    const tx = await this.routerContract.registerSong(
      ethers.id(songId),
      ethers.id(config.strategyId)
    );

    const receipt = await tx.wait();

    // Configure strategy-specific settings
    await this._configureStrategy(songId, config);

    return receipt;
  }

  /**
   * Configure strategy-specific settings
   */
  private async _configureStrategy(
    songId: string,
    config: EconomicConfig
  ): Promise<void> {
    const songHash = ethers.id(songId);
    const strategyAddress = await this.routerContract.songStrategy(songHash);

    if (config.paymentModel === PaymentModel.PAY_PER_STREAM) {
      // Configure PayPerStreamStrategy
      const strategy = new ethers.Contract(
        strategyAddress,
        PAY_PER_STREAM_ABI,
        this.signer!
      );

      const recipients = config.distributionSplits.map(s => s.recipient);
      const basisPoints = config.distributionSplits.map(s => s.basisPoints);
      const roles = config.distributionSplits.map(s => s.role);

      const tx = await strategy.setRoyaltySplit(
        songHash,
        recipients,
        basisPoints,
        roles
      );

      await tx.wait();
    }

    else if (config.paymentModel === PaymentModel.GIFT_ECONOMY) {
      // Configure GiftEconomyStrategy
      const strategy = new ethers.Contract(
        strategyAddress,
        GIFT_ECONOMY_ABI,
        this.signer!
      );

      const recipients = config.distributionSplits.map(s => s.recipient);
      const splits = config.distributionSplits.map(s => s.basisPoints);

      const tx = await strategy.configureGifts(
        songHash,
        config.acceptsGifts ?? true,
        config.minimumPayment ?? 0,
        recipients,
        splits
      );

      await tx.wait();
    }
  }

  /**
   * Change economic strategy for an existing song
   */
  async changeSongStrategy(
    songId: string,
    newStrategyId: string
  ): Promise<ethers.TransactionReceipt> {
    if (!this.signer) throw new Error('Signer required');

    const tx = await this.routerContract.changeSongStrategy(
      ethers.id(songId),
      ethers.id(newStrategyId)
    );

    return await tx.wait();
  }

  // ========== LISTENER FUNCTIONS ==========

  /**
   * Stream a song (pay-per-stream model)
   */
  async streamSong(
    songId: string,
    paymentAmount: string  // In FLOW tokens
  ): Promise<PaymentReceipt> {
    if (!this.signer) throw new Error('Signer required');

    const songHash = ethers.id(songId);
    const amount = ethers.parseEther(paymentAmount);

    // Approve FLOW token spending
    const flowToken = await this._getFlowToken();
    const approveTx = await flowToken.approve(
      await this.routerContract.getAddress(),
      amount
    );
    await approveTx.wait();

    // Process payment
    const tx = await this.routerContract.processPayment(
      songHash,
      amount,
      PaymentType.STREAM
    );

    const receipt = await tx.wait();

    // Get split details
    const splits = await this.routerContract.previewSplits(songHash, amount);

    return {
      songId,
      listener: await this.signer.getAddress(),
      amount: paymentAmount,
      paymentType: PaymentType.STREAM,
      timestamp: Date.now(),
      transactionHash: receipt!.hash,
      splits: splits.map((s: any) => ({
        recipient: s.recipient,
        basisPoints: Number(s.basisPoints),
        role: s.role,
      })),
    };
  }

  /**
   * Tip an artist (gift economy)
   */
  async tipArtist(
    songId: string,
    tipAmount: string
  ): Promise<PaymentReceipt> {
    if (!this.signer) throw new Error('Signer required');

    const songHash = ethers.id(songId);
    const amount = ethers.parseEther(tipAmount);

    const flowToken = await this._getFlowToken();
    const approveTx = await flowToken.approve(
      await this.routerContract.getAddress(),
      amount
    );
    await approveTx.wait();

    const tx = await this.routerContract.processPayment(
      songHash,
      amount,
      PaymentType.TIP
    );

    const receipt = await tx.wait();
    const splits = await this.routerContract.previewSplits(songHash, amount);

    return {
      songId,
      listener: await this.signer.getAddress(),
      amount: tipAmount,
      paymentType: PaymentType.TIP,
      timestamp: Date.now(),
      transactionHash: receipt!.hash,
      splits: splits.map((s: any) => ({
        recipient: s.recipient,
        basisPoints: Number(s.basisPoints),
        role: s.role,
      })),
    };
  }

  /**
   * Batch stream multiple songs (gas optimization)
   */
  async batchStreamSongs(
    songs: Array<{ songId: string; amount: string }>
  ): Promise<ethers.TransactionReceipt> {
    if (!this.signer) throw new Error('Signer required');

    const songHashes = songs.map(s => ethers.id(s.songId));
    const amounts = songs.map(s => ethers.parseEther(s.amount));
    const types = songs.map(() => PaymentType.STREAM);

    // Approve total amount
    const totalAmount = amounts.reduce((sum, amt) => sum + amt, 0n);
    const flowToken = await this._getFlowToken();
    const approveTx = await flowToken.approve(
      await this.routerContract.getAddress(),
      totalAmount
    );
    await approveTx.wait();

    // Batch process
    const tx = await this.routerContract.batchProcessPayments(
      songHashes,
      amounts,
      types
    );

    return await tx.wait();
  }

  /**
   * Claim CGC rewards (gift economy)
   *
   * @param artistAddress - The artist's wallet address
   * @param songId - A song ID owned by the artist to look up the gift economy strategy
   */
  async claimCGCRewards(
    artistAddress: string,
    songId: string
  ): Promise<ethers.TransactionReceipt> {
    if (!this.signer) throw new Error('Signer required');

    // Get the strategy address from the router contract using the song ID
    const songHash = ethers.id(songId);
    const strategyAddress = await this.routerContract.songStrategy(songHash);

    if (strategyAddress === ethers.ZeroAddress) {
      throw new Error(`No strategy registered for song: ${songId}`);
    }

    // Create contract instance for the gift economy strategy
    const strategy = new ethers.Contract(
      strategyAddress,
      GIFT_ECONOMY_ABI,
      this.signer
    );

    const tx = await strategy.claimCGCRewards(artistAddress);
    return await tx.wait();
  }

  // ========== VIEW FUNCTIONS ==========

  /**
   * Preview how a payment would be split
   */
  async previewPaymentSplits(
    songId: string,
    amount: string
  ): Promise<Split[]> {
    const songHash = ethers.id(songId);
    const amountWei = ethers.parseEther(amount);

    const splits = await this.routerContract.previewSplits(songHash, amountWei);

    return splits.map((s: any) => ({
      recipient: s.recipient,
      basisPoints: Number(s.basisPoints),
      role: s.role,
    }));
  }

  /**
   * Get payment history for a song
   */
  async getPaymentHistory(songId: string): Promise<PaymentReceipt[]> {
    const songHash = ethers.id(songId);
    const history = await this.routerContract.getPaymentHistory(songHash);

    return history.map((payment: any) => ({
      songId,
      listener: payment.listener,
      amount: ethers.formatEther(payment.amount),
      paymentType: payment.paymentType,
      timestamp: Number(payment.timestamp),
      transactionHash: '', // Would need to get from events
      splits: [], // Would need to calculate
    }));
  }

  /**
   * Get listener's CGC balance
   */
  async getListenerCGCBalance(
    artistAddress: string,
    listenerAddress: string
  ): Promise<string> {
    // This would query the gift economy strategy
    // Implementation depends on how we track which strategy is used
    return '0';
  }

  /**
   * Get listener stats
   */
  async getListenerStats(
    artistAddress: string,
    listenerAddress: string
  ): Promise<{
    totalGifts: string;
    totalStreams: number;
    cgcBalance: string;
  }> {
    // Query the strategy contract
    return {
      totalGifts: '0',
      totalStreams: 0,
      cgcBalance: '0',
    };
  }

  // ========== HELPERS ==========

  private async _getFlowToken(): Promise<ethers.Contract> {
    const flowTokenAddress = await this.routerContract.flowToken();
    return new ethers.Contract(
      flowTokenAddress,
      ERC20_ABI,
      this.signer || this.provider
    );
  }
}

// ========== PRESET CONFIGURATIONS ==========

export const PRESET_STRATEGIES = {
  independentArtist: {
    strategyId: 'pay-per-stream-v1',
    paymentModel: PaymentModel.PAY_PER_STREAM,
    distributionSplits: [
      { recipient: '', basisPoints: 9500, role: 'artist' },
      { recipient: '', basisPoints: 500, role: 'protocol' },
    ],
    minimumPayment: 0.001,
  },

  communityCollective: {
    strategyId: 'gift-economy-v1',
    paymentModel: PaymentModel.GIFT_ECONOMY,
    distributionSplits: [
      { recipient: '', basisPoints: 8000, role: 'artist' },
      { recipient: '', basisPoints: 1000, role: 'dao_treasury' },
      { recipient: '', basisPoints: 500, role: 'local_scene' },
      { recipient: '', basisPoints: 500, role: 'protocol' },
    ],
    acceptsGifts: true,
    minimumPayment: 0,
  },

  collaborativeSplit: {
    strategyId: 'pay-per-stream-v1',
    paymentModel: PaymentModel.PAY_PER_STREAM,
    distributionSplits: [
      { recipient: '', basisPoints: 5000, role: 'artist_1' },
      { recipient: '', basisPoints: 3000, role: 'artist_2' },
      { recipient: '', basisPoints: 1500, role: 'producer' },
      { recipient: '', basisPoints: 500, role: 'protocol' },
    ],
    minimumPayment: 0.001,
  },
};

// ========== CONTRACT ABIs ==========

const ROUTER_ABI = [
  'function registerSong(bytes32 songId, bytes32 strategyId) external',
  'function changeSongStrategy(bytes32 songId, bytes32 strategyId) external',
  'function processPayment(bytes32 songId, uint256 amount, uint8 paymentType) external',
  'function batchProcessPayments(bytes32[] calldata songIds, uint256[] calldata amounts, uint8[] calldata paymentTypes) external',
  'function previewSplits(bytes32 songId, uint256 amount) external view returns (tuple(address recipient, uint256 basisPoints, string role)[])',
  'function getPaymentHistory(bytes32 songId) external view returns (tuple(bytes32 songId, address listener, uint256 amount, uint8 paymentType, uint256 timestamp)[])',
  'function songStrategy(bytes32) external view returns (address)',
  'function flowToken() external view returns (address)',
];

const PAY_PER_STREAM_ABI = [
  'function setRoyaltySplit(bytes32 songId, address[] calldata recipients, uint256[] calldata basisPoints, string[] calldata roles) external',
  'function getRoyaltySplit(bytes32 songId) external view returns (address[] memory recipients, uint256[] memory basisPoints, string[] memory roles)',
];

const GIFT_ECONOMY_ABI = [
  'function configureGifts(bytes32 songId, bool acceptsGifts, uint256 minGiftAmount, address[] calldata recipients, uint256[] calldata splits) external',
  'function claimCGCRewards(address artist) external',
  'function getCGCBalance(address artist, address listener) external view returns (uint256)',
  'function getListenerStats(address artist, address listener) external view returns (uint256 totalGifts, uint256 totalStreams, uint256 cgcBalance)',
];

const ERC20_ABI = [
  'function approve(address spender, uint256 amount) external returns (bool)',
  'function transfer(address to, uint256 amount) external returns (bool)',
  'function balanceOf(address account) external view returns (uint256)',
];
