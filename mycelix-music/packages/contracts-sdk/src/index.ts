// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Contracts SDK
 *
 * TypeScript SDK for interacting with Mycelix smart contracts.
 * Provides type-safe wrappers for royalties, NFTs, and subscriptions.
 */

import { ethers, Contract, Signer, Provider, BigNumberish } from 'ethers';

// ==================== Types ====================

export interface Split {
  recipient: string;
  share: number; // Basis points (10000 = 100%)
}

export interface Song {
  songId: string;
  primaryArtist: string;
  totalEarned: bigint;
  totalDistributed: bigint;
  active: boolean;
  splitCount: number;
}

export interface Release {
  songId: string;
  artist: string;
  metadataUri: string;
  editionType: EditionType;
  maxSupply: bigint;
  price: bigint;
  paymentToken: string;
  startTime: bigint;
  endTime: bigint;
  paused: boolean;
  exists: boolean;
}

export interface MintStats {
  totalMinted: bigint;
  totalRevenue: bigint;
  uniqueHolders: bigint;
}

export interface Plan {
  name: string;
  price: bigint;
  duration: bigint;
  streamAllowance: bigint;
  active: boolean;
  exists: boolean;
}

export interface Subscription {
  planId: bigint;
  startTime: bigint;
  endTime: bigint;
  streamsUsed: bigint;
  autoRenew: boolean;
  active: boolean;
}

export enum EditionType {
  STANDARD = 0,
  LIMITED = 1,
  EXCLUSIVE = 2,
  FREE = 3,
}

// ==================== ABI Fragments ====================

const ROYALTY_SPLITTER_ABI = [
  'function registerSong(string songId, tuple(address recipient, uint256 share)[] splits) returns (bytes32)',
  'function updateSplits(bytes32 songHash, tuple(address recipient, uint256 share)[] newSplits)',
  'function deactivateSong(bytes32 songHash)',
  'function receiveRoyalty(bytes32 songHash, uint256 amount)',
  'function batchReceiveRoyalties(bytes32[] songHashes, uint256[] amounts)',
  'function claimPayout()',
  'function claimableBalance(address account) view returns (uint256)',
  'function getSong(bytes32 songHash) view returns (string, address, uint256, uint256, bool, uint256)',
  'function getSplits(bytes32 songHash) view returns (tuple(address recipient, uint256 share)[])',
  'function getArtistSongs(address artist) view returns (bytes32[])',
  'function platformFee() view returns (uint256)',
  'function minPayout() view returns (uint256)',
  'event SongRegistered(bytes32 indexed songHash, string songId, address indexed primaryArtist, uint256 splitCount)',
  'event RoyaltyReceived(bytes32 indexed songHash, uint256 amount, uint256 platformCut, uint256 artistCut)',
  'event PayoutClaimed(address indexed recipient, uint256 amount)',
];

const MUSIC_NFT_ABI = [
  'function createRelease(string songId, string metadataUri, uint8 editionType, uint256 maxSupply, uint256 price, address paymentToken, uint256 startTime, uint256 endTime, uint256 maxPerWallet) returns (uint256)',
  'function updateRelease(uint256 tokenId, uint256 price, uint256 startTime, uint256 endTime)',
  'function pauseRelease(uint256 tokenId, bool paused)',
  'function mint(uint256 tokenId, uint256 quantity) payable',
  'function freeMint(uint256 tokenId)',
  'function airdrop(uint256 tokenId, address[] recipients, uint256[] quantities)',
  'function setAllowlist(uint256 tokenId, address[] addresses, bool status)',
  'function setAllowlistRequired(uint256 tokenId, bool required)',
  'function getRelease(uint256 tokenId) view returns (tuple(string songId, address artist, string metadataUri, uint8 editionType, uint256 maxSupply, uint256 price, address paymentToken, uint256 startTime, uint256 endTime, bool paused, bool exists))',
  'function getMintStats(uint256 tokenId) view returns (tuple(uint256 totalMinted, uint256 totalRevenue, uint256 uniqueHolders))',
  'function getArtistReleases(address artist) view returns (uint256[])',
  'function isMintable(uint256 tokenId) view returns (bool)',
  'function uri(uint256 tokenId) view returns (string)',
  'function balanceOf(address account, uint256 id) view returns (uint256)',
  'function totalSupply(uint256 id) view returns (uint256)',
  'function allowlist(uint256 tokenId, address account) view returns (bool)',
  'function platformFee() view returns (uint256)',
  'event ReleaseCreated(uint256 indexed tokenId, address indexed artist, string songId, uint8 editionType, uint256 maxSupply, uint256 price)',
  'event ReleaseMinted(uint256 indexed tokenId, address indexed minter, uint256 quantity, uint256 totalPaid)',
];

const SUBSCRIPTION_ABI = [
  'function createPlan(string name, uint256 price, uint256 duration, uint256 streamAllowance) returns (uint256)',
  'function updatePlan(uint256 planId, uint256 price, uint256 streamAllowance)',
  'function deactivatePlan(uint256 planId)',
  'function subscribe(uint256 planId, bool autoRenew)',
  'function renew()',
  'function cancelAutoRenew()',
  'function isSubscribed(address user) view returns (bool)',
  'function getSubscription(address user) view returns (tuple(uint256 planId, uint256 startTime, uint256 endTime, uint256 streamsUsed, bool autoRenew, bool active))',
  'function getPlan(uint256 planId) view returns (tuple(string name, uint256 price, uint256 duration, uint256 streamAllowance, bool active, bool exists))',
  'function recordStream(address user, address artist, string songId)',
  'function batchRecordStreams(address[] users, address[] artists, string[] songIds)',
  'function distributePeriodRevenue(address[] artists)',
  'function getArtistPool(address artist) view returns (tuple(uint256 totalStreams, uint256 totalEarnings, uint256 lastDistribution))',
  'function getArtistPeriodStreams(address artist) view returns (uint256)',
  'function getCurrentPeriodInfo() view returns (uint256 period, uint256 startTime, uint256 endTime, uint256 revenue)',
  'event Subscribed(address indexed user, uint256 indexed planId, uint256 startTime, uint256 endTime)',
  'event StreamRecorded(address indexed user, address indexed artist, string songId, uint256 period)',
  'event ArtistPaid(address indexed artist, uint256 indexed period, uint256 amount, uint256 streams)',
];

const ERC20_ABI = [
  'function approve(address spender, uint256 amount) returns (bool)',
  'function allowance(address owner, address spender) view returns (uint256)',
  'function balanceOf(address account) view returns (uint256)',
];

// ==================== RoyaltySplitter Client ====================

export class RoyaltySplitterClient {
  private contract: Contract;
  private signer?: Signer;

  constructor(
    address: string,
    providerOrSigner: Provider | Signer
  ) {
    this.contract = new ethers.Contract(address, ROYALTY_SPLITTER_ABI, providerOrSigner);

    if ('getAddress' in providerOrSigner) {
      this.signer = providerOrSigner as Signer;
    }
  }

  /**
   * Register a new song with royalty splits
   */
  async registerSong(songId: string, splits: Split[]): Promise<string> {
    if (!this.signer) throw new Error('Signer required');

    const tx = await this.contract.registerSong(songId, splits);
    const receipt = await tx.wait();

    const event = receipt.logs.find(
      (log: any) => log.fragment?.name === 'SongRegistered'
    );

    return event?.args?.songHash;
  }

  /**
   * Update splits for an existing song
   */
  async updateSplits(songHash: string, splits: Split[]): Promise<void> {
    if (!this.signer) throw new Error('Signer required');

    const tx = await this.contract.updateSplits(songHash, splits);
    await tx.wait();
  }

  /**
   * Deactivate a song
   */
  async deactivateSong(songHash: string): Promise<void> {
    if (!this.signer) throw new Error('Signer required');

    const tx = await this.contract.deactivateSong(songHash);
    await tx.wait();
  }

  /**
   * Send royalty payment for a song
   */
  async sendRoyalty(
    songHash: string,
    amount: BigNumberish,
    paymentToken: string
  ): Promise<void> {
    if (!this.signer) throw new Error('Signer required');

    // Approve token spending
    const token = new ethers.Contract(paymentToken, ERC20_ABI, this.signer);
    const allowance = await token.allowance(
      await this.signer.getAddress(),
      await this.contract.getAddress()
    );

    if (allowance < amount) {
      const approveTx = await token.approve(await this.contract.getAddress(), amount);
      await approveTx.wait();
    }

    const tx = await this.contract.receiveRoyalty(songHash, amount);
    await tx.wait();
  }

  /**
   * Claim pending payout
   */
  async claimPayout(): Promise<void> {
    if (!this.signer) throw new Error('Signer required');

    const tx = await this.contract.claimPayout();
    await tx.wait();
  }

  /**
   * Get claimable balance
   */
  async getClaimableBalance(address: string): Promise<bigint> {
    return this.contract.claimableBalance(address);
  }

  /**
   * Get song details
   */
  async getSong(songHash: string): Promise<Song> {
    const result = await this.contract.getSong(songHash);
    return {
      songId: result[0],
      primaryArtist: result[1],
      totalEarned: result[2],
      totalDistributed: result[3],
      active: result[4],
      splitCount: Number(result[5]),
    };
  }

  /**
   * Get splits for a song
   */
  async getSplits(songHash: string): Promise<Split[]> {
    const splits = await this.contract.getSplits(songHash);
    return splits.map((s: any) => ({
      recipient: s.recipient,
      share: Number(s.share),
    }));
  }

  /**
   * Get songs by artist
   */
  async getArtistSongs(artist: string): Promise<string[]> {
    return this.contract.getArtistSongs(artist);
  }

  /**
   * Calculate song hash from ID
   */
  static getSongHash(songId: string): string {
    return ethers.keccak256(ethers.toUtf8Bytes(songId));
  }
}

// ==================== MusicNFT Client ====================

export class MusicNFTClient {
  private contract: Contract;
  private signer?: Signer;

  constructor(
    address: string,
    providerOrSigner: Provider | Signer
  ) {
    this.contract = new ethers.Contract(address, MUSIC_NFT_ABI, providerOrSigner);

    if ('getAddress' in providerOrSigner) {
      this.signer = providerOrSigner as Signer;
    }
  }

  /**
   * Create a new music release
   */
  async createRelease(params: {
    songId: string;
    metadataUri: string;
    editionType: EditionType;
    maxSupply: BigNumberish;
    price: BigNumberish;
    paymentToken: string;
    startTime: number;
    endTime: number;
    maxPerWallet: number;
  }): Promise<bigint> {
    if (!this.signer) throw new Error('Signer required');

    const tx = await this.contract.createRelease(
      params.songId,
      params.metadataUri,
      params.editionType,
      params.maxSupply,
      params.price,
      params.paymentToken,
      params.startTime,
      params.endTime,
      params.maxPerWallet
    );

    const receipt = await tx.wait();
    const event = receipt.logs.find(
      (log: any) => log.fragment?.name === 'ReleaseCreated'
    );

    return event?.args?.tokenId;
  }

  /**
   * Mint a release
   */
  async mint(
    tokenId: BigNumberish,
    quantity: number,
    value?: BigNumberish
  ): Promise<void> {
    if (!this.signer) throw new Error('Signer required');

    const release = await this.getRelease(tokenId);

    if (release.paymentToken === ethers.ZeroAddress) {
      // Native token payment
      const total = release.price * BigInt(quantity);
      const tx = await this.contract.mint(tokenId, quantity, { value: total });
      await tx.wait();
    } else {
      // ERC20 payment
      const total = release.price * BigInt(quantity);
      const token = new ethers.Contract(release.paymentToken, ERC20_ABI, this.signer);

      const allowance = await token.allowance(
        await this.signer.getAddress(),
        await this.contract.getAddress()
      );

      if (allowance < total) {
        const approveTx = await token.approve(await this.contract.getAddress(), total);
        await approveTx.wait();
      }

      const tx = await this.contract.mint(tokenId, quantity);
      await tx.wait();
    }
  }

  /**
   * Free mint
   */
  async freeMint(tokenId: BigNumberish): Promise<void> {
    if (!this.signer) throw new Error('Signer required');

    const tx = await this.contract.freeMint(tokenId);
    await tx.wait();
  }

  /**
   * Airdrop tokens
   */
  async airdrop(
    tokenId: BigNumberish,
    recipients: string[],
    quantities: number[]
  ): Promise<void> {
    if (!this.signer) throw new Error('Signer required');

    const tx = await this.contract.airdrop(tokenId, recipients, quantities);
    await tx.wait();
  }

  /**
   * Set allowlist
   */
  async setAllowlist(
    tokenId: BigNumberish,
    addresses: string[],
    status: boolean
  ): Promise<void> {
    if (!this.signer) throw new Error('Signer required');

    const tx = await this.contract.setAllowlist(tokenId, addresses, status);
    await tx.wait();
  }

  /**
   * Get release details
   */
  async getRelease(tokenId: BigNumberish): Promise<Release> {
    const result = await this.contract.getRelease(tokenId);
    return {
      songId: result.songId,
      artist: result.artist,
      metadataUri: result.metadataUri,
      editionType: result.editionType,
      maxSupply: result.maxSupply,
      price: result.price,
      paymentToken: result.paymentToken,
      startTime: result.startTime,
      endTime: result.endTime,
      paused: result.paused,
      exists: result.exists,
    };
  }

  /**
   * Get mint stats
   */
  async getMintStats(tokenId: BigNumberish): Promise<MintStats> {
    const result = await this.contract.getMintStats(tokenId);
    return {
      totalMinted: result.totalMinted,
      totalRevenue: result.totalRevenue,
      uniqueHolders: result.uniqueHolders,
    };
  }

  /**
   * Check if mintable
   */
  async isMintable(tokenId: BigNumberish): Promise<boolean> {
    return this.contract.isMintable(tokenId);
  }

  /**
   * Get token balance
   */
  async balanceOf(address: string, tokenId: BigNumberish): Promise<bigint> {
    return this.contract.balanceOf(address, tokenId);
  }

  /**
   * Get total supply
   */
  async totalSupply(tokenId: BigNumberish): Promise<bigint> {
    return this.contract.totalSupply(tokenId);
  }

  /**
   * Check allowlist status
   */
  async isAllowlisted(tokenId: BigNumberish, address: string): Promise<boolean> {
    return this.contract.allowlist(tokenId, address);
  }
}

// ==================== Subscription Client ====================

export class SubscriptionClient {
  private contract: Contract;
  private signer?: Signer;

  constructor(
    address: string,
    providerOrSigner: Provider | Signer
  ) {
    this.contract = new ethers.Contract(address, SUBSCRIPTION_ABI, providerOrSigner);

    if ('getAddress' in providerOrSigner) {
      this.signer = providerOrSigner as Signer;
    }
  }

  /**
   * Subscribe to a plan
   */
  async subscribe(
    planId: BigNumberish,
    autoRenew: boolean,
    paymentToken: string
  ): Promise<void> {
    if (!this.signer) throw new Error('Signer required');

    const plan = await this.getPlan(planId);

    // Approve token spending
    const token = new ethers.Contract(paymentToken, ERC20_ABI, this.signer);
    const allowance = await token.allowance(
      await this.signer.getAddress(),
      await this.contract.getAddress()
    );

    if (allowance < plan.price) {
      const approveTx = await token.approve(await this.contract.getAddress(), plan.price);
      await approveTx.wait();
    }

    const tx = await this.contract.subscribe(planId, autoRenew);
    await tx.wait();
  }

  /**
   * Renew subscription
   */
  async renew(paymentToken: string): Promise<void> {
    if (!this.signer) throw new Error('Signer required');

    const sub = await this.getSubscription(await this.signer.getAddress());
    const plan = await this.getPlan(sub.planId);

    // Approve token spending
    const token = new ethers.Contract(paymentToken, ERC20_ABI, this.signer);
    const allowance = await token.allowance(
      await this.signer.getAddress(),
      await this.contract.getAddress()
    );

    if (allowance < plan.price) {
      const approveTx = await token.approve(await this.contract.getAddress(), plan.price);
      await approveTx.wait();
    }

    const tx = await this.contract.renew();
    await tx.wait();
  }

  /**
   * Cancel auto-renewal
   */
  async cancelAutoRenew(): Promise<void> {
    if (!this.signer) throw new Error('Signer required');

    const tx = await this.contract.cancelAutoRenew();
    await tx.wait();
  }

  /**
   * Check if user is subscribed
   */
  async isSubscribed(address: string): Promise<boolean> {
    return this.contract.isSubscribed(address);
  }

  /**
   * Get subscription details
   */
  async getSubscription(address: string): Promise<Subscription> {
    const result = await this.contract.getSubscription(address);
    return {
      planId: result.planId,
      startTime: result.startTime,
      endTime: result.endTime,
      streamsUsed: result.streamsUsed,
      autoRenew: result.autoRenew,
      active: result.active,
    };
  }

  /**
   * Get plan details
   */
  async getPlan(planId: BigNumberish): Promise<Plan> {
    const result = await this.contract.getPlan(planId);
    return {
      name: result.name,
      price: result.price,
      duration: result.duration,
      streamAllowance: result.streamAllowance,
      active: result.active,
      exists: result.exists,
    };
  }

  /**
   * Get current period info
   */
  async getCurrentPeriodInfo(): Promise<{
    period: bigint;
    startTime: bigint;
    endTime: bigint;
    revenue: bigint;
  }> {
    const result = await this.contract.getCurrentPeriodInfo();
    return {
      period: result.period,
      startTime: result.startTime,
      endTime: result.endTime,
      revenue: result.revenue,
    };
  }

  /**
   * Get artist pool stats
   */
  async getArtistPool(artist: string): Promise<{
    totalStreams: bigint;
    totalEarnings: bigint;
    lastDistribution: bigint;
  }> {
    const result = await this.contract.getArtistPool(artist);
    return {
      totalStreams: result.totalStreams,
      totalEarnings: result.totalEarnings,
      lastDistribution: result.lastDistribution,
    };
  }
}

// ==================== Main SDK ====================

export interface MycelixContractsConfig {
  royaltySplitter: string;
  musicNFT: string;
  subscription: string;
}

export class MycelixContractsSDK {
  public readonly royaltySplitter: RoyaltySplitterClient;
  public readonly musicNFT: MusicNFTClient;
  public readonly subscription: SubscriptionClient;

  constructor(
    config: MycelixContractsConfig,
    providerOrSigner: Provider | Signer
  ) {
    this.royaltySplitter = new RoyaltySplitterClient(
      config.royaltySplitter,
      providerOrSigner
    );
    this.musicNFT = new MusicNFTClient(config.musicNFT, providerOrSigner);
    this.subscription = new SubscriptionClient(config.subscription, providerOrSigner);
  }
}

export default MycelixContractsSDK;
