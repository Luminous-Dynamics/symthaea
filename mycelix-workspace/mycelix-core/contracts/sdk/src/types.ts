// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix SDK Type Definitions
 */

import type { TransactionResponse } from 'ethers';

/**
 * Network configuration for connecting to Mycelix contracts
 */
export interface NetworkConfig {
  /** Chain ID of the network */
  chainId: number;
  /** RPC URL for connecting to the network */
  rpcUrl: string;
  /** Contract addresses on this network */
  contracts: ContractAddresses;
  /** Block explorer URL for this network */
  blockExplorer: string;
}

/**
 * Contract addresses for a network deployment
 */
export interface ContractAddresses {
  /** MycelixRegistry contract address */
  registry: string;
  /** ReputationAnchor contract address */
  reputation: string;
  /** PaymentRouter contract address */
  payment: string;
}

/**
 * Client configuration options
 */
export interface ClientConfig {
  /** Network name or custom network configuration */
  network: 'sepolia' | 'local' | NetworkConfig;
  /** Private key for signing transactions (optional) */
  privateKey?: string;
  /** Custom provider instance (optional) */
  provider?: any;
}

/**
 * DID record information returned from the registry
 */
export interface DIDRecord {
  /** Owner address of the DID */
  owner: string;
  /** Timestamp when DID was registered (Unix seconds) */
  registeredAt: bigint;
  /** Timestamp when DID was last updated (Unix seconds) */
  updatedAt: bigint;
  /** Keccak256 hash of the metadata URI */
  metadataHash: string;
  /** Whether the DID has been revoked */
  revoked: boolean;
}

/**
 * Reputation information for an address
 */
export interface ReputationInfo {
  /** Cumulative reputation score (can be negative) */
  score: bigint;
  /** Number of reputation submissions received */
  submissions: bigint;
  /** Timestamp of last reputation update (Unix seconds) */
  lastUpdate: bigint;
}

/**
 * Payment/escrow information
 */
export interface PaymentInfo {
  /** Address that created the payment */
  sender: string;
  /** Address that will receive the payment */
  recipient: string;
  /** Payment amount in wei */
  amount: bigint;
  /** Unix timestamp when payment can be released */
  releaseTime: bigint;
  /** Whether payment has been released to recipient */
  released: boolean;
  /** Whether payment has been refunded to sender */
  refunded: boolean;
}

/**
 * Transaction result with hash and wait function
 */
export type TransactionResult = TransactionResponse;

/**
 * Supported network names
 */
export type NetworkName = 'sepolia' | 'local';

/**
 * Registry operations interface
 */
export interface IRegistryClient {
  registerDID(didString: string, metadataUri: string): Promise<TransactionResult>;
  getDID(didString: string): Promise<DIDRecord>;
  updateMetadata(didString: string, newMetadataUri: string): Promise<TransactionResult>;
  revokeDID(didString: string): Promise<TransactionResult>;
  getRegistrationFee(): Promise<bigint>;
  getTotalDids(): Promise<bigint>;
}

/**
 * Reputation operations interface
 */
export interface IReputationClient {
  submitReputation(
    subject: string,
    contextHash: string,
    score: number,
    proof?: string
  ): Promise<TransactionResult>;
  getReputation(subject: string): Promise<ReputationInfo>;
  anchorMerkleRoot(root: string, epoch: number): Promise<TransactionResult>;
}

/**
 * Payment operations interface
 */
export interface IPaymentClient {
  createPayment(
    recipient: string,
    amount: bigint,
    releaseTime: number,
    conditionHash?: string
  ): Promise<TransactionResult>;
  releasePayment(paymentId: number): Promise<TransactionResult>;
  refundPayment(paymentId: number): Promise<TransactionResult>;
  getPayment(paymentId: number): Promise<PaymentInfo>;
}

/**
 * Main client interface
 */
export interface IMycelixClient {
  readonly registry: IRegistryClient;
  readonly reputation: IReputationClient;
  readonly payment: IPaymentClient;
  readonly config: NetworkConfig;

  getAddress(): string | undefined;
  getBalance(): Promise<bigint>;
  getTxUrl(txHash: string): string;
  getAddressUrl(address: string): string;
}
