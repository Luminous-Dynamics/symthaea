// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix SDK
 *
 * TypeScript SDK for interacting with Mycelix smart contracts.
 *
 * @example
 * ```typescript
 * import { MycelixClient } from '@mycelix/sdk';
 *
 * const client = new MycelixClient({
 *   network: 'sepolia',
 *   privateKey: process.env.PRIVATE_KEY
 * });
 *
 * // Register a DID
 * const tx = await client.registry.registerDID('did:mycelix:example', 'ipfs://metadata');
 * console.log('DID registered:', tx.hash);
 * ```
 */

import { ethers, Contract, Wallet, Provider, JsonRpcProvider } from 'ethers';

// Re-export types for consumers
export type {
  NetworkConfig,
  ContractAddresses,
  ClientConfig,
  DIDRecord,
  ReputationInfo,
  PaymentInfo,
  TransactionResult,
  NetworkName,
  IRegistryClient,
  IReputationClient,
  IPaymentClient,
  IMycelixClient,
} from './types.ts';

// Re-export event listener
export {
  MycelixEvents,
  type DIDRegisteredEvent,
  type DIDUpdatedEvent,
  type ReputationSubmittedEvent,
  type PaymentCreatedEvent,
  type EventCallback,
} from './events.ts';

// Contract ABIs (simplified for key functions)
const REGISTRY_ABI = [
  "function registerDID(bytes32 did, bytes32 metadataHash) external payable",
  "function didRecords(bytes32 did) external view returns (address owner, uint256 registeredAt, uint256 updatedAt, bytes32 metadataHash, bool revoked)",
  "function updateMetadata(bytes32 did, bytes32 newMetadataHash) external",
  "function revokeDID(bytes32 did) external",
  "function registrationFee() external view returns (uint256)",
  "function totalDids() external view returns (uint256)",
  "event DIDRegistered(bytes32 indexed did, address indexed owner)"
];

const REPUTATION_ABI = [
  "function submitReputation(address subject, bytes32 contextHash, int256 score, bytes calldata proof) external",
  "function getReputation(address subject) external view returns (int256 score, uint256 submissions, uint256 lastUpdate)",
  "function getReputationInContext(address subject, bytes32 contextHash) external view returns (int256)",
  "function anchorMerkleRoot(bytes32 root, uint256 epoch) external",
  "event ReputationSubmitted(address indexed subject, address indexed submitter, bytes32 contextHash, int256 score)"
];

const PAYMENT_ABI = [
  "function createPayment(address recipient, uint256 amount, uint256 releaseTime, bytes32 conditionHash) external payable returns (uint256 paymentId)",
  "function releasePayment(uint256 paymentId) external",
  "function refundPayment(uint256 paymentId) external",
  "function getPayment(uint256 paymentId) external view returns (address sender, address recipient, uint256 amount, uint256 releaseTime, bool released, bool refunded)",
  "event PaymentCreated(uint256 indexed paymentId, address indexed sender, address indexed recipient, uint256 amount)"
];

// Network configurations
export interface NetworkConfig {
  chainId: number;
  rpcUrl: string;
  contracts: {
    registry: string;
    reputation: string;
    payment: string;
  };
  blockExplorer: string;
}

export const NETWORKS: Record<string, NetworkConfig> = {
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
  local: {
    chainId: 31337,
    rpcUrl: "http://127.0.0.1:8545",
    contracts: {
      registry: "0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0",
      reputation: "0x5FbDB2315678afecb367f032d93F642f64180aa3",
      payment: "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512",
    },
    blockExplorer: "http://localhost:8545",
  },
  // Polygon Amoy Testnet (addresses TBD after deployment)
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
  // Polygon Mainnet (addresses TBD after deployment)
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
  // Ethereum Mainnet (addresses TBD after deployment)
  mainnet: {
    chainId: 1,
    rpcUrl: "https://eth.llamarpc.com",
    contracts: {
      registry: "0x0000000000000000000000000000000000000000",
      reputation: "0x0000000000000000000000000000000000000000",
      payment: "0x0000000000000000000000000000000000000000",
    },
    blockExplorer: "https://etherscan.io",
  },
};

export interface ClientConfig {
  network: keyof typeof NETWORKS | NetworkConfig;
  privateKey?: string;
  provider?: Provider;
}

/**
 * Registry client for DID operations
 */
export class RegistryClient {
  private contract: Contract;
  private signer?: Wallet;

  constructor(
    private config: NetworkConfig,
    provider: Provider,
    signer?: Wallet
  ) {
    const contractAddress = config.contracts.registry;
    this.signer = signer;
    this.contract = new Contract(
      contractAddress,
      REGISTRY_ABI,
      signer || provider
    );
  }

  /**
   * Register a new DID
   * @param didString The DID string (e.g., "did:mycelix:abc123")
   * @param metadataUri The metadata URI (e.g., "ipfs://Qm...")
   */
  async registerDID(didString: string, metadataUri: string): Promise<ethers.TransactionResponse> {
    if (!this.signer) throw new Error("Signer required for write operations");
    const didHash = ethers.keccak256(ethers.toUtf8Bytes(didString));
    const metadataHash = ethers.keccak256(ethers.toUtf8Bytes(metadataUri));
    const fee = await this.contract.registrationFee();
    return this.contract.registerDID(didHash, metadataHash, { value: fee });
  }

  /**
   * Get DID information by hash
   */
  async getDID(didString: string): Promise<{
    owner: string;
    registeredAt: bigint;
    updatedAt: bigint;
    metadataHash: string;
    revoked: boolean;
  }> {
    const didHash = ethers.keccak256(ethers.toUtf8Bytes(didString));
    const [owner, registeredAt, updatedAt, metadataHash, revoked] = await this.contract.didRecords(didHash);
    return { owner, registeredAt, updatedAt, metadataHash, revoked };
  }

  /**
   * Get registration fee
   */
  async getRegistrationFee(): Promise<bigint> {
    return this.contract.registrationFee();
  }

  /**
   * Get total registered DIDs
   */
  async getTotalDids(): Promise<bigint> {
    return this.contract.totalDids();
  }

  /**
   * Update DID metadata
   */
  async updateMetadata(didString: string, newMetadataUri: string): Promise<ethers.TransactionResponse> {
    if (!this.signer) throw new Error("Signer required for write operations");
    const didHash = ethers.keccak256(ethers.toUtf8Bytes(didString));
    const metadataHash = ethers.keccak256(ethers.toUtf8Bytes(newMetadataUri));
    return this.contract.updateMetadata(didHash, metadataHash);
  }

  /**
   * Revoke a DID
   */
  async revokeDID(didString: string): Promise<ethers.TransactionResponse> {
    if (!this.signer) throw new Error("Signer required for write operations");
    const didHash = ethers.keccak256(ethers.toUtf8Bytes(didString));
    return this.contract.revokeDID(didHash);
  }
}

/**
 * Reputation client for reputation operations
 */
export class ReputationClient {
  private contract: Contract;
  private signer?: Wallet;

  constructor(
    private config: NetworkConfig,
    provider: Provider,
    signer?: Wallet
  ) {
    const contractAddress = config.contracts.reputation;
    this.signer = signer;
    this.contract = new Contract(
      contractAddress,
      REPUTATION_ABI,
      signer || provider
    );
  }

  /**
   * Submit reputation for a subject
   */
  async submitReputation(
    subject: string,
    contextHash: string,
    score: number,
    proof: string = "0x"
  ): Promise<ethers.TransactionResponse> {
    if (!this.signer) throw new Error("Signer required for write operations");
    return this.contract.submitReputation(subject, contextHash, score, proof);
  }

  /**
   * Get reputation for a subject
   */
  async getReputation(subject: string): Promise<{
    score: bigint;
    submissions: bigint;
    lastUpdate: bigint;
  }> {
    const [score, submissions, lastUpdate] = await this.contract.getReputation(subject);
    return { score, submissions, lastUpdate };
  }

  /**
   * Anchor a Merkle root for batch reputation updates
   */
  async anchorMerkleRoot(root: string, epoch: number): Promise<ethers.TransactionResponse> {
    if (!this.signer) throw new Error("Signer required for write operations");
    return this.contract.anchorMerkleRoot(root, epoch);
  }
}

/**
 * Payment client for escrow and payment operations
 */
export class PaymentClient {
  private contract: Contract;
  private signer?: Wallet;

  constructor(
    private config: NetworkConfig,
    provider: Provider,
    signer?: Wallet
  ) {
    const contractAddress = config.contracts.payment;
    this.signer = signer;
    this.contract = new Contract(
      contractAddress,
      PAYMENT_ABI,
      signer || provider
    );
  }

  /**
   * Create a new escrow payment
   */
  async createPayment(
    recipient: string,
    amount: bigint,
    releaseTime: number,
    conditionHash: string = ethers.ZeroHash
  ): Promise<ethers.TransactionResponse> {
    if (!this.signer) throw new Error("Signer required for write operations");
    return this.contract.createPayment(recipient, amount, releaseTime, conditionHash, {
      value: amount,
    });
  }

  /**
   * Release a payment to recipient
   */
  async releasePayment(paymentId: number): Promise<ethers.TransactionResponse> {
    if (!this.signer) throw new Error("Signer required for write operations");
    return this.contract.releasePayment(paymentId);
  }

  /**
   * Refund a payment to sender
   */
  async refundPayment(paymentId: number): Promise<ethers.TransactionResponse> {
    if (!this.signer) throw new Error("Signer required for write operations");
    return this.contract.refundPayment(paymentId);
  }

  /**
   * Get payment details
   */
  async getPayment(paymentId: number): Promise<{
    sender: string;
    recipient: string;
    amount: bigint;
    releaseTime: bigint;
    released: boolean;
    refunded: boolean;
  }> {
    const [sender, recipient, amount, releaseTime, released, refunded] =
      await this.contract.getPayment(paymentId);
    return { sender, recipient, amount, releaseTime, released, refunded };
  }
}

/**
 * Main Mycelix client
 */
export class MycelixClient {
  public readonly registry: RegistryClient;
  public readonly reputation: ReputationClient;
  public readonly payment: PaymentClient;
  public readonly config: NetworkConfig;
  public readonly provider: Provider;
  public readonly signer?: Wallet;

  constructor(clientConfig: ClientConfig) {
    // Resolve network config
    if (typeof clientConfig.network === "string") {
      const netConfig = NETWORKS[clientConfig.network];
      if (!netConfig) {
        throw new Error(`Unknown network: ${clientConfig.network}`);
      }
      this.config = netConfig;
    } else {
      this.config = clientConfig.network;
    }

    // Set up provider
    this.provider = clientConfig.provider || new JsonRpcProvider(this.config.rpcUrl);

    // Set up signer if private key provided
    if (clientConfig.privateKey) {
      this.signer = new Wallet(clientConfig.privateKey, this.provider);
    }

    // Initialize clients
    this.registry = new RegistryClient(this.config, this.provider, this.signer);
    this.reputation = new ReputationClient(this.config, this.provider, this.signer);
    this.payment = new PaymentClient(this.config, this.provider, this.signer);
  }

  /**
   * Get the connected wallet address
   */
  getAddress(): string | undefined {
    return this.signer?.address;
  }

  /**
   * Get wallet balance
   */
  async getBalance(): Promise<bigint> {
    if (!this.signer) throw new Error("No signer configured");
    return this.provider.getBalance(this.signer.address);
  }

  /**
   * Get block explorer URL for a transaction
   */
  getTxUrl(txHash: string): string {
    return `${this.config.blockExplorer}/tx/${txHash}`;
  }

  /**
   * Get block explorer URL for an address
   */
  getAddressUrl(address: string): string {
    return `${this.config.blockExplorer}/address/${address}`;
  }
}

export default MycelixClient;
