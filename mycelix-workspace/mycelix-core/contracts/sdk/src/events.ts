// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Event Listening
 *
 * Subscribe to contract events for real-time updates.
 */

import { ethers, Contract, Provider, EventLog, Log } from 'ethers';

// Event ABIs
const REGISTRY_EVENTS = [
  "event DIDRegistered(bytes32 indexed did, address indexed owner)",
  "event DIDUpdated(bytes32 indexed did, bytes32 newMetadataHash)",
  "event DIDRevoked(bytes32 indexed did)",
];

const REPUTATION_EVENTS = [
  "event ReputationSubmitted(address indexed subject, address indexed submitter, bytes32 contextHash, int256 score)",
  "event MerkleRootAnchored(bytes32 indexed root, uint256 epoch)",
];

const PAYMENT_EVENTS = [
  "event PaymentCreated(uint256 indexed paymentId, address indexed sender, address indexed recipient, uint256 amount)",
  "event PaymentReleased(uint256 indexed paymentId)",
  "event PaymentRefunded(uint256 indexed paymentId)",
];

export interface DIDRegisteredEvent {
  did: string;
  owner: string;
  blockNumber: number;
  transactionHash: string;
}

export interface DIDUpdatedEvent {
  did: string;
  newMetadataHash: string;
  blockNumber: number;
  transactionHash: string;
}

export interface ReputationSubmittedEvent {
  subject: string;
  submitter: string;
  contextHash: string;
  score: bigint;
  blockNumber: number;
  transactionHash: string;
}

export interface PaymentCreatedEvent {
  paymentId: bigint;
  sender: string;
  recipient: string;
  amount: bigint;
  blockNumber: number;
  transactionHash: string;
}

export type EventCallback<T> = (event: T) => void;

/**
 * Event listener for Mycelix contracts
 */
export class MycelixEvents {
  private registryContract: Contract;
  private reputationContract: Contract;
  private paymentContract: Contract;
  private listeners: Map<string, Function[]> = new Map();

  constructor(
    provider: Provider,
    contracts: {
      registry: string;
      reputation: string;
      payment: string;
    }
  ) {
    this.registryContract = new Contract(
      contracts.registry,
      REGISTRY_EVENTS,
      provider
    );
    this.reputationContract = new Contract(
      contracts.reputation,
      REPUTATION_EVENTS,
      provider
    );
    this.paymentContract = new Contract(
      contracts.payment,
      PAYMENT_EVENTS,
      provider
    );
  }

  /**
   * Subscribe to DID registration events
   */
  onDIDRegistered(callback: EventCallback<DIDRegisteredEvent>): () => void {
    const handler = (did: string, owner: string, event: EventLog) => {
      callback({
        did,
        owner,
        blockNumber: event.blockNumber,
        transactionHash: event.transactionHash,
      });
    };

    this.registryContract.on('DIDRegistered', handler);

    return () => {
      this.registryContract.off('DIDRegistered', handler);
    };
  }

  /**
   * Subscribe to DID update events
   */
  onDIDUpdated(callback: EventCallback<DIDUpdatedEvent>): () => void {
    const handler = (did: string, newMetadataHash: string, event: EventLog) => {
      callback({
        did,
        newMetadataHash,
        blockNumber: event.blockNumber,
        transactionHash: event.transactionHash,
      });
    };

    this.registryContract.on('DIDUpdated', handler);

    return () => {
      this.registryContract.off('DIDUpdated', handler);
    };
  }

  /**
   * Subscribe to reputation submission events
   */
  onReputationSubmitted(callback: EventCallback<ReputationSubmittedEvent>): () => void {
    const handler = (
      subject: string,
      submitter: string,
      contextHash: string,
      score: bigint,
      event: EventLog
    ) => {
      callback({
        subject,
        submitter,
        contextHash,
        score,
        blockNumber: event.blockNumber,
        transactionHash: event.transactionHash,
      });
    };

    this.reputationContract.on('ReputationSubmitted', handler);

    return () => {
      this.reputationContract.off('ReputationSubmitted', handler);
    };
  }

  /**
   * Subscribe to payment creation events
   */
  onPaymentCreated(callback: EventCallback<PaymentCreatedEvent>): () => void {
    const handler = (
      paymentId: bigint,
      sender: string,
      recipient: string,
      amount: bigint,
      event: EventLog
    ) => {
      callback({
        paymentId,
        sender,
        recipient,
        amount,
        blockNumber: event.blockNumber,
        transactionHash: event.transactionHash,
      });
    };

    this.paymentContract.on('PaymentCreated', handler);

    return () => {
      this.paymentContract.off('PaymentCreated', handler);
    };
  }

  /**
   * Query historical DID registrations
   */
  async getDIDRegistrations(
    fromBlock: number = 0,
    toBlock: number | 'latest' = 'latest'
  ): Promise<DIDRegisteredEvent[]> {
    const filter = this.registryContract.filters.DIDRegistered();
    const events = await this.registryContract.queryFilter(filter, fromBlock, toBlock);

    return events.map((event) => {
      const log = event as EventLog;
      return {
        did: log.args[0],
        owner: log.args[1],
        blockNumber: log.blockNumber,
        transactionHash: log.transactionHash,
      };
    });
  }

  /**
   * Query historical payments
   */
  async getPayments(
    fromBlock: number = 0,
    toBlock: number | 'latest' = 'latest'
  ): Promise<PaymentCreatedEvent[]> {
    const filter = this.paymentContract.filters.PaymentCreated();
    const events = await this.paymentContract.queryFilter(filter, fromBlock, toBlock);

    return events.map((event) => {
      const log = event as EventLog;
      return {
        paymentId: log.args[0],
        sender: log.args[1],
        recipient: log.args[2],
        amount: log.args[3],
        blockNumber: log.blockNumber,
        transactionHash: log.transactionHash,
      };
    });
  }

  /**
   * Remove all event listeners
   */
  removeAllListeners(): void {
    this.registryContract.removeAllListeners();
    this.reputationContract.removeAllListeners();
    this.paymentContract.removeAllListeners();
  }
}

export default MycelixEvents;
