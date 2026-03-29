// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Web3.Storage Provider
 *
 * Uses web3.storage (w3up) for decentralized file storage.
 * Provides IPFS + Filecoin storage with pinning.
 */

import { create } from '@web3-storage/w3up-client';
import { StoreMemory } from '@web3-storage/access/stores/store-memory';
import * as Signer from '@ucanto/principal/ed25519';
import { CarReader } from '@ipld/car';
import pRetry from 'p-retry';
import mime from 'mime-types';

import {
  StorageProvider,
  UploadOptions,
  UploadResult,
  DirectoryUploadResult,
  DEFAULT_GATEWAYS,
} from '../types';

interface Web3StorageConfig {
  email?: string;
  space?: string;
  privateKey?: string;
  gatewayUrl?: string;
}

export class Web3StorageProvider implements StorageProvider {
  name = 'web3.storage';
  private client: Awaited<ReturnType<typeof create>> | null = null;
  private config: Web3StorageConfig;
  private gatewayUrl: string;
  private initialized = false;

  constructor(config: Web3StorageConfig = {}) {
    this.config = config;
    this.gatewayUrl = config.gatewayUrl || DEFAULT_GATEWAYS[0].url;
  }

  /**
   * Initialize the client with stored credentials or new registration
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    // Create client with memory store (or use persistent store in production)
    const principal = this.config.privateKey
      ? Signer.parse(this.config.privateKey)
      : await Signer.generate();

    const store = new StoreMemory();
    this.client = await create({ principal, store });

    // If email provided, handle authorization
    if (this.config.email) {
      // In production, this would handle the email verification flow
      console.log(`Web3.Storage client initialized for ${this.config.email}`);
    }

    this.initialized = true;
  }

  /**
   * Upload a single file
   */
  async upload(
    file: Buffer | Blob,
    name: string,
    options: UploadOptions = {}
  ): Promise<UploadResult> {
    await this.initialize();

    if (!this.client) {
      throw new Error('Web3.Storage client not initialized');
    }

    const { onProgress, metadata } = options;

    // Convert to File object
    const mimeType = mime.lookup(name) || 'application/octet-stream';
    const fileObj = new File([file], name, { type: mimeType });

    // Upload with retry
    const cid = await pRetry(
      async () => {
        const result = await this.client!.uploadFile(fileObj, {
          onShardStored: (shard) => {
            if (onProgress) {
              // Estimate progress based on shard uploads
              onProgress(50); // Simplified progress
            }
          },
        });
        return result;
      },
      { retries: 3 }
    );

    const cidString = cid.toString();
    const size = file instanceof Buffer ? file.length : file.size;

    return {
      cid: cidString,
      url: this.getUrl(cidString),
      size,
      metadata,
    };
  }

  /**
   * Upload a directory of files
   */
  async uploadDirectory(
    files: Map<string, Buffer>,
    options: UploadOptions = {}
  ): Promise<DirectoryUploadResult> {
    await this.initialize();

    if (!this.client) {
      throw new Error('Web3.Storage client not initialized');
    }

    // Convert to File array
    const fileArray: File[] = [];
    for (const [name, buffer] of files) {
      const mimeType = mime.lookup(name) || 'application/octet-stream';
      fileArray.push(new File([buffer], name, { type: mimeType }));
    }

    // Upload directory
    const cid = await pRetry(
      async () => {
        const result = await this.client!.uploadDirectory(fileArray);
        return result;
      },
      { retries: 3 }
    );

    const cidString = cid.toString();

    // Build file list
    const fileList = Array.from(files.entries()).map(([name, buffer]) => ({
      name,
      cid: '', // Individual CIDs would need to be resolved
      size: buffer.length,
    }));

    return {
      cid: cidString,
      url: this.getUrl(cidString),
      files: fileList,
    };
  }

  /**
   * Get file content by CID
   */
  async get(cid: string): Promise<Buffer | null> {
    const url = this.getUrl(cid);

    try {
      const response = await pRetry(
        async () => {
          const res = await fetch(url);
          if (!res.ok) throw new Error(`Failed to fetch: ${res.status}`);
          return res;
        },
        { retries: 3 }
      );

      const arrayBuffer = await response.arrayBuffer();
      return Buffer.from(arrayBuffer);
    } catch (error) {
      console.error('Failed to get file from IPFS:', error);
      return null;
    }
  }

  /**
   * Pin a CID (already pinned by default in web3.storage)
   */
  async pin(cid: string): Promise<boolean> {
    // Web3.storage automatically pins content
    return true;
  }

  /**
   * Unpin a CID
   */
  async unpin(cid: string): Promise<boolean> {
    // Would need to use the web3.storage API to remove
    console.log(`Unpin requested for ${cid}`);
    return true;
  }

  /**
   * Get gateway URL for a CID
   */
  getUrl(cid: string): string {
    return `${this.gatewayUrl}/ipfs/${cid}`;
  }

  /**
   * Get subdomain-style URL (better for browser caching)
   */
  getSubdomainUrl(cid: string): string {
    return `https://${cid}.ipfs.w3s.link`;
  }
}
