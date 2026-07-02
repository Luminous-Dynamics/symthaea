// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * IPFS HTTP Client Provider
 *
 * Direct IPFS node connection for self-hosted or Infura IPFS.
 */

import { create, IPFSHTTPClient } from 'ipfs-http-client';
import { CID } from 'multiformats/cid';
import pRetry from 'p-retry';
import mime from 'mime-types';

import {
  StorageProvider,
  UploadOptions,
  UploadResult,
  DirectoryUploadResult,
  DEFAULT_GATEWAYS,
} from '../types';

interface IPFSConfig {
  url?: string;
  projectId?: string;
  projectSecret?: string;
  gatewayUrl?: string;
}

export class IPFSProvider implements StorageProvider {
  name = 'ipfs';
  private client: IPFSHTTPClient;
  private gatewayUrl: string;

  constructor(config: IPFSConfig = {}) {
    const url = config.url || 'http://localhost:5001';
    this.gatewayUrl = config.gatewayUrl || DEFAULT_GATEWAYS[2].url;

    // Build authorization header for Infura
    const auth = config.projectId && config.projectSecret
      ? 'Basic ' + Buffer.from(`${config.projectId}:${config.projectSecret}`).toString('base64')
      : undefined;

    this.client = create({
      url,
      headers: auth ? { authorization: auth } : undefined,
    });
  }

  /**
   * Upload a single file
   */
  async upload(
    file: Buffer | Blob,
    name: string,
    options: UploadOptions = {}
  ): Promise<UploadResult> {
    const { onProgress, metadata, pin = true } = options;

    const buffer = file instanceof Blob
      ? Buffer.from(await file.arrayBuffer())
      : file;

    // Add file to IPFS
    const result = await pRetry(
      async () => {
        const added = await this.client.add(buffer, {
          pin,
          progress: (bytes) => {
            if (onProgress) {
              const progress = (bytes / buffer.length) * 100;
              onProgress(Math.min(progress, 100));
            }
          },
        });
        return added;
      },
      { retries: 3 }
    );

    return {
      cid: result.cid.toString(),
      url: this.getUrl(result.cid.toString()),
      size: buffer.length,
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
    const { pin = true } = options;

    // Prepare files for upload
    const fileEntries = Array.from(files.entries()).map(([name, content]) => ({
      path: name,
      content,
    }));

    // Add all files to IPFS
    let rootCid = '';
    const uploadedFiles: { name: string; cid: string; size: number }[] = [];

    for await (const result of this.client.addAll(fileEntries, {
      pin,
      wrapWithDirectory: true,
    })) {
      if (result.path === '') {
        // This is the root directory
        rootCid = result.cid.toString();
      } else {
        uploadedFiles.push({
          name: result.path,
          cid: result.cid.toString(),
          size: files.get(result.path)?.length || 0,
        });
      }
    }

    return {
      cid: rootCid,
      url: this.getUrl(rootCid),
      files: uploadedFiles,
    };
  }

  /**
   * Get file content by CID
   */
  async get(cid: string): Promise<Buffer | null> {
    try {
      const chunks: Uint8Array[] = [];

      for await (const chunk of this.client.cat(cid)) {
        chunks.push(chunk);
      }

      return Buffer.concat(chunks);
    } catch (error) {
      console.error('Failed to get file from IPFS:', error);
      return null;
    }
  }

  /**
   * Pin a CID
   */
  async pin(cid: string): Promise<boolean> {
    try {
      await this.client.pin.add(CID.parse(cid));
      return true;
    } catch (error) {
      console.error('Failed to pin CID:', error);
      return false;
    }
  }

  /**
   * Unpin a CID
   */
  async unpin(cid: string): Promise<boolean> {
    try {
      await this.client.pin.rm(CID.parse(cid));
      return true;
    } catch (error) {
      console.error('Failed to unpin CID:', error);
      return false;
    }
  }

  /**
   * Get gateway URL for a CID
   */
  getUrl(cid: string): string {
    return `${this.gatewayUrl}/ipfs/${cid}`;
  }

  /**
   * Check if IPFS node is online
   */
  async isOnline(): Promise<boolean> {
    try {
      const id = await this.client.id();
      return !!id.id;
    } catch {
      return false;
    }
  }

  /**
   * Get node info
   */
  async getNodeInfo(): Promise<{ id: string; addresses: string[] } | null> {
    try {
      const info = await this.client.id();
      return {
        id: info.id.toString(),
        addresses: info.addresses.map((a) => a.toString()),
      };
    } catch {
      return null;
    }
  }
}
