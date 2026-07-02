// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/storage
 *
 * Decentralized storage for Mycelix Music platform.
 * Supports IPFS and Web3.Storage backends.
 */

import { Web3StorageProvider } from './providers/web3storage';
import { IPFSProvider } from './providers/ipfs';
import {
  StorageProvider,
  StoredFile,
  StorageMetadata,
  UploadOptions,
  UploadResult,
  DirectoryUploadResult,
  DEFAULT_GATEWAYS,
  GatewayConfig,
} from './types';

// ============================================================================
// Storage Manager
// ============================================================================

interface StorageManagerConfig {
  primaryProvider: 'web3storage' | 'ipfs';
  web3storage?: {
    email?: string;
    privateKey?: string;
  };
  ipfs?: {
    url?: string;
    projectId?: string;
    projectSecret?: string;
  };
  gateways?: GatewayConfig[];
}

/**
 * Unified storage manager for Mycelix
 */
export class StorageManager {
  private provider: StorageProvider;
  private gateways: GatewayConfig[];

  constructor(config: StorageManagerConfig) {
    this.gateways = config.gateways || DEFAULT_GATEWAYS;

    // Initialize primary provider
    if (config.primaryProvider === 'web3storage') {
      this.provider = new Web3StorageProvider({
        email: config.web3storage?.email,
        privateKey: config.web3storage?.privateKey,
        gatewayUrl: this.gateways[0]?.url,
      });
    } else {
      this.provider = new IPFSProvider({
        url: config.ipfs?.url,
        projectId: config.ipfs?.projectId,
        projectSecret: config.ipfs?.projectSecret,
        gatewayUrl: this.gateways[0]?.url,
      });
    }
  }

  /**
   * Upload audio file
   */
  async uploadAudio(
    file: Buffer | Blob,
    trackId: string,
    originalName: string,
    options: Omit<UploadOptions, 'metadata'> = {}
  ): Promise<UploadResult> {
    return this.provider.upload(file, originalName, {
      ...options,
      metadata: {
        type: 'audio',
        trackId,
        originalName,
      },
    });
  }

  /**
   * Upload album/track artwork
   */
  async uploadArtwork(
    file: Buffer | Blob,
    entityId: string,
    entityType: 'track' | 'album' | 'artist',
    options: Omit<UploadOptions, 'metadata'> = {}
  ): Promise<UploadResult> {
    const name = `${entityType}_${entityId}_artwork.png`;
    return this.provider.upload(file, name, {
      ...options,
      metadata: {
        type: 'image',
        [entityType === 'track' ? 'trackId' : entityType === 'album' ? 'albumId' : 'artistId']: entityId,
        originalName: name,
      },
    });
  }

  /**
   * Upload track metadata as JSON
   */
  async uploadMetadata(
    metadata: Record<string, unknown>,
    trackId: string
  ): Promise<UploadResult> {
    const json = JSON.stringify(metadata, null, 2);
    const buffer = Buffer.from(json, 'utf-8');
    const name = `metadata_${trackId}.json`;

    return this.provider.upload(buffer, name, {
      metadata: {
        type: 'metadata',
        trackId,
        originalName: name,
      },
    });
  }

  /**
   * Upload waveform data
   */
  async uploadWaveform(
    waveformData: { peaks: number[]; duration: number },
    trackId: string
  ): Promise<UploadResult> {
    const json = JSON.stringify(waveformData);
    const buffer = Buffer.from(json, 'utf-8');
    const name = `waveform_${trackId}.json`;

    return this.provider.upload(buffer, name, {
      metadata: {
        type: 'waveform',
        trackId,
        originalName: name,
      },
    });
  }

  /**
   * Upload separated stems as a directory
   */
  async uploadStems(
    stems: {
      drums?: Buffer;
      bass?: Buffer;
      vocals?: Buffer;
      other?: Buffer;
    },
    trackId: string
  ): Promise<DirectoryUploadResult> {
    const files = new Map<string, Buffer>();

    if (stems.drums) files.set('drums.wav', stems.drums);
    if (stems.bass) files.set('bass.wav', stems.bass);
    if (stems.vocals) files.set('vocals.wav', stems.vocals);
    if (stems.other) files.set('other.wav', stems.other);

    return this.provider.uploadDirectory(files, {
      metadata: {
        type: 'stems',
        trackId,
        originalName: `stems_${trackId}`,
      },
    });
  }

  /**
   * Get file by CID
   */
  async get(cid: string): Promise<Buffer | null> {
    return this.provider.get(cid);
  }

  /**
   * Get JSON metadata by CID
   */
  async getMetadata<T = Record<string, unknown>>(cid: string): Promise<T | null> {
    const buffer = await this.get(cid);
    if (!buffer) return null;

    try {
      return JSON.parse(buffer.toString('utf-8'));
    } catch {
      return null;
    }
  }

  /**
   * Get URL for a CID using the fastest gateway
   */
  getUrl(cid: string): string {
    return this.provider.getUrl(cid);
  }

  /**
   * Get multiple gateway URLs for redundancy
   */
  getGatewayUrls(cid: string): string[] {
    return this.gateways.map((g) =>
      g.supportsSubdomains
        ? `https://${cid}.ipfs.${new URL(g.url).host}`
        : `${g.url}/ipfs/${cid}`
    );
  }

  /**
   * Pin content for long-term storage
   */
  async pin(cid: string): Promise<boolean> {
    return this.provider.pin(cid);
  }

  /**
   * Unpin content
   */
  async unpin(cid: string): Promise<boolean> {
    return this.provider.unpin(cid);
  }
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Parse IPFS URL to extract CID
 */
export function parseCID(url: string): string | null {
  // Handle various IPFS URL formats
  const patterns = [
    /ipfs:\/\/([a-zA-Z0-9]+)/,           // ipfs://Qm...
    /\/ipfs\/([a-zA-Z0-9]+)/,            // /ipfs/Qm...
    /([a-zA-Z0-9]+)\.ipfs\./,            // Qm....ipfs.dweb.link
    /^(Qm[a-zA-Z0-9]{44})$/,             // Raw CIDv0
    /^(bafy[a-zA-Z0-9]+)$/,              // Raw CIDv1
  ];

  for (const pattern of patterns) {
    const match = url.match(pattern);
    if (match) return match[1];
  }

  return null;
}

/**
 * Validate CID format
 */
export function isValidCID(cid: string): boolean {
  return /^(Qm[a-zA-Z0-9]{44}|bafy[a-zA-Z0-9]+)$/.test(cid);
}

/**
 * Convert IPFS URL to HTTP gateway URL
 */
export function toGatewayUrl(ipfsUrl: string, gateway = DEFAULT_GATEWAYS[0].url): string {
  const cid = parseCID(ipfsUrl);
  if (!cid) return ipfsUrl;
  return `${gateway}/ipfs/${cid}`;
}

/**
 * Convert HTTP gateway URL to IPFS protocol URL
 */
export function toIpfsUrl(gatewayUrl: string): string {
  const cid = parseCID(gatewayUrl);
  if (!cid) return gatewayUrl;
  return `ipfs://${cid}`;
}

// ============================================================================
// Exports
// ============================================================================

export {
  Web3StorageProvider,
  IPFSProvider,
  StorageProvider,
  StoredFile,
  StorageMetadata,
  UploadOptions,
  UploadResult,
  DirectoryUploadResult,
  GatewayConfig,
  DEFAULT_GATEWAYS,
};
