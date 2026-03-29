// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Storage Types for Mycelix
 */

export interface StoredFile {
  cid: string;
  name: string;
  size: number;
  mimeType: string;
  url: string;
  gateway: string;
  uploadedAt: Date;
}

export interface StorageMetadata {
  trackId?: string;
  artistId?: string;
  albumId?: string;
  type: 'audio' | 'image' | 'metadata' | 'waveform' | 'stems';
  originalName: string;
  uploadedBy?: string;
}

export interface UploadOptions {
  metadata?: StorageMetadata;
  pin?: boolean;
  wrapWithDirectory?: boolean;
  onProgress?: (progress: number) => void;
}

export interface UploadResult {
  cid: string;
  url: string;
  size: number;
  metadata?: StorageMetadata;
}

export interface DirectoryUploadResult {
  cid: string;
  url: string;
  files: {
    name: string;
    cid: string;
    size: number;
  }[];
}

export interface StorageProvider {
  name: string;
  upload(file: Buffer | Blob, name: string, options?: UploadOptions): Promise<UploadResult>;
  uploadDirectory(files: Map<string, Buffer>, options?: UploadOptions): Promise<DirectoryUploadResult>;
  get(cid: string): Promise<Buffer | null>;
  pin(cid: string): Promise<boolean>;
  unpin(cid: string): Promise<boolean>;
  getUrl(cid: string): string;
}

export interface GatewayConfig {
  url: string;
  priority: number;
  supportsSubdomains: boolean;
}

export const DEFAULT_GATEWAYS: GatewayConfig[] = [
  { url: 'https://w3s.link', priority: 1, supportsSubdomains: true },
  { url: 'https://dweb.link', priority: 2, supportsSubdomains: true },
  { url: 'https://ipfs.io', priority: 3, supportsSubdomains: false },
  { url: 'https://cloudflare-ipfs.com', priority: 4, supportsSubdomains: false },
];
