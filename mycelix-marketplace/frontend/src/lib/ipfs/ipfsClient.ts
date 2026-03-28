// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * IPFS Client Wrapper
 *
 * Handles file uploads to IPFS via Pinata
 */

/**
 * Default IPFS gateway fallback chain for censorship-resistant content retrieval.
 *
 * If the primary gateway (Pinata) is down, reads fall through to public gateways.
 * Uploads always go to Pinata (the only authenticated gateway).
 */
const DEFAULT_FALLBACK_GATEWAYS: string[] = [
  'https://w3s.link',
  'https://dweb.link',
  'https://ipfs.io',
  'https://cloudflare-ipfs.com',
];

/**
 * Pinata configuration
 */
export interface PinataConfig {
  jwt: string;
  gateway: string;
  allowMock: boolean;
}

/**
 * Resolve Pinata configuration from environment
 */
export function resolvePinataConfig(env: Record<string, string | undefined>): PinataConfig {
  return {
    jwt: env?.VITE_PINATA_JWT || '',
    gateway: env?.VITE_PINATA_GATEWAY || 'https://gateway.pinata.cloud',
    allowMock: env?.VITE_ALLOW_MOCK_IPFS === 'true',
  };
}

/**
 * Get Pinata configuration from environment
 */
function getPinataConfig(): PinataConfig {
  return resolvePinataConfig(import.meta.env as Record<string, string | undefined>);
}

/**
 * Upload a single file to IPFS via Pinata
 *
 * @param file - File to upload
 * @returns IPFS CID
 */
export async function uploadFile(file: File, configOverride?: PinataConfig): Promise<string> {
  const config = configOverride ?? getPinataConfig();

  if (!config.jwt) {
    if (config.allowMock) {
      console.warn('IPFS upload: No Pinata JWT configured, using mock CID');
      return generateMockCID(file.name);
    }
    throw new Error(
      'Pinata JWT missing. Set VITE_PINATA_JWT or enable VITE_ALLOW_MOCK_IPFS=true for mock uploads.'
    );
  }

  try {
    // Real Pinata implementation (Phase 5)
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('https://api.pinata.cloud/pinning/pinFileToIPFS', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${config.jwt}`,
      },
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Pinata upload failed: ${response.statusText}`);
    }

    const data = await response.json();
    return data.IpfsHash;
  } catch (error) {
    console.error('IPFS upload failed:', error);
    if (config.allowMock) {
      console.warn('Using mock CID due to IPFS upload failure');
      return generateMockCID(file.name);
    }
    throw error instanceof Error
      ? error
      : new Error('IPFS upload failed and mock uploads are disabled.');
  }
}

/**
 * Upload multiple files to IPFS
 *
 * @param files - Array of files to upload
 * @returns Array of IPFS CIDs
 */
export async function uploadFiles(files: File[], configOverride?: PinataConfig): Promise<string[]> {
  const uploadPromises = files.map((file) => uploadFile(file, configOverride));
  return Promise.all(uploadPromises);
}

/**
 * Get IPFS URL for a CID
 *
 * @param cid - IPFS content identifier
 * @returns Full IPFS gateway URL
 */
export function getIpfsUrl(cid: string): string {
  const config = getPinataConfig();
  return `${config.gateway}/ipfs/${cid}`;
}

/**
 * Get all IPFS gateway URLs for a CID (for retry logic).
 *
 * Returns the primary gateway first, then all fallback gateways.
 *
 * @param cid - IPFS content identifier
 * @returns Array of IPFS gateway URLs, ordered by preference
 */
export function getIpfsUrls(cid: string): string[] {
  const config = getPinataConfig();
  const urls = [`${config.gateway}/ipfs/${cid}`];
  for (const gw of DEFAULT_FALLBACK_GATEWAYS) {
    urls.push(`${gw}/ipfs/${cid}`);
  }
  return urls;
}

/**
 * Fetch content from IPFS with automatic gateway fallback.
 *
 * Tries the primary Pinata gateway first, then falls through to public
 * gateways. Each gateway gets a 10-second timeout before the next is tried.
 *
 * @param cid - IPFS content identifier
 * @returns Response from the first successful gateway
 * @throws Error if all gateways fail
 */
export async function fetchFromIpfs(cid: string): Promise<Response> {
  const urls = getIpfsUrls(cid);
  let lastError: Error | undefined;

  for (const url of urls) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 10_000);

      const response = await fetch(url, { signal: controller.signal });
      clearTimeout(timeout);

      if (response.ok) {
        return response;
      }
      lastError = new Error(`Gateway ${url} returned ${response.status}`);
    } catch (error) {
      lastError =
        error instanceof Error ? error : new Error(`Gateway ${url} failed`);
    }
  }

  throw lastError ?? new Error(`All ${urls.length} IPFS gateways failed for CID: ${cid}`);
}

/**
 * Generate mock CID for development (Phase 4)
 * Format matches real IPFS CIDs (Qm...)
 *
 * @param filename - Original filename
 * @returns Mock CID
 */
function generateMockCID(filename: string): string {
  const timestamp = Date.now();
  const hash = btoa(`${filename}:${timestamp}`).replace(/[^a-zA-Z0-9]/g, '');
  return `Qm${hash.substring(0, 44).padEnd(44, '0')}`;
}

/**
 * Validate if string is a valid IPFS CID
 *
 * @param cid - String to validate
 * @returns true if valid CID format
 */
export function isValidCID(cid: string): boolean {
  // Basic validation: starts with Qm and is 46 characters
  return /^Qm[a-zA-Z0-9]{44}$/.test(cid);
}

/**
 * Check if Pinata is configured
 *
 * @returns true if Pinata JWT is available
 */
export function isPinataConfigured(): boolean {
  const config = getPinataConfig();
  return config.jwt.length > 0;
}
