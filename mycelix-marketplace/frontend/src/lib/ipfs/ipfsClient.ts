/**
 * IPFS Client Wrapper
 *
 * Handles file uploads to IPFS via Pinata
 */

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
