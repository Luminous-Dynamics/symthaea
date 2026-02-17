import { ethers } from 'ethers';

const DOMAIN_VERSION = '1';

export function buildDomain(chainId: number, verifyingContract: string) {
  return {
    name: 'MycelixMusic',
    version: DOMAIN_VERSION,
    chainId,
    verifyingContract,
  };
}

export const songType = {
  Song: [
    { name: 'id', type: 'string' },
    { name: 'artistAddress', type: 'address' },
    { name: 'ipfsHash', type: 'string' },
    { name: 'paymentModel', type: 'string' },
    { name: 'nonce', type: 'string' },
    { name: 'timestamp', type: 'uint256' },
  ],
};

export const playType = {
  Play: [
    { name: 'songId', type: 'string' },
    { name: 'listener', type: 'address' },
    { name: 'amount', type: 'string' },
    { name: 'paymentType', type: 'string' },
    { name: 'nonce', type: 'string' },
    { name: 'timestamp', type: 'uint256' },
  ],
};

export const claimType = {
  Claim: [
    { name: 'songId', type: 'string' },
    { name: 'artistAddress', type: 'address' },
    { name: 'ipfsHash', type: 'string' },
    { name: 'title', type: 'string' },
    { name: 'nonce', type: 'string' },
    { name: 'timestamp', type: 'uint256' },
  ],
};

export async function signTypedData(
  signer: ethers.Signer,
  domain: Record<string, any>,
  types: Record<string, Array<{ name: string; type: string }>>,
  value: Record<string, any>,
) {
  // ethers v6 uses signTypedData directly on the signer
  const signature = await signer.signTypedData(domain, types, value);
  return signature;
}

// ========== High-level signing functions ==========

function generateNonce(): string {
  return ethers.hexlify(ethers.randomBytes(16));
}

export interface SongPayload {
  id: string;
  artistAddress: string;
  ipfsHash: string;
  paymentModel: string;
  nonce?: string;
}

export interface PlayPayload {
  songId: string;
  listener: string;
  amount: string;
  paymentType: string;
  nonce?: string;
}

export interface ClaimPayload {
  songId: string;
  artistAddress: string;
  ipfsHash: string;
  title: string;
  nonce?: string;
}

export interface SignedResult {
  signer: string;
  timestamp: number;
  signature: string;
  nonce: string;
}

/**
 * Sign a song registration payload with EIP-191 personal_sign
 */
export async function signSongPayload(
  signer: ethers.Signer,
  payload: SongPayload
): Promise<SignedResult> {
  const address = await signer.getAddress();
  const timestamp = Math.floor(Date.now() / 1000);
  const nonce = payload.nonce || generateNonce();

  const message = JSON.stringify({
    ...payload,
    nonce,
    timestamp,
  });

  const signature = await signer.signMessage(message);

  return { signer: address, timestamp, signature, nonce };
}

/**
 * Sign a play event payload with EIP-191 personal_sign
 */
export async function signPlayPayload(
  signer: ethers.Signer,
  payload: PlayPayload
): Promise<SignedResult> {
  const address = await signer.getAddress();
  const timestamp = Math.floor(Date.now() / 1000);
  const nonce = payload.nonce || generateNonce();

  const message = JSON.stringify({
    ...payload,
    nonce,
    timestamp,
  });

  const signature = await signer.signMessage(message);

  return { signer: address, timestamp, signature, nonce };
}

/**
 * Sign a claim payload with EIP-191 personal_sign
 */
export async function signClaimPayload(
  signer: ethers.Signer,
  payload: ClaimPayload
): Promise<SignedResult> {
  const address = await signer.getAddress();
  const timestamp = Math.floor(Date.now() / 1000);
  const nonce = payload.nonce || generateNonce();

  const message = JSON.stringify({
    ...payload,
    nonce,
    timestamp,
  });

  const signature = await signer.signMessage(message);

  return { signer: address, timestamp, signature, nonce };
}

/**
 * Sign a song registration payload with EIP-712 typed data
 */
export async function signSongTypedPayload(
  signer: ethers.Signer,
  chainId: number,
  verifyingContract: string,
  payload: SongPayload
): Promise<SignedResult> {
  const address = await signer.getAddress();
  const timestamp = Math.floor(Date.now() / 1000);
  const nonce = payload.nonce || generateNonce();

  const domain = buildDomain(chainId, verifyingContract);
  const value = {
    id: payload.id,
    artistAddress: payload.artistAddress,
    ipfsHash: payload.ipfsHash,
    paymentModel: payload.paymentModel,
    nonce,
    timestamp,
  };

  const signature = await signTypedData(signer, domain, songType, value);

  return { signer: address, timestamp, signature, nonce };
}

/**
 * Sign a play event payload with EIP-712 typed data
 */
export async function signPlayTypedPayload(
  signer: ethers.Signer,
  chainId: number,
  verifyingContract: string,
  payload: PlayPayload
): Promise<SignedResult> {
  const address = await signer.getAddress();
  const timestamp = Math.floor(Date.now() / 1000);
  const nonce = payload.nonce || generateNonce();

  const domain = buildDomain(chainId, verifyingContract);
  const value = {
    songId: payload.songId,
    listener: payload.listener,
    amount: payload.amount,
    paymentType: payload.paymentType,
    nonce,
    timestamp,
  };

  const signature = await signTypedData(signer, domain, playType, value);

  return { signer: address, timestamp, signature, nonce };
}

/**
 * Sign a claim payload with EIP-712 typed data
 */
export async function signClaimTypedPayload(
  signer: ethers.Signer,
  chainId: number,
  verifyingContract: string,
  payload: ClaimPayload
): Promise<SignedResult> {
  const address = await signer.getAddress();
  const timestamp = Math.floor(Date.now() / 1000);
  const nonce = payload.nonce || generateNonce();

  const domain = buildDomain(chainId, verifyingContract);
  const value = {
    songId: payload.songId,
    artistAddress: payload.artistAddress,
    ipfsHash: payload.ipfsHash,
    title: payload.title,
    nonce,
    timestamp,
  };

  const signature = await signTypedData(signer, domain, claimType, value);

  return { signer: address, timestamp, signature, nonce };
}
