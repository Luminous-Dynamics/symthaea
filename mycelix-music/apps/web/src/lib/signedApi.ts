import { ethers } from 'ethers';
import {
  signSongPayload,
  signPlayPayload,
  signClaimPayload,
  signSongTypedPayload,
  signPlayTypedPayload,
  signClaimTypedPayload,
} from '@mycelix/sdk';

type SignedApiError = Error & {
  status?: number;
  reason?: string;
  meta?: Record<string, unknown>;
};

type BaseBody = Record<string, unknown>;

const chainId = parseInt(process.env.NEXT_PUBLIC_CHAIN_ID || '31337', 10);
const routerAddress = process.env.NEXT_PUBLIC_ROUTER_ADDRESS;
const preferEip712 = String(process.env.NEXT_PUBLIC_ENABLE_EIP712 || 'true').toLowerCase() === 'true';

function shouldUseEip712() {
  return preferEip712 && !!routerAddress;
}

async function postJson(path: string, body: BaseBody) {
  const adminKey = process.env.NEXT_PUBLIC_API_KEY;
  const resp = await fetch(path, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(adminKey ? { 'x-api-key': adminKey } : {}),
    },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    let reason = '';
    let meta: Record<string, unknown> | undefined;
    try {
      const data = await resp.json();
      reason = data?.reason || data?.error || '';
      meta = data;
    } catch {
      reason = await resp.text().catch(() => '');
    }
    const suffix = reason ? `: ${reason}` : '';
    const err: SignedApiError = new Error(`HTTP ${resp.status}${suffix}`);
    err.status = resp.status;
    err.reason = reason || undefined;
    err.meta = meta;
    throw err;
  }
  return resp;
}

export async function postSignedSong({
  signer,
  data,
}: {
  signer: ethers.Signer;
  data: {
    id: string;
    title: string;
    artist: string;
    artistAddress: string;
    genre: string;
    description?: string;
    ipfsHash: string;
    paymentModel: string;
    coverArt?: string | null;
    audioUrl?: string | null;
    claimStreamId?: string | null;
    nonce?: string;
  };
}) {
  const { signer: signerAddr, timestamp, signature } = await signSongPayload(signer, {
    id: data.id,
    artistAddress: data.artistAddress,
    ipfsHash: data.ipfsHash,
    paymentModel: data.paymentModel,
    nonce: data.nonce,
  });
  const typed = shouldUseEip712()
    ? await signSongTypedPayload(signer, chainId, routerAddress!, {
      id: data.id,
      artistAddress: data.artistAddress,
      ipfsHash: data.ipfsHash,
      paymentModel: data.paymentModel,
      nonce: data.nonce,
    })
    : null;
  return postJson('/api/songs', {
    ...data,
    signer: signerAddr,
    timestamp,
    signature,
    method: typed ? 'eip712' : 'eip191',
    ...(typed ? { signature: typed.signature, timestamp: typed.timestamp, nonce: typed.nonce } : {}),
  });
}

export async function postSignedPlay({
  signer,
  songId,
  listenerAddress,
  amount,
  paymentType,
  nonce,
}: {
  signer: ethers.Signer;
  songId: string;
  listenerAddress: string;
  amount: string;
  paymentType: string;
  nonce?: string;
}) {
  const { signer: signerAddr, timestamp, signature } = await signPlayPayload(signer, {
    songId,
    listener: listenerAddress,
    amount,
    paymentType,
    nonce,
  });
  const typed = shouldUseEip712()
    ? await signPlayTypedPayload(signer, chainId, routerAddress!, {
      songId,
      listener: listenerAddress,
      amount,
      paymentType,
      nonce,
    })
    : null;
  return postJson(`/api/songs/${encodeURIComponent(songId)}/play`, {
    listenerAddress,
    amount: Number(amount),
    paymentType,
    signer: signerAddr,
    timestamp,
    signature,
    method: typed ? 'eip712' : 'eip191',
    ...(typed ? { signature: typed.signature, timestamp: typed.timestamp, nonce: typed.nonce } : {}),
    nonce,
  });
}

export async function postSignedClaim({
  signer,
  data,
}: {
  signer: ethers.Signer;
  data: {
    songId: string;
    title: string;
    artist: string;
    ipfsHash: string;
    artistAddress: string;
    nonce?: string;
  };
}) {
  const { signer: signerAddr, timestamp, signature } = await signClaimPayload(signer, {
    songId: data.songId,
    artistAddress: data.artistAddress,
    ipfsHash: data.ipfsHash,
    title: data.title,
    nonce: data.nonce,
  });
  const typed = shouldUseEip712()
    ? await signClaimTypedPayload(signer, chainId, routerAddress!, {
      songId: data.songId,
      artistAddress: data.artistAddress,
      ipfsHash: data.ipfsHash,
      title: data.title,
      nonce: data.nonce,
    })
    : null;
  return postJson('/api/create-dkg-claim', {
    ...data,
    signer: signerAddr,
    timestamp,
    signature,
    method: typed ? 'eip712' : 'eip191',
    ...(typed ? { signature: typed.signature, timestamp: typed.timestamp, nonce: typed.nonce } : {}),
  });
}
