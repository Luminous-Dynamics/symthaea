import { TypedDataDomain, TypedDataField } from 'ethers';

type TypedPayload = {
  types: Record<string, Array<TypedDataField>>;
  value: Record<string, any>;
};

export function buildTypedPayload(context: string, req: any, timestamp: number, nonce?: string): TypedPayload | null {
  const n = nonce ?? '';
  if (context === 'song') {
    return {
      types: {
        Song: [
          { name: 'id', type: 'string' },
          { name: 'artistAddress', type: 'address' },
          { name: 'ipfsHash', type: 'string' },
          { name: 'paymentModel', type: 'string' },
          { name: 'nonce', type: 'string' },
          { name: 'timestamp', type: 'uint256' },
        ],
      },
      value: {
        id: String(req.body?.id || ''),
        artistAddress: String(req.body?.artistAddress || ''),
        ipfsHash: String(req.body?.ipfsHash || ''),
        paymentModel: String(req.body?.paymentModel || ''),
        nonce: n,
        timestamp,
      },
    };
  }
  if (context === 'play') {
    return {
      types: {
        Play: [
          { name: 'songId', type: 'string' },
          { name: 'listener', type: 'address' },
          { name: 'amount', type: 'string' },
          { name: 'paymentType', type: 'string' },
          { name: 'nonce', type: 'string' },
          { name: 'timestamp', type: 'uint256' },
        ],
      },
      value: {
        songId: String(req.params?.id || ''),
        listener: String(req.body?.listenerAddress || ''),
        amount: String(req.body?.amount ?? ''),
        paymentType: String(req.body?.paymentType || ''),
        nonce: n,
        timestamp,
      },
    };
  }
  if (context === 'claim') {
    return {
      types: {
        Claim: [
          { name: 'songId', type: 'string' },
          { name: 'artistAddress', type: 'address' },
          { name: 'ipfsHash', type: 'string' },
          { name: 'title', type: 'string' },
          { name: 'nonce', type: 'string' },
          { name: 'timestamp', type: 'uint256' },
        ],
      },
      value: {
        songId: String(req.body?.songId || ''),
        artistAddress: String(req.body?.artistAddress || ''),
        ipfsHash: String(req.body?.ipfsHash || ''),
        title: String(req.body?.title || ''),
        nonce: n,
        timestamp,
      },
    };
  }
  return null;
}
