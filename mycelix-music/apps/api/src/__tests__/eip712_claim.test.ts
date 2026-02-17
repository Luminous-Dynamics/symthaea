import request from 'supertest';
import { ethers } from 'ethers';

const EIP712_VERIFIER = process.env.EIP712_VERIFIER || process.env.ROUTER_ADDRESS || process.env.NEXT_PUBLIC_ROUTER_ADDRESS || '';
const HAS_DB = Boolean(process.env.DATABASE_URL);
const SHOULD_RUN = Boolean(EIP712_VERIFIER && HAS_DB);

(SHOULD_RUN ? describe : describe.skip)('EIP-712 claim signatures', () => {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const { app, pool, redis } = require('../index');
  const wallet = new ethers.Wallet('0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80');
  const signerAddr = wallet.address;

  afterAll(async () => {
    try { await pool.end(); } catch {}
    try { await redis.disconnect(); } catch {}
  });

  it('accepts typed signature for claim creation', async () => {
    const domain = {
      name: 'MycelixMusic',
      version: '1',
      chainId: Number(process.env.EIP712_CHAIN_ID || 31337),
      verifyingContract: EIP712_VERIFIER,
    };
    const types = {
      Claim: [
        { name: 'songId', type: 'string' },
        { name: 'artistAddress', type: 'address' },
        { name: 'ipfsHash', type: 'string' },
        { name: 'title', type: 'string' },
        { name: 'nonce', type: 'string' },
        { name: 'timestamp', type: 'uint256' },
      ],
    };
    const timestamp = Date.now();
    const nonce = `nonce-claim-${timestamp}`;
    const body = {
      songId: `song-claim-${timestamp}`,
      title: 'Typed Claim',
      artist: 'Tester',
      ipfsHash: 'QmClaim',
      artistAddress: signerAddr,
      nonce,
      timestamp,
    };
    const signature = await wallet.signTypedData(domain as any, types as any, {
      songId: body.songId,
      artistAddress: body.artistAddress,
      ipfsHash: body.ipfsHash,
      title: body.title,
      nonce,
      timestamp,
    });

    const res = await request(app)
      .post('/api/create-dkg-claim')
      .send({
        ...body,
        signer: signerAddr,
        signature,
        method: 'eip712',
      })
      .expect(200);

    expect(res.body.streamId).toBeDefined();
  });
});
