import request from 'supertest';
import { ethers } from 'ethers';

const EIP712_VERIFIER = process.env.EIP712_VERIFIER || process.env.ROUTER_ADDRESS || process.env.NEXT_PUBLIC_ROUTER_ADDRESS || '';
const HAS_DB = Boolean(process.env.DATABASE_URL);
const MANUAL_ENABLED = String(process.env.ENABLE_MANUAL_PLAY || 'false').toLowerCase() === 'true';
const SHOULD_RUN = Boolean(EIP712_VERIFIER && HAS_DB && MANUAL_ENABLED && process.env.NODE_ENV === 'test');

(SHOULD_RUN ? describe : describe.skip)('EIP-712 play signatures', () => {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const { app, pool, redis } = require('../index');
  const wallet = new ethers.Wallet('0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80');
  const signerAddr = wallet.address;

  afterAll(async () => {
    try { await pool.end(); } catch {}
    try { await redis.disconnect(); } catch {}
  });

  it('accepts typed signature for play logging', async () => {
    const domain = {
      name: 'MycelixMusic',
      version: '1',
      chainId: Number(process.env.EIP712_CHAIN_ID || 31337),
      verifyingContract: EIP712_VERIFIER,
    };
    const songTypes = {
      Song: [
        { name: 'id', type: 'string' },
        { name: 'artistAddress', type: 'address' },
        { name: 'ipfsHash', type: 'string' },
        { name: 'paymentModel', type: 'string' },
        { name: 'nonce', type: 'string' },
        { name: 'timestamp', type: 'uint256' },
      ],
    };
    const playTypes = {
      Play: [
        { name: 'songId', type: 'string' },
        { name: 'listener', type: 'address' },
        { name: 'amount', type: 'string' },
        { name: 'paymentType', type: 'string' },
        { name: 'nonce', type: 'string' },
        { name: 'timestamp', type: 'uint256' },
      ],
    };

    // Create song first
    const songTs = Date.now();
    const songNonce = `nonce-song-${songTs}`;
    const songBody = {
      id: `song-play-${songTs}`,
      title: 'Typed Play Song',
      artist: 'Tester',
      artistAddress: signerAddr,
      genre: 'Rock',
      ipfsHash: 'QmPlay',
      paymentModel: 'pay_per_stream',
      nonce: songNonce,
      timestamp: songTs,
    };
    const songSig = await wallet.signTypedData(domain as any, songTypes as any, {
      id: songBody.id,
      artistAddress: songBody.artistAddress,
      ipfsHash: songBody.ipfsHash,
      paymentModel: songBody.paymentModel,
      nonce: songNonce,
      timestamp: songTs,
    });
    await request(app)
      .post('/api/songs')
      .send({
        ...songBody,
        signer: signerAddr,
        signature: songSig,
        method: 'eip712',
      })
      .expect(201);

    // Log play with typed signature
    const playTs = Date.now();
    const playNonce = `nonce-play-${playTs}`;
    const playBody = {
      listenerAddress: signerAddr,
      amount: '0.01',
      paymentType: 'stream',
      nonce: playNonce,
      timestamp: playTs,
    };
    const playSig = await wallet.signTypedData(domain as any, playTypes as any, {
      songId: songBody.id,
      listener: signerAddr,
      amount: playBody.amount,
      paymentType: playBody.paymentType,
      nonce: playNonce,
      timestamp: playTs,
    });

    const res = await request(app)
      .post(`/api/songs/${encodeURIComponent(songBody.id)}/play`)
      .send({
        ...playBody,
        signer: signerAddr,
        signature: playSig,
        method: 'eip712',
      })
      .expect(200);

    expect(res.body.success).toBe(true);
  });
});
