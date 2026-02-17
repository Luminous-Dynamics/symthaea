import request from 'supertest';
import { ethers } from 'ethers';
import { app, SIGNATURE_TTL_MS, pool, redis } from '../index';
import { messageForSong, messageForPlay } from '../types/auth';

// Helper to sign messages with deterministic wallet
const wallet = new ethers.Wallet('0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80');
const signerAddr = wallet.address;

function buildSignature(messageParts: string[], timestamp: number) {
  const msg = messageParts.slice(0, -1).concat(String(timestamp)).join('|');
  const sig = wallet.signMessage(msg);
  return { signaturePromise: sig, timestamp };
}

describe('auth guards', () => {
  const songId = 'test-song';
  const songBody = {
    id: songId,
    title: 'Test',
    artist: 'Artist',
    artistAddress: signerAddr,
    genre: 'Rock',
    ipfsHash: 'Qm123',
    paymentModel: 'pay_per_stream',
  };

  afterAll(async () => {
    try { await pool.end(); } catch {}
    try { await redis.disconnect(); } catch {}
  });

  it('accepts valid signature for song create', async () => {
    const now = Date.now();
    const parts = messageForSong({ ...songBody, timestamp: now });
    const { signaturePromise, timestamp } = buildSignature(parts, now);
    const signature = await signaturePromise;

    const res = await request(app)
      .post('/api/songs')
      .send({ ...songBody, signer: signerAddr, signature, timestamp })
      .expect(201);

    expect(res.body.id).toBe(songId);
  });

  it('rejects expired signature', async () => {
    const ts = Date.now() - (SIGNATURE_TTL_MS + 1000);
    const parts = messageForSong({ ...songBody, timestamp: ts });
    const { signaturePromise, timestamp } = buildSignature(parts, ts);
    const signature = await signaturePromise;

    const res = await request(app)
      .post('/api/songs')
      .send({ ...songBody, signer: signerAddr, signature, timestamp })
      .expect(400);

    expect(res.body.reason).toBe('expired');
    expect(Number(res.body.ts_diff_ms || 0)).toBeGreaterThan(0);
  });

  it('rejects invalid signature', async () => {
    const ts = Date.now();
    const res = await request(app)
      .post('/api/songs')
      .send({ ...songBody, signer: signerAddr, signature: '0xdead', timestamp: ts })
      .expect(403);

    expect(res.body.reason).toBe('invalid_signature');
  });

  it('rejects replayed signature', async () => {
    const ts = Date.now();
    const parts = messageForSong({ ...songBody, timestamp: ts });
    const { signaturePromise, timestamp } = buildSignature(parts, ts);
    const signature = await signaturePromise;
    const payload = { ...songBody, signer: signerAddr, signature, timestamp };

    await request(app).post('/api/songs').send(payload).expect(201);
    await request(app).post('/api/songs').send(payload).expect(403);
  });

  it('applies guard to play logging', async () => {
    const ts = Date.now();
    const playParts = messageForPlay({
      songId,
      listener: signerAddr,
      amount: '0.01',
      paymentType: 'stream',
      timestamp: ts,
    });
    const { signaturePromise, timestamp } = buildSignature(playParts, ts);
    const signature = await signaturePromise;

    await request(app)
      .post(`/api/songs/${songId}/play`)
      .send({
        listenerAddress: signerAddr,
        amount: 0.01,
        paymentType: 'stream',
        signer: signerAddr,
        signature,
        timestamp,
      })
      .expect(200);
  });
});
