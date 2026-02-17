# Mycelix Music SDK

TypeScript helpers for interacting with the Mycelix Music smart contracts and for signing payloads expected by the backend API.

## Install

```bash
npm install @mycelix/sdk ethers
```

## Usage

### Register a song

```ts
import { EconomicStrategySDK, PaymentModel, signSongPayload } from '@mycelix/sdk';
import { ethers } from 'ethers';

const provider = new ethers.BrowserProvider((window as any).ethereum);
const signer = await provider.getSigner();

const sdk = new EconomicStrategySDK(provider, process.env.NEXT_PUBLIC_ROUTER_ADDRESS!, signer);

const config = {
  strategyId: 'pay-per-stream-v1',
  paymentModel: PaymentModel.PAY_PER_STREAM,
  distributionSplits: [
    { recipient: await signer.getAddress(), basisPoints: 10000, role: 'artist' },
  ],
};

await sdk.registerSong('artist-title', config);

// Prepare signature for API indexing
const songSig = await signSongPayload(signer, {
  id: 'artist-title',
  artistAddress: await signer.getAddress(),
  ipfsHash: '<ipfs hash>',
  paymentModel: config.paymentModel,
});
await fetch('/api/songs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...songSig, /* song fields */ }) });
```

### Stream or tip (with play logging)

```ts
import { signPlayPayload } from '@mycelix/sdk';

await sdk.streamSong('artist-title', '0.01');
const playSig = await signPlayPayload(signer, {
  songId: 'artist-title',
  listener: await signer.getAddress(),
  amount: '0.01',
  paymentType: 'stream',
});
await fetch(`/api/songs/artist-title/play`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ ...playSig, listenerAddress: playSig.signer, amount: 0.01, paymentType: 'stream' }),
});
```

### Create a claim

```ts
import { signClaimPayload } from '@mycelix/sdk';

const claimSig = await signClaimPayload(signer, {
  songId: 'artist-title',
  artistAddress: await signer.getAddress(),
  ipfsHash: '<ipfs hash>',
  title: 'Song Title',
});
await fetch('/api/create-dkg-claim', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ ...claimSig, songId: 'artist-title', artist: 'Artist', title: 'Song Title', ipfsHash: '<ipfs hash>' }),
});
```

### Message formats

The helpers sign the exact payloads enforced by the API:
- Song: `mycelix-song|id|artistAddress|ipfsHash|paymentModel|timestamp`
- Play: `mycelix-play|songId|listener|amount|paymentType|timestamp`
- Claim: `mycelix-claim|songId|artistAddress|ipfsHash|title|timestamp`

Include `signer`, `timestamp`, and `signature` from the helper output alongside your request body. `SIGNATURE_TTL_MS` on the API controls expiry (default 5 minutes).

### Testing the helpers

Run the SDK tests to lock message formats and signatures:

```bash
npm run test --workspace=@mycelix/sdk
```
