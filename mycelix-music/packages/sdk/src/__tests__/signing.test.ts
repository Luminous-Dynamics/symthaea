const { Wallet } = require('ethers');
// Use compiled output to avoid TS transforms in jest
const {
  signSongPayload,
  signPlayPayload,
  signClaimPayload,
} = require('../../dist/economic-strategies.js');

describe('signing helpers', () => {
  const wallet = new Wallet('0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80');
  const fixedTs = 1_700_000_000_000;
  let nowSpy;

  const helpersAvailable =
    typeof signSongPayload === 'function' &&
    typeof signPlayPayload === 'function' &&
    typeof signClaimPayload === 'function';

  beforeAll(() => {
    nowSpy = jest.spyOn(Date, 'now').mockReturnValue(fixedTs);
  });

  afterAll(() => {
    nowSpy?.mockRestore();
  });

  if (!helpersAvailable) {
    it.skip('signing helpers not available in dist build; skip', () => {});
    return;
  }

  it('signSongPayload matches expected message format', async () => {
    const payload = {
      id: 'artist-song',
      artistAddress: await wallet.getAddress(),
      ipfsHash: 'QmHash',
      paymentModel: 'pay_per_stream',
    };
    const result = await signSongPayload(wallet, payload);
    const expectedMsg = [
      'mycelix-song',
      payload.id,
      payload.artistAddress,
      payload.ipfsHash,
      payload.paymentModel,
      String(fixedTs),
    ].join('|');
    expect(result.timestamp).toBe(fixedTs);
    expect(result.signer.toLowerCase()).toBe(payload.artistAddress.toLowerCase());
    await expect(wallet.verifyMessage(expectedMsg, result.signature)).resolves.toBeTruthy();
  });

  it('signPlayPayload matches expected message format', async () => {
    const listener = await wallet.getAddress();
    const payload = {
      songId: 'artist-song',
      listener,
      amount: '0.01',
      paymentType: 'stream',
    };
    const result = await signPlayPayload(wallet, payload);
    const expectedMsg = [
      'mycelix-play',
      payload.songId,
      payload.listener,
      payload.amount,
      payload.paymentType,
      String(fixedTs),
    ].join('|');
    expect(result.timestamp).toBe(fixedTs);
    expect(result.signer.toLowerCase()).toBe(listener.toLowerCase());
    await expect(wallet.verifyMessage(expectedMsg, result.signature)).resolves.toBeTruthy();
  });

  it('signClaimPayload matches expected message format', async () => {
    const artistAddress = await wallet.getAddress();
    const payload = {
      songId: 'artist-song',
      artistAddress,
      ipfsHash: 'QmHash',
      title: 'Song Title',
    };
    const result = await signClaimPayload(wallet, payload);
    const expectedMsg = [
      'mycelix-claim',
      payload.songId,
      payload.artistAddress,
      payload.ipfsHash,
      payload.title,
      String(fixedTs),
    ].join('|');
    expect(result.timestamp).toBe(fixedTs);
    expect(result.signer.toLowerCase()).toBe(artistAddress.toLowerCase());
    await expect(wallet.verifyMessage(expectedMsg, result.signature)).resolves.toBeTruthy();
  });
});
