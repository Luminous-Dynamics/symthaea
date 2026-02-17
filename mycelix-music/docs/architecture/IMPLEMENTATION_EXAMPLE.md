# Mycelix Music: Modular Economics Implementation Example

## Complete Working Example: From Artist Upload to Listener Payment

This document shows the complete flow of how the modular economic system works in practice.

## Scenario: DJ Nova Uploads a Track

**Artist:** DJ Nova (electronic music producer)
**Song:** "Cosmic Dreams"
**Choice:** Gift Economy model (free listening + optional tips)
**Goal:** Build community first, monetize later

---

## Step 1: Artist Uploads Song & Chooses Strategy

```typescript
// apps/web/src/pages/upload/index.tsx

import { EconomicStrategyWizard } from '@/components/EconomicStrategyWizard';
import { uploadToIPFS } from '@/lib/ipfs';
import { publishToDKG } from '@/lib/ceramic';

export default function UploadPage() {
  const [songFile, setSongFile] = useState<File>();
  const [metadata, setMetadata] = useState({ title: '', artist: '' });
  const [showEconomicWizard, setShowEconomicWizard] = useState(false);
  const [songId, setSongId] = useState<string>();

  async function handleUpload() {
    // 1. Upload audio file to IPFS
    const audioCID = await uploadToIPFS(songFile);

    // 2. Create claim on DKG (Ceramic)
    const claimId = await publishToDKG({
      title: metadata.title,
      artist: metadata.artist,
      audioCID,
      timestamp: Date.now(),
    });

    setSongId(claimId);
    setShowEconomicWizard(true);
  }

  return (
    <div>
      {!showEconomicWizard ? (
        <UploadForm onComplete={handleUpload} />
      ) : (
        <EconomicStrategyWizard
          songId={songId!}
          onComplete={() => router.push('/dashboard')}
        />
      )}
    </div>
  );
}
```

**Artist's Choices in Wizard:**
1. **Preset:** Community Collective (gift economy)
2. **Payment Model:** Free listening, optional tips
3. **Revenue Split:**
   - DJ Nova: 80%
   - Featured vocalist: 15%
   - Protocol: 5%
4. **Incentives:**
   - ✅ Early listener bonus (first 100 get 10 CGC)
   - ✅ Repeat listener bonus (1.5x CGC for loyal fans)

**Result:** Smart contract transaction creates:
```solidity
// On-chain storage after deployment:
songStrategy[keccak256("cosmic-dreams")] = GiftEconomyStrategyAddress;

giftConfigs[keccak256("cosmic-dreams")] = {
  acceptsGifts: true,
  minGiftAmount: 0,
  defaultRecipients: [
    "0x...DJNova",
    "0x...Vocalist"
  ],
  defaultSplits: [8000, 1500, 500]  // Basis points
}
```

---

## Step 2: Listener Discovers & Plays Song

```typescript
// apps/web/src/components/Player.tsx

import { EconomicStrategySDK } from '@mycelix/sdk';
import { useWallet } from '@/hooks/useWallet';

export function MusicPlayer({ song }: { song: Song }) {
  const { signer } = useWallet();
  const [isPlaying, setIsPlaying] = useState(false);
  const [cgcEarned, setCgcEarned] = useState(0);

  async function handlePlay() {
    setIsPlaying(true);

    // Stream audio from IPFS
    const audioUrl = `https://w3s.link/ipfs/${song.audioCID}`;
    audioRef.current.src = audioUrl;
    audioRef.current.play();

    // THIS IS FREE - Gift economy model
    // But listener earns CGC for listening!

    if (signer) {
      const sdk = new EconomicStrategySDK(
        signer.provider!,
        process.env.NEXT_PUBLIC_ROUTER_ADDRESS!,
        signer
      );

      // Record play (triggers CGC reward)
      await sdk.streamSong(song.id, '0');  // Zero FLOW (free!)

      // Check CGC balance
      const cgcBalance = await sdk.getListenerCGCBalance(
        song.artistAddress,
        await signer.getAddress()
      );

      setCgcEarned(Number(cgcBalance));
    }
  }

  return (
    <div className="music-player">
      <audio ref={audioRef} />

      <button onClick={handlePlay}>
        {isPlaying ? '⏸' : '▶️'} Play
      </button>

      {cgcEarned > 0 && (
        <div className="cgc-reward animate-bounce">
          +{cgcEarned} CGC earned! 🎁
        </div>
      )}

      <TipButton song={song} />
    </div>
  );
}
```

**What Happens On-Chain:**

```solidity
// GiftEconomyStrategy.processPayment() is called:

function processPayment(
    bytes32 songId,
    address listener,
    uint256 amount,  // 0 for free listening
    PaymentType paymentType
) external {
    // Listener pays nothing!
    require(amount == 0, "Gift economy is free");

    // But listener RECEIVES CGC
    uint256 reward = cgcPerListen;  // 1 CGC base

    // Early listener bonus
    if (listenerCount[songId] < 100) {
        reward += 10 ether;  // 10 CGC bonus!
        listenerCount[songId]++;
    }

    // Repeat listener bonus
    if (listenerProfile[listener].totalStreams > 5) {
        reward = (reward * 150) / 100;  // 1.5x multiplier
    }

    // Credit CGC to listener (claimable)
    listenerProfiles[artistAddress][listener].cgcBalance += reward;

    emit CGCDistributed(listener, reward);
}
```

**Result:** Listener just got PAID to listen to music! 🤯

---

## Step 3: Listener Tips Artist (Optional)

```typescript
// apps/web/src/components/TipButton.tsx

export function TipButton({ song }: { song: Song }) {
  const { signer } = useWallet();
  const [tipAmount, setTipAmount] = useState('1');
  const [showModal, setShowModal] = useState(false);

  async function sendTip() {
    if (!signer) return;

    const sdk = new EconomicStrategySDK(
      signer.provider!,
      process.env.NEXT_PUBLIC_ROUTER_ADDRESS!,
      signer
    );

    // Preview where money goes
    const splits = await sdk.previewPaymentSplits(song.id, tipAmount);
    console.log('Your tip will be split:', splits);
    // [
    //   { recipient: "0x...DJNova", basisPoints: 8000, role: "artist" },  // $0.80
    //   { recipient: "0x...Vocalist", basisPoints: 1500, role: "featured" }, // $0.15
    //   { recipient: "0x...Protocol", basisPoints: 500, role: "protocol" }  // $0.05
    // ]

    // Send tip
    const receipt = await sdk.tipArtist(song.id, tipAmount);

    alert(`Tip sent! TX: ${receipt.transactionHash}`);
  }

  return (
    <>
      <button onClick={() => setShowModal(true)} className="tip-button">
        💝 Tip Artist
      </button>

      {showModal && (
        <Modal>
          <h3>Support DJ Nova</h3>
          <p>This artist chose gift economy. Listening is free, but you can tip!</p>

          <input
            type="number"
            value={tipAmount}
            onChange={(e) => setTipAmount(e.target.value)}
            placeholder="Amount in FLOW"
          />

          <button onClick={sendTip}>
            Send {tipAmount} FLOW
          </button>

          <p className="text-sm text-gray-500">
            80% goes to DJ Nova, 15% to vocalist, 5% to protocol
          </p>
        </Modal>
      )}
    </>
  );
}
```

**On-Chain Result:**

```solidity
// GiftEconomyStrategy.processPayment() with PaymentType.TIP:

function _distributeGift(
    bytes32 songId,
    uint256 amount,  // 1 FLOW = 1e18 wei
    GiftConfig memory config
) internal {
    // Split according to artist's configuration
    for (uint i = 0; i < config.defaultRecipients.length; i++) {
        uint256 recipientAmount = (amount * config.defaultSplits[i]) / 10000;

        flowToken.transfer(config.defaultRecipients[i], recipientAmount);
    }

    // Instant payment! No waiting 90 days like Spotify
}
```

**Result:** DJ Nova receives 0.8 FLOW in her wallet **instantly**. No middleman, no delay.

---

## Step 4: Compare to Another Artist's Strategy

Let's say **Indie Rock Band "The Echoes"** uploads a song on the SAME platform but chooses a **different economic model**:

```typescript
// The Echoes choose: Pay Per Stream Strategy

const echoesConfig = {
  strategyId: 'pay-per-stream-v1',
  paymentModel: PaymentModel.PAY_PER_STREAM,
  distributionSplits: [
    { recipient: '0x...guitarist', basisPoints: 3000, role: 'guitarist' },
    { recipient: '0x...drummer', basisPoints: 3000, role: 'drummer' },
    { recipient: '0x...vocalist', basisPoints: 3000, role: 'vocalist' },
    { recipient: '0x...producer', basisPoints: 500, role: 'producer' },
    { recipient: '0x...protocol', basisPoints: 500, role: 'protocol' },
  ],
  minimumPayment: 0.01,  // $0.01 per stream (10x Spotify!)
};
```

**When listener plays The Echoes' song:**

```typescript
// Different strategy contract is called!

const sdk = new EconomicStrategySDK(...);

// This costs 0.01 FLOW (not free like DJ Nova's song)
await sdk.streamSong('the-echoes-song-id', '0.01');
```

**On-Chain:**

```solidity
// PayPerStreamStrategy.processPayment() is called:

function processPayment(
    bytes32 songId,
    address listener,
    uint256 amount,  // 0.01 FLOW
    PaymentType paymentType
) external {
    require(amount >= MIN_PAYMENT, "Must pay 0.01 FLOW");

    // Split instantly based on configured splits
    RoyaltySplit memory split = royaltySplits[songId];

    for (uint i = 0; i < split.recipients.length; i++) {
        uint256 recipientAmount = (amount * split.basisPoints[i]) / 10000;

        flowToken.transfer(split.recipients[i], recipientAmount);
        // Guitarist gets 0.003 FLOW instantly
        // Drummer gets 0.003 FLOW instantly
        // Vocalist gets 0.003 FLOW instantly
        // Producer gets 0.0005 FLOW instantly
        // Protocol gets 0.0005 FLOW
    }
}
```

**Result:** Different artists, different economics, SAME platform. This is the power of modularity.

---

## Step 5: Artist Changes Strategy Mid-Career

**6 months later:** DJ Nova has built a huge fanbase (10K listeners). She decides to switch to paid streaming:

```typescript
// apps/web/src/pages/dashboard/economics.tsx

export function ManageEconomics({ artistSongs }: { artistSongs: Song[] }) {
  const { signer } = useWallet();

  async function changeStrategy(songId: string, newStrategyId: string) {
    const sdk = new EconomicStrategySDK(
      signer.provider!,
      process.env.NEXT_PUBLIC_ROUTER_ADDRESS!,
      signer
    );

    // Artist can change strategy at any time
    await sdk.changeSongStrategy(songId, newStrategyId);

    alert('Strategy updated! New listeners will use the new model.');
  }

  return (
    <div>
      {artistSongs.map(song => (
        <div key={song.id}>
          <h3>{song.title}</h3>
          <p>Current: {song.currentStrategy}</p>

          <select onChange={(e) => changeStrategy(song.id, e.target.value)}>
            <option value="gift-economy-v1">Gift Economy</option>
            <option value="pay-per-stream-v1">Pay Per Stream</option>
            <option value="patronage-v1">Patronage</option>
          </select>
        </div>
      ))}
    </div>
  );
}
```

**On-Chain:**

```solidity
// EconomicStrategyRouter.changeSongStrategy()

function changeSongStrategy(
    bytes32 songId,
    bytes32 strategyId
) external {
    require(songArtist[songId] == msg.sender, "Not song owner");

    // Update strategy pointer
    songStrategy[songId] = registeredStrategies[strategyId];

    emit StrategySet(songId, strategyId);
}
```

**Result:** Future streams use the new strategy. Past payments were under old strategy (immutable). Perfect migration!

---

## Step 6: Analytics Dashboard

Artists can see exactly how their economic model is performing:

```typescript
// apps/web/src/components/EconomicsDashboard.tsx

export function EconomicsDashboard({ artistAddress }: { artistAddress: string }) {
  const { provider } = useWallet();
  const [stats, setStats] = useState<any>();

  useEffect(() => {
    loadStats();
  }, []);

  async function loadStats() {
    const sdk = new EconomicStrategySDK(
      provider,
      process.env.NEXT_PUBLIC_ROUTER_ADDRESS!
    );

    // Get payment history for all songs
    const songs = await getArtistSongs(artistAddress);

    const allPayments = await Promise.all(
      songs.map(song => sdk.getPaymentHistory(song.id))
    );

    const totalEarnings = allPayments
      .flat()
      .reduce((sum, p) => sum + parseFloat(p.amount), 0);

    const avgPerStream = totalEarnings / allPayments.length;

    setStats({
      totalEarnings,
      totalStreams: allPayments.length,
      avgPerStream,
      topSong: songs[0],
    });
  }

  return (
    <div className="dashboard">
      <MetricCard
        title="Total Earnings"
        value={`${stats?.totalEarnings} FLOW`}
        subtitle="All-time across all songs"
      />

      <MetricCard
        title="Avg Per Stream"
        value={`${stats?.avgPerStream} FLOW`}
        comparison="Spotify: $0.003"
        highlight={stats?.avgPerStream > 0.003}
      />

      <MetricCard
        title="Total Streams"
        value={stats?.totalStreams}
        subtitle="Across all economic models"
      />

      <h3>Strategy Performance Comparison</h3>
      <Table>
        <thead>
          <tr>
            <th>Song</th>
            <th>Strategy</th>
            <th>Streams</th>
            <th>Earnings</th>
            <th>Avg/Stream</th>
          </tr>
        </thead>
        <tbody>
          {songs.map(song => (
            <tr>
              <td>{song.title}</td>
              <td>{song.strategyName}</td>
              <td>{song.streams}</td>
              <td>{song.earnings} FLOW</td>
              <td>{(song.earnings / song.streams).toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </Table>
    </div>
  );
}
```

**Result:** Artist can see which economic model works best for each song and optimize accordingly!

---

## Summary: Why This Architecture Wins

### For Artists:
- ✅ **Choice:** Pick the economic model that fits YOUR music
- ✅ **Experimentation:** Try different models, see what works
- ✅ **Evolution:** Change strategy as your career grows
- ✅ **Transparency:** See exactly where money goes
- ✅ **Instant Payments:** No 90-day wait like Spotify

### For Listeners:
- ✅ **Variety:** Some music is free, some is paid, all on one platform
- ✅ **Rewards:** Earn CGC for listening to gift economy artists
- ✅ **Clarity:** Always know how your payment splits
- ✅ **Support:** Directly support artists you love

### For The Platform:
- ✅ **Innovation:** New economic models can be added without changing core
- ✅ **Community:** Different genres can have different norms
- ✅ **Sustainability:** Protocol fee on all transactions funds development
- ✅ **Decentralization:** No central company decides the rules

---

## Next Steps

1. **Deploy Contracts:** Deploy router + 2 strategies to Gnosis Chiado testnet
2. **Test Flow:** Complete end-to-end test with real transactions
3. **Launch Beta:** Onboard first 50 artists from target community
4. **Gather Data:** See which economic models perform best
5. **Iterate:** Add new strategies based on artist feedback

The modular architecture makes all of this possible. Every artist becomes their own economic experiment. 🚀
