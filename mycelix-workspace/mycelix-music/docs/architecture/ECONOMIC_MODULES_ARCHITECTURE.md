# Mycelix Music: Modular Economic Architecture v1.0

## Vision: Economic Operating Systems for Music

Each artist, album, or DAO can compose their own economic model from pluggable primitives.

## Core Principle: Composable Economic Strategies

```typescript
interface EconomicStrategy {
  id: string;
  name: string;
  description: string;

  // What value flows does this support?
  supportedTokens: ('FLOW' | 'CGC' | 'TEND' | 'USD')[];

  // How does money flow?
  paymentModel: PaymentModel;
  distributionModel: DistributionModel;

  // What behaviors does it incentivize?
  incentives: IncentiveModel;
}
```

## The Economic Primitive Library

### 1. Payment Models (How Listeners Pay)

```typescript
enum PaymentModel {
  // Traditional Models
  PAY_PER_STREAM = "pay_per_stream",           // $0.01 per play
  SUBSCRIPTION = "subscription",                 // $10/month unlimited
  PAY_PER_DOWNLOAD = "pay_per_download",        // $1 per song

  // Web3 Models
  NFT_GATED = "nft_gated",                      // Own NFT to access
  STAKING_GATED = "staking_gated",              // Stake FLOW to unlock
  TOKEN_TIP = "token_tip",                       // Pay what you want

  // Plural Models
  GIFT_ECONOMY = "gift_economy",                // Free, but listeners gift CGC
  TIME_BARTER = "time_barter",                  // Trade TEND for access
  PATRONAGE = "patronage",                       // Monthly patron subscription

  // Hybrid Models
  FREEMIUM = "freemium",                        // Free low-quality, pay for Hi-Fi
  PAY_WHAT_YOU_WANT = "pay_what_you_want",      // Listener chooses amount
  AUCTION = "auction",                          // Dutch auction for limited releases
}
```

### 2. Distribution Models (How Artists Split Revenue)

```typescript
interface DistributionModel {
  type: 'fixed' | 'dynamic' | 'algorithmic';

  // Fixed splits (traditional)
  fixedSplits?: {
    recipients: Array<{
      did: string;
      basisPoints: number;  // 10000 = 100%
      role: 'artist' | 'producer' | 'label' | 'dao' | 'protocol';
    }>;
  };

  // Dynamic splits (changes based on conditions)
  dynamicSplits?: {
    rules: Array<{
      condition: string;  // "if streams > 10000"
      then: DistributionRule;
    }>;
  };

  // Algorithmic (e.g., streams proportional to listening time)
  algorithmicSplits?: {
    algorithm: 'user_centric' | 'attention_weighted' | 'quality_adjusted';
    parameters: Record<string, any>;
  };
}
```

### 3. Incentive Models (What Behaviors Are Rewarded)

```typescript
interface IncentiveModel {
  // Listener incentives
  listenerRewards: {
    earlyListener?: { bonus: number };        // First 1000 listeners get bonus CGC
    repeatListener?: { multiplier: number };  // Loyal fans earn more voting power
    shareBonus?: { amount: number };          // Get FLOW for successful referrals
  };

  // Creator incentives
  creatorRewards: {
    qualityBonus?: { threshold: number };     // High-rated songs earn more
    collaborationBonus?: { multiplier: number }; // Multi-artist songs boosted
    curatorBonus?: { percentage: number };    // Curators earn % of discovery plays
  };

  // Community incentives
  communityRewards: {
    treasuryContribution?: { percentage: number }; // % to DAO treasury
    publicGoodsFunding?: { percentage: number };   // % to protocol development
    localCommunity?: { percentage: number };       // % to artist's local scene
  };
}
```

## Pre-Built Economic Configurations

### Config 1: "Independent Artist" (Pure Market)

```typescript
const independentArtistEconomy: EconomicStrategy = {
  id: 'independent-artist-v1',
  name: 'Independent Artist Economy',
  description: 'Pure market model - you keep what you earn',

  supportedTokens: ['FLOW', 'USD'],

  paymentModel: {
    primary: PaymentModel.PAY_PER_STREAM,
    rates: {
      base: 0.01,  // $0.01 per stream (10x Spotify!)
      hiRes: 0.02, // $0.02 for lossless
    },
    alternatives: [
      PaymentModel.PAY_PER_DOWNLOAD,  // $1 per song
      PaymentModel.TOKEN_TIP,         // Optional tipping
    ]
  },

  distributionModel: {
    type: 'fixed',
    fixedSplits: {
      recipients: [
        { did: 'artist_did', basisPoints: 9500, role: 'artist' },     // 95%
        { did: 'protocol_did', basisPoints: 500, role: 'protocol' },  // 5%
      ]
    }
  },

  incentives: {
    listenerRewards: {
      shareBonus: { amount: 0.001 } // Earn FLOW for successful shares
    },
    creatorRewards: {
      qualityBonus: { threshold: 4.5 } // 4.5+ star ratings earn 10% bonus
    }
  }
};
```

### Config 2: "Community Collective" (Gift + Market Hybrid)

```typescript
const communityCollectiveEconomy: EconomicStrategy = {
  id: 'community-collective-v1',
  name: 'Community Collective Economy',
  description: 'Free music supported by gifts and patronage',

  supportedTokens: ['FLOW', 'CGC', 'TEND'],

  paymentModel: {
    primary: PaymentModel.GIFT_ECONOMY,  // Listening is free
    secondary: PaymentModel.PATRONAGE,    // Optional $5/month patron tier
    rates: {
      patronTier: 5.00,
      cgcPerListen: 1,  // Listeners auto-gift 1 CGC per play
    }
  },

  distributionModel: {
    type: 'algorithmic',
    algorithmicSplits: {
      algorithm: 'attention_weighted',
      parameters: {
        // Artists earn proportional to total listening time
        // More engaged listeners = more value
        weightByDuration: true,
        weightByRepeatPlays: true,
      }
    }
  },

  incentives: {
    listenerRewards: {
      earlyListener: { bonus: 10 },     // First 100 listeners get 10 CGC
      repeatListener: { multiplier: 1.5 } // Repeat plays worth more CGC
    },
    creatorRewards: {
      collaborationBonus: { multiplier: 1.2 } // Collabs earn 20% more
    },
    communityRewards: {
      treasuryContribution: { percentage: 10 },    // 10% to DAO
      localCommunity: { percentage: 5 }            // 5% to local scene fund
    }
  }
};
```

### Config 3: "Experimental Label" (Complex Multi-Tier)

```typescript
const experimentalLabelEconomy: EconomicStrategy = {
  id: 'experimental-label-v1',
  name: 'Experimental Label Economy',
  description: 'Multi-tier with dynamic splits and NFT gating',

  supportedTokens: ['FLOW', 'CGC', 'LABEL_TOKEN'],

  paymentModel: {
    primary: PaymentModel.FREEMIUM,
    tiers: {
      free: {
        quality: '128kbps MP3',
        features: ['streaming_only']
      },
      premium: {
        cost: 0.01,
        quality: 'FLAC',
        features: ['streaming', 'download', 'stems']
      },
      collector: {
        type: PaymentModel.NFT_GATED,
        nftContract: '0x...',
        quality: 'Studio Master 24bit/96kHz',
        features: ['streaming', 'download', 'stems', 'voting', 'revenue_share']
      }
    }
  },

  distributionModel: {
    type: 'dynamic',
    dynamicSplits: {
      rules: [
        {
          condition: 'streams < 10000',
          then: {
            artist: 7000,   // 70%
            label: 2500,    // 25%
            protocol: 500   // 5%
          }
        },
        {
          condition: 'streams >= 10000',
          then: {
            artist: 8500,   // Artist gets more after hitting milestone
            label: 1000,
            protocol: 500
          }
        }
      ]
    }
  },

  incentives: {
    listenerRewards: {
      earlyListener: { bonus: 100 },  // Early supporters get label tokens
      shareBonus: { amount: 0.002 }
    },
    creatorRewards: {
      qualityBonus: { threshold: 4.8 },
      curatorBonus: { percentage: 5 }  // Curators earn 5% of plays they drive
    },
    communityRewards: {
      publicGoodsFunding: { percentage: 10 }  // Reinvest in experimental music
    }
  }
};
```

## Implementation: Smart Contract Architecture

```solidity
// contracts/EconomicModules.sol
pragma solidity ^0.8.20;

/**
 * @title Economic Strategy Router
 * @notice Allows each song to have its own economic operating system
 */
contract EconomicStrategyRouter {

    // Song ID => Economic Strategy ID
    mapping(bytes32 => bytes32) public songStrategy;

    // Strategy implementations
    mapping(bytes32 => IEconomicStrategy) public strategies;

    /**
     * @notice Artist chooses economic model when uploading
     */
    function setSongStrategy(
        bytes32 songId,
        bytes32 strategyId
    ) external onlyArtist(songId) {
        require(strategies[strategyId] != address(0), "Strategy not found");
        songStrategy[songId] = strategyId;

        emit StrategySet(songId, strategyId);
    }

    /**
     * @notice Route payment through appropriate strategy
     */
    function processPayment(
        bytes32 songId,
        address listener,
        uint256 amount,
        PaymentType paymentType
    ) external {
        bytes32 strategyId = songStrategy[songId];
        IEconomicStrategy strategy = strategies[strategyId];

        // Strategy handles payment logic
        strategy.processPayment(songId, listener, amount, paymentType);
    }
}

/**
 * @title Economic Strategy Interface
 * @notice All strategies must implement this
 */
interface IEconomicStrategy {
    function processPayment(
        bytes32 songId,
        address listener,
        uint256 amount,
        PaymentType paymentType
    ) external;

    function calculateSplits(
        bytes32 songId,
        uint256 amount
    ) external view returns (Split[] memory);

    function applyIncentives(
        bytes32 songId,
        address agent,
        IncentiveType incentiveType
    ) external;
}

/**
 * @title Pay Per Stream Strategy
 */
contract PayPerStreamStrategy is IEconomicStrategy {
    function processPayment(
        bytes32 songId,
        address listener,
        uint256 amount,
        PaymentType paymentType
    ) external override {
        // Get song metadata from DKG
        SongMetadata memory song = dkg.getSong(songId);

        // Calculate splits
        Split[] memory splits = calculateSplits(songId, amount);

        // Distribute instantly
        for (uint i = 0; i < splits.length; i++) {
            flowToken.transfer(splits[i].recipient, splits[i].amount);
        }

        // Apply listener incentives
        applyIncentives(songId, listener, IncentiveType.REPEAT_LISTEN);
    }

    function calculateSplits(
        bytes32 songId,
        uint256 amount
    ) public view override returns (Split[] memory) {
        // Read royalty split from song claim
        RoyaltySplit memory split = dkg.getRoyaltySplit(songId);

        Split[] memory result = new Split[](split.recipients.length);
        for (uint i = 0; i < split.recipients.length; i++) {
            result[i] = Split({
                recipient: split.recipients[i].address,
                amount: (amount * split.recipients[i].basisPoints) / 10000
            });
        }

        return result;
    }
}
```

## Artist Configuration UI

```typescript
// apps/web/src/components/EconomicConfigWizard.tsx

export function EconomicConfigWizard({ songId }: { songId: string }) {
  const [config, setConfig] = useState<EconomicStrategy>();

  return (
    <Wizard title="Choose Your Economic Model">

      {/* Step 1: Select preset or custom */}
      <Step title="Start with a Template">
        <PresetSelector
          presets={[
            independentArtistEconomy,
            communityCollectiveEconomy,
            experimentalLabelEconomy,
          ]}
          onSelect={setConfig}
        />
        <Button variant="ghost">
          Start from scratch (Advanced)
        </Button>
      </Step>

      {/* Step 2: Customize payment model */}
      <Step title="How Should Listeners Pay?">
        <PaymentModelSelector
          value={config.paymentModel}
          onChange={(model) => setConfig({...config, paymentModel: model})}
        />

        <InfoBox>
          <strong>Pay Per Stream:</strong> Listeners pay $0.01 each time
          they play your song. Best for: artists with viral potential.

          <strong>Gift Economy:</strong> Listening is free, but listeners
          can gift you CGC tokens. Best for: community-focused artists.

          <strong>Patronage:</strong> Fans subscribe monthly to support you.
          Best for: artists with dedicated fanbase.
        </InfoBox>
      </Step>

      {/* Step 3: Configure distribution */}
      <Step title="How Should Revenue Split?">
        <DistributionEditor
          value={config.distributionModel}
          onChange={(model) => setConfig({...config, distributionModel: model})}
        />

        <VisualizationPreview>
          {/* Show pie chart of where money goes */}
          <PieChart data={simulateSplits(config, 1000)} />

          <Table>
            <caption>If your song earns $1000:</caption>
            <tbody>
              <tr><td>You</td><td>$950</td></tr>
              <tr><td>Producer</td><td>$0</td></tr>
              <tr><td>DAO Treasury</td><td>$0</td></tr>
              <tr><td>Protocol Fee</td><td>$50</td></tr>
            </tbody>
          </Table>
        </VisualizationPreview>
      </Step>

      {/* Step 4: Set incentives */}
      <Step title="Reward Your Community">
        <IncentiveBuilder
          value={config.incentives}
          onChange={(incentives) => setConfig({...config, incentives})}
        />

        <CheckboxGroup label="Listener Rewards">
          <Checkbox label="Early Listener Bonus: First 100 listeners get 10 CGC" />
          <Checkbox label="Share Bonus: Earn 0.001 FLOW for each new listener you bring" />
          <Checkbox label="Repeat Listener Bonus: Loyal fans earn 1.5x CGC" />
        </CheckboxGroup>

        <CheckboxGroup label="Creator Incentives">
          <Checkbox label="Quality Bonus: Songs rated 4.5+ stars earn 10% more" />
          <Checkbox label="Collaboration Bonus: Multi-artist songs earn 20% more" />
        </CheckboxGroup>
      </Step>

      {/* Step 5: Review & Deploy */}
      <Step title="Review Your Economy">
        <EconomicSummary config={config} />

        <Button onClick={() => deployStrategy(config)}>
          Deploy This Economic Model
        </Button>

        <Alert>
          You can change this later, but existing payments will use the
          original configuration. We recommend testing with a demo song first.
        </Alert>
      </Step>

    </Wizard>
  );
}
```

## Storage: Economic Config in Ceramic (DKG)

```typescript
// src/dkg/economic-config-schema.ts

export const SongEconomicConfigSchema = {
  $schema: 'https://json-schema.org/draft/2020-12/schema',
  title: 'SongEconomicConfig',
  type: 'object',
  properties: {
    songCID: { type: 'string' },
    artistDID: { type: 'string' },

    // Economic strategy reference
    strategyId: { type: 'string' },
    strategyVersion: { type: 'string' },

    // Full config (denormalized for performance)
    paymentModel: {
      type: 'object',
      properties: {
        primary: { type: 'string', enum: Object.values(PaymentModel) },
        rates: { type: 'object' },
        alternatives: { type: 'array' }
      }
    },

    distributionModel: {
      type: 'object',
      properties: {
        type: { type: 'string', enum: ['fixed', 'dynamic', 'algorithmic'] },
        fixedSplits: { type: 'object' },
        dynamicSplits: { type: 'object' },
        algorithmicSplits: { type: 'object' }
      }
    },

    incentives: {
      type: 'object',
      properties: {
        listenerRewards: { type: 'object' },
        creatorRewards: { type: 'object' },
        communityRewards: { type: 'object' }
      }
    },

    // Audit trail
    createdAt: { type: 'string', format: 'date-time' },
    modifiedAt: { type: 'string', format: 'date-time' },
    version: { type: 'number' }
  }
};
```

## Benefits of Modular Economics

1. **Experimentation**: Communities can discover what works
2. **Flexibility**: Different genres need different economics
3. **Evolution**: Strategies can change as artists grow
4. **Transparency**: All rules are on-chain and auditable
5. **Innovation**: New economic models can be added without protocol changes

## Examples in Practice

### Example 1: Classical Orchestra
```typescript
const orchestraEconomy: EconomicStrategy = {
  // High-quality recordings with patronage model
  paymentModel: {
    primary: PaymentModel.PATRONAGE,  // $20/month patrons
    secondary: PaymentModel.PAY_PER_DOWNLOAD  // $5 per album
  },
  distributionModel: {
    type: 'fixed',
    splits: [
      { role: 'conductor', basisPoints: 2000 },
      { role: 'soloists', basisPoints: 3000 },
      { role: 'ensemble', basisPoints: 4000 },  // Split among 50 musicians
      { role: 'venue', basisPoints: 1000 }
    ]
  }
};
```

### Example 2: Underground Hip-Hop Collective
```typescript
const hipHopCollectiveEconomy: EconomicStrategy = {
  // Free music, earn through battles and features
  paymentModel: {
    primary: PaymentModel.GIFT_ECONOMY,
    secondary: PaymentModel.TIME_BARTER  // Trade verses for beats
  },
  distributionModel: {
    type: 'algorithmic',
    algorithm: 'attention_weighted'  // Most-played artists earn most
  },
  incentives: {
    collaborationBonus: { multiplier: 2.0 },  // Heavily reward collabs
    curatorBonus: { percentage: 10 }          // DJs earn for curation
  }
};
```

### Example 3: Ambient Music for Meditation
```typescript
const meditationMusicEconomy: EconomicStrategy = {
  // Pay what you want with suggested price
  paymentModel: {
    primary: PaymentModel.PAY_WHAT_YOU_WANT,
    suggested: 0.005,  // Suggest $0.005 per stream
    minimum: 0,        // But allow free
  },
  distributionModel: {
    type: 'fixed',
    splits: [
      { role: 'artist', basisPoints: 7000 },
      { role: 'meditation_app', basisPoints: 2000 },
      { role: 'mental_health_charity', basisPoints: 1000 }
    ]
  }
};
```

## Next: Implementation Priority

This modular design means we can:

1. **Build 1-2 strategies first** (e.g., Pay-Per-Stream + Gift Economy)
2. **Test with real artists** to validate
3. **Add more strategies incrementally** based on demand
4. **Let community create new strategies** via governance

---

**Status:** Architecture designed, ready to implement
**Next Step:** Build the strategy router smart contract + 2 initial strategies
