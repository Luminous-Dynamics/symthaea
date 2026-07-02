# Mycelix Mobile

React Native mobile app for interacting with Mycelix smart contracts.

## Features

- View network statistics (total DIDs, block height)
- Look up DIDs by identifier
- View contract addresses
- Connect to Sepolia testnet

## Setup

### Prerequisites

- Node.js 18+
- React Native CLI
- Xcode (for iOS)
- Android Studio (for Android)

### Installation

```bash
# Install dependencies
npm install

# Install iOS pods (macOS only)
cd ios && pod install && cd ..
```

### Running

```bash
# Start Metro bundler
npm start

# Run on iOS
npm run ios

# Run on Android
npm run android
```

## Configuration

Contract addresses are configured in `src/App.tsx`:

```typescript
const CONTRACTS = {
  registry: '0x556b810371e3d8D9E5753117514F03cC6C93b835',
  reputation: '0xf3B343888a9b82274cEfaa15921252DB6c5f48C9',
  payment: '0x94417A3645824CeBDC0872c358cf810e4Ce4D1cB',
};
```

## Architecture

```
mobile/
├── src/
│   └── App.tsx          # Main app component
├── android/             # Android native code
├── ios/                 # iOS native code
├── package.json
└── README.md
```

## Future Features

- [ ] Wallet connection (WalletConnect)
- [ ] DID registration
- [ ] QR code scanning for DIDs
- [ ] Push notifications for events
- [ ] Biometric authentication

## License

MIT
