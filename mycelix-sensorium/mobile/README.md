# Mycelix Mobile

Capacitor wrapper for Mycelix Sensorium. Wraps the Leptos WASM Sensorium shell
as a native mobile app.

## Setup

```bash
cd /srv/luminous-dynamics/mycelix-sensorium/mobile
npm install
npx cap add android   # For Android
npx cap add ios       # For iOS

# Build and sync
cd ..
trunk build --release
cd mobile
npx cap sync

# Open in IDE
npx cap open android  # Opens Android Studio
npx cap open ios      # Opens Xcode
```

## Test Device
Pixel 8 Pro (Tensor G3, arm64-v8a, SDK 34, 12GB RAM)

## Features
- Full Sensorium shell (Consciousness Orb, 4 phenotypes, WebGL background)
- Local conductor connection via WebSocket
- Haptic feedback on domain interactions (planned)
- Biometric vault unlock (planned)
