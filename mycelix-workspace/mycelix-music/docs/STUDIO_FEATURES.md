# Mycelix Studio Features

Professional audio production and broadcasting tools powered by AI and WebAssembly.

## Table of Contents

1. [DJ Mixer](#dj-mixer)
2. [AI Stem Separation](#ai-stem-separation)
3. [Audio Effects](#audio-effects)
4. [Live Broadcasting](#live-broadcasting)
5. [Voice Commands](#voice-commands)
6. [Audio Pipeline](#audio-pipeline)

---

## DJ Mixer

Professional dual-deck mixing interface with real-time audio processing.

### Features

- **Dual Decks**: Load and mix two tracks simultaneously
- **Crossfader**: Multiple curve types (linear, constant power, smooth, scratch)
- **BPM Detection**: Automatic tempo analysis
- **Beat Sync**: One-click beatmatching between decks
- **3-Band EQ**: Low, mid, high frequency control with kill switches
- **Hot Cues**: 8 cue points per deck for instant jumps
- **Looping**: 1/4, 1/2, 1, 2, 4, 8 bar loops
- **Waveform Display**: Visual representation of audio

### Usage

```tsx
import { DJMixer } from '@/components/dj';
import { useDJMixer } from '@/hooks';

function MyDJApp() {
  return <DJMixer />;
}
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Play/Pause active deck |
| Q/W | Cue Deck A/B |
| A/S | Play Deck A/B |
| Z/X | Sync Deck A/B |

---

## AI Stem Separation

Isolate vocals, drums, bass, and instruments from any audio file using AI-powered spectral analysis.

### Supported Stems

- **Vocals**: Voice and vocal harmonies
- **Drums**: Kick, snare, hi-hats, cymbals
- **Bass**: Bass guitar, synth bass
- **Other**: Remaining instruments
- **Piano**: Piano and keys (extended mode)
- **Guitar**: Electric and acoustic guitar (extended mode)
- **Synth**: Synthesizers (extended mode)

### Features

- Real-time playback with individual stem control
- Level faders and pan controls
- Mute/Solo for each stem
- WAV export for individual stems or full mix
- Drag-and-drop file upload

### Usage

```tsx
import { StemMixer } from '@/components/stems';
import { useStemSeparation } from '@/hooks';

function MyStemApp() {
  const { separateAudio, stems, controls } = useStemSeparation();

  const handleFile = async (file: File) => {
    const ctx = new AudioContext();
    const buffer = await ctx.decodeAudioData(await file.arrayBuffer());
    await separateAudio(buffer, ['vocals', 'drums', 'bass', 'other']);
  };

  return <StemMixer onExport={(blob, name) => console.log(name)} />;
}
```

---

## Audio Effects

Real-time audio processing with professional-grade effects.

### Available Effects

#### Equalizer
- 5-band parametric EQ
- Frequency, gain, and Q controls per band
- Visual frequency response curve

#### Reverb
- Room size control
- Damping adjustment
- Wet/dry mix
- Pre-delay

#### Spatial Audio
- 3D positioning (azimuth, elevation, distance)
- HRTF-based binaural processing
- Room simulation

#### Dynamics
- Compressor with threshold, ratio, attack, release
- Limiter for peak control
- Makeup gain

### Usage

```tsx
import { EffectsPanel } from '@/components/effects';
import { useAudioEffects } from '@/hooks';

function MyEffectsApp() {
  const { eq, reverb, spatial, compressor } = useAudioEffects();

  return <EffectsPanel />;
}
```

---

## Live Broadcasting

Stream your DJ sets to listeners worldwide with real-time chat and reactions.

### Features

- WebRTC-based peer-to-peer streaming
- Real-time listener count and stats
- Chat with emoji reactions
- Broadcast recording option
- Share links for easy access
- Listener management (kick/ban)

### Broadcaster Usage

```tsx
import { BroadcastStudio } from '@/components/broadcast';
import { useBroadcast } from '@/hooks';

function MyBroadcastApp() {
  const { startBroadcast, stopBroadcast, isLive, stats } = useBroadcast();

  return <BroadcastStudio />;
}
```

### Listener Usage

```tsx
import { useBroadcastListener } from '@/hooks';

function MyListenerApp({ streamId }: { streamId: string }) {
  const { connect, disconnect, isPlaying, messages } = useBroadcastListener(streamId);

  return (
    <button onClick={connect}>
      {isPlaying ? 'Listening...' : 'Join Stream'}
    </button>
  );
}
```

---

## Voice Commands

Hands-free control using the Web Speech API.

### Supported Commands

#### Playback
- "play", "pause", "stop"
- "next", "previous"
- "shuffle", "repeat"

#### Volume
- "volume up", "volume down"
- "mute", "unmute"
- "volume [0-100]"

#### Navigation
- "go home", "library", "explore"
- "show queue", "now playing"

#### DJ
- "crossfade", "loop", "sync"
- "cue point", "drop"

#### Search
- "search for [query]"
- "play [song/artist]"

### Wake Word

The default wake word is "hey mycelix". Commands are only processed after hearing the wake word (configurable).

### Usage

```tsx
import { VoiceControl } from '@/components/voice';
import { useVoiceCommands } from '@/hooks';

function MyVoiceApp() {
  const {
    isListening,
    startListening,
    registerHandler
  } = useVoiceCommands();

  // Register custom handler
  registerHandler('play', (params) => {
    console.log('Play command received:', params);
  });

  return <VoiceControl />;
}
```

---

## Audio Pipeline

Unified audio routing system connecting all features.

### Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Sources   │────▶│   Effects   │────▶│   Output    │
│  (Player,   │     │  (EQ, Rev,  │     │  (Speakers, │
│   DJ, etc)  │     │   Spatial)  │     │  Broadcast) │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Features

- Central AudioContext management
- Node registration and routing
- Master volume and mute controls
- Audio level monitoring
- Broadcast stream output

### Usage

```tsx
import { AudioPipelineProvider, useAudioPipeline } from '@/components/audio';

// Wrap your app
function App() {
  return (
    <AudioPipelineProvider>
      <MyAudioApp />
    </AudioPipelineProvider>
  );
}

// Use in components
function MyAudioApp() {
  const {
    registerNode,
    connect,
    setMasterVolume,
    getBroadcastStream
  } = useAudioPipeline();

  // Route DJ output to broadcast
  connect('dj-mixer', 'master');
  const stream = getBroadcastStream();
}
```

---

## Best Practices

### Performance

1. **Lazy load audio features** - Only initialize AudioContext on user interaction
2. **Use Web Workers** - Offload heavy processing to workers
3. **Optimize buffer sizes** - Balance latency vs. CPU usage

### Browser Support

| Feature | Chrome | Firefox | Safari | Edge |
|---------|--------|---------|--------|------|
| DJ Mixer | ✅ | ✅ | ✅ | ✅ |
| Stem Separation | ✅ | ✅ | ⚠️ | ✅ |
| Voice Commands | ✅ | ❌ | ❌ | ✅ |
| Broadcasting | ✅ | ✅ | ⚠️ | ✅ |

⚠️ = Partial support

### Accessibility

- All controls are keyboard navigable
- ARIA labels on interactive elements
- Screen reader announcements for state changes
- Reduced motion support for animations

---

## API Reference

See individual hook documentation:

- [`useDJMixer`](./api/useDJMixer.md)
- [`useStemSeparation`](./api/useStemSeparation.md)
- [`useAudioEffects`](./api/useAudioEffects.md)
- [`useBroadcast`](./api/useBroadcast.md)
- [`useVoiceCommands`](./api/useVoiceCommands.md)
- [`useAudioPipeline`](./api/useAudioPipeline.md)
