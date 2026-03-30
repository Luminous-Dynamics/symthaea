// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * DJ Mixer Component
 *
 * Full-featured DJ interface with dual decks, crossfader, and effects.
 */

import { useCallback, useMemo } from 'react';
import { useDJMixer, type DeckState, type CrossfaderCurve } from '@/hooks/useDJMixer';

// === Types ===

interface DJMixerProps {
  onLoadTrack?: (deck: 'A' | 'B') => void;
}

// === Helpers ===

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// === Sub-Components ===

function Waveform({
  peaks,
  position,
  duration,
  loop,
  cuePoints,
  onSeek,
}: {
  peaks: number[];
  position: number;
  duration: number;
  loop: DeckState['loop'];
  cuePoints: DeckState['cuePoints'];
  onSeek: (position: number) => void;
}) {
  const progress = duration > 0 ? position / duration : 0;

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      const rect = e.currentTarget.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width;
      onSeek(x * duration);
    },
    [duration, onSeek]
  );

  return (
    <div style={styles.waveformContainer} onClick={handleClick}>
      {/* Waveform bars */}
      <div style={styles.waveformBars}>
        {peaks.length > 0 ? (
          peaks.slice(0, 200).map((peak, i) => (
            <div
              key={i}
              style={{
                ...styles.waveformBar,
                height: `${peak * 100}%`,
                backgroundColor: i / 200 < progress ? '#818cf8' : 'rgba(255, 255, 255, 0.3)',
              }}
            />
          ))
        ) : (
          <div style={styles.waveformPlaceholder}>No waveform data</div>
        )}
      </div>

      {/* Loop region */}
      {loop && (
        <div
          style={{
            ...styles.loopRegion,
            left: `${(loop.start / duration) * 100}%`,
            width: `${((loop.end - loop.start) / duration) * 100}%`,
            opacity: loop.active ? 0.3 : 0.15,
          }}
        />
      )}

      {/* Cue points */}
      {cuePoints.map((cue, i) => (
        <div
          key={cue.id}
          style={{
            ...styles.cueMarker,
            left: `${(cue.position / duration) * 100}%`,
            backgroundColor: cue.color,
          }}
          title={cue.label}
        />
      ))}

      {/* Playhead */}
      <div
        style={{
          ...styles.playhead,
          left: `${progress * 100}%`,
        }}
      />
    </div>
  );
}

function DeckDisplay({ deck, label }: { deck: DeckState; label: string }) {
  return (
    <div style={styles.deckDisplay}>
      <div style={styles.deckLabel}>{label}</div>
      <div style={styles.trackInfo}>
        <div style={styles.trackTitle}>{deck.trackTitle || 'No track loaded'}</div>
        <div style={styles.trackArtist}>{deck.trackArtist}</div>
      </div>
      <div style={styles.timeDisplay}>
        <span style={styles.currentTime}>{formatTime(deck.position)}</span>
        <span style={styles.remainingTime}>-{formatTime(deck.duration - deck.position)}</span>
      </div>
      <div style={styles.bpmDisplay}>
        <span style={styles.bpmValue}>{deck.bpm.toFixed(1)}</span>
        <span style={styles.bpmLabel}>BPM</span>
        {deck.pitch !== 0 && (
          <span style={styles.pitchValue}>
            {deck.pitch > 0 ? '+' : ''}{deck.pitch.toFixed(1)}%
          </span>
        )}
      </div>
    </div>
  );
}

function PitchSlider({
  value,
  onChange,
}: {
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <div style={styles.pitchSlider}>
      <div style={styles.pitchLabel}>PITCH</div>
      <input
        type="range"
        min={-50}
        max={50}
        step={0.1}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={styles.pitchInput}
      />
      <div style={styles.pitchValue}>{value.toFixed(1)}%</div>
    </div>
  );
}

function EQKnob({
  label,
  value,
  onChange,
  onKill,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  onKill: () => void;
}) {
  return (
    <div style={styles.eqKnob}>
      <div style={styles.eqLabel}>{label}</div>
      <input
        type="range"
        min={-24}
        max={12}
        step={0.5}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={styles.eqInput}
      />
      <button style={styles.killButton} onClick={onKill}>
        {value <= -24 ? 'ON' : 'KILL'}
      </button>
    </div>
  );
}

function LoopControls({
  loop,
  onSetLoop,
  onToggle,
  onClear,
  onHalve,
  onDouble,
}: {
  loop: DeckState['loop'];
  onSetLoop: (bars: number) => void;
  onToggle: () => void;
  onClear: () => void;
  onHalve: () => void;
  onDouble: () => void;
}) {
  const loopSizes = [0.25, 0.5, 1, 2, 4, 8];

  return (
    <div style={styles.loopControls}>
      <div style={styles.loopLabel}>LOOP</div>
      <div style={styles.loopButtons}>
        {loopSizes.map((size) => (
          <button
            key={size}
            style={styles.loopSizeButton}
            onClick={() => onSetLoop(size)}
          >
            {size < 1 ? `1/${1 / size}` : size}
          </button>
        ))}
      </div>
      <div style={styles.loopActions}>
        <button
          style={{
            ...styles.loopActionButton,
            backgroundColor: loop?.active ? '#22c55e' : undefined,
          }}
          onClick={onToggle}
          disabled={!loop}
        >
          {loop?.active ? 'EXIT' : 'IN'}
        </button>
        <button style={styles.loopActionButton} onClick={onHalve} disabled={!loop}>
          /2
        </button>
        <button style={styles.loopActionButton} onClick={onDouble} disabled={!loop}>
          x2
        </button>
        <button style={styles.loopActionButton} onClick={onClear} disabled={!loop}>
          CLR
        </button>
      </div>
    </div>
  );
}

function CueButtons({
  cuePoints,
  onSet,
  onJump,
}: {
  cuePoints: DeckState['cuePoints'];
  onSet: (index: number) => void;
  onJump: (index: number) => void;
}) {
  const colors = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#06b6d4', '#3b82f6', '#8b5cf6', '#ec4899'];

  return (
    <div style={styles.cueButtons}>
      <div style={styles.cueLabel}>HOT CUE</div>
      <div style={styles.cueGrid}>
        {[0, 1, 2, 3, 4, 5, 6, 7].map((i) => {
          const cue = cuePoints[i];
          return (
            <button
              key={i}
              style={{
                ...styles.cueButton,
                backgroundColor: cue ? cue.color : 'rgba(255, 255, 255, 0.1)',
              }}
              onClick={() => (cue ? onJump(i) : onSet(i))}
            >
              {i + 1}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function Crossfader({
  value,
  curve,
  onValueChange,
  onCurveChange,
}: {
  value: number;
  curve: CrossfaderCurve;
  onValueChange: (value: number) => void;
  onCurveChange: (curve: CrossfaderCurve) => void;
}) {
  const curves: CrossfaderCurve[] = ['linear', 'constant', 'smooth', 'scratch'];

  return (
    <div style={styles.crossfader}>
      <div style={styles.crossfaderLabel}>CROSSFADER</div>
      <div style={styles.crossfaderTrack}>
        <span style={styles.deckIndicator}>A</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={value}
          onChange={(e) => onValueChange(parseFloat(e.target.value))}
          style={styles.crossfaderInput}
        />
        <span style={styles.deckIndicator}>B</span>
      </div>
      <div style={styles.curveSelector}>
        {curves.map((c) => (
          <button
            key={c}
            style={{
              ...styles.curveButton,
              backgroundColor: curve === c ? '#818cf8' : 'rgba(255, 255, 255, 0.1)',
            }}
            onClick={() => onCurveChange(c)}
          >
            {c.charAt(0).toUpperCase()}
          </button>
        ))}
      </div>
    </div>
  );
}

// === Main Component ===

export function DJMixer({ onLoadTrack }: DJMixerProps) {
  const dj = useDJMixer();

  const handlePlayPause = useCallback(
    (deck: 'A' | 'B') => {
      const deckState = deck === 'A' ? dj.deckA : dj.deckB;
      if (deckState.isPlaying) {
        dj.pause(deck);
      } else {
        dj.play(deck);
      }
    },
    [dj]
  );

  return (
    <div style={styles.container}>
      {/* Deck A */}
      <div style={styles.deck}>
        <DeckDisplay deck={dj.deckA} label="DECK A" />

        <Waveform
          peaks={dj.deckA.waveformPeaks}
          position={dj.deckA.position}
          duration={dj.deckA.duration}
          loop={dj.deckA.loop}
          cuePoints={dj.deckA.cuePoints}
          onSeek={(pos) => dj.seek('A', pos)}
        />

        <div style={styles.deckControls}>
          {/* Transport */}
          <div style={styles.transport}>
            <button
              style={{
                ...styles.transportButton,
                ...styles.playButton,
                backgroundColor: dj.deckA.isPlaying ? '#22c55e' : undefined,
              }}
              onClick={() => handlePlayPause('A')}
              disabled={!dj.deckA.isLoaded}
            >
              {dj.deckA.isPlaying ? '\u{23F8}' : '\u{25B6}'}
            </button>
            <button
              style={styles.transportButton}
              onClick={() => dj.stop('A')}
              disabled={!dj.deckA.isLoaded}
            >
              \u{23F9}
            </button>
            <button
              style={styles.transportButton}
              onClick={() => onLoadTrack?.('A')}
            >
              LOAD
            </button>
            <button
              style={{
                ...styles.transportButton,
                backgroundColor: dj.mixer.isSyncEnabled ? '#818cf8' : undefined,
              }}
              onClick={() => dj.syncBpm('A')}
              disabled={!dj.deckA.isLoaded || !dj.deckB.isLoaded}
            >
              SYNC
            </button>
          </div>

          {/* Pitch */}
          <PitchSlider
            value={dj.deckA.pitch}
            onChange={(v) => dj.setPitch('A', v)}
          />

          {/* EQ */}
          <div style={styles.eqSection}>
            <EQKnob
              label="HI"
              value={dj.deckA.eqHigh}
              onChange={(v) => dj.setEQ('A', 'high', v)}
              onKill={() => dj.killEQ('A', 'high')}
            />
            <EQKnob
              label="MID"
              value={dj.deckA.eqMid}
              onChange={(v) => dj.setEQ('A', 'mid', v)}
              onKill={() => dj.killEQ('A', 'mid')}
            />
            <EQKnob
              label="LOW"
              value={dj.deckA.eqLow}
              onChange={(v) => dj.setEQ('A', 'low', v)}
              onKill={() => dj.killEQ('A', 'low')}
            />
          </div>

          {/* Loop */}
          <LoopControls
            loop={dj.deckA.loop}
            onSetLoop={(bars) => dj.setLoop('A', bars)}
            onToggle={() => dj.toggleLoop('A')}
            onClear={() => dj.clearLoop('A')}
            onHalve={() => dj.halveLoop('A')}
            onDouble={() => dj.doubleLoop('A')}
          />

          {/* Cue Points */}
          <CueButtons
            cuePoints={dj.deckA.cuePoints}
            onSet={(i) => dj.setCuePoint('A', i)}
            onJump={(i) => dj.jumpToCue('A', i)}
          />
        </div>

        {/* Volume Fader */}
        <div style={styles.volumeFader}>
          <div style={styles.volumeLabel}>VOL</div>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={dj.deckA.volume}
            onChange={(e) => dj.setDeckVolume('A', parseFloat(e.target.value))}
            style={styles.volumeInput}
          />
        </div>
      </div>

      {/* Mixer Section */}
      <div style={styles.mixerSection}>
        {/* Master Controls */}
        <div style={styles.masterControls}>
          <div style={styles.masterVolume}>
            <div style={styles.masterLabel}>MASTER</div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={dj.mixer.masterVolume}
              onChange={(e) => dj.setMasterVolume(parseFloat(e.target.value))}
              style={styles.masterInput}
            />
          </div>

          <div style={styles.headphoneMix}>
            <div style={styles.headphoneLabel}>CUE/MIX</div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={dj.mixer.headphoneMix}
              onChange={(e) => dj.setHeadphoneMix(parseFloat(e.target.value))}
              style={styles.headphoneInput}
            />
          </div>
        </div>

        {/* Crossfader */}
        <Crossfader
          value={dj.mixer.crossfader}
          curve={dj.mixer.crossfaderCurve}
          onValueChange={dj.setCrossfader}
          onCurveChange={dj.setCrossfaderCurve}
        />

        {/* Cue Buttons */}
        <div style={styles.cueToggleSection}>
          <button
            style={{
              ...styles.cueToggle,
              backgroundColor: dj.mixer.deckACue ? '#818cf8' : undefined,
            }}
            onClick={() => dj.toggleCue('A')}
          >
            CUE A
          </button>
          <button
            style={{
              ...styles.cueToggle,
              backgroundColor: dj.mixer.deckBCue ? '#818cf8' : undefined,
            }}
            onClick={() => dj.toggleCue('B')}
          >
            CUE B
          </button>
        </div>
      </div>

      {/* Deck B */}
      <div style={styles.deck}>
        <DeckDisplay deck={dj.deckB} label="DECK B" />

        <Waveform
          peaks={dj.deckB.waveformPeaks}
          position={dj.deckB.position}
          duration={dj.deckB.duration}
          loop={dj.deckB.loop}
          cuePoints={dj.deckB.cuePoints}
          onSeek={(pos) => dj.seek('B', pos)}
        />

        <div style={styles.deckControls}>
          <div style={styles.transport}>
            <button
              style={{
                ...styles.transportButton,
                ...styles.playButton,
                backgroundColor: dj.deckB.isPlaying ? '#22c55e' : undefined,
              }}
              onClick={() => handlePlayPause('B')}
              disabled={!dj.deckB.isLoaded}
            >
              {dj.deckB.isPlaying ? '\u{23F8}' : '\u{25B6}'}
            </button>
            <button
              style={styles.transportButton}
              onClick={() => dj.stop('B')}
              disabled={!dj.deckB.isLoaded}
            >
              \u{23F9}
            </button>
            <button
              style={styles.transportButton}
              onClick={() => onLoadTrack?.('B')}
            >
              LOAD
            </button>
            <button
              style={{
                ...styles.transportButton,
                backgroundColor: dj.mixer.isSyncEnabled ? '#818cf8' : undefined,
              }}
              onClick={() => dj.syncBpm('B')}
              disabled={!dj.deckA.isLoaded || !dj.deckB.isLoaded}
            >
              SYNC
            </button>
          </div>

          <PitchSlider
            value={dj.deckB.pitch}
            onChange={(v) => dj.setPitch('B', v)}
          />

          <div style={styles.eqSection}>
            <EQKnob
              label="HI"
              value={dj.deckB.eqHigh}
              onChange={(v) => dj.setEQ('B', 'high', v)}
              onKill={() => dj.killEQ('B', 'high')}
            />
            <EQKnob
              label="MID"
              value={dj.deckB.eqMid}
              onChange={(v) => dj.setEQ('B', 'mid', v)}
              onKill={() => dj.killEQ('B', 'mid')}
            />
            <EQKnob
              label="LOW"
              value={dj.deckB.eqLow}
              onChange={(v) => dj.setEQ('B', 'low', v)}
              onKill={() => dj.killEQ('B', 'low')}
            />
          </div>

          <LoopControls
            loop={dj.deckB.loop}
            onSetLoop={(bars) => dj.setLoop('B', bars)}
            onToggle={() => dj.toggleLoop('B')}
            onClear={() => dj.clearLoop('B')}
            onHalve={() => dj.halveLoop('B')}
            onDouble={() => dj.doubleLoop('B')}
          />

          <CueButtons
            cuePoints={dj.deckB.cuePoints}
            onSet={(i) => dj.setCuePoint('B', i)}
            onJump={(i) => dj.jumpToCue('B', i)}
          />
        </div>

        <div style={styles.volumeFader}>
          <div style={styles.volumeLabel}>VOL</div>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={dj.deckB.volume}
            onChange={(e) => dj.setDeckVolume('B', parseFloat(e.target.value))}
            style={styles.volumeInput}
          />
        </div>
      </div>
    </div>
  );
}

// === Styles ===

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    gap: '16px',
    padding: '16px',
    backgroundColor: '#0a0a0a',
    borderRadius: '12px',
    fontFamily: 'system-ui, -apple-system, sans-serif',
    color: '#ffffff',
  },
  deck: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    padding: '12px',
    backgroundColor: '#1a1a1a',
    borderRadius: '8px',
  },
  deckDisplay: {
    display: 'flex',
    alignItems: 'center',
    gap: '16px',
    padding: '8px',
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: '6px',
  },
  deckLabel: {
    fontSize: '12px',
    fontWeight: 700,
    color: '#818cf8',
  },
  trackInfo: {
    flex: 1,
  },
  trackTitle: {
    fontSize: '14px',
    fontWeight: 500,
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  trackArtist: {
    fontSize: '12px',
    color: '#9ca3af',
  },
  timeDisplay: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-end',
  },
  currentTime: {
    fontSize: '16px',
    fontFamily: 'monospace',
    fontWeight: 600,
  },
  remainingTime: {
    fontSize: '12px',
    fontFamily: 'monospace',
    color: '#9ca3af',
  },
  bpmDisplay: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: '4px 12px',
    backgroundColor: 'rgba(129, 140, 248, 0.1)',
    borderRadius: '4px',
  },
  bpmValue: {
    fontSize: '18px',
    fontWeight: 700,
    color: '#818cf8',
  },
  bpmLabel: {
    fontSize: '10px',
    color: '#9ca3af',
  },
  pitchValue: {
    fontSize: '10px',
    color: '#22c55e',
  },
  waveformContainer: {
    position: 'relative',
    height: '80px',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    borderRadius: '4px',
    overflow: 'hidden',
    cursor: 'pointer',
  },
  waveformBars: {
    display: 'flex',
    alignItems: 'center',
    height: '100%',
    gap: '1px',
  },
  waveformBar: {
    flex: 1,
    transition: 'height 50ms',
  },
  waveformPlaceholder: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '100%',
    height: '100%',
    color: '#6b7280',
    fontSize: '12px',
  },
  loopRegion: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    backgroundColor: '#818cf8',
    pointerEvents: 'none',
  },
  cueMarker: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: '2px',
    pointerEvents: 'none',
  },
  playhead: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: '2px',
    backgroundColor: '#ffffff',
    boxShadow: '0 0 8px rgba(255, 255, 255, 0.5)',
    pointerEvents: 'none',
  },
  deckControls: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  transport: {
    display: 'flex',
    gap: '8px',
  },
  transportButton: {
    flex: 1,
    padding: '10px',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    border: 'none',
    borderRadius: '4px',
    color: '#ffffff',
    fontSize: '12px',
    fontWeight: 500,
    cursor: 'pointer',
  },
  playButton: {
    fontSize: '16px',
  },
  pitchSlider: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  pitchLabel: {
    fontSize: '10px',
    color: '#9ca3af',
    width: '40px',
  },
  pitchInput: {
    flex: 1,
    accentColor: '#818cf8',
  },
  eqSection: {
    display: 'flex',
    gap: '8px',
  },
  eqKnob: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '4px',
  },
  eqLabel: {
    fontSize: '10px',
    color: '#9ca3af',
  },
  eqInput: {
    width: '100%',
    accentColor: '#818cf8',
  },
  killButton: {
    padding: '2px 8px',
    backgroundColor: 'rgba(239, 68, 68, 0.2)',
    border: 'none',
    borderRadius: '2px',
    color: '#ef4444',
    fontSize: '9px',
    cursor: 'pointer',
  },
  loopControls: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  },
  loopLabel: {
    fontSize: '10px',
    color: '#9ca3af',
  },
  loopButtons: {
    display: 'flex',
    gap: '4px',
  },
  loopSizeButton: {
    flex: 1,
    padding: '6px',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    border: 'none',
    borderRadius: '3px',
    color: '#ffffff',
    fontSize: '10px',
    cursor: 'pointer',
  },
  loopActions: {
    display: 'flex',
    gap: '4px',
  },
  loopActionButton: {
    flex: 1,
    padding: '6px',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    border: 'none',
    borderRadius: '3px',
    color: '#ffffff',
    fontSize: '10px',
    cursor: 'pointer',
  },
  cueButtons: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  },
  cueLabel: {
    fontSize: '10px',
    color: '#9ca3af',
  },
  cueGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(8, 1fr)',
    gap: '4px',
  },
  cueButton: {
    padding: '8px',
    border: 'none',
    borderRadius: '3px',
    color: '#ffffff',
    fontSize: '10px',
    fontWeight: 600,
    cursor: 'pointer',
  },
  volumeFader: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  volumeLabel: {
    fontSize: '10px',
    color: '#9ca3af',
    width: '30px',
  },
  volumeInput: {
    flex: 1,
    accentColor: '#818cf8',
  },
  mixerSection: {
    width: '200px',
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
    padding: '12px',
    backgroundColor: '#1a1a1a',
    borderRadius: '8px',
  },
  masterControls: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  masterVolume: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  masterLabel: {
    fontSize: '10px',
    color: '#9ca3af',
    textAlign: 'center',
  },
  masterInput: {
    accentColor: '#22c55e',
  },
  headphoneMix: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  headphoneLabel: {
    fontSize: '10px',
    color: '#9ca3af',
    textAlign: 'center',
  },
  headphoneInput: {
    accentColor: '#eab308',
  },
  crossfader: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  crossfaderLabel: {
    fontSize: '10px',
    color: '#9ca3af',
    textAlign: 'center',
  },
  crossfaderTrack: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  deckIndicator: {
    fontSize: '12px',
    fontWeight: 700,
    color: '#818cf8',
  },
  crossfaderInput: {
    flex: 1,
    accentColor: '#818cf8',
  },
  curveSelector: {
    display: 'flex',
    gap: '4px',
  },
  curveButton: {
    flex: 1,
    padding: '4px',
    border: 'none',
    borderRadius: '3px',
    color: '#ffffff',
    fontSize: '10px',
    cursor: 'pointer',
  },
  cueToggleSection: {
    display: 'flex',
    gap: '8px',
  },
  cueToggle: {
    flex: 1,
    padding: '8px',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    border: 'none',
    borderRadius: '4px',
    color: '#ffffff',
    fontSize: '11px',
    fontWeight: 500,
    cursor: 'pointer',
  },
};

export default DJMixer;
