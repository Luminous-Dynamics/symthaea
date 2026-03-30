// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Effects Panel Component
 *
 * Provides UI controls for audio effects: EQ, Reverb, Spatial, Compressor.
 */

import { useCallback, useState } from 'react';
import { useAudioEffects, type EQSettings } from '@/hooks/useAudioEffects';

// === Types ===

type EffectTab = 'eq' | 'reverb' | 'spatial' | 'dynamics';

interface EffectsPanelProps {
  onClose?: () => void;
  compact?: boolean;
}

// === Sub-Components ===

function Knob({
  value,
  min,
  max,
  label,
  unit = '',
  onChange,
  size = 48,
}: {
  value: number;
  min: number;
  max: number;
  label: string;
  unit?: string;
  onChange: (value: number) => void;
  size?: number;
}) {
  const normalized = (value - min) / (max - min);
  const angle = -135 + normalized * 270;

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      const startY = e.clientY;
      const startValue = value;

      const handleMouseMove = (moveEvent: MouseEvent) => {
        const delta = (startY - moveEvent.clientY) / 100;
        const newValue = Math.max(min, Math.min(max, startValue + delta * (max - min)));
        onChange(newValue);
      };

      const handleMouseUp = () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };

      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    },
    [value, min, max, onChange]
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px' }}>
      <div
        style={{
          width: size,
          height: size,
          borderRadius: '50%',
          backgroundColor: 'rgba(255, 255, 255, 0.1)',
          border: '2px solid rgba(255, 255, 255, 0.2)',
          cursor: 'pointer',
          position: 'relative',
        }}
        onMouseDown={handleMouseDown}
      >
        <div
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            width: '2px',
            height: size / 2 - 6,
            backgroundColor: '#818cf8',
            transformOrigin: 'bottom center',
            transform: `translate(-50%, -100%) rotate(${angle}deg)`,
          }}
        />
      </div>
      <span style={{ fontSize: '10px', color: '#9ca3af' }}>{label}</span>
      <span style={{ fontSize: '11px', color: '#ffffff' }}>
        {value.toFixed(1)}{unit}
      </span>
    </div>
  );
}

function Slider({
  value,
  min,
  max,
  label,
  unit = '',
  onChange,
  vertical = false,
}: {
  value: number;
  min: number;
  max: number;
  label: string;
  unit?: string;
  onChange: (value: number) => void;
  vertical?: boolean;
}) {
  const normalized = (value - min) / (max - min);

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: vertical ? 'column' : 'row',
        alignItems: 'center',
        gap: '8px',
      }}
    >
      <span style={{ fontSize: '11px', color: '#9ca3af', minWidth: '60px' }}>{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={(max - min) / 100}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{
          flex: 1,
          accentColor: '#818cf8',
          cursor: 'pointer',
        }}
      />
      <span style={{ fontSize: '11px', color: '#ffffff', minWidth: '50px', textAlign: 'right' }}>
        {value.toFixed(1)}{unit}
      </span>
    </div>
  );
}

function Toggle({
  enabled,
  label,
  onChange,
}: {
  enabled: boolean;
  label: string;
  onChange: (enabled: boolean) => void;
}) {
  return (
    <button
      onClick={() => onChange(!enabled)}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        padding: '6px 12px',
        backgroundColor: enabled ? 'rgba(129, 140, 248, 0.2)' : 'rgba(255, 255, 255, 0.05)',
        border: `1px solid ${enabled ? '#818cf8' : 'rgba(255, 255, 255, 0.1)'}`,
        borderRadius: '6px',
        color: enabled ? '#818cf8' : '#9ca3af',
        cursor: 'pointer',
        fontSize: '12px',
        fontWeight: 500,
      }}
    >
      <div
        style={{
          width: '8px',
          height: '8px',
          borderRadius: '50%',
          backgroundColor: enabled ? '#22c55e' : '#6b7280',
        }}
      />
      {label}
    </button>
  );
}

// === EQ Tab ===

function EQTab({ effects }: { effects: ReturnType<typeof useAudioEffects> }) {
  const { eq, setEQEnabled, setEQBand, resetEQ } = effects;

  return (
    <div style={styles.tabContent}>
      <div style={styles.tabHeader}>
        <Toggle enabled={eq.enabled} label="EQ" onChange={setEQEnabled} />
        <button style={styles.resetButton} onClick={resetEQ}>Reset</button>
      </div>

      <div style={styles.eqBands}>
        {/* Low Shelf */}
        <div style={styles.eqBand}>
          <Slider
            value={eq.settings.lowShelf.gain}
            min={-12}
            max={12}
            label="Low"
            unit="dB"
            onChange={(gain) => setEQBand('lowShelf', { ...eq.settings.lowShelf, gain })}
          />
          <Slider
            value={eq.settings.lowShelf.frequency}
            min={20}
            max={500}
            label="Freq"
            unit="Hz"
            onChange={(frequency) => setEQBand('lowShelf', { ...eq.settings.lowShelf, frequency })}
          />
        </div>

        {/* Low Mid */}
        <div style={styles.eqBand}>
          <Slider
            value={eq.settings.lowMid.gain}
            min={-12}
            max={12}
            label="Low Mid"
            unit="dB"
            onChange={(gain) => setEQBand('lowMid', { ...eq.settings.lowMid, gain })}
          />
          <Slider
            value={eq.settings.lowMid.frequency}
            min={100}
            max={1000}
            label="Freq"
            unit="Hz"
            onChange={(frequency) => setEQBand('lowMid', { ...eq.settings.lowMid, frequency })}
          />
          <Slider
            value={eq.settings.lowMid.q}
            min={0.1}
            max={10}
            label="Q"
            onChange={(q) => setEQBand('lowMid', { ...eq.settings.lowMid, q })}
          />
        </div>

        {/* Mid */}
        <div style={styles.eqBand}>
          <Slider
            value={eq.settings.mid.gain}
            min={-12}
            max={12}
            label="Mid"
            unit="dB"
            onChange={(gain) => setEQBand('mid', { ...eq.settings.mid, gain })}
          />
          <Slider
            value={eq.settings.mid.frequency}
            min={500}
            max={4000}
            label="Freq"
            unit="Hz"
            onChange={(frequency) => setEQBand('mid', { ...eq.settings.mid, frequency })}
          />
          <Slider
            value={eq.settings.mid.q}
            min={0.1}
            max={10}
            label="Q"
            onChange={(q) => setEQBand('mid', { ...eq.settings.mid, q })}
          />
        </div>

        {/* High Mid */}
        <div style={styles.eqBand}>
          <Slider
            value={eq.settings.highMid.gain}
            min={-12}
            max={12}
            label="High Mid"
            unit="dB"
            onChange={(gain) => setEQBand('highMid', { ...eq.settings.highMid, gain })}
          />
          <Slider
            value={eq.settings.highMid.frequency}
            min={2000}
            max={10000}
            label="Freq"
            unit="Hz"
            onChange={(frequency) => setEQBand('highMid', { ...eq.settings.highMid, frequency })}
          />
          <Slider
            value={eq.settings.highMid.q}
            min={0.1}
            max={10}
            label="Q"
            onChange={(q) => setEQBand('highMid', { ...eq.settings.highMid, q })}
          />
        </div>

        {/* High Shelf */}
        <div style={styles.eqBand}>
          <Slider
            value={eq.settings.highShelf.gain}
            min={-12}
            max={12}
            label="High"
            unit="dB"
            onChange={(gain) => setEQBand('highShelf', { ...eq.settings.highShelf, gain })}
          />
          <Slider
            value={eq.settings.highShelf.frequency}
            min={5000}
            max={20000}
            label="Freq"
            unit="Hz"
            onChange={(frequency) => setEQBand('highShelf', { ...eq.settings.highShelf, frequency })}
          />
        </div>
      </div>
    </div>
  );
}

// === Reverb Tab ===

function ReverbTab({ effects }: { effects: ReturnType<typeof useAudioEffects> }) {
  const { reverb, setReverbEnabled, setReverbSettings, resetReverb } = effects;

  return (
    <div style={styles.tabContent}>
      <div style={styles.tabHeader}>
        <Toggle enabled={reverb.enabled} label="Reverb" onChange={setReverbEnabled} />
        <button style={styles.resetButton} onClick={resetReverb}>Reset</button>
      </div>

      <div style={styles.knobRow}>
        <Knob
          value={reverb.settings.roomSize}
          min={0}
          max={1}
          label="Room Size"
          onChange={(roomSize) => setReverbSettings({ roomSize })}
        />
        <Knob
          value={reverb.settings.damping}
          min={0}
          max={1}
          label="Damping"
          onChange={(damping) => setReverbSettings({ damping })}
        />
        <Knob
          value={reverb.settings.mix}
          min={0}
          max={1}
          label="Mix"
          onChange={(mix) => setReverbSettings({ mix })}
        />
      </div>

      <div style={styles.presets}>
        <span style={{ fontSize: '11px', color: '#9ca3af' }}>Presets:</span>
        <button
          style={styles.presetButton}
          onClick={() => setReverbSettings({ roomSize: 0.3, damping: 0.7, mix: 0.2 })}
        >
          Small Room
        </button>
        <button
          style={styles.presetButton}
          onClick={() => setReverbSettings({ roomSize: 0.6, damping: 0.5, mix: 0.3 })}
        >
          Hall
        </button>
        <button
          style={styles.presetButton}
          onClick={() => setReverbSettings({ roomSize: 0.9, damping: 0.2, mix: 0.5 })}
        >
          Cathedral
        </button>
      </div>
    </div>
  );
}

// === Spatial Tab ===

function SpatialTab({ effects }: { effects: ReturnType<typeof useAudioEffects> }) {
  const { spatial, setSpatialEnabled, setSpatialPosition, resetSpatial } = effects;

  return (
    <div style={styles.tabContent}>
      <div style={styles.tabHeader}>
        <Toggle enabled={spatial.enabled} label="3D Audio" onChange={setSpatialEnabled} />
        <button style={styles.resetButton} onClick={resetSpatial}>Reset</button>
      </div>

      {/* 3D Position Visualizer */}
      <div style={styles.spatialViz}>
        <div
          style={{
            width: '150px',
            height: '150px',
            borderRadius: '50%',
            border: '2px solid rgba(255, 255, 255, 0.2)',
            position: 'relative',
            backgroundColor: 'rgba(255, 255, 255, 0.05)',
          }}
        >
          {/* Listener (center) */}
          <div
            style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              backgroundColor: '#818cf8',
            }}
          />
          {/* Sound source */}
          <div
            style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: `translate(-50%, -50%) rotate(${spatial.settings.azimuth}deg) translateY(${-60 / spatial.settings.distance}px)`,
              width: '10px',
              height: '10px',
              borderRadius: '50%',
              backgroundColor: '#22c55e',
            }}
          />
        </div>
      </div>

      <div style={styles.sliderGroup}>
        <Slider
          value={spatial.settings.azimuth}
          min={-180}
          max={180}
          label="Azimuth"
          unit="°"
          onChange={(azimuth) =>
            setSpatialPosition(azimuth, spatial.settings.elevation, spatial.settings.distance)
          }
        />
        <Slider
          value={spatial.settings.elevation}
          min={-90}
          max={90}
          label="Elevation"
          unit="°"
          onChange={(elevation) =>
            setSpatialPosition(spatial.settings.azimuth, elevation, spatial.settings.distance)
          }
        />
        <Slider
          value={spatial.settings.distance}
          min={0.1}
          max={10}
          label="Distance"
          unit="m"
          onChange={(distance) =>
            setSpatialPosition(spatial.settings.azimuth, spatial.settings.elevation, distance)
          }
        />
      </div>
    </div>
  );
}

// === Dynamics Tab ===

function DynamicsTab({ effects }: { effects: ReturnType<typeof useAudioEffects> }) {
  const {
    compressor,
    limiter,
    gainReduction,
    setCompressorEnabled,
    setCompressorSettings,
    resetCompressor,
    setLimiterEnabled,
    setLimiterSettings,
    resetLimiter,
  } = effects;

  return (
    <div style={styles.tabContent}>
      {/* Compressor */}
      <div style={styles.section}>
        <div style={styles.tabHeader}>
          <Toggle enabled={compressor.enabled} label="Compressor" onChange={setCompressorEnabled} />
          <button style={styles.resetButton} onClick={resetCompressor}>Reset</button>
        </div>

        <div style={styles.sliderGroup}>
          <Slider
            value={compressor.settings.threshold}
            min={-60}
            max={0}
            label="Threshold"
            unit="dB"
            onChange={(threshold) => setCompressorSettings({ threshold })}
          />
          <Slider
            value={compressor.settings.ratio}
            min={1}
            max={20}
            label="Ratio"
            unit=":1"
            onChange={(ratio) => setCompressorSettings({ ratio })}
          />
          <Slider
            value={compressor.settings.attack}
            min={0.1}
            max={100}
            label="Attack"
            unit="ms"
            onChange={(attack) => setCompressorSettings({ attack })}
          />
          <Slider
            value={compressor.settings.release}
            min={10}
            max={1000}
            label="Release"
            unit="ms"
            onChange={(release) => setCompressorSettings({ release })}
          />
          <Slider
            value={compressor.settings.makeupGain}
            min={-12}
            max={24}
            label="Makeup"
            unit="dB"
            onChange={(makeupGain) => setCompressorSettings({ makeupGain })}
          />
        </div>

        {/* Gain Reduction Meter */}
        <div style={styles.meter}>
          <span style={{ fontSize: '11px', color: '#9ca3af' }}>GR</span>
          <div style={styles.meterTrack}>
            <div
              style={{
                ...styles.meterFill,
                width: `${Math.min(100, gainReduction * 5)}%`,
                backgroundColor: gainReduction > 10 ? '#ef4444' : '#eab308',
              }}
            />
          </div>
          <span style={{ fontSize: '11px', color: '#ffffff', minWidth: '40px' }}>
            -{gainReduction.toFixed(1)}dB
          </span>
        </div>
      </div>

      {/* Limiter */}
      <div style={styles.section}>
        <div style={styles.tabHeader}>
          <Toggle enabled={limiter.enabled} label="Limiter" onChange={setLimiterEnabled} />
          <button style={styles.resetButton} onClick={resetLimiter}>Reset</button>
        </div>

        <div style={styles.sliderGroup}>
          <Slider
            value={limiter.settings.ceiling}
            min={-12}
            max={0}
            label="Ceiling"
            unit="dB"
            onChange={(ceiling) => setLimiterSettings({ ceiling })}
          />
          <Slider
            value={limiter.settings.release}
            min={1}
            max={500}
            label="Release"
            unit="ms"
            onChange={(release) => setLimiterSettings({ release })}
          />
        </div>
      </div>
    </div>
  );
}

// === Main Component ===

export function EffectsPanel({ onClose, compact = false }: EffectsPanelProps) {
  const effects = useAudioEffects();
  const [activeTab, setActiveTab] = useState<EffectTab>('eq');

  const tabs: { id: EffectTab; label: string; icon: string }[] = [
    { id: 'eq', label: 'EQ', icon: '\u{1F3DA}' },
    { id: 'reverb', label: 'Reverb', icon: '\u{1F3DB}' },
    { id: 'spatial', label: '3D', icon: '\u{1F30D}' },
    { id: 'dynamics', label: 'Dynamics', icon: '\u{1F4CA}' },
  ];

  return (
    <div style={{ ...styles.container, ...(compact && styles.containerCompact) }}>
      {/* Header */}
      <div style={styles.header}>
        <h3 style={styles.title}>Audio Effects</h3>
        {onClose && (
          <button style={styles.closeButton} onClick={onClose}>
            X
          </button>
        )}
      </div>

      {/* Tab Navigation */}
      <div style={styles.tabNav}>
        {tabs.map((tab) => (
          <button
            key={tab.id}
            style={{
              ...styles.tabButton,
              ...(activeTab === tab.id && styles.tabButtonActive),
            }}
            onClick={() => setActiveTab(tab.id)}
          >
            <span>{tab.icon}</span>
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'eq' && <EQTab effects={effects} />}
      {activeTab === 'reverb' && <ReverbTab effects={effects} />}
      {activeTab === 'spatial' && <SpatialTab effects={effects} />}
      {activeTab === 'dynamics' && <DynamicsTab effects={effects} />}

      {/* Status */}
      <div style={styles.status}>
        <span
          style={{
            ...styles.statusDot,
            backgroundColor: effects.isLoaded ? '#22c55e' : '#6b7280',
          }}
        />
        <span style={{ fontSize: '11px', color: '#9ca3af' }}>
          {effects.isLoaded ? 'WASM Effects Ready' : 'Loading...'}
        </span>
      </div>
    </div>
  );
}

// === Styles ===

const styles: Record<string, React.CSSProperties> = {
  container: {
    backgroundColor: '#1a1a1a',
    borderRadius: '12px',
    padding: '16px',
    width: '400px',
    fontFamily: 'system-ui, -apple-system, sans-serif',
    color: '#ffffff',
  },
  containerCompact: {
    width: '320px',
    padding: '12px',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '12px',
  },
  title: {
    fontSize: '16px',
    fontWeight: 600,
    margin: 0,
  },
  closeButton: {
    background: 'none',
    border: 'none',
    color: '#9ca3af',
    fontSize: '16px',
    cursor: 'pointer',
    padding: '4px 8px',
  },
  tabNav: {
    display: 'flex',
    gap: '4px',
    marginBottom: '16px',
  },
  tabButton: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '4px',
    padding: '8px',
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    border: '1px solid transparent',
    borderRadius: '8px',
    color: '#9ca3af',
    cursor: 'pointer',
    fontSize: '11px',
  },
  tabButtonActive: {
    backgroundColor: 'rgba(129, 140, 248, 0.1)',
    borderColor: '#818cf8',
    color: '#818cf8',
  },
  tabContent: {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
  },
  tabHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  resetButton: {
    background: 'none',
    border: '1px solid rgba(255, 255, 255, 0.2)',
    borderRadius: '4px',
    padding: '4px 8px',
    color: '#9ca3af',
    fontSize: '11px',
    cursor: 'pointer',
  },
  eqBands: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  eqBand: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    padding: '8px',
    backgroundColor: 'rgba(255, 255, 255, 0.03)',
    borderRadius: '8px',
  },
  knobRow: {
    display: 'flex',
    justifyContent: 'space-around',
    padding: '16px 0',
  },
  presets: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    flexWrap: 'wrap',
  },
  presetButton: {
    padding: '4px 10px',
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    border: '1px solid rgba(255, 255, 255, 0.1)',
    borderRadius: '4px',
    color: '#ffffff',
    fontSize: '11px',
    cursor: 'pointer',
  },
  spatialViz: {
    display: 'flex',
    justifyContent: 'center',
    padding: '16px 0',
  },
  sliderGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  section: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    padding: '12px',
    backgroundColor: 'rgba(255, 255, 255, 0.03)',
    borderRadius: '8px',
  },
  meter: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  meterTrack: {
    flex: 1,
    height: '6px',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: '3px',
    overflow: 'hidden',
  },
  meterFill: {
    height: '100%',
    transition: 'width 50ms',
  },
  status: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginTop: '12px',
    paddingTop: '12px',
    borderTop: '1px solid rgba(255, 255, 255, 0.1)',
  },
  statusDot: {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
  },
};

export default EffectsPanel;
