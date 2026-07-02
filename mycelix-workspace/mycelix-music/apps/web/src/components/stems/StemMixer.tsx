// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * StemMixer Component
 *
 * Professional stem mixing interface for AI-separated audio
 * Features individual stem control, export, and visualization
 */

import React, { useState, useRef, useCallback } from 'react';
import { useStemSeparation, StemType, Stem } from '../../hooks/useStemSeparation';

interface StemMixerProps {
  className?: string;
  onExport?: (blob: Blob, filename: string) => void;
}

export function StemMixer({ className = '', onExport }: StemMixerProps) {
  const {
    stems,
    progress,
    isProcessing,
    isPlaying,
    currentTime,
    duration,
    separateAudio,
    play,
    pause,
    stop,
    seek,
    controls,
    exportStem,
    exportMix,
    STEM_COLORS,
    STEM_LABELS,
  } = useStemSeparation();

  const [dragOver, setDragOver] = useState(false);
  const [selectedStems, setSelectedStems] = useState<StemType[]>(['vocals', 'drums', 'bass', 'other']);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Handle file drop
  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('audio/')) {
      await processFile(file);
    }
  }, [selectedStems]);

  // Handle file selection
  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      await processFile(file);
    }
  }, [selectedStems]);

  // Process audio file
  const processFile = async (file: File) => {
    const audioContext = new AudioContext();
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    await separateAudio(audioBuffer, selectedStems);
  };

  // Format time
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Handle export
  const handleExportStem = async (type: StemType) => {
    const blob = await exportStem(type);
    if (blob && onExport) {
      onExport(blob, `${STEM_LABELS[type].toLowerCase()}.wav`);
    } else if (blob) {
      downloadBlob(blob, `${STEM_LABELS[type].toLowerCase()}.wav`);
    }
  };

  const handleExportMix = async () => {
    const blob = await exportMix();
    if (blob && onExport) {
      onExport(blob, 'stem-mix.wav');
    } else if (blob) {
      downloadBlob(blob, 'stem-mix.wav');
    }
  };

  const downloadBlob = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Toggle stem selection
  const toggleStemSelection = (type: StemType) => {
    setSelectedStems(prev =>
      prev.includes(type)
        ? prev.filter(t => t !== type)
        : [...prev, type]
    );
  };

  return (
    <div className={`stem-mixer ${className}`} style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <h2 style={styles.title}>AI Stem Separation</h2>
        <p style={styles.subtitle}>Isolate vocals, drums, bass, and instruments</p>
      </div>

      {/* Stem Type Selection */}
      <div style={styles.stemSelection}>
        <span style={styles.selectionLabel}>Stems to extract:</span>
        <div style={styles.stemChips}>
          {(['vocals', 'drums', 'bass', 'other', 'piano', 'guitar', 'synth'] as StemType[]).map(type => (
            <button
              key={type}
              onClick={() => toggleStemSelection(type)}
              style={{
                ...styles.stemChip,
                backgroundColor: selectedStems.includes(type) ? STEM_COLORS[type] : '#374151',
                borderColor: STEM_COLORS[type],
              }}
            >
              {STEM_LABELS[type]}
            </button>
          ))}
        </div>
      </div>

      {/* Drop Zone */}
      {stems.size === 0 && !isProcessing && (
        <div
          style={{
            ...styles.dropZone,
            borderColor: dragOver ? '#818cf8' : '#4b5563',
            backgroundColor: dragOver ? 'rgba(129, 140, 248, 0.1)' : 'transparent',
          }}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <div style={styles.dropIcon}>🎵</div>
          <p style={styles.dropText}>Drop audio file here or click to browse</p>
          <p style={styles.dropHint}>Supports MP3, WAV, FLAC, and more</p>
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />
        </div>
      )}

      {/* Processing Progress */}
      {isProcessing && (
        <div style={styles.progressContainer}>
          <div style={styles.progressHeader}>
            <span>{progress.message}</span>
            <span>{Math.round(progress.progress)}%</span>
          </div>
          <div style={styles.progressBar}>
            <div
              style={{
                ...styles.progressFill,
                width: `${progress.progress}%`,
              }}
            />
          </div>
          {progress.currentStem && (
            <div style={styles.currentStem}>
              <span
                style={{
                  ...styles.stemIndicator,
                  backgroundColor: STEM_COLORS[progress.currentStem],
                }}
              />
              {STEM_LABELS[progress.currentStem]}
            </div>
          )}
        </div>
      )}

      {/* Stem Mixer Controls */}
      {stems.size > 0 && !isProcessing && (
        <>
          {/* Transport Controls */}
          <div style={styles.transport}>
            <button onClick={stop} style={styles.transportBtn} title="Stop">
              ⏹
            </button>
            <button
              onClick={isPlaying ? pause : play}
              style={{ ...styles.transportBtn, ...styles.playBtn }}
              title={isPlaying ? 'Pause' : 'Play'}
            >
              {isPlaying ? '⏸' : '▶'}
            </button>
            <div style={styles.timeDisplay}>
              <span>{formatTime(currentTime)}</span>
              <input
                type="range"
                min={0}
                max={duration}
                value={currentTime}
                onChange={(e) => seek(parseFloat(e.target.value))}
                style={styles.seekBar}
              />
              <span>{formatTime(duration)}</span>
            </div>
          </div>

          {/* Stem Channels */}
          <div style={styles.channels}>
            {Array.from(stems.entries()).map(([type, stem]) => (
              <StemChannel
                key={type}
                stem={stem}
                color={STEM_COLORS[type]}
                label={STEM_LABELS[type]}
                onLevelChange={(level) => controls.setLevel(type, level)}
                onMuteToggle={() => controls.setMuted(type, !stem.muted)}
                onSoloToggle={() => controls.setSolo(type, !stem.solo)}
                onPanChange={(pan) => controls.setPan(type, pan)}
                onExport={() => handleExportStem(type)}
              />
            ))}
          </div>

          {/* Master Controls */}
          <div style={styles.masterControls}>
            <button onClick={controls.resetAll} style={styles.resetBtn}>
              Reset All
            </button>
            <button onClick={handleExportMix} style={styles.exportBtn}>
              Export Mix
            </button>
            <button
              onClick={() => {
                stop();
                setSelectedStems(['vocals', 'drums', 'bass', 'other']);
              }}
              style={styles.newFileBtn}
            >
              New File
            </button>
          </div>
        </>
      )}
    </div>
  );
}

// Individual Stem Channel Component
interface StemChannelProps {
  stem: Stem;
  color: string;
  label: string;
  onLevelChange: (level: number) => void;
  onMuteToggle: () => void;
  onSoloToggle: () => void;
  onPanChange: (pan: number) => void;
  onExport: () => void;
}

function StemChannel({
  stem,
  color,
  label,
  onLevelChange,
  onMuteToggle,
  onSoloToggle,
  onPanChange,
  onExport,
}: StemChannelProps) {
  return (
    <div style={styles.channel}>
      {/* Header */}
      <div style={styles.channelHeader}>
        <span style={{ ...styles.channelDot, backgroundColor: color }} />
        <span style={styles.channelLabel}>{label}</span>
      </div>

      {/* Fader */}
      <div style={styles.faderContainer}>
        <input
          type="range"
          min={0}
          max={1.5}
          step={0.01}
          value={stem.level}
          onChange={(e) => onLevelChange(parseFloat(e.target.value))}
          style={{
            ...styles.fader,
            background: `linear-gradient(to top, ${color} ${(stem.level / 1.5) * 100}%, #374151 ${(stem.level / 1.5) * 100}%)`,
          }}
          className="stem-fader"
        />
        <span style={styles.faderValue}>{Math.round(stem.level * 100)}%</span>
      </div>

      {/* Pan */}
      <div style={styles.panContainer}>
        <span style={styles.panLabel}>L</span>
        <input
          type="range"
          min={-1}
          max={1}
          step={0.01}
          value={stem.pan}
          onChange={(e) => onPanChange(parseFloat(e.target.value))}
          style={styles.panSlider}
        />
        <span style={styles.panLabel}>R</span>
      </div>

      {/* Buttons */}
      <div style={styles.channelButtons}>
        <button
          onClick={onMuteToggle}
          style={{
            ...styles.channelBtn,
            backgroundColor: stem.muted ? '#ef4444' : '#374151',
          }}
        >
          M
        </button>
        <button
          onClick={onSoloToggle}
          style={{
            ...styles.channelBtn,
            backgroundColor: stem.solo ? '#eab308' : '#374151',
          }}
        >
          S
        </button>
        <button onClick={onExport} style={styles.exportStemBtn} title="Export">
          ↓
        </button>
      </div>
    </div>
  );
}

// Styles
const styles: Record<string, React.CSSProperties> = {
  container: {
    backgroundColor: '#1f2937',
    borderRadius: '12px',
    padding: '24px',
    color: '#f9fafb',
    fontFamily: 'system-ui, -apple-system, sans-serif',
  },
  header: {
    marginBottom: '20px',
  },
  title: {
    fontSize: '24px',
    fontWeight: 700,
    margin: 0,
    background: 'linear-gradient(135deg, #818cf8 0%, #c084fc 100%)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
  },
  subtitle: {
    fontSize: '14px',
    color: '#9ca3af',
    margin: '4px 0 0 0',
  },
  stemSelection: {
    marginBottom: '20px',
  },
  selectionLabel: {
    fontSize: '12px',
    color: '#9ca3af',
    marginBottom: '8px',
    display: 'block',
  },
  stemChips: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '8px',
  },
  stemChip: {
    padding: '6px 12px',
    borderRadius: '16px',
    border: '2px solid',
    fontSize: '12px',
    fontWeight: 500,
    cursor: 'pointer',
    color: '#fff',
    transition: 'all 0.2s',
  },
  dropZone: {
    border: '2px dashed',
    borderRadius: '12px',
    padding: '48px',
    textAlign: 'center',
    cursor: 'pointer',
    transition: 'all 0.2s',
  },
  dropIcon: {
    fontSize: '48px',
    marginBottom: '16px',
  },
  dropText: {
    fontSize: '16px',
    color: '#d1d5db',
    margin: '0 0 8px 0',
  },
  dropHint: {
    fontSize: '12px',
    color: '#6b7280',
    margin: 0,
  },
  progressContainer: {
    padding: '24px',
    backgroundColor: '#111827',
    borderRadius: '8px',
  },
  progressHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    marginBottom: '12px',
    fontSize: '14px',
  },
  progressBar: {
    height: '8px',
    backgroundColor: '#374151',
    borderRadius: '4px',
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    background: 'linear-gradient(90deg, #818cf8, #c084fc)',
    transition: 'width 0.3s',
  },
  currentStem: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginTop: '12px',
    fontSize: '12px',
    color: '#9ca3af',
  },
  stemIndicator: {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
  },
  transport: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    marginBottom: '24px',
    padding: '16px',
    backgroundColor: '#111827',
    borderRadius: '8px',
  },
  transportBtn: {
    width: '40px',
    height: '40px',
    borderRadius: '50%',
    border: 'none',
    backgroundColor: '#374151',
    color: '#fff',
    fontSize: '16px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  playBtn: {
    backgroundColor: '#818cf8',
    width: '48px',
    height: '48px',
    fontSize: '20px',
  },
  timeDisplay: {
    flex: 1,
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    fontSize: '12px',
    fontFamily: 'monospace',
  },
  seekBar: {
    flex: 1,
    height: '4px',
    cursor: 'pointer',
  },
  channels: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(100px, 1fr))',
    gap: '12px',
    marginBottom: '24px',
  },
  channel: {
    backgroundColor: '#111827',
    borderRadius: '8px',
    padding: '16px',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '12px',
  },
  channelHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  channelDot: {
    width: '10px',
    height: '10px',
    borderRadius: '50%',
  },
  channelLabel: {
    fontSize: '12px',
    fontWeight: 600,
  },
  faderContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '8px',
  },
  fader: {
    writingMode: 'vertical-lr',
    direction: 'rtl',
    width: '8px',
    height: '120px',
    cursor: 'pointer',
    appearance: 'none',
    backgroundColor: '#374151',
    borderRadius: '4px',
  },
  faderValue: {
    fontSize: '10px',
    color: '#9ca3af',
  },
  panContainer: {
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
    width: '100%',
  },
  panLabel: {
    fontSize: '10px',
    color: '#6b7280',
  },
  panSlider: {
    flex: 1,
    height: '4px',
    cursor: 'pointer',
  },
  channelButtons: {
    display: 'flex',
    gap: '4px',
  },
  channelBtn: {
    width: '28px',
    height: '28px',
    borderRadius: '4px',
    border: 'none',
    fontSize: '10px',
    fontWeight: 700,
    cursor: 'pointer',
    color: '#fff',
  },
  exportStemBtn: {
    width: '28px',
    height: '28px',
    borderRadius: '4px',
    border: 'none',
    backgroundColor: '#374151',
    fontSize: '12px',
    cursor: 'pointer',
    color: '#9ca3af',
  },
  masterControls: {
    display: 'flex',
    gap: '12px',
    justifyContent: 'center',
  },
  resetBtn: {
    padding: '10px 20px',
    borderRadius: '8px',
    border: '1px solid #4b5563',
    backgroundColor: 'transparent',
    color: '#9ca3af',
    fontSize: '14px',
    cursor: 'pointer',
  },
  exportBtn: {
    padding: '10px 20px',
    borderRadius: '8px',
    border: 'none',
    background: 'linear-gradient(135deg, #818cf8, #c084fc)',
    color: '#fff',
    fontSize: '14px',
    fontWeight: 600,
    cursor: 'pointer',
  },
  newFileBtn: {
    padding: '10px 20px',
    borderRadius: '8px',
    border: '1px solid #4b5563',
    backgroundColor: 'transparent',
    color: '#9ca3af',
    fontSize: '14px',
    cursor: 'pointer',
  },
};

export default StemMixer;
