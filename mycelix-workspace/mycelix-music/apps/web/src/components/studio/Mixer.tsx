// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { cn } from '@/lib/utils';
import {
  Volume2,
  VolumeX,
  Headphones,
  MoreVertical,
  Settings,
  Waves,
  Activity,
  Minus,
  Plus,
  Link,
  Link2Off,
  RotateCcw,
} from 'lucide-react';
import { Button } from '../ui/design-system/Button';

// Types
interface MixerChannel {
  id: string;
  name: string;
  type: 'audio' | 'instrument' | 'bus' | 'master';
  color: string;
  volume: number; // dB, typically -60 to +6
  pan: number; // -1 to 1
  mute: boolean;
  solo: boolean;
  armed: boolean;
  sends: Send[];
  eq: EQBand[];
  compressor: CompressorSettings;
  meter: {
    left: number;
    right: number;
    peak: number;
  };
}

interface Send {
  id: string;
  name: string;
  amount: number;
  preFader: boolean;
}

interface EQBand {
  id: string;
  type: 'lowShelf' | 'highShelf' | 'peak' | 'lowPass' | 'highPass';
  frequency: number;
  gain: number;
  q: number;
  enabled: boolean;
}

interface CompressorSettings {
  enabled: boolean;
  threshold: number;
  ratio: number;
  attack: number;
  release: number;
  gain: number;
}

interface MixerProps {
  channels: MixerChannel[];
  onChannelChange: (channelId: string, updates: Partial<MixerChannel>) => void;
  masterChannel: MixerChannel;
  className?: string;
}

// Meter Component
function VUMeter({
  left,
  right,
  peak,
  height = 200
}: {
  left: number;
  right: number;
  peak: number;
  height?: number;
}) {
  const getColor = (level: number) => {
    if (level > -3) return 'bg-red-500';
    if (level > -6) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  const getHeight = (level: number) => {
    // Convert dB to percentage (assuming -60 to 0 range)
    const normalized = Math.max(0, Math.min(1, (level + 60) / 60));
    return normalized * 100;
  };

  return (
    <div className="flex gap-1 h-full">
      {/* Left channel */}
      <div className="w-2 bg-gray-800 rounded-sm overflow-hidden relative">
        <div
          className={cn('absolute bottom-0 w-full transition-all duration-75', getColor(left))}
          style={{ height: `${getHeight(left)}%` }}
        />
        {/* Peak indicator */}
        <div
          className="absolute w-full h-0.5 bg-red-400"
          style={{ bottom: `${getHeight(peak)}%` }}
        />
      </div>
      {/* Right channel */}
      <div className="w-2 bg-gray-800 rounded-sm overflow-hidden relative">
        <div
          className={cn('absolute bottom-0 w-full transition-all duration-75', getColor(right))}
          style={{ height: `${getHeight(right)}%` }}
        />
      </div>
    </div>
  );
}

// Fader Component
function Fader({
  value,
  onChange,
  min = -60,
  max = 6,
  label,
  disabled = false,
}: {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  label?: string;
  disabled?: boolean;
}) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const percentage = ((value - min) / (max - min)) * 100;

  const updateValue = useCallback(
    (clientY: number) => {
      if (!trackRef.current || disabled) return;
      const rect = trackRef.current.getBoundingClientRect();
      let newPercentage = ((rect.bottom - clientY) / rect.height) * 100;
      newPercentage = Math.max(0, Math.min(100, newPercentage));
      const newValue = min + (newPercentage / 100) * (max - min);
      onChange(Math.round(newValue * 10) / 10);
    },
    [disabled, max, min, onChange]
  );

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e: MouseEvent) => updateValue(e.clientY);
    const handleMouseUp = () => setIsDragging(false);

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, updateValue]);

  const handleDoubleClick = () => {
    if (!disabled) onChange(0); // Reset to unity gain
  };

  return (
    <div className="flex flex-col items-center gap-2">
      <div
        ref={trackRef}
        className="relative w-8 h-40 cursor-ns-resize"
        onMouseDown={(e) => {
          setIsDragging(true);
          updateValue(e.clientY);
        }}
        onDoubleClick={handleDoubleClick}
      >
        {/* Track */}
        <div className="absolute left-1/2 -translate-x-1/2 w-1 h-full bg-gray-700 rounded-full" />

        {/* Unity gain marker */}
        <div
          className="absolute left-0 right-0 h-px bg-gray-500"
          style={{ bottom: `${((0 - min) / (max - min)) * 100}%` }}
        />

        {/* Fill */}
        <div
          className="absolute left-1/2 -translate-x-1/2 w-1 bg-primary-500 rounded-full bottom-0"
          style={{ height: `${Math.max(0, percentage)}%` }}
        />

        {/* Handle */}
        <div
          className={cn(
            'absolute left-1/2 -translate-x-1/2 w-6 h-8 bg-gray-300 dark:bg-gray-600 rounded shadow-md border border-gray-400 dark:border-gray-500',
            'flex items-center justify-center',
            isDragging && 'bg-primary-400 dark:bg-primary-500'
          )}
          style={{ bottom: `calc(${percentage}% - 16px)` }}
        >
          <div className="w-4 h-px bg-gray-500 dark:bg-gray-400" />
        </div>
      </div>

      {label && (
        <span className="text-xs text-gray-400 font-mono">
          {value > 0 ? '+' : ''}{value.toFixed(1)}
        </span>
      )}
    </div>
  );
}

// Pan Knob Component
function PanKnob({
  value,
  onChange,
  disabled = false,
}: {
  value: number;
  onChange: (value: number) => void;
  disabled?: boolean;
}) {
  const [isDragging, setIsDragging] = useState(false);
  const startY = useRef(0);
  const startValue = useRef(0);

  const rotation = value * 135; // -135 to 135 degrees

  const handleMouseDown = (e: React.MouseEvent) => {
    if (disabled) return;
    setIsDragging(true);
    startY.current = e.clientY;
    startValue.current = value;
  };

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      const delta = (startY.current - e.clientY) / 100;
      const newValue = Math.max(-1, Math.min(1, startValue.current + delta));
      onChange(Math.round(newValue * 100) / 100);
    };

    const handleMouseUp = () => setIsDragging(false);

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, onChange]);

  const handleDoubleClick = () => {
    if (!disabled) onChange(0);
  };

  return (
    <div className="flex flex-col items-center gap-1">
      <div className="flex justify-between w-full text-[10px] text-gray-500 px-1">
        <span>L</span>
        <span>R</span>
      </div>
      <div
        className={cn(
          'w-8 h-8 rounded-full bg-gray-700 border-2 border-gray-600 relative cursor-ns-resize',
          isDragging && 'border-primary-500'
        )}
        onMouseDown={handleMouseDown}
        onDoubleClick={handleDoubleClick}
      >
        {/* Indicator line */}
        <div
          className="absolute top-1 left-1/2 w-0.5 h-3 bg-white rounded-full origin-bottom"
          style={{ transform: `translateX(-50%) rotate(${rotation}deg)` }}
        />
      </div>
      <span className="text-[10px] text-gray-400 font-mono">
        {value === 0 ? 'C' : value < 0 ? `L${Math.abs(Math.round(value * 100))}` : `R${Math.round(value * 100)}`}
      </span>
    </div>
  );
}

// Channel Strip Component
function ChannelStrip({
  channel,
  onChange,
  isSelected,
  onSelect,
}: {
  channel: MixerChannel;
  onChange: (updates: Partial<MixerChannel>) => void;
  isSelected: boolean;
  onSelect: () => void;
}) {
  return (
    <div
      className={cn(
        'flex flex-col bg-gray-900 border border-gray-800 rounded-lg p-2 min-w-[80px] transition-all',
        isSelected && 'ring-2 ring-primary-500 border-primary-500'
      )}
      onClick={onSelect}
    >
      {/* Channel header */}
      <div className="flex items-center justify-between mb-2">
        <div
          className="w-2 h-2 rounded-full"
          style={{ backgroundColor: channel.color }}
        />
        <Button variant="ghost" size="icon-sm" className="h-5 w-5">
          <MoreVertical className="h-3 w-3" />
        </Button>
      </div>

      {/* EQ indicator */}
      <div className="h-6 mb-2 flex items-center justify-center">
        <Activity className="h-4 w-4 text-gray-500" />
      </div>

      {/* Sends */}
      <div className="space-y-1 mb-2">
        {channel.sends.slice(0, 2).map((send) => (
          <div key={send.id} className="flex items-center gap-1">
            <div className="flex-1 h-1 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500"
                style={{ width: `${send.amount}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Pan */}
      <div className="flex justify-center mb-2">
        <PanKnob
          value={channel.pan}
          onChange={(pan) => onChange({ pan })}
        />
      </div>

      {/* Mute/Solo/Arm */}
      <div className="flex gap-1 justify-center mb-2">
        <button
          onClick={(e) => {
            e.stopPropagation();
            onChange({ mute: !channel.mute });
          }}
          className={cn(
            'w-6 h-5 rounded text-[10px] font-bold transition-colors',
            channel.mute
              ? 'bg-red-600 text-white'
              : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
          )}
        >
          M
        </button>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onChange({ solo: !channel.solo });
          }}
          className={cn(
            'w-6 h-5 rounded text-[10px] font-bold transition-colors',
            channel.solo
              ? 'bg-yellow-500 text-black'
              : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
          )}
        >
          S
        </button>
        {channel.type !== 'master' && channel.type !== 'bus' && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onChange({ armed: !channel.armed });
            }}
            className={cn(
              'w-6 h-5 rounded text-[10px] font-bold transition-colors',
              channel.armed
                ? 'bg-red-500 text-white animate-pulse'
                : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
            )}
          >
            R
          </button>
        )}
      </div>

      {/* Meter and Fader */}
      <div className="flex gap-2 justify-center">
        <VUMeter
          left={channel.meter.left}
          right={channel.meter.right}
          peak={channel.meter.peak}
          height={160}
        />
        <Fader
          value={channel.volume}
          onChange={(volume) => onChange({ volume })}
          label={channel.volume.toFixed(1)}
        />
      </div>

      {/* Channel name */}
      <div className="mt-2 text-center">
        <input
          type="text"
          value={channel.name}
          onChange={(e) => onChange({ name: e.target.value })}
          className="w-full bg-transparent text-center text-xs text-gray-300 focus:outline-none focus:bg-gray-800 rounded px-1"
        />
      </div>

      {/* Output indicator */}
      <div className="mt-1 text-center text-[10px] text-gray-500">
        {channel.type === 'master' ? 'OUT' : 'Master'}
      </div>
    </div>
  );
}

// Master Channel Component
function MasterChannel({
  channel,
  onChange,
}: {
  channel: MixerChannel;
  onChange: (updates: Partial<MixerChannel>) => void;
}) {
  return (
    <div className="flex flex-col bg-gray-800 border-2 border-gray-700 rounded-lg p-3 min-w-[100px]">
      <div className="text-center text-sm font-bold text-white mb-3">MASTER</div>

      {/* Meters */}
      <div className="flex gap-2 justify-center mb-3">
        <VUMeter
          left={channel.meter.left}
          right={channel.meter.right}
          peak={channel.meter.peak}
          height={180}
        />
      </div>

      {/* Fader */}
      <div className="flex justify-center mb-3">
        <Fader
          value={channel.volume}
          onChange={(volume) => onChange({ volume })}
        />
      </div>

      {/* Volume display */}
      <div className="text-center text-sm font-mono text-gray-300 mb-2">
        {channel.volume > 0 ? '+' : ''}{channel.volume.toFixed(1)} dB
      </div>

      {/* Mute */}
      <div className="flex justify-center">
        <button
          onClick={() => onChange({ mute: !channel.mute })}
          className={cn(
            'px-3 py-1 rounded text-xs font-bold transition-colors',
            channel.mute
              ? 'bg-red-600 text-white'
              : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
          )}
        >
          {channel.mute ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
        </button>
      </div>
    </div>
  );
}

// Main Mixer Component
export function Mixer({ channels, onChannelChange, masterChannel, className }: MixerProps) {
  const [selectedChannel, setSelectedChannel] = useState<string | null>(null);
  const [linkedChannels, setLinkedChannels] = useState<Set<string>>(new Set());
  const scrollRef = useRef<HTMLDivElement>(null);

  const handleChannelChange = useCallback(
    (channelId: string, updates: Partial<MixerChannel>) => {
      onChannelChange(channelId, updates);

      // Handle linked channels
      if (linkedChannels.has(channelId) && updates.volume !== undefined) {
        linkedChannels.forEach((id) => {
          if (id !== channelId) {
            onChannelChange(id, { volume: updates.volume });
          }
        });
      }
    },
    [linkedChannels, onChannelChange]
  );

  const toggleChannelLink = (channelId: string) => {
    setLinkedChannels((prev) => {
      const next = new Set(prev);
      if (next.has(channelId)) {
        next.delete(channelId);
      } else {
        next.add(channelId);
      }
      return next;
    });
  };

  return (
    <div className={cn('flex flex-col bg-gray-950 rounded-xl overflow-hidden', className)}>
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-800">
        <div className="flex items-center gap-2">
          <Waves className="h-5 w-5 text-primary-500" />
          <span className="text-sm font-medium text-white">Mixer</span>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="icon-sm">
            <Link className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="icon-sm">
            <RotateCcw className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="icon-sm">
            <Settings className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Channel strips */}
      <div className="flex-1 overflow-hidden">
        <div
          ref={scrollRef}
          className="flex gap-2 p-4 overflow-x-auto h-full"
        >
          {/* Regular channels */}
          {channels.map((channel) => (
            <ChannelStrip
              key={channel.id}
              channel={channel}
              onChange={(updates) => handleChannelChange(channel.id, updates)}
              isSelected={selectedChannel === channel.id}
              onSelect={() => setSelectedChannel(channel.id)}
            />
          ))}

          {/* Separator */}
          <div className="w-px bg-gray-700 mx-2" />

          {/* Master channel */}
          <MasterChannel
            channel={masterChannel}
            onChange={(updates) => onChannelChange(masterChannel.id, updates)}
          />
        </div>
      </div>

      {/* Bottom section - EQ/Effects for selected channel */}
      {selectedChannel && (
        <div className="border-t border-gray-800 p-4">
          <ChannelDetail
            channel={channels.find((c) => c.id === selectedChannel)!}
            onChange={(updates) => handleChannelChange(selectedChannel, updates)}
          />
        </div>
      )}
    </div>
  );
}

// Channel Detail Panel
function ChannelDetail({
  channel,
  onChange,
}: {
  channel: MixerChannel;
  onChange: (updates: Partial<MixerChannel>) => void;
}) {
  const [activeTab, setActiveTab] = useState<'eq' | 'comp' | 'sends'>('eq');

  return (
    <div className="flex flex-col gap-3">
      {/* Tabs */}
      <div className="flex gap-2">
        {(['eq', 'comp', 'sends'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={cn(
              'px-3 py-1 rounded text-xs font-medium transition-colors',
              activeTab === tab
                ? 'bg-primary-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            )}
          >
            {tab.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="bg-gray-900 rounded-lg p-3 min-h-[100px]">
        {activeTab === 'eq' && (
          <EQPanel eq={channel.eq} onChange={(eq) => onChange({ eq })} />
        )}
        {activeTab === 'comp' && (
          <CompressorPanel
            settings={channel.compressor}
            onChange={(compressor) => onChange({ compressor })}
          />
        )}
        {activeTab === 'sends' && (
          <SendsPanel
            sends={channel.sends}
            onChange={(sends) => onChange({ sends })}
          />
        )}
      </div>
    </div>
  );
}

// EQ Panel
function EQPanel({
  eq,
  onChange,
}: {
  eq: EQBand[];
  onChange: (eq: EQBand[]) => void;
}) {
  return (
    <div className="flex gap-4 items-end">
      {eq.map((band, index) => (
        <div key={band.id} className="flex flex-col items-center gap-2">
          <div className="text-[10px] text-gray-400 uppercase">{band.type}</div>
          <input
            type="range"
            min="-12"
            max="12"
            step="0.5"
            value={band.gain}
            onChange={(e) => {
              const newEq = [...eq];
              newEq[index] = { ...band, gain: parseFloat(e.target.value) };
              onChange(newEq);
            }}
            className="w-16 h-2 appearance-none bg-gray-700 rounded-full"
            style={{ writingMode: 'vertical-lr' as any }}
          />
          <div className="text-[10px] text-gray-500 font-mono">
            {band.gain > 0 ? '+' : ''}{band.gain.toFixed(1)}
          </div>
          <div className="text-[10px] text-gray-400">{band.frequency}Hz</div>
        </div>
      ))}
    </div>
  );
}

// Compressor Panel
function CompressorPanel({
  settings,
  onChange,
}: {
  settings: CompressorSettings;
  onChange: (settings: CompressorSettings) => void;
}) {
  return (
    <div className="grid grid-cols-5 gap-4">
      {[
        { key: 'threshold', label: 'Threshold', min: -60, max: 0, unit: 'dB' },
        { key: 'ratio', label: 'Ratio', min: 1, max: 20, unit: ':1' },
        { key: 'attack', label: 'Attack', min: 0.1, max: 100, unit: 'ms' },
        { key: 'release', label: 'Release', min: 10, max: 1000, unit: 'ms' },
        { key: 'gain', label: 'Gain', min: 0, max: 24, unit: 'dB' },
      ].map(({ key, label, min, max, unit }) => (
        <div key={key} className="flex flex-col items-center gap-1">
          <div className="text-[10px] text-gray-400">{label}</div>
          <input
            type="range"
            min={min}
            max={max}
            step={(max - min) / 100}
            value={settings[key as keyof CompressorSettings] as number}
            onChange={(e) =>
              onChange({ ...settings, [key]: parseFloat(e.target.value) })
            }
            className="w-full h-1 appearance-none bg-gray-700 rounded-full"
          />
          <div className="text-[10px] text-gray-500 font-mono">
            {(settings[key as keyof CompressorSettings] as number).toFixed(1)}{unit}
          </div>
        </div>
      ))}
    </div>
  );
}

// Sends Panel
function SendsPanel({
  sends,
  onChange,
}: {
  sends: Send[];
  onChange: (sends: Send[]) => void;
}) {
  return (
    <div className="space-y-2">
      {sends.map((send, index) => (
        <div key={send.id} className="flex items-center gap-3">
          <span className="text-xs text-gray-400 w-16">{send.name}</span>
          <input
            type="range"
            min="0"
            max="100"
            value={send.amount}
            onChange={(e) => {
              const newSends = [...sends];
              newSends[index] = { ...send, amount: parseFloat(e.target.value) };
              onChange(newSends);
            }}
            className="flex-1 h-1 appearance-none bg-gray-700 rounded-full"
          />
          <span className="text-[10px] text-gray-500 font-mono w-8">
            {send.amount.toFixed(0)}%
          </span>
          <button
            onClick={() => {
              const newSends = [...sends];
              newSends[index] = { ...send, preFader: !send.preFader };
              onChange(newSends);
            }}
            className={cn(
              'text-[10px] px-1.5 py-0.5 rounded',
              send.preFader ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-400'
            )}
          >
            {send.preFader ? 'PRE' : 'POST'}
          </button>
        </div>
      ))}
    </div>
  );
}

export type { MixerChannel, Send, EQBand, CompressorSettings };
