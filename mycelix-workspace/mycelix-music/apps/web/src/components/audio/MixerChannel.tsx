// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mixer Channel Component
 *
 * Complete channel strip with:
 * - Volume fader with meter
 * - Pan knob
 * - Mute/Solo/Record buttons
 * - Insert effect slots
 * - Send controls
 * - Input/Output routing
 */

import React, { memo, useCallback } from 'react';
import { Fader } from './Fader';
import { Knob } from './Knob';
import { VUMeter } from './VUMeter';

// ==================== Types ====================

export interface ChannelState {
  id: string;
  name: string;
  color: string;
  volume: number;       // dB (-70 to +6)
  pan: number;          // -1 to 1
  mute: boolean;
  solo: boolean;
  record: boolean;
  meterLevel: number;   // dB for meter
  meterPeak: number;
  sends: SendState[];
  inserts: InsertSlot[];
}

export interface SendState {
  id: string;
  name: string;
  level: number;        // dB
  enabled: boolean;
  preFader: boolean;
}

export interface InsertSlot {
  id: string;
  pluginName: string | null;
  enabled: boolean;
}

export interface MixerChannelProps {
  channel: ChannelState;
  onVolumeChange: (volume: number) => void;
  onPanChange: (pan: number) => void;
  onMuteToggle: () => void;
  onSoloToggle: () => void;
  onRecordToggle: () => void;
  onSendChange?: (sendId: string, level: number) => void;
  onInsertClick?: (slotIndex: number) => void;
  onNameChange?: (name: string) => void;
  onColorChange?: (color: string) => void;
  /** Show sends section */
  showSends?: boolean;
  /** Show inserts section */
  showInserts?: boolean;
  /** Number of insert slots */
  insertSlots?: number;
  /** Compact mode */
  compact?: boolean;
  /** Selected state */
  selected?: boolean;
  /** Class name */
  className?: string;
}

// ==================== Sub-components ====================

interface ChannelButtonProps {
  label: string;
  active: boolean;
  activeColor: string;
  onClick: () => void;
  size?: 'sm' | 'md';
}

const ChannelButton = memo(function ChannelButton({
  label,
  active,
  activeColor,
  onClick,
  size = 'md',
}: ChannelButtonProps) {
  return (
    <button
      onClick={onClick}
      style={{
        width: size === 'sm' ? 20 : 28,
        height: size === 'sm' ? 20 : 28,
        border: 'none',
        borderRadius: 4,
        background: active ? activeColor : 'var(--color-surface, #1F1F1F)',
        color: active ? '#000' : 'var(--color-text-muted, #A1A1A1)',
        fontSize: size === 'sm' ? 9 : 11,
        fontWeight: 600,
        cursor: 'pointer',
        transition: 'all 150ms ease',
        boxShadow: active ? `0 0 8px ${activeColor}` : 'none',
      }}
    >
      {label}
    </button>
  );
});

// ==================== Main Component ====================

export const MixerChannel = memo(function MixerChannel({
  channel,
  onVolumeChange,
  onPanChange,
  onMuteToggle,
  onSoloToggle,
  onRecordToggle,
  onSendChange,
  onInsertClick,
  onNameChange,
  onColorChange,
  showSends = true,
  showInserts = true,
  insertSlots = 4,
  compact = false,
  selected = false,
  className,
}: MixerChannelProps) {
  const channelWidth = compact ? 60 : 80;
  const faderHeight = compact ? 150 : 200;

  const handleNameEdit = useCallback(() => {
    if (!onNameChange) return;
    const newName = prompt('Channel name:', channel.name);
    if (newName && newName !== channel.name) {
      onNameChange(newName);
    }
  }, [channel.name, onNameChange]);

  return (
    <div
      className={className}
      style={{
        display: 'flex',
        flexDirection: 'column',
        width: channelWidth,
        background: selected
          ? 'var(--color-surface-active, #333)'
          : 'var(--color-surface, #1F1F1F)',
        borderRadius: 8,
        padding: compact ? 8 : 12,
        gap: compact ? 8 : 12,
        border: `1px solid ${selected ? channel.color : 'var(--color-surface-border, #333)'}`,
        transition: 'border-color 150ms',
      }}
    >
      {/* Color indicator */}
      <div
        style={{
          height: 4,
          background: channel.color,
          borderRadius: 2,
          cursor: onColorChange ? 'pointer' : 'default',
        }}
        onClick={() => {
          if (!onColorChange) return;
          const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'];
          const currentIndex = colors.indexOf(channel.color);
          onColorChange(colors[(currentIndex + 1) % colors.length]);
        }}
      />

      {/* Channel name */}
      <div
        onClick={handleNameEdit}
        style={{
          fontSize: 11,
          fontWeight: 500,
          color: 'var(--color-text, #FAFAFA)',
          textAlign: 'center',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
          cursor: onNameChange ? 'pointer' : 'default',
          padding: '2px 4px',
          borderRadius: 4,
          background: 'var(--color-background-sunken, #080808)',
        }}
        title={channel.name}
      >
        {channel.name}
      </div>

      {/* Insert slots */}
      {showInserts && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {Array.from({ length: insertSlots }).map((_, i) => {
            const insert = channel.inserts[i];
            return (
              <div
                key={i}
                onClick={() => onInsertClick?.(i)}
                style={{
                  height: 18,
                  background: insert?.pluginName
                    ? insert.enabled
                      ? 'var(--color-primary-muted, rgba(139, 92, 246, 0.2))'
                      : 'var(--color-surface-hover, #2A2A2A)'
                    : 'var(--color-background-sunken, #080808)',
                  borderRadius: 3,
                  fontSize: 9,
                  color: insert?.pluginName
                    ? 'var(--color-text, #FAFAFA)'
                    : 'var(--color-text-subtle, #6B6B6B)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  padding: '0 4px',
                }}
              >
                {insert?.pluginName || '—'}
              </div>
            );
          })}
        </div>
      )}

      {/* Sends */}
      {showSends && channel.sends.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {channel.sends.slice(0, compact ? 2 : 4).map(send => (
            <div
              key={send.id}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 4,
              }}
            >
              <span
                style={{
                  fontSize: 9,
                  color: 'var(--color-text-subtle, #6B6B6B)',
                  width: 20,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                }}
              >
                {send.name}
              </span>
              <Knob
                value={send.level}
                min={-70}
                max={6}
                onChange={(level) => onSendChange?.(send.id, level)}
                size={compact ? 20 : 24}
                variant="minimal"
                color={send.enabled ? 'var(--color-secondary, #06B6D4)' : 'var(--color-text-subtle, #6B6B6B)'}
                showValue={false}
              />
            </div>
          ))}
        </div>
      )}

      {/* Pan control */}
      <div style={{ display: 'flex', justifyContent: 'center' }}>
        <Knob
          value={channel.pan}
          min={-1}
          max={1}
          step={0.01}
          defaultValue={0}
          onChange={onPanChange}
          size={compact ? 32 : 40}
          label="PAN"
          variant="modern"
          bipolar
          formatValue={(v) => {
            if (Math.abs(v) < 0.01) return 'C';
            return v < 0 ? `L${Math.round(Math.abs(v) * 100)}` : `R${Math.round(v * 100)}`;
          }}
        />
      </div>

      {/* Fader and meter section */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          gap: 4,
          flex: 1,
        }}
      >
        <VUMeter
          level={channel.meterLevel}
          peak={channel.meterPeak}
          height={faderHeight}
          width={8}
          orientation="vertical"
          variant="segmented"
          segments={compact ? 20 : 30}
          showScale={false}
        />

        <Fader
          value={channel.volume}
          min={-70}
          max={6}
          step={0.1}
          defaultValue={0}
          onChange={onVolumeChange}
          orientation="vertical"
          length={faderHeight}
          trackWidth={6}
          variant="studio"
          showScale={!compact}
          scaleMarks={compact ? [0, -12, -48] : [6, 0, -6, -12, -24, -48, -70]}
        />
      </div>

      {/* Channel buttons */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          gap: 4,
        }}
      >
        <ChannelButton
          label="M"
          active={channel.mute}
          activeColor="#EF4444"
          onClick={onMuteToggle}
          size={compact ? 'sm' : 'md'}
        />
        <ChannelButton
          label="S"
          active={channel.solo}
          activeColor="#FBBF24"
          onClick={onSoloToggle}
          size={compact ? 'sm' : 'md'}
        />
        <ChannelButton
          label="R"
          active={channel.record}
          activeColor="#EF4444"
          onClick={onRecordToggle}
          size={compact ? 'sm' : 'md'}
        />
      </div>

      {/* Volume value */}
      <div
        style={{
          fontSize: 10,
          fontFamily: 'var(--font-family-mono, monospace)',
          color: 'var(--color-text-muted, #A1A1A1)',
          textAlign: 'center',
          background: 'var(--color-background-sunken, #080808)',
          padding: '2px 4px',
          borderRadius: 4,
        }}
      >
        {channel.volume > -70 ? `${channel.volume.toFixed(1)} dB` : '-∞ dB'}
      </div>
    </div>
  );
});

// ==================== Mixer Console ====================

export interface MixerConsoleProps {
  channels: ChannelState[];
  masterChannel?: ChannelState;
  onChannelUpdate: (channelId: string, updates: Partial<ChannelState>) => void;
  selectedChannelId?: string;
  onChannelSelect?: (channelId: string) => void;
  showMaster?: boolean;
  compact?: boolean;
  className?: string;
}

export const MixerConsole = memo(function MixerConsole({
  channels,
  masterChannel,
  onChannelUpdate,
  selectedChannelId,
  onChannelSelect,
  showMaster = true,
  compact = false,
  className,
}: MixerConsoleProps) {
  const handleUpdate = useCallback((channelId: string, key: keyof ChannelState, value: unknown) => {
    onChannelUpdate(channelId, { [key]: value });
  }, [onChannelUpdate]);

  return (
    <div
      className={className}
      style={{
        display: 'flex',
        gap: compact ? 4 : 8,
        padding: compact ? 8 : 16,
        background: 'var(--color-background, #0F0F0F)',
        borderRadius: 12,
        overflowX: 'auto',
      }}
    >
      {/* Regular channels */}
      {channels.map(channel => (
        <MixerChannel
          key={channel.id}
          channel={channel}
          onVolumeChange={(v) => handleUpdate(channel.id, 'volume', v)}
          onPanChange={(v) => handleUpdate(channel.id, 'pan', v)}
          onMuteToggle={() => handleUpdate(channel.id, 'mute', !channel.mute)}
          onSoloToggle={() => handleUpdate(channel.id, 'solo', !channel.solo)}
          onRecordToggle={() => handleUpdate(channel.id, 'record', !channel.record)}
          onNameChange={(v) => handleUpdate(channel.id, 'name', v)}
          onColorChange={(v) => handleUpdate(channel.id, 'color', v)}
          selected={channel.id === selectedChannelId}
          compact={compact}
        />
      ))}

      {/* Divider */}
      {showMaster && masterChannel && (
        <div
          style={{
            width: 1,
            background: 'var(--color-surface-border, #333)',
            margin: '0 8px',
          }}
        />
      )}

      {/* Master channel */}
      {showMaster && masterChannel && (
        <MixerChannel
          channel={masterChannel}
          onVolumeChange={(v) => handleUpdate(masterChannel.id, 'volume', v)}
          onPanChange={(v) => handleUpdate(masterChannel.id, 'pan', v)}
          onMuteToggle={() => handleUpdate(masterChannel.id, 'mute', !masterChannel.mute)}
          onSoloToggle={() => handleUpdate(masterChannel.id, 'solo', !masterChannel.solo)}
          onRecordToggle={() => handleUpdate(masterChannel.id, 'record', !masterChannel.record)}
          showSends={false}
          compact={compact}
        />
      )}
    </div>
  );
});

export default MixerChannel;
