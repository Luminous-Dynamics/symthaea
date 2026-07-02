// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState, useMemo } from 'react';
import { X, Users, Lock, AlertCircle, CheckCircle, Info } from 'lucide-react';

interface Stem {
  id: string;
  name: string;
  type: string;
  duration: number;
}

interface Participant {
  id: string;
  name: string;
  color: string;
}

interface ContributionSplitsProps {
  projectId: string;
  stems: Stem[];
  participants: Participant[];
  onClose: () => void;
}

interface SplitEntry {
  userId: string;
  percent: number;
  role: string;
  locked: boolean;
}

export function ContributionSplits({
  projectId,
  stems,
  participants,
  onClose,
}: ContributionSplitsProps) {
  // Auto-calculate splits based on stem contributions
  const autoSplits = useMemo(() => {
    // In production, this would use actual contributor data from stems
    // For now, simulate based on participants
    const totalContributors = participants.length;
    const basePercent = Math.floor(100 / totalContributors);
    const remainder = 100 - (basePercent * totalContributors);

    return participants.map((p, i) => ({
      userId: p.id,
      percent: basePercent + (i === 0 ? remainder : 0),
      role: i === 0 ? 'Lead Producer' : 'Contributor',
      locked: false,
    }));
  }, [participants]);

  const [splits, setSplits] = useState<SplitEntry[]>(autoSplits);
  const [useAutoCalculation, setUseAutoCalculation] = useState(true);

  const totalPercent = splits.reduce((sum, s) => sum + s.percent, 0);
  const isValid = Math.abs(totalPercent - 100) < 0.01;

  const handlePercentChange = (userId: string, newPercent: number) => {
    if (useAutoCalculation) {
      setUseAutoCalculation(false);
    }

    setSplits(prev => prev.map(s =>
      s.userId === userId ? { ...s, percent: Math.max(0, Math.min(100, newPercent)) } : s
    ));
  };

  const handleRoleChange = (userId: string, role: string) => {
    setSplits(prev => prev.map(s =>
      s.userId === userId ? { ...s, role } : s
    ));
  };

  const handleLockToggle = (userId: string) => {
    setSplits(prev => prev.map(s =>
      s.userId === userId ? { ...s, locked: !s.locked } : s
    ));
  };

  const redistributeEvenly = () => {
    const unlocked = splits.filter(s => !s.locked);
    const lockedTotal = splits.filter(s => s.locked).reduce((sum, s) => sum + s.percent, 0);
    const remainingPercent = 100 - lockedTotal;
    const perPerson = remainingPercent / unlocked.length;

    setSplits(prev => prev.map(s =>
      s.locked ? s : { ...s, percent: perPerson }
    ));
  };

  const saveSplits = async () => {
    // Save to backend
    console.log('Saving splits:', splits);
  };

  return (
    <div className="w-80 border-l border-white/10 bg-gray-900 flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-white/10 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Users className="w-5 h-5 text-purple-400" />
          <span className="font-semibold">Contribution Splits</span>
        </div>
        <button onClick={onClose} className="p-1 hover:bg-white/10 rounded">
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Philosophy Note */}
      <div className="p-3 bg-purple-500/10 border-b border-white/10">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-purple-400 flex-shrink-0 mt-0.5" />
          <p className="text-xs text-muted-foreground">
            Splits are recorded on-chain. All contributors receive automatic royalties
            in perpetuity when this work generates revenue.
          </p>
        </div>
      </div>

      {/* Auto-calculation toggle */}
      <div className="p-4 border-b border-white/10">
        <label className="flex items-center justify-between cursor-pointer">
          <span className="text-sm">Auto-calculate from contributions</span>
          <button
            onClick={() => setUseAutoCalculation(!useAutoCalculation)}
            className={`relative w-10 h-5 rounded-full transition-colors ${
              useAutoCalculation ? 'bg-purple-500' : 'bg-gray-600'
            }`}
          >
            <div
              className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${
                useAutoCalculation ? 'left-5' : 'left-0.5'
              }`}
            />
          </button>
        </label>
      </div>

      {/* Splits List */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {splits.map((split) => {
          const participant = participants.find(p => p.id === split.userId);
          if (!participant) return null;

          return (
            <div key={split.userId} className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div
                    className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium"
                    style={{ backgroundColor: participant.color }}
                  >
                    {participant.name[0].toUpperCase()}
                  </div>
                  <span className="text-sm font-medium">{participant.name}</span>
                </div>
                <button
                  onClick={() => handleLockToggle(split.userId)}
                  className={`p-1 rounded ${split.locked ? 'text-yellow-400' : 'text-muted-foreground hover:text-white'}`}
                  title={split.locked ? 'Unlock' : 'Lock'}
                >
                  {split.locked ? <Lock className="w-4 h-4" /> : <Lock className="w-4 h-4 opacity-30" />}
                </button>
              </div>

              <div className="flex items-center gap-2">
                <input
                  type="number"
                  min="0"
                  max="100"
                  step="0.1"
                  value={split.percent}
                  onChange={(e) => handlePercentChange(split.userId, parseFloat(e.target.value) || 0)}
                  disabled={useAutoCalculation || split.locked}
                  className="w-20 px-2 py-1 bg-white/5 border border-white/10 rounded text-sm text-right disabled:opacity-50"
                />
                <span className="text-sm text-muted-foreground">%</span>

                <select
                  value={split.role}
                  onChange={(e) => handleRoleChange(split.userId, e.target.value)}
                  className="flex-1 px-2 py-1 bg-white/5 border border-white/10 rounded text-sm"
                >
                  <option value="Lead Producer">Lead Producer</option>
                  <option value="Producer">Producer</option>
                  <option value="Contributor">Contributor</option>
                  <option value="Vocalist">Vocalist</option>
                  <option value="Instrumentalist">Instrumentalist</option>
                  <option value="Engineer">Engineer</option>
                  <option value="Writer">Writer</option>
                </select>
              </div>

              {/* Contribution bar */}
              <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all"
                  style={{
                    width: `${split.percent}%`,
                    backgroundColor: participant.color,
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Total & Actions */}
      <div className="p-4 border-t border-white/10 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">Total</span>
          <div className="flex items-center gap-2">
            {isValid ? (
              <CheckCircle className="w-4 h-4 text-green-400" />
            ) : (
              <AlertCircle className="w-4 h-4 text-red-400" />
            )}
            <span className={`font-mono ${isValid ? 'text-green-400' : 'text-red-400'}`}>
              {totalPercent.toFixed(1)}%
            </span>
          </div>
        </div>

        {!isValid && (
          <p className="text-xs text-red-400">
            Splits must total exactly 100%
          </p>
        )}

        <div className="flex gap-2">
          <button
            onClick={redistributeEvenly}
            className="flex-1 py-2 text-sm bg-white/5 rounded-lg hover:bg-white/10 transition-colors"
          >
            Split Evenly
          </button>
          <button
            onClick={saveSplits}
            disabled={!isValid}
            className="flex-1 py-2 text-sm bg-purple-500 rounded-lg hover:bg-purple-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Save Splits
          </button>
        </div>
      </div>
    </div>
  );
}
