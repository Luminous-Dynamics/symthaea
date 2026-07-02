// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState } from 'react';
import { useAccount } from 'wagmi';
import {
  Heart,
  Sparkles,
  TreeDeciduous,
  Trees,
  Leaf,
  CheckCircle,
  Clock,
  Users,
  ChevronRight,
} from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';

export type PatronageTier = 'seedling' | 'sapling' | 'grove' | 'ancient';

interface PatronageCardProps {
  artistId: string;
  artistName: string;
  artistImage: string;
  currentTier?: PatronageTier;
  patronCount: number;
  onBeginPatronage: (tier: PatronageTier) => void;
}

const tierInfo: Record<PatronageTier, {
  name: string;
  icon: React.ReactNode;
  period: string;
  commitment: string;
  color: string;
  benefits: string[];
}> = {
  seedling: {
    name: 'Seedling',
    icon: <Leaf className="w-5 h-5" />,
    period: 'Monthly',
    commitment: '$5/month',
    color: 'from-green-500 to-emerald-500',
    benefits: [
      'Early access to new releases',
      'Exclusive behind-the-scenes content',
      'Patron-only community access',
    ],
  },
  sapling: {
    name: 'Sapling',
    icon: <TreeDeciduous className="w-5 h-5" />,
    period: 'Quarterly',
    commitment: '$12/quarter',
    color: 'from-emerald-500 to-teal-500',
    benefits: [
      'All Seedling benefits',
      'Governance voting rights',
      'Monthly live streams',
      'Name in credits',
    ],
  },
  grove: {
    name: 'Grove',
    icon: <Trees className="w-5 h-5" />,
    period: 'Annual',
    commitment: '$40/year',
    color: 'from-teal-500 to-cyan-500',
    benefits: [
      'All Sapling benefits',
      'Revenue share (micro-royalties)',
      'Co-creation opportunities',
      'Physical merchandise',
    ],
  },
  ancient: {
    name: 'Ancient',
    icon: <Sparkles className="w-5 h-5" />,
    period: 'Multi-year',
    commitment: '$100/2 years',
    color: 'from-purple-500 to-pink-500',
    benefits: [
      'All Grove benefits',
      'Advisory role',
      'Legacy patron recognition',
      'Direct artist access',
      'Limited edition collectibles',
    ],
  },
};

export function PatronageCard({
  artistId,
  artistName,
  artistImage,
  currentTier,
  patronCount,
  onBeginPatronage,
}: PatronageCardProps) {
  const { isConnected } = useAccount();
  const [selectedTier, setSelectedTier] = useState<PatronageTier | null>(null);
  const [isExpanded, setIsExpanded] = useState(false);

  const tiers: PatronageTier[] = ['seedling', 'sapling', 'grove', 'ancient'];

  const handlePatronage = (tier: PatronageTier) => {
    if (!isConnected) {
      // Prompt wallet connection
      return;
    }
    onBeginPatronage(tier);
  };

  return (
    <div className="bg-white/5 rounded-2xl overflow-hidden">
      {/* Header */}
      <div className="p-6 border-b border-white/10">
        <div className="flex items-center gap-4">
          <div className="relative w-16 h-16 rounded-full overflow-hidden">
            <Image
              src={artistImage}
              alt={artistName}
              fill
              className="object-cover"
            />
          </div>
          <div className="flex-1">
            <h3 className="text-xl font-bold">{artistName}</h3>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Users className="w-4 h-4" />
              <span>{patronCount.toLocaleString()} patrons</span>
            </div>
          </div>
          {currentTier && (
            <div className={`px-3 py-1.5 rounded-full bg-gradient-to-r ${tierInfo[currentTier].color} text-sm font-medium flex items-center gap-2`}>
              {tierInfo[currentTier].icon}
              {tierInfo[currentTier].name} Patron
            </div>
          )}
        </div>
      </div>

      {/* Philosophy Note */}
      <div className="px-6 py-4 bg-gradient-to-r from-purple-500/10 to-pink-500/10 border-b border-white/10">
        <p className="text-sm text-muted-foreground">
          <Heart className="w-4 h-4 inline mr-2 text-pink-400" />
          Patronage is a relationship, not an asset. Your support directly funds the
          artist&apos;s creative journey while earning you Spores for governance.
        </p>
      </div>

      {/* Tier Selection */}
      <div className="p-6">
        <h4 className="font-semibold mb-4">Choose Your Commitment</h4>

        <div className="space-y-3">
          {tiers.map((tier) => {
            const info = tierInfo[tier];
            const isSelected = selectedTier === tier;
            const isCurrent = currentTier === tier;

            return (
              <button
                key={tier}
                onClick={() => setSelectedTier(isSelected ? null : tier)}
                className={`w-full p-4 rounded-xl border transition-all text-left ${
                  isSelected
                    ? 'border-purple-500 bg-purple-500/10'
                    : isCurrent
                    ? 'border-green-500/50 bg-green-500/5'
                    : 'border-white/10 hover:border-white/20 bg-white/5'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className={`w-10 h-10 rounded-full bg-gradient-to-r ${info.color} flex items-center justify-center`}>
                      {info.icon}
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{info.name}</span>
                        {isCurrent && (
                          <span className="text-xs text-green-400 flex items-center gap-1">
                            <CheckCircle className="w-3 h-3" /> Current
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-3 text-sm text-muted-foreground">
                        <span>{info.commitment}</span>
                        <span className="flex items-center gap-1">
                          <Clock className="w-3 h-3" /> {info.period}
                        </span>
                      </div>
                    </div>
                  </div>
                  <ChevronRight className={`w-5 h-5 text-muted-foreground transition-transform ${isSelected ? 'rotate-90' : ''}`} />
                </div>

                {/* Expanded Benefits */}
                {isSelected && (
                  <div className="mt-4 pt-4 border-t border-white/10">
                    <p className="text-sm font-medium mb-3">Benefits:</p>
                    <ul className="space-y-2">
                      {info.benefits.map((benefit, i) => (
                        <li key={i} className="flex items-start gap-2 text-sm text-muted-foreground">
                          <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                          {benefit}
                        </li>
                      ))}
                    </ul>

                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handlePatronage(tier);
                      }}
                      disabled={isCurrent}
                      className={`mt-4 w-full py-3 rounded-lg font-medium transition-colors ${
                        isCurrent
                          ? 'bg-white/10 text-muted-foreground cursor-not-allowed'
                          : `bg-gradient-to-r ${info.color} hover:opacity-90`
                      }`}
                    >
                      {isCurrent ? 'Current Tier' : `Become a ${info.name}`}
                    </button>
                  </div>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Footer - What happens with your support */}
      <div className="p-6 bg-white/5 border-t border-white/10">
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex items-center justify-between w-full text-sm"
        >
          <span className="text-muted-foreground">Where does my support go?</span>
          <ChevronRight className={`w-4 h-4 text-muted-foreground transition-transform ${isExpanded ? 'rotate-90' : ''}`} />
        </button>

        {isExpanded && (
          <div className="mt-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm">Artist</span>
              <span className="text-sm font-medium text-purple-400">85%</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">Collaborators</span>
              <span className="text-sm font-medium text-pink-400">10%</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">Ecosystem Treasury</span>
              <span className="text-sm font-medium text-green-400">5%</span>
            </div>
            <p className="text-xs text-muted-foreground pt-2 border-t border-white/10">
              Treasury funds support emerging artists, education, and platform development.
              No extraction - all value stays in the ecosystem.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
