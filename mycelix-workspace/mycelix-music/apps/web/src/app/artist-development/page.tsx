// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState } from 'react';
import { useAuth } from '@/hooks/useAuth';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { Player } from '@/components/player/Player';
import {
  Sprout,
  TreeDeciduous,
  Target,
  Users,
  Award,
  TrendingUp,
  Clock,
  CheckCircle,
  Circle,
  ChevronRight,
  Heart,
  DollarSign,
  Music2,
  Headphones,
  Star,
  Gift,
  Sparkles,
  ArrowRight,
} from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';

interface Milestone {
  id: string;
  title: string;
  description: string;
  target: number;
  current: number;
  reward: string;
  rewardAmount: number;
  isCompleted: boolean;
  icon: React.ReactNode;
}

interface Mentor {
  id: string;
  name: string;
  avatar: string;
  genre: string;
  followers: number;
  specialties: string[];
  availability: 'open' | 'limited' | 'closed';
  rating: number;
  menteeCount: number;
}

interface Residency {
  id: string;
  title: string;
  description: string;
  duration: string;
  funding: number;
  requirements: string[];
  deadline: Date;
  applicants: number;
  votes: number;
  status: 'open' | 'voting' | 'active' | 'completed';
}

const mockMilestones: Milestone[] = [
  {
    id: '1',
    title: 'First 100 Listeners',
    description: 'Reach 100 unique listeners on your tracks',
    target: 100,
    current: 67,
    reward: 'Emerging Artist Badge',
    rewardAmount: 25,
    isCompleted: false,
    icon: <Headphones className="w-5 h-5" />,
  },
  {
    id: '2',
    title: 'Community Builder',
    description: 'Gain 50 followers on your profile',
    target: 50,
    current: 50,
    reward: 'Promotion boost',
    rewardAmount: 50,
    isCompleted: true,
    icon: <Users className="w-5 h-5" />,
  },
  {
    id: '3',
    title: 'Consistent Creator',
    description: 'Release 5 tracks in a single month',
    target: 5,
    current: 3,
    reward: 'Studio credits',
    rewardAmount: 100,
    isCompleted: false,
    icon: <Music2 className="w-5 h-5" />,
  },
  {
    id: '4',
    title: 'Collaboration Star',
    description: 'Collaborate with 3 other artists',
    target: 3,
    current: 1,
    reward: 'Featured placement',
    rewardAmount: 75,
    isCompleted: false,
    icon: <Heart className="w-5 h-5" />,
  },
];

const mockMentors: Mentor[] = [
  {
    id: '1',
    name: 'Ethereal Waves',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=ethereal',
    genre: 'Ambient',
    followers: 12500,
    specialties: ['Production', 'Sound Design', 'Mixing'],
    availability: 'open',
    rating: 4.9,
    menteeCount: 8,
  },
  {
    id: '2',
    name: 'Neon Pulse',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=neon',
    genre: 'Synthwave',
    followers: 28000,
    specialties: ['Synthesis', 'Arrangement', 'Branding'],
    availability: 'limited',
    rating: 4.8,
    menteeCount: 12,
  },
  {
    id: '3',
    name: 'Forest Protocol',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=forest',
    genre: 'Organic Electronic',
    followers: 8900,
    specialties: ['Field Recording', 'Live Performance', 'Sustainability'],
    availability: 'open',
    rating: 4.7,
    menteeCount: 5,
  },
];

const mockResidencies: Residency[] = [
  {
    id: '1',
    title: 'Emerging Voices Residency',
    description: 'A 3-month program for new artists to develop their debut EP with full support.',
    duration: '3 months',
    funding: 5000,
    requirements: ['Less than 1000 followers', 'At least 3 released tracks', 'Active for 6+ months'],
    deadline: new Date('2024-02-15'),
    applicants: 45,
    votes: 1234,
    status: 'open',
  },
  {
    id: '2',
    title: 'Collaboration Lab',
    description: 'Bring together 5 artists for a collaborative album project.',
    duration: '2 months',
    funding: 3000,
    requirements: ['Open to all artists', 'Must commit to weekly sessions', 'Collaborative spirit'],
    deadline: new Date('2024-01-30'),
    applicants: 78,
    votes: 2567,
    status: 'voting',
  },
];

export default function ArtistDevelopmentPage() {
  const { authenticated, user } = useAuth();
  const [activeTab, setActiveTab] = useState<'milestones' | 'mentors' | 'residencies'>('milestones');

  // Calculate overall progress
  const totalProgress = mockMilestones.reduce((sum, m) => sum + (m.current / m.target), 0) / mockMilestones.length * 100;
  const completedMilestones = mockMilestones.filter(m => m.isCompleted).length;

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />

      <main className="ml-64 pb-24">
        <Header />

        <div className="px-6 py-4">
          {/* Hero Section */}
          <div className="relative rounded-2xl overflow-hidden mb-8 bg-gradient-to-br from-green-600 via-emerald-600 to-teal-600">
            <div className="absolute inset-0 bg-black/20" />
            <div className="relative p-8 md:p-12">
              <div className="flex items-center gap-3 mb-4">
                <Sprout className="w-8 h-8" />
                <span className="text-sm font-medium uppercase tracking-wider opacity-80">
                  Artist Development
                </span>
              </div>
              <h1 className="text-4xl md:text-5xl font-bold mb-4">
                Grow Your Artistry
              </h1>
              <p className="text-lg opacity-80 max-w-xl mb-6">
                The mycelium network supports emerging artists through mentorship,
                milestone-based funding, and community-backed residencies.
              </p>

              {/* Progress Summary */}
              <div className="flex items-center gap-8">
                <div>
                  <p className="text-3xl font-bold">{completedMilestones}/{mockMilestones.length}</p>
                  <p className="text-sm opacity-80">Milestones completed</p>
                </div>
                <div>
                  <p className="text-3xl font-bold">${125}</p>
                  <p className="text-sm opacity-80">Earned from development</p>
                </div>
                <div className="flex-1 max-w-xs">
                  <div className="flex justify-between text-sm mb-1">
                    <span>Overall Progress</span>
                    <span>{Math.round(totalProgress)}%</span>
                  </div>
                  <div className="h-2 bg-white/20 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-white rounded-full transition-all"
                      style={{ width: `${totalProgress}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Tabs */}
          <div className="flex items-center gap-2 mb-8 border-b border-white/10 pb-4">
            <button
              onClick={() => setActiveTab('milestones')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                activeTab === 'milestones'
                  ? 'bg-green-500 text-white'
                  : 'text-muted-foreground hover:text-white hover:bg-white/5'
              }`}
            >
              <Target className="w-4 h-4" />
              Milestones
            </button>
            <button
              onClick={() => setActiveTab('mentors')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                activeTab === 'mentors'
                  ? 'bg-green-500 text-white'
                  : 'text-muted-foreground hover:text-white hover:bg-white/5'
              }`}
            >
              <Users className="w-4 h-4" />
              Mentorship
            </button>
            <button
              onClick={() => setActiveTab('residencies')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                activeTab === 'residencies'
                  ? 'bg-green-500 text-white'
                  : 'text-muted-foreground hover:text-white hover:bg-white/5'
              }`}
            >
              <Award className="w-4 h-4" />
              Residencies
            </button>
          </div>

          {/* Tab Content */}
          {activeTab === 'milestones' && (
            <MilestonesTab milestones={mockMilestones} />
          )}
          {activeTab === 'mentors' && (
            <MentorsTab mentors={mockMentors} />
          )}
          {activeTab === 'residencies' && (
            <ResidenciesTab residencies={mockResidencies} />
          )}
        </div>
      </main>

      <Player />
    </div>
  );
}

function MilestonesTab({ milestones }: { milestones: Milestone[] }) {
  return (
    <div className="space-y-6">
      {/* Active Milestones */}
      <div className="grid grid-cols-2 gap-4">
        {milestones.filter(m => !m.isCompleted).map((milestone) => (
          <MilestoneCard key={milestone.id} milestone={milestone} />
        ))}
      </div>

      {/* Completed */}
      {milestones.some(m => m.isCompleted) && (
        <div>
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <CheckCircle className="w-5 h-5 text-green-400" />
            Completed
          </h3>
          <div className="grid grid-cols-2 gap-4">
            {milestones.filter(m => m.isCompleted).map((milestone) => (
              <MilestoneCard key={milestone.id} milestone={milestone} />
            ))}
          </div>
        </div>
      )}

      {/* Upcoming Milestones Preview */}
      <div className="p-6 bg-white/5 rounded-xl border border-white/10">
        <h3 className="font-semibold mb-4">Coming Soon</h3>
        <div className="space-y-3">
          <div className="flex items-center gap-3 text-muted-foreground">
            <Circle className="w-4 h-4" />
            <span>1,000 Listener Club - Reach 1,000 unique listeners</span>
          </div>
          <div className="flex items-center gap-3 text-muted-foreground">
            <Circle className="w-4 h-4" />
            <span>Viral Moment - Get a track featured in Discover</span>
          </div>
          <div className="flex items-center gap-3 text-muted-foreground">
            <Circle className="w-4 h-4" />
            <span>Genre Pioneer - Top 10 in a genre chart</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function MilestoneCard({ milestone }: { milestone: Milestone }) {
  const progress = (milestone.current / milestone.target) * 100;

  return (
    <div className={`p-6 rounded-xl border ${
      milestone.isCompleted
        ? 'bg-green-500/10 border-green-500/30'
        : 'bg-white/5 border-white/10'
    }`}>
      <div className="flex items-start justify-between mb-4">
        <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
          milestone.isCompleted ? 'bg-green-500/20 text-green-400' : 'bg-white/10'
        }`}>
          {milestone.icon}
        </div>
        {milestone.isCompleted && (
          <CheckCircle className="w-6 h-6 text-green-400" />
        )}
      </div>

      <h4 className="font-semibold mb-1">{milestone.title}</h4>
      <p className="text-sm text-muted-foreground mb-4">{milestone.description}</p>

      {/* Progress */}
      <div className="mb-4">
        <div className="flex justify-between text-sm mb-1">
          <span>{milestone.current} / {milestone.target}</span>
          <span>{Math.round(progress)}%</span>
        </div>
        <div className="h-2 bg-white/10 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${
              milestone.isCompleted ? 'bg-green-500' : 'bg-purple-500'
            }`}
            style={{ width: `${Math.min(progress, 100)}%` }}
          />
        </div>
      </div>

      {/* Reward */}
      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center gap-2">
          <Gift className="w-4 h-4 text-yellow-400" />
          <span>{milestone.reward}</span>
        </div>
        <span className="text-green-400 font-medium">+${milestone.rewardAmount}</span>
      </div>
    </div>
  );
}

function MentorsTab({ mentors }: { mentors: Mentor[] }) {
  return (
    <div className="space-y-6">
      {/* Info */}
      <div className="p-6 bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-xl border border-white/10">
        <div className="flex items-start gap-4">
          <Sparkles className="w-8 h-8 text-purple-400 flex-shrink-0" />
          <div>
            <h3 className="font-semibold mb-2">Mentorship Program</h3>
            <p className="text-muted-foreground">
              Connect with established artists who volunteer their time to help emerging creators.
              Mentorship is free - it's part of how the mycelium network gives back.
            </p>
          </div>
        </div>
      </div>

      {/* Mentor Grid */}
      <div className="grid grid-cols-3 gap-4">
        {mentors.map((mentor) => (
          <MentorCard key={mentor.id} mentor={mentor} />
        ))}
      </div>

      {/* Become a Mentor */}
      <div className="p-6 bg-white/5 rounded-xl border border-white/10">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-semibold mb-1">Want to Give Back?</h3>
            <p className="text-sm text-muted-foreground">
              Established artists can apply to become mentors and earn Spores for their contributions.
            </p>
          </div>
          <button className="px-4 py-2 bg-purple-500 rounded-lg font-medium hover:bg-purple-600 transition-colors">
            Apply to Mentor
          </button>
        </div>
      </div>
    </div>
  );
}

function MentorCard({ mentor }: { mentor: Mentor }) {
  const availabilityColors = {
    open: 'bg-green-500',
    limited: 'bg-yellow-500',
    closed: 'bg-red-500',
  };

  return (
    <div className="bg-white/5 rounded-xl p-6 hover:bg-white/10 transition-colors">
      <div className="flex items-start gap-4 mb-4">
        <div className="relative">
          <Image
            src={mentor.avatar}
            alt={mentor.name}
            width={48}
            height={48}
            className="rounded-full"
          />
          <div className={`absolute -bottom-1 -right-1 w-4 h-4 rounded-full border-2 border-gray-900 ${availabilityColors[mentor.availability]}`} />
        </div>
        <div>
          <h4 className="font-semibold">{mentor.name}</h4>
          <p className="text-sm text-muted-foreground">{mentor.genre}</p>
        </div>
      </div>

      <div className="flex items-center gap-4 text-sm text-muted-foreground mb-4">
        <div className="flex items-center gap-1">
          <Users className="w-4 h-4" />
          {(mentor.followers / 1000).toFixed(1)}k followers
        </div>
        <div className="flex items-center gap-1">
          <Star className="w-4 h-4 text-yellow-400" />
          {mentor.rating}
        </div>
      </div>

      <div className="flex flex-wrap gap-1 mb-4">
        {mentor.specialties.map((specialty) => (
          <span
            key={specialty}
            className="px-2 py-0.5 bg-white/10 rounded text-xs"
          >
            {specialty}
          </span>
        ))}
      </div>

      <div className="flex items-center justify-between">
        <span className="text-sm text-muted-foreground">
          {mentor.menteeCount} mentees
        </span>
        <button
          disabled={mentor.availability === 'closed'}
          className="px-3 py-1.5 bg-purple-500 rounded-lg text-sm font-medium hover:bg-purple-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {mentor.availability === 'closed' ? 'Unavailable' : 'Request'}
        </button>
      </div>
    </div>
  );
}

function ResidenciesTab({ residencies }: { residencies: Residency[] }) {
  return (
    <div className="space-y-6">
      {/* Info */}
      <div className="p-6 bg-gradient-to-r from-green-500/10 to-teal-500/10 rounded-xl border border-white/10">
        <div className="flex items-start gap-4">
          <Award className="w-8 h-8 text-green-400 flex-shrink-0" />
          <div>
            <h3 className="font-semibold mb-2">Community-Voted Residencies</h3>
            <p className="text-muted-foreground">
              Residencies are funded by the community treasury and selected through Spore-weighted voting.
              Apply to receive funding and support for your next project.
            </p>
          </div>
        </div>
      </div>

      {/* Residency Cards */}
      <div className="space-y-4">
        {residencies.map((residency) => (
          <ResidencyCard key={residency.id} residency={residency} />
        ))}
      </div>

      {/* Propose New */}
      <div className="p-6 bg-white/5 rounded-xl border border-white/10 border-dashed">
        <div className="text-center">
          <h3 className="font-semibold mb-2">Have an Idea?</h3>
          <p className="text-sm text-muted-foreground mb-4">
            Propose a new residency program for community consideration
          </p>
          <button className="px-4 py-2 bg-white/10 rounded-lg font-medium hover:bg-white/20 transition-colors">
            Submit Proposal
          </button>
        </div>
      </div>
    </div>
  );
}

function ResidencyCard({ residency }: { residency: Residency }) {
  const statusColors = {
    open: 'bg-green-500',
    voting: 'bg-purple-500',
    active: 'bg-blue-500',
    completed: 'bg-gray-500',
  };

  const daysUntilDeadline = Math.ceil(
    (residency.deadline.getTime() - Date.now()) / (1000 * 60 * 60 * 24)
  );

  return (
    <div className="bg-white/5 rounded-xl p-6 hover:bg-white/10 transition-colors">
      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <h4 className="text-lg font-semibold">{residency.title}</h4>
            <span className={`px-2 py-0.5 rounded text-xs font-medium ${statusColors[residency.status]}`}>
              {residency.status.toUpperCase()}
            </span>
          </div>
          <p className="text-muted-foreground">{residency.description}</p>
        </div>
        <div className="text-right">
          <p className="text-2xl font-bold text-green-400">${residency.funding}</p>
          <p className="text-sm text-muted-foreground">{residency.duration}</p>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4 mb-4">
        <div>
          <p className="text-sm text-muted-foreground">Applicants</p>
          <p className="font-semibold">{residency.applicants}</p>
        </div>
        <div>
          <p className="text-sm text-muted-foreground">Community Votes</p>
          <p className="font-semibold">{residency.votes.toLocaleString()}</p>
        </div>
        <div>
          <p className="text-sm text-muted-foreground">Deadline</p>
          <p className="font-semibold">{daysUntilDeadline} days left</p>
        </div>
      </div>

      <div className="mb-4">
        <p className="text-sm font-medium mb-2">Requirements:</p>
        <ul className="space-y-1">
          {residency.requirements.map((req, i) => (
            <li key={i} className="text-sm text-muted-foreground flex items-center gap-2">
              <CheckCircle className="w-3 h-3 text-green-400" />
              {req}
            </li>
          ))}
        </ul>
      </div>

      <div className="flex items-center gap-3">
        {residency.status === 'open' && (
          <button className="flex-1 py-2 bg-green-500 rounded-lg font-medium hover:bg-green-600 transition-colors">
            Apply Now
          </button>
        )}
        {residency.status === 'voting' && (
          <button className="flex-1 py-2 bg-purple-500 rounded-lg font-medium hover:bg-purple-600 transition-colors">
            Vote for Applicants
          </button>
        )}
        <button className="px-4 py-2 bg-white/10 rounded-lg hover:bg-white/20 transition-colors">
          Learn More
        </button>
      </div>
    </div>
  );
}
