// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * TeamTrust - Shared Trust Networks & Team Collaboration
 *
 * This component provides:
 * - Shared trust network management
 * - Team workspaces with role-based access
 * - Collaborative trust attestations
 * - Trust policy management
 * - Network visualization
 */

import React, { useState, useCallback, useMemo } from 'react';
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import {
  Users,
  Network,
  Shield,
  Plus,
  Settings,
  UserPlus,
  Mail,
  CheckCircle,
  XCircle,
  Clock,
  Eye,
  Edit,
  Trash2,
  Copy,
  Link,
  Lock,
  Globe,
  Building,
  Crown,
  UserCheck,
  AlertTriangle,
  TrendingUp,
  Activity,
  Share2,
  GitBranch,
  Zap,
} from 'lucide-react';

// ============================================================================
// Types
// ============================================================================

export interface TeamMember {
  id: string;
  email: string;
  name: string;
  avatar?: string;
  role: 'owner' | 'admin' | 'member' | 'viewer';
  joinedAt: Date;
  lastActive?: Date;
  trustContributions: number;
  status: 'active' | 'invited' | 'suspended';
}

export interface TrustPolicy {
  id: string;
  name: string;
  description: string;
  rules: TrustRule[];
  isDefault: boolean;
  createdBy: string;
  createdAt: Date;
}

export interface TrustRule {
  id: string;
  type: 'auto_trust' | 'require_approval' | 'block' | 'verify_domain' | 'threshold';
  condition: {
    field: 'domain' | 'sender' | 'trust_score' | 'attestation_count' | 'member_vouched';
    operator: 'equals' | 'contains' | 'greater_than' | 'less_than' | 'in_list';
    value: string | number | string[];
  };
  action: {
    type: 'set_trust' | 'flag' | 'notify' | 'quarantine';
    value?: number | string;
  };
}

export interface SharedNetwork {
  id: string;
  name: string;
  description: string;
  icon?: string;
  visibility: 'private' | 'invite_only' | 'public';
  members: TeamMember[];
  policies: TrustPolicy[];
  trustGraph: TrustEdge[];
  stats: NetworkStats;
  createdAt: Date;
  updatedAt: Date;
}

export interface TrustEdge {
  from: string;
  to: string;
  score: number;
  attestations: string[];
  lastUpdated: Date;
}

export interface NetworkStats {
  totalMembers: number;
  totalTrustedContacts: number;
  totalAttestations: number;
  averageTrustScore: number;
  activeThisWeek: number;
  pendingApprovals: number;
}

export interface CollaborativeAttestation {
  id: string;
  networkId: string;
  targetEmail: string;
  targetName?: string;
  proposedScore: number;
  reason: string;
  proposedBy: TeamMember;
  proposedAt: Date;
  votes: AttestationVote[];
  status: 'pending' | 'approved' | 'rejected' | 'expired';
  requiredVotes: number;
  expiresAt: Date;
}

export interface AttestationVote {
  memberId: string;
  memberName: string;
  vote: 'approve' | 'reject' | 'abstain';
  comment?: string;
  votedAt: Date;
}

// ============================================================================
// Store
// ============================================================================

interface TeamTrustState {
  networks: SharedNetwork[];
  activeNetworkId: string | null;
  pendingAttestations: CollaborativeAttestation[];
  invitations: NetworkInvitation[];

  // Actions
  createNetwork: (network: Omit<SharedNetwork, 'id' | 'createdAt' | 'updatedAt' | 'stats'>) => string;
  updateNetwork: (id: string, updates: Partial<SharedNetwork>) => void;
  deleteNetwork: (id: string) => void;
  setActiveNetwork: (id: string | null) => void;

  addMember: (networkId: string, member: Omit<TeamMember, 'id' | 'joinedAt' | 'trustContributions'>) => void;
  updateMember: (networkId: string, memberId: string, updates: Partial<TeamMember>) => void;
  removeMember: (networkId: string, memberId: string) => void;

  addPolicy: (networkId: string, policy: Omit<TrustPolicy, 'id' | 'createdAt'>) => void;
  updatePolicy: (networkId: string, policyId: string, updates: Partial<TrustPolicy>) => void;
  removePolicy: (networkId: string, policyId: string) => void;

  proposeAttestation: (attestation: Omit<CollaborativeAttestation, 'id' | 'proposedAt' | 'votes' | 'status'>) => string;
  voteOnAttestation: (attestationId: string, vote: AttestationVote) => void;
}

interface NetworkInvitation {
  id: string;
  networkId: string;
  networkName: string;
  invitedBy: string;
  invitedAt: Date;
  expiresAt: Date;
  role: TeamMember['role'];
}

export const useTeamTrustStore = create<TeamTrustState>()(
  persist(
    (set, get) => ({
      networks: [],
      activeNetworkId: null,
      pendingAttestations: [],
      invitations: [],

      createNetwork: (network) => {
        const id = `network_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        const newNetwork: SharedNetwork = {
          ...network,
          id,
          createdAt: new Date(),
          updatedAt: new Date(),
          stats: {
            totalMembers: network.members.length,
            totalTrustedContacts: 0,
            totalAttestations: 0,
            averageTrustScore: 0,
            activeThisWeek: 0,
            pendingApprovals: 0,
          },
        };
        set((state) => ({ networks: [...state.networks, newNetwork] }));
        return id;
      },

      updateNetwork: (id, updates) => {
        set((state) => ({
          networks: state.networks.map((n) =>
            n.id === id ? { ...n, ...updates, updatedAt: new Date() } : n
          ),
        }));
      },

      deleteNetwork: (id) => {
        set((state) => ({
          networks: state.networks.filter((n) => n.id !== id),
          activeNetworkId: state.activeNetworkId === id ? null : state.activeNetworkId,
        }));
      },

      setActiveNetwork: (id) => {
        set({ activeNetworkId: id });
      },

      addMember: (networkId, member) => {
        const newMember: TeamMember = {
          ...member,
          id: `member_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          joinedAt: new Date(),
          trustContributions: 0,
        };
        set((state) => ({
          networks: state.networks.map((n) =>
            n.id === networkId
              ? {
                  ...n,
                  members: [...n.members, newMember],
                  stats: { ...n.stats, totalMembers: n.stats.totalMembers + 1 },
                  updatedAt: new Date(),
                }
              : n
          ),
        }));
      },

      updateMember: (networkId, memberId, updates) => {
        set((state) => ({
          networks: state.networks.map((n) =>
            n.id === networkId
              ? {
                  ...n,
                  members: n.members.map((m) =>
                    m.id === memberId ? { ...m, ...updates } : m
                  ),
                  updatedAt: new Date(),
                }
              : n
          ),
        }));
      },

      removeMember: (networkId, memberId) => {
        set((state) => ({
          networks: state.networks.map((n) =>
            n.id === networkId
              ? {
                  ...n,
                  members: n.members.filter((m) => m.id !== memberId),
                  stats: { ...n.stats, totalMembers: n.stats.totalMembers - 1 },
                  updatedAt: new Date(),
                }
              : n
          ),
        }));
      },

      addPolicy: (networkId, policy) => {
        const newPolicy: TrustPolicy = {
          ...policy,
          id: `policy_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          createdAt: new Date(),
        };
        set((state) => ({
          networks: state.networks.map((n) =>
            n.id === networkId
              ? { ...n, policies: [...n.policies, newPolicy], updatedAt: new Date() }
              : n
          ),
        }));
      },

      updatePolicy: (networkId, policyId, updates) => {
        set((state) => ({
          networks: state.networks.map((n) =>
            n.id === networkId
              ? {
                  ...n,
                  policies: n.policies.map((p) =>
                    p.id === policyId ? { ...p, ...updates } : p
                  ),
                  updatedAt: new Date(),
                }
              : n
          ),
        }));
      },

      removePolicy: (networkId, policyId) => {
        set((state) => ({
          networks: state.networks.map((n) =>
            n.id === networkId
              ? {
                  ...n,
                  policies: n.policies.filter((p) => p.id !== policyId),
                  updatedAt: new Date(),
                }
              : n
          ),
        }));
      },

      proposeAttestation: (attestation) => {
        const id = `attestation_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        const newAttestation: CollaborativeAttestation = {
          ...attestation,
          id,
          proposedAt: new Date(),
          votes: [],
          status: 'pending',
        };
        set((state) => ({
          pendingAttestations: [...state.pendingAttestations, newAttestation],
        }));
        return id;
      },

      voteOnAttestation: (attestationId, vote) => {
        set((state) => {
          const attestations = state.pendingAttestations.map((a) => {
            if (a.id !== attestationId) return a;

            const votes = [...a.votes.filter((v) => v.memberId !== vote.memberId), vote];
            const approvals = votes.filter((v) => v.vote === 'approve').length;
            const rejections = votes.filter((v) => v.vote === 'reject').length;

            let status = a.status;
            if (approvals >= a.requiredVotes) status = 'approved';
            else if (rejections > a.votes.length - a.requiredVotes) status = 'rejected';

            return { ...a, votes, status };
          });

          return { pendingAttestations: attestations };
        });
      },
    }),
    {
      name: 'mycelix-team-trust',
    }
  )
);

// ============================================================================
// Hooks
// ============================================================================

export function useActiveNetwork() {
  const { networks, activeNetworkId } = useTeamTrustStore();
  return useMemo(
    () => networks.find((n) => n.id === activeNetworkId) || null,
    [networks, activeNetworkId]
  );
}

export function useNetworkMembers(networkId: string | null) {
  const { networks } = useTeamTrustStore();
  return useMemo(() => {
    const network = networks.find((n) => n.id === networkId);
    return network?.members || [];
  }, [networks, networkId]);
}

export function usePendingAttestations(networkId: string | null) {
  const { pendingAttestations } = useTeamTrustStore();
  return useMemo(
    () => pendingAttestations.filter((a) => a.networkId === networkId && a.status === 'pending'),
    [pendingAttestations, networkId]
  );
}

// ============================================================================
// Components
// ============================================================================

// Network List Sidebar
export function NetworkList() {
  const { networks, activeNetworkId, setActiveNetwork, invitations } = useTeamTrustStore();
  const [showCreate, setShowCreate] = useState(false);

  return (
    <div className="w-64 border-r bg-gray-50 dark:bg-gray-900 flex flex-col">
      <div className="p-4 border-b">
        <h2 className="font-semibold flex items-center gap-2">
          <Network className="w-5 h-5" />
          Trust Networks
        </h2>
      </div>

      <div className="flex-1 overflow-y-auto">
        {networks.length === 0 ? (
          <div className="p-4 text-center text-gray-500">
            <Network className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No networks yet</p>
            <button
              onClick={() => setShowCreate(true)}
              className="mt-2 text-blue-500 hover:underline text-sm"
            >
              Create your first network
            </button>
          </div>
        ) : (
          <div className="p-2 space-y-1">
            {networks.map((network) => (
              <button
                key={network.id}
                onClick={() => setActiveNetwork(network.id)}
                className={`w-full p-3 rounded-lg text-left transition-colors ${
                  activeNetworkId === network.id
                    ? 'bg-blue-100 dark:bg-blue-900 text-blue-900 dark:text-blue-100'
                    : 'hover:bg-gray-100 dark:hover:bg-gray-800'
                }`}
              >
                <div className="flex items-center gap-2">
                  {network.visibility === 'private' ? (
                    <Lock className="w-4 h-4" />
                  ) : network.visibility === 'public' ? (
                    <Globe className="w-4 h-4" />
                  ) : (
                    <Users className="w-4 h-4" />
                  )}
                  <span className="font-medium truncate">{network.name}</span>
                </div>
                <div className="flex items-center gap-3 mt-1 text-xs text-gray-500">
                  <span>{network.stats.totalMembers} members</span>
                  <span>{network.stats.totalTrustedContacts} contacts</span>
                </div>
              </button>
            ))}
          </div>
        )}

        {invitations.length > 0 && (
          <div className="p-2 border-t">
            <h3 className="px-2 py-1 text-xs font-semibold text-gray-500 uppercase">
              Invitations ({invitations.length})
            </h3>
            {invitations.map((inv) => (
              <div
                key={inv.id}
                className="p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg mt-1"
              >
                <p className="font-medium text-sm">{inv.networkName}</p>
                <p className="text-xs text-gray-500">from {inv.invitedBy}</p>
                <div className="flex gap-2 mt-2">
                  <button className="px-2 py-1 bg-blue-500 text-white rounded text-xs">
                    Accept
                  </button>
                  <button className="px-2 py-1 bg-gray-200 dark:bg-gray-700 rounded text-xs">
                    Decline
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="p-4 border-t">
        <button
          onClick={() => setShowCreate(true)}
          className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
        >
          <Plus className="w-4 h-4" />
          New Network
        </button>
      </div>

      {showCreate && (
        <CreateNetworkModal onClose={() => setShowCreate(false)} />
      )}
    </div>
  );
}

// Create Network Modal
function CreateNetworkModal({ onClose }: { onClose: () => void }) {
  const { createNetwork, setActiveNetwork } = useTeamTrustStore();
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [visibility, setVisibility] = useState<SharedNetwork['visibility']>('private');

  const handleCreate = () => {
    if (!name.trim()) return;

    const id = createNetwork({
      name: name.trim(),
      description: description.trim(),
      visibility,
      members: [
        {
          id: 'self',
          email: 'you@example.com', // Would come from auth
          name: 'You',
          role: 'owner',
          joinedAt: new Date(),
          trustContributions: 0,
          status: 'active',
        },
      ],
      policies: [],
      trustGraph: [],
    });

    setActiveNetwork(id);
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl w-full max-w-md m-4">
        <div className="p-6 border-b">
          <h2 className="text-xl font-semibold">Create Trust Network</h2>
          <p className="text-sm text-gray-500 mt-1">
            Share trust attestations with your team or community
          </p>
        </div>

        <div className="p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">Network Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., Engineering Team"
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="What is this network for?"
              rows={3}
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">Visibility</label>
            <div className="space-y-2">
              {[
                { value: 'private', icon: Lock, label: 'Private', desc: 'Only invited members' },
                { value: 'invite_only', icon: Users, label: 'Invite Only', desc: 'Members can invite others' },
                { value: 'public', icon: Globe, label: 'Public', desc: 'Anyone can join' },
              ].map((opt) => (
                <label
                  key={opt.value}
                  className={`flex items-center gap-3 p-3 border rounded-lg cursor-pointer transition-colors ${
                    visibility === opt.value
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'hover:bg-gray-50 dark:hover:bg-gray-700'
                  }`}
                >
                  <input
                    type="radio"
                    name="visibility"
                    value={opt.value}
                    checked={visibility === opt.value}
                    onChange={(e) => setVisibility(e.target.value as SharedNetwork['visibility'])}
                    className="sr-only"
                  />
                  <opt.icon className="w-5 h-5" />
                  <div>
                    <p className="font-medium">{opt.label}</p>
                    <p className="text-xs text-gray-500">{opt.desc}</p>
                  </div>
                </label>
              ))}
            </div>
          </div>
        </div>

        <div className="p-6 border-t flex justify-end gap-3">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleCreate}
            disabled={!name.trim()}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Create Network
          </button>
        </div>
      </div>
    </div>
  );
}

// Network Dashboard
export function NetworkDashboard() {
  const network = useActiveNetwork();
  const pendingAttestations = usePendingAttestations(network?.id || null);

  if (!network) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-500">
        <div className="text-center">
          <Network className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>Select a network to view</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto">
      {/* Header */}
      <div className="p-6 border-b bg-gradient-to-r from-blue-500 to-purple-500 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">{network.name}</h1>
            <p className="text-blue-100 mt-1">{network.description}</p>
          </div>
          <div className="flex items-center gap-2">
            <button className="p-2 hover:bg-white/20 rounded-lg transition-colors">
              <Share2 className="w-5 h-5" />
            </button>
            <button className="p-2 hover:bg-white/20 rounded-lg transition-colors">
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-4 gap-4 mt-6">
          {[
            { label: 'Members', value: network.stats.totalMembers, icon: Users },
            { label: 'Trusted Contacts', value: network.stats.totalTrustedContacts, icon: UserCheck },
            { label: 'Attestations', value: network.stats.totalAttestations, icon: Shield },
            { label: 'Avg Trust Score', value: `${network.stats.averageTrustScore}%`, icon: TrendingUp },
          ].map((stat) => (
            <div key={stat.label} className="bg-white/10 rounded-lg p-4">
              <stat.icon className="w-5 h-5 mb-2" />
              <p className="text-2xl font-bold">{stat.value}</p>
              <p className="text-sm text-blue-100">{stat.label}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="p-6 grid grid-cols-2 gap-6">
        {/* Pending Attestations */}
        <div className="border rounded-xl">
          <div className="p-4 border-b flex items-center justify-between">
            <h2 className="font-semibold flex items-center gap-2">
              <Clock className="w-5 h-5" />
              Pending Approvals
            </h2>
            <span className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded-full text-xs font-medium">
              {pendingAttestations.length}
            </span>
          </div>
          <div className="p-4 space-y-3 max-h-80 overflow-y-auto">
            {pendingAttestations.length === 0 ? (
              <p className="text-center text-gray-500 py-4">No pending approvals</p>
            ) : (
              pendingAttestations.map((attestation) => (
                <PendingAttestationCard key={attestation.id} attestation={attestation} />
              ))
            )}
          </div>
        </div>

        {/* Recent Activity */}
        <div className="border rounded-xl">
          <div className="p-4 border-b">
            <h2 className="font-semibold flex items-center gap-2">
              <Activity className="w-5 h-5" />
              Recent Activity
            </h2>
          </div>
          <div className="p-4 space-y-3 max-h-80 overflow-y-auto">
            <ActivityItem
              icon={UserPlus}
              title="New member joined"
              description="alice@example.com joined the network"
              time="2 hours ago"
            />
            <ActivityItem
              icon={Shield}
              title="Trust attestation approved"
              description="bob@vendor.com now trusted (85%)"
              time="5 hours ago"
            />
            <ActivityItem
              icon={AlertTriangle}
              title="Suspicious sender flagged"
              description="unknown@scam.com blocked by policy"
              time="1 day ago"
            />
          </div>
        </div>

        {/* Members */}
        <div className="border rounded-xl">
          <div className="p-4 border-b flex items-center justify-between">
            <h2 className="font-semibold flex items-center gap-2">
              <Users className="w-5 h-5" />
              Members
            </h2>
            <button className="flex items-center gap-1 text-blue-500 hover:underline text-sm">
              <UserPlus className="w-4 h-4" />
              Invite
            </button>
          </div>
          <div className="p-4 space-y-2 max-h-80 overflow-y-auto">
            {network.members.map((member) => (
              <MemberCard key={member.id} member={member} />
            ))}
          </div>
        </div>

        {/* Trust Policies */}
        <div className="border rounded-xl">
          <div className="p-4 border-b flex items-center justify-between">
            <h2 className="font-semibold flex items-center gap-2">
              <Shield className="w-5 h-5" />
              Trust Policies
            </h2>
            <button className="flex items-center gap-1 text-blue-500 hover:underline text-sm">
              <Plus className="w-4 h-4" />
              Add Rule
            </button>
          </div>
          <div className="p-4 space-y-2 max-h-80 overflow-y-auto">
            {network.policies.length === 0 ? (
              <div className="text-center py-4">
                <Shield className="w-8 h-8 mx-auto mb-2 text-gray-400" />
                <p className="text-gray-500 text-sm">No policies configured</p>
                <button className="mt-2 text-blue-500 hover:underline text-sm">
                  Add your first policy
                </button>
              </div>
            ) : (
              network.policies.map((policy) => (
                <PolicyCard key={policy.id} policy={policy} />
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// Pending Attestation Card
function PendingAttestationCard({ attestation }: { attestation: CollaborativeAttestation }) {
  const { voteOnAttestation } = useTeamTrustStore();

  const approvals = attestation.votes.filter((v) => v.vote === 'approve').length;
  const rejections = attestation.votes.filter((v) => v.vote === 'reject').length;

  return (
    <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
      <div className="flex items-start justify-between">
        <div>
          <p className="font-medium">{attestation.targetEmail}</p>
          <p className="text-sm text-gray-500">{attestation.targetName}</p>
        </div>
        <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs font-medium">
          {attestation.proposedScore}%
        </span>
      </div>
      <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">{attestation.reason}</p>
      <div className="flex items-center justify-between mt-3">
        <span className="text-xs text-gray-500">
          {approvals}/{attestation.requiredVotes} votes • by {attestation.proposedBy.name}
        </span>
        <div className="flex gap-2">
          <button
            onClick={() =>
              voteOnAttestation(attestation.id, {
                memberId: 'self',
                memberName: 'You',
                vote: 'approve',
                votedAt: new Date(),
              })
            }
            className="p-1 text-green-600 hover:bg-green-100 rounded"
          >
            <CheckCircle className="w-5 h-5" />
          </button>
          <button
            onClick={() =>
              voteOnAttestation(attestation.id, {
                memberId: 'self',
                memberName: 'You',
                vote: 'reject',
                votedAt: new Date(),
              })
            }
            className="p-1 text-red-600 hover:bg-red-100 rounded"
          >
            <XCircle className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
}

// Activity Item
function ActivityItem({
  icon: Icon,
  title,
  description,
  time,
}: {
  icon: React.ElementType;
  title: string;
  description: string;
  time: string;
}) {
  return (
    <div className="flex items-start gap-3">
      <div className="p-2 bg-gray-100 dark:bg-gray-700 rounded-lg">
        <Icon className="w-4 h-4" />
      </div>
      <div className="flex-1">
        <p className="font-medium text-sm">{title}</p>
        <p className="text-xs text-gray-500">{description}</p>
      </div>
      <span className="text-xs text-gray-400">{time}</span>
    </div>
  );
}

// Member Card
function MemberCard({ member }: { member: TeamMember }) {
  const roleColors = {
    owner: 'text-yellow-600',
    admin: 'text-purple-600',
    member: 'text-blue-600',
    viewer: 'text-gray-600',
  };

  const roleIcons = {
    owner: Crown,
    admin: Shield,
    member: Users,
    viewer: Eye,
  };

  const RoleIcon = roleIcons[member.role];

  return (
    <div className="flex items-center gap-3 p-2 hover:bg-gray-50 dark:hover:bg-gray-800 rounded-lg">
      <div className="w-10 h-10 bg-gradient-to-br from-blue-400 to-purple-500 rounded-full flex items-center justify-center text-white font-medium">
        {member.name.charAt(0).toUpperCase()}
      </div>
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <p className="font-medium">{member.name}</p>
          <RoleIcon className={`w-4 h-4 ${roleColors[member.role]}`} />
        </div>
        <p className="text-xs text-gray-500">{member.email}</p>
      </div>
      <div className="text-right">
        <p className="text-sm font-medium">{member.trustContributions}</p>
        <p className="text-xs text-gray-500">contributions</p>
      </div>
    </div>
  );
}

// Policy Card
function PolicyCard({ policy }: { policy: TrustPolicy }) {
  return (
    <div className="p-3 border rounded-lg">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Shield className="w-4 h-4 text-blue-500" />
          <span className="font-medium">{policy.name}</span>
        </div>
        {policy.isDefault && (
          <span className="bg-green-100 text-green-800 px-2 py-0.5 rounded text-xs">
            Default
          </span>
        )}
      </div>
      <p className="text-sm text-gray-500 mt-1">{policy.description}</p>
      <div className="flex items-center gap-2 mt-2">
        <span className="text-xs bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
          {policy.rules.length} rules
        </span>
      </div>
    </div>
  );
}

// Trust Network Visualization
export function TrustNetworkGraph({ networkId }: { networkId: string }) {
  const { networks } = useTeamTrustStore();
  const network = networks.find((n) => n.id === networkId);

  if (!network) return null;

  // Simple force-directed layout placeholder
  // In production, use d3-force or similar
  return (
    <div className="w-full h-96 bg-gray-50 dark:bg-gray-900 rounded-xl border flex items-center justify-center">
      <div className="text-center text-gray-500">
        <GitBranch className="w-12 h-12 mx-auto mb-4 opacity-50" />
        <p>Trust Network Visualization</p>
        <p className="text-sm mt-1">
          {network.trustGraph.length} connections between {network.stats.totalTrustedContacts} contacts
        </p>
      </div>
    </div>
  );
}

// Propose Attestation Modal
export function ProposeAttestationModal({
  networkId,
  onClose,
}: {
  networkId: string;
  onClose: () => void;
}) {
  const { proposeAttestation, networks } = useTeamTrustStore();
  const network = networks.find((n) => n.id === networkId);

  const [email, setEmail] = useState('');
  const [name, setName] = useState('');
  const [score, setScore] = useState(75);
  const [reason, setReason] = useState('');

  const handlePropose = () => {
    if (!email.trim() || !reason.trim() || !network) return;

    proposeAttestation({
      networkId,
      targetEmail: email.trim(),
      targetName: name.trim() || undefined,
      proposedScore: score,
      reason: reason.trim(),
      proposedBy: network.members.find((m) => m.role === 'owner') || network.members[0],
      requiredVotes: Math.ceil(network.members.length / 2),
      expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
    });

    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl w-full max-w-md m-4">
        <div className="p-6 border-b">
          <h2 className="text-xl font-semibold">Propose Trust Attestation</h2>
          <p className="text-sm text-gray-500 mt-1">
            Submit for team approval
          </p>
        </div>

        <div className="p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">Email Address</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="contact@example.com"
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Name (optional)</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Contact Name"
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">
              Trust Score: {score}%
            </label>
            <input
              type="range"
              min={0}
              max={100}
              value={score}
              onChange={(e) => setScore(Number(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>Untrusted</span>
              <span>Fully Trusted</span>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Reason</label>
            <textarea
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="Why should this contact be trusted?"
              rows={3}
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600"
            />
          </div>
        </div>

        <div className="p-6 border-t flex justify-end gap-3">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handlePropose}
            disabled={!email.trim() || !reason.trim()}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Submit for Approval
          </button>
        </div>
      </div>
    </div>
  );
}

// Main Team Trust Page
export function TeamTrustPage() {
  return (
    <div className="flex h-full">
      <NetworkList />
      <NetworkDashboard />
    </div>
  );
}

export default TeamTrustPage;
