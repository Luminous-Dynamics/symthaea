'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

// Types
interface Initiative {
  id: string
  title: string
  description: string
  category: 'infrastructure' | 'education' | 'governance' | 'sustainability' | 'economic'
  proposedBy: string
  proposedDate: string
  votingEndDate: string
  requiredInvestment: number
  currentVotes: {
    yes: number
    no: number
    abstain: number
  }
  status: 'active' | 'passed' | 'failed' | 'pending'
  communityImpact: {
    jobs: number
    households: number
    co2Reduction: number
  }
  userVoted?: 'yes' | 'no' | 'abstain' | null
  quorum: number // Minimum participation percentage
  passingThreshold: number // Percentage of yes votes needed
}

interface VotingPower {
  userId: string
  totalInvestment: number
  votingPower: number // Calculated based on investment
  multiplier: number // Based on community engagement
}

// Sample initiatives data
const sampleInitiatives: Initiative[] = [
  {
    id: '1',
    title: 'Community Solar Garden Expansion',
    description: 'Expand our community solar program to include 500 additional households in underserved areas',
    category: 'infrastructure',
    proposedBy: 'Community Energy Coalition',
    proposedDate: '2025-01-15',
    votingEndDate: '2025-02-15',
    requiredInvestment: 2500000,
    currentVotes: {
      yes: 342,
      no: 89,
      abstain: 45
    },
    status: 'active',
    communityImpact: {
      jobs: 45,
      households: 500,
      co2Reduction: 2800
    },
    quorum: 30,
    passingThreshold: 66
  },
  {
    id: '2',
    title: 'Energy Education Program',
    description: 'Launch comprehensive renewable energy education in local schools and community centers',
    category: 'education',
    proposedBy: 'Local Teachers Union',
    proposedDate: '2025-01-20',
    votingEndDate: '2025-02-20',
    requiredInvestment: 450000,
    currentVotes: {
      yes: 567,
      no: 23,
      abstain: 78
    },
    status: 'active',
    communityImpact: {
      jobs: 12,
      households: 2000,
      co2Reduction: 0
    },
    quorum: 30,
    passingThreshold: 60
  },
  {
    id: '3',
    title: 'Community Ownership Transition Fund',
    description: 'Establish a fund to help transition projects to full community ownership over 7 years',
    category: 'governance',
    proposedBy: 'Investor Coalition',
    proposedDate: '2025-01-10',
    votingEndDate: '2025-02-10',
    requiredInvestment: 10000000,
    currentVotes: {
      yes: 892,
      no: 156,
      abstain: 203
    },
    status: 'passed',
    communityImpact: {
      jobs: 200,
      households: 5000,
      co2Reduction: 15000
    },
    quorum: 40,
    passingThreshold: 75
  }
]

const categoryIcons = {
  infrastructure: '🏗️',
  education: '📚',
  governance: '⚖️',
  sustainability: '🌱',
  economic: '💰'
}

const categoryColors = {
  infrastructure: 'from-blue-500 to-indigo-500',
  education: 'from-purple-500 to-pink-500',
  governance: 'from-amber-500 to-orange-500',
  sustainability: 'from-emerald-500 to-green-500',
  economic: 'from-cyan-500 to-teal-500'
}

export default function InvestorVotingMechanism() {
  const [initiatives, setInitiatives] = useState<Initiative[]>(sampleInitiatives)
  const [selectedInitiative, setSelectedInitiative] = useState<Initiative | null>(null)
  const [filter, setFilter] = useState<'all' | 'active' | 'passed' | 'failed'>('all')
  const [showProposalForm, setShowProposalForm] = useState(false)
  const [votingPower] = useState<VotingPower>({
    userId: 'user123',
    totalInvestment: 50000,
    votingPower: 50, // 1 vote per $1000
    multiplier: 1.2 // 20% bonus for engagement
  })

  // Calculate voting statistics
  const calculateVotePercentage = (votes: Initiative['currentVotes']) => {
    const total = votes.yes + votes.no + votes.abstain
    if (total === 0) return { yes: 0, no: 0, abstain: 0 }
    return {
      yes: (votes.yes / total) * 100,
      no: (votes.no / total) * 100,
      abstain: (votes.abstain / total) * 100
    }
  }

  const calculateTimeRemaining = (endDate: string) => {
    const end = new Date(endDate).getTime()
    const now = new Date().getTime()
    const diff = end - now
    
    if (diff <= 0) return 'Voting ended'
    
    const days = Math.floor(diff / (1000 * 60 * 60 * 24))
    const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60))
    
    if (days > 0) return `${days} days remaining`
    return `${hours} hours remaining`
  }

  const handleVote = (initiativeId: string, vote: 'yes' | 'no' | 'abstain') => {
    setInitiatives(prev => prev.map(init => {
      if (init.id === initiativeId) {
        // Remove previous vote if exists
        const updatedVotes = { ...init.currentVotes }
        if (init.userVoted) {
          updatedVotes[init.userVoted] -= votingPower.votingPower
        }
        
        // Add new vote
        updatedVotes[vote] += votingPower.votingPower
        
        return {
          ...init,
          currentVotes: updatedVotes,
          userVoted: vote
        }
      }
      return init
    }))
  }

  const filteredInitiatives = initiatives.filter(init => {
    if (filter === 'all') return true
    return init.status === filter
  })

  return (
    <div className="w-full max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-3xl font-light text-white mb-3">
          <span className="text-4xl mr-3">🗳️</span>
          Community Building Initiatives
        </h2>
        <p className="text-white/60 max-w-3xl mx-auto">
          Vote on proposals that shape your community's energy future. Your voting power is based on your investment and engagement.
        </p>
      </div>

      {/* Voting Power Display */}
      <div className="bg-gradient-to-br from-purple-500/10 to-pink-500/10 backdrop-blur-md rounded-2xl border border-purple-500/20 p-6">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-light text-white mb-2">Your Voting Power</h3>
            <p className="text-white/60 text-sm">
              Based on ${votingPower.totalInvestment.toLocaleString()} investment
            </p>
          </div>
          <div className="text-right">
            <div className="text-3xl font-light text-purple-400">
              {Math.round(votingPower.votingPower * votingPower.multiplier)} votes
            </div>
            <div className="text-xs text-purple-400/60">
              {votingPower.multiplier > 1 && `${((votingPower.multiplier - 1) * 100).toFixed(0)}% engagement bonus`}
            </div>
          </div>
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="flex justify-center">
        <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-full p-1 inline-flex">
          {(['all', 'active', 'passed', 'failed'] as const).map((status) => (
            <button
              key={status}
              onClick={() => setFilter(status)}
              className={`px-6 py-2 rounded-full transition-all capitalize ${
                filter === status
                  ? 'bg-white/10 text-white'
                  : 'text-white/60 hover:text-white/80'
              }`}
            >
              {status}
            </button>
          ))}
        </div>
      </div>

      {/* Initiatives Grid */}
      <div className="grid md:grid-cols-2 gap-6">
        {filteredInitiatives.map((initiative, index) => {
          const percentages = calculateVotePercentage(initiative.currentVotes)
          const totalVotes = initiative.currentVotes.yes + initiative.currentVotes.no + initiative.currentVotes.abstain
          
          return (
            <motion.div
              key={initiative.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="bg-white/5 backdrop-blur-md rounded-2xl border border-white/10 p-6 hover:border-white/20 transition-all"
            >
              {/* Initiative Header */}
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-2xl">{categoryIcons[initiative.category]}</span>
                    <span className={`px-3 py-1 rounded-full text-xs bg-gradient-to-r ${categoryColors[initiative.category]} text-white`}>
                      {initiative.category}
                    </span>
                    <span className={`px-3 py-1 rounded-full text-xs ${
                      initiative.status === 'active' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' :
                      initiative.status === 'passed' ? 'bg-green-500/20 text-green-400 border border-green-500/30' :
                      initiative.status === 'failed' ? 'bg-red-500/20 text-red-400 border border-red-500/30' :
                      'bg-gray-500/20 text-gray-400 border border-gray-500/30'
                    }`}>
                      {initiative.status}
                    </span>
                  </div>
                  <h3 className="text-lg font-medium text-white mb-2">{initiative.title}</h3>
                  <p className="text-sm text-white/60 mb-3">{initiative.description}</p>
                </div>
              </div>

              {/* Community Impact */}
              <div className="grid grid-cols-3 gap-3 mb-4 p-3 bg-white/5 rounded-xl">
                <div className="text-center">
                  <div className="text-xs text-white/60">Jobs</div>
                  <div className="text-lg text-emerald-400">+{initiative.communityImpact.jobs}</div>
                </div>
                <div className="text-center">
                  <div className="text-xs text-white/60">Households</div>
                  <div className="text-lg text-cyan-400">{initiative.communityImpact.households}</div>
                </div>
                <div className="text-center">
                  <div className="text-xs text-white/60">CO₂ (tons)</div>
                  <div className="text-lg text-purple-400">-{initiative.communityImpact.co2Reduction}</div>
                </div>
              </div>

              {/* Voting Progress */}
              <div className="mb-4">
                <div className="flex justify-between text-xs text-white/60 mb-2">
                  <span>Yes: {percentages.yes.toFixed(1)}%</span>
                  <span>{totalVotes} votes</span>
                </div>
                <div className="h-3 bg-white/10 rounded-full overflow-hidden flex">
                  <div 
                    className="bg-emerald-500 transition-all"
                    style={{ width: `${percentages.yes}%` }}
                  />
                  <div 
                    className="bg-red-500 transition-all"
                    style={{ width: `${percentages.no}%` }}
                  />
                  <div 
                    className="bg-gray-500 transition-all"
                    style={{ width: `${percentages.abstain}%` }}
                  />
                </div>
                <div className="flex justify-between text-xs text-white/40 mt-1">
                  <span>Quorum: {initiative.quorum}%</span>
                  <span>Required: {initiative.passingThreshold}%</span>
                </div>
              </div>

              {/* Metadata */}
              <div className="flex justify-between items-center text-xs text-white/60 mb-4">
                <span>by {initiative.proposedBy}</span>
                <span>{calculateTimeRemaining(initiative.votingEndDate)}</span>
              </div>

              {/* Investment Required */}
              <div className="mb-4 p-3 bg-gradient-to-r from-cyan-500/10 to-blue-500/10 rounded-xl">
                <div className="text-xs text-white/60 mb-1">Required Investment</div>
                <div className="text-xl text-cyan-400">${(initiative.requiredInvestment / 1000000).toFixed(1)}M</div>
              </div>

              {/* Voting Buttons */}
              {initiative.status === 'active' && (
                <div className="grid grid-cols-3 gap-2">
                  <button
                    onClick={() => handleVote(initiative.id, 'yes')}
                    className={`px-4 py-2 rounded-lg transition-all text-sm font-medium ${
                      initiative.userVoted === 'yes'
                        ? 'bg-emerald-500 text-white'
                        : 'bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30 border border-emerald-500/30'
                    }`}
                  >
                    Yes
                  </button>
                  <button
                    onClick={() => handleVote(initiative.id, 'no')}
                    className={`px-4 py-2 rounded-lg transition-all text-sm font-medium ${
                      initiative.userVoted === 'no'
                        ? 'bg-red-500 text-white'
                        : 'bg-red-500/20 text-red-400 hover:bg-red-500/30 border border-red-500/30'
                    }`}
                  >
                    No
                  </button>
                  <button
                    onClick={() => handleVote(initiative.id, 'abstain')}
                    className={`px-4 py-2 rounded-lg transition-all text-sm font-medium ${
                      initiative.userVoted === 'abstain'
                        ? 'bg-gray-500 text-white'
                        : 'bg-gray-500/20 text-gray-400 hover:bg-gray-500/30 border border-gray-500/30'
                    }`}
                  >
                    Abstain
                  </button>
                </div>
              )}

              {/* View Details Button */}
              <button
                onClick={() => setSelectedInitiative(initiative)}
                className="w-full mt-3 px-4 py-2 bg-white/5 border border-white/20 rounded-lg text-white/60 hover:bg-white/10 hover:text-white transition-all text-sm"
              >
                View Details →
              </button>
            </motion.div>
          )
        })}
      </div>

      {/* Propose New Initiative Button */}
      <div className="text-center">
        <button
          onClick={() => setShowProposalForm(true)}
          className="px-8 py-3 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full text-white font-medium hover:shadow-lg hover:shadow-purple-500/25 transition-all transform hover:scale-105"
        >
          <span className="mr-2">💡</span> Propose New Initiative
        </button>
      </div>

      {/* Initiative Details Modal */}
      <AnimatePresence>
        {selectedInitiative && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 backdrop-blur-md z-50 flex items-center justify-center p-6"
            onClick={() => setSelectedInitiative(null)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-gradient-to-br from-gray-900 to-black border border-white/20 rounded-2xl p-8 max-w-3xl w-full max-h-[90vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex justify-between items-start mb-6">
                <div>
                  <div className="flex items-center gap-3 mb-3">
                    <span className="text-3xl">{categoryIcons[selectedInitiative.category]}</span>
                    <span className={`px-3 py-1 rounded-full text-sm bg-gradient-to-r ${categoryColors[selectedInitiative.category]} text-white`}>
                      {selectedInitiative.category}
                    </span>
                  </div>
                  <h3 className="text-2xl font-light text-white mb-2">{selectedInitiative.title}</h3>
                  <p className="text-white/60 text-sm">
                    Proposed by {selectedInitiative.proposedBy} on {new Date(selectedInitiative.proposedDate).toLocaleDateString()}
                  </p>
                </div>
                <button
                  onClick={() => setSelectedInitiative(null)}
                  className="text-white/60 hover:text-white transition-all text-2xl"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-6">
                {/* Description */}
                <div>
                  <h4 className="text-sm text-white/60 uppercase tracking-wider mb-2">Description</h4>
                  <p className="text-white/80">{selectedInitiative.description}</p>
                </div>

                {/* Voting Progress Detailed */}
                <div>
                  <h4 className="text-sm text-white/60 uppercase tracking-wider mb-3">Voting Progress</h4>
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm text-emerald-400">Yes</span>
                        <span className="text-sm text-white">{selectedInitiative.currentVotes.yes} votes</span>
                      </div>
                      <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-emerald-500"
                          style={{ width: `${calculateVotePercentage(selectedInitiative.currentVotes).yes}%` }}
                        />
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm text-red-400">No</span>
                        <span className="text-sm text-white">{selectedInitiative.currentVotes.no} votes</span>
                      </div>
                      <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-red-500"
                          style={{ width: `${calculateVotePercentage(selectedInitiative.currentVotes).no}%` }}
                        />
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm text-gray-400">Abstain</span>
                        <span className="text-sm text-white">{selectedInitiative.currentVotes.abstain} votes</span>
                      </div>
                      <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-gray-500"
                          style={{ width: `${calculateVotePercentage(selectedInitiative.currentVotes).abstain}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>

                {/* Community Impact Details */}
                <div>
                  <h4 className="text-sm text-white/60 uppercase tracking-wider mb-3">Expected Community Impact</h4>
                  <div className="grid grid-cols-3 gap-4 p-4 bg-white/5 rounded-xl">
                    <div>
                      <div className="text-2xl text-emerald-400">+{selectedInitiative.communityImpact.jobs}</div>
                      <div className="text-xs text-white/60">Jobs Created</div>
                    </div>
                    <div>
                      <div className="text-2xl text-cyan-400">{selectedInitiative.communityImpact.households}</div>
                      <div className="text-xs text-white/60">Households Benefited</div>
                    </div>
                    <div>
                      <div className="text-2xl text-purple-400">-{selectedInitiative.communityImpact.co2Reduction}</div>
                      <div className="text-xs text-white/60">Tons CO₂ Reduced</div>
                    </div>
                  </div>
                </div>

                {/* Investment Details */}
                <div>
                  <h4 className="text-sm text-white/60 uppercase tracking-wider mb-3">Financial Requirements</h4>
                  <div className="p-4 bg-gradient-to-r from-cyan-500/10 to-blue-500/10 rounded-xl">
                    <div className="text-3xl text-cyan-400 mb-2">
                      ${(selectedInitiative.requiredInvestment / 1000000).toFixed(2)}M
                    </div>
                    <div className="text-sm text-white/60">Total Investment Required</div>
                  </div>
                </div>

                {/* Voting Requirements */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 bg-white/5 rounded-xl">
                    <div className="text-xl text-amber-400">{selectedInitiative.quorum}%</div>
                    <div className="text-xs text-white/60">Minimum Participation</div>
                  </div>
                  <div className="p-4 bg-white/5 rounded-xl">
                    <div className="text-xl text-green-400">{selectedInitiative.passingThreshold}%</div>
                    <div className="text-xs text-white/60">Approval Required</div>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Proposal Form Modal */}
      <AnimatePresence>
        {showProposalForm && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 backdrop-blur-md z-50 flex items-center justify-center p-6"
            onClick={() => setShowProposalForm(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-gradient-to-br from-gray-900 to-black border border-white/20 rounded-2xl p-8 max-w-2xl w-full"
              onClick={(e) => e.stopPropagation()}
            >
              <h3 className="text-2xl font-light text-white mb-6">Propose New Initiative</h3>
              <form className="space-y-4">
                <div>
                  <label className="block text-sm text-white/60 mb-2">Title</label>
                  <input
                    type="text"
                    className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-lg text-white placeholder-white/40 focus:border-purple-500 focus:outline-none"
                    placeholder="Enter initiative title"
                  />
                </div>
                <div>
                  <label className="block text-sm text-white/60 mb-2">Category</label>
                  <select className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-lg text-white focus:border-purple-500 focus:outline-none">
                    <option value="infrastructure">Infrastructure</option>
                    <option value="education">Education</option>
                    <option value="governance">Governance</option>
                    <option value="sustainability">Sustainability</option>
                    <option value="economic">Economic</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm text-white/60 mb-2">Description</label>
                  <textarea
                    className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-lg text-white placeholder-white/40 focus:border-purple-500 focus:outline-none h-32"
                    placeholder="Describe your initiative"
                  />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm text-white/60 mb-2">Required Investment ($)</label>
                    <input
                      type="number"
                      className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-lg text-white placeholder-white/40 focus:border-purple-500 focus:outline-none"
                      placeholder="0"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-white/60 mb-2">Voting Period (days)</label>
                    <input
                      type="number"
                      className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-lg text-white placeholder-white/40 focus:border-purple-500 focus:outline-none"
                      placeholder="30"
                    />
                  </div>
                </div>
                <div className="flex gap-4">
                  <button
                    type="button"
                    onClick={() => setShowProposalForm(false)}
                    className="flex-1 px-6 py-3 bg-white/5 border border-white/20 rounded-lg text-white hover:bg-white/10 transition-all"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="flex-1 px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg text-white font-medium hover:shadow-lg hover:shadow-purple-500/25 transition-all"
                  >
                    Submit Proposal
                  </button>
                </div>
              </form>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}