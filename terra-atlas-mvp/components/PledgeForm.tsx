'use client'

import { useState } from 'react'
import { ConsciousnessScore } from './ConsciousnessScore'

interface PledgeFormProps {
  projectId: string
  projectName: string
  consciousnessScore?: number
  consciousnessTier?: string
  availableBalance?: number
  currency?: string
  onSubmit?: (pledge: PledgeData) => void
  className?: string
}

interface PledgeData {
  projectId: string
  amount: number
  currency: string
  harmonyIntent: string
  consciousnessScore: number
  consciousnessTier: string
}

const HARMONY_OPTIONS = [
  'Resonant Coherence',
  'Pan-Sentient Flourishing',
  'Ecological Reciprocity',
  'Epistemic Humility',
  'Radical Translucency',
  'Compassionate Action',
  'Temporal Stewardship',
  'Sacred Stillness',
]

const MIN_CONSCIOUSNESS = 0.2

/**
 * Consciousness-gated pledge submission form.
 * Requires Participant tier (consciousness >= 0.2) to submit.
 */
export function PledgeForm({
  projectId,
  projectName,
  consciousnessScore = 0,
  consciousnessTier = 'Observer',
  availableBalance = 0,
  currency = 'TEND',
  onSubmit,
  className = '',
}: PledgeFormProps) {
  const [amount, setAmount] = useState('')
  const [harmonyIntent, setHarmonyIntent] = useState(HARMONY_OPTIONS[2])
  const [customIntent, setCustomIntent] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [submitted, setSubmitted] = useState(false)

  const canPledge = consciousnessScore >= MIN_CONSCIOUSNESS
  const parsedAmount = parseInt(amount) || 0
  const validAmount = parsedAmount > 0 && parsedAmount <= availableBalance

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!canPledge || !validAmount) return

    setSubmitting(true)
    try {
      const pledge: PledgeData = {
        projectId,
        amount: parsedAmount,
        currency,
        harmonyIntent: harmonyIntent === 'Custom' ? customIntent : harmonyIntent,
        consciousnessScore,
        consciousnessTier,
      }
      onSubmit?.(pledge)
      setSubmitted(true)
    } finally {
      setSubmitting(false)
    }
  }

  if (submitted) {
    return (
      <div className={`rounded-xl bg-emerald-900/20 border border-emerald-700/50 p-4 ${className}`}>
        <div className="text-center">
          <p className="text-emerald-300 font-medium mt-2">Pledge Submitted</p>
          <p className="text-gray-400 text-sm mt-1">
            {parsedAmount} {currency} pledged toward {projectName}
          </p>
          <p className="text-gray-500 text-xs mt-2">
            Expires in 24 hours. You will be notified if matched.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className={`rounded-xl bg-gray-900/60 border border-gray-800 p-4 ${className}`}>
      <h3 className="text-sm font-semibold text-gray-200 mb-3">Pledge Resources</h3>

      <div className="flex items-center justify-between mb-4 p-2.5 rounded-lg bg-gray-800/60">
        <div>
          <span className="text-xs text-gray-400">Your Consciousness</span>
          <div className="flex items-center gap-2 mt-0.5">
            <ConsciousnessScore phiScore={consciousnessScore} harmonyAlignment={consciousnessScore} size="sm" showLabel={false} />
            <span className="text-xs text-gray-400">{consciousnessTier}</span>
          </div>
        </div>
        {canPledge ? (
          <span className="text-xs text-emerald-400 bg-emerald-900/30 px-2 py-1 rounded">Eligible</span>
        ) : (
          <span className="text-xs text-red-400 bg-red-900/30 px-2 py-1 rounded">
            Requires Participant tier
          </span>
        )}
      </div>

      <form onSubmit={handleSubmit} className="space-y-3">
        <div>
          <label className="text-xs text-gray-400 block mb-1">Amount ({currency})</label>
          <div className="flex items-center gap-2">
            <input
              type="number"
              min="1"
              max={availableBalance}
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              placeholder="0"
              disabled={!canPledge}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 font-mono focus:border-violet-500 focus:outline-none disabled:opacity-50"
            />
            <span className="text-xs text-gray-500 whitespace-nowrap">
              / {availableBalance} avail
            </span>
          </div>
        </div>

        <div>
          <label className="text-xs text-gray-400 block mb-1">Harmony Intent</label>
          <select
            value={harmonyIntent}
            onChange={(e) => setHarmonyIntent(e.target.value)}
            disabled={!canPledge}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 focus:border-violet-500 focus:outline-none disabled:opacity-50"
          >
            {HARMONY_OPTIONS.map((h) => (
              <option key={h} value={h}>{h}</option>
            ))}
            <option value="Custom">Custom...</option>
          </select>
          {harmonyIntent === 'Custom' && (
            <input
              type="text"
              value={customIntent}
              onChange={(e) => setCustomIntent(e.target.value)}
              placeholder="Describe your harmony intent..."
              className="w-full mt-2 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 focus:border-violet-500 focus:outline-none"
            />
          )}
        </div>

        <button
          type="submit"
          disabled={!canPledge || !validAmount || submitting}
          className="w-full py-2.5 rounded-lg font-medium text-sm transition-all bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500 text-white shadow-lg shadow-violet-500/20 disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none"
        >
          {submitting ? 'Submitting...' : `Pledge ${parsedAmount || '...'} ${currency}`}
        </button>

        <p className="text-xs text-gray-500 text-center">
          Pledges expire in 24h. Demurrage applies to matched pledges.
        </p>
      </form>
    </div>
  )
}
