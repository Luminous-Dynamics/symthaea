'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

// 9 Stages of Community Readiness (based on strategic recommendations)
const readinessStages = [
  { id: 1, name: 'Unaware', description: 'No awareness of energy transition opportunities' },
  { id: 2, name: 'Denial', description: 'Recognition but belief that it doesn\'t apply locally' },
  { id: 3, name: 'Vague Awareness', description: 'General awareness but no concrete knowledge' },
  { id: 4, name: 'Preplanning', description: 'Clear recognition, gathering information' },
  { id: 5, name: 'Preparation', description: 'Active planning, building leadership' },
  { id: 6, name: 'Initiation', description: 'Beginning implementation, early projects' },
  { id: 7, name: 'Stabilization', description: 'Ongoing implementation with stable support' },
  { id: 8, name: 'Expansion', description: 'Growing programs, increasing capacity' },
  { id: 9, name: 'Advocacy', description: 'Community ownership, helping other communities' }
]

// 5 Dimensions of Community Capacity
const dimensions = [
  {
    name: 'Knowledge',
    icon: '🧠',
    description: 'Understanding of energy transition and investment opportunities',
    indicators: [
      'Technical literacy',
      'Financial understanding',
      'Regulatory awareness',
      'Environmental knowledge'
    ]
  },
  {
    name: 'Leadership',
    icon: '👥',
    description: 'Local champions and organized governance structures',
    indicators: [
      'Community champions',
      'Organizational capacity',
      'Decision-making processes',
      'External partnerships'
    ]
  },
  {
    name: 'Resources',
    icon: '💰',
    description: 'Financial, human, and technical resources available',
    indicators: [
      'Local investment capacity',
      'Technical expertise',
      'Available land/sites',
      'Infrastructure access'
    ]
  },
  {
    name: 'Community Climate',
    icon: '🌡️',
    description: 'Social cohesion and collective will for change',
    indicators: [
      'Public support',
      'Political will',
      'Trust levels',
      'Collaboration history'
    ]
  },
  {
    name: 'Infrastructure',
    icon: '🏗️',
    description: 'Physical and institutional foundations',
    indicators: [
      'Grid connectivity',
      'Transportation access',
      'Digital infrastructure',
      'Regulatory framework'
    ]
  }
]

interface CommunityAssessment {
  knowledge: number
  leadership: number
  resources: number
  climate: number
  infrastructure: number
}

export default function CommunityReadinessFramework() {
  const [selectedDimension, setSelectedDimension] = useState<string>('Knowledge')
  const [communityStage, setCommunityStage] = useState<number>(4) // Default to Preplanning
  const [showAssessment, setShowAssessment] = useState(false)
  const [assessment, setAssessment] = useState<CommunityAssessment>({
    knowledge: 4,
    leadership: 3,
    resources: 5,
    climate: 4,
    infrastructure: 6
  })

  // Calculate overall readiness stage based on dimension scores
  const calculateOverallStage = () => {
    const scores = Object.values(assessment)
    const average = scores.reduce((a, b) => a + b, 0) / scores.length
    return Math.round(average)
  }

  // Get stage color based on progress
  const getStageColor = (stage: number) => {
    if (stage <= 3) return 'from-red-500 to-orange-500'
    if (stage <= 6) return 'from-amber-500 to-yellow-500'
    return 'from-emerald-500 to-green-500'
  }

  // Get progress percentage
  const getProgress = () => {
    return (calculateOverallStage() / 9) * 100
  }

  return (
    <div className="w-full max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-3xl font-light text-white mb-3">
          <span className="text-4xl mr-3">🌱</span>
          Community Readiness Framework
        </h2>
        <p className="text-white/60 max-w-3xl mx-auto">
          Assess your community's capacity for energy transition and identify pathways to community ownership
        </p>
      </div>

      {/* Overall Progress Indicator */}
      <div className="bg-white/5 backdrop-blur-md rounded-2xl border border-white/10 p-8">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-xl font-light text-white mb-2">
              Current Stage: <span className="text-emerald-400 font-medium">
                {readinessStages[calculateOverallStage() - 1].name}
              </span>
            </h3>
            <p className="text-white/60 text-sm max-w-lg">
              {readinessStages[calculateOverallStage() - 1].description}
            </p>
          </div>
          <div className="text-right">
            <div className="text-4xl font-light text-white">
              {Math.round(getProgress())}%
            </div>
            <div className="text-xs text-white/60 uppercase tracking-wider">
              Ready
            </div>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="relative h-3 bg-white/10 rounded-full overflow-hidden">
          <motion.div
            className={`h-full bg-gradient-to-r ${getStageColor(calculateOverallStage())}`}
            initial={{ width: 0 }}
            animate={{ width: `${getProgress()}%` }}
            transition={{ duration: 1, ease: 'easeOut' }}
          />
        </div>

        {/* Stage Markers */}
        <div className="mt-6 grid grid-cols-9 gap-2">
          {readinessStages.map((stage) => (
            <button
              key={stage.id}
              onClick={() => setCommunityStage(stage.id)}
              className={`group relative`}
            >
              <div className={`h-2 rounded-full transition-all ${
                stage.id <= calculateOverallStage()
                  ? 'bg-emerald-500'
                  : 'bg-white/20'
              } ${communityStage === stage.id ? 'ring-2 ring-emerald-500 ring-offset-2 ring-offset-black' : ''}`} />
              
              {/* Stage tooltip */}
              <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                <div className="bg-black/90 backdrop-blur-xl border border-white/20 rounded-lg px-3 py-2 whitespace-nowrap">
                  <div className="text-xs text-white font-medium">{stage.name}</div>
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Dimension Assessment */}
      <div className="grid md:grid-cols-5 gap-4">
        {dimensions.map((dimension) => (
          <motion.button
            key={dimension.name}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setSelectedDimension(dimension.name)}
            className={`relative p-6 rounded-2xl bg-white/5 backdrop-blur-md border transition-all ${
              selectedDimension === dimension.name
                ? 'border-emerald-500/40 bg-white/10'
                : 'border-white/10 hover:border-white/20'
            }`}
          >
            <div className="text-3xl mb-3">{dimension.icon}</div>
            <h4 className="text-white font-medium mb-1">{dimension.name}</h4>
            
            {/* Mini progress bar */}
            <div className="mt-3 h-1 bg-white/10 rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500"
                style={{ width: `${(assessment[dimension.name.toLowerCase() as keyof CommunityAssessment] / 9) * 100}%` }}
              />
            </div>
            
            <div className="mt-2 text-xs text-white/60">
              Stage {assessment[dimension.name.toLowerCase() as keyof CommunityAssessment]}
            </div>

            {selectedDimension === dimension.name && (
              <motion.div
                layoutId="dimensionHighlight"
                className="absolute inset-0 bg-gradient-to-br from-emerald-500/10 to-cyan-500/10 rounded-2xl"
              />
            )}
          </motion.button>
        ))}
      </div>

      {/* Dimension Details */}
      <AnimatePresence mode="wait">
        {selectedDimension && (
          <motion.div
            key={selectedDimension}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="bg-gradient-to-br from-emerald-500/10 to-cyan-500/10 backdrop-blur-md rounded-2xl border border-emerald-500/20 p-8"
          >
            <div className="flex items-start justify-between mb-6">
              <div>
                <h3 className="text-2xl font-light text-white mb-2 flex items-center gap-3">
                  <span className="text-3xl">
                    {dimensions.find(d => d.name === selectedDimension)?.icon}
                  </span>
                  {selectedDimension}
                </h3>
                <p className="text-white/60 max-w-2xl">
                  {dimensions.find(d => d.name === selectedDimension)?.description}
                </p>
              </div>
              <div className="text-right">
                <div className="text-3xl font-light text-emerald-400">
                  Stage {assessment[selectedDimension.toLowerCase() as keyof CommunityAssessment]}
                </div>
                <div className="text-xs text-white/60 uppercase tracking-wider">
                  Current Level
                </div>
              </div>
            </div>

            {/* Key Indicators */}
            <div className="mb-6">
              <h4 className="text-sm text-white/60 uppercase tracking-wider mb-3">
                Key Indicators
              </h4>
              <div className="grid grid-cols-2 gap-3">
                {dimensions.find(d => d.name === selectedDimension)?.indicators.map((indicator, i) => (
                  <div key={i} className="flex items-center gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                    <span className="text-sm text-white/80">{indicator}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Actions to Advance */}
            <div className="bg-white/5 rounded-xl p-4">
              <h4 className="text-sm text-white/60 uppercase tracking-wider mb-3">
                Actions to Advance
              </h4>
              <div className="space-y-2">
                {getAdvancementActions(
                  selectedDimension, 
                  assessment[selectedDimension.toLowerCase() as keyof CommunityAssessment]
                ).map((action, i) => (
                  <div key={i} className="flex items-start gap-2">
                    <span className="text-emerald-400 mt-0.5">→</span>
                    <span className="text-sm text-white/80">{action}</span>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Assessment Tool */}
      <div className="text-center">
        <button
          onClick={() => setShowAssessment(!showAssessment)}
          className="px-8 py-3 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-full text-white font-medium hover:shadow-lg hover:shadow-emerald-500/25 transition-all transform hover:scale-105"
        >
          {showAssessment ? 'Hide' : 'Take'} Full Assessment
        </button>
      </div>

      {/* Full Assessment Modal */}
      <AnimatePresence>
        {showAssessment && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-white/5 backdrop-blur-md rounded-2xl border border-white/10 p-8"
          >
            <h3 className="text-xl font-light text-white mb-6">
              Community Readiness Assessment
            </h3>
            
            <div className="space-y-6">
              {dimensions.map((dimension) => (
                <div key={dimension.name}>
                  <div className="flex items-center justify-between mb-3">
                    <label className="text-white flex items-center gap-2">
                      <span className="text-xl">{dimension.icon}</span>
                      {dimension.name}
                    </label>
                    <span className="text-emerald-400">
                      Stage {assessment[dimension.name.toLowerCase() as keyof CommunityAssessment]}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="9"
                    value={assessment[dimension.name.toLowerCase() as keyof CommunityAssessment]}
                    onChange={(e) => setAssessment({
                      ...assessment,
                      [dimension.name.toLowerCase()]: parseInt(e.target.value)
                    })}
                    className="w-full accent-emerald-500"
                  />
                  <div className="flex justify-between text-xs text-white/40 mt-1">
                    <span>Unaware</span>
                    <span>Advocacy</span>
                  </div>
                </div>
              ))}
            </div>

            {/* Results Summary */}
            <div className="mt-8 p-6 bg-gradient-to-br from-emerald-500/10 to-cyan-500/10 rounded-xl">
              <h4 className="text-lg font-light text-white mb-4">
                Assessment Results
              </h4>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-sm text-white/60 mb-1">Overall Stage</div>
                  <div className="text-2xl text-emerald-400 font-light">
                    {readinessStages[calculateOverallStage() - 1].name}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-white/60 mb-1">Readiness Score</div>
                  <div className="text-2xl text-cyan-400 font-light">
                    {Math.round(getProgress())}%
                  </div>
                </div>
              </div>
              
              {/* Recommendations */}
              <div className="mt-4 pt-4 border-t border-white/10">
                <p className="text-sm text-white/80">
                  {getRecommendation(calculateOverallStage())}
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// Helper function to get advancement actions
function getAdvancementActions(dimension: string, currentStage: number): string[] {
  const actions: Record<string, Record<number, string[]>> = {
    'Knowledge': {
      1: ['Host initial community information session', 'Distribute basic educational materials'],
      2: ['Bring in expert speakers', 'Create local success stories database'],
      3: ['Organize site visits to operating projects', 'Start technical training programs'],
      4: ['Develop local expertise through partnerships', 'Create knowledge-sharing networks'],
      5: ['Establish ongoing education programs', 'Build technical assistance capacity'],
      6: ['Document and share learnings', 'Mentor other communities'],
      7: ['Lead regional knowledge networks', 'Develop best practices'],
      8: ['Innovate new approaches', 'Publish research'],
      9: ['Guide national policy', 'Shape industry standards']
    },
    'Leadership': {
      1: ['Identify potential champions', 'Gauge initial interest'],
      2: ['Build core team', 'Define vision'],
      3: ['Formalize organizational structure', 'Develop governance'],
      4: ['Expand leadership circle', 'Build coalitions'],
      5: ['Strengthen decision processes', 'Increase representation'],
      6: ['Develop succession planning', 'Build youth engagement'],
      7: ['Create leadership pipeline', 'Expand partnerships'],
      8: ['Lead regional initiatives', 'Influence policy'],
      9: ['Model for other communities', 'Drive systemic change']
    },
    'Resources': {
      1: ['Assess baseline capacity', 'Identify funding sources'],
      2: ['Secure initial seed funding', 'Map available assets'],
      3: ['Develop funding strategy', 'Build technical capacity'],
      4: ['Secure project development funds', 'Attract expertise'],
      5: ['Structure investment vehicles', 'Build local ownership'],
      6: ['Scale funding capacity', 'Develop revenue streams'],
      7: ['Create sustainable financing', 'Build reserves'],
      8: ['Leverage for larger projects', 'Create funds for others'],
      9: ['Lead investment initiatives', 'Support other communities']
    },
    'Community Climate': {
      1: ['Conduct initial surveys', 'Understand concerns'],
      2: ['Address misconceptions', 'Build awareness'],
      3: ['Create dialogue spaces', 'Build trust'],
      4: ['Develop shared vision', 'Build consensus'],
      5: ['Strengthen collaboration', 'Celebrate early wins'],
      6: ['Deepen engagement', 'Expand participation'],
      7: ['Institutionalize support', 'Create traditions'],
      8: ['Model collaboration', 'Share practices'],
      9: ['Inspire movements', 'Transform culture']
    },
    'Infrastructure': {
      1: ['Assess current infrastructure', 'Identify gaps'],
      2: ['Plan basic improvements', 'Map opportunities'],
      3: ['Secure infrastructure funding', 'Start upgrades'],
      4: ['Implement key improvements', 'Build connectivity'],
      5: ['Complete critical infrastructure', 'Enhance capacity'],
      6: ['Optimize systems', 'Build redundancy'],
      7: ['Create advanced infrastructure', 'Lead innovation'],
      8: ['Share infrastructure models', 'Support region'],
      9: ['Set infrastructure standards', 'Enable others']
    }
  }

  const stageActions = actions[dimension]?.[Math.min(currentStage, 9)] || []
  return stageActions
}

// Helper function to get recommendation based on stage
function getRecommendation(stage: number): string {
  const recommendations: Record<number, string> = {
    1: "Your community is just beginning its journey. Focus on awareness-building and identifying early champions.",
    2: "Address concerns and misconceptions while building a core group of supporters.",
    3: "Build concrete knowledge and start developing organizational capacity.",
    4: "You're ready for serious planning. Develop comprehensive strategies and secure initial resources.",
    5: "Begin implementing pilot projects while strengthening community support systems.",
    6: "Scale successful initiatives and establish sustainable financing mechanisms.",
    7: "Maintain momentum while building long-term institutional capacity.",
    8: "Share your success with other communities while continuing to innovate.",
    9: "Your community is a leader! Focus on systemic change and supporting others."
  }
  
  return recommendations[stage] || "Continue building capacity across all dimensions."
}