# Governance UI/UX Design & Implementation Plan

**Version**: 1.0
**Date**: November 11, 2025
**Status**: Design Phase
**Dependencies**: Week 9-10 Production Deployment

---

## Executive Summary

Design and implement accessible web interface for Zero-TrustML governance, enabling non-technical participants to:
- Create and vote on proposals via web forms
- Track proposal status in real-time
- Review and approve guardian authorization requests
- Visualize governance analytics
- Participate via mobile devices

**Timeline**: 15-23 days
**Stack**: React + Next.js + Web3 + Tailwind CSS + Recharts

---

## Design Principles

### 1. Progressive Disclosure
- Show simple interface by default
- Advanced features available but hidden
- "Explain" tooltips for complex concepts

### 2. Accessibility First
- WCAG 2.1 AA compliance
- Keyboard navigation throughout
- Screen reader friendly
- High contrast mode

### 3. Mobile-First
- Touch-friendly interface
- Responsive design
- Progressive Web App (PWA)
- Offline-capable where possible

### 4. Real-Time Updates
- WebSocket connections for live data
- Optimistic UI updates
- Background sync

---

## Information Architecture

```
Governance Dashboard
├── Home
│   ├── Active Proposals (Card Grid)
│   ├── My Voting Power (Widget)
│   ├── Recent Activity (Timeline)
│   └── Quick Actions (CTAs)
├── Proposals
│   ├── All Proposals (List/Grid View)
│   ├── Create Proposal (Form Wizard)
│   ├── Proposal Detail (Full View)
│   └── My Proposals (Filtered List)
├── Voting
│   ├── Cast Vote (Modal)
│   ├── Vote History (Table)
│   └── Vote Calculator (Tool)
├── Guardian
│   ├── Authorization Requests (Queue)
│   ├── Guardian Network (Graph)
│   └── Response History (Timeline)
├── Analytics
│   ├── Participation Metrics (Charts)
│   ├── Vote Distribution (Visualizations)
│   ├── Proposal Success Rate (Graphs)
│   └── Network Health (Dashboard)
└── Profile
    ├── Identity Overview (Card)
    ├── Reputation Score (Badge)
    ├── Guardian Relationships (List)
    └── Notification Settings (Form)
```

---

## Track 3.1: Core Governance Interface (7-10 days)

### Home Dashboard (2 days)

#### Wireframe

```
┌─────────────────────────────────────────────────────────────┐
│ Zero-TrustML Governance                          [Profile]▼ │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Your Voting Power                                     │  │
│  │  ┌────┐  Weight: 5.13x                               │  │
│  │  │5.13│  Budget: 513 credits                          │  │
│  │  └────┘  Reputation: 0.9 (High)                       │  │
│  │          Assurance: E4 (Verified)                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
│  Active Proposals (3)              [Create Proposal] [View]  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ PARAM_CH... │ │ BAN_PARTIC..│ │ EMERGENCY.. │          │
│  │             │ │             │ │             │          │
│  │ 42% FOR     │ │ 78% FOR     │ │ 35% FOR     │          │
│  │ 3 days left │ │ 1 day left  │ │ 5 days left │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
│                                                               │
│  Recent Activity                                              │
│  🗳️  Alice voted FOR on "Increase learning rate"           │
│  📝  Bob created proposal "Ban malicious node"              │
│  ✅  Proposal "Update parameters" APPROVED                   │
│  🛡️  Guardian request pending your approval                 │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

#### Implementation

```typescript
// pages/index.tsx
import { useGovernance } from '../hooks/useGovernance';
import { VotingPowerWidget } from '../components/VotingPowerWidget';
import { ProposalCard } from '../components/ProposalCard';
import { ActivityTimeline } from '../components/ActivityTimeline';

export default function Home() {
  const { activeProposals, votingPower, recentActivity } = useGovernance();

  return (
    <div className="container mx-auto px-4 py-8">
      <VotingPowerWidget
        voteWeight={votingPower.weight}
        budget={votingPower.budget}
        reputation={votingPower.reputation}
        assurance={votingPower.assurance}
      />

      <section className="mt-8">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-bold">Active Proposals</h2>
          <div className="space-x-2">
            <Button onClick={() => router.push('/proposals/create')}>
              Create Proposal
            </Button>
            <Button variant="outline" onClick={() => router.push('/proposals')}>
              View All
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {activeProposals.map(proposal => (
            <ProposalCard key={proposal.id} proposal={proposal} />
          ))}
        </div>
      </section>

      <section className="mt-8">
        <h2 className="text-2xl font-bold mb-4">Recent Activity</h2>
        <ActivityTimeline activities={recentActivity} />
      </section>
    </div>
  );
}
```

### Proposal List & Detail (2-3 days)

#### Proposal List View

```typescript
// pages/proposals/index.tsx
import { useState } from 'react';
import { useProposals } from '../../hooks/useProposals';
import { ProposalFilters } from '../../components/ProposalFilters';
import { ProposalTable } from '../../components/ProposalTable';

export default function ProposalsPage() {
  const [filters, setFilters] = useState({
    status: 'all',
    type: 'all',
    sortBy: 'created_at'
  });

  const { proposals, loading } = useProposals(filters);

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-6">Governance Proposals</h1>

      <ProposalFilters
        filters={filters}
        onChange={setFilters}
      />

      {loading ? (
        <LoadingSpinner />
      ) : (
        <ProposalTable proposals={proposals} />
      )}
    </div>
  );
}
```

#### Proposal Detail View

```typescript
// pages/proposals/[id].tsx
import { useProposal } from '../../hooks/useProposal';
import { VoteButton } from '../../components/VoteButton';
import { ProposalTimeline } from '../../components/ProposalTimeline';
import { VoteDistribution } from '../../components/VoteDistribution';

export default function ProposalDetailPage({ id }) {
  const { proposal, votes, canVote } = useProposal(id);

  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      {/* Proposal Header */}
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <div className="flex justify-between items-start mb-4">
          <div>
            <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
              {proposal.type}
            </span>
            <h1 className="text-3xl font-bold mt-2">{proposal.title}</h1>
          </div>
          <ProposalStatus status={proposal.status} />
        </div>

        <div className="flex space-x-4 text-sm text-gray-600">
          <span>Proposed by {proposal.proposer}</span>
          <span>•</span>
          <span>{proposal.votingTimeLeft} left</span>
          <span>•</span>
          <span>{proposal.participation}% participation</span>
        </div>
      </div>

      {/* Voting Interface */}
      {canVote && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
          <p className="font-semibold mb-2">You haven't voted yet</p>
          <VoteButton proposalId={id} />
        </div>
      )}

      {/* Vote Distribution */}
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <h2 className="text-xl font-bold mb-4">Vote Distribution</h2>
        <VoteDistribution votes={votes} />
      </div>

      {/* Proposal Description */}
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <h2 className="text-xl font-bold mb-4">Description</h2>
        <ReactMarkdown>{proposal.description}</ReactMarkdown>
      </div>

      {/* Execution Parameters */}
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <h2 className="text-xl font-bold mb-4">Execution Parameters</h2>
        <pre className="bg-gray-100 p-4 rounded overflow-x-auto">
          {JSON.stringify(proposal.executionParams, null, 2)}
        </pre>
      </div>

      {/* Evidence (if any) */}
      {proposal.evidenceCids && (
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h2 className="text-xl font-bold mb-4">Evidence</h2>
          <EvidenceViewer cids={proposal.evidenceCids} />
        </div>
      )}

      {/* Timeline */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-bold mb-4">Timeline</h2>
        <ProposalTimeline proposal={proposal} />
      </div>
    </div>
  );
}
```

### Create Proposal Wizard (2-3 days)

```typescript
// components/CreateProposalWizard.tsx
import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { ProposalType } from '../types';

const STEPS = [
  'Type Selection',
  'Basic Info',
  'Execution Parameters',
  'Evidence Upload',
  'Review & Submit'
];

export function CreateProposalWizard() {
  const [currentStep, setCurrentStep] = useState(0);
  const [proposalType, setProposalType] = useState<ProposalType | null>(null);
  const { register, handleSubmit, watch, formState: { errors } } = useForm();

  const onSubmit = async (data) => {
    // Submit proposal
    const result = await createProposal({
      type: proposalType,
      title: data.title,
      description: data.description,
      executionParams: data.executionParams,
      evidenceFiles: data.evidenceFiles
    });

    if (result.success) {
      router.push(`/proposals/${result.proposalId}`);
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      {/* Progress Steps */}
      <div className="mb-8">
        <div className="flex justify-between">
          {STEPS.map((step, index) => (
            <div
              key={step}
              className={`flex-1 ${index < currentStep ? 'text-green-600' : index === currentStep ? 'text-blue-600' : 'text-gray-400'}`}
            >
              <div className="flex items-center">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center border-2 ${
                  index < currentStep ? 'border-green-600 bg-green-600 text-white' :
                  index === currentStep ? 'border-blue-600 text-blue-600' :
                  'border-gray-300'
                }`}>
                  {index < currentStep ? '✓' : index + 1}
                </div>
                <span className="ml-2 text-sm">{step}</span>
              </div>
              {index < STEPS.length - 1 && (
                <div className={`h-1 mt-4 ${index < currentStep ? 'bg-green-600' : 'bg-gray-300'}`} />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Step Content */}
      <form onSubmit={handleSubmit(onSubmit)}>
        {currentStep === 0 && (
          <ProposalTypeSelector
            selected={proposalType}
            onChange={setProposalType}
          />
        )}

        {currentStep === 1 && (
          <BasicInfoForm
            register={register}
            errors={errors}
          />
        )}

        {currentStep === 2 && (
          <ExecutionParamsForm
            proposalType={proposalType}
            register={register}
            errors={errors}
          />
        )}

        {currentStep === 3 && (
          <EvidenceUploadForm
            onUpload={(files) => setEvidenceFiles(files)}
          />
        )}

        {currentStep === 4 && (
          <ReviewAndSubmit
            data={watch()}
            proposalType={proposalType}
          />
        )}

        {/* Navigation Buttons */}
        <div className="flex justify-between mt-8">
          <Button
            type="button"
            variant="outline"
            onClick={() => setCurrentStep(currentStep - 1)}
            disabled={currentStep === 0}
          >
            Previous
          </Button>

          {currentStep < STEPS.length - 1 ? (
            <Button
              type="button"
              onClick={() => setCurrentStep(currentStep + 1)}
            >
              Next
            </Button>
          ) : (
            <Button type="submit">
              Submit Proposal
            </Button>
          )}
        </div>
      </form>
    </div>
  );
}
```

### Vote Casting Interface (1-2 days)

```typescript
// components/VoteCastModal.tsx
import { useState } from 'react';
import { Slider } from './ui/Slider';
import { VoteChoice } from '../types';

export function VoteCastModal({ proposalId, votingPower, onClose }) {
  const [choice, setChoice] = useState<VoteChoice>(VoteChoice.FOR);
  const [credits, setCredits] = useState(100);

  const effectiveVotes = Math.sqrt(credits) * votingPower.weight;
  const remainingBudget = votingPower.budget - credits;

  const handleVote = async () => {
    const result = await castVote({
      proposalId,
      choice,
      credits
    });

    if (result.success) {
      toast.success('Vote cast successfully!');
      onClose();
    }
  };

  return (
    <Modal isOpen onClose={onClose}>
      <div className="p-6">
        <h2 className="text-2xl font-bold mb-4">Cast Your Vote</h2>

        {/* Vote Choice */}
        <div className="mb-6">
          <label className="block text-sm font-medium mb-2">Your Choice</label>
          <div className="flex space-x-4">
            <button
              onClick={() => setChoice(VoteChoice.FOR)}
              className={`flex-1 py-3 rounded-lg border-2 ${
                choice === VoteChoice.FOR
                  ? 'border-green-500 bg-green-50'
                  : 'border-gray-300'
              }`}
            >
              👍 For
            </button>
            <button
              onClick={() => setChoice(VoteChoice.AGAINST)}
              className={`flex-1 py-3 rounded-lg border-2 ${
                choice === VoteChoice.AGAINST
                  ? 'border-red-500 bg-red-50'
                  : 'border-gray-300'
              }`}
            >
              👎 Against
            </button>
            <button
              onClick={() => setChoice(VoteChoice.ABSTAIN)}
              className={`flex-1 py-3 rounded-lg border-2 ${
                choice === VoteChoice.ABSTAIN
                  ? 'border-gray-500 bg-gray-50'
                  : 'border-gray-300'
              }`}
            >
              🤷 Abstain
            </button>
          </div>
        </div>

        {/* Credits Slider */}
        <div className="mb-6">
          <div className="flex justify-between items-center mb-2">
            <label className="block text-sm font-medium">Credits to Spend</label>
            <span className="text-sm text-gray-600">
              Budget: {votingPower.budget} credits
            </span>
          </div>
          <Slider
            value={credits}
            onChange={setCredits}
            min={1}
            max={votingPower.budget}
            step={1}
          />
          <div className="text-sm text-gray-600 mt-1">
            Spending {credits} credits ({(credits / votingPower.budget * 100).toFixed(1)}% of budget)
          </div>
        </div>

        {/* Vote Power Calculation */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
          <h3 className="font-semibold mb-2">Your Vote Power</h3>
          <div className="space-y-1 text-sm">
            <div className="flex justify-between">
              <span>Credits:</span>
              <span className="font-mono">{credits}</span>
            </div>
            <div className="flex justify-between">
              <span>Vote Weight:</span>
              <span className="font-mono">{votingPower.weight.toFixed(2)}x</span>
            </div>
            <div className="flex justify-between border-t border-blue-300 pt-1 mt-1">
              <span className="font-semibold">Effective Votes:</span>
              <span className="font-mono font-semibold">{effectiveVotes.toFixed(2)}</span>
            </div>
          </div>
          <div className="text-xs text-gray-600 mt-2">
            Formula: √{credits} × {votingPower.weight.toFixed(2)} = {effectiveVotes.toFixed(2)}
          </div>
        </div>

        {/* Remaining Budget Warning */}
        {remainingBudget < votingPower.budget * 0.2 && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 mb-4">
            <p className="text-sm">
              ⚠️ You'll have {remainingBudget} credits remaining ({(remainingBudget / votingPower.budget * 100).toFixed(1)}% of budget)
            </p>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex space-x-4">
          <Button
            variant="outline"
            onClick={onClose}
            className="flex-1"
          >
            Cancel
          </Button>
          <Button
            onClick={handleVote}
            className="flex-1"
          >
            Cast Vote
          </Button>
        </div>
      </div>
    </Modal>
  );
}
```

**Deliverable**: Fully functional governance interface with proposal creation, viewing, and voting.

---

## Track 3.2: Guardian Interface (3-5 days)

### Authorization Request Queue (2-3 days)

```typescript
// pages/guardian/index.tsx
import { useGuardianRequests } from '../../hooks/useGuardianRequests';
import { AuthorizationRequestCard } from '../../components/AuthorizationRequestCard';

export default function GuardianDashboard() {
  const { pendingRequests, pastRequests } = useGuardianRequests();

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-6">Guardian Dashboard</h1>

      {pendingRequests.length > 0 ? (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
          <p className="font-semibold">
            🛡️ You have {pendingRequests.length} pending authorization request{pendingRequests.length > 1 ? 's' : ''}
          </p>
        </div>
      ) : (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-6">
          <p>✅ No pending authorization requests</p>
        </div>
      )}

      <section className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Pending Requests</h2>
        <div className="space-y-4">
          {pendingRequests.map(request => (
            <AuthorizationRequestCard
              key={request.id}
              request={request}
              onApprove={() => handleApprove(request.id)}
              onReject={() => handleReject(request.id)}
            />
          ))}
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">Past Requests</h2>
        <AuthorizationRequestTable requests={pastRequests} />
      </section>
    </div>
  );
}
```

### Guardian Network Visualization (1-2 days)

```typescript
// components/GuardianNetworkGraph.tsx
import { ForceGraph2D } from 'react-force-graph';
import { useGuardianNetwork } from '../hooks/useGuardianNetwork';

export function GuardianNetworkGraph({ participantId }) {
  const { nodes, links } = useGuardianNetwork(participantId);

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-xl font-bold mb-4">Your Guardian Network</h2>

      <ForceGraph2D
        graphData={{ nodes, links }}
        nodeLabel="name"
        nodeColor={node => {
          if (node.id === participantId) return '#3b82f6';  // Blue for you
          return node.isGuardian ? '#10b981' : '#6b7280';  // Green for guardians, gray for others
        }}
        linkColor={() => '#d1d5db'}
        nodeRelSize={8}
        width={800}
        height={600}
      />

      <div className="mt-4 flex space-x-4 text-sm">
        <div className="flex items-center">
          <div className="w-3 h-3 bg-blue-500 rounded-full mr-2" />
          <span>You</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 bg-green-500 rounded-full mr-2" />
          <span>Your Guardians</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 bg-gray-500 rounded-full mr-2" />
          <span>Others</span>
        </div>
      </div>
    </div>
  );
}
```

**Deliverable**: Guardian authorization interface with network visualization.

---

## Track 3.3: Analytics Dashboard (3-5 days)

### Governance Metrics (2-3 days)

```typescript
// pages/analytics/index.tsx
import { Line, Bar, Pie } from 'recharts';
import { useGovernanceMetrics } from '../../hooks/useGovernanceMetrics';

export default function AnalyticsPage() {
  const {
    proposalStats,
    participationTrend,
    voteDistribution,
    topVoters
  } = useGovernanceMetrics();

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-6">Governance Analytics</h1>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <MetricCard
          title="Total Proposals"
          value={proposalStats.total}
          change="+12%"
          trend="up"
        />
        <MetricCard
          title="Approval Rate"
          value={`${(proposalStats.approvalRate * 100).toFixed(1)}%`}
          change="-2%"
          trend="down"
        />
        <MetricCard
          title="Participation Rate"
          value={`${(proposalStats.participationRate * 100).toFixed(1)}%`}
          change="+5%"
          trend="up"
        />
        <MetricCard
          title="Active Voters"
          value={proposalStats.activeVoters}
          change="+18"
          trend="up"
        />
      </div>

      {/* Participation Trend */}
      <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
        <h2 className="text-xl font-bold mb-4">Participation Trend (Last 30 Days)</h2>
        <Line
          data={participationTrend}
          xDataKey="date"
          yDataKey="participation"
          height={300}
        />
      </div>

      {/* Vote Distribution */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-bold mb-4">Vote Distribution</h2>
          <Pie
            data={voteDistribution}
            dataKey="value"
            nameKey="name"
            cx="50%"
            cy="50%"
            outerRadius={80}
          />
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-bold mb-4">Top Voters</h2>
          <Bar
            data={topVoters}
            xDataKey="name"
            yDataKey="votesCast"
            height={300}
          />
        </div>
      </div>
    </div>
  );
}
```

### Real-Time Activity Feed (1-2 days)

```typescript
// components/RealTimeActivityFeed.tsx
import { useEffect, useState } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';

export function RealTimeActivityFeed() {
  const [activities, setActivities] = useState([]);
  const ws = useWebSocket('wss://governance.zerotrustml.io/ws');

  useEffect(() => {
    if (!ws) return;

    ws.on('activity', (activity) => {
      setActivities(prev => [activity, ...prev].slice(0, 50));
    });

    return () => ws.disconnect();
  }, [ws]);

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-xl font-bold mb-4">Live Activity</h2>
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {activities.map(activity => (
          <ActivityItem key={activity.id} activity={activity} />
        ))}
      </div>
    </div>
  );
}
```

**Deliverable**: Comprehensive analytics with real-time updates.

---

## Track 3.4: Mobile & PWA Support (2-3 days)

### Progressive Web App Configuration

```typescript
// next.config.js
const withPWA = require('next-pwa');

module.exports = withPWA({
  pwa: {
    dest: 'public',
    register: true,
    skipWaiting: true,
    disable: process.env.NODE_ENV === 'development'
  }
});
```

```json
// public/manifest.json
{
  "name": "Zero-TrustML Governance",
  "short_name": "ZTM Governance",
  "description": "Decentralized governance for Zero-TrustML",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#ffffff",
  "theme_color": "#3b82f6",
  "icons": [
    {
      "src": "/icons/icon-192x192.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "/icons/icon-512x512.png",
      "sizes": "512x512",
      "type": "image/png"
    }
  ]
}
```

### Mobile-Optimized Components

- Touch-friendly buttons (min 44x44px)
- Bottom navigation bar
- Swipe gestures for proposal navigation
- Pull-to-refresh
- Haptic feedback for votes

**Deliverable**: PWA with full mobile support.

---

## Track 3.5: Testing & Documentation (2-3 days)

### Testing Strategy

1. **Unit Tests** (Jest + React Testing Library)
2. **Integration Tests** (Cypress)
3. **Accessibility Tests** (axe-core)
4. **Performance Tests** (Lighthouse)
5. **User Acceptance Testing** (Real users)

### Documentation

- User guide (screencast videos)
- Admin guide
- Component storybook
- API documentation

**Deliverable**: Production-ready UI with full test coverage.

---

## Summary: Track 3 (UI/UX)

### Timeline

| Component | Duration | Key Deliverable |
|-----------|----------|-----------------|
| Core Interface | 7-10 days | Proposals + Voting |
| Guardian Interface | 3-5 days | Authorization UI |
| Analytics Dashboard | 3-5 days | Metrics + Charts |
| Mobile/PWA | 2-3 days | Mobile support |
| Testing & Docs | 2-3 days | Complete coverage |
| **Total** | **17-26 days** | Full governance UI |

### Tech Stack

- **Frontend**: React 18 + Next.js 14
- **Styling**: Tailwind CSS + Headless UI
- **Charts**: Recharts + D3.js
- **Forms**: React Hook Form + Zod
- **State**: Zustand + React Query
- **Web3**: Ethers.js + Wagmi
- **Testing**: Jest + Cypress + Axe
- **Deployment**: Vercel

### Cost Estimate

- **Vercel Hosting**: $20-100/month (Pro plan)
- **CDN**: Included
- **WebSocket**: $10-50/month (Pusher or self-hosted)
- **Monitoring**: $0-20/month (Vercel Analytics)

**Total**: ~$30-170/month

### Success Criteria

- ✅ WCAG 2.1 AA compliance
- ✅ Lighthouse score >90
- ✅ Mobile-responsive
- ✅ PWA installable
- ✅ Real-time updates
- ✅ <3s initial load
- ✅ Comprehensive user testing

---

**Status**: Design Complete - Ready for Implementation
**Recommendation**: Implement after Track 1 (Production Readiness)
**Next**: Create high-fidelity mockups in Figma
