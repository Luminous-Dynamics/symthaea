import TierDistribution from './components/TierDistribution';
import GateRejections from './components/GateRejections';
import AuditTrail from './components/AuditTrail';
import ProfileInspector from './components/ProfileInspector';
import { useConsciousness } from './hooks/useConsciousness';

export default function App() {
  const {
    tierDistribution,
    auditTrail,
    gateTimeSeries,
    agents,
    loading,
    error,
    isLive,
    lookupProfile,
    refreshData,
  } = useConsciousness();

  return (
    <div className="min-h-screen p-4 max-w-screen-2xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-100">
            Mycelix Consciousness Dashboard
          </h1>
          <p className="text-sm text-gray-400 mt-1">
            4D consciousness gating system -- community operator view
          </p>
          {error && (
            <p className="text-xs text-amber-400 mt-1">{error}</p>
          )}
        </div>
        <div className="flex items-center gap-3">
          <span className={`text-xs px-2 py-0.5 rounded-full ${
            isLive
              ? 'bg-green-900/50 text-green-400 border border-green-800'
              : 'bg-gray-800 text-gray-500 border border-gray-700'
          }`}>
            {isLive ? 'LIVE' : 'MOCK'}
          </span>
          {loading && (
            <span className="text-xs text-gray-500 animate-pulse">Loading...</span>
          )}
          <button
            onClick={refreshData}
            className="px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-gray-300 text-sm rounded border border-gray-700"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Grid layout: 2 columns on large screens */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Top left: Tier distribution */}
        <TierDistribution data={tierDistribution} />

        {/* Top right: Gate decisions time series */}
        <GateRejections data={gateTimeSeries} />

        {/* Bottom left: Profile inspector */}
        <ProfileInspector lookupProfile={lookupProfile} agents={agents} />

        {/* Bottom right: Audit trail */}
        <AuditTrail entries={auditTrail} />
      </div>

      {/* Footer */}
      <div className="mt-6 text-center text-xs text-gray-600">
        Mycelix Consciousness Gating | Observer (0.0) &rarr; Participant (0.3) &rarr; Citizen (0.4) &rarr; Steward (0.6) &rarr; Guardian (0.8)
      </div>
    </div>
  );
}
