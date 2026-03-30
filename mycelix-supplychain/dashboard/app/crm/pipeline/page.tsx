// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState } from 'react';
import {
  TrendingUp, Plus, DollarSign, Calendar, Building2,
  ChevronRight, MoreHorizontal, Filter, ArrowUpRight, ArrowDownRight
} from 'lucide-react';

interface Opportunity {
  id: string;
  name: string;
  account_name: string;
  stage: string;
  amount: number;
  probability: number;
  close_date: string;
  owner: string;
  next_step?: string;
}

const STAGES = [
  { id: 'PROSPECTING', name: 'Prospecting', probability: 10 },
  { id: 'QUALIFICATION', name: 'Qualification', probability: 25 },
  { id: 'PROPOSAL', name: 'Proposal', probability: 50 },
  { id: 'NEGOTIATION', name: 'Negotiation', probability: 75 },
  { id: 'CLOSED_WON', name: 'Closed Won', probability: 100 },
  { id: 'CLOSED_LOST', name: 'Closed Lost', probability: 0 },
];

const mockOpportunities: Opportunity[] = [
  { id: '1', name: 'Enterprise License Deal', account_name: 'Acme Corp', stage: 'NEGOTIATION', amount: 250000, probability: 75, close_date: '2024-02-15', owner: 'John D.', next_step: 'Final contract review' },
  { id: '2', name: 'Platform Migration', account_name: 'TechStart Inc', stage: 'PROPOSAL', amount: 85000, probability: 50, close_date: '2024-02-28', owner: 'Sarah M.', next_step: 'Send updated proposal' },
  { id: '3', name: 'Annual Support Contract', account_name: 'Global Partners', stage: 'QUALIFICATION', amount: 45000, probability: 25, close_date: '2024-03-15', owner: 'Mike R.', next_step: 'Discovery call scheduled' },
  { id: '4', name: 'Cloud Deployment', account_name: 'DataDriven Inc', stage: 'PROSPECTING', amount: 120000, probability: 10, close_date: '2024-04-01', owner: 'John D.', next_step: 'Initial outreach' },
  { id: '5', name: 'API Integration Package', account_name: 'CloudFirst', stage: 'PROPOSAL', amount: 65000, probability: 50, close_date: '2024-02-20', owner: 'Sarah M.' },
  { id: '6', name: 'Security Upgrade', account_name: 'NextGen Solutions', stage: 'NEGOTIATION', amount: 180000, probability: 80, close_date: '2024-02-10', owner: 'Mike R.', next_step: 'Pending legal approval' },
  { id: '7', name: 'Custom Development', account_name: 'Innovate Labs', stage: 'CLOSED_WON', amount: 95000, probability: 100, close_date: '2024-01-25', owner: 'John D.' },
  { id: '8', name: 'Consulting Services', account_name: 'FutureScale', stage: 'CLOSED_LOST', amount: 50000, probability: 0, close_date: '2024-01-20', owner: 'Sarah M.' },
];

export default function PipelinePage() {
  const [viewMode, setViewMode] = useState<'kanban' | 'table'>('kanban');
  const [selectedOpp, setSelectedOpp] = useState<Opportunity | null>(null);

  const activeStages = STAGES.filter(s => s.id !== 'CLOSED_WON' && s.id !== 'CLOSED_LOST');

  const getOpportunitiesByStage = (stageId: string) => {
    return mockOpportunities.filter(opp => opp.stage === stageId);
  };

  const calculatePipelineValue = () => {
    return mockOpportunities
      .filter(o => o.stage !== 'CLOSED_WON' && o.stage !== 'CLOSED_LOST')
      .reduce((sum, o) => sum + o.amount, 0);
  };

  const calculateWeightedValue = () => {
    return mockOpportunities
      .filter(o => o.stage !== 'CLOSED_WON' && o.stage !== 'CLOSED_LOST')
      .reduce((sum, o) => sum + (o.amount * o.probability / 100), 0);
  };

  const calculateStageValue = (stageId: string) => {
    return getOpportunitiesByStage(stageId).reduce((sum, o) => sum + o.amount, 0);
  };

  const wonThisMonth = mockOpportunities
    .filter(o => o.stage === 'CLOSED_WON')
    .reduce((sum, o) => sum + o.amount, 0);

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Sales Pipeline</h1>
          <p className="text-gray-600">Track opportunities through the sales process</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => setViewMode('kanban')}
              className={`px-3 py-1.5 text-sm font-medium rounded ${viewMode === 'kanban' ? 'bg-white shadow' : 'text-gray-600'}`}
            >
              Kanban
            </button>
            <button
              onClick={() => setViewMode('table')}
              className={`px-3 py-1.5 text-sm font-medium rounded ${viewMode === 'table' ? 'bg-white shadow' : 'text-gray-600'}`}
            >
              Table
            </button>
          </div>
          <button className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700">
            <Plus className="w-4 h-4" />
            New Opportunity
          </button>
        </div>
      </div>

      {/* Pipeline Summary */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between">
            <p className="text-sm text-gray-500">Pipeline Value</p>
            <TrendingUp className="w-5 h-5 text-blue-500" />
          </div>
          <p className="text-2xl font-bold text-gray-900 mt-2">{formatCurrency(calculatePipelineValue())}</p>
          <p className="text-xs text-green-600 flex items-center gap-1 mt-1">
            <ArrowUpRight className="w-3 h-3" /> +12% from last month
          </p>
        </div>
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between">
            <p className="text-sm text-gray-500">Weighted Pipeline</p>
            <DollarSign className="w-5 h-5 text-green-500" />
          </div>
          <p className="text-2xl font-bold text-gray-900 mt-2">{formatCurrency(calculateWeightedValue())}</p>
          <p className="text-xs text-gray-500 mt-1">Based on probability</p>
        </div>
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between">
            <p className="text-sm text-gray-500">Won This Month</p>
            <TrendingUp className="w-5 h-5 text-green-500" />
          </div>
          <p className="text-2xl font-bold text-green-600 mt-2">{formatCurrency(wonThisMonth)}</p>
          <p className="text-xs text-green-600 flex items-center gap-1 mt-1">
            <ArrowUpRight className="w-3 h-3" /> +25% vs target
          </p>
        </div>
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between">
            <p className="text-sm text-gray-500">Open Deals</p>
            <Calendar className="w-5 h-5 text-purple-500" />
          </div>
          <p className="text-2xl font-bold text-gray-900 mt-2">
            {mockOpportunities.filter(o => o.stage !== 'CLOSED_WON' && o.stage !== 'CLOSED_LOST').length}
          </p>
          <p className="text-xs text-gray-500 mt-1">Active opportunities</p>
        </div>
      </div>

      {/* Kanban View */}
      {viewMode === 'kanban' && (
        <div className="flex gap-4 overflow-x-auto pb-4">
          {activeStages.map((stage) => {
            const stageOpps = getOpportunitiesByStage(stage.id);
            const stageValue = calculateStageValue(stage.id);

            return (
              <div key={stage.id} className="flex-shrink-0 w-72">
                {/* Stage Header */}
                <div className="bg-gray-100 rounded-t-lg px-4 py-3">
                  <div className="flex items-center justify-between">
                    <h3 className="font-medium text-gray-900">{stage.name}</h3>
                    <span className="text-xs bg-gray-200 px-2 py-0.5 rounded-full">
                      {stageOpps.length}
                    </span>
                  </div>
                  <p className="text-sm text-gray-500 mt-1">{formatCurrency(stageValue)}</p>
                </div>

                {/* Stage Cards */}
                <div className="bg-gray-50 rounded-b-lg p-2 min-h-[400px] space-y-2">
                  {stageOpps.map((opp) => (
                    <div
                      key={opp.id}
                      onClick={() => setSelectedOpp(opp)}
                      className="bg-white rounded-lg p-3 border border-gray-200 shadow-sm cursor-pointer hover:shadow-md transition-shadow"
                    >
                      <div className="flex items-start justify-between">
                        <h4 className="font-medium text-gray-900 text-sm">{opp.name}</h4>
                        <button className="text-gray-400 hover:text-gray-600">
                          <MoreHorizontal className="w-4 h-4" />
                        </button>
                      </div>
                      <div className="flex items-center gap-1 mt-2 text-xs text-gray-500">
                        <Building2 className="w-3 h-3" />
                        {opp.account_name}
                      </div>
                      <div className="flex items-center justify-between mt-3">
                        <span className="font-semibold text-gray-900">
                          {formatCurrency(opp.amount)}
                        </span>
                        <span className="text-xs text-gray-500">
                          {opp.probability}%
                        </span>
                      </div>
                      <div className="flex items-center justify-between mt-2 pt-2 border-t border-gray-100">
                        <span className="text-xs text-gray-500">
                          Close: {new Date(opp.close_date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                        </span>
                        <span className="text-xs text-gray-500">{opp.owner}</span>
                      </div>
                      {opp.next_step && (
                        <p className="mt-2 text-xs text-blue-600 bg-blue-50 px-2 py-1 rounded">
                          Next: {opp.next_step}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Table View */}
      {viewMode === 'table' && (
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Opportunity</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Account</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Stage</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Amount</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Probability</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Close Date</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Owner</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {mockOpportunities
                .filter(o => o.stage !== 'CLOSED_WON' && o.stage !== 'CLOSED_LOST')
                .map((opp) => (
                  <tr key={opp.id} className="hover:bg-gray-50 cursor-pointer" onClick={() => setSelectedOpp(opp)}>
                    <td className="px-6 py-4">
                      <div className="font-medium text-gray-900">{opp.name}</div>
                      {opp.next_step && (
                        <div className="text-xs text-gray-500 mt-1">Next: {opp.next_step}</div>
                      )}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-500">{opp.account_name}</td>
                    <td className="px-6 py-4">
                      <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded">
                        {STAGES.find(s => s.id === opp.stage)?.name}
                      </span>
                    </td>
                    <td className="px-6 py-4 font-medium text-gray-900">{formatCurrency(opp.amount)}</td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-blue-500 rounded-full"
                            style={{ width: `${opp.probability}%` }}
                          />
                        </div>
                        <span className="text-sm text-gray-600">{opp.probability}%</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-500">
                      {new Date(opp.close_date).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-500">{opp.owner}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Opportunity Detail Modal */}
      {selectedOpp && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-lg">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold">{selectedOpp.name}</h2>
              <button onClick={() => setSelectedOpp(null)} className="text-gray-400 hover:text-gray-600">
                &times;
              </button>
            </div>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-500">Account</p>
                  <p className="font-medium">{selectedOpp.account_name}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Owner</p>
                  <p className="font-medium">{selectedOpp.owner}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Amount</p>
                  <p className="font-medium text-lg">{formatCurrency(selectedOpp.amount)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Probability</p>
                  <p className="font-medium">{selectedOpp.probability}%</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Stage</p>
                  <p className="font-medium">{STAGES.find(s => s.id === selectedOpp.stage)?.name}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Close Date</p>
                  <p className="font-medium">{new Date(selectedOpp.close_date).toLocaleDateString()}</p>
                </div>
              </div>
              {selectedOpp.next_step && (
                <div className="bg-blue-50 p-3 rounded-lg">
                  <p className="text-sm text-blue-800 font-medium">Next Step</p>
                  <p className="text-blue-700">{selectedOpp.next_step}</p>
                </div>
              )}
              <div className="flex gap-3 pt-4">
                <button
                  onClick={() => setSelectedOpp(null)}
                  className="flex-1 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
                >
                  Close
                </button>
                <button className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
                  Edit Opportunity
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
