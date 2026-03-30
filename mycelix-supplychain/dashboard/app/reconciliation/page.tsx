// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useEffect, useState } from 'react';
import DashboardLayout from '@/components/DashboardLayout';
import { api, BankTransaction, ReconciliationStats, MatchSuggestion } from '@/lib/api';
import {
  RefreshCw,
  Check,
  X,
  Sparkles,
  ArrowUpRight,
  ArrowDownLeft,
  AlertCircle,
  Building2,
  Link2,
  Clock,
} from 'lucide-react';

export default function ReconciliationPage() {
  const [transactions, setTransactions] = useState<BankTransaction[]>([]);
  const [stats, setStats] = useState<ReconciliationStats | null>(null);
  const [suggestions, setSuggestions] = useState<MatchSuggestion[]>([]);
  const [loading, setLoading] = useState(true);
  const [autoMatching, setAutoMatching] = useState(false);
  const [selectedTxn, setSelectedTxn] = useState<string | null>(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const [txns, statsData] = await Promise.all([
        api.getUnmatchedTransactions(),
        api.getReconciliationStats(),
      ]);
      setTransactions(txns);
      setStats(statsData);
    } catch (err) {
      console.error('Failed to load reconciliation data', err);
    } finally {
      setLoading(false);
    }
  };

  const handleAutoMatch = async () => {
    try {
      setAutoMatching(true);
      const matches = await api.autoMatchTransactions();
      setSuggestions(matches);
    } catch (err) {
      console.error('Auto-match failed', err);
    } finally {
      setAutoMatching(false);
    }
  };

  const handleAcceptMatch = async (txnId: string, suggestion: MatchSuggestion) => {
    try {
      await api.matchTransaction(txnId, suggestion.suggested_type, suggestion.suggested_id);
      setTransactions(transactions.filter((t) => t.id !== txnId));
      setSuggestions(suggestions.filter((s) => s.bank_transaction_id !== txnId));
      loadData();
    } catch (err) {
      console.error('Match failed', err);
    }
  };

  const handleIgnore = async (txnId: string) => {
    try {
      await api.ignoreTransaction(txnId, 'Manually ignored');
      setTransactions(transactions.filter((t) => t.id !== txnId));
      loadData();
    } catch (err) {
      console.error('Ignore failed', err);
    }
  };

  const getSuggestionForTxn = (txnId: string) =>
    suggestions.find((s) => s.bank_transaction_id === txnId);

  return (
    <DashboardLayout
      title="Bank Reconciliation"
      subtitle="Match bank transactions to invoices and bills"
    >
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-6">
        <div className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
          <p className="text-sm text-gray-500">Total Transactions</p>
          <p className="text-2xl font-bold text-gray-900">{stats?.total_transactions || 0}</p>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
          <p className="text-sm text-gray-500">Matched</p>
          <p className="text-2xl font-bold text-green-600">{stats?.matched_count || 0}</p>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
          <p className="text-sm text-gray-500">Unmatched</p>
          <p className="text-2xl font-bold text-orange-600">{stats?.unmatched_count || 0}</p>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
          <p className="text-sm text-gray-500">Pending</p>
          <p className="text-2xl font-bold text-blue-600">{stats?.pending_count || 0}</p>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
          <p className="text-sm text-gray-500">Auto-Match Rate</p>
          <p className="text-2xl font-bold text-primary-600">
            {stats?.auto_match_rate?.toFixed(1) || 0}%
          </p>
        </div>
      </div>

      {/* Action bar */}
      <div className="flex flex-col sm:flex-row gap-4 mb-6">
        <button
          onClick={handleAutoMatch}
          disabled={autoMatching}
          className="flex items-center justify-center px-4 py-2 bg-primary-600 hover:bg-primary-700 disabled:bg-primary-400 text-white font-medium rounded-lg transition-colors"
        >
          {autoMatching ? (
            <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
          ) : (
            <Sparkles className="w-5 h-5 mr-2" />
          )}
          {autoMatching ? 'Matching...' : 'Auto-Match Transactions'}
        </button>
        <button
          onClick={loadData}
          className="flex items-center justify-center px-4 py-2 border border-gray-300 text-gray-700 font-medium rounded-lg hover:bg-gray-50 transition-colors"
        >
          <RefreshCw className="w-5 h-5 mr-2" />
          Refresh
        </button>
      </div>

      {/* Suggestions banner */}
      {suggestions.length > 0 && (
        <div className="mb-6 bg-primary-50 border border-primary-200 rounded-xl p-4">
          <div className="flex items-center">
            <Sparkles className="w-5 h-5 text-primary-600 mr-2" />
            <span className="font-medium text-primary-800">
              {suggestions.length} potential matches found!
            </span>
            <span className="ml-2 text-primary-600">
              Review suggestions below to reconcile transactions.
            </span>
          </div>
        </div>
      )}

      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
        </div>
      ) : transactions.length === 0 ? (
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-12 text-center">
          <Check className="w-12 h-12 text-green-500 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">All Caught Up!</h3>
          <p className="text-gray-500">No unmatched transactions to review.</p>
        </div>
      ) : (
        <div className="space-y-4">
          {transactions.map((txn) => {
            const suggestion = getSuggestionForTxn(txn.id);
            const isCredit = txn.amount > 0;

            return (
              <div
                key={txn.id}
                className={`bg-white rounded-xl shadow-sm border ${
                  selectedTxn === txn.id ? 'border-primary-300 ring-2 ring-primary-100' : 'border-gray-100'
                } overflow-hidden`}
              >
                <div
                  className="p-4 cursor-pointer hover:bg-gray-50"
                  onClick={() => setSelectedTxn(selectedTxn === txn.id ? null : txn.id)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                      <div
                        className={`w-10 h-10 rounded-full flex items-center justify-center ${
                          isCredit ? 'bg-green-100' : 'bg-red-100'
                        }`}
                      >
                        {isCredit ? (
                          <ArrowDownLeft className="w-5 h-5 text-green-600" />
                        ) : (
                          <ArrowUpRight className="w-5 h-5 text-red-600" />
                        )}
                      </div>
                      <div>
                        <p className="font-medium text-gray-900">
                          {txn.merchant_name || txn.description || 'Unknown'}
                        </p>
                        <div className="flex items-center text-sm text-gray-500 space-x-2">
                          <span>{new Date(txn.transaction_date).toLocaleDateString()}</span>
                          {txn.category && (
                            <>
                              <span>•</span>
                              <span>{txn.category}</span>
                            </>
                          )}
                          {txn.is_pending && (
                            <>
                              <span>•</span>
                              <span className="inline-flex items-center text-orange-600">
                                <Clock className="w-3 h-3 mr-1" />
                                Pending
                              </span>
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <p
                        className={`text-lg font-semibold ${
                          isCredit ? 'text-green-600' : 'text-red-600'
                        }`}
                      >
                        {isCredit ? '+' : '-'}${Math.abs(txn.amount).toLocaleString()}
                      </p>
                      <p className="text-sm text-gray-500">{txn.currency}</p>
                    </div>
                  </div>

                  {/* Suggestion badge */}
                  {suggestion && (
                    <div className="mt-3 flex items-center">
                      <div className="flex items-center bg-primary-50 rounded-lg px-3 py-1">
                        <Link2 className="w-4 h-4 text-primary-600 mr-2" />
                        <span className="text-sm text-primary-700">
                          Matches {suggestion.suggested_type} ({suggestion.confidence.toFixed(0)}%
                          confidence)
                        </span>
                      </div>
                    </div>
                  )}
                </div>

                {/* Expanded details */}
                {selectedTxn === txn.id && (
                  <div className="border-t border-gray-100 p-4 bg-gray-50">
                    {suggestion ? (
                      <div className="space-y-4">
                        <div>
                          <h4 className="font-medium text-gray-900 mb-2">Suggested Match</h4>
                          <div className="bg-white rounded-lg border border-gray-200 p-3">
                            <div className="flex items-center justify-between">
                              <div>
                                <span className="text-sm font-medium text-gray-700">
                                  {suggestion.suggested_type}
                                </span>
                                <p className="text-sm text-gray-500 mt-1">
                                  {suggestion.match_reasons.join(' • ')}
                                </p>
                              </div>
                              <span className="text-lg font-semibold text-primary-600">
                                {suggestion.confidence.toFixed(0)}%
                              </span>
                            </div>
                          </div>
                        </div>
                        <div className="flex space-x-3">
                          <button
                            onClick={() => handleAcceptMatch(txn.id, suggestion)}
                            className="flex-1 flex items-center justify-center px-4 py-2 bg-green-600 hover:bg-green-700 text-white font-medium rounded-lg transition-colors"
                          >
                            <Check className="w-5 h-5 mr-2" />
                            Accept Match
                          </button>
                          <button
                            onClick={() => handleIgnore(txn.id)}
                            className="flex items-center justify-center px-4 py-2 border border-gray-300 text-gray-700 font-medium rounded-lg hover:bg-gray-100 transition-colors"
                          >
                            <X className="w-5 h-5 mr-2" />
                            Ignore
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-4">
                        <div className="flex items-center text-gray-500">
                          <AlertCircle className="w-5 h-5 mr-2" />
                          <span>No automatic match found. Match manually or ignore.</span>
                        </div>
                        <div className="flex space-x-3">
                          <button className="flex-1 flex items-center justify-center px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white font-medium rounded-lg transition-colors">
                            <Link2 className="w-5 h-5 mr-2" />
                            Match Manually
                          </button>
                          <button
                            onClick={() => handleIgnore(txn.id)}
                            className="flex items-center justify-center px-4 py-2 border border-gray-300 text-gray-700 font-medium rounded-lg hover:bg-gray-100 transition-colors"
                          >
                            <X className="w-5 h-5 mr-2" />
                            Ignore
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </DashboardLayout>
  );
}
