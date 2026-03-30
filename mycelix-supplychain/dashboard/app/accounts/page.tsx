// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useEffect, useState } from 'react';
import DashboardLayout from '@/components/DashboardLayout';
import { api, GlAccount } from '@/lib/api';
import { ChevronRight, Search, Plus } from 'lucide-react';

export default function AccountsPage() {
  const [accounts, setAccounts] = useState<GlAccount[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');

  useEffect(() => {
    const loadAccounts = async () => {
      try {
        const data = await api.getAccounts();
        setAccounts(data);
      } catch (err) {
        console.error('Failed to load accounts', err);
      } finally {
        setLoading(false);
      }
    };
    loadAccounts();
  }, []);

  const filteredAccounts = accounts.filter(
    (a) =>
      a.account_name.toLowerCase().includes(search.toLowerCase()) ||
      a.account_number.includes(search)
  );

  const groupedAccounts = filteredAccounts.reduce((groups, account) => {
    const type = account.account_type;
    if (!groups[type]) groups[type] = [];
    groups[type].push(account);
    return groups;
  }, {} as Record<string, GlAccount[]>);

  const typeOrder = ['ASSET', 'LIABILITY', 'EQUITY', 'REVENUE', 'EXPENSE'];
  const typeColors: Record<string, string> = {
    ASSET: 'bg-blue-100 text-blue-700',
    LIABILITY: 'bg-orange-100 text-orange-700',
    EQUITY: 'bg-purple-100 text-purple-700',
    REVENUE: 'bg-green-100 text-green-700',
    EXPENSE: 'bg-red-100 text-red-700',
  };

  return (
    <DashboardLayout title="Chart of Accounts" subtitle="Manage your general ledger accounts">
      {/* Actions bar */}
      <div className="flex flex-col sm:flex-row gap-4 mb-6">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search accounts..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
          />
        </div>
        <button className="flex items-center justify-center px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white font-medium rounded-lg transition-colors">
          <Plus className="w-5 h-5 mr-2" />
          Add Account
        </button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
        </div>
      ) : (
        <div className="space-y-6">
          {typeOrder.map((type) => {
            const typeAccounts = groupedAccounts[type];
            if (!typeAccounts || typeAccounts.length === 0) return null;

            return (
              <div key={type} className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
                <div className="px-6 py-4 border-b border-gray-100 flex items-center justify-between">
                  <div className="flex items-center">
                    <span className={`px-3 py-1 text-sm font-medium rounded-full ${typeColors[type]}`}>
                      {type}
                    </span>
                    <span className="ml-3 text-sm text-gray-500">{typeAccounts.length} accounts</span>
                  </div>
                </div>
                <div className="divide-y divide-gray-100">
                  {typeAccounts.map((account) => (
                    <div
                      key={account.id}
                      className="px-6 py-4 flex items-center justify-between hover:bg-gray-50 cursor-pointer"
                    >
                      <div className="flex items-center">
                        <span className="font-mono text-sm text-gray-500 w-16">{account.account_number}</span>
                        <span className="font-medium text-gray-900 ml-4">{account.account_name}</span>
                        {account.parent_account_id && (
                          <span className="ml-2 text-xs text-gray-400">(sub-account)</span>
                        )}
                      </div>
                      <div className="flex items-center">
                        <span className={`px-2 py-1 text-xs font-medium rounded ${
                          account.is_active ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'
                        }`}>
                          {account.is_active ? 'Active' : 'Inactive'}
                        </span>
                        <ChevronRight className="w-5 h-5 text-gray-400 ml-4" />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </DashboardLayout>
  );
}
