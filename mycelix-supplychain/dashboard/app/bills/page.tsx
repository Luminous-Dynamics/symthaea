// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useEffect, useState } from 'react';
import DashboardLayout from '@/components/DashboardLayout';
import { api, Bill } from '@/lib/api';
import { Plus, Search, FileCheck, Clock, CheckCircle, XCircle, Eye } from 'lucide-react';

export default function BillsPage() {
  const [bills, setBills] = useState<Bill[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');

  useEffect(() => {
    const loadBills = async () => {
      try {
        const data = await api.getBills();
        setBills(data);
      } catch (err) {
        console.error('Failed to load bills', err);
      } finally {
        setLoading(false);
      }
    };
    loadBills();
  }, []);

  const filteredBills = bills.filter((bill) => {
    const matchesSearch =
      bill.bill_number.toLowerCase().includes(search.toLowerCase());
    const matchesStatus = statusFilter === 'all' || bill.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const statusColors: Record<string, string> = {
    DRAFT: 'bg-gray-100 text-gray-700',
    PENDING: 'bg-yellow-100 text-yellow-700',
    APPROVED: 'bg-blue-100 text-blue-700',
    PAID: 'bg-green-100 text-green-700',
    CANCELLED: 'bg-gray-100 text-gray-500',
  };

  const statusIcons: Record<string, React.ReactNode> = {
    DRAFT: <Clock className="w-4 h-4" />,
    PENDING: <Clock className="w-4 h-4" />,
    APPROVED: <FileCheck className="w-4 h-4" />,
    PAID: <CheckCircle className="w-4 h-4" />,
    CANCELLED: <XCircle className="w-4 h-4" />,
  };

  const totalPayables = bills
    .filter((b) => b.status !== 'PAID' && b.status !== 'CANCELLED')
    .reduce((sum, b) => sum + b.total_amount, 0);

  const pendingApproval = bills.filter((b) => b.status === 'PENDING').length;

  return (
    <DashboardLayout title="Bills" subtitle="Manage vendor bills and accounts payable">
      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
          <p className="text-sm text-gray-500">Total Bills</p>
          <p className="text-2xl font-bold text-gray-900">{bills.length}</p>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
          <p className="text-sm text-gray-500">Total Payables</p>
          <p className="text-2xl font-bold text-orange-600">${totalPayables.toLocaleString()}</p>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
          <p className="text-sm text-gray-500">Pending Approval</p>
          <p className="text-2xl font-bold text-yellow-600">{pendingApproval}</p>
        </div>
      </div>

      {/* Actions bar */}
      <div className="flex flex-col sm:flex-row gap-4 mb-6">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search bills..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
          />
        </div>
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
        >
          <option value="all">All Status</option>
          <option value="DRAFT">Draft</option>
          <option value="PENDING">Pending</option>
          <option value="APPROVED">Approved</option>
          <option value="PAID">Paid</option>
          <option value="CANCELLED">Cancelled</option>
        </select>
        <button className="flex items-center justify-center px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white font-medium rounded-lg transition-colors">
          <Plus className="w-5 h-5 mr-2" />
          New Bill
        </button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
        </div>
      ) : (
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-100">
                  <th className="text-left px-6 py-3 text-sm font-medium text-gray-500">Bill #</th>
                  <th className="text-left px-6 py-3 text-sm font-medium text-gray-500">Date</th>
                  <th className="text-left px-6 py-3 text-sm font-medium text-gray-500">Due Date</th>
                  <th className="text-right px-6 py-3 text-sm font-medium text-gray-500">Amount</th>
                  <th className="text-center px-6 py-3 text-sm font-medium text-gray-500">Status</th>
                  <th className="text-center px-6 py-3 text-sm font-medium text-gray-500">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {filteredBills.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="px-6 py-8 text-center text-gray-500">
                      No bills found
                    </td>
                  </tr>
                ) : (
                  filteredBills.map((bill) => (
                    <tr key={bill.id} className="hover:bg-gray-50">
                      <td className="px-6 py-4">
                        <span className="font-medium text-gray-900">{bill.bill_number}</span>
                      </td>
                      <td className="px-6 py-4 text-gray-600">
                        {new Date(bill.bill_date).toLocaleDateString()}
                      </td>
                      <td className="px-6 py-4 text-gray-600">
                        {new Date(bill.due_date).toLocaleDateString()}
                      </td>
                      <td className="px-6 py-4 text-right font-medium text-gray-900">
                        ${bill.total_amount.toLocaleString()}
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex justify-center">
                          <span className={`inline-flex items-center px-2.5 py-1 text-xs font-medium rounded-full ${statusColors[bill.status]}`}>
                            {statusIcons[bill.status]}
                            <span className="ml-1">{bill.status}</span>
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex justify-center gap-1">
                          <button className="p-2 text-gray-400 hover:text-primary-600 rounded-lg hover:bg-gray-100">
                            <Eye className="w-5 h-5" />
                          </button>
                          {bill.status === 'PENDING' && (
                            <button className="p-2 text-gray-400 hover:text-green-600 rounded-lg hover:bg-gray-100">
                              <FileCheck className="w-5 h-5" />
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </DashboardLayout>
  );
}
