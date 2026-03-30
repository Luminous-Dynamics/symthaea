// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useEffect, useState } from 'react';
import DashboardLayout from '@/components/DashboardLayout';
import { api, Payment } from '@/lib/api';
import { Plus, Search, ArrowDownCircle, ArrowUpCircle, CheckCircle, Clock, XCircle, Eye } from 'lucide-react';

export default function PaymentsPage() {
  const [payments, setPayments] = useState<Payment[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [typeFilter, setTypeFilter] = useState<string>('all');

  useEffect(() => {
    const loadPayments = async () => {
      try {
        const data = await api.getPayments();
        setPayments(data);
      } catch (err) {
        console.error('Failed to load payments', err);
      } finally {
        setLoading(false);
      }
    };
    loadPayments();
  }, []);

  const filteredPayments = payments.filter((payment) => {
    const matchesSearch =
      payment.payment_number.toLowerCase().includes(search.toLowerCase());
    const matchesType = typeFilter === 'all' || payment.payment_type === typeFilter;
    return matchesSearch && matchesType;
  });

  const statusColors: Record<string, string> = {
    PENDING: 'bg-yellow-100 text-yellow-700',
    COMPLETED: 'bg-green-100 text-green-700',
    FAILED: 'bg-red-100 text-red-700',
    CANCELLED: 'bg-gray-100 text-gray-500',
  };

  const statusIcons: Record<string, React.ReactNode> = {
    PENDING: <Clock className="w-4 h-4" />,
    COMPLETED: <CheckCircle className="w-4 h-4" />,
    FAILED: <XCircle className="w-4 h-4" />,
    CANCELLED: <XCircle className="w-4 h-4" />,
  };

  const totalReceived = payments
    .filter((p) => p.payment_type === 'RECEIPT' && p.status === 'COMPLETED')
    .reduce((sum, p) => sum + p.amount, 0);

  const totalPaid = payments
    .filter((p) => p.payment_type === 'PAYMENT' && p.status === 'COMPLETED')
    .reduce((sum, p) => sum + p.amount, 0);

  return (
    <DashboardLayout title="Payments" subtitle="Track incoming and outgoing payments">
      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
          <p className="text-sm text-gray-500">Total Transactions</p>
          <p className="text-2xl font-bold text-gray-900">{payments.length}</p>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
          <div className="flex items-center">
            <ArrowDownCircle className="w-5 h-5 text-green-500 mr-2" />
            <div>
              <p className="text-sm text-gray-500">Received</p>
              <p className="text-2xl font-bold text-green-600">${totalReceived.toLocaleString()}</p>
            </div>
          </div>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
          <div className="flex items-center">
            <ArrowUpCircle className="w-5 h-5 text-orange-500 mr-2" />
            <div>
              <p className="text-sm text-gray-500">Paid Out</p>
              <p className="text-2xl font-bold text-orange-600">${totalPaid.toLocaleString()}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Actions bar */}
      <div className="flex flex-col sm:flex-row gap-4 mb-6">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search payments..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
          />
        </div>
        <select
          value={typeFilter}
          onChange={(e) => setTypeFilter(e.target.value)}
          className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
        >
          <option value="all">All Types</option>
          <option value="RECEIPT">Receipts</option>
          <option value="PAYMENT">Payments</option>
        </select>
        <button className="flex items-center justify-center px-4 py-2 bg-green-600 hover:bg-green-700 text-white font-medium rounded-lg transition-colors">
          <ArrowDownCircle className="w-5 h-5 mr-2" />
          Receive Payment
        </button>
        <button className="flex items-center justify-center px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white font-medium rounded-lg transition-colors">
          <ArrowUpCircle className="w-5 h-5 mr-2" />
          Make Payment
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
                  <th className="text-left px-6 py-3 text-sm font-medium text-gray-500">Payment #</th>
                  <th className="text-left px-6 py-3 text-sm font-medium text-gray-500">Date</th>
                  <th className="text-left px-6 py-3 text-sm font-medium text-gray-500">Type</th>
                  <th className="text-left px-6 py-3 text-sm font-medium text-gray-500">Method</th>
                  <th className="text-right px-6 py-3 text-sm font-medium text-gray-500">Amount</th>
                  <th className="text-center px-6 py-3 text-sm font-medium text-gray-500">Status</th>
                  <th className="text-center px-6 py-3 text-sm font-medium text-gray-500">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {filteredPayments.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="px-6 py-8 text-center text-gray-500">
                      No payments found
                    </td>
                  </tr>
                ) : (
                  filteredPayments.map((payment) => (
                    <tr key={payment.id} className="hover:bg-gray-50">
                      <td className="px-6 py-4">
                        <span className="font-medium text-gray-900">{payment.payment_number}</span>
                      </td>
                      <td className="px-6 py-4 text-gray-600">
                        {new Date(payment.payment_date).toLocaleDateString()}
                      </td>
                      <td className="px-6 py-4">
                        <span className={`inline-flex items-center ${
                          payment.payment_type === 'RECEIPT' ? 'text-green-600' : 'text-orange-600'
                        }`}>
                          {payment.payment_type === 'RECEIPT' ? (
                            <ArrowDownCircle className="w-4 h-4 mr-1" />
                          ) : (
                            <ArrowUpCircle className="w-4 h-4 mr-1" />
                          )}
                          {payment.payment_type}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-gray-600 capitalize">
                        {payment.payment_method.replace('_', ' ').toLowerCase()}
                      </td>
                      <td className="px-6 py-4 text-right font-medium text-gray-900">
                        ${payment.amount.toLocaleString()}
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex justify-center">
                          <span className={`inline-flex items-center px-2.5 py-1 text-xs font-medium rounded-full ${statusColors[payment.status]}`}>
                            {statusIcons[payment.status]}
                            <span className="ml-1">{payment.status}</span>
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex justify-center">
                          <button className="p-2 text-gray-400 hover:text-primary-600 rounded-lg hover:bg-gray-100">
                            <Eye className="w-5 h-5" />
                          </button>
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
