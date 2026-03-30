// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import {
  LayoutDashboard,
  FileText,
  Receipt,
  CreditCard,
  BookOpen,
  LogOut,
  TrendingUp,
  TrendingDown,
  DollarSign,
  AlertCircle
} from 'lucide-react';
import { api, Invoice, Bill, Payment, GlAccount } from '@/lib/api';

interface DashboardStats {
  totalReceivables: number;
  totalPayables: number;
  cashBalance: number;
  overdueInvoices: number;
}

export default function Dashboard() {
  const router = useRouter();
  const [user, setUser] = useState<{ name: string; email: string; role: string } | null>(null);
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [recentInvoices, setRecentInvoices] = useState<Invoice[]>([]);
  const [recentBills, setRecentBills] = useState<Bill[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadDashboard = async () => {
      try {
        // Check authentication
        const token = api.getToken();
        if (!token) {
          router.push('/login');
          return;
        }

        // Load user profile
        const profile = await api.getProfile();
        setUser(profile);

        // Load data
        const [invoices, bills, payments, accounts] = await Promise.all([
          api.getInvoices().catch(() => []),
          api.getBills().catch(() => []),
          api.getPayments().catch(() => []),
          api.getAccounts().catch(() => []),
        ]);

        // Calculate stats
        const receivables = invoices
          .filter(i => i.status !== 'PAID' && i.status !== 'CANCELLED')
          .reduce((sum, i) => sum + i.total_amount, 0);

        const payables = bills
          .filter(b => b.status !== 'PAID' && b.status !== 'CANCELLED')
          .reduce((sum, b) => sum + b.total_amount, 0);

        const overdue = invoices.filter(i => {
          const dueDate = new Date(i.due_date);
          return i.status === 'SENT' && dueDate < new Date();
        }).length;

        setStats({
          totalReceivables: receivables,
          totalPayables: payables,
          cashBalance: 150000, // Would come from GL balance
          overdueInvoices: overdue,
        });

        setRecentInvoices(invoices.slice(0, 5));
        setRecentBills(bills.slice(0, 5));
      } catch (err: any) {
        if (err.status === 401) {
          router.push('/login');
        } else {
          setError(err.error || 'Failed to load dashboard');
        }
      } finally {
        setLoading(false);
      }
    };

    loadDashboard();
  }, [router]);

  const handleLogout = async () => {
    await api.logout();
    router.push('/login');
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Sidebar */}
      <aside className="fixed inset-y-0 left-0 w-64 bg-white border-r border-gray-200">
        <div className="flex flex-col h-full">
          {/* Logo */}
          <div className="flex items-center h-16 px-6 border-b border-gray-200">
            <span className="text-xl font-bold text-primary-600">Mycelix</span>
            <span className="ml-2 text-sm text-gray-500">ERP</span>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-4 py-6 space-y-1">
            <Link href="/" className="flex items-center px-4 py-2 text-sm font-medium text-primary-600 bg-primary-50 rounded-lg">
              <LayoutDashboard className="w-5 h-5 mr-3" />
              Dashboard
            </Link>
            <Link href="/accounts" className="flex items-center px-4 py-2 text-sm font-medium text-gray-600 hover:bg-gray-100 rounded-lg">
              <BookOpen className="w-5 h-5 mr-3" />
              Chart of Accounts
            </Link>
            <Link href="/invoices" className="flex items-center px-4 py-2 text-sm font-medium text-gray-600 hover:bg-gray-100 rounded-lg">
              <FileText className="w-5 h-5 mr-3" />
              Invoices
            </Link>
            <Link href="/bills" className="flex items-center px-4 py-2 text-sm font-medium text-gray-600 hover:bg-gray-100 rounded-lg">
              <Receipt className="w-5 h-5 mr-3" />
              Bills
            </Link>
            <Link href="/payments" className="flex items-center px-4 py-2 text-sm font-medium text-gray-600 hover:bg-gray-100 rounded-lg">
              <CreditCard className="w-5 h-5 mr-3" />
              Payments
            </Link>
          </nav>

          {/* User */}
          <div className="p-4 border-t border-gray-200">
            <div className="flex items-center">
              <div className="w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center">
                <span className="text-sm font-medium text-primary-600">
                  {user?.name?.charAt(0) || 'U'}
                </span>
              </div>
              <div className="ml-3 flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-700 truncate">{user?.name}</p>
                <p className="text-xs text-gray-500 truncate">{user?.role}</p>
              </div>
              <button onClick={handleLogout} className="p-2 text-gray-400 hover:text-gray-600">
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="ml-64 p-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-500">Welcome back, {user?.name?.split(' ')[0]}</p>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-center">
            <AlertCircle className="w-5 h-5 text-red-500 mr-2" />
            <span className="text-red-700">{error}</span>
          </div>
        )}

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">Cash Balance</p>
                <p className="text-2xl font-bold text-gray-900">
                  ${stats?.cashBalance.toLocaleString()}
                </p>
              </div>
              <div className="p-3 bg-green-100 rounded-full">
                <DollarSign className="w-6 h-6 text-green-600" />
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">Receivables</p>
                <p className="text-2xl font-bold text-gray-900">
                  ${stats?.totalReceivables.toLocaleString()}
                </p>
              </div>
              <div className="p-3 bg-blue-100 rounded-full">
                <TrendingUp className="w-6 h-6 text-blue-600" />
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">Payables</p>
                <p className="text-2xl font-bold text-gray-900">
                  ${stats?.totalPayables.toLocaleString()}
                </p>
              </div>
              <div className="p-3 bg-orange-100 rounded-full">
                <TrendingDown className="w-6 h-6 text-orange-600" />
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">Overdue Invoices</p>
                <p className="text-2xl font-bold text-gray-900">{stats?.overdueInvoices}</p>
              </div>
              <div className="p-3 bg-red-100 rounded-full">
                <AlertCircle className="w-6 h-6 text-red-600" />
              </div>
            </div>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Recent Invoices */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-100">
            <div className="px-6 py-4 border-b border-gray-100">
              <h2 className="text-lg font-semibold text-gray-900">Recent Invoices</h2>
            </div>
            <div className="divide-y divide-gray-100">
              {recentInvoices.length === 0 ? (
                <p className="p-6 text-gray-500 text-center">No invoices yet</p>
              ) : (
                recentInvoices.map((invoice) => (
                  <div key={invoice.id} className="px-6 py-4 flex items-center justify-between">
                    <div>
                      <p className="font-medium text-gray-900">{invoice.invoice_number}</p>
                      <p className="text-sm text-gray-500">
                        Due {new Date(invoice.due_date).toLocaleDateString()}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="font-medium text-gray-900">
                        ${invoice.total_amount.toLocaleString()}
                      </p>
                      <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${
                        invoice.status === 'PAID' ? 'bg-green-100 text-green-700' :
                        invoice.status === 'SENT' ? 'bg-blue-100 text-blue-700' :
                        invoice.status === 'OVERDUE' ? 'bg-red-100 text-red-700' :
                        'bg-gray-100 text-gray-700'
                      }`}>
                        {invoice.status}
                      </span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Recent Bills */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-100">
            <div className="px-6 py-4 border-b border-gray-100">
              <h2 className="text-lg font-semibold text-gray-900">Recent Bills</h2>
            </div>
            <div className="divide-y divide-gray-100">
              {recentBills.length === 0 ? (
                <p className="p-6 text-gray-500 text-center">No bills yet</p>
              ) : (
                recentBills.map((bill) => (
                  <div key={bill.id} className="px-6 py-4 flex items-center justify-between">
                    <div>
                      <p className="font-medium text-gray-900">{bill.bill_number}</p>
                      <p className="text-sm text-gray-500">
                        Due {new Date(bill.due_date).toLocaleDateString()}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="font-medium text-gray-900">
                        ${bill.total_amount.toLocaleString()}
                      </p>
                      <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${
                        bill.status === 'PAID' ? 'bg-green-100 text-green-700' :
                        bill.status === 'APPROVED' ? 'bg-blue-100 text-blue-700' :
                        'bg-gray-100 text-gray-700'
                      }`}>
                        {bill.status}
                      </span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
