// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState, useMemo } from 'react';
import {
  Banknote, Plus, Search, Calendar, Download,
  CheckCircle, Clock, AlertCircle, RefreshCw,
  DollarSign, Users, FileText, Play
} from 'lucide-react';
import { api, HrPayRun, HrPayStub } from '@/lib/api';
import { useApiWithFallback, formatCurrency, formatDate } from '@/lib/hooks';
import DashboardLayout from '@/components/DashboardLayout';

// Mock data for demo/development
const mockPayRuns: HrPayRun[] = [
  { id: '1', tenant_id: '', pay_period_start: '2025-12-01', pay_period_end: '2025-12-15', pay_date: '2025-12-20', status: 'COMPLETED', total_gross: 125000, total_deductions: 31250, total_net: 93750, employee_count: 9, processed_by: null, processed_at: '2025-12-18T10:00:00Z', approved_by: null, approved_at: '2025-12-19T14:00:00Z', created_at: '2025-12-16T08:00:00Z' },
  { id: '2', tenant_id: '', pay_period_start: '2025-11-16', pay_period_end: '2025-11-30', pay_date: '2025-12-05', status: 'COMPLETED', total_gross: 125000, total_deductions: 31250, total_net: 93750, employee_count: 9, processed_by: null, processed_at: '2025-12-03T10:00:00Z', approved_by: null, approved_at: '2025-12-04T14:00:00Z', created_at: '2025-12-01T08:00:00Z' },
  { id: '3', tenant_id: '', pay_period_start: '2025-12-16', pay_period_end: '2025-12-31', pay_date: '2026-01-05', status: 'DRAFT', total_gross: 0, total_deductions: 0, total_net: 0, employee_count: 0, processed_by: null, processed_at: null, approved_by: null, approved_at: null, created_at: '2025-12-20T08:00:00Z' },
];

const mockPayStubs: HrPayStub[] = [
  { id: '1', tenant_id: '', pay_run_id: '1', employee_id: '1', gross_pay: 8333.33, federal_tax: 1666.67, state_tax: 500, social_security: 516.67, medicare: 120.83, other_deductions: 200, net_pay: 5329.16, hours_worked: 80, overtime_hours: 0, created_at: '', employee_name: 'Sarah Chen' },
  { id: '2', tenant_id: '', pay_run_id: '1', employee_id: '2', gross_pay: 10000, federal_tax: 2000, state_tax: 600, social_security: 620, medicare: 145, other_deductions: 300, net_pay: 6335, hours_worked: 80, overtime_hours: 5, created_at: '', employee_name: 'Michael Johnson' },
  { id: '3', tenant_id: '', pay_run_id: '1', employee_id: '3', gross_pay: 7500, federal_tax: 1500, state_tax: 450, social_security: 465, medicare: 108.75, other_deductions: 150, net_pay: 4826.25, hours_worked: 80, overtime_hours: 0, created_at: '', employee_name: 'Emily Davis' },
];

export default function PayrollPage() {
  const [selectedPayRun, setSelectedPayRun] = useState<HrPayRun | null>(null);
  const [showNewPayRunModal, setShowNewPayRunModal] = useState(false);
  const [newPayRun, setNewPayRun] = useState({
    pay_period_start: '',
    pay_period_end: '',
    pay_date: '',
  });

  const { data: payRuns, loading, error, refetch, isUsingFallback } = useApiWithFallback(
    () => api.getHrPayRuns(),
    mockPayRuns
  );

  const stats = useMemo(() => {
    const runs = payRuns || [];
    const lastCompleted = runs.find(r => r.status === 'COMPLETED');
    const drafts = runs.filter(r => r.status === 'DRAFT').length;
    const ytdGross = runs
      .filter(r => r.status === 'COMPLETED')
      .reduce((sum, r) => sum + r.total_gross, 0);

    return {
      lastPayrollDate: lastCompleted?.pay_date || 'N/A',
      lastPayrollAmount: lastCompleted?.total_net || 0,
      drafts,
      ytdGross,
    };
  }, [payRuns]);

  const getStatusBadge = (status: string) => {
    const styles: Record<string, { bg: string; icon: React.ElementType }> = {
      DRAFT: { bg: 'bg-gray-100 text-gray-800', icon: FileText },
      PROCESSING: { bg: 'bg-blue-100 text-blue-800', icon: Clock },
      PENDING_APPROVAL: { bg: 'bg-yellow-100 text-yellow-800', icon: Clock },
      APPROVED: { bg: 'bg-green-100 text-green-800', icon: CheckCircle },
      COMPLETED: { bg: 'bg-green-100 text-green-800', icon: CheckCircle },
      CANCELLED: { bg: 'bg-red-100 text-red-800', icon: AlertCircle },
    };
    const config = styles[status] || styles.DRAFT;
    const Icon = config.icon;
    return (
      <span className={`inline-flex items-center gap-1 px-2 py-1 text-xs font-medium rounded-full ${config.bg}`}>
        <Icon className="w-3 h-3" />
        {status.replace('_', ' ')}
      </span>
    );
  };

  const handleCreatePayRun = async () => {
    try {
      await api.createHrPayRun(newPayRun);
      setShowNewPayRunModal(false);
      setNewPayRun({ pay_period_start: '', pay_period_end: '', pay_date: '' });
      refetch();
    } catch (err) {
      console.error('Failed to create pay run:', err);
    }
  };

  const handleGeneratePayStubs = async (payRunId: string) => {
    try {
      await api.generateHrPayStubs(payRunId);
      refetch();
    } catch (err) {
      console.error('Failed to generate pay stubs:', err);
    }
  };

  return (
    <DashboardLayout title="Payroll" subtitle="Manage pay runs and employee compensation">
      <div className="space-y-6">
        {/* Demo mode banner */}
        {isUsingFallback && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-yellow-600" />
            <span className="text-sm text-yellow-800">
              Using demo data. Start the backend server to see real data.
            </span>
            <button
              onClick={refetch}
              className="ml-auto flex items-center gap-1 text-sm text-yellow-700 hover:text-yellow-900"
            >
              <RefreshCw className="w-4 h-4" />
              Retry
            </button>
          </div>
        )}

        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Payroll</h1>
            <p className="text-gray-600">Manage pay runs and employee compensation</p>
          </div>
          <button
            onClick={() => setShowNewPayRunModal(true)}
            className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
          >
            <Plus className="w-4 h-4" />
            New Pay Run
          </button>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-4 gap-4">
          <div className="bg-white p-4 rounded-lg border border-gray-200">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-green-100 rounded-lg">
                <DollarSign className="w-5 h-5 text-green-600" />
              </div>
              <div>
                <p className="text-sm text-gray-500">Last Payroll</p>
                <p className="text-2xl font-bold text-gray-900">
                  {loading ? '...' : formatCurrency(stats.lastPayrollAmount)}
                </p>
              </div>
            </div>
          </div>
          <div className="bg-white p-4 rounded-lg border border-gray-200">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Calendar className="w-5 h-5 text-blue-600" />
              </div>
              <div>
                <p className="text-sm text-gray-500">Last Pay Date</p>
                <p className="text-lg font-bold text-gray-900">
                  {loading ? '...' : stats.lastPayrollDate !== 'N/A' ? formatDate(stats.lastPayrollDate) : 'N/A'}
                </p>
              </div>
            </div>
          </div>
          <div className="bg-white p-4 rounded-lg border border-gray-200">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-yellow-100 rounded-lg">
                <FileText className="w-5 h-5 text-yellow-600" />
              </div>
              <div>
                <p className="text-sm text-gray-500">Draft Pay Runs</p>
                <p className="text-2xl font-bold text-yellow-600">{loading ? '...' : stats.drafts}</p>
              </div>
            </div>
          </div>
          <div className="bg-white p-4 rounded-lg border border-gray-200">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-purple-100 rounded-lg">
                <Banknote className="w-5 h-5 text-purple-600" />
              </div>
              <div>
                <p className="text-sm text-gray-500">YTD Gross</p>
                <p className="text-2xl font-bold text-purple-600">
                  {loading ? '...' : formatCurrency(stats.ytdGross)}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="w-8 h-8 text-blue-500 animate-spin" />
          </div>
        )}

        {/* Pay Runs Table */}
        {!loading && (
          <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-200">
              <h2 className="text-lg font-semibold text-gray-900">Pay Runs</h2>
            </div>
            <table className="w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Pay Period</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Pay Date</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Employees</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Gross Pay</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Net Pay</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {(payRuns || []).map((run) => (
                  <tr key={run.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 text-sm text-gray-900">
                      {formatDate(run.pay_period_start)} - {formatDate(run.pay_period_end)}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-900">
                      {formatDate(run.pay_date)}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-900">
                      <div className="flex items-center gap-1">
                        <Users className="w-4 h-4 text-gray-400" />
                        {run.employee_count}
                      </div>
                    </td>
                    <td className="px-6 py-4 text-sm font-medium text-gray-900">
                      {formatCurrency(run.total_gross)}
                    </td>
                    <td className="px-6 py-4 text-sm font-medium text-green-600">
                      {formatCurrency(run.total_net)}
                    </td>
                    <td className="px-6 py-4">
                      {getStatusBadge(run.status)}
                    </td>
                    <td className="px-6 py-4 text-right">
                      <div className="flex items-center justify-end gap-2">
                        {run.status === 'DRAFT' && (
                          <button
                            onClick={() => handleGeneratePayStubs(run.id)}
                            className="flex items-center gap-1 px-3 py-1 text-sm bg-blue-50 text-blue-600 rounded hover:bg-blue-100"
                          >
                            <Play className="w-3 h-3" />
                            Process
                          </button>
                        )}
                        <button
                          onClick={() => setSelectedPayRun(run)}
                          className="px-3 py-1 text-sm text-gray-600 hover:text-gray-900"
                        >
                          View
                        </button>
                        {run.status === 'COMPLETED' && (
                          <button className="flex items-center gap-1 px-3 py-1 text-sm text-gray-600 hover:text-gray-900">
                            <Download className="w-3 h-3" />
                            Export
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Empty State */}
        {!loading && (payRuns || []).length === 0 && (
          <div className="text-center py-12 bg-white rounded-lg border border-gray-200">
            <Banknote className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">No pay runs found. Create your first pay run to get started.</p>
          </div>
        )}

        {/* New Pay Run Modal */}
        {showNewPayRunModal && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-6 w-full max-w-md">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold">Create New Pay Run</h2>
                <button onClick={() => setShowNewPayRunModal(false)} className="text-gray-400 hover:text-gray-600 text-2xl">
                  &times;
                </button>
              </div>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Pay Period Start</label>
                  <input
                    type="date"
                    value={newPayRun.pay_period_start}
                    onChange={(e) => setNewPayRun({ ...newPayRun, pay_period_start: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Pay Period End</label>
                  <input
                    type="date"
                    value={newPayRun.pay_period_end}
                    onChange={(e) => setNewPayRun({ ...newPayRun, pay_period_end: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Pay Date</label>
                  <input
                    type="date"
                    value={newPayRun.pay_date}
                    onChange={(e) => setNewPayRun({ ...newPayRun, pay_date: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div className="flex gap-3 pt-4">
                  <button
                    onClick={() => setShowNewPayRunModal(false)}
                    className="flex-1 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleCreatePayRun}
                    disabled={!newPayRun.pay_period_start || !newPayRun.pay_period_end || !newPayRun.pay_date}
                    className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
                  >
                    Create Pay Run
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Pay Run Detail Modal */}
        {selectedPayRun && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-6 w-full max-w-4xl max-h-[90vh] overflow-y-auto">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-xl font-semibold">Pay Run Details</h2>
                  <p className="text-gray-500">
                    {formatDate(selectedPayRun.pay_period_start)} - {formatDate(selectedPayRun.pay_period_end)}
                  </p>
                </div>
                <button onClick={() => setSelectedPayRun(null)} className="text-gray-400 hover:text-gray-600 text-2xl">
                  &times;
                </button>
              </div>

              {/* Summary */}
              <div className="grid grid-cols-4 gap-4 mb-6">
                <div className="bg-gray-50 p-4 rounded-lg">
                  <p className="text-sm text-gray-500">Pay Date</p>
                  <p className="text-lg font-semibold">{formatDate(selectedPayRun.pay_date)}</p>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <p className="text-sm text-gray-500">Employees</p>
                  <p className="text-lg font-semibold">{selectedPayRun.employee_count}</p>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <p className="text-sm text-gray-500">Total Gross</p>
                  <p className="text-lg font-semibold">{formatCurrency(selectedPayRun.total_gross)}</p>
                </div>
                <div className="bg-green-50 p-4 rounded-lg">
                  <p className="text-sm text-green-600">Total Net</p>
                  <p className="text-lg font-semibold text-green-700">{formatCurrency(selectedPayRun.total_net)}</p>
                </div>
              </div>

              {/* Pay Stubs Table */}
              <div className="border border-gray-200 rounded-lg overflow-hidden">
                <table className="w-full">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Employee</th>
                      <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Gross</th>
                      <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Federal Tax</th>
                      <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">State Tax</th>
                      <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">SS</th>
                      <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Medicare</th>
                      <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Net Pay</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {mockPayStubs.filter(s => s.pay_run_id === selectedPayRun.id).map((stub) => (
                      <tr key={stub.id}>
                        <td className="px-4 py-3 text-sm font-medium text-gray-900">{stub.employee_name}</td>
                        <td className="px-4 py-3 text-sm text-right text-gray-900">{formatCurrency(stub.gross_pay)}</td>
                        <td className="px-4 py-3 text-sm text-right text-red-600">-{formatCurrency(stub.federal_tax)}</td>
                        <td className="px-4 py-3 text-sm text-right text-red-600">-{formatCurrency(stub.state_tax)}</td>
                        <td className="px-4 py-3 text-sm text-right text-red-600">-{formatCurrency(stub.social_security)}</td>
                        <td className="px-4 py-3 text-sm text-right text-red-600">-{formatCurrency(stub.medicare)}</td>
                        <td className="px-4 py-3 text-sm text-right font-semibold text-green-600">{formatCurrency(stub.net_pay)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="flex gap-3 pt-6">
                <button
                  onClick={() => setSelectedPayRun(null)}
                  className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
                >
                  Close
                </button>
                <button className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
                  <Download className="w-4 h-4" />
                  Export Pay Stubs
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
