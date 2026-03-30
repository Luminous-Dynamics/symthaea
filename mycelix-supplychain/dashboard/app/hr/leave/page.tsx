// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState } from 'react';
import {
  Calendar, Plus, Clock, CheckCircle, XCircle, AlertCircle,
  User, ChevronLeft, ChevronRight, Sun, Thermometer, Heart
} from 'lucide-react';

interface LeaveRequest {
  id: string;
  employee_name: string;
  employee_avatar: string;
  leave_type: string;
  start_date: string;
  end_date: string;
  total_days: number;
  reason?: string;
  status: string;
  approved_by?: string;
}

interface LeaveBalance {
  employee_id: string;
  annual_entitled: number;
  annual_taken: number;
  annual_remaining: number;
  sick_entitled: number;
  sick_taken: number;
  sick_remaining: number;
}

const mockRequests: LeaveRequest[] = [
  { id: '1', employee_name: 'Sarah Chen', employee_avatar: 'from-purple-500 to-pink-500', leave_type: 'ANNUAL', start_date: '2024-02-15', end_date: '2024-02-20', total_days: 4, reason: 'Family vacation', status: 'PENDING' },
  { id: '2', employee_name: 'Michael Johnson', employee_avatar: 'from-blue-500 to-cyan-500', leave_type: 'SICK', start_date: '2024-02-10', end_date: '2024-02-11', total_days: 2, reason: 'Doctor appointment', status: 'APPROVED', approved_by: 'Anna Martinez' },
  { id: '3', employee_name: 'Emily Davis', employee_avatar: 'from-green-500 to-emerald-500', leave_type: 'PERSONAL', start_date: '2024-02-28', end_date: '2024-02-28', total_days: 1, reason: 'Personal matter', status: 'PENDING' },
  { id: '4', employee_name: 'James Wilson', employee_avatar: 'from-orange-500 to-red-500', leave_type: 'ANNUAL', start_date: '2024-03-01', end_date: '2024-03-08', total_days: 6, reason: 'Spring break trip', status: 'APPROVED', approved_by: 'David Brown' },
  { id: '5', employee_name: 'Lisa Wong', employee_avatar: 'from-indigo-500 to-purple-500', leave_type: 'PARENTAL', start_date: '2024-04-01', end_date: '2024-06-30', total_days: 65, reason: 'Maternity leave', status: 'APPROVED', approved_by: 'CEO' },
  { id: '6', employee_name: 'Robert Taylor', employee_avatar: 'from-yellow-500 to-orange-500', leave_type: 'SICK', start_date: '2024-02-05', end_date: '2024-02-05', total_days: 1, status: 'REJECTED', approved_by: 'John Smith' },
];

const whosOutToday = [
  { name: 'Michael Johnson', department: 'Engineering', type: 'SICK', until: '2024-02-11' },
  { name: 'Jennifer Lee', department: 'Marketing', type: 'ANNUAL', until: '2024-02-14' },
];

const myBalance: LeaveBalance = {
  employee_id: 'current',
  annual_entitled: 20,
  annual_taken: 5,
  annual_remaining: 15,
  sick_entitled: 10,
  sick_taken: 2,
  sick_remaining: 8,
};

export default function LeavePage() {
  const [activeTab, setActiveTab] = useState<'requests' | 'calendar' | 'balance'>('requests');
  const [statusFilter, setStatusFilter] = useState('ALL');
  const [showNewRequest, setShowNewRequest] = useState(false);
  const [selectedRequest, setSelectedRequest] = useState<LeaveRequest | null>(null);

  const filteredRequests = mockRequests.filter(req => {
    if (statusFilter === 'ALL') return true;
    return req.status === statusFilter;
  });

  const getStatusBadge = (status: string) => {
    const styles: Record<string, { bg: string; icon: React.ReactNode }> = {
      PENDING: { bg: 'bg-yellow-100 text-yellow-800', icon: <Clock className="w-3 h-3" /> },
      APPROVED: { bg: 'bg-green-100 text-green-800', icon: <CheckCircle className="w-3 h-3" /> },
      REJECTED: { bg: 'bg-red-100 text-red-800', icon: <XCircle className="w-3 h-3" /> },
      CANCELLED: { bg: 'bg-gray-100 text-gray-800', icon: <AlertCircle className="w-3 h-3" /> },
    };
    return styles[status] || styles.PENDING;
  };

  const getLeaveTypeIcon = (type: string) => {
    switch (type) {
      case 'ANNUAL': return <Sun className="w-4 h-4 text-yellow-500" />;
      case 'SICK': return <Thermometer className="w-4 h-4 text-red-500" />;
      case 'PERSONAL': return <User className="w-4 h-4 text-blue-500" />;
      case 'PARENTAL': return <Heart className="w-4 h-4 text-pink-500" />;
      default: return <Calendar className="w-4 h-4 text-gray-500" />;
    }
  };

  const pendingCount = mockRequests.filter(r => r.status === 'PENDING').length;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Time Off</h1>
          <p className="text-gray-600">Manage leave requests and balances</p>
        </div>
        <button
          onClick={() => setShowNewRequest(true)}
          className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
        >
          <Plus className="w-4 h-4" />
          Request Time Off
        </button>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-4 gap-4">
        {/* My Balance Card */}
        <div className="col-span-2 bg-white p-4 rounded-lg border border-gray-200">
          <h3 className="font-medium text-gray-900 mb-4">My Leave Balance</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="p-3 bg-yellow-50 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Sun className="w-5 h-5 text-yellow-600" />
                <span className="font-medium text-gray-900">Annual Leave</span>
              </div>
              <div className="flex items-end gap-2">
                <span className="text-3xl font-bold text-yellow-600">{myBalance.annual_remaining}</span>
                <span className="text-gray-500 mb-1">/ {myBalance.annual_entitled} days</span>
              </div>
              <div className="mt-2 h-2 bg-yellow-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-yellow-500 rounded-full"
                  style={{ width: `${(myBalance.annual_remaining / myBalance.annual_entitled) * 100}%` }}
                />
              </div>
            </div>
            <div className="p-3 bg-red-50 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Thermometer className="w-5 h-5 text-red-600" />
                <span className="font-medium text-gray-900">Sick Leave</span>
              </div>
              <div className="flex items-end gap-2">
                <span className="text-3xl font-bold text-red-600">{myBalance.sick_remaining}</span>
                <span className="text-gray-500 mb-1">/ {myBalance.sick_entitled} days</span>
              </div>
              <div className="mt-2 h-2 bg-red-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-red-500 rounded-full"
                  style={{ width: `${(myBalance.sick_remaining / myBalance.sick_entitled) * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Who's Out Today */}
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <h3 className="font-medium text-gray-900 mb-3">Who's Out Today</h3>
          {whosOutToday.length > 0 ? (
            <div className="space-y-2">
              {whosOutToday.map((person, i) => (
                <div key={i} className="flex items-center gap-2 text-sm">
                  <div className="w-6 h-6 rounded-full bg-gray-200 flex items-center justify-center text-xs">
                    {person.name.split(' ').map(n => n[0]).join('')}
                  </div>
                  <div>
                    <p className="font-medium text-gray-900">{person.name}</p>
                    <p className="text-xs text-gray-500">{person.type} until {new Date(person.until).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}</p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-gray-500">Everyone is in today!</p>
          )}
        </div>

        {/* Pending Approvals */}
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <h3 className="font-medium text-gray-900 mb-3">Pending Approvals</h3>
          <div className="flex items-center gap-3">
            <div className="p-3 bg-yellow-100 rounded-full">
              <Clock className="w-6 h-6 text-yellow-600" />
            </div>
            <div>
              <p className="text-3xl font-bold text-gray-900">{pendingCount}</p>
              <p className="text-sm text-gray-500">requests waiting</p>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="flex gap-8">
          <button
            onClick={() => setActiveTab('requests')}
            className={`pb-4 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'requests'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            Requests
          </button>
          <button
            onClick={() => setActiveTab('calendar')}
            className={`pb-4 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'calendar'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            Calendar
          </button>
        </nav>
      </div>

      {/* Filter */}
      {activeTab === 'requests' && (
        <div className="flex gap-2">
          {['ALL', 'PENDING', 'APPROVED', 'REJECTED'].map((status) => (
            <button
              key={status}
              onClick={() => setStatusFilter(status)}
              className={`px-4 py-2 rounded-lg text-sm font-medium ${
                statusFilter === status
                  ? 'bg-blue-100 text-blue-700'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {status === 'ALL' ? 'All' : status.charAt(0) + status.slice(1).toLowerCase()}
            </button>
          ))}
        </div>
      )}

      {/* Requests Table */}
      {activeTab === 'requests' && (
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Employee</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Type</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Dates</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Days</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {filteredRequests.map((request) => {
                const statusStyle = getStatusBadge(request.status);
                return (
                  <tr key={request.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-3">
                        <div className={`w-10 h-10 rounded-full bg-gradient-to-br ${request.employee_avatar} flex items-center justify-center text-white font-medium`}>
                          {request.employee_name.split(' ').map(n => n[0]).join('')}
                        </div>
                        <div>
                          <p className="font-medium text-gray-900">{request.employee_name}</p>
                          {request.reason && (
                            <p className="text-xs text-gray-500 truncate max-w-[200px]">{request.reason}</p>
                          )}
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        {getLeaveTypeIcon(request.leave_type)}
                        <span className="text-sm text-gray-900">
                          {request.leave_type.charAt(0) + request.leave_type.slice(1).toLowerCase()}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-500">
                      {new Date(request.start_date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                      {request.start_date !== request.end_date && (
                        <> - {new Date(request.end_date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}</>
                      )}
                    </td>
                    <td className="px-6 py-4 text-sm font-medium text-gray-900">
                      {request.total_days} {request.total_days === 1 ? 'day' : 'days'}
                    </td>
                    <td className="px-6 py-4">
                      <span className={`inline-flex items-center gap-1 px-2 py-1 text-xs font-medium rounded-full ${statusStyle.bg}`}>
                        {statusStyle.icon}
                        {request.status}
                      </span>
                      {request.approved_by && (
                        <p className="text-xs text-gray-400 mt-1">by {request.approved_by}</p>
                      )}
                    </td>
                    <td className="px-6 py-4 text-right">
                      {request.status === 'PENDING' && (
                        <div className="flex items-center justify-end gap-2">
                          <button className="px-3 py-1 text-sm text-green-600 hover:bg-green-50 rounded">
                            Approve
                          </button>
                          <button className="px-3 py-1 text-sm text-red-600 hover:bg-red-50 rounded">
                            Reject
                          </button>
                        </div>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Calendar View */}
      {activeTab === 'calendar' && (
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-medium text-gray-900">February 2024</h3>
            <div className="flex items-center gap-2">
              <button className="p-2 hover:bg-gray-100 rounded">
                <ChevronLeft className="w-5 h-5" />
              </button>
              <button className="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded">Today</button>
              <button className="p-2 hover:bg-gray-100 rounded">
                <ChevronRight className="w-5 h-5" />
              </button>
            </div>
          </div>
          <div className="text-center text-gray-500 py-12">
            <Calendar className="w-12 h-12 mx-auto text-gray-300 mb-4" />
            <p>Calendar view with leave requests</p>
            <p className="text-sm">Coming soon...</p>
          </div>
        </div>
      )}

      {/* New Request Modal */}
      {showNewRequest && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h2 className="text-lg font-semibold mb-4">Request Time Off</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Leave Type</label>
                <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500">
                  <option value="ANNUAL">Annual Leave</option>
                  <option value="SICK">Sick Leave</option>
                  <option value="PERSONAL">Personal</option>
                  <option value="UNPAID">Unpaid</option>
                </select>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Start Date</label>
                  <input type="date" className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">End Date</label>
                  <input type="date" className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500" />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Reason (optional)</label>
                <textarea
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  placeholder="Provide a reason for your request..."
                />
              </div>
            </div>
            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => setShowNewRequest(false)}
                className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg"
              >
                Cancel
              </button>
              <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
                Submit Request
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
