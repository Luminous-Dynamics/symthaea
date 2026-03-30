// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState } from 'react';
import {
  Target, Plus, Search, Filter, ArrowRight, Star,
  Mail, Phone, Building2, Calendar, MoreHorizontal
} from 'lucide-react';

interface Lead {
  id: string;
  first_name: string;
  last_name: string;
  company?: string;
  title?: string;
  email?: string;
  phone?: string;
  source: string;
  status: string;
  rating?: string;
  score: number;
  created_at: string;
}

// Mock data
const mockLeads: Lead[] = [
  { id: '1', first_name: 'Alex', last_name: 'Thompson', company: 'Innovate Labs', title: 'VP Engineering', email: 'alex@innovatelabs.com', phone: '+1-555-1001', source: 'WEB', status: 'NEW', rating: 'HOT', score: 85, created_at: '2024-01-15' },
  { id: '2', first_name: 'Maria', last_name: 'Garcia', company: 'FutureScale', title: 'Director of Ops', email: 'maria@futurescale.io', phone: '+1-555-1002', source: 'REFERRAL', status: 'CONTACTED', rating: 'WARM', score: 65, created_at: '2024-01-14' },
  { id: '3', first_name: 'James', last_name: 'Wilson', company: 'DataDriven Inc', title: 'CTO', email: 'james@datadriven.com', phone: '+1-555-1003', source: 'TRADE_SHOW', status: 'QUALIFIED', rating: 'HOT', score: 92, created_at: '2024-01-12' },
  { id: '4', first_name: 'Sophie', last_name: 'Brown', company: 'CloudFirst', title: 'CEO', email: 'sophie@cloudfirst.co', phone: '+1-555-1004', source: 'COLD_CALL', status: 'NEW', rating: 'COLD', score: 35, created_at: '2024-01-16' },
  { id: '5', first_name: 'David', last_name: 'Kim', company: 'NextGen Solutions', title: 'Head of Product', email: 'david@nextgen.io', phone: '+1-555-1005', source: 'WEB', status: 'CONTACTED', rating: 'WARM', score: 58, created_at: '2024-01-13' },
];

export default function LeadsPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('ALL');
  const [showConvertModal, setShowConvertModal] = useState(false);
  const [selectedLead, setSelectedLead] = useState<Lead | null>(null);

  const filteredLeads = mockLeads.filter(lead => {
    const matchesSearch = `${lead.first_name} ${lead.last_name} ${lead.company}`.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = statusFilter === 'ALL' || lead.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const getStatusBadge = (status: string) => {
    const styles: Record<string, string> = {
      NEW: 'bg-blue-100 text-blue-800',
      CONTACTED: 'bg-yellow-100 text-yellow-800',
      QUALIFIED: 'bg-green-100 text-green-800',
      UNQUALIFIED: 'bg-gray-100 text-gray-800',
      CONVERTED: 'bg-purple-100 text-purple-800',
    };
    return styles[status] || 'bg-gray-100 text-gray-800';
  };

  const getRatingColor = (rating?: string) => {
    switch (rating) {
      case 'HOT': return 'text-red-500';
      case 'WARM': return 'text-orange-500';
      case 'COLD': return 'text-blue-400';
      default: return 'text-gray-300';
    }
  };

  const getSourceLabel = (source: string) => {
    const labels: Record<string, string> = {
      WEB: 'Website',
      REFERRAL: 'Referral',
      TRADE_SHOW: 'Trade Show',
      COLD_CALL: 'Cold Call',
      SOCIAL: 'Social Media',
      ADVERTISING: 'Advertising',
    };
    return labels[source] || source;
  };

  const stats = {
    total: mockLeads.length,
    new: mockLeads.filter(l => l.status === 'NEW').length,
    contacted: mockLeads.filter(l => l.status === 'CONTACTED').length,
    qualified: mockLeads.filter(l => l.status === 'QUALIFIED').length,
    avgScore: Math.round(mockLeads.reduce((sum, l) => sum + l.score, 0) / mockLeads.length),
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Leads</h1>
          <p className="text-gray-600">Track and convert potential customers</p>
        </div>
        <button className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700">
          <Plus className="w-4 h-4" />
          New Lead
        </button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-5 gap-4">
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <p className="text-sm text-gray-500">Total Leads</p>
          <p className="text-2xl font-bold text-gray-900">{stats.total}</p>
        </div>
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <p className="text-sm text-gray-500">New</p>
          <p className="text-2xl font-bold text-blue-600">{stats.new}</p>
        </div>
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <p className="text-sm text-gray-500">Contacted</p>
          <p className="text-2xl font-bold text-yellow-600">{stats.contacted}</p>
        </div>
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <p className="text-sm text-gray-500">Qualified</p>
          <p className="text-2xl font-bold text-green-600">{stats.qualified}</p>
        </div>
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <p className="text-sm text-gray-500">Avg Score</p>
          <p className="text-2xl font-bold text-purple-600">{stats.avgScore}</p>
        </div>
      </div>

      {/* Filters */}
      <div className="flex gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search leads..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
        >
          <option value="ALL">All Status</option>
          <option value="NEW">New</option>
          <option value="CONTACTED">Contacted</option>
          <option value="QUALIFIED">Qualified</option>
          <option value="UNQUALIFIED">Unqualified</option>
        </select>
        <button className="flex items-center gap-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50">
          <Filter className="w-4 h-4" />
          More Filters
        </button>
      </div>

      {/* Leads Table */}
      <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Lead</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Company</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Source</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Score</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Rating</th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {filteredLeads.map((lead) => (
              <tr key={lead.id} className="hover:bg-gray-50">
                <td className="px-6 py-4">
                  <div className="flex items-center">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center text-white font-medium">
                      {lead.first_name[0]}{lead.last_name[0]}
                    </div>
                    <div className="ml-4">
                      <div className="text-sm font-medium text-gray-900">
                        {lead.first_name} {lead.last_name}
                      </div>
                      <div className="text-sm text-gray-500">{lead.title}</div>
                    </div>
                  </div>
                </td>
                <td className="px-6 py-4">
                  <div className="flex items-center gap-2">
                    <Building2 className="w-4 h-4 text-gray-400" />
                    <span className="text-sm text-gray-900">{lead.company}</span>
                  </div>
                  <div className="flex items-center gap-4 mt-1">
                    {lead.email && (
                      <a href={`mailto:${lead.email}`} className="flex items-center gap-1 text-xs text-gray-500 hover:text-blue-600">
                        <Mail className="w-3 h-3" />
                      </a>
                    )}
                    {lead.phone && (
                      <a href={`tel:${lead.phone}`} className="flex items-center gap-1 text-xs text-gray-500 hover:text-blue-600">
                        <Phone className="w-3 h-3" />
                      </a>
                    )}
                  </div>
                </td>
                <td className="px-6 py-4 text-sm text-gray-500">
                  {getSourceLabel(lead.source)}
                </td>
                <td className="px-6 py-4">
                  <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusBadge(lead.status)}`}>
                    {lead.status}
                  </span>
                </td>
                <td className="px-6 py-4">
                  <div className="flex items-center gap-2">
                    <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${
                          lead.score >= 80 ? 'bg-green-500' :
                          lead.score >= 50 ? 'bg-yellow-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${lead.score}%` }}
                      />
                    </div>
                    <span className="text-sm font-medium text-gray-700">{lead.score}</span>
                  </div>
                </td>
                <td className="px-6 py-4">
                  <div className="flex items-center gap-1">
                    {[1, 2, 3].map((i) => (
                      <Star
                        key={i}
                        className={`w-4 h-4 ${
                          lead.rating === 'HOT' ? 'text-red-500 fill-red-500' :
                          lead.rating === 'WARM' && i <= 2 ? 'text-orange-500 fill-orange-500' :
                          lead.rating === 'COLD' && i === 1 ? 'text-blue-400 fill-blue-400' :
                          'text-gray-300'
                        }`}
                      />
                    ))}
                    <span className={`ml-1 text-xs font-medium ${getRatingColor(lead.rating)}`}>
                      {lead.rating}
                    </span>
                  </div>
                </td>
                <td className="px-6 py-4 text-right">
                  <div className="flex items-center justify-end gap-2">
                    <button
                      onClick={() => {
                        setSelectedLead(lead);
                        setShowConvertModal(true);
                      }}
                      className="flex items-center gap-1 px-3 py-1 text-sm text-blue-600 hover:bg-blue-50 rounded"
                    >
                      Convert
                      <ArrowRight className="w-3 h-3" />
                    </button>
                    <button className="p-1 text-gray-400 hover:text-gray-600">
                      <MoreHorizontal className="w-4 h-4" />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Convert Modal */}
      {showConvertModal && selectedLead && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h2 className="text-lg font-semibold mb-4">Convert Lead</h2>
            <p className="text-sm text-gray-600 mb-4">
              Convert <strong>{selectedLead.first_name} {selectedLead.last_name}</strong> from {selectedLead.company} to:
            </p>
            <div className="space-y-3">
              <label className="flex items-center gap-3 p-3 border rounded-lg cursor-pointer hover:bg-gray-50">
                <input type="checkbox" defaultChecked className="w-4 h-4 text-blue-600" />
                <div>
                  <p className="font-medium">Account</p>
                  <p className="text-sm text-gray-500">Create new account for {selectedLead.company}</p>
                </div>
              </label>
              <label className="flex items-center gap-3 p-3 border rounded-lg cursor-pointer hover:bg-gray-50">
                <input type="checkbox" defaultChecked className="w-4 h-4 text-blue-600" />
                <div>
                  <p className="font-medium">Contact</p>
                  <p className="text-sm text-gray-500">Create contact for {selectedLead.first_name} {selectedLead.last_name}</p>
                </div>
              </label>
              <label className="flex items-center gap-3 p-3 border rounded-lg cursor-pointer hover:bg-gray-50">
                <input type="checkbox" className="w-4 h-4 text-blue-600" />
                <div>
                  <p className="font-medium">Opportunity</p>
                  <p className="text-sm text-gray-500">Create new sales opportunity</p>
                </div>
              </label>
            </div>
            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => setShowConvertModal(false)}
                className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg"
              >
                Cancel
              </button>
              <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
                Convert Lead
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
