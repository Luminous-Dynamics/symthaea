// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState } from 'react';
import {
  Users, Building2, Plus, Search, Filter, Mail, Phone,
  MoreHorizontal, Edit, Trash2, ExternalLink, UserPlus
} from 'lucide-react';

// Types
interface Account {
  id: string;
  name: string;
  account_type: string;
  industry?: string;
  website?: string;
  phone?: string;
  email?: string;
  annual_revenue?: number;
  employee_count?: number;
  owner_id?: string;
  is_active: boolean;
}

interface Contact {
  id: string;
  account_id?: string;
  first_name: string;
  last_name: string;
  title?: string;
  email?: string;
  phone?: string;
  mobile?: string;
  is_primary: boolean;
  is_active: boolean;
}

// Mock data
const mockAccounts: Account[] = [
  { id: '1', name: 'Acme Corporation', account_type: 'CUSTOMER', industry: 'Technology', website: 'https://acme.com', phone: '+1-555-0100', annual_revenue: 5000000, employee_count: 250, is_active: true },
  { id: '2', name: 'TechStart Inc', account_type: 'PROSPECT', industry: 'Software', website: 'https://techstart.io', phone: '+1-555-0200', annual_revenue: 1500000, employee_count: 45, is_active: true },
  { id: '3', name: 'Global Partners', account_type: 'PARTNER', industry: 'Consulting', website: 'https://globalpartners.com', phone: '+1-555-0300', annual_revenue: 10000000, employee_count: 500, is_active: true },
];

const mockContacts: Contact[] = [
  { id: '1', account_id: '1', first_name: 'John', last_name: 'Smith', title: 'CEO', email: 'john@acme.com', phone: '+1-555-0101', is_primary: true, is_active: true },
  { id: '2', account_id: '1', first_name: 'Sarah', last_name: 'Johnson', title: 'CTO', email: 'sarah@acme.com', phone: '+1-555-0102', is_primary: false, is_active: true },
  { id: '3', account_id: '2', first_name: 'Mike', last_name: 'Chen', title: 'Founder', email: 'mike@techstart.io', phone: '+1-555-0201', is_primary: true, is_active: true },
  { id: '4', account_id: '3', first_name: 'Emily', last_name: 'Davis', title: 'Partner', email: 'emily@globalpartners.com', phone: '+1-555-0301', is_primary: true, is_active: true },
];

export default function ContactsPage() {
  const [activeTab, setActiveTab] = useState<'contacts' | 'accounts'>('contacts');
  const [searchQuery, setSearchQuery] = useState('');
  const [showNewModal, setShowNewModal] = useState(false);
  const [selectedAccount, setSelectedAccount] = useState<Account | null>(null);

  const filteredContacts = mockContacts.filter(c =>
    `${c.first_name} ${c.last_name}`.toLowerCase().includes(searchQuery.toLowerCase()) ||
    c.email?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const filteredAccounts = mockAccounts.filter(a =>
    a.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    a.industry?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getAccountName = (accountId?: string) => {
    if (!accountId) return '-';
    const account = mockAccounts.find(a => a.id === accountId);
    return account?.name || '-';
  };

  const getAccountTypeBadge = (type: string) => {
    const colors: Record<string, string> = {
      CUSTOMER: 'bg-green-100 text-green-800',
      PROSPECT: 'bg-blue-100 text-blue-800',
      PARTNER: 'bg-purple-100 text-purple-800',
      VENDOR: 'bg-orange-100 text-orange-800',
    };
    return colors[type] || 'bg-gray-100 text-gray-800';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">CRM</h1>
          <p className="text-gray-600">Manage contacts and accounts</p>
        </div>
        <button
          onClick={() => setShowNewModal(true)}
          className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
        >
          <Plus className="w-4 h-4" />
          {activeTab === 'contacts' ? 'New Contact' : 'New Account'}
        </button>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="flex gap-8">
          <button
            onClick={() => setActiveTab('contacts')}
            className={`pb-4 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'contacts'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            <div className="flex items-center gap-2">
              <Users className="w-4 h-4" />
              Contacts ({mockContacts.length})
            </div>
          </button>
          <button
            onClick={() => setActiveTab('accounts')}
            className={`pb-4 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'accounts'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            <div className="flex items-center gap-2">
              <Building2 className="w-4 h-4" />
              Accounts ({mockAccounts.length})
            </div>
          </button>
        </nav>
      </div>

      {/* Search and Filters */}
      <div className="flex gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            placeholder={`Search ${activeTab}...`}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
        <button className="flex items-center gap-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50">
          <Filter className="w-4 h-4" />
          Filters
        </button>
      </div>

      {/* Contacts Table */}
      {activeTab === 'contacts' && (
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Name
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Account
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Title
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Email
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Phone
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {filteredContacts.map((contact) => (
                <tr key={contact.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center">
                        <span className="text-blue-600 font-medium">
                          {contact.first_name[0]}{contact.last_name[0]}
                        </span>
                      </div>
                      <div className="ml-4">
                        <div className="text-sm font-medium text-gray-900">
                          {contact.first_name} {contact.last_name}
                          {contact.is_primary && (
                            <span className="ml-2 px-2 py-0.5 text-xs bg-yellow-100 text-yellow-800 rounded">
                              Primary
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {getAccountName(contact.account_id)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {contact.title || '-'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {contact.email && (
                      <a href={`mailto:${contact.email}`} className="flex items-center gap-1 text-sm text-blue-600 hover:text-blue-800">
                        <Mail className="w-4 h-4" />
                        {contact.email}
                      </a>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {contact.phone && (
                      <a href={`tel:${contact.phone}`} className="flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700">
                        <Phone className="w-4 h-4" />
                        {contact.phone}
                      </a>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                    <div className="flex items-center justify-end gap-2">
                      <button className="p-1 text-gray-400 hover:text-gray-600">
                        <Edit className="w-4 h-4" />
                      </button>
                      <button className="p-1 text-gray-400 hover:text-red-600">
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Accounts Table */}
      {activeTab === 'accounts' && (
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Account Name
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Type
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Industry
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Revenue
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Employees
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {filteredAccounts.map((account) => (
                <tr key={account.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className="w-10 h-10 rounded-lg bg-purple-100 flex items-center justify-center">
                        <Building2 className="w-5 h-5 text-purple-600" />
                      </div>
                      <div className="ml-4">
                        <div className="text-sm font-medium text-gray-900">{account.name}</div>
                        {account.website && (
                          <a href={account.website} target="_blank" rel="noopener noreferrer" className="text-xs text-blue-600 hover:underline flex items-center gap-1">
                            {account.website.replace('https://', '')}
                            <ExternalLink className="w-3 h-3" />
                          </a>
                        )}
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${getAccountTypeBadge(account.account_type)}`}>
                      {account.account_type}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {account.industry || '-'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {account.annual_revenue
                      ? `$${(account.annual_revenue / 1000000).toFixed(1)}M`
                      : '-'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {account.employee_count?.toLocaleString() || '-'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                    <div className="flex items-center justify-end gap-2">
                      <button
                        onClick={() => setSelectedAccount(account)}
                        className="p-1 text-gray-400 hover:text-blue-600"
                        title="Add Contact"
                      >
                        <UserPlus className="w-4 h-4" />
                      </button>
                      <button className="p-1 text-gray-400 hover:text-gray-600">
                        <Edit className="w-4 h-4" />
                      </button>
                      <button className="p-1 text-gray-400 hover:text-red-600">
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
