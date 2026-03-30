// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Tenant Management
 *
 * Manage organizations/tenants in the system
 */

import React, { useEffect, useState } from 'react';

interface Tenant {
  id: string;
  name: string;
  domain: string;
  status: 'active' | 'suspended' | 'trial' | 'cancelled';
  plan: string;
  user_count: number;
  storage_used_mb: number;
  created_at: string;
}

interface CreateTenantForm {
  name: string;
  domain: string;
  plan: string;
  admin_email: string;
}

export default function TenantManagement() {
  const [tenants, setTenants] = useState<Tenant[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [selectedTenant, setSelectedTenant] = useState<Tenant | null>(null);

  useEffect(() => {
    fetchTenants();
  }, [statusFilter]);

  async function fetchTenants() {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (statusFilter !== 'all') params.append('status', statusFilter);
      if (searchQuery) params.append('search', searchQuery);

      const response = await fetch(`/api/admin/tenants?${params}`);
      if (response.ok) {
        setTenants(await response.json());
      }
    } catch (error) {
      console.error('Failed to fetch tenants:', error);
    } finally {
      setLoading(false);
    }
  }

  async function createTenant(form: CreateTenantForm) {
    try {
      const response = await fetch('/api/admin/tenants', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });

      if (response.ok) {
        setShowCreateModal(false);
        fetchTenants();
      }
    } catch (error) {
      console.error('Failed to create tenant:', error);
    }
  }

  async function suspendTenant(id: string) {
    if (!confirm('Are you sure you want to suspend this tenant?')) return;

    try {
      await fetch(`/api/admin/tenants/${id}/suspend`, { method: 'POST' });
      fetchTenants();
    } catch (error) {
      console.error('Failed to suspend tenant:', error);
    }
  }

  async function activateTenant(id: string) {
    try {
      await fetch(`/api/admin/tenants/${id}/activate`, { method: 'POST' });
      fetchTenants();
    } catch (error) {
      console.error('Failed to activate tenant:', error);
    }
  }

  async function deleteTenant(id: string) {
    if (!confirm('Are you sure you want to delete this tenant? This action cannot be undone.')) {
      return;
    }

    try {
      await fetch(`/api/admin/tenants/${id}`, { method: 'DELETE' });
      fetchTenants();
    } catch (error) {
      console.error('Failed to delete tenant:', error);
    }
  }

  const statusColors = {
    active: 'bg-green-100 text-green-800',
    suspended: 'bg-red-100 text-red-800',
    trial: 'bg-yellow-100 text-yellow-800',
    cancelled: 'bg-gray-100 text-gray-800',
  };

  const filteredTenants = tenants.filter((tenant) =>
    tenant.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    tenant.domain.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Tenant Management</h1>
        <button
          onClick={() => setShowCreateModal(true)}
          className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90"
        >
          Create Tenant
        </button>
      </div>

      {/* Filters */}
      <div className="flex gap-4">
        <input
          type="text"
          placeholder="Search tenants..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="flex-1 px-4 py-2 border border-border rounded-lg bg-background"
        />
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          className="px-4 py-2 border border-border rounded-lg bg-background"
        >
          <option value="all">All Status</option>
          <option value="active">Active</option>
          <option value="suspended">Suspended</option>
          <option value="trial">Trial</option>
          <option value="cancelled">Cancelled</option>
        </select>
      </div>

      {/* Tenant Table */}
      <div className="bg-surface rounded-lg border border-border overflow-hidden">
        <table className="w-full">
          <thead className="bg-muted/50">
            <tr>
              <th className="px-4 py-3 text-left text-sm font-medium">Name</th>
              <th className="px-4 py-3 text-left text-sm font-medium">Domain</th>
              <th className="px-4 py-3 text-left text-sm font-medium">Status</th>
              <th className="px-4 py-3 text-left text-sm font-medium">Plan</th>
              <th className="px-4 py-3 text-left text-sm font-medium">Users</th>
              <th className="px-4 py-3 text-left text-sm font-medium">Storage</th>
              <th className="px-4 py-3 text-left text-sm font-medium">Created</th>
              <th className="px-4 py-3 text-right text-sm font-medium">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {loading ? (
              <tr>
                <td colSpan={8} className="px-4 py-8 text-center">
                  Loading...
                </td>
              </tr>
            ) : filteredTenants.length === 0 ? (
              <tr>
                <td colSpan={8} className="px-4 py-8 text-center text-muted">
                  No tenants found
                </td>
              </tr>
            ) : (
              filteredTenants.map((tenant) => (
                <tr key={tenant.id} className="hover:bg-muted/30">
                  <td className="px-4 py-3">
                    <button
                      onClick={() => setSelectedTenant(tenant)}
                      className="font-medium hover:text-primary"
                    >
                      {tenant.name}
                    </button>
                  </td>
                  <td className="px-4 py-3 text-sm">{tenant.domain}</td>
                  <td className="px-4 py-3">
                    <span
                      className={`px-2 py-1 text-xs rounded-full ${
                        statusColors[tenant.status]
                      }`}
                    >
                      {tenant.status}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm capitalize">{tenant.plan}</td>
                  <td className="px-4 py-3 text-sm">{tenant.user_count}</td>
                  <td className="px-4 py-3 text-sm">{tenant.storage_used_mb} MB</td>
                  <td className="px-4 py-3 text-sm">
                    {new Date(tenant.created_at).toLocaleDateString()}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <div className="flex justify-end gap-2">
                      {tenant.status === 'active' ? (
                        <button
                          onClick={() => suspendTenant(tenant.id)}
                          className="text-sm text-yellow-600 hover:text-yellow-700"
                        >
                          Suspend
                        </button>
                      ) : tenant.status === 'suspended' ? (
                        <button
                          onClick={() => activateTenant(tenant.id)}
                          className="text-sm text-green-600 hover:text-green-700"
                        >
                          Activate
                        </button>
                      ) : null}
                      <button
                        onClick={() => deleteTenant(tenant.id)}
                        className="text-sm text-red-600 hover:text-red-700"
                      >
                        Delete
                      </button>
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Create Tenant Modal */}
      {showCreateModal && (
        <CreateTenantModal
          onClose={() => setShowCreateModal(false)}
          onCreate={createTenant}
        />
      )}

      {/* Tenant Details Modal */}
      {selectedTenant && (
        <TenantDetailsModal
          tenant={selectedTenant}
          onClose={() => setSelectedTenant(null)}
        />
      )}
    </div>
  );
}

function CreateTenantModal({
  onClose,
  onCreate,
}: {
  onClose: () => void;
  onCreate: (form: CreateTenantForm) => void;
}) {
  const [form, setForm] = useState<CreateTenantForm>({
    name: '',
    domain: '',
    plan: 'starter',
    admin_email: '',
  });

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    onCreate(form);
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-surface rounded-lg p-6 w-full max-w-md">
        <h2 className="text-xl font-bold mb-4">Create Tenant</h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">Name</label>
            <input
              type="text"
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              className="w-full px-3 py-2 border border-border rounded-lg bg-background"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Domain</label>
            <input
              type="text"
              value={form.domain}
              onChange={(e) => setForm({ ...form, domain: e.target.value })}
              className="w-full px-3 py-2 border border-border rounded-lg bg-background"
              placeholder="example.com"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Plan</label>
            <select
              value={form.plan}
              onChange={(e) => setForm({ ...form, plan: e.target.value })}
              className="w-full px-3 py-2 border border-border rounded-lg bg-background"
            >
              <option value="starter">Starter</option>
              <option value="professional">Professional</option>
              <option value="enterprise">Enterprise</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Admin Email</label>
            <input
              type="email"
              value={form.admin_email}
              onChange={(e) => setForm({ ...form, admin_email: e.target.value })}
              className="w-full px-3 py-2 border border-border rounded-lg bg-background"
              required
            />
          </div>
          <div className="flex justify-end gap-2 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 border border-border rounded-lg hover:bg-muted"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90"
            >
              Create
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function TenantDetailsModal({
  tenant,
  onClose,
}: {
  tenant: Tenant;
  onClose: () => void;
}) {
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-surface rounded-lg p-6 w-full max-w-2xl">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold">{tenant.name}</h2>
          <button onClick={onClose} className="text-muted hover:text-foreground">
            Close
          </button>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-sm text-muted">Domain</p>
            <p className="font-medium">{tenant.domain}</p>
          </div>
          <div>
            <p className="text-sm text-muted">Status</p>
            <p className="font-medium capitalize">{tenant.status}</p>
          </div>
          <div>
            <p className="text-sm text-muted">Plan</p>
            <p className="font-medium capitalize">{tenant.plan}</p>
          </div>
          <div>
            <p className="text-sm text-muted">Users</p>
            <p className="font-medium">{tenant.user_count}</p>
          </div>
          <div>
            <p className="text-sm text-muted">Storage Used</p>
            <p className="font-medium">{tenant.storage_used_mb} MB</p>
          </div>
          <div>
            <p className="text-sm text-muted">Created</p>
            <p className="font-medium">
              {new Date(tenant.created_at).toLocaleDateString()}
            </p>
          </div>
        </div>

        <div className="mt-6 pt-4 border-t border-border">
          <h3 className="font-medium mb-2">Quick Actions</h3>
          <div className="flex gap-2">
            <button className="px-3 py-1 text-sm border border-border rounded hover:bg-muted">
              View Users
            </button>
            <button className="px-3 py-1 text-sm border border-border rounded hover:bg-muted">
              View Billing
            </button>
            <button className="px-3 py-1 text-sm border border-border rounded hover:bg-muted">
              Export Data
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
