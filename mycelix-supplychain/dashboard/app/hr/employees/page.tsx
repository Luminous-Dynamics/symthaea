// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState, useMemo } from 'react';
import {
  Users, Plus, Search, Building2, Mail, Phone,
  Calendar, UserCircle, AlertCircle, RefreshCw
} from 'lucide-react';
import { api, HrEmployee } from '@/lib/api';
import { useApiWithFallback, formatDate } from '@/lib/hooks';
import DashboardLayout from '@/components/DashboardLayout';

// Mock data fallback for demo/development
const mockEmployees: HrEmployee[] = [
  { id: '1', tenant_id: '', employee_number: 'EMP001', first_name: 'Sarah', last_name: 'Chen', email: 'sarah.chen@company.com', phone: '+1-555-0101', date_of_birth: null, hire_date: '2022-03-15', termination_date: null, department_id: null, job_title: 'Senior Software Engineer', manager_id: null, employment_type: 'FULL_TIME', employment_status: 'ACTIVE', work_location: null, salary: null, salary_currency: 'USD', pay_frequency: 'BIWEEKLY', bank_account_number: null, bank_routing_number: null, tax_id: null, emergency_contact_name: null, emergency_contact_phone: null, is_active: true, created_at: '', updated_at: '', department_name: 'Engineering', manager_name: 'John Smith' },
  { id: '2', tenant_id: '', employee_number: 'EMP002', first_name: 'Michael', last_name: 'Johnson', email: 'michael.j@company.com', phone: '+1-555-0102', date_of_birth: null, hire_date: '2021-06-01', termination_date: null, department_id: null, job_title: 'Engineering Manager', manager_id: null, employment_type: 'FULL_TIME', employment_status: 'ACTIVE', work_location: null, salary: null, salary_currency: 'USD', pay_frequency: 'BIWEEKLY', bank_account_number: null, bank_routing_number: null, tax_id: null, emergency_contact_name: null, emergency_contact_phone: null, is_active: true, created_at: '', updated_at: '', department_name: 'Engineering' },
  { id: '3', tenant_id: '', employee_number: 'EMP003', first_name: 'Emily', last_name: 'Davis', email: 'emily.d@company.com', phone: '+1-555-0103', date_of_birth: null, hire_date: '2023-01-10', termination_date: null, department_id: null, job_title: 'Product Manager', manager_id: null, employment_type: 'FULL_TIME', employment_status: 'ACTIVE', work_location: null, salary: null, salary_currency: 'USD', pay_frequency: 'BIWEEKLY', bank_account_number: null, bank_routing_number: null, tax_id: null, emergency_contact_name: null, emergency_contact_phone: null, is_active: true, created_at: '', updated_at: '', department_name: 'Product', manager_name: 'Lisa Wong' },
  { id: '4', tenant_id: '', employee_number: 'EMP004', first_name: 'James', last_name: 'Wilson', email: 'james.w@company.com', phone: '+1-555-0104', date_of_birth: null, hire_date: '2022-09-01', termination_date: null, department_id: null, job_title: 'Account Executive', manager_id: null, employment_type: 'FULL_TIME', employment_status: 'ACTIVE', work_location: null, salary: null, salary_currency: 'USD', pay_frequency: 'BIWEEKLY', bank_account_number: null, bank_routing_number: null, tax_id: null, emergency_contact_name: null, emergency_contact_phone: null, is_active: true, created_at: '', updated_at: '', department_name: 'Sales', manager_name: 'David Brown' },
  { id: '5', tenant_id: '', employee_number: 'EMP005', first_name: 'Lisa', last_name: 'Wong', email: 'lisa.w@company.com', phone: '+1-555-0105', date_of_birth: null, hire_date: '2020-01-15', termination_date: null, department_id: null, job_title: 'VP of Product', manager_id: null, employment_type: 'FULL_TIME', employment_status: 'ACTIVE', work_location: null, salary: null, salary_currency: 'USD', pay_frequency: 'BIWEEKLY', bank_account_number: null, bank_routing_number: null, tax_id: null, emergency_contact_name: null, emergency_contact_phone: null, is_active: true, created_at: '', updated_at: '', department_name: 'Product' },
  { id: '6', tenant_id: '', employee_number: 'EMP006', first_name: 'David', last_name: 'Brown', email: 'david.b@company.com', phone: '+1-555-0106', date_of_birth: null, hire_date: '2021-03-01', termination_date: null, department_id: null, job_title: 'Sales Director', manager_id: null, employment_type: 'FULL_TIME', employment_status: 'ACTIVE', work_location: null, salary: null, salary_currency: 'USD', pay_frequency: 'BIWEEKLY', bank_account_number: null, bank_routing_number: null, tax_id: null, emergency_contact_name: null, emergency_contact_phone: null, is_active: true, created_at: '', updated_at: '', department_name: 'Sales' },
  { id: '7', tenant_id: '', employee_number: 'EMP007', first_name: 'Anna', last_name: 'Martinez', email: 'anna.m@company.com', phone: '+1-555-0107', date_of_birth: null, hire_date: '2022-05-15', termination_date: null, department_id: null, job_title: 'HR Manager', manager_id: null, employment_type: 'FULL_TIME', employment_status: 'ACTIVE', work_location: null, salary: null, salary_currency: 'USD', pay_frequency: 'BIWEEKLY', bank_account_number: null, bank_routing_number: null, tax_id: null, emergency_contact_name: null, emergency_contact_phone: null, is_active: true, created_at: '', updated_at: '', department_name: 'HR' },
  { id: '8', tenant_id: '', employee_number: 'EMP008', first_name: 'Robert', last_name: 'Taylor', email: 'robert.t@company.com', phone: null, date_of_birth: null, hire_date: '2023-06-01', termination_date: null, department_id: null, job_title: 'DevOps Engineer', manager_id: null, employment_type: 'CONTRACTOR', employment_status: 'ACTIVE', work_location: null, salary: null, salary_currency: 'USD', pay_frequency: 'BIWEEKLY', bank_account_number: null, bank_routing_number: null, tax_id: null, emergency_contact_name: null, emergency_contact_phone: null, is_active: true, created_at: '', updated_at: '', department_name: 'Engineering', manager_name: 'John Smith' },
  { id: '9', tenant_id: '', employee_number: 'EMP009', first_name: 'Jennifer', last_name: 'Lee', email: 'jennifer.l@company.com', phone: '+1-555-0109', date_of_birth: null, hire_date: '2022-11-01', termination_date: null, department_id: null, job_title: 'Marketing Specialist', manager_id: null, employment_type: 'FULL_TIME', employment_status: 'ON_LEAVE', work_location: null, salary: null, salary_currency: 'USD', pay_frequency: 'BIWEEKLY', bank_account_number: null, bank_routing_number: null, tax_id: null, emergency_contact_name: null, emergency_contact_phone: null, is_active: true, created_at: '', updated_at: '', department_name: 'Marketing', manager_name: 'Alex Kim' },
];

const departments = ['All', 'Engineering', 'Product', 'Sales', 'HR', 'Marketing', 'Finance'];

// Generate avatar color from name
function getAvatarColor(name: string): string {
  const colors = [
    'from-purple-500 to-pink-500',
    'from-blue-500 to-cyan-500',
    'from-green-500 to-emerald-500',
    'from-orange-500 to-red-500',
    'from-indigo-500 to-purple-500',
    'from-teal-500 to-green-500',
    'from-pink-500 to-rose-500',
    'from-yellow-500 to-orange-500',
    'from-cyan-500 to-blue-500',
  ];
  const index = name.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0) % colors.length;
  return colors[index];
}

export default function EmployeesPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [departmentFilter, setDepartmentFilter] = useState('All');
  const [statusFilter, setStatusFilter] = useState('ALL');
  const [selectedEmployee, setSelectedEmployee] = useState<HrEmployee | null>(null);

  const { data: employees, loading, error, refetch, isUsingFallback } = useApiWithFallback(
    () => api.getHrEmployees(),
    mockEmployees
  );

  const filteredEmployees = useMemo(() => {
    return (employees || []).filter(emp => {
      const matchesSearch = `${emp.first_name} ${emp.last_name} ${emp.email} ${emp.job_title}`
        .toLowerCase()
        .includes(searchQuery.toLowerCase());
      const matchesDept = departmentFilter === 'All' || emp.department_name === departmentFilter;
      const matchesStatus = statusFilter === 'ALL' || emp.employment_status === statusFilter;
      return matchesSearch && matchesDept && matchesStatus;
    });
  }, [employees, searchQuery, departmentFilter, statusFilter]);

  const getStatusBadge = (status: string) => {
    const styles: Record<string, string> = {
      ACTIVE: 'bg-green-100 text-green-800',
      ON_LEAVE: 'bg-yellow-100 text-yellow-800',
      TERMINATED: 'bg-red-100 text-red-800',
      PENDING: 'bg-gray-100 text-gray-800',
    };
    return styles[status] || 'bg-gray-100 text-gray-800';
  };

  const getTypeBadge = (type: string) => {
    const styles: Record<string, string> = {
      FULL_TIME: 'bg-blue-100 text-blue-800',
      PART_TIME: 'bg-purple-100 text-purple-800',
      CONTRACTOR: 'bg-orange-100 text-orange-800',
      INTERN: 'bg-pink-100 text-pink-800',
    };
    return styles[type] || 'bg-gray-100 text-gray-800';
  };

  const stats = useMemo(() => ({
    total: (employees || []).length,
    active: (employees || []).filter(e => e.employment_status === 'ACTIVE').length,
    onLeave: (employees || []).filter(e => e.employment_status === 'ON_LEAVE').length,
    contractors: (employees || []).filter(e => e.employment_type === 'CONTRACTOR').length,
  }), [employees]);

  return (
    <DashboardLayout title="Employees" subtitle="Manage your team directory">
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
            <h1 className="text-2xl font-bold text-gray-900">Employees</h1>
            <p className="text-gray-600">Manage your team directory</p>
          </div>
          <button className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700">
            <Plus className="w-4 h-4" />
            Add Employee
          </button>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-4 gap-4">
          <div className="bg-white p-4 rounded-lg border border-gray-200">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Users className="w-5 h-5 text-blue-600" />
              </div>
              <div>
                <p className="text-sm text-gray-500">Total Employees</p>
                <p className="text-2xl font-bold text-gray-900">{loading ? '...' : stats.total}</p>
              </div>
            </div>
          </div>
          <div className="bg-white p-4 rounded-lg border border-gray-200">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-green-100 rounded-lg">
                <UserCircle className="w-5 h-5 text-green-600" />
              </div>
              <div>
                <p className="text-sm text-gray-500">Active</p>
                <p className="text-2xl font-bold text-green-600">{loading ? '...' : stats.active}</p>
              </div>
            </div>
          </div>
          <div className="bg-white p-4 rounded-lg border border-gray-200">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-yellow-100 rounded-lg">
                <Calendar className="w-5 h-5 text-yellow-600" />
              </div>
              <div>
                <p className="text-sm text-gray-500">On Leave</p>
                <p className="text-2xl font-bold text-yellow-600">{loading ? '...' : stats.onLeave}</p>
              </div>
            </div>
          </div>
          <div className="bg-white p-4 rounded-lg border border-gray-200">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-orange-100 rounded-lg">
                <Building2 className="w-5 h-5 text-orange-600" />
              </div>
              <div>
                <p className="text-sm text-gray-500">Contractors</p>
                <p className="text-2xl font-bold text-orange-600">{loading ? '...' : stats.contractors}</p>
              </div>
            </div>
          </div>
        </div>

        {/* Filters */}
        <div className="flex gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search employees..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <select
            value={departmentFilter}
            onChange={(e) => setDepartmentFilter(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          >
            {departments.map(dept => (
              <option key={dept} value={dept}>{dept}</option>
            ))}
          </select>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          >
            <option value="ALL">All Status</option>
            <option value="ACTIVE">Active</option>
            <option value="ON_LEAVE">On Leave</option>
            <option value="TERMINATED">Terminated</option>
          </select>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="w-8 h-8 text-blue-500 animate-spin" />
          </div>
        )}

        {/* Error State */}
        {error && !isUsingFallback && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-center">
            <p className="text-red-700">{error}</p>
            <button onClick={refetch} className="mt-2 text-red-600 underline">
              Try again
            </button>
          </div>
        )}

        {/* Employee Grid */}
        {!loading && (
          <div className="grid grid-cols-3 gap-4">
            {filteredEmployees.map((employee) => {
              const avatarColor = getAvatarColor(`${employee.first_name} ${employee.last_name}`);
              return (
                <div
                  key={employee.id}
                  onClick={() => setSelectedEmployee(employee)}
                  className="bg-white rounded-lg border border-gray-200 p-4 hover:shadow-md transition-shadow cursor-pointer"
                >
                  <div className="flex items-start gap-4">
                    <div className={`w-12 h-12 rounded-full bg-gradient-to-br ${avatarColor} flex items-center justify-center text-white font-semibold`}>
                      {employee.first_name[0]}{employee.last_name[0]}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <h3 className="font-medium text-gray-900 truncate">
                          {employee.first_name} {employee.last_name}
                        </h3>
                        <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${getStatusBadge(employee.employment_status)}`}>
                          {employee.employment_status.replace('_', ' ')}
                        </span>
                      </div>
                      <p className="text-sm text-gray-500">{employee.job_title}</p>
                      <div className="flex items-center gap-2 mt-1">
                        <Building2 className="w-3 h-3 text-gray-400" />
                        <span className="text-xs text-gray-500">{employee.department_name || 'Unassigned'}</span>
                      </div>
                    </div>
                  </div>
                  <div className="mt-4 pt-4 border-t border-gray-100">
                    <div className="flex items-center justify-between text-sm">
                      <a href={`mailto:${employee.email}`} className="flex items-center gap-1 text-gray-500 hover:text-blue-600">
                        <Mail className="w-4 h-4" />
                        <span className="truncate max-w-[150px]">{employee.email}</span>
                      </a>
                      {employee.phone && (
                        <a href={`tel:${employee.phone}`} className="flex items-center gap-1 text-gray-500 hover:text-blue-600">
                          <Phone className="w-4 h-4" />
                        </a>
                      )}
                    </div>
                  </div>
                  <div className="mt-2 flex items-center justify-between">
                    <span className={`px-2 py-0.5 text-xs font-medium rounded ${getTypeBadge(employee.employment_type)}`}>
                      {employee.employment_type.replace('_', ' ')}
                    </span>
                    <span className="text-xs text-gray-400">
                      Since {formatDate(employee.hire_date, { month: 'short', year: 'numeric' })}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Empty State */}
        {!loading && filteredEmployees.length === 0 && (
          <div className="text-center py-12">
            <Users className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">No employees found matching your filters.</p>
          </div>
        )}

        {/* Employee Detail Modal */}
        {selectedEmployee && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-6 w-full max-w-lg">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-4">
                  <div className={`w-16 h-16 rounded-full bg-gradient-to-br ${getAvatarColor(`${selectedEmployee.first_name} ${selectedEmployee.last_name}`)} flex items-center justify-center text-white text-xl font-semibold`}>
                    {selectedEmployee.first_name[0]}{selectedEmployee.last_name[0]}
                  </div>
                  <div>
                    <h2 className="text-xl font-semibold">{selectedEmployee.first_name} {selectedEmployee.last_name}</h2>
                    <p className="text-gray-500">{selectedEmployee.job_title}</p>
                    <p className="text-sm text-gray-400">{selectedEmployee.employee_number}</p>
                  </div>
                </div>
                <button onClick={() => setSelectedEmployee(null)} className="text-gray-400 hover:text-gray-600 text-2xl">
                  &times;
                </button>
              </div>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-gray-500">Department</p>
                    <p className="font-medium">{selectedEmployee.department_name || 'Unassigned'}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Manager</p>
                    <p className="font-medium">{selectedEmployee.manager_name || 'None'}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Email</p>
                    <p className="font-medium">{selectedEmployee.email}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Phone</p>
                    <p className="font-medium">{selectedEmployee.phone || 'N/A'}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Employment Type</p>
                    <span className={`px-2 py-1 text-xs font-medium rounded ${getTypeBadge(selectedEmployee.employment_type)}`}>
                      {selectedEmployee.employment_type.replace('_', ' ')}
                    </span>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Status</p>
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusBadge(selectedEmployee.employment_status)}`}>
                      {selectedEmployee.employment_status.replace('_', ' ')}
                    </span>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Start Date</p>
                    <p className="font-medium">{formatDate(selectedEmployee.hire_date)}</p>
                  </div>
                </div>
                <div className="flex gap-3 pt-4">
                  <button
                    onClick={() => setSelectedEmployee(null)}
                    className="flex-1 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
                  >
                    Close
                  </button>
                  <button className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
                    Edit Profile
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
