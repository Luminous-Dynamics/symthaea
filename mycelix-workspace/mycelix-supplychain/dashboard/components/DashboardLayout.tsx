// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useEffect, useState } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import Link from 'next/link';
import {
  LayoutDashboard,
  FileText,
  Receipt,
  CreditCard,
  BookOpen,
  LogOut,
  Menu,
  X,
  Building2,
  DollarSign,
  Sparkles,
  Users,
  Target,
  TrendingUp,
  UserCircle,
  Calendar,
  Banknote,
  Package,
  Warehouse,
  ArrowDownUp,
} from 'lucide-react';
import { api } from '@/lib/api';

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  // Finance
  { name: 'Chart of Accounts', href: '/accounts', icon: BookOpen },
  { name: 'Invoices', href: '/invoices', icon: FileText },
  { name: 'Bills', href: '/bills', icon: Receipt },
  { name: 'Payments', href: '/payments', icon: CreditCard },
  { name: 'Bank Reconciliation', href: '/reconciliation', icon: Building2 },
  { name: 'Currencies', href: '/currencies', icon: DollarSign },
  { name: 'AI Inbox', href: '/ai-inbox', icon: Sparkles },
  // Inventory
  { name: 'Inventory', href: '/inventory', icon: Package },
  { name: 'Products', href: '/inventory/products', icon: Package },
  { name: 'Warehouses', href: '/inventory/warehouses', icon: Warehouse },
  { name: 'Stock Levels', href: '/inventory/stock', icon: ArrowDownUp },
  // CRM
  { name: 'Contacts', href: '/crm/contacts', icon: Users },
  { name: 'Leads', href: '/crm/leads', icon: Target },
  { name: 'Pipeline', href: '/crm/pipeline', icon: TrendingUp },
  // HR
  { name: 'Employees', href: '/hr/employees', icon: UserCircle },
  { name: 'Time Off', href: '/hr/leave', icon: Calendar },
  { name: 'Payroll', href: '/hr/payroll', icon: Banknote },
];

interface DashboardLayoutProps {
  children: React.ReactNode;
  title: string;
  subtitle?: string;
}

export default function DashboardLayout({ children, title, subtitle }: DashboardLayoutProps) {
  const router = useRouter();
  const pathname = usePathname();
  const [user, setUser] = useState<{ name: string; role: string } | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    const loadUser = async () => {
      try {
        const profile = await api.getProfile();
        setUser(profile);
      } catch {
        router.push('/login');
      }
    };
    loadUser();
  }, [router]);

  const handleLogout = async () => {
    await api.logout();
    router.push('/login');
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Mobile sidebar backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-gray-900/50 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside className={`fixed inset-y-0 left-0 w-64 bg-white border-r border-gray-200 z-50 transform transition-transform lg:translate-x-0 ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}>
        <div className="flex flex-col h-full">
          {/* Logo */}
          <div className="flex items-center justify-between h-16 px-6 border-b border-gray-200">
            <div className="flex items-center">
              <span className="text-xl font-bold text-primary-600">Mycelix</span>
              <span className="ml-2 text-sm text-gray-500">ERP</span>
            </div>
            <button
              className="lg:hidden p-2 text-gray-400 hover:text-gray-600"
              onClick={() => setSidebarOpen(false)}
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-4 py-6 space-y-1 overflow-y-auto">
            {navigation.map((item) => {
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={`flex items-center px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                    isActive
                      ? 'text-primary-600 bg-primary-50'
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                  onClick={() => setSidebarOpen(false)}
                >
                  <item.icon className="w-5 h-5 mr-3" />
                  {item.name}
                </Link>
              );
            })}
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
      <div className="lg:ml-64">
        {/* Mobile header */}
        <header className="lg:hidden flex items-center h-16 px-4 bg-white border-b border-gray-200">
          <button
            className="p-2 text-gray-600"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu className="w-6 h-6" />
          </button>
          <span className="ml-4 text-lg font-semibold text-gray-900">{title}</span>
        </header>

        <main className="p-6 lg:p-8">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-2xl font-bold text-gray-900">{title}</h1>
            {subtitle && <p className="text-gray-500">{subtitle}</p>}
          </div>

          {children}
        </main>
      </div>
    </div>
  );
}
