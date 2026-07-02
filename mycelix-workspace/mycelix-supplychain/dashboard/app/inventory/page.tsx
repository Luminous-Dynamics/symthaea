// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { Package, Warehouse, TrendingDown, DollarSign, ArrowUpRight, ArrowDownRight, AlertTriangle } from 'lucide-react';
import { useApiWithFallback } from '@/lib/hooks';
import { api } from '@/lib/api';

interface InventorySummary {
  total_products: number;
  total_warehouses: number;
  low_stock_count: number;
  total_value: number;
}

interface LowStockProduct {
  id: string;
  sku: string;
  name: string;
  quantity_on_hand: number;
  reorder_point: number;
}

const mockSummary: InventorySummary = {
  total_products: 156,
  total_warehouses: 3,
  low_stock_count: 8,
  total_value: 125000.00,
};

const mockLowStock: LowStockProduct[] = [
  { id: '1', sku: 'LAPTOP-PRO-15', name: 'Professional Laptop 15"', quantity_on_hand: 5, reorder_point: 10 },
  { id: '2', sku: 'MONITOR-27', name: '27" LED Monitor', quantity_on_hand: 12, reorder_point: 15 },
  { id: '3', sku: 'KEYBOARD-MECH', name: 'Mechanical Keyboard', quantity_on_hand: 20, reorder_point: 25 },
];

export default function InventoryPage() {
  const [summary, setSummary] = useState<InventorySummary>(mockSummary);
  const [isDemo, setIsDemo] = useState(true);

  const { data: lowStockProducts, isUsingFallback } = useApiWithFallback<LowStockProduct[]>(
    () => api.getInventoryLowStock(),
    mockLowStock
  );

  useEffect(() => {
    setIsDemo(isUsingFallback);
  }, [isUsingFallback]);

  const stats = [
    {
      name: 'Total Products',
      value: summary.total_products.toLocaleString(),
      icon: Package,
      href: '/inventory/products',
      color: 'text-blue-600',
      bgColor: 'bg-blue-50',
    },
    {
      name: 'Warehouses',
      value: summary.total_warehouses.toLocaleString(),
      icon: Warehouse,
      href: '/inventory/warehouses',
      color: 'text-green-600',
      bgColor: 'bg-green-50',
    },
    {
      name: 'Low Stock Items',
      value: summary.low_stock_count.toLocaleString(),
      icon: TrendingDown,
      href: '/inventory/stock?filter=low',
      color: 'text-amber-600',
      bgColor: 'bg-amber-50',
    },
    {
      name: 'Inventory Value',
      value: `$${summary.total_value.toLocaleString()}`,
      icon: DollarSign,
      href: '/inventory/stock',
      color: 'text-purple-600',
      bgColor: 'bg-purple-50',
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Inventory Management</h1>
          <p className="mt-1 text-sm text-gray-500">
            Track products, manage warehouses, and monitor stock levels
          </p>
        </div>
        <div className="flex space-x-3">
          <Link
            href="/inventory/products/new"
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700"
          >
            Add Product
          </Link>
          <Link
            href="/inventory/stock/receive"
            className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md shadow-sm text-gray-700 bg-white hover:bg-gray-50"
          >
            Receive Stock
          </Link>
        </div>
      </div>

      {/* Demo Mode Banner */}
      {isDemo && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
          <div className="flex items-center">
            <AlertTriangle className="h-5 w-5 text-amber-600 mr-2" />
            <span className="text-amber-800 text-sm">
              Demo Mode: Showing sample data. Connect to the backend to see real inventory data.
            </span>
          </div>
        </div>
      )}

      {/* Stats Grid */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => (
          <Link
            key={stat.name}
            href={stat.href}
            className="relative bg-white overflow-hidden rounded-lg shadow hover:shadow-md transition-shadow"
          >
            <div className="p-5">
              <div className="flex items-center">
                <div className={`${stat.bgColor} rounded-md p-3`}>
                  <stat.icon className={`h-6 w-6 ${stat.color}`} />
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">{stat.name}</dt>
                    <dd className="text-lg font-semibold text-gray-900">{stat.value}</dd>
                  </dl>
                </div>
              </div>
            </div>
            <div className="bg-gray-50 px-5 py-3">
              <div className="text-sm flex items-center text-blue-600 hover:text-blue-700">
                View details
                <ArrowUpRight className="ml-1 h-4 w-4" />
              </div>
            </div>
          </Link>
        ))}
      </div>

      {/* Low Stock Alert */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 border-b border-gray-200 sm:px-6">
          <div className="flex items-center justify-between">
            <h3 className="text-lg leading-6 font-medium text-gray-900 flex items-center">
              <AlertTriangle className="h-5 w-5 text-amber-500 mr-2" />
              Low Stock Alerts
            </h3>
            <Link
              href="/inventory/stock?filter=low"
              className="text-sm text-blue-600 hover:text-blue-700"
            >
              View all
            </Link>
          </div>
        </div>
        <div className="divide-y divide-gray-200">
          {lowStockProducts.map((product) => (
            <div key={product.id} className="px-4 py-4 sm:px-6 hover:bg-gray-50">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <Package className="h-8 w-8 text-gray-400 mr-3" />
                  <div>
                    <p className="text-sm font-medium text-gray-900">{product.name}</p>
                    <p className="text-sm text-gray-500">SKU: {product.sku}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm text-gray-900">
                    <span className="font-semibold text-red-600">{product.quantity_on_hand}</span>
                    <span className="text-gray-500"> / {product.reorder_point} min</span>
                  </p>
                  <div className="mt-1 flex items-center text-sm text-red-600">
                    <ArrowDownRight className="h-4 w-4 mr-1" />
                    Below reorder point
                  </div>
                </div>
              </div>
            </div>
          ))}
          {lowStockProducts.length === 0 && (
            <div className="px-4 py-8 text-center text-gray-500">
              No low stock alerts
            </div>
          )}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-3">
        <Link
          href="/inventory/stock/receive"
          className="bg-white shadow rounded-lg p-6 hover:shadow-md transition-shadow"
        >
          <div className="flex items-center">
            <div className="bg-green-100 rounded-lg p-3">
              <ArrowDownRight className="h-6 w-6 text-green-600" />
            </div>
            <div className="ml-4">
              <h3 className="text-lg font-medium text-gray-900">Receive Stock</h3>
              <p className="text-sm text-gray-500">Record incoming inventory</p>
            </div>
          </div>
        </Link>

        <Link
          href="/inventory/stock/transfer"
          className="bg-white shadow rounded-lg p-6 hover:shadow-md transition-shadow"
        >
          <div className="flex items-center">
            <div className="bg-blue-100 rounded-lg p-3">
              <Package className="h-6 w-6 text-blue-600" />
            </div>
            <div className="ml-4">
              <h3 className="text-lg font-medium text-gray-900">Transfer Stock</h3>
              <p className="text-sm text-gray-500">Move between locations</p>
            </div>
          </div>
        </Link>

        <Link
          href="/inventory/stock/adjust"
          className="bg-white shadow rounded-lg p-6 hover:shadow-md transition-shadow"
        >
          <div className="flex items-center">
            <div className="bg-purple-100 rounded-lg p-3">
              <TrendingDown className="h-6 w-6 text-purple-600" />
            </div>
            <div className="ml-4">
              <h3 className="text-lg font-medium text-gray-900">Adjust Stock</h3>
              <p className="text-sm text-gray-500">Correct inventory counts</p>
            </div>
          </div>
        </Link>
      </div>
    </div>
  );
}
