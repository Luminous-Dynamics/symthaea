// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Package, Warehouse, ArrowDownRight, ArrowUpRight, Search, Filter, AlertTriangle, RefreshCw } from 'lucide-react';
import { useApiWithFallback, formatCurrency } from '@/lib/hooks';
import { api, StockLevel } from '@/lib/api';

interface StockLevelWithDetails extends StockLevel {
  product_sku?: string;
  product_name?: string;
  warehouse_code?: string;
  warehouse_name?: string;
}

const mockStockLevels: StockLevelWithDetails[] = [
  {
    id: '1',
    tenant_id: 'demo',
    product_id: 'p1',
    warehouse_id: 'w1',
    quantity_on_hand: 45,
    quantity_reserved: 5,
    quantity_available: 40,
    quantity_on_order: 25,
    unit_cost: 850.00,
    total_value: 38250.00,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    product_sku: 'LAPTOP-PRO-15',
    product_name: 'Professional Laptop 15"',
    warehouse_code: 'MAIN',
    warehouse_name: 'Main Distribution Center',
  },
  {
    id: '2',
    tenant_id: 'demo',
    product_id: 'p2',
    warehouse_id: 'w1',
    quantity_on_hand: 78,
    quantity_reserved: 10,
    quantity_available: 68,
    quantity_on_order: 0,
    unit_cost: 220.00,
    total_value: 17160.00,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    product_sku: 'MONITOR-27',
    product_name: '27" LED Monitor',
    warehouse_code: 'MAIN',
    warehouse_name: 'Main Distribution Center',
  },
  {
    id: '3',
    tenant_id: 'demo',
    product_id: 'p3',
    warehouse_id: 'w1',
    quantity_on_hand: 156,
    quantity_reserved: 20,
    quantity_available: 136,
    quantity_on_order: 50,
    unit_cost: 45.00,
    total_value: 7020.00,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    product_sku: 'KEYBOARD-MECH',
    product_name: 'Mechanical Keyboard',
    warehouse_code: 'MAIN',
    warehouse_name: 'Main Distribution Center',
  },
  {
    id: '4',
    tenant_id: 'demo',
    product_id: 'p1',
    warehouse_id: 'w2',
    quantity_on_hand: 20,
    quantity_reserved: 3,
    quantity_available: 17,
    quantity_on_order: 0,
    unit_cost: 850.00,
    total_value: 17000.00,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    product_sku: 'LAPTOP-PRO-15',
    product_name: 'Professional Laptop 15"',
    warehouse_code: 'EAST',
    warehouse_name: 'East Coast Warehouse',
  },
  {
    id: '5',
    tenant_id: 'demo',
    product_id: 'p4',
    warehouse_id: 'w1',
    quantity_on_hand: 500,
    quantity_reserved: 50,
    quantity_available: 450,
    quantity_on_order: 200,
    unit_cost: 4.50,
    total_value: 2250.00,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    product_sku: 'PAPER-A4-500',
    product_name: 'A4 Copy Paper (500 sheets)',
    warehouse_code: 'MAIN',
    warehouse_name: 'Main Distribution Center',
  },
];

export default function StockPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedWarehouse, setSelectedWarehouse] = useState<string>('all');
  const [showLowStockOnly, setShowLowStockOnly] = useState(false);

  const { data: stockLevels, loading, isUsingFallback } = useApiWithFallback<StockLevelWithDetails[]>(
    () => api.getInventoryStockValuation() as Promise<StockLevelWithDetails[]>,
    mockStockLevels
  );

  const warehouses = Array.from(new Set(stockLevels.map(s => s.warehouse_code))).filter(Boolean);

  const filteredStock = stockLevels.filter((stock) => {
    const matchesSearch =
      !searchQuery ||
      stock.product_name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      stock.product_sku?.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesWarehouse = selectedWarehouse === 'all' || stock.warehouse_code === selectedWarehouse;
    const matchesLowStock = !showLowStockOnly || stock.quantity_available < 20; // Simple threshold
    return matchesSearch && matchesWarehouse && matchesLowStock;
  });

  const totalValue = filteredStock.reduce((sum, s) => sum + (s.total_value || 0), 0);
  const totalOnHand = filteredStock.reduce((sum, s) => sum + s.quantity_on_hand, 0);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Stock Levels</h1>
          <p className="mt-1 text-sm text-gray-500">
            View and manage inventory across all locations
          </p>
        </div>
        <div className="flex space-x-3">
          <Link
            href="/inventory/stock/receive"
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700"
          >
            <ArrowDownRight className="h-4 w-4 mr-2" />
            Receive
          </Link>
          <Link
            href="/inventory/stock/transfer"
            className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md shadow-sm text-gray-700 bg-white hover:bg-gray-50"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Transfer
          </Link>
        </div>
      </div>

      {/* Demo Mode Banner */}
      {isUsingFallback && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
          <div className="flex items-center">
            <AlertTriangle className="h-5 w-5 text-amber-600 mr-2" />
            <span className="text-amber-800 text-sm">
              Demo Mode: Showing sample data.
            </span>
          </div>
        </div>
      )}

      {/* Summary Cards */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-3">
        <div className="bg-white overflow-hidden shadow rounded-lg">
          <div className="p-5">
            <div className="flex items-center">
              <div className="bg-blue-100 rounded-md p-3">
                <Package className="h-6 w-6 text-blue-600" />
              </div>
              <div className="ml-5">
                <p className="text-sm font-medium text-gray-500">Total Units</p>
                <p className="text-2xl font-semibold text-gray-900">{totalOnHand.toLocaleString()}</p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white overflow-hidden shadow rounded-lg">
          <div className="p-5">
            <div className="flex items-center">
              <div className="bg-green-100 rounded-md p-3">
                <ArrowUpRight className="h-6 w-6 text-green-600" />
              </div>
              <div className="ml-5">
                <p className="text-sm font-medium text-gray-500">Total Value</p>
                <p className="text-2xl font-semibold text-gray-900">{formatCurrency(totalValue)}</p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white overflow-hidden shadow rounded-lg">
          <div className="p-5">
            <div className="flex items-center">
              <div className="bg-amber-100 rounded-md p-3">
                <Warehouse className="h-6 w-6 text-amber-600" />
              </div>
              <div className="ml-5">
                <p className="text-sm font-medium text-gray-500">Locations</p>
                <p className="text-2xl font-semibold text-gray-900">{warehouses.length}</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-white shadow rounded-lg p-4">
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search by product name or SKU..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md leading-5 bg-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Filter className="h-5 w-5 text-gray-400" />
              <select
                value={selectedWarehouse}
                onChange={(e) => setSelectedWarehouse(e.target.value)}
                className="block pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 rounded-md"
              >
                <option value="all">All Warehouses</option>
                {warehouses.map((wh) => (
                  <option key={wh} value={wh}>{wh}</option>
                ))}
              </select>
            </div>
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={showLowStockOnly}
                onChange={(e) => setShowLowStockOnly(e.target.checked)}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <span className="ml-2 text-sm text-gray-600">Low stock only</span>
            </label>
          </div>
        </div>
      </div>

      {/* Stock Table */}
      <div className="bg-white shadow rounded-lg overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Product
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Warehouse
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                On Hand
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                Reserved
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                Available
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                On Order
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                Unit Cost
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                Total Value
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {loading ? (
              <tr>
                <td colSpan={8} className="px-6 py-12 text-center text-gray-500">
                  Loading stock levels...
                </td>
              </tr>
            ) : filteredStock.length === 0 ? (
              <tr>
                <td colSpan={8} className="px-6 py-12 text-center text-gray-500">
                  No stock records found
                </td>
              </tr>
            ) : (
              filteredStock.map((stock) => (
                <tr key={stock.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <Package className="h-8 w-8 text-gray-400 mr-3" />
                      <div>
                        <div className="text-sm font-medium text-gray-900">
                          {stock.product_name}
                        </div>
                        <div className="text-sm text-gray-500 font-mono">
                          {stock.product_sku}
                        </div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <Warehouse className="h-4 w-4 text-gray-400 mr-2" />
                      <div>
                        <div className="text-sm font-medium text-gray-900">
                          {stock.warehouse_code}
                        </div>
                        <div className="text-sm text-gray-500">
                          {stock.warehouse_name}
                        </div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-semibold text-gray-900">
                    {stock.quantity_on_hand}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm text-amber-600">
                    {stock.quantity_reserved > 0 ? stock.quantity_reserved : '-'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm">
                    <span className={`font-semibold ${stock.quantity_available < 20 ? 'text-red-600' : 'text-green-600'}`}>
                      {stock.quantity_available}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm text-blue-600">
                    {stock.quantity_on_order > 0 ? `+${stock.quantity_on_order}` : '-'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm text-gray-900">
                    {stock.unit_cost ? formatCurrency(stock.unit_cost) : '-'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-semibold text-gray-900">
                    {stock.total_value ? formatCurrency(stock.total_value) : '-'}
                  </td>
                </tr>
              ))
            )}
          </tbody>
          <tfoot className="bg-gray-50">
            <tr>
              <td colSpan={2} className="px-6 py-3 text-sm font-medium text-gray-900">
                Total
              </td>
              <td className="px-6 py-3 text-right text-sm font-bold text-gray-900">
                {totalOnHand.toLocaleString()}
              </td>
              <td colSpan={4}></td>
              <td className="px-6 py-3 text-right text-sm font-bold text-gray-900">
                {formatCurrency(totalValue)}
              </td>
            </tr>
          </tfoot>
        </table>
      </div>
    </div>
  );
}
