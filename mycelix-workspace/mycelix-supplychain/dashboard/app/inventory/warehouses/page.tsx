// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Warehouse, MapPin, Package, Plus, Edit2, ArrowUpRight, AlertTriangle } from 'lucide-react';
import { useApiWithFallback, formatCurrency } from '@/lib/hooks';
import { api, InventoryWarehouse } from '@/lib/api';

const mockWarehouses: InventoryWarehouse[] = [
  {
    id: '1',
    tenant_id: 'demo',
    code: 'MAIN',
    name: 'Main Distribution Center',
    warehouse_type: 'DISTRIBUTION',
    city: 'Dallas',
    state: 'TX',
    country: 'USA',
    contact_name: 'John Smith',
    contact_email: 'john.smith@example.com',
    is_active: true,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  },
  {
    id: '2',
    tenant_id: 'demo',
    code: 'EAST',
    name: 'East Coast Warehouse',
    warehouse_type: 'DISTRIBUTION',
    city: 'Atlanta',
    state: 'GA',
    country: 'USA',
    contact_name: 'Jane Doe',
    contact_email: 'jane.doe@example.com',
    is_active: true,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  },
  {
    id: '3',
    tenant_id: 'demo',
    code: 'RETAIL-01',
    name: 'Downtown Retail Store',
    warehouse_type: 'RETAIL',
    city: 'Dallas',
    state: 'TX',
    country: 'USA',
    contact_name: 'Mike Johnson',
    contact_email: 'mike.johnson@example.com',
    is_active: true,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  },
];

interface WarehouseStats {
  locationCount: number;
  productCount: number;
  totalValue: number;
}

const mockStats: Record<string, WarehouseStats> = {
  '1': { locationCount: 24, productCount: 156, totalValue: 125000 },
  '2': { locationCount: 12, productCount: 89, totalValue: 67500 },
  '3': { locationCount: 6, productCount: 45, totalValue: 34200 },
};

export default function WarehousesPage() {
  const { data: warehouses, loading, isUsingFallback } = useApiWithFallback<InventoryWarehouse[]>(
    () => api.getInventoryWarehouses(),
    mockWarehouses
  );

  const getWarehouseTypeStyle = (type: string) => {
    switch (type) {
      case 'DISTRIBUTION':
        return 'bg-blue-100 text-blue-800';
      case 'RETAIL':
        return 'bg-green-100 text-green-800';
      case 'MANUFACTURING':
        return 'bg-purple-100 text-purple-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Warehouses</h1>
          <p className="mt-1 text-sm text-gray-500">
            Manage storage locations and facilities
          </p>
        </div>
        <Link
          href="/inventory/warehouses/new"
          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700"
        >
          <Plus className="h-4 w-4 mr-2" />
          Add Warehouse
        </Link>
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

      {/* Warehouse Grid */}
      {loading ? (
        <div className="text-center py-12 text-gray-500">Loading warehouses...</div>
      ) : (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2 xl:grid-cols-3">
          {warehouses.map((warehouse) => {
            const stats = mockStats[warehouse.id] || { locationCount: 0, productCount: 0, totalValue: 0 };

            return (
              <div
                key={warehouse.id}
                className="bg-white shadow rounded-lg overflow-hidden hover:shadow-md transition-shadow"
              >
                <div className="p-6">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center">
                      <div className="bg-blue-100 rounded-lg p-3">
                        <Warehouse className="h-6 w-6 text-blue-600" />
                      </div>
                      <div className="ml-4">
                        <h3 className="text-lg font-medium text-gray-900">{warehouse.name}</h3>
                        <p className="text-sm text-gray-500 font-mono">{warehouse.code}</p>
                      </div>
                    </div>
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${getWarehouseTypeStyle(warehouse.warehouse_type)}`}>
                      {warehouse.warehouse_type}
                    </span>
                  </div>

                  <div className="mt-4 flex items-center text-sm text-gray-500">
                    <MapPin className="h-4 w-4 mr-1" />
                    {[warehouse.city, warehouse.state, warehouse.country]
                      .filter(Boolean)
                      .join(', ')}
                  </div>

                  {warehouse.contact_name && (
                    <div className="mt-2 text-sm text-gray-500">
                      Contact: {warehouse.contact_name}
                    </div>
                  )}

                  <div className="mt-6 grid grid-cols-3 gap-4 border-t border-gray-200 pt-4">
                    <div>
                      <p className="text-sm font-medium text-gray-500">Locations</p>
                      <p className="mt-1 text-lg font-semibold text-gray-900">
                        {stats.locationCount}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-gray-500">Products</p>
                      <p className="mt-1 text-lg font-semibold text-gray-900">
                        {stats.productCount}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-gray-500">Value</p>
                      <p className="mt-1 text-lg font-semibold text-gray-900">
                        {formatCurrency(stats.totalValue)}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-50 px-6 py-3 flex items-center justify-between">
                  <Link
                    href={`/inventory/warehouses/${warehouse.id}`}
                    className="text-sm text-blue-600 hover:text-blue-700 flex items-center"
                  >
                    View Details
                    <ArrowUpRight className="ml-1 h-4 w-4" />
                  </Link>
                  <div className="flex items-center space-x-3">
                    <Link
                      href={`/inventory/warehouses/${warehouse.id}/locations`}
                      className="text-sm text-gray-600 hover:text-gray-900"
                    >
                      Locations
                    </Link>
                    <Link
                      href={`/inventory/warehouses/${warehouse.id}/stock`}
                      className="text-sm text-gray-600 hover:text-gray-900 flex items-center"
                    >
                      <Package className="h-4 w-4 mr-1" />
                      Stock
                    </Link>
                    <button className="text-gray-400 hover:text-gray-600">
                      <Edit2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Empty State */}
      {!loading && warehouses.length === 0 && (
        <div className="text-center py-12 bg-white shadow rounded-lg">
          <Warehouse className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">No warehouses</h3>
          <p className="mt-1 text-sm text-gray-500">
            Get started by creating a new warehouse.
          </p>
          <div className="mt-6">
            <Link
              href="/inventory/warehouses/new"
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700"
            >
              <Plus className="h-4 w-4 mr-2" />
              Add Warehouse
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}
