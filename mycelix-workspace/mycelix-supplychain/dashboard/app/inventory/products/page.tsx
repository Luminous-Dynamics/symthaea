// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Package, Search, Plus, Edit2, Trash2, AlertTriangle, Filter } from 'lucide-react';
import { useApiWithFallback, formatCurrency } from '@/lib/hooks';
import { api, Product } from '@/lib/api';

const mockProducts: Product[] = [
  {
    id: '1',
    tenant_id: 'demo',
    sku: 'LAPTOP-PRO-15',
    name: 'Professional Laptop 15"',
    description: 'High-performance business laptop with 15" display',
    category_id: 'elec',
    product_type: 'STOCKABLE',
    status: 'ACTIVE',
    unit_of_measure: 'EACH',
    cost_price: 850.00,
    sale_price: 1299.99,
    currency: 'USD',
    barcode: '123456789012',
    reorder_point: 10,
    reorder_quantity: 25,
    is_active: true,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  },
  {
    id: '2',
    tenant_id: 'demo',
    sku: 'MONITOR-27',
    name: '27" LED Monitor',
    description: 'High-resolution LED monitor for professional use',
    category_id: 'elec',
    product_type: 'STOCKABLE',
    status: 'ACTIVE',
    unit_of_measure: 'EACH',
    cost_price: 220.00,
    sale_price: 349.99,
    currency: 'USD',
    barcode: '123456789013',
    reorder_point: 15,
    reorder_quantity: 30,
    is_active: true,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  },
  {
    id: '3',
    tenant_id: 'demo',
    sku: 'KEYBOARD-MECH',
    name: 'Mechanical Keyboard',
    description: 'Ergonomic mechanical keyboard with RGB',
    category_id: 'elec',
    product_type: 'STOCKABLE',
    status: 'ACTIVE',
    unit_of_measure: 'EACH',
    cost_price: 45.00,
    sale_price: 89.99,
    currency: 'USD',
    barcode: '123456789014',
    reorder_point: 25,
    reorder_quantity: 50,
    is_active: true,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  },
  {
    id: '4',
    tenant_id: 'demo',
    sku: 'PAPER-A4-500',
    name: 'A4 Copy Paper (500 sheets)',
    description: 'Standard white A4 copy paper, 500 sheet ream',
    category_id: 'office',
    product_type: 'CONSUMABLE',
    status: 'ACTIVE',
    unit_of_measure: 'BOX',
    cost_price: 4.50,
    sale_price: 8.99,
    currency: 'USD',
    reorder_point: 100,
    reorder_quantity: 200,
    is_active: true,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  },
  {
    id: '5',
    tenant_id: 'demo',
    sku: 'PENS-BLUE-12',
    name: 'Blue Ballpoint Pens (12 pack)',
    description: 'Standard blue ballpoint pens, 12 per pack',
    category_id: 'office',
    product_type: 'CONSUMABLE',
    status: 'ACTIVE',
    unit_of_measure: 'BOX',
    cost_price: 2.00,
    sale_price: 5.99,
    currency: 'USD',
    reorder_point: 50,
    reorder_quantity: 100,
    is_active: true,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  },
];

export default function ProductsPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedType, setSelectedType] = useState<string>('all');

  const { data: products, loading, isUsingFallback } = useApiWithFallback<Product[]>(
    () => api.getInventoryProducts(),
    mockProducts
  );

  const filteredProducts = products.filter((product) => {
    const matchesSearch =
      product.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      product.sku.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesType = selectedType === 'all' || product.product_type === selectedType;
    return matchesSearch && matchesType;
  });

  const productTypes = ['all', 'STOCKABLE', 'CONSUMABLE', 'SERVICE'];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Products</h1>
          <p className="mt-1 text-sm text-gray-500">
            Manage your product catalog
          </p>
        </div>
        <Link
          href="/inventory/products/new"
          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700"
        >
          <Plus className="h-4 w-4 mr-2" />
          Add Product
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

      {/* Search and Filter */}
      <div className="bg-white shadow rounded-lg p-4">
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search products by name or SKU..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md leading-5 bg-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
          <div className="flex items-center space-x-2">
            <Filter className="h-5 w-5 text-gray-400" />
            <select
              value={selectedType}
              onChange={(e) => setSelectedType(e.target.value)}
              className="block pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 rounded-md"
            >
              {productTypes.map((type) => (
                <option key={type} value={type}>
                  {type === 'all' ? 'All Types' : type}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Products Table */}
      <div className="bg-white shadow rounded-lg overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Product
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                SKU
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Type
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Cost
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Price
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Status
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {loading ? (
              <tr>
                <td colSpan={7} className="px-6 py-12 text-center text-gray-500">
                  Loading products...
                </td>
              </tr>
            ) : filteredProducts.length === 0 ? (
              <tr>
                <td colSpan={7} className="px-6 py-12 text-center text-gray-500">
                  No products found
                </td>
              </tr>
            ) : (
              filteredProducts.map((product) => (
                <tr key={product.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <Package className="h-8 w-8 text-gray-400 mr-3" />
                      <div>
                        <div className="text-sm font-medium text-gray-900">
                          {product.name}
                        </div>
                        <div className="text-sm text-gray-500">
                          {product.description?.substring(0, 50)}...
                        </div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="text-sm font-mono text-gray-900">{product.sku}</span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                      product.product_type === 'STOCKABLE'
                        ? 'bg-blue-100 text-blue-800'
                        : product.product_type === 'CONSUMABLE'
                        ? 'bg-green-100 text-green-800'
                        : 'bg-purple-100 text-purple-800'
                    }`}>
                      {product.product_type}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {product.cost_price ? formatCurrency(product.cost_price) : '-'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {product.sale_price ? formatCurrency(product.sale_price) : '-'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                      product.status === 'ACTIVE'
                        ? 'bg-green-100 text-green-800'
                        : product.status === 'INACTIVE'
                        ? 'bg-gray-100 text-gray-800'
                        : 'bg-red-100 text-red-800'
                    }`}>
                      {product.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                    <div className="flex items-center justify-end space-x-2">
                      <Link
                        href={`/inventory/products/${product.id}`}
                        className="text-blue-600 hover:text-blue-900 p-1"
                      >
                        <Edit2 className="h-4 w-4" />
                      </Link>
                      <button className="text-red-600 hover:text-red-900 p-1">
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Summary */}
      <div className="flex items-center justify-between text-sm text-gray-500">
        <span>
          Showing {filteredProducts.length} of {products.length} products
        </span>
      </div>
    </div>
  );
}
