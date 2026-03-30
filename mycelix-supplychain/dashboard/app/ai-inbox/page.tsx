// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useEffect, useState, useRef } from 'react';
import DashboardLayout from '@/components/DashboardLayout';
import { api, ProcessingQueueItem, ExtractedInvoiceData } from '@/lib/api';
import {
  Upload,
  FileText,
  Clock,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Eye,
  Sparkles,
  RefreshCw,
  Zap,
  FileCheck,
  Inbox,
} from 'lucide-react';

type QueueStatus = 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED' | 'REVIEW';

export default function AiInboxPage() {
  const [queue, setQueue] = useState<ProcessingQueueItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [selectedItem, setSelectedItem] = useState<ProcessingQueueItem | null>(null);
  const [extractedData, setExtractedData] = useState<ExtractedInvoiceData | null>(null);
  const [processing, setProcessing] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadQueue();
  }, []);

  const loadQueue = async () => {
    try {
      setLoading(true);
      const items = await api.getAiQueue();
      setQueue(items);
    } catch (err) {
      console.error('Failed to load AI queue', err);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setUploading(true);
    try {
      for (const file of Array.from(files)) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('tenant_id', 'demo-tenant'); // In production, get from auth context
        await api.uploadDocument(formData);
      }
      loadQueue();
    } catch (err) {
      console.error('Upload failed', err);
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleProcess = async (item: ProcessingQueueItem) => {
    try {
      setProcessing(item.id);
      const extracted = await api.processDocument(item.id);
      setExtractedData(extracted);
      setSelectedItem(item);
      loadQueue();
    } catch (err) {
      console.error('Processing failed', err);
    } finally {
      setProcessing(null);
    }
  };

  const handleApprove = async (createAs: 'INVOICE' | 'BILL') => {
    if (!selectedItem) return;
    try {
      await api.approveExtraction(selectedItem.id, createAs);
      setSelectedItem(null);
      setExtractedData(null);
      loadQueue();
    } catch (err) {
      console.error('Approval failed', err);
    }
  };

  const statusConfig: Record<QueueStatus, { icon: React.ReactNode; color: string; bg: string }> = {
    PENDING: { icon: <Clock className="w-4 h-4" />, color: 'text-gray-600', bg: 'bg-gray-100' },
    PROCESSING: {
      icon: <RefreshCw className="w-4 h-4 animate-spin" />,
      color: 'text-blue-600',
      bg: 'bg-blue-100',
    },
    COMPLETED: {
      icon: <CheckCircle className="w-4 h-4" />,
      color: 'text-green-600',
      bg: 'bg-green-100',
    },
    FAILED: { icon: <XCircle className="w-4 h-4" />, color: 'text-red-600', bg: 'bg-red-100' },
    REVIEW: {
      icon: <AlertTriangle className="w-4 h-4" />,
      color: 'text-orange-600',
      bg: 'bg-orange-100',
    },
  };

  const pendingCount = queue.filter((i) => i.status === 'PENDING').length;
  const processingCount = queue.filter((i) => i.status === 'PROCESSING').length;
  const completedCount = queue.filter((i) => i.status === 'COMPLETED').length;
  const reviewCount = queue.filter((i) => i.status === 'REVIEW').length;

  return (
    <DashboardLayout
      title="AI Invoice Inbox"
      subtitle="Upload and automatically process invoices with AI"
    >
      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500">Pending</p>
              <p className="text-2xl font-bold text-gray-900">{pendingCount}</p>
            </div>
            <div className="w-10 h-10 bg-gray-100 rounded-full flex items-center justify-center">
              <Inbox className="w-5 h-5 text-gray-600" />
            </div>
          </div>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500">Processing</p>
              <p className="text-2xl font-bold text-blue-600">{processingCount}</p>
            </div>
            <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
              <Zap className="w-5 h-5 text-blue-600" />
            </div>
          </div>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500">Needs Review</p>
              <p className="text-2xl font-bold text-orange-600">{reviewCount}</p>
            </div>
            <div className="w-10 h-10 bg-orange-100 rounded-full flex items-center justify-center">
              <Eye className="w-5 h-5 text-orange-600" />
            </div>
          </div>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500">Completed</p>
              <p className="text-2xl font-bold text-green-600">{completedCount}</p>
            </div>
            <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center">
              <FileCheck className="w-5 h-5 text-green-600" />
            </div>
          </div>
        </div>
      </div>

      {/* Upload Area */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6 mb-6">
        <div
          className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-primary-400 transition-colors cursor-pointer"
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.png,.jpg,.jpeg,.tiff"
            onChange={handleFileUpload}
            className="hidden"
          />
          {uploading ? (
            <div className="flex flex-col items-center">
              <RefreshCw className="w-12 h-12 text-primary-600 animate-spin mb-4" />
              <p className="text-lg font-medium text-gray-900">Uploading...</p>
            </div>
          ) : (
            <>
              <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <p className="text-lg font-medium text-gray-900 mb-1">
                Drop invoices here or click to upload
              </p>
              <p className="text-sm text-gray-500">
                Supports PDF, PNG, JPG, TIFF • Max 10MB per file
              </p>
            </>
          )}
        </div>
      </div>

      {/* Queue List */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-100 flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center">
            <Sparkles className="w-5 h-5 mr-2 text-primary-600" />
            Processing Queue
          </h3>
          <button
            onClick={loadQueue}
            className="flex items-center text-sm text-gray-600 hover:text-gray-900"
          >
            <RefreshCw className="w-4 h-4 mr-1" />
            Refresh
          </button>
        </div>

        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
          </div>
        ) : queue.length === 0 ? (
          <div className="p-12 text-center">
            <Inbox className="w-12 h-12 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No Documents in Queue</h3>
            <p className="text-gray-500">Upload invoices to get started with AI processing.</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-100">
            {queue.map((item) => {
              const status = statusConfig[item.status as QueueStatus] || statusConfig.PENDING;
              const isProcessingThis = processing === item.id;

              return (
                <div
                  key={item.id}
                  className="p-4 hover:bg-gray-50 flex items-center justify-between"
                >
                  <div className="flex items-center space-x-4">
                    <div className="w-10 h-10 bg-primary-50 rounded-lg flex items-center justify-center">
                      <FileText className="w-5 h-5 text-primary-600" />
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">
                        {item.file_name || `Document ${item.id.slice(0, 8)}`}
                      </p>
                      <div className="flex items-center text-sm text-gray-500 space-x-3">
                        <span>{new Date(item.created_at).toLocaleString()}</span>
                        {item.file_size && (
                          <span>{(item.file_size / 1024).toFixed(1)} KB</span>
                        )}
                        <span>{item.source_type}</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-4">
                    <span
                      className={`inline-flex items-center px-2.5 py-1 text-xs font-medium rounded-full ${status.bg} ${status.color}`}
                    >
                      {status.icon}
                      <span className="ml-1">{item.status}</span>
                    </span>
                    {item.status === 'PENDING' && (
                      <button
                        onClick={() => handleProcess(item)}
                        disabled={isProcessingThis}
                        className="flex items-center px-3 py-1.5 text-sm bg-primary-600 hover:bg-primary-700 disabled:bg-primary-400 text-white font-medium rounded-lg transition-colors"
                      >
                        {isProcessingThis ? (
                          <RefreshCw className="w-4 h-4 mr-1 animate-spin" />
                        ) : (
                          <Zap className="w-4 h-4 mr-1" />
                        )}
                        Process
                      </button>
                    )}
                    {(item.status === 'COMPLETED' || item.status === 'REVIEW') && (
                      <button
                        onClick={() => {
                          setSelectedItem(item);
                          // Load extracted data
                        }}
                        className="flex items-center px-3 py-1.5 text-sm border border-gray-300 text-gray-700 font-medium rounded-lg hover:bg-gray-50 transition-colors"
                      >
                        <Eye className="w-4 h-4 mr-1" />
                        Review
                      </button>
                    )}
                    {item.status === 'FAILED' && item.error_message && (
                      <span className="text-sm text-red-600 max-w-xs truncate">
                        {item.error_message}
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Extracted Data Modal */}
      {selectedItem && extractedData && (
        <div className="fixed inset-0 bg-gray-900/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl shadow-xl w-full max-w-2xl max-h-[90vh] overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-100 flex items-center justify-between">
              <h3 className="text-lg font-semibold text-gray-900">Extracted Invoice Data</h3>
              <div className="flex items-center space-x-2">
                <span
                  className={`px-2 py-1 text-xs font-medium rounded-full ${
                    extractedData.confidence >= 0.9
                      ? 'bg-green-100 text-green-700'
                      : extractedData.confidence >= 0.7
                      ? 'bg-yellow-100 text-yellow-700'
                      : 'bg-red-100 text-red-700'
                  }`}
                >
                  {(extractedData.confidence * 100).toFixed(0)}% confidence
                </span>
              </div>
            </div>
            <div className="p-6 overflow-y-auto max-h-[60vh]">
              <div className="grid grid-cols-2 gap-4 mb-6">
                <div>
                  <label className="block text-sm font-medium text-gray-500 mb-1">Vendor</label>
                  <p className="text-gray-900 font-medium">
                    {extractedData.vendor_name || 'Unknown'}
                  </p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-500 mb-1">
                    Invoice Number
                  </label>
                  <p className="text-gray-900 font-medium">
                    {extractedData.invoice_number || 'Not found'}
                  </p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-500 mb-1">
                    Invoice Date
                  </label>
                  <p className="text-gray-900">
                    {extractedData.invoice_date
                      ? new Date(extractedData.invoice_date).toLocaleDateString()
                      : 'Not found'}
                  </p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-500 mb-1">Due Date</label>
                  <p className="text-gray-900">
                    {extractedData.due_date
                      ? new Date(extractedData.due_date).toLocaleDateString()
                      : 'Not found'}
                  </p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-500 mb-1">
                    Document Type
                  </label>
                  <p className="text-gray-900">{extractedData.document_type || 'Invoice'}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-500 mb-1">Currency</label>
                  <p className="text-gray-900">{extractedData.currency}</p>
                </div>
              </div>

              {/* Amounts */}
              <div className="bg-gray-50 rounded-lg p-4 mb-6">
                <div className="grid grid-cols-3 gap-4 text-center">
                  <div>
                    <p className="text-sm text-gray-500">Subtotal</p>
                    <p className="text-lg font-semibold text-gray-900">
                      ${extractedData.subtotal?.toLocaleString() || '0.00'}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Tax</p>
                    <p className="text-lg font-semibold text-gray-900">
                      ${extractedData.tax_amount?.toLocaleString() || '0.00'}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Total</p>
                    <p className="text-xl font-bold text-primary-600">
                      ${extractedData.total_amount?.toLocaleString() || '0.00'}
                    </p>
                  </div>
                </div>
              </div>

              {/* Line Items */}
              {extractedData.line_items.length > 0 && (
                <div>
                  <h4 className="font-medium text-gray-900 mb-2">Line Items</h4>
                  <div className="border border-gray-200 rounded-lg overflow-hidden">
                    <table className="w-full text-sm">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="text-left px-3 py-2 font-medium text-gray-600">
                            Description
                          </th>
                          <th className="text-right px-3 py-2 font-medium text-gray-600">Qty</th>
                          <th className="text-right px-3 py-2 font-medium text-gray-600">
                            Unit Price
                          </th>
                          <th className="text-right px-3 py-2 font-medium text-gray-600">Amount</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-100">
                        {extractedData.line_items.map((item, idx) => (
                          <tr key={idx}>
                            <td className="px-3 py-2 text-gray-900">
                              {item.description || 'No description'}
                            </td>
                            <td className="px-3 py-2 text-right text-gray-600">
                              {item.quantity || '-'}
                            </td>
                            <td className="px-3 py-2 text-right text-gray-600">
                              ${item.unit_price?.toFixed(2) || '-'}
                            </td>
                            <td className="px-3 py-2 text-right font-medium text-gray-900">
                              ${item.amount?.toFixed(2) || '-'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
            <div className="px-6 py-4 border-t border-gray-100 flex justify-end space-x-3">
              <button
                onClick={() => {
                  setSelectedItem(null);
                  setExtractedData(null);
                }}
                className="px-4 py-2 border border-gray-300 text-gray-700 font-medium rounded-lg hover:bg-gray-50 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => handleApprove('BILL')}
                className="px-4 py-2 border border-primary-600 text-primary-600 font-medium rounded-lg hover:bg-primary-50 transition-colors"
              >
                Create as Bill
              </button>
              <button
                onClick={() => handleApprove('INVOICE')}
                className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white font-medium rounded-lg transition-colors"
              >
                Create as Invoice
              </button>
            </div>
          </div>
        </div>
      )}
    </DashboardLayout>
  );
}
