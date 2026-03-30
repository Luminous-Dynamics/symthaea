// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useEffect, useState } from 'react';
import DashboardLayout from '@/components/DashboardLayout';
import { api, Currency, CurrencyConfig, ExchangeRate } from '@/lib/api';
import {
  DollarSign,
  Euro,
  RefreshCw,
  Plus,
  Check,
  ArrowRight,
  Settings,
  TrendingUp,
  Calendar,
} from 'lucide-react';

export default function CurrenciesPage() {
  const [currencies, setCurrencies] = useState<Currency[]>([]);
  const [config, setConfig] = useState<CurrencyConfig | null>(null);
  const [rates, setRates] = useState<ExchangeRate[]>([]);
  const [loading, setLoading] = useState(true);
  const [showRateModal, setShowRateModal] = useState(false);
  const [newRate, setNewRate] = useState({
    from: '',
    to: '',
    rate: '',
    date: new Date().toISOString().split('T')[0],
  });

  // Conversion calculator
  const [convertAmount, setConvertAmount] = useState('100');
  const [convertFrom, setConvertFrom] = useState('USD');
  const [convertTo, setConvertTo] = useState('EUR');
  const [conversionResult, setConversionResult] = useState<number | null>(null);
  const [converting, setConverting] = useState(false);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const [currencyList, currencyConfig, rateList] = await Promise.all([
        api.getCurrencies(),
        api.getCurrencyConfig().catch(() => null),
        api.getExchangeRates(
          new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
          new Date().toISOString().split('T')[0]
        ).catch(() => []),
      ]);
      setCurrencies(currencyList);
      setConfig(currencyConfig);
      setRates(rateList);
    } catch (err) {
      console.error('Failed to load currency data', err);
    } finally {
      setLoading(false);
    }
  };

  const handleAddRate = async () => {
    try {
      await api.setExchangeRate(newRate.from, newRate.to, parseFloat(newRate.rate), newRate.date);
      setShowRateModal(false);
      setNewRate({ from: '', to: '', rate: '', date: new Date().toISOString().split('T')[0] });
      loadData();
    } catch (err) {
      console.error('Failed to add rate', err);
    }
  };

  const handleConvert = async () => {
    try {
      setConverting(true);
      const result = await api.convertAmount(parseFloat(convertAmount), convertFrom, convertTo);
      setConversionResult(result.amount);
    } catch (err) {
      console.error('Conversion failed', err);
    } finally {
      setConverting(false);
    }
  };

  const currencyIcon = (code: string) => {
    switch (code) {
      case 'USD':
        return <DollarSign className="w-5 h-5" />;
      case 'EUR':
        return <Euro className="w-5 h-5" />;
      default:
        return <DollarSign className="w-5 h-5" />;
    }
  };

  return (
    <DashboardLayout
      title="Multi-Currency Settings"
      subtitle="Configure currencies and exchange rates"
    >
      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
          <p className="text-sm text-gray-500">Base Currency</p>
          <p className="text-2xl font-bold text-gray-900">{config?.base_currency || 'USD'}</p>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
          <p className="text-sm text-gray-500">Enabled Currencies</p>
          <p className="text-2xl font-bold text-primary-600">
            {config?.enabled_currencies?.length || currencies.length}
          </p>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
          <p className="text-sm text-gray-500">Exchange Rates</p>
          <p className="text-2xl font-bold text-green-600">{rates.length}</p>
        </div>
        <div className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
          <p className="text-sm text-gray-500">Last Updated</p>
          <p className="text-lg font-medium text-gray-700">
            {config?.last_rate_update
              ? new Date(config.last_rate_update).toLocaleDateString()
              : 'Never'}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Currency Converter */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <TrendingUp className="w-5 h-5 mr-2 text-primary-600" />
            Currency Converter
          </h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Amount</label>
              <input
                type="number"
                value={convertAmount}
                onChange={(e) => setConvertAmount(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
            </div>
            <div className="grid grid-cols-5 gap-2 items-center">
              <div className="col-span-2">
                <label className="block text-sm font-medium text-gray-700 mb-1">From</label>
                <select
                  value={convertFrom}
                  onChange={(e) => setConvertFrom(e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                >
                  {currencies.map((c) => (
                    <option key={c.code} value={c.code}>
                      {c.code} - {c.name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex justify-center pt-6">
                <ArrowRight className="w-5 h-5 text-gray-400" />
              </div>
              <div className="col-span-2">
                <label className="block text-sm font-medium text-gray-700 mb-1">To</label>
                <select
                  value={convertTo}
                  onChange={(e) => setConvertTo(e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                >
                  {currencies.map((c) => (
                    <option key={c.code} value={c.code}>
                      {c.code} - {c.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <button
              onClick={handleConvert}
              disabled={converting}
              className="w-full flex items-center justify-center px-4 py-2 bg-primary-600 hover:bg-primary-700 disabled:bg-primary-400 text-white font-medium rounded-lg transition-colors"
            >
              {converting ? (
                <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
              ) : (
                <TrendingUp className="w-5 h-5 mr-2" />
              )}
              Convert
            </button>
            {conversionResult !== null && (
              <div className="mt-4 p-4 bg-primary-50 rounded-lg">
                <p className="text-sm text-primary-700">Result</p>
                <p className="text-2xl font-bold text-primary-900">
                  {currencies.find((c) => c.code === convertTo)?.symbol}
                  {conversionResult.toLocaleString(undefined, {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  })}
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Recent Exchange Rates */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900 flex items-center">
              <Calendar className="w-5 h-5 mr-2 text-primary-600" />
              Exchange Rates
            </h3>
            <button
              onClick={() => setShowRateModal(true)}
              className="flex items-center px-3 py-1.5 text-sm bg-primary-600 hover:bg-primary-700 text-white font-medium rounded-lg transition-colors"
            >
              <Plus className="w-4 h-4 mr-1" />
              Add Rate
            </button>
          </div>
          {loading ? (
            <div className="flex items-center justify-center h-32">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
            </div>
          ) : rates.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <DollarSign className="w-12 h-12 mx-auto mb-2 text-gray-300" />
              <p>No exchange rates configured</p>
            </div>
          ) : (
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {rates.slice(0, 10).map((rate) => (
                <div
                  key={rate.id}
                  className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                >
                  <div className="flex items-center">
                    <span className="font-medium text-gray-900">{rate.from_currency}</span>
                    <ArrowRight className="w-4 h-4 mx-2 text-gray-400" />
                    <span className="font-medium text-gray-900">{rate.to_currency}</span>
                  </div>
                  <div className="text-right">
                    <p className="font-semibold text-gray-900">{rate.rate.toFixed(4)}</p>
                    <p className="text-xs text-gray-500">
                      {new Date(rate.rate_date).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Available Currencies */}
      <div className="mt-6 bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-100 flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900">Available Currencies</h3>
          <button
            onClick={loadData}
            className="flex items-center text-sm text-gray-600 hover:text-gray-900"
          >
            <RefreshCw className="w-4 h-4 mr-1" />
            Refresh
          </button>
        </div>
        {loading ? (
          <div className="flex items-center justify-center h-32">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 divide-y sm:divide-y-0 sm:divide-x divide-gray-100">
            {currencies.map((currency) => {
              const isEnabled = config?.enabled_currencies?.includes(currency.code) ?? true;
              const isBase = config?.base_currency === currency.code;

              return (
                <div key={currency.code} className="p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <div className="w-10 h-10 bg-primary-50 rounded-full flex items-center justify-center text-primary-600 mr-3">
                        {currencyIcon(currency.code)}
                      </div>
                      <div>
                        <p className="font-medium text-gray-900">
                          {currency.code}
                          {isBase && (
                            <span className="ml-2 text-xs bg-primary-100 text-primary-700 px-2 py-0.5 rounded-full">
                              Base
                            </span>
                          )}
                        </p>
                        <p className="text-sm text-gray-500">{currency.name}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <span className="text-lg text-gray-700">{currency.symbol}</span>
                      {isEnabled && (
                        <Check className="w-4 h-4 text-green-500 ml-auto mt-1" />
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Add Rate Modal */}
      {showRateModal && (
        <div className="fixed inset-0 bg-gray-900/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-xl p-6 w-full max-w-md mx-4">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Add Exchange Rate</h3>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">From</label>
                  <select
                    value={newRate.from}
                    onChange={(e) => setNewRate({ ...newRate, from: e.target.value })}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                  >
                    <option value="">Select...</option>
                    {currencies.map((c) => (
                      <option key={c.code} value={c.code}>
                        {c.code}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">To</label>
                  <select
                    value={newRate.to}
                    onChange={(e) => setNewRate({ ...newRate, to: e.target.value })}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                  >
                    <option value="">Select...</option>
                    {currencies.map((c) => (
                      <option key={c.code} value={c.code}>
                        {c.code}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Rate</label>
                <input
                  type="number"
                  step="0.0001"
                  value={newRate.rate}
                  onChange={(e) => setNewRate({ ...newRate, rate: e.target.value })}
                  placeholder="1.0850"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Date</label>
                <input
                  type="date"
                  value={newRate.date}
                  onChange={(e) => setNewRate({ ...newRate, date: e.target.value })}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                />
              </div>
              <div className="flex space-x-3 pt-2">
                <button
                  onClick={handleAddRate}
                  disabled={!newRate.from || !newRate.to || !newRate.rate}
                  className="flex-1 flex items-center justify-center px-4 py-2 bg-primary-600 hover:bg-primary-700 disabled:bg-gray-300 text-white font-medium rounded-lg transition-colors"
                >
                  <Plus className="w-5 h-5 mr-2" />
                  Add Rate
                </button>
                <button
                  onClick={() => setShowRateModal(false)}
                  className="px-4 py-2 border border-gray-300 text-gray-700 font-medium rounded-lg hover:bg-gray-50 transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </DashboardLayout>
  );
}
