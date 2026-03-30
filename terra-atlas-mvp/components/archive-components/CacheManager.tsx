// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client'

import React, { useState, useEffect } from 'react'
import { 
  FiWifi, FiWifiOff, FiDatabase, FiTrash2, 
  FiDownload, FiRefreshCw, FiHardDrive 
} from 'react-icons/fi'
import { useOfflineStatus, useCacheStats } from '../hooks/useCache'
import { cache } from '../lib/cache'

export default function CacheManager() {
  const { isOnline, status } = useOfflineStatus()
  const { stats, size, clearAll, clearStorage, refresh } = useCacheStats()
  const [showDetails, setShowDetails] = useState(false)
  const [syncing, setSyncing] = useState(false)

  // Register service worker
  useEffect(() => {
    if (typeof navigator !== 'undefined' && 'serviceWorker' in navigator) {
      navigator.serviceWorker.register('/sw.js')
        .then(registration => {
          console.log('Service Worker registered:', registration)
        })
        .catch(error => {
          console.error('Service Worker registration failed:', error)
        })
    }
  }, [])

  const handleSync = async () => {
    setSyncing(true)
    
    try {
      // Trigger background sync
      if ('sync' in navigator && 'serviceWorker' in navigator) {
        const registration = await navigator.serviceWorker.ready
        await registration.sync.register('sync-all-data')
      }
      
      // Manual sync for older browsers
      const response = await fetch('/api/user/sync', {
        method: 'POST'
      })
      
      if (response.ok) {
        console.log('Data synced successfully')
      }
    } catch (error) {
      console.error('Sync failed:', error)
    } finally {
      setSyncing(false)
      refresh()
    }
  }

  const handleClearCache = (storage?: 'memory' | 'localStorage' | 'sessionStorage') => {
    if (window.confirm(`Clear ${storage || 'all'} cache? This will remove cached data.`)) {
      if (storage) {
        clearStorage(storage)
      } else {
        clearAll()
      }
    }
  }

  return (
    <>
      {/* Offline Banner */}
      {!isOnline && (
        <div className="fixed top-16 left-0 right-0 z-40 bg-orange-500/90 backdrop-blur-sm text-white px-4 py-2">
          <div className="max-w-7xl mx-auto flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FiWifiOff className="animate-pulse" />
              <span className="text-sm font-medium">
                You&rsquo;re offline - showing cached data
              </span>
            </div>
            <button
              onClick={() => window.location.reload()}
              className="text-xs px-3 py-1 bg-white/20 rounded hover:bg-white/30 transition-colors"
            >
              Retry
            </button>
          </div>
        </div>
      )}

      {/* Cache Status Button */}
      <button
        onClick={() => setShowDetails(!showDetails)}
        className="fixed bottom-4 right-4 z-30 bg-gray-900/90 backdrop-blur-sm text-white p-3 rounded-full shadow-lg hover:scale-105 transition-transform"
        title="Cache Manager"
      >
        <div className="relative">
          <FiDatabase size={20} />
          {!isOnline && (
            <div className="absolute -top-1 -right-1 w-3 h-3 bg-orange-500 rounded-full animate-pulse" />
          )}
        </div>
      </button>

      {/* Cache Details Panel */}
      {showDetails && (
        <div className="fixed bottom-20 right-4 z-30 w-80 bg-gray-900/95 backdrop-blur-xl rounded-lg shadow-xl border border-white/10 overflow-hidden">
          {/* Header */}
          <div className="px-4 py-3 border-b border-white/10">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${isOnline ? 'bg-green-500' : 'bg-orange-500'} animate-pulse`} />
                <span className="text-white font-medium">
                  {isOnline ? 'Online' : 'Offline Mode'}
                </span>
              </div>
              <button
                onClick={() => setShowDetails(false)}
                className="text-white/60 hover:text-white"
              >
                ×
              </button>
            </div>
          </div>

          {/* Cache Stats */}
          <div className="p-4 space-y-3">
            {/* Total Size */}
            <div className="flex items-center justify-between text-sm">
              <span className="text-white/60">Cache Size</span>
              <span className="text-white font-mono">{size}</span>
            </div>

            {/* Storage Breakdown */}
            <div className="space-y-2">
              {/* Memory Cache */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm">
                  <FiHardDrive size={14} className="text-cyan-400" />
                  <span className="text-white/60">Memory</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-white text-xs">{stats.memory.size} items</span>
                  <button
                    onClick={() => handleClearCache('memory')}
                    className="text-red-400 hover:text-red-300"
                    title="Clear memory cache"
                  >
                    <FiTrash2 size={14} />
                  </button>
                </div>
              </div>

              {/* Local Storage */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm">
                  <FiDatabase size={14} className="text-green-400" />
                  <span className="text-white/60">Local Storage</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-white text-xs">
                    {stats.localStorage.size} items
                  </span>
                  <button
                    onClick={() => handleClearCache('localStorage')}
                    className="text-red-400 hover:text-red-300"
                    title="Clear local storage"
                  >
                    <FiTrash2 size={14} />
                  </button>
                </div>
              </div>

              {/* Session Storage */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm">
                  <FiDownload size={14} className="text-purple-400" />
                  <span className="text-white/60">Session</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-white text-xs">
                    {stats.sessionStorage.size} items
                  </span>
                  <button
                    onClick={() => handleClearCache('sessionStorage')}
                    className="text-red-400 hover:text-red-300"
                    title="Clear session storage"
                  >
                    <FiTrash2 size={14} />
                  </button>
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-2 pt-2">
              <button
                onClick={handleSync}
                disabled={!isOnline || syncing}
                className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-cyan-500/20 text-cyan-400 rounded hover:bg-cyan-500/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <FiRefreshCw size={14} className={syncing ? 'animate-spin' : ''} />
                <span className="text-sm">
                  {syncing ? 'Syncing...' : 'Sync Now'}
                </span>
              </button>
              
              <button
                onClick={() => handleClearCache()}
                className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-red-500/20 text-red-400 rounded hover:bg-red-500/30 transition-colors"
              >
                <FiTrash2 size={14} />
                <span className="text-sm">Clear All</span>
              </button>
            </div>

            {/* Info */}
            <div className="pt-2 border-t border-white/10">
              <p className="text-xs text-white/40">
                {isOnline 
                  ? 'Data is automatically cached for offline use'
                  : 'Changes will sync when connection is restored'
                }
              </p>
            </div>
          </div>
        </div>
      )}
    </>
  )
}