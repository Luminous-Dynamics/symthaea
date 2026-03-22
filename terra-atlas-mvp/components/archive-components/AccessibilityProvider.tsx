'use client'

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'

interface AccessibilitySettings {
  screenReaderMode: boolean
  highContrast: boolean
  reduceMotion: boolean
  keyboardNavigation: boolean
  fontSize: 'normal' | 'large' | 'extra-large'
  focusIndicator: 'default' | 'enhanced' | 'high-visibility'
}

interface AccessibilityContextType {
  settings: AccessibilitySettings
  updateSettings: (updates: Partial<AccessibilitySettings>) => void
  announceToScreenReader: (message: string, priority?: 'polite' | 'assertive') => void
  isUsingKeyboard: boolean
}

const AccessibilityContext = createContext<AccessibilityContextType | undefined>(undefined)

export function useAccessibility() {
  const context = useContext(AccessibilityContext)
  if (!context) {
    throw new Error('useAccessibility must be used within AccessibilityProvider')
  }
  return context
}

interface AccessibilityProviderProps {
  children: ReactNode
}

export function AccessibilityProvider({ children }: AccessibilityProviderProps) {
  const [settings, setSettings] = useState<AccessibilitySettings>(() => {
    // Load saved settings from localStorage
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('accessibility-settings')
      if (saved) {
        return JSON.parse(saved)
      }
    }
    
    // Default settings
    return {
      screenReaderMode: false,
      highContrast: false,
      reduceMotion: false,
      keyboardNavigation: true,
      fontSize: 'normal',
      focusIndicator: 'default'
    }
  })
  
  const [isUsingKeyboard, setIsUsingKeyboard] = useState(false)

  // Detect keyboard vs mouse usage
  useEffect(() => {
    const handleMouseDown = () => setIsUsingKeyboard(false)
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Tab' || e.key === 'ArrowUp' || e.key === 'ArrowDown') {
        setIsUsingKeyboard(true)
      }
    }

    window.addEventListener('mousedown', handleMouseDown)
    window.addEventListener('keydown', handleKeyDown)

    return () => {
      window.removeEventListener('mousedown', handleMouseDown)
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [])

  // Detect user preferences from system
  useEffect(() => {
    if (typeof window !== 'undefined') {
      // Check for prefers-reduced-motion
      const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
      if (motionQuery.matches) {
        setSettings(prev => ({ ...prev, reduceMotion: true }))
      }

      // Check for high contrast preference
      const contrastQuery = window.matchMedia('(prefers-contrast: high)')
      if (contrastQuery.matches) {
        setSettings(prev => ({ ...prev, highContrast: true }))
      }
    }
  }, [])

  // Save settings to localStorage
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('accessibility-settings', JSON.stringify(settings))
    }
  }, [settings])

  // Apply settings to document
  useEffect(() => {
    const root = document.documentElement

    // High contrast mode
    if (settings.highContrast) {
      root.classList.add('high-contrast')
    } else {
      root.classList.remove('high-contrast')
    }

    // Reduce motion
    if (settings.reduceMotion) {
      root.classList.add('reduce-motion')
    } else {
      root.classList.remove('reduce-motion')
    }

    // Font size
    root.setAttribute('data-font-size', settings.fontSize)

    // Focus indicator
    root.setAttribute('data-focus-style', settings.focusIndicator)

    // Screen reader mode
    if (settings.screenReaderMode) {
      root.setAttribute('data-screen-reader', 'true')
    } else {
      root.removeAttribute('data-screen-reader')
    }
  }, [settings])

  const updateSettings = (updates: Partial<AccessibilitySettings>) => {
    setSettings(prev => ({ ...prev, ...updates }))
  }

  const announceToScreenReader = (message: string, priority: 'polite' | 'assertive' = 'polite') => {
    const announcement = document.createElement('div')
    announcement.setAttribute('role', 'status')
    announcement.setAttribute('aria-live', priority)
    announcement.setAttribute('aria-atomic', 'true')
    announcement.className = 'sr-only'
    announcement.textContent = message
    
    document.body.appendChild(announcement)
    
    setTimeout(() => {
      document.body.removeChild(announcement)
    }, 1000)
  }

  return (
    <AccessibilityContext.Provider value={{
      settings,
      updateSettings,
      announceToScreenReader,
      isUsingKeyboard
    }}>
      {children}
      
      {/* Accessibility Settings Panel */}
      <AccessibilityPanel />
    </AccessibilityContext.Provider>
  )
}

function AccessibilityPanel() {
  const { settings, updateSettings } = useAccessibility()
  const [isOpen, setIsOpen] = useState(false)

  // Keyboard shortcut to open panel (Alt + A)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.altKey && e.key === 'a') {
        e.preventDefault()
        setIsOpen(!isOpen)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen])

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-4 left-4 z-50 p-2 bg-blue-600 text-white rounded-lg shadow-lg hover:bg-blue-700 transition-colors"
        aria-label="Open accessibility settings (Alt + A)"
        title="Accessibility Settings (Alt + A)"
      >
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      </button>
    )
  }

  return (
    <div 
      className="fixed bottom-4 left-4 z-50 w-80 bg-gray-900 border border-blue-500/30 rounded-lg shadow-2xl"
      role="dialog"
      aria-label="Accessibility Settings"
    >
      <div className="p-4 border-b border-gray-800">
        <div className="flex justify-between items-center">
          <h2 className="text-white font-semibold">Accessibility Settings</h2>
          <button
            onClick={() => setIsOpen(false)}
            className="text-gray-400 hover:text-white transition-colors"
            aria-label="Close accessibility settings"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>

      <div className="p-4 space-y-4 max-h-96 overflow-y-auto">
        {/* Screen Reader Mode */}
        <label className="flex items-center justify-between">
          <span className="text-gray-300 text-sm">Screen Reader Mode</span>
          <input
            type="checkbox"
            checked={settings.screenReaderMode}
            onChange={(e) => updateSettings({ screenReaderMode: e.target.checked })}
            className="w-5 h-5 accent-blue-500"
            aria-label="Toggle screen reader mode"
          />
        </label>

        {/* High Contrast */}
        <label className="flex items-center justify-between">
          <span className="text-gray-300 text-sm">High Contrast</span>
          <input
            type="checkbox"
            checked={settings.highContrast}
            onChange={(e) => updateSettings({ highContrast: e.target.checked })}
            className="w-5 h-5 accent-blue-500"
            aria-label="Toggle high contrast mode"
          />
        </label>

        {/* Reduce Motion */}
        <label className="flex items-center justify-between">
          <span className="text-gray-300 text-sm">Reduce Motion</span>
          <input
            type="checkbox"
            checked={settings.reduceMotion}
            onChange={(e) => updateSettings({ reduceMotion: e.target.checked })}
            className="w-5 h-5 accent-blue-500"
            aria-label="Toggle reduced motion"
          />
        </label>

        {/* Keyboard Navigation */}
        <label className="flex items-center justify-between">
          <span className="text-gray-300 text-sm">Enhanced Keyboard Navigation</span>
          <input
            type="checkbox"
            checked={settings.keyboardNavigation}
            onChange={(e) => updateSettings({ keyboardNavigation: e.target.checked })}
            className="w-5 h-5 accent-blue-500"
            aria-label="Toggle enhanced keyboard navigation"
          />
        </label>

        {/* Font Size */}
        <div>
          <label className="text-gray-300 text-sm block mb-2">Font Size</label>
          <select
            value={settings.fontSize}
            onChange={(e) => updateSettings({ fontSize: e.target.value as any })}
            className="w-full bg-gray-800 text-white rounded px-3 py-2 border border-gray-700 focus:border-blue-500 focus:outline-none"
            aria-label="Select font size"
          >
            <option value="normal">Normal</option>
            <option value="large">Large</option>
            <option value="extra-large">Extra Large</option>
          </select>
        </div>

        {/* Focus Indicator */}
        <div>
          <label className="text-gray-300 text-sm block mb-2">Focus Indicator</label>
          <select
            value={settings.focusIndicator}
            onChange={(e) => updateSettings({ focusIndicator: e.target.value as any })}
            className="w-full bg-gray-800 text-white rounded px-3 py-2 border border-gray-700 focus:border-blue-500 focus:outline-none"
            aria-label="Select focus indicator style"
          >
            <option value="default">Default</option>
            <option value="enhanced">Enhanced</option>
            <option value="high-visibility">High Visibility</option>
          </select>
        </div>

        {/* Keyboard Shortcuts Info */}
        <div className="pt-4 border-t border-gray-800">
          <h3 className="text-gray-300 text-sm font-semibold mb-2">Keyboard Shortcuts</h3>
          <ul className="text-gray-400 text-xs space-y-1">
            <li><kbd className="px-1 bg-gray-800 rounded">Tab</kbd> Navigate forward</li>
            <li><kbd className="px-1 bg-gray-800 rounded">Shift+Tab</kbd> Navigate backward</li>
            <li><kbd className="px-1 bg-gray-800 rounded">Enter</kbd> Activate button/link</li>
            <li><kbd className="px-1 bg-gray-800 rounded">Space</kbd> Toggle checkbox</li>
            <li><kbd className="px-1 bg-gray-800 rounded">Esc</kbd> Close dialogs</li>
            <li><kbd className="px-1 bg-gray-800 rounded">Alt+A</kbd> Accessibility settings</li>
            <li><kbd className="px-1 bg-gray-800 rounded">/</kbd> Open search</li>
            <li><kbd className="px-1 bg-gray-800 rounded">F</kbd> Toggle filters</li>
          </ul>
        </div>
      </div>
    </div>
  )
}