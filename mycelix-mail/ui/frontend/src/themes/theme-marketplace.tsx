// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Theme Marketplace
 *
 * Browse, preview, and install community themes.
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Search,
  Download,
  Star,
  Check,
  Palette,
  Sun,
  Moon,
  Monitor,
  Eye,
  Heart,
  Trash2,
} from 'lucide-react';

// ============================================================================
// Types
// ============================================================================

export interface Theme {
  id: string;
  name: string;
  author: string;
  description: string;
  version: string;
  type: 'light' | 'dark' | 'auto';
  colors: ThemeColors;
  preview: string; // URL to preview image
  downloads: number;
  rating: number;
  ratingCount: number;
  tags: string[];
  createdAt: string;
  updatedAt: string;
  isInstalled?: boolean;
  isActive?: boolean;
}

export interface ThemeColors {
  // Core colors
  primary: string;
  primaryForeground: string;
  secondary: string;
  secondaryForeground: string;
  accent: string;
  accentForeground: string;

  // Background colors
  background: string;
  foreground: string;
  card: string;
  cardForeground: string;
  popover: string;
  popoverForeground: string;

  // UI colors
  muted: string;
  mutedForeground: string;
  border: string;
  input: string;
  ring: string;

  // Semantic colors
  destructive: string;
  destructiveForeground: string;
  success: string;
  successForeground: string;
  warning: string;
  warningForeground: string;
  info: string;
  infoForeground: string;

  // Trust levels
  trustVeryHigh: string;
  trustHigh: string;
  trustMedium: string;
  trustLow: string;
  trustVeryLow: string;
  trustUnknown: string;

  // Sidebar
  sidebarBackground: string;
  sidebarForeground: string;
  sidebarAccent: string;

  // Email list
  emailUnread: string;
  emailStarred: string;
  emailSelected: string;

  // Custom
  radius: string;
  fontFamily?: string;
}

// ============================================================================
// Built-in Themes
// ============================================================================

const builtInThemes: Theme[] = [
  {
    id: 'default-light',
    name: 'Mycelix Light',
    author: 'Mycelix Team',
    description: 'The default light theme with clean, professional aesthetics',
    version: '1.0.0',
    type: 'light',
    colors: {
      primary: '#6366f1',
      primaryForeground: '#ffffff',
      secondary: '#f1f5f9',
      secondaryForeground: '#1e293b',
      accent: '#f1f5f9',
      accentForeground: '#1e293b',
      background: '#ffffff',
      foreground: '#0f172a',
      card: '#ffffff',
      cardForeground: '#0f172a',
      popover: '#ffffff',
      popoverForeground: '#0f172a',
      muted: '#f1f5f9',
      mutedForeground: '#64748b',
      border: '#e2e8f0',
      input: '#e2e8f0',
      ring: '#6366f1',
      destructive: '#ef4444',
      destructiveForeground: '#ffffff',
      success: '#22c55e',
      successForeground: '#ffffff',
      warning: '#f59e0b',
      warningForeground: '#ffffff',
      info: '#3b82f6',
      infoForeground: '#ffffff',
      trustVeryHigh: '#22c55e',
      trustHigh: '#84cc16',
      trustMedium: '#eab308',
      trustLow: '#f97316',
      trustVeryLow: '#ef4444',
      trustUnknown: '#94a3b8',
      sidebarBackground: '#f8fafc',
      sidebarForeground: '#0f172a',
      sidebarAccent: '#e2e8f0',
      emailUnread: '#1e293b',
      emailStarred: '#fbbf24',
      emailSelected: '#eff6ff',
      radius: '0.5rem',
    },
    preview: '/themes/previews/default-light.png',
    downloads: 0,
    rating: 5,
    ratingCount: 0,
    tags: ['official', 'light', 'professional'],
    createdAt: '2024-01-01',
    updatedAt: '2024-01-01',
    isInstalled: true,
  },
  {
    id: 'default-dark',
    name: 'Mycelix Dark',
    author: 'Mycelix Team',
    description: 'The default dark theme for comfortable night reading',
    version: '1.0.0',
    type: 'dark',
    colors: {
      primary: '#818cf8',
      primaryForeground: '#1e1b4b',
      secondary: '#1e293b',
      secondaryForeground: '#f1f5f9',
      accent: '#1e293b',
      accentForeground: '#f1f5f9',
      background: '#0f172a',
      foreground: '#f1f5f9',
      card: '#1e293b',
      cardForeground: '#f1f5f9',
      popover: '#1e293b',
      popoverForeground: '#f1f5f9',
      muted: '#334155',
      mutedForeground: '#94a3b8',
      border: '#334155',
      input: '#334155',
      ring: '#818cf8',
      destructive: '#f87171',
      destructiveForeground: '#1e1b4b',
      success: '#4ade80',
      successForeground: '#1e1b4b',
      warning: '#fbbf24',
      warningForeground: '#1e1b4b',
      info: '#60a5fa',
      infoForeground: '#1e1b4b',
      trustVeryHigh: '#4ade80',
      trustHigh: '#a3e635',
      trustMedium: '#facc15',
      trustLow: '#fb923c',
      trustVeryLow: '#f87171',
      trustUnknown: '#64748b',
      sidebarBackground: '#0c1222',
      sidebarForeground: '#f1f5f9',
      sidebarAccent: '#1e293b',
      emailUnread: '#f1f5f9',
      emailStarred: '#fbbf24',
      emailSelected: '#1e3a5f',
      radius: '0.5rem',
    },
    preview: '/themes/previews/default-dark.png',
    downloads: 0,
    rating: 5,
    ratingCount: 0,
    tags: ['official', 'dark', 'professional'],
    createdAt: '2024-01-01',
    updatedAt: '2024-01-01',
    isInstalled: true,
  },
  {
    id: 'nord',
    name: 'Nord',
    author: 'Community',
    description: 'An arctic, north-bluish clean and elegant theme',
    version: '1.0.0',
    type: 'dark',
    colors: {
      primary: '#88c0d0',
      primaryForeground: '#2e3440',
      secondary: '#3b4252',
      secondaryForeground: '#eceff4',
      accent: '#81a1c1',
      accentForeground: '#2e3440',
      background: '#2e3440',
      foreground: '#eceff4',
      card: '#3b4252',
      cardForeground: '#eceff4',
      popover: '#3b4252',
      popoverForeground: '#eceff4',
      muted: '#434c5e',
      mutedForeground: '#d8dee9',
      border: '#4c566a',
      input: '#4c566a',
      ring: '#88c0d0',
      destructive: '#bf616a',
      destructiveForeground: '#eceff4',
      success: '#a3be8c',
      successForeground: '#2e3440',
      warning: '#ebcb8b',
      warningForeground: '#2e3440',
      info: '#81a1c1',
      infoForeground: '#2e3440',
      trustVeryHigh: '#a3be8c',
      trustHigh: '#8fbcbb',
      trustMedium: '#ebcb8b',
      trustLow: '#d08770',
      trustVeryLow: '#bf616a',
      trustUnknown: '#4c566a',
      sidebarBackground: '#2e3440',
      sidebarForeground: '#eceff4',
      sidebarAccent: '#3b4252',
      emailUnread: '#eceff4',
      emailStarred: '#ebcb8b',
      emailSelected: '#434c5e',
      radius: '0.375rem',
    },
    preview: '/themes/previews/nord.png',
    downloads: 12500,
    rating: 4.8,
    ratingCount: 156,
    tags: ['popular', 'dark', 'nordic'],
    createdAt: '2024-02-15',
    updatedAt: '2024-03-10',
  },
  {
    id: 'dracula',
    name: 'Dracula',
    author: 'Community',
    description: 'A dark theme for vampires',
    version: '1.0.0',
    type: 'dark',
    colors: {
      primary: '#bd93f9',
      primaryForeground: '#282a36',
      secondary: '#44475a',
      secondaryForeground: '#f8f8f2',
      accent: '#ff79c6',
      accentForeground: '#282a36',
      background: '#282a36',
      foreground: '#f8f8f2',
      card: '#44475a',
      cardForeground: '#f8f8f2',
      popover: '#44475a',
      popoverForeground: '#f8f8f2',
      muted: '#6272a4',
      mutedForeground: '#f8f8f2',
      border: '#6272a4',
      input: '#6272a4',
      ring: '#bd93f9',
      destructive: '#ff5555',
      destructiveForeground: '#f8f8f2',
      success: '#50fa7b',
      successForeground: '#282a36',
      warning: '#ffb86c',
      warningForeground: '#282a36',
      info: '#8be9fd',
      infoForeground: '#282a36',
      trustVeryHigh: '#50fa7b',
      trustHigh: '#8be9fd',
      trustMedium: '#f1fa8c',
      trustLow: '#ffb86c',
      trustVeryLow: '#ff5555',
      trustUnknown: '#6272a4',
      sidebarBackground: '#21222c',
      sidebarForeground: '#f8f8f2',
      sidebarAccent: '#44475a',
      emailUnread: '#f8f8f2',
      emailStarred: '#f1fa8c',
      emailSelected: '#44475a',
      radius: '0.5rem',
    },
    preview: '/themes/previews/dracula.png',
    downloads: 8900,
    rating: 4.7,
    ratingCount: 89,
    tags: ['popular', 'dark', 'vibrant'],
    createdAt: '2024-01-20',
    updatedAt: '2024-02-28',
  },
  {
    id: 'solarized-light',
    name: 'Solarized Light',
    author: 'Community',
    description: 'A precision color scheme designed for readability',
    version: '1.0.0',
    type: 'light',
    colors: {
      primary: '#268bd2',
      primaryForeground: '#fdf6e3',
      secondary: '#eee8d5',
      secondaryForeground: '#657b83',
      accent: '#2aa198',
      accentForeground: '#fdf6e3',
      background: '#fdf6e3',
      foreground: '#657b83',
      card: '#eee8d5',
      cardForeground: '#657b83',
      popover: '#eee8d5',
      popoverForeground: '#657b83',
      muted: '#eee8d5',
      mutedForeground: '#93a1a1',
      border: '#93a1a1',
      input: '#93a1a1',
      ring: '#268bd2',
      destructive: '#dc322f',
      destructiveForeground: '#fdf6e3',
      success: '#859900',
      successForeground: '#fdf6e3',
      warning: '#b58900',
      warningForeground: '#fdf6e3',
      info: '#268bd2',
      infoForeground: '#fdf6e3',
      trustVeryHigh: '#859900',
      trustHigh: '#2aa198',
      trustMedium: '#b58900',
      trustLow: '#cb4b16',
      trustVeryLow: '#dc322f',
      trustUnknown: '#93a1a1',
      sidebarBackground: '#eee8d5',
      sidebarForeground: '#657b83',
      sidebarAccent: '#fdf6e3',
      emailUnread: '#073642',
      emailStarred: '#b58900',
      emailSelected: '#fdf6e3',
      radius: '0.25rem',
    },
    preview: '/themes/previews/solarized-light.png',
    downloads: 5600,
    rating: 4.5,
    ratingCount: 67,
    tags: ['light', 'readable', 'classic'],
    createdAt: '2024-02-01',
    updatedAt: '2024-02-15',
  },
  {
    id: 'monokai',
    name: 'Monokai Pro',
    author: 'Community',
    description: 'A sophisticated dark theme inspired by the classic Monokai',
    version: '1.0.0',
    type: 'dark',
    colors: {
      primary: '#a6e22e',
      primaryForeground: '#272822',
      secondary: '#3e3d32',
      secondaryForeground: '#f8f8f2',
      accent: '#f92672',
      accentForeground: '#f8f8f2',
      background: '#272822',
      foreground: '#f8f8f2',
      card: '#3e3d32',
      cardForeground: '#f8f8f2',
      popover: '#3e3d32',
      popoverForeground: '#f8f8f2',
      muted: '#49483e',
      mutedForeground: '#75715e',
      border: '#49483e',
      input: '#49483e',
      ring: '#a6e22e',
      destructive: '#f92672',
      destructiveForeground: '#f8f8f2',
      success: '#a6e22e',
      successForeground: '#272822',
      warning: '#fd971f',
      warningForeground: '#272822',
      info: '#66d9ef',
      infoForeground: '#272822',
      trustVeryHigh: '#a6e22e',
      trustHigh: '#66d9ef',
      trustMedium: '#e6db74',
      trustLow: '#fd971f',
      trustVeryLow: '#f92672',
      trustUnknown: '#75715e',
      sidebarBackground: '#1e1f1c',
      sidebarForeground: '#f8f8f2',
      sidebarAccent: '#3e3d32',
      emailUnread: '#f8f8f2',
      emailStarred: '#e6db74',
      emailSelected: '#49483e',
      radius: '0.375rem',
    },
    preview: '/themes/previews/monokai.png',
    downloads: 7200,
    rating: 4.6,
    ratingCount: 78,
    tags: ['dark', 'code', 'classic'],
    createdAt: '2024-01-25',
    updatedAt: '2024-03-01',
  },
];

// ============================================================================
// Component
// ============================================================================

interface ThemeMarketplaceProps {
  isOpen: boolean;
  onClose: () => void;
  onApplyTheme: (theme: Theme) => void;
  currentThemeId?: string;
}

export const ThemeMarketplace: React.FC<ThemeMarketplaceProps> = ({
  isOpen,
  onClose,
  onApplyTheme,
  currentThemeId,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState<'all' | 'light' | 'dark' | 'installed'>('all');
  const [previewTheme, setPreviewTheme] = useState<Theme | null>(null);
  const [installedThemes, setInstalledThemes] = useState<Set<string>>(
    new Set(builtInThemes.filter(t => t.isInstalled).map(t => t.id))
  );

  const filteredThemes = useMemo(() => {
    return builtInThemes.filter(theme => {
      const matchesSearch =
        searchQuery === '' ||
        theme.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        theme.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
        theme.author.toLowerCase().includes(searchQuery.toLowerCase()) ||
        theme.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));

      const matchesTab =
        activeTab === 'all' ||
        (activeTab === 'installed' && installedThemes.has(theme.id)) ||
        (activeTab === 'light' && theme.type === 'light') ||
        (activeTab === 'dark' && theme.type === 'dark');

      return matchesSearch && matchesTab;
    }).sort((a, b) => b.downloads - a.downloads);
  }, [searchQuery, activeTab, installedThemes]);

  const handleInstall = async (theme: Theme) => {
    // Simulate installation
    setInstalledThemes(prev => new Set([...prev, theme.id]));
  };

  const handleUninstall = async (themeId: string) => {
    if (themeId === 'default-light' || themeId === 'default-dark') {
      // Can't uninstall default themes
      return;
    }
    setInstalledThemes(prev => {
      const next = new Set(prev);
      next.delete(themeId);
      return next;
    });
  };

  const handleApply = (theme: Theme) => {
    onApplyTheme(theme);
    onClose();
  };

  const TypeIcon = ({ type }: { type: 'light' | 'dark' | 'auto' }) => {
    switch (type) {
      case 'light':
        return <Sun className="h-4 w-4" />;
      case 'dark':
        return <Moon className="h-4 w-4" />;
      case 'auto':
        return <Monitor className="h-4 w-4" />;
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Palette className="h-5 w-5" />
            Theme Marketplace
          </DialogTitle>
        </DialogHeader>

        <div className="flex gap-4 h-[600px]">
          {/* Left Panel - Theme List */}
          <div className="w-1/2 flex flex-col">
            <div className="mb-4 space-y-3">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
                <Input
                  placeholder="Search themes..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10"
                />
              </div>

              <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as typeof activeTab)}>
                <TabsList className="w-full">
                  <TabsTrigger value="all" className="flex-1">All</TabsTrigger>
                  <TabsTrigger value="light" className="flex-1">
                    <Sun className="h-3 w-3 mr-1" /> Light
                  </TabsTrigger>
                  <TabsTrigger value="dark" className="flex-1">
                    <Moon className="h-3 w-3 mr-1" /> Dark
                  </TabsTrigger>
                  <TabsTrigger value="installed" className="flex-1">
                    <Check className="h-3 w-3 mr-1" /> Installed
                  </TabsTrigger>
                </TabsList>
              </Tabs>
            </div>

            <ScrollArea className="flex-1">
              <div className="space-y-2 pr-4">
                {filteredThemes.map(theme => {
                  const isInstalled = installedThemes.has(theme.id);
                  const isActive = theme.id === currentThemeId;

                  return (
                    <div
                      key={theme.id}
                      onClick={() => setPreviewTheme(theme)}
                      className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                        previewTheme?.id === theme.id
                          ? 'border-primary bg-primary/5'
                          : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                      }`}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex items-center gap-2">
                          <div
                            className="w-8 h-8 rounded"
                            style={{ backgroundColor: theme.colors.primary }}
                          />
                          <div>
                            <div className="font-medium text-sm flex items-center gap-1">
                              {theme.name}
                              {isActive && (
                                <Badge variant="secondary" className="text-xs">Active</Badge>
                              )}
                            </div>
                            <div className="text-xs text-gray-500">by {theme.author}</div>
                          </div>
                        </div>
                        <TypeIcon type={theme.type} />
                      </div>

                      <p className="text-xs text-gray-500 mt-2 line-clamp-2">
                        {theme.description}
                      </p>

                      <div className="flex items-center justify-between mt-2">
                        <div className="flex items-center gap-3 text-xs text-gray-500">
                          <span className="flex items-center gap-1">
                            <Star className="h-3 w-3 fill-yellow-400 text-yellow-400" />
                            {theme.rating.toFixed(1)}
                          </span>
                          <span className="flex items-center gap-1">
                            <Download className="h-3 w-3" />
                            {theme.downloads.toLocaleString()}
                          </span>
                        </div>
                        {isInstalled && (
                          <Badge variant="outline" className="text-xs">
                            <Check className="h-3 w-3 mr-1" /> Installed
                          </Badge>
                        )}
                      </div>
                    </div>
                  );
                })}

                {filteredThemes.length === 0 && (
                  <div className="text-center py-8 text-gray-500">
                    No themes found matching your search.
                  </div>
                )}
              </div>
            </ScrollArea>
          </div>

          {/* Right Panel - Theme Preview */}
          <div className="w-1/2 flex flex-col border-l pl-4">
            {previewTheme ? (
              <>
                <div className="mb-4">
                  <h3 className="text-lg font-medium">{previewTheme.name}</h3>
                  <p className="text-sm text-gray-500">by {previewTheme.author}</p>
                  <p className="text-sm mt-2">{previewTheme.description}</p>

                  <div className="flex flex-wrap gap-1 mt-2">
                    {previewTheme.tags.map(tag => (
                      <Badge key={tag} variant="secondary" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                </div>

                {/* Color Preview */}
                <div className="mb-4">
                  <h4 className="text-sm font-medium mb-2">Color Palette</h4>
                  <div className="grid grid-cols-6 gap-2">
                    {[
                      { name: 'Primary', color: previewTheme.colors.primary },
                      { name: 'Secondary', color: previewTheme.colors.secondary },
                      { name: 'Accent', color: previewTheme.colors.accent },
                      { name: 'Background', color: previewTheme.colors.background },
                      { name: 'Foreground', color: previewTheme.colors.foreground },
                      { name: 'Muted', color: previewTheme.colors.muted },
                    ].map(({ name, color }) => (
                      <div key={name} className="text-center">
                        <div
                          className="w-full h-8 rounded border"
                          style={{ backgroundColor: color }}
                          title={`${name}: ${color}`}
                        />
                        <span className="text-xs text-gray-500">{name}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Trust Colors */}
                <div className="mb-4">
                  <h4 className="text-sm font-medium mb-2">Trust Indicators</h4>
                  <div className="flex gap-1">
                    {[
                      { level: 'Very High', color: previewTheme.colors.trustVeryHigh },
                      { level: 'High', color: previewTheme.colors.trustHigh },
                      { level: 'Medium', color: previewTheme.colors.trustMedium },
                      { level: 'Low', color: previewTheme.colors.trustLow },
                      { level: 'Very Low', color: previewTheme.colors.trustVeryLow },
                    ].map(({ level, color }) => (
                      <div
                        key={level}
                        className="flex-1 h-6 rounded"
                        style={{ backgroundColor: color }}
                        title={level}
                      />
                    ))}
                  </div>
                </div>

                {/* Mock Preview */}
                <div className="flex-1 rounded-lg overflow-hidden border mb-4">
                  <div
                    className="h-full p-4"
                    style={{
                      backgroundColor: previewTheme.colors.background,
                      color: previewTheme.colors.foreground,
                    }}
                  >
                    <div
                      className="p-3 rounded mb-3"
                      style={{ backgroundColor: previewTheme.colors.card }}
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <div
                          className="w-8 h-8 rounded-full"
                          style={{ backgroundColor: previewTheme.colors.primary }}
                        />
                        <div>
                          <div className="font-medium text-sm">John Doe</div>
                          <div
                            className="text-xs"
                            style={{ color: previewTheme.colors.mutedForeground }}
                          >
                            john@example.com
                          </div>
                        </div>
                        <Badge
                          className="ml-auto text-xs"
                          style={{
                            backgroundColor: previewTheme.colors.trustHigh,
                            color: previewTheme.colors.background,
                          }}
                        >
                          High Trust
                        </Badge>
                      </div>
                      <div className="text-sm font-medium">Re: Project Update</div>
                      <div
                        className="text-xs"
                        style={{ color: previewTheme.colors.mutedForeground }}
                      >
                        Thanks for the update! Looking forward to...
                      </div>
                    </div>

                    <Button
                      className="w-full"
                      style={{
                        backgroundColor: previewTheme.colors.primary,
                        color: previewTheme.colors.primaryForeground,
                      }}
                    >
                      Sample Button
                    </Button>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-2">
                  {installedThemes.has(previewTheme.id) ? (
                    <>
                      <Button
                        onClick={() => handleApply(previewTheme)}
                        className="flex-1"
                        disabled={previewTheme.id === currentThemeId}
                      >
                        {previewTheme.id === currentThemeId ? (
                          <>
                            <Check className="h-4 w-4 mr-2" /> Active
                          </>
                        ) : (
                          <>
                            <Eye className="h-4 w-4 mr-2" /> Apply Theme
                          </>
                        )}
                      </Button>
                      {!previewTheme.id.startsWith('default') && (
                        <Button
                          variant="outline"
                          onClick={() => handleUninstall(previewTheme.id)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      )}
                    </>
                  ) : (
                    <Button
                      onClick={() => handleInstall(previewTheme)}
                      className="flex-1"
                    >
                      <Download className="h-4 w-4 mr-2" /> Install Theme
                    </Button>
                  )}
                </div>
              </>
            ) : (
              <div className="flex-1 flex items-center justify-center text-gray-500">
                <div className="text-center">
                  <Palette className="h-12 w-12 mx-auto mb-2 opacity-50" />
                  <p>Select a theme to preview</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ThemeMarketplace;
