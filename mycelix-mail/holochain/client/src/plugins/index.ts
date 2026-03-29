// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Plugin SDK Module
 *
 * Extensible plugin architecture for Mycelix Mail
 */

export {
  PluginManager,
  type Plugin,
  type PluginManifest,
  type PluginContext,
  type PluginPermission,
  type PluginHook,
  type PluginStorage,
  type PluginAPI,
  type PluginUI,
  type PluginSetting,
  type UIExtension,
} from './PluginSDK';

export default PluginManager;
