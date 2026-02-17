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
