/**
 * Settings Components Index
 *
 * Exports for settings-related components:
 * - Epistemic settings panel with trust, AI, and privacy configuration
 */

export {
  default as EpistemicSettings,
  useEpistemicSettings,
  SettingToggle,
  SettingSlider,
  SettingSelect,
} from './EpistemicSettings';

export type {
  EpistemicSettingsState,
  TrustSettings,
  QuarantineSettings,
  AISettings,
  NotificationSettings,
  PrivacySettings,
  DisplaySettings,
} from './EpistemicSettings';

export {
  VoiceAccessibilitySettings,
  default as VoiceAccessibilitySettingsDefault,
} from './VoiceAccessibilitySettings';
