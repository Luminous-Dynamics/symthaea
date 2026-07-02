// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Voice Commands Hook
 *
 * Hands-free control using Web Speech API
 * Supports playback, navigation, search, and DJ commands
 */

import { useState, useCallback, useRef, useEffect } from 'react';

export interface VoiceCommand {
  patterns: string[];
  action: string;
  category: 'playback' | 'navigation' | 'search' | 'dj' | 'volume' | 'playlist' | 'system';
  description: string;
  handler?: (params: CommandParams) => void;
}

export interface CommandParams {
  raw: string;
  command: string;
  args: string[];
  number?: number;
  text?: string;
}

export interface VoiceCommandResult {
  success: boolean;
  command: string;
  action: string;
  transcript: string;
  timestamp: number;
}

interface SpeechRecognitionEvent {
  results: SpeechRecognitionResultList;
  resultIndex: number;
}

interface SpeechRecognitionResultList {
  [index: number]: SpeechRecognitionResult;
  length: number;
}

interface SpeechRecognitionResult {
  [index: number]: SpeechRecognitionAlternative;
  isFinal: boolean;
  length: number;
}

interface SpeechRecognitionAlternative {
  transcript: string;
  confidence: number;
}

// Default command set
const DEFAULT_COMMANDS: VoiceCommand[] = [
  // Playback commands
  { patterns: ['play', 'start', 'resume'], action: 'play', category: 'playback', description: 'Start playback' },
  { patterns: ['pause', 'stop'], action: 'pause', category: 'playback', description: 'Pause playback' },
  { patterns: ['next', 'skip', 'next track', 'next song'], action: 'next', category: 'playback', description: 'Skip to next track' },
  { patterns: ['previous', 'back', 'go back', 'previous track'], action: 'previous', category: 'playback', description: 'Go to previous track' },
  { patterns: ['shuffle', 'shuffle on', 'shuffle mode'], action: 'shuffle', category: 'playback', description: 'Toggle shuffle mode' },
  { patterns: ['repeat', 'loop', 'repeat mode'], action: 'repeat', category: 'playback', description: 'Toggle repeat mode' },
  { patterns: ['restart', 'start over', 'from beginning'], action: 'restart', category: 'playback', description: 'Restart current track' },

  // Volume commands
  { patterns: ['volume up', 'louder', 'turn up', 'increase volume'], action: 'volumeUp', category: 'volume', description: 'Increase volume' },
  { patterns: ['volume down', 'quieter', 'turn down', 'decrease volume'], action: 'volumeDown', category: 'volume', description: 'Decrease volume' },
  { patterns: ['mute', 'mute audio', 'silence'], action: 'mute', category: 'volume', description: 'Mute audio' },
  { patterns: ['unmute', 'unmute audio'], action: 'unmute', category: 'volume', description: 'Unmute audio' },
  { patterns: ['volume {number}', 'set volume to {number}', 'set volume {number}'], action: 'setVolume', category: 'volume', description: 'Set volume to specific level' },

  // Navigation commands
  { patterns: ['go home', 'home', 'go to home'], action: 'goHome', category: 'navigation', description: 'Navigate to home' },
  { patterns: ['library', 'my library', 'go to library'], action: 'goLibrary', category: 'navigation', description: 'Navigate to library' },
  { patterns: ['explore', 'discover', 'go to explore'], action: 'goExplore', category: 'navigation', description: 'Navigate to explore' },
  { patterns: ['queue', 'show queue', 'up next'], action: 'showQueue', category: 'navigation', description: 'Show play queue' },
  { patterns: ['now playing', 'current track', 'what\'s playing'], action: 'nowPlaying', category: 'navigation', description: 'Show now playing' },

  // Search commands
  { patterns: ['search for {text}', 'find {text}', 'look for {text}'], action: 'search', category: 'search', description: 'Search for music' },
  { patterns: ['play {text}', 'play song {text}', 'play artist {text}'], action: 'playSearch', category: 'search', description: 'Search and play' },

  // Playlist commands
  { patterns: ['add to playlist', 'save to playlist', 'add this song'], action: 'addToPlaylist', category: 'playlist', description: 'Add current track to playlist' },
  { patterns: ['like', 'love', 'favorite', 'add to favorites'], action: 'like', category: 'playlist', description: 'Like current track' },
  { patterns: ['unlike', 'remove from favorites'], action: 'unlike', category: 'playlist', description: 'Unlike current track' },
  { patterns: ['create playlist {text}', 'new playlist {text}'], action: 'createPlaylist', category: 'playlist', description: 'Create new playlist' },

  // DJ commands
  { patterns: ['crossfade', 'fade over', 'transition'], action: 'crossfade', category: 'dj', description: 'Crossfade to next track' },
  { patterns: ['loop', 'start loop', 'loop this'], action: 'startLoop', category: 'dj', description: 'Start loop' },
  { patterns: ['end loop', 'stop loop', 'exit loop'], action: 'endLoop', category: 'dj', description: 'End loop' },
  { patterns: ['drop', 'cue point', 'mark'], action: 'setCue', category: 'dj', description: 'Set cue point' },
  { patterns: ['sync', 'beat sync', 'sync tempo'], action: 'sync', category: 'dj', description: 'Sync beat' },

  // System commands
  { patterns: ['help', 'show commands', 'what can you do'], action: 'help', category: 'system', description: 'Show available commands' },
  { patterns: ['stop listening', 'voice off', 'disable voice'], action: 'stopListening', category: 'system', description: 'Stop voice recognition' },
];

export function useVoiceCommands(customCommands?: VoiceCommand[]) {
  const [isListening, setIsListening] = useState(false);
  const [isSupported, setIsSupported] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [interimTranscript, setInterimTranscript] = useState('');
  const [lastResult, setLastResult] = useState<VoiceCommandResult | null>(null);
  const [history, setHistory] = useState<VoiceCommandResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [wakeWord, setWakeWord] = useState<string | null>('hey mycelix');
  const [isAwake, setIsAwake] = useState(false);
  const [confidence, setConfidence] = useState(0);

  const recognitionRef = useRef<any>(null);
  const commandsRef = useRef<VoiceCommand[]>([]);
  const handlersRef = useRef<Map<string, (params: CommandParams) => void>>(new Map());
  const awakeTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Merge default and custom commands
  useEffect(() => {
    commandsRef.current = [...DEFAULT_COMMANDS, ...(customCommands || [])];
  }, [customCommands]);

  // Check for browser support
  useEffect(() => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    setIsSupported(!!SpeechRecognition);

    if (SpeechRecognition) {
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;
      recognitionRef.current.lang = 'en-US';
      recognitionRef.current.maxAlternatives = 3;
    }
  }, []);

  // Parse command from transcript
  const parseCommand = useCallback((text: string): { command: VoiceCommand; params: CommandParams } | null => {
    const normalizedText = text.toLowerCase().trim();

    for (const command of commandsRef.current) {
      for (const pattern of command.patterns) {
        // Check for variable patterns ({text}, {number})
        if (pattern.includes('{')) {
          const regex = pattern
            .replace(/\{text\}/g, '(.+)')
            .replace(/\{number\}/g, '(\\d+)');

          const match = normalizedText.match(new RegExp(`^${regex}$`, 'i'));
          if (match) {
            const args = match.slice(1);
            const params: CommandParams = {
              raw: text,
              command: pattern,
              args,
              number: pattern.includes('{number}') ? parseInt(args[0], 10) : undefined,
              text: pattern.includes('{text}') ? args[0] : undefined,
            };
            return { command, params };
          }
        } else if (normalizedText === pattern || normalizedText.startsWith(pattern + ' ')) {
          return {
            command,
            params: {
              raw: text,
              command: pattern,
              args: normalizedText.replace(pattern, '').trim().split(' ').filter(Boolean),
            },
          };
        }
      }
    }

    return null;
  }, []);

  // Handle recognition result
  const handleResult = useCallback((event: SpeechRecognitionEvent) => {
    let interim = '';
    let final = '';

    for (let i = event.resultIndex; i < event.results.length; i++) {
      const result = event.results[i];
      const transcript = result[0].transcript;
      const conf = result[0].confidence;

      if (result.isFinal) {
        final += transcript;
        setConfidence(conf);
      } else {
        interim += transcript;
      }
    }

    setInterimTranscript(interim);

    if (final) {
      setTranscript(final);

      // Check for wake word if enabled
      if (wakeWord && !isAwake) {
        if (final.toLowerCase().includes(wakeWord.toLowerCase())) {
          setIsAwake(true);
          // Auto-sleep after 30 seconds of inactivity
          if (awakeTimeoutRef.current) {
            clearTimeout(awakeTimeoutRef.current);
          }
          awakeTimeoutRef.current = setTimeout(() => {
            setIsAwake(false);
          }, 30000);
          return;
        }
        return;
      }

      // Reset awake timeout on activity
      if (wakeWord && isAwake && awakeTimeoutRef.current) {
        clearTimeout(awakeTimeoutRef.current);
        awakeTimeoutRef.current = setTimeout(() => {
          setIsAwake(false);
        }, 30000);
      }

      // Parse and execute command
      const parsed = parseCommand(final);
      if (parsed) {
        const { command, params } = parsed;

        // Execute handler if registered
        const handler = handlersRef.current.get(command.action) || command.handler;
        if (handler) {
          handler(params);
        }

        const result: VoiceCommandResult = {
          success: true,
          command: command.patterns[0],
          action: command.action,
          transcript: final,
          timestamp: Date.now(),
        };

        setLastResult(result);
        setHistory(prev => [...prev.slice(-49), result]);

        // Handle system commands
        if (command.action === 'stopListening') {
          stopListening();
        }
      } else {
        setLastResult({
          success: false,
          command: '',
          action: '',
          transcript: final,
          timestamp: Date.now(),
        });
      }
    }
  }, [wakeWord, isAwake, parseCommand]);

  // Start listening
  const startListening = useCallback(() => {
    if (!recognitionRef.current) {
      setError('Speech recognition not supported');
      return;
    }

    try {
      recognitionRef.current.onresult = handleResult;

      recognitionRef.current.onerror = (event: any) => {
        if (event.error === 'no-speech') {
          // Ignore no-speech errors
          return;
        }
        setError(event.error);
        setIsListening(false);
      };

      recognitionRef.current.onend = () => {
        // Restart if still meant to be listening
        if (isListening) {
          try {
            recognitionRef.current.start();
          } catch (e) {
            // Already started
          }
        }
      };

      recognitionRef.current.start();
      setIsListening(true);
      setError(null);
    } catch (e) {
      setError('Failed to start speech recognition');
    }
  }, [handleResult, isListening]);

  // Stop listening
  const stopListening = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
    setIsListening(false);
    setIsAwake(false);
    setInterimTranscript('');
  }, []);

  // Toggle listening
  const toggleListening = useCallback(() => {
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  }, [isListening, startListening, stopListening]);

  // Register command handler
  const registerHandler = useCallback((action: string, handler: (params: CommandParams) => void) => {
    handlersRef.current.set(action, handler);
    return () => {
      handlersRef.current.delete(action);
    };
  }, []);

  // Register multiple handlers
  const registerHandlers = useCallback((handlers: Record<string, (params: CommandParams) => void>) => {
    Object.entries(handlers).forEach(([action, handler]) => {
      handlersRef.current.set(action, handler);
    });
    return () => {
      Object.keys(handlers).forEach(action => {
        handlersRef.current.delete(action);
      });
    };
  }, []);

  // Get commands by category
  const getCommandsByCategory = useCallback((category: VoiceCommand['category']) => {
    return commandsRef.current.filter(cmd => cmd.category === category);
  }, []);

  // Get all commands
  const getAllCommands = useCallback(() => {
    return commandsRef.current;
  }, []);

  // Speak text (TTS)
  const speak = useCallback((text: string, options?: SpeechSynthesisUtterance) => {
    if (!('speechSynthesis' in window)) return;

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = options?.rate ?? 1;
    utterance.pitch = options?.pitch ?? 1;
    utterance.volume = options?.volume ?? 1;

    // Find a good voice
    const voices = speechSynthesis.getVoices();
    const preferredVoice = voices.find(v =>
      v.lang.startsWith('en') && (v.name.includes('Google') || v.name.includes('Samantha'))
    );
    if (preferredVoice) {
      utterance.voice = preferredVoice;
    }

    speechSynthesis.speak(utterance);
  }, []);

  // Cancel speech
  const cancelSpeak = useCallback(() => {
    if ('speechSynthesis' in window) {
      speechSynthesis.cancel();
    }
  }, []);

  // Cleanup
  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      if (awakeTimeoutRef.current) {
        clearTimeout(awakeTimeoutRef.current);
      }
    };
  }, []);

  return {
    // State
    isListening,
    isSupported,
    isAwake,
    transcript,
    interimTranscript,
    lastResult,
    history,
    error,
    confidence,
    wakeWord,

    // Controls
    startListening,
    stopListening,
    toggleListening,
    setWakeWord,

    // Handlers
    registerHandler,
    registerHandlers,

    // Commands
    getCommandsByCategory,
    getAllCommands,

    // TTS
    speak,
    cancelSpeak,
  };
}
