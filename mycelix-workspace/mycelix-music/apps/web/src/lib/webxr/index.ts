// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * WebXR Studio
 *
 * Immersive VR/AR music production:
 * - VR environment and scene management
 * - Hand tracking and controllers
 * - 3D mixer and instruments
 * - Spatial audio integration
 * - Collaborative VR sessions
 * - Gesture recognition
 */

// ==================== Types ====================

export interface XREnvironment {
  id: string;
  name: string;
  skybox: string;
  lighting: XRLighting;
  objects: XRSceneObject[];
  spawnPoint: Vector3;
}

export interface XRLighting {
  ambient: { color: string; intensity: number };
  directional?: { color: string; intensity: number; position: Vector3 };
  point?: Array<{ color: string; intensity: number; position: Vector3; range: number }>;
}

export interface XRSceneObject {
  id: string;
  type: 'mixer' | 'keyboard' | 'drums' | 'turntable' | 'speakers' | 'display' | 'custom';
  position: Vector3;
  rotation: Quaternion;
  scale: Vector3;
  interactable: boolean;
  model?: string;
}

export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

export interface Quaternion {
  x: number;
  y: number;
  z: number;
  w: number;
}

export interface XRControllerState {
  hand: 'left' | 'right';
  position: Vector3;
  rotation: Quaternion;
  trigger: number;
  grip: number;
  thumbstick: { x: number; y: number };
  buttons: {
    a: boolean;
    b: boolean;
    x?: boolean;
    y?: boolean;
  };
  hapticActuator?: GamepadHapticActuator;
}

export interface XRHandState {
  hand: 'left' | 'right';
  joints: Map<XRHandJoint, { position: Vector3; rotation: Quaternion; radius: number }>;
  pinchStrength: number;
  gripStrength: number;
}

export type XRHandJoint =
  | 'wrist'
  | 'thumb-metacarpal' | 'thumb-phalanx-proximal' | 'thumb-phalanx-distal' | 'thumb-tip'
  | 'index-finger-metacarpal' | 'index-finger-phalanx-proximal' | 'index-finger-phalanx-intermediate' | 'index-finger-phalanx-distal' | 'index-finger-tip'
  | 'middle-finger-metacarpal' | 'middle-finger-phalanx-proximal' | 'middle-finger-phalanx-intermediate' | 'middle-finger-phalanx-distal' | 'middle-finger-tip'
  | 'ring-finger-metacarpal' | 'ring-finger-phalanx-proximal' | 'ring-finger-phalanx-intermediate' | 'ring-finger-phalanx-distal' | 'ring-finger-tip'
  | 'pinky-finger-metacarpal' | 'pinky-finger-phalanx-proximal' | 'pinky-finger-phalanx-intermediate' | 'pinky-finger-phalanx-distal' | 'pinky-finger-tip';

export interface XRGesture {
  type: 'point' | 'fist' | 'open' | 'pinch' | 'thumbs-up' | 'peace' | 'grab' | 'release';
  hand: 'left' | 'right';
  confidence: number;
  position: Vector3;
}

export interface VRMixerChannel {
  id: string;
  name: string;
  position: Vector3;
  fader: number;
  pan: number;
  mute: boolean;
  solo: boolean;
  eq: { low: number; mid: number; high: number };
  sends: Array<{ id: string; level: number }>;
}

export interface VRInstrument {
  id: string;
  type: 'keyboard' | 'drums' | 'strings' | 'pads' | 'turntable';
  position: Vector3;
  rotation: Quaternion;
  settings: Record<string, any>;
  onNote?: (note: number, velocity: number, hand: 'left' | 'right') => void;
}

// ==================== WebXR Session Manager ====================

export class XRSessionManager {
  private xrSession: XRSession | null = null;
  private referenceSpace: XRReferenceSpace | null = null;
  private renderer: WebGLRenderer | null = null;
  private gl: WebGLRenderingContext | null = null;
  private isSupported = false;
  private mode: 'immersive-vr' | 'immersive-ar' | 'inline' = 'immersive-vr';

  public onSessionStart?: () => void;
  public onSessionEnd?: () => void;
  public onFrame?: (time: DOMHighResTimeStamp, frame: XRFrame) => void;

  constructor() {
    this.checkSupport();
  }

  private async checkSupport(): Promise<void> {
    if ('xr' in navigator) {
      this.isSupported = await (navigator as any).xr.isSessionSupported('immersive-vr');
    }
  }

  isXRSupported(): boolean {
    return this.isSupported;
  }

  async startSession(
    mode: 'immersive-vr' | 'immersive-ar' = 'immersive-vr',
    canvas: HTMLCanvasElement
  ): Promise<void> {
    if (!this.isSupported) {
      throw new Error('WebXR not supported');
    }

    this.mode = mode;

    const sessionInit: XRSessionInit = {
      requiredFeatures: ['local-floor'],
      optionalFeatures: ['hand-tracking', 'bounded-floor', 'layers'],
    };

    this.xrSession = await (navigator as any).xr.requestSession(mode, sessionInit);

    // Set up WebGL
    this.gl = canvas.getContext('webgl2', { xrCompatible: true }) as WebGLRenderingContext;
    if (!this.gl) {
      this.gl = canvas.getContext('webgl', { xrCompatible: true }) as WebGLRenderingContext;
    }

    await this.xrSession.updateRenderState({
      baseLayer: new XRWebGLLayer(this.xrSession, this.gl),
    });

    this.referenceSpace = await this.xrSession.requestReferenceSpace('local-floor');

    this.xrSession.addEventListener('end', () => {
      this.xrSession = null;
      this.onSessionEnd?.();
    });

    // Start render loop
    this.xrSession.requestAnimationFrame(this.onXRFrame.bind(this));

    this.onSessionStart?.();
  }

  async endSession(): Promise<void> {
    if (this.xrSession) {
      await this.xrSession.end();
      this.xrSession = null;
    }
  }

  private onXRFrame(time: DOMHighResTimeStamp, frame: XRFrame): void {
    if (!this.xrSession) return;

    this.xrSession.requestAnimationFrame(this.onXRFrame.bind(this));
    this.onFrame?.(time, frame);
  }

  getSession(): XRSession | null {
    return this.xrSession;
  }

  getReferenceSpace(): XRReferenceSpace | null {
    return this.referenceSpace;
  }

  getGL(): WebGLRenderingContext | null {
    return this.gl;
  }
}

// WebGL renderer stub for type safety
interface WebGLRenderer {
  context: WebGLRenderingContext;
}

// ==================== Controller Manager ====================

export class XRControllerManager {
  private session: XRSession | null = null;
  private controllers: Map<string, XRControllerState> = new Map();
  private inputSources: Map<string, XRInputSource> = new Map();

  public onControllerConnected?: (controller: XRControllerState) => void;
  public onControllerDisconnected?: (hand: 'left' | 'right') => void;
  public onTriggerPress?: (hand: 'left' | 'right', value: number) => void;
  public onGripPress?: (hand: 'left' | 'right', value: number) => void;
  public onButtonPress?: (hand: 'left' | 'right', button: string) => void;
  public onThumbstickMove?: (hand: 'left' | 'right', x: number, y: number) => void;

  initialize(session: XRSession): void {
    this.session = session;

    session.addEventListener('inputsourceschange', (event) => {
      for (const source of (event as XRInputSourcesChangeEvent).added) {
        this.addController(source);
      }
      for (const source of (event as XRInputSourcesChangeEvent).removed) {
        this.removeController(source);
      }
    });
  }

  private addController(source: XRInputSource): void {
    const hand = source.handedness === 'left' ? 'left' : 'right';
    const id = `controller-${hand}`;

    this.inputSources.set(id, source);

    const state: XRControllerState = {
      hand,
      position: { x: 0, y: 0, z: 0 },
      rotation: { x: 0, y: 0, z: 0, w: 1 },
      trigger: 0,
      grip: 0,
      thumbstick: { x: 0, y: 0 },
      buttons: { a: false, b: false },
    };

    this.controllers.set(id, state);
    this.onControllerConnected?.(state);
  }

  private removeController(source: XRInputSource): void {
    const hand = source.handedness === 'left' ? 'left' : 'right';
    const id = `controller-${hand}`;

    this.inputSources.delete(id);
    this.controllers.delete(id);
    this.onControllerDisconnected?.(hand);
  }

  update(frame: XRFrame, referenceSpace: XRReferenceSpace): void {
    for (const [id, source] of this.inputSources) {
      const state = this.controllers.get(id);
      if (!state) continue;

      // Update pose
      const pose = frame.getPose(source.gripSpace!, referenceSpace);
      if (pose) {
        state.position = {
          x: pose.transform.position.x,
          y: pose.transform.position.y,
          z: pose.transform.position.z,
        };
        state.rotation = {
          x: pose.transform.orientation.x,
          y: pose.transform.orientation.y,
          z: pose.transform.orientation.z,
          w: pose.transform.orientation.w,
        };
      }

      // Update buttons
      const gamepad = source.gamepad;
      if (gamepad) {
        // Trigger
        const triggerValue = gamepad.buttons[0]?.value || 0;
        if (triggerValue !== state.trigger) {
          state.trigger = triggerValue;
          this.onTriggerPress?.(state.hand, triggerValue);
        }

        // Grip
        const gripValue = gamepad.buttons[1]?.value || 0;
        if (gripValue !== state.grip) {
          state.grip = gripValue;
          this.onGripPress?.(state.hand, gripValue);
        }

        // Thumbstick
        const thumbX = gamepad.axes[2] || 0;
        const thumbY = gamepad.axes[3] || 0;
        if (thumbX !== state.thumbstick.x || thumbY !== state.thumbstick.y) {
          state.thumbstick = { x: thumbX, y: thumbY };
          this.onThumbstickMove?.(state.hand, thumbX, thumbY);
        }

        // Buttons
        const aPressed = gamepad.buttons[4]?.pressed || false;
        const bPressed = gamepad.buttons[5]?.pressed || false;
        if (aPressed !== state.buttons.a) {
          state.buttons.a = aPressed;
          if (aPressed) this.onButtonPress?.(state.hand, 'a');
        }
        if (bPressed !== state.buttons.b) {
          state.buttons.b = bPressed;
          if (bPressed) this.onButtonPress?.(state.hand, 'b');
        }

        // Haptic actuator
        state.hapticActuator = gamepad.hapticActuators?.[0];
      }
    }
  }

  getController(hand: 'left' | 'right'): XRControllerState | undefined {
    return this.controllers.get(`controller-${hand}`);
  }

  async vibrate(hand: 'left' | 'right', intensity = 1, duration = 100): Promise<void> {
    const controller = this.getController(hand);
    if (controller?.hapticActuator) {
      await controller.hapticActuator.pulse(intensity, duration);
    }
  }

  dispose(): void {
    this.controllers.clear();
    this.inputSources.clear();
  }
}

// ==================== Hand Tracking Manager ====================

export class XRHandTrackingManager {
  private session: XRSession | null = null;
  private hands: Map<string, XRHandState> = new Map();
  private gestureRecognizer: GestureRecognizer;

  public onGesture?: (gesture: XRGesture) => void;
  public onPinch?: (hand: 'left' | 'right', strength: number) => void;

  constructor() {
    this.gestureRecognizer = new GestureRecognizer();
  }

  initialize(session: XRSession): void {
    this.session = session;

    // Initialize hand states
    this.hands.set('left', this.createEmptyHandState('left'));
    this.hands.set('right', this.createEmptyHandState('right'));
  }

  private createEmptyHandState(hand: 'left' | 'right'): XRHandState {
    return {
      hand,
      joints: new Map(),
      pinchStrength: 0,
      gripStrength: 0,
    };
  }

  update(frame: XRFrame, referenceSpace: XRReferenceSpace, inputSources: XRInputSourceArray): void {
    for (const source of inputSources) {
      if (!source.hand) continue;

      const handedness = source.handedness === 'left' ? 'left' : 'right';
      const state = this.hands.get(handedness);
      if (!state) continue;

      // Update all joints
      for (const jointName of source.hand.keys()) {
        const joint = source.hand.get(jointName as XRHandJoint);
        if (!joint) continue;

        const pose = frame.getJointPose?.(joint, referenceSpace);
        if (pose) {
          state.joints.set(jointName as XRHandJoint, {
            position: {
              x: pose.transform.position.x,
              y: pose.transform.position.y,
              z: pose.transform.position.z,
            },
            rotation: {
              x: pose.transform.orientation.x,
              y: pose.transform.orientation.y,
              z: pose.transform.orientation.z,
              w: pose.transform.orientation.w,
            },
            radius: pose.radius || 0.01,
          });
        }
      }

      // Calculate pinch strength
      const thumbTip = state.joints.get('thumb-tip');
      const indexTip = state.joints.get('index-finger-tip');
      if (thumbTip && indexTip) {
        const distance = this.vectorDistance(thumbTip.position, indexTip.position);
        state.pinchStrength = Math.max(0, 1 - distance / 0.05);

        if (state.pinchStrength > 0.8) {
          this.onPinch?.(handedness, state.pinchStrength);
        }
      }

      // Calculate grip strength
      const palm = state.joints.get('wrist');
      const middleTip = state.joints.get('middle-finger-tip');
      if (palm && middleTip) {
        const distance = this.vectorDistance(palm.position, middleTip.position);
        state.gripStrength = Math.max(0, 1 - distance / 0.1);
      }

      // Recognize gestures
      const gesture = this.gestureRecognizer.recognize(state);
      if (gesture) {
        this.onGesture?.(gesture);
      }
    }
  }

  private vectorDistance(a: Vector3, b: Vector3): number {
    return Math.sqrt(
      Math.pow(a.x - b.x, 2) +
      Math.pow(a.y - b.y, 2) +
      Math.pow(a.z - b.z, 2)
    );
  }

  getHand(hand: 'left' | 'right'): XRHandState | undefined {
    return this.hands.get(hand);
  }

  dispose(): void {
    this.hands.clear();
  }
}

// ==================== Gesture Recognizer ====================

class GestureRecognizer {
  private lastGestures: Map<string, { type: string; time: number }> = new Map();
  private gestureDebounce = 500; // ms

  recognize(hand: XRHandState): XRGesture | null {
    const now = Date.now();
    const lastGesture = this.lastGestures.get(hand.hand);

    // Debounce gestures
    if (lastGesture && now - lastGesture.time < this.gestureDebounce) {
      return null;
    }

    const wrist = hand.joints.get('wrist');
    if (!wrist) return null;

    const gesture = this.detectGesture(hand);
    if (gesture && (!lastGesture || lastGesture.type !== gesture.type)) {
      this.lastGestures.set(hand.hand, { type: gesture.type, time: now });
      return gesture;
    }

    return null;
  }

  private detectGesture(hand: XRHandState): XRGesture | null {
    const wrist = hand.joints.get('wrist')!;

    // Pinch detection
    if (hand.pinchStrength > 0.9) {
      return {
        type: 'pinch',
        hand: hand.hand,
        confidence: hand.pinchStrength,
        position: wrist.position,
      };
    }

    // Fist detection
    if (hand.gripStrength > 0.8) {
      return {
        type: 'fist',
        hand: hand.hand,
        confidence: hand.gripStrength,
        position: wrist.position,
      };
    }

    // Point detection
    const indexExtended = this.isFingerExtended(hand, 'index-finger');
    const middleCurled = !this.isFingerExtended(hand, 'middle-finger');
    const ringCurled = !this.isFingerExtended(hand, 'ring-finger');
    const pinkyCurled = !this.isFingerExtended(hand, 'pinky-finger');

    if (indexExtended && middleCurled && ringCurled && pinkyCurled) {
      return {
        type: 'point',
        hand: hand.hand,
        confidence: 0.9,
        position: wrist.position,
      };
    }

    // Open hand
    const allExtended =
      this.isFingerExtended(hand, 'index-finger') &&
      this.isFingerExtended(hand, 'middle-finger') &&
      this.isFingerExtended(hand, 'ring-finger') &&
      this.isFingerExtended(hand, 'pinky-finger');

    if (allExtended && hand.gripStrength < 0.2) {
      return {
        type: 'open',
        hand: hand.hand,
        confidence: 0.85,
        position: wrist.position,
      };
    }

    // Peace sign
    if (
      this.isFingerExtended(hand, 'index-finger') &&
      this.isFingerExtended(hand, 'middle-finger') &&
      !this.isFingerExtended(hand, 'ring-finger') &&
      !this.isFingerExtended(hand, 'pinky-finger')
    ) {
      return {
        type: 'peace',
        hand: hand.hand,
        confidence: 0.85,
        position: wrist.position,
      };
    }

    return null;
  }

  private isFingerExtended(hand: XRHandState, finger: string): boolean {
    const proximal = hand.joints.get(`${finger}-phalanx-proximal` as XRHandJoint);
    const tip = hand.joints.get(`${finger}-tip` as XRHandJoint);
    const wrist = hand.joints.get('wrist');

    if (!proximal || !tip || !wrist) return false;

    const proximalDist = this.vectorDistance(proximal.position, wrist.position);
    const tipDist = this.vectorDistance(tip.position, wrist.position);

    return tipDist > proximalDist * 1.5;
  }

  private vectorDistance(a: Vector3, b: Vector3): number {
    return Math.sqrt(
      Math.pow(a.x - b.x, 2) +
      Math.pow(a.y - b.y, 2) +
      Math.pow(a.z - b.z, 2)
    );
  }
}

// ==================== VR Mixer ====================

export class VRMixer {
  private channels: Map<string, VRMixerChannel> = new Map();
  private masterVolume = 1;
  private selectedChannel: string | null = null;

  public onChannelChange?: (channel: VRMixerChannel) => void;
  public onMasterChange?: (volume: number) => void;

  constructor() {
    // Create default channels
    this.createDefaultChannels();
  }

  private createDefaultChannels(): void {
    const channelNames = ['Drums', 'Bass', 'Keys', 'Guitar', 'Vocals', 'FX', 'Master'];

    channelNames.forEach((name, i) => {
      const channel: VRMixerChannel = {
        id: `channel-${i}`,
        name,
        position: { x: -1.5 + i * 0.5, y: 0.8, z: -1 },
        fader: 0.75,
        pan: 0,
        mute: false,
        solo: false,
        eq: { low: 0, mid: 0, high: 0 },
        sends: [],
      };
      this.channels.set(channel.id, channel);
    });
  }

  setFader(channelId: string, value: number): void {
    const channel = this.channels.get(channelId);
    if (channel) {
      channel.fader = Math.max(0, Math.min(1, value));
      this.onChannelChange?.(channel);
    }
  }

  setPan(channelId: string, value: number): void {
    const channel = this.channels.get(channelId);
    if (channel) {
      channel.pan = Math.max(-1, Math.min(1, value));
      this.onChannelChange?.(channel);
    }
  }

  toggleMute(channelId: string): void {
    const channel = this.channels.get(channelId);
    if (channel) {
      channel.mute = !channel.mute;
      this.onChannelChange?.(channel);
    }
  }

  toggleSolo(channelId: string): void {
    const channel = this.channels.get(channelId);
    if (channel) {
      channel.solo = !channel.solo;
      this.onChannelChange?.(channel);
    }
  }

  setEQ(channelId: string, band: 'low' | 'mid' | 'high', value: number): void {
    const channel = this.channels.get(channelId);
    if (channel) {
      channel.eq[band] = Math.max(-12, Math.min(12, value));
      this.onChannelChange?.(channel);
    }
  }

  setMasterVolume(volume: number): void {
    this.masterVolume = Math.max(0, Math.min(1, volume));
    this.onMasterChange?.(this.masterVolume);
  }

  selectChannel(channelId: string | null): void {
    this.selectedChannel = channelId;
  }

  getChannels(): VRMixerChannel[] {
    return Array.from(this.channels.values());
  }

  getChannel(id: string): VRMixerChannel | undefined {
    return this.channels.get(id);
  }

  getSelectedChannel(): VRMixerChannel | undefined {
    return this.selectedChannel ? this.channels.get(this.selectedChannel) : undefined;
  }

  // Handle VR interaction
  handleControllerInteraction(
    controller: XRControllerState,
    action: 'grab' | 'release' | 'move'
  ): void {
    // Find channel at controller position
    for (const channel of this.channels.values()) {
      const dx = controller.position.x - channel.position.x;
      const dz = controller.position.z - channel.position.z;
      const distance = Math.sqrt(dx * dx + dz * dz);

      if (distance < 0.15) {
        if (action === 'grab' && controller.grip > 0.8) {
          this.selectChannel(channel.id);
        } else if (action === 'move' && this.selectedChannel === channel.id) {
          // Map vertical movement to fader
          const faderChange = (controller.position.y - 0.8) * 2;
          this.setFader(channel.id, 0.5 + faderChange);
        }
        break;
      }
    }

    if (action === 'release') {
      this.selectChannel(null);
    }
  }
}

// ==================== VR Keyboard ====================

export class VRKeyboard {
  private keys: Map<number, { position: Vector3; pressed: boolean }> = new Map();
  private position: Vector3 = { x: 0, y: 0.8, z: -0.5 };
  private octave = 4;

  public onNoteOn?: (note: number, velocity: number, hand: 'left' | 'right') => void;
  public onNoteOff?: (note: number, hand: 'left' | 'right') => void;

  constructor() {
    this.createKeys();
  }

  private createKeys(): void {
    // Create 2 octaves of keys
    const whiteNotes = [0, 2, 4, 5, 7, 9, 11]; // C, D, E, F, G, A, B
    const blackNotes = [1, 3, 6, 8, 10]; // C#, D#, F#, G#, A#

    for (let octave = 0; octave < 2; octave++) {
      // White keys
      whiteNotes.forEach((note, i) => {
        const midiNote = 60 + octave * 12 + note;
        this.keys.set(midiNote, {
          position: {
            x: this.position.x - 0.5 + (octave * 7 + i) * 0.07,
            y: this.position.y,
            z: this.position.z,
          },
          pressed: false,
        });
      });

      // Black keys
      blackNotes.forEach((note, i) => {
        const midiNote = 60 + octave * 12 + note;
        const whiteIndex = note < 5 ? Math.floor(note / 2) : Math.floor((note - 1) / 2) + 1;
        this.keys.set(midiNote, {
          position: {
            x: this.position.x - 0.5 + (octave * 7 + whiteIndex) * 0.07 + 0.035,
            y: this.position.y + 0.02,
            z: this.position.z - 0.03,
          },
          pressed: false,
        });
      });
    }
  }

  handleHandInteraction(hand: XRHandState): void {
    const indexTip = hand.joints.get('index-finger-tip');
    if (!indexTip) return;

    for (const [note, key] of this.keys) {
      const dx = indexTip.position.x - key.position.x;
      const dy = indexTip.position.y - key.position.y;
      const dz = indexTip.position.z - key.position.z;
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

      // Key collision (pressing down)
      if (distance < 0.02 && dy < 0) {
        if (!key.pressed) {
          key.pressed = true;
          const velocity = Math.min(127, Math.floor(Math.abs(dy) * 1000));
          this.onNoteOn?.(note, velocity, hand.hand);
        }
      } else if (key.pressed && distance > 0.03) {
        key.pressed = false;
        this.onNoteOff?.(note, hand.hand);
      }
    }
  }

  handleControllerInteraction(controller: XRControllerState): void {
    for (const [note, key] of this.keys) {
      const dx = controller.position.x - key.position.x;
      const dy = controller.position.y - key.position.y;
      const dz = controller.position.z - key.position.z;
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

      if (distance < 0.03 && controller.trigger > 0.5) {
        if (!key.pressed) {
          key.pressed = true;
          const velocity = Math.floor(controller.trigger * 127);
          this.onNoteOn?.(note, velocity, controller.hand);
        }
      } else if (key.pressed && (distance > 0.05 || controller.trigger < 0.3)) {
        key.pressed = false;
        this.onNoteOff?.(note, controller.hand);
      }
    }
  }

  setOctave(octave: number): void {
    this.octave = Math.max(0, Math.min(8, octave));
    this.createKeys();
  }

  setPosition(position: Vector3): void {
    this.position = position;
    this.createKeys();
  }

  getKeys(): Map<number, { position: Vector3; pressed: boolean }> {
    return this.keys;
  }
}

// ==================== VR Drum Kit ====================

export class VRDrumKit {
  private pads: Map<string, { position: Vector3; radius: number; lastHit: number }> = new Map();
  private position: Vector3 = { x: 1, y: 0.8, z: -0.5 };

  public onHit?: (pad: string, velocity: number, hand: 'left' | 'right') => void;

  constructor() {
    this.createPads();
  }

  private createPads(): void {
    const padLayout = [
      { id: 'kick', pos: { x: 0, y: -0.3, z: 0 }, radius: 0.15 },
      { id: 'snare', pos: { x: -0.15, y: 0, z: 0.1 }, radius: 0.1 },
      { id: 'hihat', pos: { x: -0.3, y: 0.1, z: -0.1 }, radius: 0.08 },
      { id: 'tom1', pos: { x: -0.1, y: 0.15, z: -0.1 }, radius: 0.08 },
      { id: 'tom2', pos: { x: 0.1, y: 0.15, z: -0.1 }, radius: 0.08 },
      { id: 'floor-tom', pos: { x: 0.25, y: -0.1, z: 0.1 }, radius: 0.1 },
      { id: 'crash', pos: { x: -0.25, y: 0.3, z: -0.2 }, radius: 0.12 },
      { id: 'ride', pos: { x: 0.25, y: 0.25, z: -0.2 }, radius: 0.12 },
    ];

    for (const pad of padLayout) {
      this.pads.set(pad.id, {
        position: {
          x: this.position.x + pad.pos.x,
          y: this.position.y + pad.pos.y,
          z: this.position.z + pad.pos.z,
        },
        radius: pad.radius,
        lastHit: 0,
      });
    }
  }

  handleControllerInteraction(controller: XRControllerState, velocity: Vector3): void {
    const now = Date.now();
    const speed = Math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2);

    for (const [padId, pad] of this.pads) {
      const dx = controller.position.x - pad.position.x;
      const dy = controller.position.y - pad.position.y;
      const dz = controller.position.z - pad.position.z;
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

      // Hit detection
      if (distance < pad.radius && speed > 0.5 && now - pad.lastHit > 50) {
        pad.lastHit = now;
        const hitVelocity = Math.min(127, Math.floor(speed * 50));
        this.onHit?.(padId, hitVelocity, controller.hand);
      }
    }
  }

  handleHandInteraction(hand: XRHandState, velocity: Vector3): void {
    const now = Date.now();
    const indexTip = hand.joints.get('index-finger-tip');
    if (!indexTip) return;

    const speed = Math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2);

    for (const [padId, pad] of this.pads) {
      const dx = indexTip.position.x - pad.position.x;
      const dy = indexTip.position.y - pad.position.y;
      const dz = indexTip.position.z - pad.position.z;
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

      if (distance < pad.radius && speed > 0.3 && now - pad.lastHit > 50) {
        pad.lastHit = now;
        const hitVelocity = Math.min(127, Math.floor(speed * 80));
        this.onHit?.(padId, hitVelocity, hand.hand);
      }
    }
  }

  setPosition(position: Vector3): void {
    this.position = position;
    this.createPads();
  }

  getPads(): Map<string, { position: Vector3; radius: number }> {
    return new Map(Array.from(this.pads).map(([id, pad]) => [id, { position: pad.position, radius: pad.radius }]));
  }
}

// ==================== XR Studio Manager ====================

export class XRStudioManager {
  private sessionManager: XRSessionManager;
  private controllerManager: XRControllerManager;
  private handTrackingManager: XRHandTrackingManager;
  private mixer: VRMixer;
  private keyboard: VRKeyboard;
  private drumKit: VRDrumKit;
  private environment: XREnvironment | null = null;

  public onMIDINote?: (note: number, velocity: number, channel: number) => void;
  public onDrumHit?: (pad: string, velocity: number) => void;
  public onMixerChange?: (channelId: string, param: string, value: number) => void;

  constructor() {
    this.sessionManager = new XRSessionManager();
    this.controllerManager = new XRControllerManager();
    this.handTrackingManager = new XRHandTrackingManager();
    this.mixer = new VRMixer();
    this.keyboard = new VRKeyboard();
    this.drumKit = new VRDrumKit();

    this.setupCallbacks();
  }

  private setupCallbacks(): void {
    // Keyboard callbacks
    this.keyboard.onNoteOn = (note, velocity) => {
      this.onMIDINote?.(note, velocity, 1);
    };

    // Drum callbacks
    this.drumKit.onHit = (pad, velocity) => {
      this.onDrumHit?.(pad, velocity);
    };

    // Mixer callbacks
    this.mixer.onChannelChange = (channel) => {
      this.onMixerChange?.(channel.id, 'fader', channel.fader);
    };

    // Controller haptics on note
    this.keyboard.onNoteOn = (note, velocity, hand) => {
      this.controllerManager.vibrate(hand, velocity / 127 * 0.5, 50);
      this.onMIDINote?.(note, velocity, 1);
    };

    this.drumKit.onHit = (pad, velocity, hand) => {
      this.controllerManager.vibrate(hand, velocity / 127, 100);
      this.onDrumHit?.(pad, velocity);
    };

    // Session frame callback
    this.sessionManager.onFrame = (time, frame) => {
      this.update(frame);
    };
  }

  async startVRSession(canvas: HTMLCanvasElement): Promise<void> {
    await this.sessionManager.startSession('immersive-vr', canvas);

    const session = this.sessionManager.getSession();
    if (session) {
      this.controllerManager.initialize(session);
      this.handTrackingManager.initialize(session);
    }
  }

  async endVRSession(): Promise<void> {
    await this.sessionManager.endSession();
    this.controllerManager.dispose();
    this.handTrackingManager.dispose();
  }

  private update(frame: XRFrame): void {
    const referenceSpace = this.sessionManager.getReferenceSpace();
    if (!referenceSpace) return;

    const session = this.sessionManager.getSession();
    if (!session) return;

    // Update controllers
    this.controllerManager.update(frame, referenceSpace);

    // Update hand tracking
    this.handTrackingManager.update(frame, referenceSpace, session.inputSources);

    // Process controller interactions
    const leftController = this.controllerManager.getController('left');
    const rightController = this.controllerManager.getController('right');

    if (leftController) {
      this.keyboard.handleControllerInteraction(leftController);
      this.drumKit.handleControllerInteraction(leftController, { x: 0, y: -0.5, z: 0 });
    }

    if (rightController) {
      this.keyboard.handleControllerInteraction(rightController);
      this.drumKit.handleControllerInteraction(rightController, { x: 0, y: -0.5, z: 0 });
    }

    // Process hand interactions
    const leftHand = this.handTrackingManager.getHand('left');
    const rightHand = this.handTrackingManager.getHand('right');

    if (leftHand) {
      this.keyboard.handleHandInteraction(leftHand);
      this.drumKit.handleHandInteraction(leftHand, { x: 0, y: -0.3, z: 0 });
    }

    if (rightHand) {
      this.keyboard.handleHandInteraction(rightHand);
      this.drumKit.handleHandInteraction(rightHand, { x: 0, y: -0.3, z: 0 });
    }
  }

  setEnvironment(environment: XREnvironment): void {
    this.environment = environment;
  }

  getMixer(): VRMixer {
    return this.mixer;
  }

  getKeyboard(): VRKeyboard {
    return this.keyboard;
  }

  getDrumKit(): VRDrumKit {
    return this.drumKit;
  }

  isVRSupported(): boolean {
    return this.sessionManager.isXRSupported();
  }

  dispose(): void {
    this.controllerManager.dispose();
    this.handTrackingManager.dispose();
  }
}

// ==================== Singleton ====================

let xrStudioManager: XRStudioManager | null = null;

export function getXRStudioManager(): XRStudioManager {
  if (!xrStudioManager) {
    xrStudioManager = new XRStudioManager();
  }
  return xrStudioManager;
}

export default {
  XRStudioManager,
  getXRStudioManager,
  XRSessionManager,
  XRControllerManager,
  XRHandTrackingManager,
  VRMixer,
  VRKeyboard,
  VRDrumKit,
};
