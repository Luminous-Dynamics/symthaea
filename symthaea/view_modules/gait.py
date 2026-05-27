import numpy as np

class BallisticGaitGenerator:
    def __init__(self):
        self.current_freq = 0.0
        self.current_stride_amp = 0.0
        self.current_spine_amp = 0.0
        self.gait_phase = 0.0

    def compute_stride(self, elapsed_time, dt, nominal_targets, r_pressure=0.0, l_pressure=0.0):
        # ADVANCED TERRAIN ADAPTATION CURRICULUM
        if elapsed_time < 4.0:
            state_label, target_freq, stride_amp, spine_amp = "STAND", 0.0, 0.0, 0.0
        elif elapsed_time < 14.0:
            # High-clearance stepping gait designed to scale vertical step barriers
            state_label, target_freq, stride_amp, spine_amp = "TERRAIN_STEP", 1.2, 0.42, 0.05
        else:
            state_label, target_freq, stride_amp, spine_amp = "TERRAIN_RUN ", 2.0, 0.52, 0.10

        smoothing_factor = 0.05
        self.current_freq += (target_freq - self.current_freq) * smoothing_factor
        self.current_stride_amp += (stride_amp - self.current_stride_amp) * smoothing_factor
        self.current_spine_amp += (spine_amp - self.current_spine_amp) * smoothing_factor

        # Tactile phase regulation gating via foot touch pressure values
        phase_warp = 0.0
        if state_label != "STAND":
            r_is_swing = np.sin(self.gait_phase) > 0
            l_is_swing = not r_is_swing
            if r_is_swing and r_pressure > 10.0:  phase_warp = 3.0 * dt
            if l_is_swing and l_pressure > 10.0:  phase_warp = 3.0 * dt

        self.gait_phase += 2.0 * np.pi * self.current_freq * dt + phase_warp
        
        # Explicit 180-degree phase separation between left and right channels
        r_p = self.gait_phase
        l_p = self.gait_phase + np.pi

        r_is_swing = np.sin(r_p) > 0
        l_is_swing = not r_is_swing

        ctrl_targets = nominal_targets.copy()

        if state_label != "STAND":
            # Pitch the core abdomen slightly forward to align center of mass naturally
            ctrl_targets[0] = -0.12 

            # ===================================================================
            # TRUE MONOTONIC HUMAN BIPEDAL TRAJECTORY MATRIX
            # Hip Pitch:  Negative = Forward Swing, Positive = Backward Push
            # Knee Pitch: More Negative = Deep Flexion, Less Negative = Extension
            # ===================================================================
            
            # --- Right Leg Stride Profiles ---
            # Monotonic Cosine Sweep glides smoothly from +0.22 (Back) to -0.62 (Forward)
            ctrl_targets[5] = -0.20 + np.cos(r_p) * (self.current_stride_amp * 0.9 + 0.1)
            
            if r_is_swing:
                # Swing Phase: Flex the knee deeply backward (more negative) to clear step lips
                ctrl_targets[6] = -0.35 - np.sin(r_p) * (self.current_stride_amp * 2.2 + 0.2)
                ctrl_targets[7] = -0.30  # Dorsiflexion: Pull toes up to clear obstacle edges
            else:
                # Stance Phase: Extend the knee to lock out straight (-0.25 rad) and bear weight
                ctrl_targets[6] = -0.28
                ctrl_targets[7] = -0.05

            # --- Left Leg Stride Profiles ---
            ctrl_targets[11] = -0.20 + np.cos(l_p) * (self.current_stride_amp * 0.9 + 0.1)
            
            if l_is_swing:
                ctrl_targets[12] = -0.35 - np.sin(l_p) * (self.current_stride_amp * 2.2 + 0.2)
                ctrl_targets[13] = -0.30
            else:
                ctrl_targets[12] = -0.28
                ctrl_targets[13] = -0.05

            # Coordinated upper body counterbalance swings
            ctrl_targets[15] = np.sin(r_p) * self.current_stride_amp * 0.5
            ctrl_targets[17] = -0.40 + np.cos(r_p) * 0.2
            ctrl_targets[18] = np.sin(l_p) * self.current_stride_amp * 0.5
            ctrl_targets[20] = -0.40 + np.cos(l_p) * 0.2

            # Visual gimbal spine counter-waves
            for s in range(53, 64):
                ctrl_targets[s] = np.sin(self.gait_phase - (s - 53) * 0.15) * self.current_spine_amp

        return ctrl_targets, state_label, r_is_swing, l_is_swing
