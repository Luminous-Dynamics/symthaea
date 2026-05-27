import time
import numpy as np

class PosturalObserver:
    def __init__(self):
        self.recovery_state = "NORMAL"
        self.recovery_start_time = 0.0

    def evaluate_fall(self, upright_projection, elapsed_time):
        if self.recovery_state == "NORMAL" and upright_projection < 0.48 and elapsed_time > 2.0:
            print("🚨 FALL ENCOUNTERED: Triggering Modular Recovery Sequence.")
            self.recovery_state = "TUCK"
            self.recovery_start_time = time.time()
        return self.recovery_state

    def compute_recovery_targets(self, xmat, data, pelvis_id, gravity_compensation, nominal_targets):
        rec_elapsed = time.time() - self.recovery_start_time
        ctrl_targets = nominal_targets.copy()
        
        if rec_elapsed < 1.2:
            state_label = "REC_STAGE_1_TUCK"
            ctrl_targets[5], ctrl_targets[11] = 0.60, 0.60   # Pull hips in
            ctrl_targets[6], ctrl_targets[12] = -1.20, -1.20 # Deep knee tuck
            ctrl_targets[7], ctrl_targets[13] = 0.40, 0.40
        elif rec_elapsed < 2.4:
            state_label = "REC_STAGE_2_PUSH"
            ctrl_targets[5], ctrl_targets[11] = -0.10, -0.10 # Explode legs downward
            ctrl_targets[6], ctrl_targets[12] = -0.20, -0.20
            ctrl_targets[7], ctrl_targets[13] = 0.05, 0.05
            data.xfrc_applied[pelvis_id, 2] = 250.0  # Kinetic recovery impulse
        else:
            state_label = "REC_STAGE_3_SNAP"
            if rec_elapsed > 4.0 and xmat[2, 2] > 0.82:
                print("🎯 STAND RECLAIMED: Returning to trajectory curriculum.")
                self.recovery_state = "NORMAL"
                
        return ctrl_targets, state_label
