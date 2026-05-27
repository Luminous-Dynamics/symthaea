import numpy as np

# Spatial Target Destination
GOAL_X = 25.0

# Nominal Athletic Standing Posture Constants (Sign-Aligned to Human Anatomy)
NOMINAL_TARGETS = np.zeros(64)
NOMINAL_TARGETS[5]  = -0.22  # Right Hip pitch (Negative = Forward Flexion)
NOMINAL_TARGETS[11] = -0.22  # Left Hip pitch
NOMINAL_TARGETS[6]  = 0.45   # Right Knee pitch (Positive = Human Backward Flexion)
NOMINAL_TARGETS[12] = 0.45   # Left Knee pitch
NOMINAL_TARGETS[7]  = -0.12  # Right Ankle pitch
NOMINAL_TARGETS[13] = -0.12  # Left Ankle pitch

def get_velocity_limits(num_actuators):
    limits = np.zeros(num_actuators)
    for i in range(num_actuators):
        if i < 3:    limits[i] = 2.5   # Core Abdomen
        elif i < 15: limits[i] = 4.5   # Locomotion Legs
        elif i < 21: limits[i] = 6.0   # Counterbalancing Arms
        elif i < 53: limits[i] = 8.0   # Dexterous Hands
        else:        limits[i] = 3.0   # Flexible Spine Chain
    return limits
