# --- General Simulation Parameters ---
NUM_EPISODES_HETNET = 100 # Might need more for complex scenarios
STEPS_PER_EPISODE_HETNET = 300
NUM_UES = 2

# --- HetNet Area & gNB Deployment ---
TOWN_WIDTH = 1000  # meters (1km)
TOWN_HEIGHT = 1000 # meters

GNB_CONFIGS = {
    'macro': {'num': 5, 'tx_power_dbm': 20, 'deployment': 'grid_sparse'}, # Tx power relative to a baseline, actual RSRP will be calculated
    'pico':  {'num': 15, 'tx_power_dbm': 10, 'deployment': 'random_cluster'}, # Clustered around macro or specific zones
    'femto': {'num': 30, 'tx_power_dbm': 0,  'deployment': 'random_uniform'}  # Uniformly spread, low power
}
# Note: tx_power_dbm here is a relative power level.
# Actual RSRP will depend on this, path loss, and a baseline.
# For RSRP calculation: RSRP = Base_RSRP_at_Ref_Dist + TxPower_gNB_config - PathLoss_from_Ref_Dist
# Let's define a base RSRP at reference distance, e.g., if TxPower_gNB_config = 0.
# This means TxPower_gNB_config is an *offset* to a baseline power.
# OR, more simply: RSRP = (Fixed_Tx_Power_for_PathLoss_Formula + gNB_type_tx_offset) - PathLoss.
# Let's use a clearer method:
# TxPower (dBm) for each cell type, PathLoss (dB), RSRP = TxPower - PathLoss
# Macro: e.g., 46 dBm (typical Tx power)
# Pico:  e.g., 30 dBm
# Femto: e.g., 20 dBm
# We'll use these Tx powers directly in RSRP calculation.

CELL_TYPE_SPECS = {
    'macro': {'tx_power': 46, 'color': 'red', 'marker': '^', 'size': 150},    # dBm
    'pico':  {'tx_power': 30, 'color': 'orange', 'marker': 's', 'size': 80}, # dBm
    'femto': {'tx_power': 20, 'color': 'green', 'marker': 'P', 'size': 50}   # dBm
}
# Total gNBs is the sum of 'num' in GNB_CONFIGS for each type that will use CELL_TYPE_SPECS

# --- UE Parameters (List of dicts, one for each UE) ---
UE_PARAMS_LIST = [
    {'id': 0, 'start_pos': (50, 50), 'end_pos': (TOWN_WIDTH - 50, TOWN_HEIGHT - 50), 'speed': 5.0}, # UE1 diagonal
    {'id': 1, 'start_pos': (50, TOWN_HEIGHT - 50), 'end_pos': (TOWN_WIDTH - 50, 50), 'speed': 4.0}  # UE2 other diagonal
]

# --- Channel Model Parameters (Same as before, but Tx power now varies per gNB) ---
PATH_LOSS_EXPONENT_HETNET = 3.5 # Can be adjusted, e.g., higher in dense urban
REFERENCE_DISTANCE_HETNET = 1.0 # meter
REFERENCE_LOSS_HETNET = 40.0    # dB (Path loss at 1m for a hypothetical isotropic antenna)
# RSRP = TxPower_gNB - (REFERENCE_LOSS_HETNET + 10 * PATH_LOSS_EXPONENT_HETNET * log10(dist / REFERENCE_DISTANCE_HETNET)) + Noise
NOISE_STD_DEV_HETNET = 4.0 # dB

# --- Traditional Agent Parameters ---
HYSTERESIS_DB_HETNET = 3.0
TIME_TO_TRIGGER_HETNET = 2

# --- DQN Parameters (shared for each UE's agent) ---
LEARNING_RATE_HETNET = 0.0005
DISCOUNT_FACTOR_HETNET = 0.99 # Longer episodes might benefit from higher gamma
EPSILON_START_HETNET = 1.0
EPSILON_END_HETNET = 0.01
# Adjusted decay steps based on more UEs and potentially more complex learning
EPSILON_DECAY_STEPS_HETNET = (NUM_EPISODES_HETNET * STEPS_PER_EPISODE_HETNET * NUM_UES) // 4
REPLAY_BUFFER_SIZE_HETNET = 20000 * NUM_UES # Larger buffer
BATCH_SIZE_HETNET = 64
TARGET_UPDATE_FREQ_HETNET = 200 * NUM_UES # Slower target updates relative to total steps

# --- Reward Function Weights (per UE) ---
RSRP_REWARD_WEIGHT_HETNET = 0.8
FAILURE_PENALTY_HETNET = -150
PINGPONG_PENALTY_HETNET = -70
HANDOVER_EXECUTION_PENALTY_HETNET = -10
# New: Penalty for being disconnected (if no suitable gNB, RSRP very low)
DISCONNECTION_PENALTY_HETNET = -200
VERY_LOW_RSRP_THRESHOLD = -125 # dBm