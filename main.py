import numpy as np
import config # Ensure this is the correct, updated config
from environment import HetNetHandoverEnv # Import the new environment
from dqn_agent import DQNAgent
from traditional_agent import TraditionalAgent
from simulation import run_simulation_hetnet # Import the new simulation runner
from plotting import plot_results, plot_rewards
import os
import tensorflow as tf

def main_hetnet():
    # GPU Config
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e: print(e)

    print("--- Initializing HetNet Environment ---")
    env_hetnet = HetNetHandoverEnv(
        town_width=config.TOWN_WIDTH,
        town_height=config.TOWN_HEIGHT,
        gnb_configs=config.GNB_CONFIGS,
        cell_type_specs=config.CELL_TYPE_SPECS,
        ue_params_list=config.UE_PARAMS_LIST,
        steps_per_episode=config.STEPS_PER_EPISODE_HETNET,
        path_loss_exponent=config.PATH_LOSS_EXPONENT_HETNET,
        ref_dist=config.REFERENCE_DISTANCE_HETNET,
        ref_loss=config.REFERENCE_LOSS_HETNET,
        noise_std_dev=config.NOISE_STD_DEV_HETNET,
        very_low_rsrp_threshold=config.VERY_LOW_RSRP_THRESHOLD,
        disconnection_penalty=config.DISCONNECTION_PENALTY_HETNET,
        seed=42
    )
    
    # --- DEFINE THESE **BEFORE** AGENT INITIALIZATION ---
    state_size = env_hetnet.observation_space.shape[0] 
    action_size = env_hetnet.action_space.n       
    num_ues = env_hetnet.num_ues # Now num_ues is defined
    print(f"HetNet Env: State size: {state_size}, Action size: {action_size}, Num UEs: {num_ues}")

    # --- Initialize DQN Agents (one per UE) ---
    dqn_agents_hetnet = []
    for i in range(num_ues): # num_ues is now available
        agent = DQNAgent(
            state_size,
            action_size,
            learning_rate=config.LEARNING_RATE_HETNET,
            gamma=config.DISCOUNT_FACTOR_HETNET,
            epsilon_start=config.EPSILON_START_HETNET,
            epsilon_end=config.EPSILON_END_HETNET,
            epsilon_decay_steps=config.EPSILON_DECAY_STEPS_HETNET,
            replay_buffer_size=config.REPLAY_BUFFER_SIZE_HETNET,
            batch_size=config.BATCH_SIZE_HETNET,
            target_update_freq=config.TARGET_UPDATE_FREQ_HETNET
        )
        dqn_agents_hetnet.append(agent)
    
    # --- Initialize Traditional Agents (one per UE) ---
    traditional_agents_hetnet = []
    for i in range(num_ues): # num_ues is now available
        agent = TraditionalAgent(
            num_gnbs=action_size, # action_size is also defined now
            hysteresis_db=config.HYSTERESIS_DB_HETNET,
            time_to_trigger_steps=config.TIME_TO_TRIGGER_HETNET
        )
        traditional_agents_hetnet.append(agent)

    # --- Train DQN Agents on HetNet ---
    # (Rest of the main_hetnet function remains the same)
    # ...
    print("\n--- Training DQN Agents on HetNet ---")
    is_dqn_list_train = [True] * num_ues
    train_dqn_list_train = [True] * num_ues
    
    # Optionally load pre-trained weights if continuing training
    # for i, agent in enumerate(dqn_agents_hetnet):
    #     if os.path.exists(f"dqn_hetnet_ue{i}_final.weights.h5"):
    #         print(f"Loading pre-trained weights for UE {i}")
    #         agent.load(f"dqn_hetnet_ue{i}_final.weights.h5")

    dqn_results_train_hetnet, dqn_rewards_hetnet = run_simulation_hetnet(
        env_hetnet, dqn_agents_hetnet, config.NUM_EPISODES_HETNET, config.STEPS_PER_EPISODE_HETNET,
        is_dqn_list_train, train_dqn_list_train
    )
    for i, agent in enumerate(dqn_agents_hetnet):
        if hasattr(agent, 'save'):
            agent.save(f"dqn_hetnet_ue{i}_final.weights.h5")
    plot_rewards(dqn_rewards_hetnet, title=f"DQN Training Rewards (HetNet - {num_ues} UEs - Aggregated)")

    # --- Evaluate Trained DQN Agents on HetNet ---
    print("\n--- Evaluating Trained DQN Agents on HetNet ---")
    for i, agent in enumerate(dqn_agents_hetnet): # Load final trained weights for eval
        if hasattr(agent, 'load') and os.path.exists(f"dqn_hetnet_ue{i}_final.weights.h5"):
             agent.load(f"dqn_hetnet_ue{i}_final.weights.h5")
        agent.epsilon = 0.0 # Turn off exploration for evaluation

    is_dqn_list_eval = [True] * num_ues
    train_dqn_list_eval = [False] * num_ues # No training during eval
    num_eval_episodes = max(10, config.NUM_EPISODES_HETNET // 5)
    
    dqn_results_eval_hetnet, _ = run_simulation_hetnet(
        env_hetnet, dqn_agents_hetnet, num_eval_episodes, config.STEPS_PER_EPISODE_HETNET,
        is_dqn_list_eval, train_dqn_list_eval
    )

    # --- Evaluate Traditional Agents on HetNet ---
    print("\n--- Evaluating Traditional Agents on HetNet ---")
    is_trad_list_eval = [False] * num_ues # Not DQN
    train_trad_list_eval = [False] * num_ues
    traditional_results_hetnet, _ = run_simulation_hetnet(
        env_hetnet, traditional_agents_hetnet, num_eval_episodes, config.STEPS_PER_EPISODE_HETNET,
        is_trad_list_eval, train_trad_list_eval
    )

    # --- Compare and Plot HetNet Results ---
    print("\n--- Plotting HetNet Comparison (Aggregated over UEs) ---")
    plot_results(dqn_results_eval_hetnet, traditional_results_hetnet, num_eval_episodes)

    print("\nHetNet Simulation and comparison complete.")
    env_hetnet.close() # Close visualizer if any


if __name__ == "__main__":
    # You can call main_udn() from the previous step or main_hetnet() here
    main_hetnet()