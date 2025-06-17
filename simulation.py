import numpy as np
from collections import defaultdict
import config # Ensure this imports the common config file

def run_simulation_hetnet(env, agents, num_episodes, steps_per_episode, is_dqn_list, train_dqn_list):
    """
    Runs the HetNet simulation for a list of agents (one per UE).
    is_dqn_list and train_dqn_list are lists of booleans.
    """
    num_ues = env.num_ues
    aggregated_results = defaultdict(list) 
    all_episode_total_rewards = [] 

    for e in range(num_episodes):
        current_observations, _ = env.reset() 
        
        episode_stats_all_ues = {
            'handovers': 0, 'failures': 0, 'ping_pongs': 0,
            'latency': [], 'bandwidth': [], 'serving_rsrp': [],
            'rewards': [0.0] * num_ues 
        }

        # Reset traditional agents if any
        for i, agent in enumerate(agents):
            if not is_dqn_list[i] and hasattr(agent, 'reset'):
                agent.reset()
        
        for step in range(steps_per_episode):
            actions = []
            for i in range(num_ues):
                if is_dqn_list[i]:
                    action = agents[i].choose_action(current_observations[i])
                else: 
                    action = agents[i].choose_action(current_observations[i], env.ue_serving_gnb_indices[i])
                actions.append(action)
            
            next_observations, rewards_tuple, terminated, truncated, infos_tuple = env.step(tuple(actions))
            done_episode = terminated or truncated 

            for i in range(num_ues):
                episode_stats_all_ues['rewards'][i] += rewards_tuple[i]

                if is_dqn_list[i]:
                    agents[i].remember(current_observations[i], actions[i], rewards_tuple[i],
                                       next_observations[i], done_episode)

                info_ue = infos_tuple[i]
                if info_ue.get('handover_occurred', False):
                    episode_stats_all_ues['handovers'] += 1
                if info_ue.get('handover_failed', False):
                    episode_stats_all_ues['failures'] += 1
                if info_ue.get('ping_pong', False):
                    episode_stats_all_ues['ping_pongs'] += 1
                episode_stats_all_ues['latency'].append(info_ue.get('latency', 0))
                episode_stats_all_ues['bandwidth'].append(info_ue.get('bandwidth', 0))
                episode_stats_all_ues['serving_rsrp'].append(info_ue.get('serving_rsrp', -150))

            current_observations = next_observations

            # Train DQN agents that are set to train
            for i in range(num_ues):
                if is_dqn_list[i] and train_dqn_list[i] and hasattr(agents[i], 'batch_size') and len(agents[i].memory) > agents[i].batch_size:
                    loss = agents[i].replay() # CORRECTED: Call replay without arguments
                    # Optional: log loss for agent i

            if done_episode:
                break
        
        # --- End of Episode ---
        total_episode_reward_sum = sum(episode_stats_all_ues['rewards'])
        all_episode_total_rewards.append(total_episode_reward_sum)

        aggregated_results['total_handovers'].append(episode_stats_all_ues['handovers'])
        aggregated_results['total_failures'].append(episode_stats_all_ues['failures'])
        aggregated_results['total_ping_pongs'].append(episode_stats_all_ues['ping_pongs'])
        
        avg_latency = np.mean(episode_stats_all_ues['latency']) if episode_stats_all_ues['latency'] else 0
        avg_bandwidth = np.mean(episode_stats_all_ues['bandwidth']) if episode_stats_all_ues['bandwidth'] else 0
        avg_rsrp = np.mean(episode_stats_all_ues['serving_rsrp']) if episode_stats_all_ues['serving_rsrp'] else -150
        
        aggregated_results['avg_latency'].append(avg_latency)
        aggregated_results['avg_bandwidth'].append(avg_bandwidth)
        aggregated_results['avg_rsrp'].append(avg_rsrp)

        total_ho_attempts = episode_stats_all_ues['handovers'] + episode_stats_all_ues['failures']
        failure_rate = episode_stats_all_ues['failures'] / total_ho_attempts if total_ho_attempts > 0 else 0
        aggregated_results['failure_rate'].append(failure_rate)
        
        ping_pong_rate = episode_stats_all_ues['ping_pongs'] / episode_stats_all_ues['handovers'] if episode_stats_all_ues['handovers'] > 0 else 0
        aggregated_results['ping_pong_rate'].append(ping_pong_rate)

        current_epsilon = -1
        for i in range(num_ues):
            if is_dqn_list[i] and hasattr(agents[i], 'epsilon'):
                current_epsilon = agents[i].epsilon
                break
        
        print(f"Episode {e+1}/{num_episodes} - Sum Reward: {total_episode_reward_sum:.2f}"
              f" - HOs: {episode_stats_all_ues['handovers']}"
              f" - Fails: {episode_stats_all_ues['failures']}"
              f" - PPs: {episode_stats_all_ues['ping_pongs']}"
              f" - Epsilon: {current_epsilon:.4f}" if current_epsilon !=-1 else "")

        if any(is_dqn_list) and any(train_dqn_list) and (e + 1) % 20 == 0: # Save weights periodically (e.g., every 20 episodes)
            for i in range(num_ues):
                if is_dqn_list[i] and train_dqn_list[i] and hasattr(agents[i], 'save'):
                    agents[i].save(f"dqn_hetnet_ue{i}_weights_ep{e+1}.weights.h5")

    agent_type_str = "DQN" if any(is_dqn_list) else "Traditional"
    print(f"Simulation finished for {agent_type_str} agents.")
    return aggregated_results, all_episode_total_rewards