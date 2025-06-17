import numpy as np
import gymnasium as gym
from gymnasium import spaces
import config # Import the updated config
import matplotlib.pyplot as plt # For rendering

class HetNetHandoverEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 5}

    def __init__(self,
                 town_width, town_height,
                 gnb_configs, cell_type_specs,
                 ue_params_list,
                 steps_per_episode,
                 path_loss_exponent, ref_dist, ref_loss, noise_std_dev,
                 very_low_rsrp_threshold, disconnection_penalty,
                 seed=None):
        super().__init__()
        if seed is not None:
            np.random.seed(seed)

        self.town_width = town_width
        self.town_height = town_height
        self.gnb_configs = gnb_configs
        self.cell_type_specs = cell_type_specs
        self.ue_params_list = ue_params_list
        self.num_ues = len(ue_params_list)
        self.steps_per_episode = steps_per_episode

        self.path_loss_exponent = path_loss_exponent
        self.ref_dist = ref_dist
        self.ref_loss = ref_loss
        self.noise_std_dev = noise_std_dev
        self.very_low_rsrp_threshold = very_low_rsrp_threshold
        self.disconnection_penalty = disconnection_penalty


        self._deploy_gnbs() # Populates self.gnb_details (list of dicts)
        # self.num_gnbs_total is set within _deploy_gnbs

        # UE specific states will be lists, one entry per UE
        self.ue_positions = [np.array(p['start_pos'], dtype=float) for p in self.ue_params_list]
        self.ue_end_positions = [np.array(p['end_pos'], dtype=float) for p in self.ue_params_list]
        self.ue_speeds = [p['speed'] for p in self.ue_params_list]
        
        self.ue_move_vectors = []
        self.ue_steps_needed = []
        for i in range(self.num_ues):
            start_pos = self.ue_positions[i]
            end_pos = self.ue_end_positions[i]
            speed = self.ue_speeds[i]
            total_distance = np.linalg.norm(end_pos - start_pos)
            steps_needed = max(1, int(np.ceil(total_distance / speed))) if speed > 0 else self.steps_per_episode
            self.ue_steps_needed.append(steps_needed)
            if steps_needed > 0 and total_distance > 0 :
                self.ue_move_vectors.append((end_pos - start_pos) / steps_needed)
            else:
                self.ue_move_vectors.append(np.zeros_like(start_pos))


        self.ue_serving_gnb_indices = [self._get_best_initial_gnb(i) for i in range(self.num_ues)]

        # For DQN agents (independent learners, so obs/action space is per UE)
        # Observation: RSRP from all gNBs
        self.observation_space = spaces.Box(low=-150, high=0, shape=(self.num_gnbs_total,), dtype=np.float32)
        # Action: Choose one of the gNBs
        self.action_space = spaces.Discrete(self.num_gnbs_total if self.num_gnbs_total > 0 else 1) # Handle case of 0 gNBs for action space

        # Tracking for metrics (list for each UE)
        self.ue_last_serving_gnb = [-1] * self.num_ues
        self.ue_previous_serving_gnb_for_pingpong = list(self.ue_serving_gnb_indices) # Copy initial serving gNBs
        self.ue_steps_on_current_gnb = [0] * self.num_ues
        self.ping_pong_window = 5 # steps

        self.current_step = 0
        self.viewer = None
        self.ue_paths_history = [[] for _ in range(self.num_ues)]


    def _deploy_gnbs(self):
        self.gnb_details = [] 
        gnb_id_counter = 0
        # Ensuring deployment order if sensitive (e.g., macros before picos if picos cluster around macros)
        # For Python 3.7+, dicts maintain insertion order. If older, use collections.OrderedDict for GNB_CONFIGS
        # or iterate over a predefined list of cell types.
        # CELL_ORDER = ['macro', 'pico', 'femto'] # Example if specific order needed for older Python
        # for cell_type in CELL_ORDER:
        #    config_params = self.gnb_configs[cell_type]
        
        for cell_type, config_params in self.gnb_configs.items(): # Uses dict insertion order (Python 3.7+)
            num_to_deploy = config_params['num']
            if num_to_deploy == 0:
                continue # Skip if no gNBs of this type are to be deployed

            tx_power = self.cell_type_specs[cell_type]['tx_power']
            
            if config_params['deployment'] == 'grid_sparse':
                cols = int(np.ceil(np.sqrt(num_to_deploy)))
                rows = int(np.ceil(num_to_deploy / cols)) if cols > 0 else 0
                
                # Avoid division by zero if cols/rows is 0 or 1
                x_spacing = self.town_width / cols if cols > 1 else self.town_width / 2
                y_offset = self.town_width / (2*cols) if cols >=1 else 0 # Center point for single col

                y_spacing = self.town_height / rows if rows > 1 else self.town_height / 2
                x_offset = self.town_height / (2*rows) if rows >=1 else 0 # Center point for single row
                
                count = 0
                for r_idx in range(rows):
                    for c_idx in range(cols):
                        if count < num_to_deploy:
                            loc_x = c_idx * x_spacing + x_offset
                            loc_y = r_idx * y_spacing + y_offset
                            self.gnb_details.append({'id': gnb_id_counter, 'loc': np.array((loc_x, loc_y)), 'tx_power': tx_power, 'type': cell_type})
                            gnb_id_counter += 1
                            count +=1
                        else: break
                    if count >= num_to_deploy: break

            elif config_params['deployment'] == 'random_uniform':
                for _ in range(num_to_deploy):
                    loc = (np.random.uniform(0, self.town_width), np.random.uniform(0, self.town_height))
                    self.gnb_details.append({'id': gnb_id_counter, 'loc': np.array(loc), 'tx_power': tx_power, 'type': cell_type})
                    gnb_id_counter += 1
            
            elif config_params['deployment'] == 'random_cluster':
                num_clusters = max(1, num_to_deploy // 5) # Ensure at least 1 cluster if deploying
                cluster_centers = []

                macro_locs_list = [gd['loc'] for gd in self.gnb_details if gd['type'] == 'macro']

                if macro_locs_list:
                    num_macros_available = len(macro_locs_list)
                    num_centers_from_macros = min(num_clusters, num_macros_available)
                    
                    chosen_indices = np.random.choice(num_macros_available, size=num_centers_from_macros, replace=False)
                    for idx in chosen_indices:
                        cluster_centers.append(macro_locs_list[idx])
                
                num_random_centers_needed = num_clusters - len(cluster_centers)
                if num_random_centers_needed > 0:
                    for _ in range(num_random_centers_needed):
                        random_center = np.array((
                            np.random.uniform(0.1 * self.town_width, 0.9 * self.town_width),
                            np.random.uniform(0.1 * self.town_height, 0.9 * self.town_height)
                        ))
                        cluster_centers.append(random_center)
                
                if not cluster_centers and num_to_deploy > 0 : # Fallback if still no centers but need to deploy
                    cluster_centers.append(np.array((self.town_width / 2, self.town_height / 2))) # Default center
                
                cluster_radius = 0
                if num_clusters > 0 : # Avoid division by zero for radius calculation
                    cluster_radius = min(self.town_width, self.town_height) / (2 * np.sqrt(num_clusters) + 2) 
                
                for i in range(num_to_deploy):
                    if not cluster_centers: # Should ideally not be reached if num_to_deploy > 0
                        loc_x = np.random.uniform(0, self.town_width)
                        loc_y = np.random.uniform(0, self.town_height)
                    else:
                        center = cluster_centers[i % len(cluster_centers)] 
                        angle = np.random.uniform(0, 2 * np.pi)
                        radius_offset = np.random.uniform(0, cluster_radius) if cluster_radius > 0 else 0
                        loc_x = np.clip(center[0] + radius_offset * np.cos(angle), 0, self.town_width)
                        loc_y = np.clip(center[1] + radius_offset * np.sin(angle), 0, self.town_height)
                    
                    self.gnb_details.append({'id': gnb_id_counter, 'loc': np.array((loc_x, loc_y)), 'tx_power': tx_power, 'type': cell_type})
                    gnb_id_counter +=1
        
        self.num_gnbs_total = len(self.gnb_details)
        if self.num_gnbs_total == 0:
            print("Warning: No gNBs were deployed. Check GNB_CONFIGS in config.py.")
            # Add a dummy gNB if none are deployed to prevent downstream errors with empty action/observation spaces
            # This isn't ideal but handles an edge case of misconfiguration.
            self.gnb_details.append({'id': 0, 'loc': np.array((self.town_width/2, self.town_height/2)),
                                     'tx_power': self.cell_type_specs['femto']['tx_power'], 'type': 'femto'})
            self.num_gnbs_total = 1
            gnb_id_counter = 1
        
        print(f"Deployed {self.num_gnbs_total} gNBs.")

    # ... (rest of the HetNetHandoverEnv class: _calculate_rsrp, _get_state_for_ue, etc.)
    # The methods _calculate_rsrp, _get_state_for_ue, _get_best_initial_gnb, reset, step,
    # _simulate_latency, _simulate_bandwidth, _get_info_for_ue, render, close
    # from your previous version should follow here. I'm omitting them for brevity as they
    # were not the source of this specific error. Make sure they are present.

    def _calculate_rsrp(self, ue_pos, gnb_detail):
        dist = np.linalg.norm(ue_pos - gnb_detail['loc'])
        dist = max(dist, self.ref_dist) 

        path_loss_db = self.ref_loss + 10 * self.path_loss_exponent * np.log10(dist / self.ref_dist)
        rsrp = gnb_detail['tx_power'] - path_loss_db + np.random.normal(0, self.noise_std_dev)
        return max(-150.0, min(0.0, rsrp)) 


    def _get_state_for_ue(self, ue_idx):
        ue_pos = self.ue_positions[ue_idx]
        state = np.array([self._calculate_rsrp(ue_pos, gnb) for gnb in self.gnb_details], dtype=np.float32)
        return state

    def _get_best_initial_gnb(self, ue_idx):
        if self.num_gnbs_total == 0: return -1 
        ue_pos = self.ue_positions[ue_idx] 
        rsrps = [self._calculate_rsrp(ue_pos, gnb) for gnb in self.gnb_details]
        return np.argmax(rsrps) if rsrps else -1


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.ue_positions = [np.array(p['start_pos'], dtype=float) for p in self.ue_params_list]
        # Re-deploy gNBs if their placement is stochastic and should change with seed at each reset
        # If gNBs are fixed for the lifetime of the env instance, then _deploy_gnbs is only called in __init__
        # self._deploy_gnbs() # Uncomment if gNBs should be re-randomized per episode based on new seed

        self.ue_serving_gnb_indices = [self._get_best_initial_gnb(i) for i in range(self.num_ues)]
        
        self.ue_last_serving_gnb = [-1] * self.num_ues
        self.ue_previous_serving_gnb_for_pingpong = list(self.ue_serving_gnb_indices)
        self.ue_steps_on_current_gnb = [0] * self.num_ues
        self.ue_paths_history = [[np.copy(pos)] for pos in self.ue_positions]


        observations = tuple(self._get_state_for_ue(i) for i in range(self.num_ues))
        infos = tuple(self._get_info_for_ue(i) for i in range(self.num_ues))
        return observations, infos


    def step(self, actions): 
        self.current_step += 1
        
        rewards = [0.0] * self.num_ues
        next_observations = [np.zeros(self.observation_space.shape, dtype=np.float32) for _ in range(self.num_ues)] # Initialize
        # dones = [False] * self.num_ues # Not used for overall episode termination directly
        infos = [{} for _ in range(self.num_ues)]

        current_rsrp_states_list = [self._get_state_for_ue(i) for i in range(self.num_ues)]

        for i in range(self.num_ues): 
            target_gnb_idx = actions[i]
            handover_occurred_ue = False
            handover_failed_ue = False
            is_ping_pong_ue = False
            
            current_serving_gnb_ue = self.ue_serving_gnb_indices[i]
            gnb_before_potential_ho = current_serving_gnb_ue


            if self.num_gnbs_total == 0 : 
                handover_failed_ue = True 
                rewards[i] += self.disconnection_penalty 
            elif target_gnb_idx != current_serving_gnb_ue: 
                handover_occurred_ue = True
                # Ensure target_gnb_idx is valid
                if not (0 <= target_gnb_idx < self.num_gnbs_total):
                    # Invalid action, treat as failure or stay on current
                    handover_failed_ue = True
                    rewards[i] += config.FAILURE_PENALTY_HETNET # Penalty for invalid action
                else:
                    rsrp_of_target = current_rsrp_states_list[i][target_gnb_idx]
                    if rsrp_of_target < (self.very_low_rsrp_threshold - 5): 
                        handover_failed_ue = True
                        rewards[i] += config.FAILURE_PENALTY_HETNET
                    else: 
                        rewards[i] += config.HANDOVER_EXECUTION_PENALTY_HETNET
                        self.ue_last_serving_gnb[i] = current_serving_gnb_ue
                        self.ue_serving_gnb_indices[i] = target_gnb_idx

                        if self.ue_serving_gnb_indices[i] == self.ue_previous_serving_gnb_for_pingpong[i] and \
                           self.ue_steps_on_current_gnb[i] < self.ping_pong_window and \
                           self.ue_last_serving_gnb[i] != -1: # Ensure there was a previous gNB connection
                            is_ping_pong_ue = True
                            rewards[i] += config.PINGPONG_PENALTY_HETNET
                        
                        self.ue_previous_serving_gnb_for_pingpong[i] = gnb_before_potential_ho
                        self.ue_steps_on_current_gnb[i] = 0
            else: 
                self.ue_steps_on_current_gnb[i] += 1

            if self.current_step < self.ue_steps_needed[i]:
                self.ue_positions[i] += self.ue_move_vectors[i]
            elif self.current_step == self.ue_steps_needed[i]:
                 self.ue_positions[i] = np.copy(self.ue_end_positions[i])
            
            self.ue_paths_history[i].append(np.copy(self.ue_positions[i]))

            next_observations[i] = self._get_state_for_ue(i)
            serving_gnb_idx_ue = self.ue_serving_gnb_indices[i]
            
            serving_rsrp_after_move = -150.0 # Default for disconnected
            if self.num_gnbs_total > 0 and 0 <= serving_gnb_idx_ue < self.num_gnbs_total:
                serving_rsrp_after_move = next_observations[i][serving_gnb_idx_ue]
            else: # Disconnected or invalid serving_gnb_idx
                rewards[i] += self.disconnection_penalty


            min_rsrp_reward = -120
            max_rsrp_reward = -40 
            norm_rsrp = (serving_rsrp_after_move - min_rsrp_reward) / (max_rsrp_reward - min_rsrp_reward)
            rsrp_based_reward = np.clip(norm_rsrp, 0, 1.5)
            rewards[i] += config.RSRP_REWARD_WEIGHT_HETNET * rsrp_based_reward

            if serving_rsrp_after_move < self.very_low_rsrp_threshold:
                rewards[i] += self.disconnection_penalty / 2 

            ue_i_reached_dest = np.allclose(self.ue_positions[i], self.ue_end_positions[i], atol=self.ue_speeds[i]*1.1) or \
                                self.current_step >= self.ue_steps_needed[i] # Make atol slightly larger than one step

            infos[i] = {
                'ue_id': self.ue_params_list[i]['id'],
                'handover_occurred': handover_occurred_ue and not handover_failed_ue,
                'handover_failed': handover_failed_ue,
                'ping_pong': is_ping_pong_ue,
                'serving_rsrp': serving_rsrp_after_move,
                'latency': self._simulate_latency(serving_rsrp_after_move),
                'bandwidth': self._simulate_bandwidth(serving_rsrp_after_move),
                'ue_pos': np.copy(self.ue_positions[i]),
                'serving_gnb': self.ue_serving_gnb_indices[i],
                'reached_destination': ue_i_reached_dest
            }

        all_ues_done_moving = all(infos[i]['reached_destination'] for i in range(self.num_ues))
        
        terminated = self.current_step >= self.steps_per_episode or all_ues_done_moving
        
        return tuple(next_observations), tuple(rewards), terminated, False, tuple(infos)


    def _simulate_latency(self, rsrp):
        latency = 10 + 140 * np.clip(1 - ((rsrp - (-120)) / (-70 - (-120))), 0, 1)
        return max(5, latency)

    def _simulate_bandwidth(self, rsrp):
        bandwidth = 1 + 299 * np.clip(((rsrp - (-120)) / (-70 - (-120))), 0, 1)
        return max(1, bandwidth)

    def _get_info_for_ue(self, ue_idx): # Helper for reset
        return {
            "ue_id": self.ue_params_list[ue_idx]['id'],
            "ue_pos": np.copy(self.ue_positions[ue_idx]),
            "serving_gnb": self.ue_serving_gnb_indices[ue_idx],
            "steps_on_gnb": self.ue_steps_on_current_gnb[ue_idx]
        }


    def render(self, mode='human'):
        if mode == 'human':
            if self.viewer is None:
                plt.ion()
                self.fig, self.ax = plt.subplots(figsize=(10, 10))
                self.viewer = True
                
                self.gnb_plots = {} 
                for cell_type, spec in self.cell_type_specs.items():
                    type_gbs = [gd for gd in self.gnb_details if gd['type'] == cell_type]
                    if type_gbs:
                        locs = np.array([gb['loc'] for gb in type_gbs])
                        self.gnb_plots[cell_type] = self.ax.scatter(locs[:,0], locs[:,1],
                                                                    c=spec['color'], marker=spec['marker'],
                                                                    s=spec['size'], label=f'{cell_type.capitalize()} gNBs',
                                                                    alpha=0.7, zorder=1) # zorder for layering
                
                self.ue_path_plots = []
                self.current_ue_scatters = []
                self.serving_lines = []
                ue_colors = plt.cm.get_cmap('viridis', self.num_ues +1) # Get distinct colors

                for i in range(self.num_ues):
                    color = ue_colors(i / max(1, self.num_ues -1 )) if self.num_ues > 1 else ue_colors(0.5)
                    path_plot, = self.ax.plot([], [], '-', color=color, alpha=0.5, label=f'UE {i} Path', zorder=2)
                    self.ue_path_plots.append(path_plot)
                    
                    current_scatter = self.ax.scatter([], [], c=[color], marker='o', s=70, edgecolors='black', label=f'UE {i}', zorder=3)
                    self.current_ue_scatters.append(current_scatter)
                    
                    line, = self.ax.plot([], [], '--', color=color, alpha=0.6, zorder=2)
                    self.serving_lines.append(line)

                self.ax.set_xlim(-0.05 * self.town_width, self.town_width * 1.05) # Add some padding
                self.ax.set_ylim(-0.05 * self.town_height, self.town_height * 1.05)
                self.ax.set_xlabel("X-coordinate (m)")
                self.ax.set_ylabel("Y-coordinate (m)")
                self.ax.legend(loc='upper right', fontsize='small')
                self.ax.grid(True)
                plt.show(block=False)

            for i in range(self.num_ues):
                if self.ue_paths_history[i]:
                    path_coords = np.array(self.ue_paths_history[i])
                    self.ue_path_plots[i].set_data(path_coords[:,0], path_coords[:,1])
                
                self.current_ue_scatters[i].set_offsets(self.ue_positions[i]) # set_offsets expects (N,2) or single (x,y)

                serving_idx = self.ue_serving_gnb_indices[i]
                if self.num_gnbs_total > 0 and 0 <= serving_idx < self.num_gnbs_total :
                    serving_gnb_pos = self.gnb_details[serving_idx]['loc']
                    self.serving_lines[i].set_data([self.ue_positions[i][0], serving_gnb_pos[0]],
                                                   [self.ue_positions[i][1], serving_gnb_pos[1]])
                else:
                    self.serving_lines[i].set_data([],[])
            
            self.ax.set_title(f"HetNet Handover - Step: {self.current_step}/{self.steps_per_episode}")
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def close(self):
        if self.viewer is not None:
            plt.ioff()
            plt.close(self.fig)
            self.viewer = None