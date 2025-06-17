import numpy as np
import config

class TraditionalAgent:
    def __init__(self, num_gnbs, hysteresis_db, time_to_trigger_steps):
        self.num_gnbs = num_gnbs
        self.hysteresis = hysteresis_db
        self.ttt_steps = time_to_trigger_steps
        self.candidate_gnb = -1
        self.ttt_counter = 0

    def reset(self):
        self.candidate_gnb = -1
        self.ttt_counter = 0

    def choose_action(self, state, current_serving_gnb):
        """
        Implements Event A3 logic: Handover if a neighbor is better
        than serving by Hysteresis margin for TimeToTrigger duration.
        Args:
            state (np.array): Array of RSRP values [RSRP_gNB0, RSRP_gNB1, ...]
            current_serving_gnb (int): Index of the currently serving gNB.
        Returns:
            int: Action (index of the target gNB, which could be the current one).
        """
        serving_rsrp = state[current_serving_gnb]
        best_neighbor_idx = -1
        best_neighbor_rsrp = -np.inf

        # Find the best neighbor cell
        for i in range(self.num_gnbs):
            if i == current_serving_gnb:
                continue
            if state[i] > best_neighbor_rsrp:
                best_neighbor_rsrp = state[i]
                best_neighbor_idx = i

        # Check A3 condition: Neighbor > Serving + Hysteresis
        if best_neighbor_idx != -1 and best_neighbor_rsrp > serving_rsrp + self.hysteresis:
            # Condition met, check if it's the same candidate as before
            if self.candidate_gnb == best_neighbor_idx:
                self.ttt_counter += 1
            else:
                # New candidate, reset counter
                self.candidate_gnb = best_neighbor_idx
                self.ttt_counter = 1

            # Check if TTT is met
            if self.ttt_counter >= self.ttt_steps:
                # Trigger handover
                self.reset() # Reset TTT state after triggering
                return self.candidate_gnb # Action: Handover to the candidate
        else:
            # Condition not met, reset TTT
            self.reset()

        # Default action: Stay with the current serving cell
        return current_serving_gnb