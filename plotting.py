import matplotlib.pyplot as plt
import numpy as np

def plot_results(dqn_results, traditional_results, num_eval_episodes):
    """Plots the comparison between DQN and Traditional methods over evaluation episodes,
       including a histogram of total handovers."""

    metrics_to_plot_lines = [ # Metrics for line plots
        ('total_failures', 'Total Failures per Episode'),
        ('total_ping_pongs', 'Total Ping-Pongs per Episode'),
        ('failure_rate', 'Handover Failure Rate per Episode'),
        ('ping_pong_rate', 'Ping-Pong Rate per Episode'),
        ('avg_latency', 'Average Latency (ms) per Episode'),
        ('avg_bandwidth', 'Average Bandwidth (Mbps) per Episode'),
        ('avg_rsrp', 'Average Serving RSRP (dBm) per Episode'),
        ('total_handovers', 'Total Handovers per Episode (Line)') # Keep the line plot for total_handovers too
    ]

    n_line_metrics = len(metrics_to_plot_lines)
    # We'll add one more plot for the histogram
    n_total_plots = n_line_metrics + 1
    n_cols = 2
    n_rows = (n_total_plots + n_cols - 1) // n_cols # Ensure enough rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() # Make it easier to index

    episodes_eval = np.arange(num_eval_episodes)

    def smooth(y, box_pts):
        if not y or len(y) < box_pts or box_pts <= 0:
            return np.array(y) # Return a copy
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='valid')
        pad_size_before = (len(y) - len(y_smooth) + 1) // 2
        pad_size_after = len(y) - len(y_smooth) - pad_size_before
        if pad_size_before >= 0 and pad_size_after >= 0:
            y_smooth = np.pad(y_smooth, (pad_size_before, pad_size_after), mode='edge')
        if len(y_smooth) != len(y): # Fallback
            return np.array(y)
        return y_smooth

    smoothing_window = max(1, num_eval_episodes // 10)

    plot_idx = 0
    for metric_key, title in metrics_to_plot_lines:
        if plot_idx >= len(axes): break
        ax = axes[plot_idx]

        if metric_key in dqn_results and dqn_results[metric_key]:
            smoothed_dqn = smooth(dqn_results[metric_key], smoothing_window)
            if len(smoothed_dqn) == num_eval_episodes:
                ax.plot(episodes_eval, smoothed_dqn, label='DQN (Smoothed)', alpha=0.9, color='blue')
            ax.plot(episodes_eval, dqn_results[metric_key], label='DQN (Raw)', alpha=0.3, color='lightblue')

        if metric_key in traditional_results and traditional_results[metric_key]:
            smoothed_trad = smooth(traditional_results[metric_key], smoothing_window)
            if len(smoothed_trad) == num_eval_episodes:
                ax.plot(episodes_eval, smoothed_trad, label='Traditional (Smoothed)', alpha=0.9, color='orange')
            ax.plot(episodes_eval, traditional_results[metric_key], label='Traditional (Raw)', alpha=0.3, color='peachpuff')

        ax.set_title(title)
        ax.set_xlabel("Evaluation Episode")
        ax.set_ylabel(metric_key.replace('_', ' ').capitalize())
        ax.legend()
        ax.grid(True)
        plot_idx += 1

    # --- Add Histogram for Total Handovers ---
    if plot_idx < len(axes): # Check if there's space for the histogram plot
        ax_hist = axes[plot_idx]
        dqn_ho_counts = dqn_results.get('total_handovers', [])
        trad_ho_counts = traditional_results.get('total_handovers', [])

        if dqn_ho_counts or trad_ho_counts:
            # Determine common bins for comparison
            all_ho_counts = []
            if dqn_ho_counts: all_ho_counts.extend(dqn_ho_counts)
            if trad_ho_counts: all_ho_counts.extend(trad_ho_counts)

            if all_ho_counts: # Ensure there's data to plot
                min_val = np.min(all_ho_counts)
                max_val = np.max(all_ho_counts)
                # Determine a reasonable number of bins, e.g., one bin per integer count if range is small
                num_bins = int(max_val - min_val + 1)
                if num_bins > 20 : # Cap max bins to avoid overly granular histograms
                    num_bins = 20
                if num_bins <= 0: # Ensure at least 1 bin
                    num_bins = 1

                bins = np.linspace(min_val, max_val, num_bins + 1)

                if dqn_ho_counts:
                    ax_hist.hist(dqn_ho_counts, bins=bins, alpha=0.7, label='DQN Handovers', color='blue', edgecolor='black')
                if trad_ho_counts:
                    ax_hist.hist(trad_ho_counts, bins=bins, alpha=0.7, label='Traditional Handovers', color='orange', edgecolor='black')

                ax_hist.set_title('Distribution of Handovers per Episode')
                ax_hist.set_xlabel('Number of Handovers in an Episode')
                ax_hist.set_ylabel('Frequency (Number of Episodes)')
                ax_hist.legend()
                ax_hist.grid(axis='y', alpha=0.75)
        plot_idx +=1


    # Hide any unused subplots
    for j in range(plot_idx, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig("comparison_results_with_histogram.png") # New filename
    plt.show()


def plot_rewards(rewards, title="DQN Training Rewards"):
    """Plots the rewards per episode."""
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(rewards)), rewards, label='Raw Reward') # Ensure x-axis starts from 0

    if len(rewards) > 10:
        smoothing_window_rewards = max(1, len(rewards) // 20)
        def smooth_rewards_data(y, box_pts): # Renamed to avoid conflict
            if not isinstance(y, list) or not y or len(y) < box_pts or box_pts <= 0: return np.array(y)
            y_arr = np.array(y)
            box = np.ones(box_pts)/box_pts
            y_smooth = np.convolve(y_arr, box, mode='valid')
            pad_size_before = (len(y_arr) - len(y_smooth) + 1) // 2
            pad_size_after = len(y_arr) - len(y_smooth) - pad_size_before
            if pad_size_before >= 0 and pad_size_after >=0:
                y_smooth = np.pad(y_smooth, (pad_size_before, pad_size_after), mode='edge')
            if len(y_smooth) != len(y_arr): return y_arr # Fallback
            return y_smooth

        smoothed_rewards = smooth_rewards_data(rewards, smoothing_window_rewards)
        if len(smoothed_rewards) == len(rewards):
            plt.plot(np.arange(len(smoothed_rewards)), smoothed_rewards, label=f'Smoothed (window {smoothing_window_rewards})', linestyle='--')
    
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("dqn_rewards.png")
    plt.show()