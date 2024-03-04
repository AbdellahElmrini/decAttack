import os
import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 12})

# Replace with the actual directory where the data is saved
exp_dir = "experiments/erdos_renyi_False_20-100"

# Load the results from the pickle file
with open(os.path.join(exp_dir, "res.pkl"), "rb") as f:
    res11, res22, res33 = pickle.load(f)

# Load the configuration from the JSON file
with open(os.path.join(exp_dir, "config.json"), "r") as f:
    config = json.load(f)

# Extract configuration variables
n_list = config['n_list']
p_list = np.array(config['p_list'])

"""
# previous figure
# Create the figure and axes
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
x_ticks = p_list * n_list[0]

# Plot for each attacker configuration
for a_idx, res_array in enumerate([res11, res22, res33]):
    for n_idx, n in enumerate(n_list):
        mean_vals = np.mean(res_array[n_idx, :], axis=1)
        std_vals = np.std(res_array[n_idx, :], axis=1)
        ax[a_idx].errorbar(x_ticks, mean_vals, yerr=std_vals, fmt="s--", label=rf'$n={n}$')

# Set legends, labels, and limits
for i, a in enumerate(ax):
    a.legend()
    a.set_xlabel(r'$n \times p$')
    a.set_ylim([0, 1.01])

# Set titles for each subplot
ax[0].set_title("1 attacker")
ax[1].set_title("2 attackers")
ax[2].set_title("3 attackers")

# Show the plot
plt.show()
"""

# Create the figure and axes
fig, ax = plt.subplots(1, 3, figsize=(18, 4))
x_ticks = p_list * n_list[0]
# Define the color palette and bar width
cmap = plt.cm.viridis
bar_width = 0.15  # Adjust as needed for the best visual appearance

# Number of datasets (n values) per attacker configuration
num_datasets = len(n_list)

# Plot histograms with error bars for each attacker configuration
for a_idx, res_array in enumerate([res11, res22, res33]):
    for n_idx, n in enumerate(n_list):
        mean_vals = np.mean(res_array[n_idx, :], axis=1)
        std_vals = np.std(res_array[n_idx, :], axis=1)
        
        # Determine the color from the palette
        color = cmap(n_idx / num_datasets)

        # Calculate the x positions for this dataset
        x_positions = x_ticks + (n_idx - num_datasets / 2) * bar_width + bar_width / 2

        # Plot histogram
        ax[a_idx].bar(x_positions, mean_vals, width=bar_width, color=color, label=rf'$n={n}$', alpha=0.7)

        # Add error bars
        ax[a_idx].errorbar(x_positions, mean_vals, yerr=std_vals, fmt='none', ecolor='black', capsize=5)

    # Set custom x-ticks and labels
    ax[a_idx].set_xticks(x_ticks)
    ax[a_idx].set_xticklabels([f'{n_list[0]*p:.2f}' for p in p_list])


# Set legends, labels, and limits
for i, a in enumerate(ax):
    if i==0:
        a.set_ylabel('Proportion of reconstructed nodes')
    a.legend()
    a.set_xlabel(r'$n \times p$')
    
    a.set_ylim([0, 1.01])

# Set titles for each subplot
ax[0].set_title("1 attacker")
ax[1].set_title("2 attackers")
ax[2].set_title("3 attackers")


plt.savefig("propER.pdf", bbox_inches='tight', pad_inches=0)
# Show the plot
plt.show()
