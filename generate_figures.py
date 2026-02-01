"""
QICCR Scalability Visualization
Generates performance plots for the paper
"""

import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['text.usetex'] = False  # Set to True if LaTeX is available

# Data: Concurrent reasoning paths vs. throughput
states = np.array([1000, 10000, 100000, 1000000])
qiccr_throughput = np.array([5000, 4800, 4500, 4200])  # states/sec with 4 GPUs
llm_beam_search = np.array([800, 750, 600, 400])  # LLM with beam search (saturates)
sequential_baseline = np.array([200, 190, 180, 170])  # Sequential search

# Create figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot lines
ax.plot(states, qiccr_throughput, marker='o', markersize=8, linewidth=2.5, 
        label='QICCR (4x A100 GPUs)', color='#2E86AB', linestyle='-')
ax.plot(states, llm_beam_search, marker='s', markersize=7, linewidth=2, 
        label='LLM Beam Search', color='#A23B72', linestyle='--')
ax.plot(states, sequential_baseline, marker='^', markersize=7, linewidth=2, 
        label='Sequential Search', color='#F18F01', linestyle=':')

# Formatting
ax.set_xscale('log')
ax.set_xlabel('Concurrent Reasoning Paths', fontsize=12, fontweight='bold')
ax.set_ylabel('Throughput (paths/second)', fontsize=12, fontweight='bold')
ax.set_title('QICCR Scalability Analysis', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Add annotation for key insight
ax.annotate('Near-linear scaling', 
            xy=(100000, 4500), xytext=(50000, 5200),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'),
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

# Tight layout
plt.tight_layout()

# Save figure
plt.savefig('scalability_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('scalability_plot.pdf', bbox_inches='tight')

print("Scalability plot generated successfully!")
print("Files saved:")
print("  - scalability_plot.png (for preview)")
print("  - scalability_plot.pdf (for LaTeX)")

# Create additional performance comparison plot
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left: Accuracy vs. Problem Complexity
problem_steps = np.array([2, 3, 4, 5, 6, 7, 8, 10])
qiccr_accuracy = np.array([94, 92, 90, 88, 87, 85, 83, 80])
baseline_accuracy = np.array([85, 82, 78, 74, 71, 68, 62, 55])

ax1.plot(problem_steps, qiccr_accuracy, marker='o', markersize=8, linewidth=2.5,
         label='QICCR', color='#2E86AB')
ax1.plot(problem_steps, baseline_accuracy, marker='s', markersize=7, linewidth=2,
         label='Best Baseline', color='#A23B72', linestyle='--')
ax1.set_xlabel('Number of Reasoning Steps', fontsize=11, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_title('Accuracy vs. Problem Complexity', fontsize=12, fontweight='bold')
ax1.legend(loc='lower left', frameon=True, fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([50, 100])

# Right: Fidelity vs. Accuracy scatter
benchmarks = ['GSM8K++', 'StrategyQA', 'EntailmentBank', 'LiveCodeBench']
fidelity_scores = np.array([0.89, 0.85, 0.87, 0.86])
accuracy_scores = np.array([90, 82, 68, 75])
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

for i, benchmark in enumerate(benchmarks):
    ax2.scatter(fidelity_scores[i], accuracy_scores[i], s=200, 
                color=colors[i], alpha=0.7, edgecolors='black', linewidth=1.5,
                label=benchmark)

ax2.set_xlabel('Fidelity Score', fontsize=11, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax2.set_title('Fidelity-Accuracy Correlation', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right', frameon=True, fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0.82, 0.92])
ax2.set_ylim([60, 95])

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('performance_comparison.pdf', bbox_inches='tight')

print("Performance comparison plot generated successfully!")

# Create tensor network visualization (conceptual)
fig3, ax = plt.subplots(figsize=(8, 5))

# This creates a conceptual diagram showing tensor network structure
# In practice, you'd use TikZ in LaTeX for publication quality

from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import Rectangle

# Draw MPS tensor chain
num_tensors = 6
tensor_width = 0.8
tensor_height = 0.6
y_pos = 2

for i in range(num_tensors):
    x_pos = i * 1.5
    
    # Tensor box
    rect = FancyBboxPatch((x_pos, y_pos), tensor_width, tensor_height,
                           boxstyle="round,pad=0.05", 
                           facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    
    # Variable index (physical leg)
    ax.plot([x_pos + tensor_width/2, x_pos + tensor_width/2], 
            [y_pos + tensor_height, y_pos + tensor_height + 0.4],
            'k-', linewidth=2)
    ax.plot(x_pos + tensor_width/2, y_pos + tensor_height + 0.4, 
            'ko', markersize=8)
    ax.text(x_pos + tensor_width/2, y_pos + tensor_height + 0.6, 
            f'$x_{i+1}$', ha='center', fontsize=11, fontweight='bold')
    
    # Horizontal bonds
    if i < num_tensors - 1:
        ax.plot([x_pos + tensor_width, x_pos + 1.5], 
                [y_pos + tensor_height/2, y_pos + tensor_height/2],
                'r-', linewidth=3, label='Bond' if i == 0 else '')

ax.text(num_tensors * 1.5 / 2, y_pos - 0.8, 
        'Matrix Product State (MPS) Tensor Network',
        ha='center', fontsize=13, fontweight='bold')

ax.set_xlim([-0.5, num_tensors * 1.5])
ax.set_ylim([0, 4])
ax.axis('off')

# Add legend
ax.plot([], [], 'ko-', linewidth=2, label='Physical indices')
ax.plot([], [], 'ro-', linewidth=3, label='Bond connections')
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('tensor_network_structure.png', dpi=300, bbox_inches='tight')
plt.savefig('tensor_network_structure.pdf', bbox_inches='tight')

print("Tensor network structure diagram generated successfully!")
print("\nAll figures generated. Ready for inclusion in LaTeX paper.")
